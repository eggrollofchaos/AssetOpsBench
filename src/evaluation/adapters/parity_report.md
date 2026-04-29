# SG ↔ AOB judge parity report (Phase 1, Step 3)

*Drafted 2026-04-28 alongside the per-trial JSON adapter. The full parity-check
run (LLM judge against our 36 canonical trials) requires Watsonx or Insomnia
access; this document captures the static analysis findings and the script
contract for the live run.*

## TL;DR

- **Rubric keys**: identical between team `scripts/judge_trajectory.py` and AOB
  `src/evaluation/graders/llm_judge.py`. Both use the six-criterion shape
  `{task_completion, data_retrieval_accuracy, generalized_result_verification,
  agent_sequence_correct, clarity_and_justification, hallucinations}`.
- **Per-dim boolean prompts**: substantially the same — both ask the judge for
  six bools against `scenario.characteristic_form`. Wording differs slightly
  but the criteria definitions are aligned.
- **Aggregate score formulas DIFFER**:
  - Team: `score_6d = (count of True among first 5 + (1 if hallucinations is False else 0)) / 6` → range `[0, 1]`, evenly weighted across all six dims, threshold for "judge-pass" is `score_6d ≥ 0.6` (4-of-6 dims).
  - AOB: `score = (count of True among first 5) / 5.0; if hallucinations is True, score = max(0, score - 0.2)` → range `[0, 1]`, weights first five evenly with a hallucination penalty rather than a sixth slot.
- **Likely consequence**: per-dim booleans will agree at high κ (same criteria,
  similar prompts); aggregate scores will diverge by up to ±0.2 on identical
  bool-vectors. Threshold-based pass/fail (`≥ 0.6`) may flip on borderline
  cases.

## Recommendation

Per spec § Q-EVAL-PARITY recommendation: behavioral parity at agreement
threshold ≥ 95% on judge-pass classification, Cohen's κ ≥ 0.8 across the six
per-dim booleans. The formula divergence does not block parity — but reviewers
should know the aggregate-score numbers will not match exactly.

Two paths to reconcile (post-Phase 1, when AOB merges feat/evaluation-module):

1. **Port team formula upstream.** Open a PR against AOB
   `feat/evaluation-module` adding a `score_6d` mode that uses the team's
   evenly-weighted formula. This is the "minimum surprise" path for our
   existing `results/metrics/scenario_scores.jsonl` numbers.
2. **Adopt AOB formula in team repo.** Re-score our 36 canonical trials with
   the AOB formula. Headline tables would shift slightly. Cleaner long-term;
   bigger short-term churn.

Pick (1) if Dhaval is open to it; else (2) post-paper.

## Live-run contract (deferred — needs LLM endpoint)

The script that produces the on-disk parity report:

```bash
# 1. Adapt our 36 canonical trials into PersistedTrajectory shape:
uv run python -c "
from pathlib import Path
import json
from evaluation.adapters import load_team_run_dir
TEAM = Path('/Users/wax/coding/hpml-assetopsbench-smart-grid-mcp')
TARGETS = [
    TEAM / 'benchmarks/cell_A_direct/raw/8979314_aat_direct',
    TEAM / 'benchmarks/cell_B_mcp_baseline/raw/8979314_aat_mcp_baseline',
    TEAM / 'benchmarks/cell_Y_plan_execute/raw/8998340_exp2_cell_Y_pe_mcp_baseline',
    TEAM / 'benchmarks/cell_Y_plan_execute/raw/8998341_exp2_cell_Y_pe_self_ask_mcp_baseline',
    TEAM / 'benchmarks/cell_Z_hybrid/raw/8998342_exp2_cell_Z_verified_pe_mcp_baseline',
    TEAM / 'benchmarks/cell_Z_hybrid/raw/8998343_exp2_cell_Z_verified_pe_self_ask_mcp_baseline',
]
out_dir = Path('/tmp/sg_trajectories'); out_dir.mkdir(exist_ok=True)
for t in TARGETS:
    for rec in load_team_run_dir(t):
        path = out_dir / f'{rec.run_id}__{rec.scenario_id}__{rec.runner}.json'
        path.write_text(rec.model_dump_json(indent=2))
print(f'wrote {len(list(out_dir.glob(\"*.json\")))} trajectories')
"

# 2. Run AOB evaluator with the LLM judge configured for Watsonx Llama-4-Maverick
#    (matches what scripts/judge_trajectory.py was using):
uv run evaluate \
    --trajectories /tmp/sg_trajectories \
    --scenarios /Users/wax/coding/hpml-assetopsbench-smart-grid-mcp/data/scenarios \
    --judge-model meta-llama/llama-4-maverick-17b-128e-instruct-fp8 \
    --judge-backend watsonx \
    --output /tmp/aob_eval_report.json

# 3. Compare against our existing scenario_scores.jsonl:
uv run python -m evaluation.adapters.compute_parity \
    --aob-report /tmp/aob_eval_report.json \
    --team-scores /Users/wax/coding/hpml-assetopsbench-smart-grid-mcp/results/metrics/scenario_scores.jsonl \
    --output /Users/wax/coding/hpml-assetopsbench-smart-grid-mcp/results/aob_eval_parity.md
```

Step 3 (`compute_parity`) is **not yet implemented** in this commit; it
becomes a Phase 1 follow-up once Step 2 produces real AOB judge output.

## Estimated parity numbers (anticipatory)

Without running, my best guess:

- **κ (per-dim Boolean agreement)**: ~0.85-0.95. Same criteria, same model
  family, similar prompts. Wording differences in `_PROMPT_TEMPLATE` may
  introduce ~5% per-dim disagreement.
- **Judge-pass classification agreement (`score_6d ≥ 0.6` vs AOB
  threshold)**: ~85-95%. Formula divergence puts borderline trials (3-of-5
  + hallucinations=False) on different sides of the threshold ~10% of the
  time.
- **Cell ranking preserved?**: yes (high confidence). The relative quality
  ordering (Z+SA > Z > Y+SA > B > A > Y baseline on N=6) is robust to
  small per-trial perturbations and should be invariant to formula
  difference.

These are predictions, not measurements; revise after Step 2 runs.

## Cross-references

- Plan: `docs/plans/aob-extraction.md` (in team repo)
- Spec: `docs/plans/aob-extraction_spec.md` § Eval parity findings (in team repo)
- Team rubric: team repo `scripts/judge_trajectory.py:107-138`
- AOB rubric: AOB fork `src/evaluation/graders/llm_judge.py:30-63`
