"""Verified Plan-Execute (Plan-Execute-Verify-Replan) for AOB.

Adds a per-step verifier-replan loop on top of :class:`PlanExecuteRunner`.
After each step executes, the verifier LLM judges whether to:

- ``continue`` — accept the result and move on
- ``retry`` — same step, with verifier-guided retry context (bounded)
- ``replan_suffix`` — discard remaining steps, regenerate a suffix plan

Originally developed in the HPML Smart Grid MCP team's
``scripts/verified_pe_runner.py`` + ``orchestration_utils.py`` and ported
here as a generic AOB helper. The Smart-Grid-specific repair logic
(sensor-task repair, invalid-sensor skip) is intentionally NOT ported;
domain-specific repair stays at the team-repo customization layer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm import LLMBackend
from observability import agent_run_span, persist_trajectory

from .executor import Executor
from .models import OrchestratorResult, Plan, PlanStep, StepResult
from .planner import Planner
from .runner import PlanExecuteRunner, _SUMMARIZE_PROMPT
from .self_ask import SelfAskDecision, _parse_json_object, maybe_self_ask

_log = logging.getLogger(__name__)


VERIFIER_PROMPT = """\
You are verifying whether a completed plan-execute step actually advanced the
goal enough to continue without repair.

Return a single raw JSON object with exactly these keys:
- decision: one of "continue", "retry", "replan_suffix"
- reason: short string
- updated_focus: short string, or empty string if not needed

Rules:
- Prefer "continue" unless the current result clearly suggests the remaining
  plan should change.
- Use "retry" only when the same step should be attempted once more with the
  same overall intent.
- Use "replan_suffix" when the completed context changes what the remaining
  steps should be.
- Keep the answer benchmarkable: avoid open-ended or conversational behavior.

Original question:
{question}

Effective planning question:
{effective_question}

Current step:
{current_step}

Current step result:
{current_result}

Completed history so far:
{history}

Remaining planned steps:
{remaining_steps}

JSON:
"""


@dataclass
class VerificationDecision:
    """One verifier verdict per step."""

    decision: str  # "continue" | "retry" | "replan_suffix"
    reason: str
    updated_focus: str


class _GenerativeLLM(Protocol):
    def generate(self, prompt: str, temperature: float = 0.0) -> str: ...


# --------------------------------------------------------------------------- helpers


def _compact(text: Any, *, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return f"{s[:limit]}\n...[truncated {len(s) - limit} chars]"


def _compact_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(none)"
    chunks = []
    for entry in history:
        status = "OK" if entry.get("success") else "ERR"
        detail = entry.get("response") if entry.get("success") else entry.get("error")
        chunks.append(
            f"Step {entry.get('step')} [{status}] {entry.get('task')} | "
            f"tool={entry.get('tool')} | detail={_compact(detail, limit=240)}"
        )
    return "\n".join(chunks)


def _compact_steps(steps: list[PlanStep]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(
        f"Step {s.step_number}: {s.task} | server={s.server} | tool={s.tool}"
        for s in steps
    )


def _serialize_step(step_result: StepResult, **extra: Any) -> dict[str, Any]:
    payload = {
        "step": step_result.step_number,
        "task": step_result.task,
        "server": step_result.server,
        "tool": step_result.tool,
        "tool_args": step_result.tool_args,
        "response": step_result.response,
        "error": step_result.error,
        "success": step_result.success,
    }
    payload.update(extra)
    return payload


def verify_step(
    question: str,
    effective_question: str,
    current_step: PlanStep,
    current_result: dict[str, Any],
    history: list[dict[str, Any]],
    remaining_steps: list[PlanStep],
    llm: _GenerativeLLM,
) -> VerificationDecision:
    """Ask the verifier LLM to judge a freshly-completed step.

    Returns a :class:`VerificationDecision`. Failures (LLM error, malformed
    JSON, unknown decision) fall back to ``decision="continue"`` so the
    pipeline never stalls.
    """
    try:
        raw = llm.generate(
            VERIFIER_PROMPT.format(
                question=question,
                effective_question=effective_question,
                current_step=(
                    f"Step {current_step.step_number}: {current_step.task} "
                    f"(server={current_step.server}, tool={current_step.tool})"
                ),
                current_result=json.dumps(current_result, indent=2, default=str),
                history=_compact_history(history),
                remaining_steps=_compact_steps(remaining_steps),
            )
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "Verifier failed (%s); falling back to decision='continue'.", exc
        )
        return VerificationDecision(
            decision="continue",
            reason="Verifier unavailable; continuing with the current result.",
            updated_focus="",
        )

    payload = _parse_json_object(raw)
    decision = str(payload.get("decision", "continue")).strip().lower()
    if decision not in {"continue", "retry", "replan_suffix"}:
        _log.warning(
            "Verifier returned unknown decision %r; falling back to 'continue'.",
            decision,
        )
        decision = "continue"

    return VerificationDecision(
        decision=decision,
        reason=str(payload.get("reason", "")).strip()
        or "No additional verifier reason recorded.",
        updated_focus=str(payload.get("updated_focus", "")).strip(),
    )


def build_retry_question(
    question: str,
    effective_question: str,
    current_step: PlanStep,
    current_result: dict[str, Any],
    decision: VerificationDecision,
    retries_used: int,
) -> str:
    """Compose a retry-guidance prompt for re-executing a step."""
    detail = (
        current_result.get("response") or current_result.get("error") or "(none)"
    )
    prompt = [
        effective_question,
        "",
        "Retry guidance for the current execution step:",
        f"- Original question: {question}",
        f"- Step: {current_step.step_number} / {current_step.task}",
        f"- Previous attempt result: {_compact(detail, limit=400)}",
        f"- Verifier reason: {decision.reason}",
        f"- Retry attempt number: {retries_used}",
        "- Keep the same overall intent, but correct the specific issue the verifier flagged.",
    ]
    if decision.updated_focus:
        prompt.append(f"- Updated focus: {decision.updated_focus}")
    return "\n".join(prompt)


def build_suffix_replan_question(
    question: str,
    effective_question: str,
    history: list[dict[str, Any]],
    remaining_steps: list[PlanStep],
    decision: VerificationDecision,
) -> str:
    """Compose a replan-suffix prompt for the planner."""
    prompt = [
        "Replan only the remaining suffix of the task.",
        "",
        f"Original question:\n{question}",
        "",
        f"Effective planning question:\n{effective_question}",
        "",
        "Verified completed context:",
        _compact_history(history),
        "",
        "Remaining plan that may need repair:",
        _compact_steps(remaining_steps),
        "",
        f"Verifier reason: {decision.reason}",
    ]
    if decision.updated_focus:
        prompt.extend(["", f"Updated focus for the suffix: {decision.updated_focus}"])
    prompt.extend(
        [
            "",
            "Only plan the remaining work. Do not repeat already completed steps.",
            "Rules for this repaired suffix plan:",
            "- Start the repaired suffix at step 1.",
            "- Completed steps are already done and available as context, not as dependencies.",
            "- The first suffix step must use Dependency1: None.",
            "- Only reference dependencies on earlier suffix steps (#S1, #S2, ...) and never on completed steps.",
            "- If a suffix step needs prior completed context, mention it in the task text instead of as a dependency.",
        ]
    )
    return "\n".join(prompt)


def renumber_plan(plan: Plan, offset: int) -> Plan:
    """Return a copy of ``plan`` with every step number shifted by ``offset``.

    Useful for splicing a suffix plan onto an existing trajectory.
    """
    renumbered = []
    for step in plan.steps:
        renumbered.append(
            PlanStep(
                step_number=step.step_number + offset,
                task=step.task,
                server=step.server,
                tool=step.tool,
                tool_args=step.tool_args,
                dependencies=[dep + offset for dep in step.dependencies],
                expected_output=step.expected_output,
            )
        )
    return Plan(steps=renumbered, raw=plan.raw)


# --------------------------------------------------------------------------- runner


class VerifiedPlanExecuteRunner(PlanExecuteRunner):
    """PlanExecuteRunner with a per-step verifier-replan loop.

    Args:
        llm: LLM backend used for planning, execution, summarisation, the
            optional Self-Ask pre-pass, and the verifier.
        server_paths: Override MCP server specs.
        max_replans: Maximum number of suffix replans the verifier may
            trigger across the whole run (default 2).
        max_retries_per_step: Maximum verifier-driven retries per step
            (default 1).
        enable_self_ask: Whether to run the Self-Ask pre-pass before
            planning (default True).

    Returns the same :class:`OrchestratorResult` shape as the base runner;
    the verifier history is exposed indirectly through OTel span attributes
    and the persisted trajectory record's ``runner_name``.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        *,
        max_replans: int = 2,
        max_retries_per_step: int = 1,
        enable_self_ask: bool = True,
    ) -> None:
        super().__init__(llm, server_paths)
        self._max_replans = max_replans
        self._max_retries_per_step = max_retries_per_step
        self._enable_self_ask = enable_self_ask

    async def run(self, question: str) -> OrchestratorResult:
        with agent_run_span(
            "verified-plan-execute",
            model=self._llm.model_id,
            question=question,
        ) as span:
            run_started = time.perf_counter()
            self._meter.reset()

            # 0. Optional Self-Ask pass.
            if self._enable_self_ask:
                self_ask = maybe_self_ask(question, self._meter)
            else:
                self_ask = SelfAskDecision(
                    needs_self_ask=False,
                    clarifying_questions=[],
                    assumptions=[],
                    augmented_question=question,
                )
            effective_q = self_ask.augmented_question

            # 1. Discover.
            _log.info("Discovering server capabilities...")
            server_descriptions = await self._executor.get_server_descriptions()

            # 2. Plan.
            _log.info("Planning...")
            planning_started = time.perf_counter()
            initial_plan = self._planner.generate_plan(effective_q, server_descriptions)
            planning_ms = (time.perf_counter() - planning_started) * 1000
            active_steps = list(initial_plan.resolved_order())
            all_plan_steps: list[PlanStep] = list(active_steps)

            # 3. Execute with per-step verification.
            history: list[dict[str, Any]] = []
            trajectory: list[StepResult] = []
            replans_used = 0
            step_index = 0

            while step_index < len(active_steps):
                step = active_steps[step_index]
                retries_used = 0
                step_question = effective_q

                while True:
                    result = await self._execute_one_step(step, trajectory, step_question)
                    trajectory.append(result)
                    remaining_steps = active_steps[step_index + 1:]

                    if not result.success:
                        history.append(
                            _serialize_step(
                                result,
                                verifier_decision="error",
                                verifier_reason=result.error
                                or "Step failed before verification.",
                            )
                        )
                        break

                    serialized = _serialize_step(result)
                    verdict = verify_step(
                        question,
                        effective_q,
                        step,
                        serialized,
                        history,
                        remaining_steps,
                        self._meter,
                    )

                    if (
                        verdict.decision == "retry"
                        and retries_used < self._max_retries_per_step
                    ):
                        retries_used += 1
                        history.append(
                            _serialize_step(
                                result,
                                verifier_decision="retry",
                                verifier_reason=verdict.reason,
                                retries_used=retries_used,
                            )
                        )
                        step_question = build_retry_question(
                            question,
                            effective_q,
                            step,
                            serialized,
                            verdict,
                            retries_used,
                        )
                        # Drop the just-appended trajectory entry so the
                        # final trajectory carries only the accepted result.
                        trajectory.pop()
                        _log.info(
                            "Verifier requested retry for step %d; retrying.",
                            step.step_number,
                        )
                        continue

                    if verdict.decision == "retry":
                        # Retry budget exhausted; treat as continue.
                        verdict = VerificationDecision(
                            decision="continue",
                            reason=(
                                f"{verdict.reason} Retry budget exhausted; "
                                "continuing with the current result."
                            ),
                            updated_focus=verdict.updated_focus,
                        )

                    history.append(
                        _serialize_step(
                            result,
                            verifier_decision=verdict.decision,
                            verifier_reason=verdict.reason,
                            verifier_updated_focus=verdict.updated_focus,
                            retries_used=retries_used,
                        )
                    )

                    if (
                        verdict.decision == "replan_suffix"
                        and remaining_steps
                        and replans_used < self._max_replans
                    ):
                        replans_used += 1
                        replan_q = build_suffix_replan_question(
                            question,
                            effective_q,
                            history,
                            remaining_steps,
                            verdict,
                        )
                        try:
                            suffix_plan = self._planner.generate_plan(
                                replan_q, server_descriptions
                            )
                        except Exception as exc:  # noqa: BLE001
                            _log.warning(
                                "Suffix replan failed (%s); continuing with original suffix.",
                                exc,
                            )
                        else:
                            shifted = renumber_plan(
                                suffix_plan,
                                max(p.step_number for p in all_plan_steps),
                            )
                            suffix_steps = list(shifted.resolved_order())
                            all_plan_steps.extend(suffix_steps)
                            active_steps = active_steps[: step_index + 1] + suffix_steps
                            _log.info(
                                "Verifier triggered suffix replan with %d new step(s).",
                                len(suffix_steps),
                            )

                    break

                step_index += 1

            # 4. Summarise against the ORIGINAL question.
            _log.info("Summarising...")
            results_text = "\n\n".join(
                f"Step {r.step_number} — {r.task} (server: {r.server}):\n"
                + (r.response if r.success else f"ERROR: {r.error}")
                for r in trajectory
            )
            summarization_started = time.perf_counter()
            answer = self._meter.generate(
                _SUMMARIZE_PROMPT.format(question=question, results=results_text)
            )
            summarization_ms = (time.perf_counter() - summarization_started) * 1000
            duration_ms = (time.perf_counter() - run_started) * 1000

            full_plan = Plan(steps=all_plan_steps, raw=initial_plan.raw)
            result = OrchestratorResult(
                question=question,
                answer=answer,
                plan=full_plan,
                trajectory=trajectory,
            )
            span.set_attribute("agent.plan.steps", len(all_plan_steps))
            span.set_attribute("agent.answer.length", len(answer or ""))
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.planning_time_ms", planning_ms)
            span.set_attribute("agent.summarization_time_ms", summarization_ms)
            span.set_attribute("agent.self_ask.fired", self_ask.needs_self_ask)
            span.set_attribute("agent.verified.replans_used", replans_used)
            span.set_attribute("gen_ai.usage.input_tokens", self._meter.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", self._meter.output_tokens)
            span.set_attribute("agent.llm_time_ms", planning_ms + summarization_ms)
            persist_trajectory(
                runner_name="verified-plan-execute",
                model=self._llm.model_id,
                question=question,
                answer=answer or "",
                trajectory=trajectory,
            )
            return result

    async def _execute_one_step(
        self,
        step: PlanStep,
        prior_trajectory: list[StepResult],
        step_question: str,
    ) -> StepResult:
        """Execute a single plan step, building the per-step context dict
        the executor expects.
        """
        # The base executor's ``execute_plan`` takes the whole plan and
        # threads context internally. For per-step verification we mimic
        # that by constructing the context-by-step-number dict from the
        # already-completed trajectory.
        context: dict[int, StepResult] = {
            r.step_number: r for r in prior_trajectory
        }
        return await self._executor.execute_step(step, context, step_question)
