"""Adapter for Smart Grid Bench per-trial JSON captures.

The HPML Team 13 Smart Grid Bench fork of AssetOpsBench writes one JSON file
per (scenario × trial) under ``benchmarks/cell_<X>/raw/<run_id>/``. This
adapter accepts those files and emits
:class:`evaluation.models.PersistedTrajectory` records that the offline
evaluation pipeline can grade.

It handles two shapes:

- **Canonical** (post Smart Grid Bench PR #143, 2026-04-27): each per-trial
  JSON carries a top-level ``scenario`` dict with ``id`` plus a top-level
  ``success`` bool.
- **Legacy** (pre PR #143): no top-level ``scenario`` field. Scenario
  identity must be recovered from the filename (the team repo's
  ``run_experiment.sh`` writes filenames as
  ``<date>_<cell>_<model>_<orch>_<mcp_mode>_<scenario_basename>_runNN.json``).

The two shapes are reconciled by ``from_per_trial_dict`` below.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..models import PersistedTrajectory

_log = logging.getLogger(__name__)

# Known Smart Grid Bench scenario basenames as of 2026-04-28. Used as a
# fallback when a legacy per-trial JSON does not carry a top-level
# ``scenario`` field. Callers can override this list when extracting from
# captures that include scenarios outside this set.
SCENARIO_BASENAMES_DEFAULT: tuple[str, ...] = (
    "multi_01_end_to_end_fault_response",
    "multi_02_dga_to_workorder_pipeline",
    "iot_01_list_transformer_sensors",
    "iot_02_voltage_imbalance_check",
    "fmsr_01_dga_fault_mode_diagnosis",
    "fmsr_02_hydrogen_spike_root_cause",
    "tsfm_01_rul_forecast_maintenance_window",
    "tsfm_02_hotspot_temp_anomaly",
    "wo_01_create_inspection_order",
    "wo_02_corrective_order_after_fault",
    "aob_fmsr_01_list_failure_modes",
)

_TRIAL_SUFFIX_RE = re.compile(r"_run(\d{2})\.json$")
_FILENAME_PREFIX_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<cell>[A-Z])_(?P<model>[a-z0-9-]+?)_(?P<rest>.+)$"
)


def _strip_trial_suffix(filename: str) -> str:
    """Drop the ``_runNN.json`` suffix from a per-trial filename."""
    return _TRIAL_SUFFIX_RE.sub("", filename)


def _model_from_filename(filename: str) -> str | None:
    """Best-effort model identifier derived from the filename pattern.

    Filename pattern is ``<date>_<cell>_<model_short>_<orch>_...``. Returns
    the model_short token (e.g. ``llama-3-1-8b-instruct``). Returns
    ``None`` if the pattern doesn't match or if no orchestration token
    follows.
    """
    stem = _strip_trial_suffix(filename)
    m = _FILENAME_PREFIX_RE.match(stem)
    if not m:
        return None
    rest = m.group("rest")
    # Truncate model_short at the first orchestration marker to recover
    # just the model name. The model token is hyphen-only by convention
    # but the regex above is non-greedy, so ``rest`` already starts at the
    # orchestration marker. The match's ``model`` group is the truth.
    if any(rest.startswith(f"{orch}_") or rest.startswith(orch)
           for orch in ("agent_as_tool", "verified_pe", "plan_execute")):
        return m.group("model")
    return None


def _scenario_basename_from_filename(
    filename: str,
    scenario_basenames: tuple[str, ...],
) -> str | None:
    """Heuristic: return the scenario basename embedded in the filename.

    Matches by longest-suffix-of-stem against the known scenario list.
    Returns ``None`` if no scenario in ``scenario_basenames`` is a suffix
    of the trial-stripped stem.
    """
    stem = _strip_trial_suffix(filename)
    matches = sorted(
        (s for s in scenario_basenames if stem.endswith(s)),
        key=len,
        reverse=True,
    )
    return matches[0] if matches else None


def _runner_label_from_filename(filename: str) -> str | None:
    """Best-effort runner identifier derived from the filename pattern.

    Returns labels like ``agent_as_tool_direct``,
    ``agent_as_tool_baseline``, ``plan_execute_baseline``,
    ``verified_pe_baseline``. Returns ``None`` if no orchestration token
    is found.
    """
    stem = _strip_trial_suffix(filename)
    # Try longest orchestration token first to avoid plan_execute matching
    # inside verified_pe or vice versa. ``agent_as_tool`` is most specific.
    for orch in ("agent_as_tool", "verified_pe", "plan_execute"):
        marker = f"_{orch}_"
        idx = stem.find(marker)
        if idx < 0:
            continue
        # After the orchestration marker, the next underscore-separated
        # token is the mcp_mode (direct/baseline/optimized/...). The rest
        # is the scenario basename and is intentionally dropped here.
        after = stem[idx + len(marker):]
        mcp_mode = after.split("_", 1)[0] if after else ""
        return f"{orch}_{mcp_mode}".rstrip("_")
    return None


def from_per_trial_dict(
    data: dict,
    *,
    run_id: str,
    runner: str | None = None,
    model: str | None = None,
    scenario_id_fallback: str | None = None,
) -> PersistedTrajectory:
    """Build a :class:`PersistedTrajectory` from a Smart Grid Bench per-trial dict.

    Parameters
    ----------
    data
        The parsed JSON dict from one ``<run-id>/<file>_runNN.json`` file.
    run_id
        Run identifier (typically the parent run-dir name, e.g.
        ``8979314_aat_direct``).
    runner
        Caller-provided runner label. If ``None``, derived from
        ``data["runner_meta"]["mcp_mode"]`` (best-effort) or left
        ``"unknown"``.
    model
        Caller-provided model identifier. If ``None``, derived from
        ``data["runner_meta"]["model_id"]`` (AaT shape) or left
        ``"unknown"``. Callers can pass a filename-derived model fallback
        for PE-family trials whose JSON omits ``runner_meta``.
    scenario_id_fallback
        Used only when the per-trial JSON does not carry a canonical
        ``scenario`` dict (legacy shape). Pass the scenario basename
        recovered from the filename or any other source.
    """
    runner_meta = data.get("runner_meta") or {}

    if model is None:
        model = str(runner_meta.get("model_id") or "unknown")
    else:
        model = str(model)

    scenario_obj = data.get("scenario")
    if isinstance(scenario_obj, dict) and scenario_obj.get("id") is not None:
        scenario_id: str | None = str(scenario_obj["id"])
    elif scenario_id_fallback is not None:
        scenario_id = scenario_id_fallback
    else:
        scenario_id = None

    if runner is None:
        mcp_mode = runner_meta.get("mcp_mode")
        runner = f"aat_{mcp_mode}" if mcp_mode else "unknown"

    # The canonical AOB ``trajectory`` field accepts ``Any``; preserve the
    # full team-repo per-trial payload (minus question/answer/scenario
    # which live as top-level PersistedTrajectory fields) so that
    # ``metrics._from_sdk_trajectory`` (looks for ``{"turns": [...]}``) and
    # ``_from_plan_execute`` (looks for a step list) have the same shape
    # they would in an AOB-native run, plus team-specific extras
    # (``failed_tools``, ``max_turns_exhausted``, ``runner_meta``, Verified
    # PE's ``verification``, the team PE shape's top-level ``trajectory``
    # step list).
    trajectory: dict = {
        k: v for k, v in data.items()
        if k not in {"question", "answer", "scenario"}
    }

    return PersistedTrajectory(
        run_id=run_id,
        scenario_id=scenario_id,
        runner=str(runner),
        model=model,
        question=str(data.get("question") or ""),
        answer=str(data.get("answer") or ""),
        trajectory=trajectory,
    )


def load_team_run_dir(
    run_dir: Path | str,
    *,
    scenario_basenames: tuple[str, ...] = SCENARIO_BASENAMES_DEFAULT,
) -> list[PersistedTrajectory]:
    """Load every per-trial JSON under a Smart Grid Bench run directory.

    ``run_dir`` is expected to point at
    ``benchmarks/cell_<X>/raw/<run-id>/``. The directory's basename
    becomes the ``run_id`` on every emitted record. Files that fail to
    parse or that do not match the expected ``_runNN.json`` pattern are
    skipped with a log entry.
    """
    p = Path(run_dir)
    run_id = p.name
    out: list[PersistedTrajectory] = []
    for child in sorted(p.iterdir()):
        if not _TRIAL_SUFFIX_RE.search(child.name):
            continue
        try:
            data = json.loads(child.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _log.exception("sg_per_trial: failed to parse %s", child)
            continue
        scenario_fallback = _scenario_basename_from_filename(
            child.name, scenario_basenames
        )
        runner_label = _runner_label_from_filename(child.name)
        model_fallback = _model_from_filename(child.name)
        try:
            out.append(
                from_per_trial_dict(
                    data,
                    run_id=run_id,
                    runner=runner_label,
                    model=model_fallback,
                    scenario_id_fallback=scenario_fallback,
                )
            )
        except Exception:  # noqa: BLE001 — keep the rest of the dir loadable
            _log.exception("sg_per_trial: failed to adapt %s", child)
            continue
    return out
