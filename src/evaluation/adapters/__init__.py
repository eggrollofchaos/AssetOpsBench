"""Per-source-shape trajectory adapters for the offline evaluation pipeline.

Each module under ``adapters/`` accepts an external benchmark's per-trial
JSON shape and emits :class:`evaluation.models.PersistedTrajectory` records
that the rest of the evaluation pipeline can consume uniformly.
"""

from .sg_per_trial import (
    from_per_trial_dict,
    load_team_run_dir,
    SCENARIO_BASENAMES_DEFAULT,
)

__all__ = [
    "from_per_trial_dict",
    "load_team_run_dir",
    "SCENARIO_BASENAMES_DEFAULT",
]
