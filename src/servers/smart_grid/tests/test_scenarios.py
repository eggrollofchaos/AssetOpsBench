"""Smart Grid scenario JSON shape tests.

Verifies the Phase-2 conversion of single-file scenarios → AOB array format
keeps every record loadable as a :class:`evaluation.models.Scenario`.

These tests skip when ``src/evaluation/`` is unavailable (Phase 1 branch
not merged into Phase 2 branch yet — the evaluation module lives on
``aob/sg-evaluation-adapter`` while these scenarios live on
``aob/sg-domain-port``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip the entire module if Phase 1's evaluation module isn't present on the
# current branch.
pytest.importorskip("evaluation.models", reason="evaluation/ from Phase 1 not on this branch")

from evaluation.models import Scenario  # noqa: E402  (import-after-skip)


_SCENARIOS_DIR = Path(__file__).resolve().parents[3] / "scenarios" / "local"


def _load(filename: str) -> list[dict]:
    path = _SCENARIOS_DIR / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list), f"{filename} must be a JSON array"
    return raw


def test_smart_grid_scenarios_count():
    records = _load("smart_grid.json")
    assert len(records) == 11


def test_smart_grid_negative_checks_count():
    records = _load("smart_grid_negative_checks.json")
    assert len(records) == 5


def test_smart_grid_scenarios_validate_via_aob_scenario_model():
    for raw in _load("smart_grid.json"):
        scenario = Scenario.from_raw(raw)
        assert scenario.id
        assert scenario.text
        # Permissive extra='allow' should preserve domain-specific fields.
        assert hasattr(scenario, "asset_id") or "asset_id" not in raw


def test_smart_grid_negative_checks_validate_via_aob_scenario_model():
    for raw in _load("smart_grid_negative_checks.json"):
        scenario = Scenario.from_raw(raw)
        assert scenario.id.startswith("SG-NEG-")


def test_smart_grid_scenario_ids_unique():
    main = _load("smart_grid.json")
    neg = _load("smart_grid_negative_checks.json")
    ids = [r["id"] for r in main + neg]
    assert len(ids) == len(set(ids)), f"duplicate scenario IDs detected: {ids}"
