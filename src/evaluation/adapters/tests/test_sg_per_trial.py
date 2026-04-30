"""Tests for the Smart Grid Bench per-trial trajectory adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.adapters import sg_per_trial
from evaluation.adapters.sg_per_trial import (
    _model_from_filename,
    _runner_label_from_filename,
    _scenario_basename_from_filename,
    SCENARIO_BASENAMES_DEFAULT,
    from_per_trial_dict,
    load_team_run_dir,
)


# --------------------------------------------------------------------------- helpers


def _aat_payload() -> dict:
    """Minimal AaT-shaped per-trial JSON payload (post-PR #143 canonical)."""
    return {
        "question": "Investigate transformer T-015.",
        "answer": "Recommend WO-7531 for replacement.",
        "success": True,
        "history": [{"step": 1, "tool": "list_assets"}],
        "failed_tools": [],
        "max_turns_exhausted": False,
        "turn_count": 2,
        "tool_call_count": 5,
        "scenario": {"id": "SGT-009", "type": "transformer_fault"},
        "runner_meta": {
            "model_id": "openai/Llama-3.1-8B-Instruct",
            "mcp_mode": "baseline",
            "max_turns": 30,
            "parallel_tool_calls": False,
        },
    }


def _legacy_aat_payload() -> dict:
    """AaT-shaped per-trial JSON without the canonical scenario field (pre-PR #143)."""
    payload = _aat_payload()
    del payload["scenario"]
    return payload


def _pe_payload() -> dict:
    """PE-shaped per-trial JSON (no runner_meta, with plan + trajectory step list)."""
    return {
        "question": "Investigate T-015.",
        "answer": "DGA fault FM-007.",
        "success": True,
        "plan": ["list_assets", "get_dga", "issue_wo"],
        "trajectory": [
            {"step": 1, "action": "list_assets", "ok": True},
            {"step": 2, "action": "get_dga", "ok": True},
        ],
        "scenario": {"id": "SGT-009", "type": "transformer_fault"},
    }


def _verified_pe_payload() -> dict:
    """Verified PE shape with verification + replans_used."""
    payload = _pe_payload()
    payload["verification"] = {"replans_used": 1, "verified": True}
    return payload


# --------------------------------------------------------------------------- from_per_trial_dict


def test_from_per_trial_dict_canonical_aat():
    rec = from_per_trial_dict(_aat_payload(), run_id="run-x")
    assert rec.run_id == "run-x"
    assert rec.scenario_id == "SGT-009"
    assert rec.runner == "aat_baseline"
    assert rec.model == "openai/Llama-3.1-8B-Instruct"
    assert rec.question.startswith("Investigate")
    assert rec.answer.startswith("Recommend")
    assert isinstance(rec.trajectory, dict)
    assert rec.trajectory.get("history") == [{"step": 1, "tool": "list_assets"}]
    assert rec.trajectory.get("success") is True


def test_from_per_trial_dict_legacy_aat_uses_fallback():
    rec = from_per_trial_dict(
        _legacy_aat_payload(),
        run_id="run-x",
        scenario_id_fallback="multi_01_end_to_end_fault_response",
    )
    assert rec.scenario_id == "multi_01_end_to_end_fault_response"


def test_from_per_trial_dict_legacy_without_fallback_yields_none():
    rec = from_per_trial_dict(_legacy_aat_payload(), run_id="run-x")
    assert rec.scenario_id is None


def test_from_per_trial_dict_pe_uses_caller_model_when_runner_meta_absent():
    rec = from_per_trial_dict(
        _pe_payload(),
        run_id="run-y",
        runner="plan_execute_baseline",
        model="llama-3-1-8b-instruct",
    )
    assert rec.runner == "plan_execute_baseline"
    assert rec.model == "llama-3-1-8b-instruct"
    assert rec.trajectory.get("plan") == ["list_assets", "get_dga", "issue_wo"]
    assert isinstance(rec.trajectory.get("trajectory"), list)


def test_from_per_trial_dict_verified_pe_preserves_verification():
    rec = from_per_trial_dict(_verified_pe_payload(), run_id="run-z")
    assert rec.trajectory.get("verification") == {"replans_used": 1, "verified": True}


def test_from_per_trial_dict_strips_top_level_q_a_scenario_from_trajectory():
    rec = from_per_trial_dict(_aat_payload(), run_id="run-x")
    assert "question" not in rec.trajectory
    assert "answer" not in rec.trajectory
    assert "scenario" not in rec.trajectory


# --------------------------------------------------------------------------- helpers under test


@pytest.mark.parametrize(
    "filename,expected",
    [
        (
            "2026-04-26_A_llama-3-1-8b-instruct_agent_as_tool_direct_multi_01_end_to_end_fault_response_run01.json",
            "multi_01_end_to_end_fault_response",
        ),
        (
            "2026-04-27_Y_llama-3-1-8b-instruct_plan_execute_baseline_iot_02_voltage_imbalance_check_run03.json",
            "iot_02_voltage_imbalance_check",
        ),
        ("not-a-known-pattern.json", None),
    ],
)
def test_scenario_basename_from_filename(filename, expected):
    assert (
        _scenario_basename_from_filename(filename, SCENARIO_BASENAMES_DEFAULT)
        == expected
    )


@pytest.mark.parametrize(
    "filename,expected_runner",
    [
        (
            "2026-04-26_A_llama-3-1-8b-instruct_agent_as_tool_direct_multi_01_end_to_end_fault_response_run01.json",
            "agent_as_tool_direct",
        ),
        (
            "2026-04-27_Y_llama-3-1-8b-instruct_plan_execute_baseline_multi_01_end_to_end_fault_response_run01.json",
            "plan_execute_baseline",
        ),
        (
            "2026-04-27_Z_llama-3-1-8b-instruct_verified_pe_baseline_multi_02_dga_to_workorder_pipeline_run02.json",
            "verified_pe_baseline",
        ),
        ("garbage.json", None),
    ],
)
def test_runner_label_from_filename(filename, expected_runner):
    assert _runner_label_from_filename(filename) == expected_runner


@pytest.mark.parametrize(
    "filename,expected_model",
    [
        (
            "2026-04-26_A_llama-3-1-8b-instruct_agent_as_tool_direct_multi_01_end_to_end_fault_response_run01.json",
            "llama-3-1-8b-instruct",
        ),
        (
            "2026-04-27_Y_some-other-model_plan_execute_baseline_iot_02_voltage_imbalance_check_run01.json",
            "some-other-model",
        ),
        ("garbage.json", None),
    ],
)
def test_model_from_filename(filename, expected_model):
    assert _model_from_filename(filename) == expected_model


# --------------------------------------------------------------------------- load_team_run_dir


def test_load_team_run_dir_writes_full_record(tmp_path: Path):
    run_dir = tmp_path / "8979314_aat_direct"
    run_dir.mkdir()
    file_a = (
        run_dir
        / "2026-04-26_A_llama-3-1-8b-instruct_agent_as_tool_direct_multi_01_end_to_end_fault_response_run01.json"
    )
    file_a.write_text(json.dumps(_aat_payload()))
    file_b = (
        run_dir
        / "2026-04-26_A_llama-3-1-8b-instruct_agent_as_tool_direct_multi_02_dga_to_workorder_pipeline_run02.json"
    )
    file_b.write_text(json.dumps(_legacy_aat_payload()))
    # Non-trial files should be ignored.
    (run_dir / "meta.json").write_text("{}")

    records = load_team_run_dir(run_dir)
    assert len(records) == 2
    by_scenario = {r.scenario_id for r in records}
    # Canonical (file_a) → SGT-009; legacy (file_b) → filename fallback.
    assert "SGT-009" in by_scenario
    assert "multi_02_dga_to_workorder_pipeline" in by_scenario
    for r in records:
        assert r.run_id == "8979314_aat_direct"
        assert r.runner == "agent_as_tool_direct"
        assert r.model == "llama-3-1-8b-instruct"


def test_load_team_run_dir_skips_unparseable(tmp_path: Path, caplog):
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir()
    (run_dir / "x_run01.json").write_text("not-json")
    records = load_team_run_dir(run_dir)
    assert records == []


# --------------------------------------------------------------------------- metrics integration


def _aat_payload_with_role_turns() -> dict:
    """AaT payload with the real role-based turn shape produced by the team runner."""
    return {
        "question": "Investigate T-015.",
        "answer": "WO-7531.",
        "success": True,
        "history": [
            {"role": "user", "content": "Investigate T-015.", "tool_calls": [], "turn": 0},
            {
                "role": "assistant",
                "content": "Looking up sensors.",
                "tool_calls": [
                    {"name": "list_assets", "arguments": "{}", "call_id": "c1", "output": "T-015"},
                    {"name": "get_dga", "arguments": "{}", "call_id": "c2", "output": "FM-007"},
                ],
                "turn": 1,
            },
        ],
        "turn_count": 2,
        "tool_call_count": 2,
        "scenario": {"id": "SGT-009", "type": "transformer_fault"},
        "runner_meta": {"model_id": "openai/Llama-3.1-8B-Instruct", "mcp_mode": "baseline"},
    }


def _pe_payload_with_history_steps() -> dict:
    """PE payload with the real step-result history shape produced by the team runner."""
    return {
        "question": "Investigate T-015.",
        "answer": "DGA fault FM-007.",
        "success": True,
        "plan": ["list_assets", "get_dga"],
        "history": [
            {"step": 1, "task": "list", "tool": "list_assets", "tool_args": {},
             "server": "iot", "response": "T-015", "success": True, "error": None,
             "executor_success": True},
            {"step": 2, "task": "DGA", "tool": "get_dga", "tool_args": {},
             "server": "fmsr", "response": "FM-007", "success": True, "error": None,
             "executor_success": True},
        ],
        "scenario": {"id": "SGT-009", "type": "transformer_fault"},
    }


def test_metrics_from_aat_record_nonzero():
    from evaluation.metrics import metrics_from_trajectory

    rec = from_per_trial_dict(_aat_payload_with_role_turns(), run_id="run-x")
    ops = metrics_from_trajectory(rec)
    assert ops.turn_count == 2
    assert ops.tool_call_count == 2
    assert ops.unique_tools == ["get_dga", "list_assets"]


def test_metrics_from_pe_record_nonzero():
    from evaluation.metrics import metrics_from_trajectory

    rec = from_per_trial_dict(
        _pe_payload_with_history_steps(),
        run_id="run-y",
        runner="plan_execute_baseline",
        model="llama-3-1-8b-instruct",
    )
    ops = metrics_from_trajectory(rec)
    assert ops.turn_count == 2
    assert ops.tool_call_count == 2
    assert ops.unique_tools == ["get_dga", "list_assets"]


def test_metrics_from_pe_record_via_nested_trajectory_key():
    """Some legacy PE captures put the step list under ``trajectory`` instead of ``history``."""
    from evaluation.metrics import metrics_from_trajectory

    payload = _pe_payload_with_history_steps()
    payload["trajectory"] = payload.pop("history")
    rec = from_per_trial_dict(payload, run_id="run-z", runner="plan_execute_baseline",
                              model="llama-3-1-8b-instruct")
    ops = metrics_from_trajectory(rec)
    assert ops.turn_count == 2
    assert ops.tool_call_count == 2
