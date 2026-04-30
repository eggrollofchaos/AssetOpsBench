"""Tests for the Verified Plan-Execute helpers (verify_step, replan utils, renumber)."""

from __future__ import annotations

import json

import pytest

from agent.plan_execute.models import Plan, PlanStep
from agent.plan_execute.verified import (
    VERIFIER_PROMPT,
    VerificationDecision,
    build_retry_question,
    build_suffix_replan_question,
    renumber_plan,
    verify_step,
    _compact,
    _compact_history,
    _compact_steps,
    _serialize_step,
)


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.response


class _FailingLLM:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise RuntimeError("verifier unavailable")


def _step(n: int = 1, task: str = "do thing") -> PlanStep:
    return PlanStep(
        step_number=n,
        task=task,
        server="iot",
        tool="list_assets",
        tool_args={},
        dependencies=[],
        expected_output="",
    )


# --------------------------------------------------------------------------- verify_step


def test_verify_step_continue_decision():
    llm = _FakeLLM(
        json.dumps(
            {"decision": "continue", "reason": "looks good", "updated_focus": ""}
        )
    )
    verdict = verify_step(
        question="q?",
        effective_question="q?",
        current_step=_step(),
        current_result={"step": 1, "success": True, "response": "ok"},
        history=[],
        remaining_steps=[],
        llm=llm,
    )
    assert verdict.decision == "continue"
    assert verdict.reason == "looks good"


def test_verify_step_retry_decision_with_focus():
    llm = _FakeLLM(
        json.dumps(
            {
                "decision": "retry",
                "reason": "missed key sensor",
                "updated_focus": "include all sensors",
            }
        )
    )
    verdict = verify_step(
        "q", "q", _step(), {"step": 1, "success": True, "response": "ok"}, [], [], llm
    )
    assert verdict.decision == "retry"
    assert verdict.updated_focus == "include all sensors"


def test_verify_step_replan_suffix_decision():
    llm = _FakeLLM(
        json.dumps(
            {
                "decision": "replan_suffix",
                "reason": "context shifted",
                "updated_focus": "now go to FMSR",
            }
        )
    )
    verdict = verify_step(
        "q", "q", _step(), {"step": 1, "success": True, "response": "ok"}, [], [], llm
    )
    assert verdict.decision == "replan_suffix"


def test_verify_step_unknown_decision_falls_back_to_continue():
    llm = _FakeLLM(json.dumps({"decision": "abandon", "reason": "weird"}))
    verdict = verify_step(
        "q", "q", _step(), {"step": 1, "success": True, "response": "ok"}, [], [], llm
    )
    assert verdict.decision == "continue"


def test_verify_step_garbage_json_falls_back_to_continue():
    llm = _FakeLLM("not json")
    verdict = verify_step(
        "q", "q", _step(), {"step": 1, "success": True, "response": "ok"}, [], [], llm
    )
    assert verdict.decision == "continue"


def test_verify_step_llm_error_falls_back_to_continue():
    verdict = verify_step(
        "q",
        "q",
        _step(),
        {"step": 1, "success": True, "response": "ok"},
        [],
        [],
        _FailingLLM(),
    )
    assert verdict.decision == "continue"
    assert "unavailable" in verdict.reason.lower()


def test_verify_step_blank_reason_gets_default():
    llm = _FakeLLM(json.dumps({"decision": "continue", "reason": "  "}))
    verdict = verify_step(
        "q", "q", _step(), {"step": 1, "success": True, "response": "ok"}, [], [], llm
    )
    assert verdict.reason  # non-empty default applied


# --------------------------------------------------------------------------- prompt builders


def test_build_retry_question_includes_step_and_reason():
    decision = VerificationDecision(
        decision="retry", reason="missed key sensor", updated_focus="add T-015"
    )
    prompt = build_retry_question(
        question="orig question?",
        effective_question="effective question",
        current_step=_step(n=2, task="task two"),
        current_result={"response": "previous response"},
        decision=decision,
        retries_used=1,
    )
    assert "orig question?" in prompt
    assert "task two" in prompt
    assert "missed key sensor" in prompt
    assert "add T-015" in prompt
    assert "Retry attempt number: 1" in prompt


def test_build_suffix_replan_question_includes_history_and_focus():
    history = [
        {
            "step": 1,
            "success": True,
            "task": "list assets",
            "tool": "list_assets",
            "response": "T-015 listed",
        }
    ]
    decision = VerificationDecision(
        decision="replan_suffix", reason="pivot to FMSR", updated_focus="DGA"
    )
    prompt = build_suffix_replan_question(
        question="orig?",
        effective_question="eff?",
        history=history,
        remaining_steps=[_step(n=2)],
        decision=decision,
    )
    assert "Replan only the remaining suffix" in prompt
    assert "orig?" in prompt
    assert "T-015 listed" in prompt
    assert "DGA" in prompt
    assert "Updated focus" in prompt


# --------------------------------------------------------------------------- renumber_plan


def test_renumber_plan_shifts_step_numbers_and_dependencies():
    plan = Plan(
        steps=[
            PlanStep(1, "a", "iot", "t1", {}, [], ""),
            PlanStep(2, "b", "iot", "t2", {}, [1], ""),
            PlanStep(3, "c", "iot", "t3", {}, [1, 2], ""),
        ],
        raw="raw",
    )
    shifted = renumber_plan(plan, offset=10)
    assert [s.step_number for s in shifted.steps] == [11, 12, 13]
    assert shifted.steps[1].dependencies == [11]
    assert shifted.steps[2].dependencies == [11, 12]
    # raw is preserved.
    assert shifted.raw == "raw"


def test_renumber_plan_offset_zero_is_identity():
    plan = Plan(steps=[PlanStep(1, "a", "iot", "t", {}, [], "")], raw="r")
    shifted = renumber_plan(plan, offset=0)
    assert shifted.steps[0].step_number == 1


# --------------------------------------------------------------------------- helpers


def test_compact_truncates_long_text_with_marker():
    out = _compact("x" * 1000, limit=100)
    assert len(out) > 100  # the truncation marker adds chars
    assert out.startswith("x" * 100)
    assert "truncated" in out


def test_compact_returns_short_text_unchanged():
    assert _compact("short", limit=100) == "short"


def test_compact_handles_none():
    assert _compact(None, limit=100) == ""


def test_compact_history_summarizes_each_entry():
    history = [
        {"step": 1, "success": True, "task": "list", "tool": "list_assets", "response": "ok"},
        {"step": 2, "success": False, "task": "fetch", "tool": "get_metadata", "error": "boom"},
    ]
    out = _compact_history(history)
    assert "[OK]" in out
    assert "[ERR]" in out
    assert "list_assets" in out
    assert "boom" in out


def test_compact_history_empty_returns_none_marker():
    assert _compact_history([]) == "(none)"


def test_compact_steps_lists_each_step():
    steps = [_step(n=1, task="first"), _step(n=2, task="second")]
    out = _compact_steps(steps)
    assert "first" in out
    assert "second" in out


def test_compact_steps_empty_returns_none_marker():
    assert _compact_steps([]) == "(none)"


def test_serialize_step_includes_extras():
    from agent.plan_execute.models import StepResult

    sr = StepResult(
        step_number=1,
        task="t",
        server="iot",
        response="ok",
        tool="list_assets",
        tool_args={},
    )
    payload = _serialize_step(sr, verifier_decision="continue")
    assert payload["verifier_decision"] == "continue"
    assert payload["step"] == 1
    assert payload["success"] is True


def test_verifier_prompt_accepts_all_placeholders():
    """VERIFIER_PROMPT must be format()-able with the helper signature."""
    formatted = VERIFIER_PROMPT.format(
        question="q",
        effective_question="eq",
        current_step="step desc",
        current_result="{}",
        history="(none)",
        remaining_steps="(none)",
    )
    assert "q" in formatted
    assert "step desc" in formatted
