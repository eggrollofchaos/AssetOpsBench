"""Tests for the Self-Ask clarification helper."""

from __future__ import annotations

import json

import pytest

from agent.plan_execute.self_ask import (
    SELF_ASK_PROMPT,
    SelfAskDecision,
    _parse_json_object,
    maybe_self_ask,
)


class _FakeLLM:
    """Tiny LLMBackend stand-in: returns canned text on each generate() call."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.response


class _FailingLLM:
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        raise RuntimeError("LLM unavailable")


# --------------------------------------------------------------------------- _parse_json_object


def test_parse_json_object_clean():
    assert _parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_with_markdown_fence():
    raw = '```json\n{"a": 1}\n```'
    assert _parse_json_object(raw) == {"a": 1}


def test_parse_json_object_with_prose_around():
    raw = 'Here is your JSON:\n{"a": 1}\nHope that helps!'
    assert _parse_json_object(raw) == {"a": 1}


def test_parse_json_object_returns_empty_on_garbage():
    assert _parse_json_object("nope, not json") == {}


def test_parse_json_object_returns_empty_on_array():
    # We expect a dict; arrays count as "not a dict".
    assert _parse_json_object("[1, 2, 3]") == {}


# --------------------------------------------------------------------------- maybe_self_ask


def test_maybe_self_ask_no_clarification_needed():
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": False,
                "clarifying_questions": [],
                "assumptions": [],
                "augmented_question": "",
            }
        )
    )
    decision = maybe_self_ask("What's the RUL of T-015?", llm)
    assert decision.needs_self_ask is False
    assert decision.clarifying_questions == []
    assert decision.assumptions == []
    # Falls back to original question when augmented_question is empty.
    assert decision.augmented_question == "What's the RUL of T-015?"


def test_maybe_self_ask_clarification_needed_with_augmented():
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": True,
                "clarifying_questions": ["Which asset?", "What time window?"],
                "assumptions": ["Recent readings preferred"],
                "augmented_question": "Investigate T-015 over the last 7 days.",
            }
        )
    )
    decision = maybe_self_ask("Investigate T-015.", llm)
    assert decision.needs_self_ask is True
    assert decision.clarifying_questions == ["Which asset?", "What time window?"]
    assert decision.assumptions == ["Recent readings preferred"]
    assert decision.augmented_question == "Investigate T-015 over the last 7 days."


def test_maybe_self_ask_builds_augmented_when_llm_omits_it():
    """LLM said clarification was needed but didn't fill in augmented_question."""
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": True,
                "clarifying_questions": ["Which asset?"],
                "assumptions": ["Recent only"],
                "augmented_question": "",
            }
        )
    )
    decision = maybe_self_ask("Investigate.", llm)
    assert decision.needs_self_ask is True
    # Augmented should contain the original question and the clarifications
    # built in.
    assert "Investigate." in decision.augmented_question
    assert "Which asset?" in decision.augmented_question
    assert "Recent only" in decision.augmented_question


def test_maybe_self_ask_caps_clarifying_questions_at_2():
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": True,
                "clarifying_questions": ["q1", "q2", "q3", "q4"],
                "assumptions": [],
                "augmented_question": "augmented",
            }
        )
    )
    decision = maybe_self_ask("q?", llm)
    assert decision.clarifying_questions == ["q1", "q2"]


def test_maybe_self_ask_caps_assumptions_at_3():
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": True,
                "clarifying_questions": [],
                "assumptions": ["a1", "a2", "a3", "a4", "a5"],
                "augmented_question": "augmented",
            }
        )
    )
    decision = maybe_self_ask("q?", llm)
    assert decision.assumptions == ["a1", "a2", "a3"]


def test_maybe_self_ask_falls_back_on_llm_error():
    decision = maybe_self_ask("Question", _FailingLLM())
    assert decision.needs_self_ask is False
    assert decision.augmented_question == "Question"


def test_maybe_self_ask_falls_back_on_garbage_json():
    decision = maybe_self_ask("Question", _FakeLLM("not json at all"))
    assert decision.needs_self_ask is False
    assert decision.augmented_question == "Question"


def test_maybe_self_ask_strips_whitespace_in_lists():
    llm = _FakeLLM(
        json.dumps(
            {
                "needs_self_ask": True,
                "clarifying_questions": ["  trim me  ", "", "   "],
                "assumptions": [" a1 ", ""],
                "augmented_question": "ok",
            }
        )
    )
    decision = maybe_self_ask("q?", llm)
    assert decision.clarifying_questions == ["trim me"]
    assert decision.assumptions == ["a1"]


def test_self_ask_prompt_includes_question_placeholder():
    """SELF_ASK_PROMPT must accept a question via .format()."""
    formatted = SELF_ASK_PROMPT.format(question="test question")
    assert "test question" in formatted


def test_maybe_self_ask_returns_dataclass_instance():
    llm = _FakeLLM(json.dumps({"needs_self_ask": False}))
    decision = maybe_self_ask("q", llm)
    assert isinstance(decision, SelfAskDecision)
