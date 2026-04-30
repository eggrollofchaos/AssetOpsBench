"""Self-Ask clarification pass for plan-execute pipelines.

A lightweight pre-planning step that asks the LLM whether the user's question
is specific enough to plan against, and, if not, augments the question with
internally-resolved clarifying notes. The result is fed into
:meth:`Planner.generate_plan` instead of the raw question.

Originally developed in the HPML Smart Grid MCP team's
``scripts/orchestration_utils.py`` and ported here as a generic AOB helper
because the design is domain-agnostic: any plan-execute workflow can benefit
from a one-shot internal clarification pass.

Usage::

    from agent.plan_execute.self_ask import maybe_self_ask
    from llm import LiteLLMBackend

    llm = LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")
    decision = maybe_self_ask("What's wrong with T-015?", llm)
    plan = planner.generate_plan(decision.augmented_question, descriptions)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

_log = logging.getLogger(__name__)


SELF_ASK_PROMPT = """\
You are deciding whether an industrial asset-operations question needs a brief
internal clarification pass before tool planning.

Return a single raw JSON object with exactly these keys:
- needs_self_ask: boolean
- clarifying_questions: list of at most 2 short strings
- assumptions: list of at most 3 short strings
- augmented_question: string

Rules:
- Use needs_self_ask=false when the original question is already specific enough.
- If needs_self_ask=false, set augmented_question to the original question.
- If needs_self_ask=true, augmented_question should keep the original question
  intact while appending the clarification points the planner should resolve
  internally before answering.
- Do not ask the human user for clarification. This is an internal planning aid.

Question:
{question}

JSON:
"""


@dataclass
class SelfAskDecision:
    """Outcome of one Self-Ask pass.

    Attributes:
        needs_self_ask: True iff the LLM judged the question as needing
            clarification before planning.
        clarifying_questions: Up to two short clarifying questions the LLM
            generated. Empty when ``needs_self_ask`` is False.
        assumptions: Up to three temporary assumptions the LLM proposed.
            Empty when ``needs_self_ask`` is False.
        augmented_question: The question to plan against. Equals the original
            question when ``needs_self_ask`` is False; otherwise carries the
            original question plus the clarification points.
    """

    needs_self_ask: bool
    clarifying_questions: list[str]
    assumptions: list[str]
    augmented_question: str


class _GenerativeLLM(Protocol):
    """Minimal LLM contract: a synchronous ``.generate(prompt)`` returning a string."""

    def generate(self, prompt: str, temperature: float = 0.0) -> str: ...


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Best-effort JSON-object parser tolerant of markdown fences and prose
    surrounding the JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                value = json.loads(text[start:end])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def maybe_self_ask(question: str, llm: _GenerativeLLM) -> SelfAskDecision:
    """Run one Self-Ask pass against ``question`` using ``llm``.

    The LLM is asked to return a structured JSON decision. Failures (LLM
    error, malformed JSON) fall back to ``needs_self_ask=False`` so the
    caller can proceed with the raw question.

    Args:
        question: The user's original question.
        llm: Any object exposing ``generate(prompt, temperature=0.0) -> str``.

    Returns:
        A :class:`SelfAskDecision`. ``augmented_question`` is always set —
        it equals ``question`` when no clarification is needed.
    """
    try:
        raw = llm.generate(SELF_ASK_PROMPT.format(question=question))
    except Exception as exc:  # noqa: BLE001 — never let Self-Ask break planning
        _log.warning("Self-Ask failed (%s); falling back to no-op decision.", exc)
        return SelfAskDecision(
            needs_self_ask=False,
            clarifying_questions=[],
            assumptions=[],
            augmented_question=question,
        )

    payload = _parse_json_object(raw)
    needs_self_ask = bool(payload.get("needs_self_ask", False))

    clarifying_questions = [
        str(item).strip()
        for item in payload.get("clarifying_questions", [])
        if str(item).strip()
    ][:2]
    assumptions = [
        str(item).strip()
        for item in payload.get("assumptions", [])
        if str(item).strip()
    ][:3]
    augmented_question = str(payload.get("augmented_question", "")).strip()

    if not needs_self_ask:
        return SelfAskDecision(
            needs_self_ask=False,
            clarifying_questions=[],
            assumptions=[],
            augmented_question=question,
        )

    if not augmented_question:
        # The LLM said clarification was needed but didn't fill in the
        # augmented question. Build one from clarifying_questions +
        # assumptions so planning still has the extra context.
        extra = []
        if clarifying_questions:
            extra.append(
                "Resolve these clarification questions internally before answering:\n- "
                + "\n- ".join(clarifying_questions)
            )
        if assumptions:
            extra.append(
                "Use these temporary assumptions if needed:\n- "
                + "\n- ".join(assumptions)
            )
        augmented_question = question + "\n\n" + "\n\n".join(extra)

    return SelfAskDecision(
        needs_self_ask=True,
        clarifying_questions=clarifying_questions,
        assumptions=assumptions,
        augmented_question=augmented_question,
    )
