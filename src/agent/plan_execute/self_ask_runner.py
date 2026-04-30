"""PlanExecuteSelfAskRunner — PlanExecuteRunner with a Self-Ask pre-pass.

Subclass of :class:`PlanExecuteRunner` that runs a one-shot Self-Ask
clarification pass before planning, so the planner sees an augmented
question instead of a possibly-underspecified one.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from llm import LLMBackend
from observability import agent_run_span, persist_trajectory

from .models import OrchestratorResult
from .runner import PlanExecuteRunner, _SUMMARIZE_PROMPT
from .self_ask import SelfAskDecision, maybe_self_ask

_log = logging.getLogger(__name__)


class PlanExecuteSelfAskRunner(PlanExecuteRunner):
    """PlanExecuteRunner with a Self-Ask clarification pass.

    Args:
        llm: LLM backend used for planning, tool selection, summarisation,
            and the Self-Ask pass.
        server_paths: Override MCP server specs (forwarded to the parent
            ``PlanExecuteRunner``).
        enable_self_ask: When False, the Self-Ask pass is skipped entirely
            and behaviour matches the base ``PlanExecuteRunner`` (default
            True).

    The augmented question (when Self-Ask fires) is fed to the planner;
    the original question is preserved on the returned
    :class:`OrchestratorResult` so the user-visible question text doesn't
    drift.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        *,
        enable_self_ask: bool = True,
    ) -> None:
        super().__init__(llm, server_paths)
        self._enable_self_ask = enable_self_ask

    async def run(self, question: str) -> OrchestratorResult:
        with agent_run_span(
            "plan-execute-self-ask",
            model=self._llm.model_id,
            question=question,
        ) as span:
            run_started = time.perf_counter()
            self._meter.reset()

            # 0. Self-Ask pass (no-op when disabled).
            if self._enable_self_ask:
                self_ask = maybe_self_ask(question, self._meter)
            else:
                self_ask = SelfAskDecision(
                    needs_self_ask=False,
                    clarifying_questions=[],
                    assumptions=[],
                    augmented_question=question,
                )

            # 1. Discover.
            _log.info("Discovering server capabilities...")
            server_descriptions = await self._executor.get_server_descriptions()

            # 2. Plan against the augmented question.
            _log.info("Planning...")
            planning_started = time.perf_counter()
            plan = self._planner.generate_plan(
                self_ask.augmented_question, server_descriptions
            )
            planning_ms = (time.perf_counter() - planning_started) * 1000
            _log.info("Plan has %d step(s).", len(plan.steps))

            # 3. Execute.
            trajectory = await self._executor.execute_plan(
                plan, self_ask.augmented_question
            )

            # 4. Summarise against the ORIGINAL question — the answer is
            #    user-visible, not planner-visible.
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

            result = OrchestratorResult(
                question=question,
                answer=answer,
                plan=plan,
                trajectory=trajectory,
            )
            span.set_attribute("agent.plan.steps", len(plan.steps))
            span.set_attribute("agent.answer.length", len(answer or ""))
            span.set_attribute("agent.duration_ms", duration_ms)
            span.set_attribute("agent.planning_time_ms", planning_ms)
            span.set_attribute("agent.summarization_time_ms", summarization_ms)
            span.set_attribute("agent.self_ask.fired", self_ask.needs_self_ask)
            span.set_attribute("gen_ai.usage.input_tokens", self._meter.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", self._meter.output_tokens)
            span.set_attribute("agent.llm_time_ms", planning_ms + summarization_ms)
            persist_trajectory(
                runner_name="plan-execute-self-ask",
                model=self._llm.model_id,
                question=question,
                answer=answer or "",
                trajectory=trajectory,
            )
            return result
