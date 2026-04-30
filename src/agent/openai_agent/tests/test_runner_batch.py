"""Unit tests for OpenAIAgentRunner.run_batch + parallel_tool_calls.

Mirrors the team-repo Cell C optimized batch mode (PR #134's
`scripts/aat_runner.py::_main_multi`): MCP servers built ONCE and reused
across every prompt × trial inside one AsyncExitStack context. Patches
`agents.Runner.run` so no real API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.models import AgentResult
from agent.openai_agent.runner import OpenAIAgentRunner


# --------------------------------------------------------------------------- helpers


def _make_run_result(answer: str = "ok"):
    return SimpleNamespace(
        new_items=[],
        raw_responses=[],
        final_output=answer,
    )


def _runner(monkeypatch, **overrides) -> OpenAIAgentRunner:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    kwargs = dict(server_paths={"iot": "iot-mcp-server", "fmsr": "fmsr-mcp-server"})
    kwargs.update(overrides)
    return OpenAIAgentRunner(**kwargs)


# --------------------------------------------------------------------------- __init__


def test_parallel_tool_calls_default_false(monkeypatch):
    runner = _runner(monkeypatch)
    assert runner._parallel_tool_calls is False


def test_parallel_tool_calls_explicit_true(monkeypatch):
    runner = _runner(monkeypatch, parallel_tool_calls=True)
    assert runner._parallel_tool_calls is True


def test_parallel_tool_calls_explicit_none(monkeypatch):
    runner = _runner(monkeypatch, parallel_tool_calls=None)
    assert runner._parallel_tool_calls is None


def test_build_agent_passes_model_settings(monkeypatch):
    runner = _runner(monkeypatch, parallel_tool_calls=True)
    fake_servers = [MagicMock(), MagicMock()]
    with patch("agent.openai_agent.runner.Agent") as agent_cls:
        runner._build_agent(fake_servers)
    agent_cls.assert_called_once()
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["mcp_servers"] == fake_servers
    assert kwargs["model_settings"].parallel_tool_calls is True


# --------------------------------------------------------------------------- run_batch


@pytest.mark.anyio
async def test_run_batch_rejects_zero_trials(monkeypatch):
    runner = _runner(monkeypatch)
    with pytest.raises(ValueError, match="trials must be >= 1"):
        await runner.run_batch(prompts=["q"], trials=0)


@pytest.mark.anyio
async def test_run_batch_returns_result_per_prompt_trial(monkeypatch):
    runner = _runner(monkeypatch)
    fake_runner_run = AsyncMock(return_value=_make_run_result("answer-x"))

    # Patch _build_mcp_servers to return MagicMocks supporting async context
    # manager protocol. AsyncExitStack will call __aenter__/__aexit__.
    fake_servers = [_async_ctx_server() for _ in range(2)]
    with (
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=fake_servers),
        patch("agent.openai_agent.runner.Runner.run", fake_runner_run),
    ):
        results = await runner.run_batch(
            prompts=["prompt-A", "prompt-B"],
            trials=3,
        )

    assert len(results) == 2 * 3
    # Order is prompt-major: A trials 1-3 then B trials 1-3.
    assert [r.question for r in results] == [
        "prompt-A", "prompt-A", "prompt-A",
        "prompt-B", "prompt-B", "prompt-B",
    ]
    for r in results:
        assert isinstance(r, AgentResult)
        assert r.error is None
        assert r.answer == "answer-x"
    # Runner.run called once per (prompt, trial).
    assert fake_runner_run.await_count == 6


@pytest.mark.anyio
async def test_run_batch_reuses_mcp_servers_across_trials(monkeypatch):
    """Each MCP server must be entered exactly once for the whole batch."""
    runner = _runner(monkeypatch)
    fake_runner_run = AsyncMock(return_value=_make_run_result())

    fake_servers = [_async_ctx_server(name=f"srv-{i}") for i in range(2)]
    with (
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=fake_servers),
        patch("agent.openai_agent.runner.Runner.run", fake_runner_run),
    ):
        await runner.run_batch(prompts=["p1", "p2"], trials=4)

    # Servers were each entered exactly once and exited exactly once,
    # not once per (prompt × trial).
    for server in fake_servers:
        assert server.aenter_count == 1, f"{server.name} entered {server.aenter_count}x"
        assert server.aexit_count == 1, f"{server.name} exited {server.aexit_count}x"

    assert fake_runner_run.await_count == 2 * 4


@pytest.mark.anyio
async def test_run_batch_isolates_per_trial_failure(monkeypatch):
    """One trial raising should not abort the batch."""
    runner = _runner(monkeypatch)

    call_count = {"n": 0}

    async def flaky_run(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated trial failure")
        return _make_run_result(answer=f"answer-{call_count['n']}")

    fake_servers = [_async_ctx_server()]
    with (
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=fake_servers),
        patch("agent.openai_agent.runner.Runner.run", side_effect=flaky_run),
    ):
        results = await runner.run_batch(prompts=["q"], trials=3)

    assert len(results) == 3
    assert results[0].error is None
    assert results[0].answer == "answer-1"
    assert results[1].error is not None
    assert "RuntimeError" in results[1].error
    assert "simulated trial failure" in results[1].error
    assert results[1].answer == ""
    assert results[2].error is None


@pytest.mark.anyio
async def test_run_batch_respects_parallel_tool_calls_setting(monkeypatch):
    runner = _runner(monkeypatch, parallel_tool_calls=True)
    fake_runner_run = AsyncMock(return_value=_make_run_result())

    fake_servers = [_async_ctx_server()]
    with (
        patch("agent.openai_agent.runner._build_mcp_servers", return_value=fake_servers),
        patch("agent.openai_agent.runner.Runner.run", fake_runner_run),
        patch("agent.openai_agent.runner.Agent") as agent_cls,
    ):
        await runner.run_batch(prompts=["q"], trials=1)

    # Agent was constructed exactly once for the batch and got the
    # parallel_tool_calls=True ModelSettings.
    agent_cls.assert_called_once()
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["model_settings"].parallel_tool_calls is True


# --------------------------------------------------------------------------- helpers


class _FakeMCPServer:
    """Minimal stand-in for MCPServerStdio that tracks aenter/aexit counts.

    AsyncExitStack.enter_async_context() resolves __aenter__/__aexit__ via
    the class, so SimpleNamespace doesn't work — they need to be class
    methods or descriptors. Hand-rolled class is the cleanest fixture.
    """

    def __init__(self, name: str = "srv") -> None:
        self.name = name
        self.aenter_count = 0
        self.aexit_count = 0

    async def __aenter__(self):
        self.aenter_count += 1
        return self

    async def __aexit__(self, *_exc):
        self.aexit_count += 1
        return False


def _async_ctx_server(name: str = "srv") -> _FakeMCPServer:
    return _FakeMCPServer(name=name)


@pytest.fixture
def anyio_backend():
    return "asyncio"
