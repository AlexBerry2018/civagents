"""Журнал вызовов модели: контекст, папка прогона, запись, сводка."""

import json

import httpx
import pytest

from civ_agents.config import LLMProfile
from civ_agents.llm.base import CallRecord
from civ_agents.llm.errors import LLMNoToolCall, LLMTimeout
from civ_agents.llm.fake import FakeLLM
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.llm.types import Message
from civ_agents.observability.call_log import CallLog
from civ_agents.observability.context import call_context, current_context
from civ_agents.observability.run import Run
from civ_agents.observability.summary import load_calls, percentile, summarize


def record(latency_s=1.0, attempts=1, response=None, error=None, warnings=None):
    return CallRecord(
        model="m",
        messages=[Message.user("привет")],
        tools=[],
        latency_s=latency_s,
        attempts=attempts,
        response=response,
        error=error,
        warnings=warnings or [],
    )


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- Контекст -------------------------------------------------------------------


def test_context_nests_and_resets():
    with call_context(turn=5):
        with call_context(agent="economist"):
            assert current_context() == {"turn": 5, "agent": "economist"}
        assert current_context() == {"turn": 5}
    assert current_context() == {}


def test_context_resets_after_exception():
    with pytest.raises(RuntimeError):
        with call_context(turn=1):
            raise RuntimeError
    assert current_context() == {}


# --- Прогон ---------------------------------------------------------------------


def test_run_creates_unique_dirs(tmp_path):
    first = Run.create(tmp_path, name="baseline")
    second = Run.create(tmp_path, name="baseline")

    assert first.dir.is_dir() and second.dir.is_dir()
    assert first.run_id != second.run_id
    assert "baseline" in first.run_id
    assert first.calls_path == first.dir / "llm_calls.jsonl"


# --- Журнал ---------------------------------------------------------------------


def test_successful_call_is_logged_with_context(tmp_path):
    run = Run.create(tmp_path)
    log = CallLog.for_run(run)

    with call_context(turn=3, agent="ruler"):
        log(record(latency_s=1.5, response=FakeLLM.tool("found_city", unit_id=1)))

    [entry] = read_lines(run.calls_path)
    assert entry["run_id"] == run.run_id
    assert (entry["turn"], entry["agent"]) == (3, "ruler")
    assert entry["ok"] is True
    assert entry["tool_calls"] == ["found_city"]
    assert entry["prompt_tokens"] == 10
    assert entry["latency_s"] == 1.5
    assert not run.prompts_path.exists()  # промпты по умолчанию не пишутся


def test_content_error_keeps_model_answer(tmp_path):
    run = Run.create(tmp_path)
    error = LLMNoToolCall("нет вызова", response=FakeLLM.text("Я бы основал город"))
    CallLog.for_run(run, save_prompts=True)(record(error=error))

    [entry] = read_lines(run.calls_path)
    assert entry["ok"] is False
    assert entry["error_type"] == "LLMNoToolCall"
    assert entry["completion_tokens"] == 5  # токены ответа сохранены, хотя вызов неудачный

    [prompt] = read_lines(run.prompts_path)
    assert prompt["call_id"] == entry["call_id"]
    assert prompt["response"]["text"] == "Я бы основал город"
    assert prompt["content_error"] is True


def test_transport_error_without_response(tmp_path):
    run = Run.create(tmp_path)
    CallLog.for_run(run)(record(attempts=3, error=LLMTimeout("сервер молчит")))

    [entry] = read_lines(run.calls_path)
    assert entry["error_type"] == "LLMTimeout"
    assert entry["prompt_tokens"] is None
    assert entry["attempts"] == 3


def test_client_writes_to_log(tmp_path):
    """Связка целиком: клиент → наблюдатель → файл."""
    completion = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion))
    profile = LLMProfile(base_url="http://test/v1", model="m", timeout_s=5, max_output_tokens=64, context_window=1000)
    run = Run.create(tmp_path)
    client = OpenAICompatClient(profile, observer=CallLog.for_run(run), http_client=httpx.Client(transport=transport))

    with call_context(turn=1, agent="ruler"):
        client.chat([Message.user("x")])

    [entry] = read_lines(run.calls_path)
    assert entry["agent"] == "ruler"
    assert entry["prompt_tokens"] == 12


# --- Сводка ---------------------------------------------------------------------


def test_percentile_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([7], 95) == 7
    assert percentile([], 50) == 0.0


def test_summary(tmp_path):
    run = Run.create(tmp_path)
    log = CallLog.for_run(run)
    with call_context(turn=1):
        with call_context(agent="ruler"):
            log(record(latency_s=2.0, response=FakeLLM.text("ok")))
        with call_context(agent="economist"):
            log(record(latency_s=9.0, attempts=3, error=LLMTimeout("t"), warnings=["обрезан"]))
            log(record(latency_s=1.0, error=LLMNoToolCall("нет", response=FakeLLM.text("текст"))))

    summary = summarize(load_calls(run.dir))

    assert summary["calls"] == 3
    assert summary["ok"] == 1
    assert summary["errors"] == {"LLMTimeout": 1, "LLMNoToolCall": 1}
    assert summary["retried"] == 1
    assert summary["with_warnings"] == 1
    assert summary["prompt_tokens"] == 20  # ответы ruler и economist, у таймаута токенов нет
    assert summary["slowest"]["agent"] == "economist"
    assert summary["by_agent"]["economist"]["calls"] == 2
    assert summary["latency_total_s"] == 12.0