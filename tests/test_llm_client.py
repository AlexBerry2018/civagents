"""OpenAICompatClient против подменённого HTTP-сервера (httpx.MockTransport) — без Ollama."""

import json
import logging

import httpx
import pytest

from civ_agents.config import LLMProfile
from civ_agents.llm.errors import (
    LLMBadRequest,
    LLMInvalidToolArguments,
    LLMNoToolCall,
    LLMTimeout,
    LLMTruncated,
    LLMUnavailable,
)
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.llm.types import FinishReason, Message, ToolCall, ToolSpec

PROFILE = LLMProfile(
    base_url="http://test/v1", model="m", timeout_s=5, max_output_tokens=64, context_window=1000
)
TOOL = ToolSpec(
    name="found_city",
    description="Found a city",
    parameters={"type": "object", "properties": {"unit_id": {"type": "integer"}}},
)


def completion(content="ok", tool_calls=None, finish_reason="stop", prompt_tokens=10, cached_tokens=0):
    """Ответ сервера в формате OpenAI /v1/chat/completions."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 5,
            "total_tokens": prompt_tokens + 5,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


class Server:
    """Поддельный сервер: отвечает по очереди из списка, запоминает запросы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, int):
            return httpx.Response(item, json={"error": {"message": f"status {item}"}})
        return httpx.Response(200, json=item)

    def body(self, index=-1) -> dict:
        return json.loads(self.requests[index].content)


def make_client(server: Server, **kwargs) -> OpenAICompatClient:
    http_client = httpx.Client(transport=httpx.MockTransport(server))
    return OpenAICompatClient(PROFILE, http_client=http_client, sleep=lambda _: None, **kwargs)


# --- Успешные ответы ----------------------------------------------------------


def test_text_response_is_cleaned():
    server = Server(completion("<think>хм</think>\n\nОтвет", cached_tokens=7))
    response = make_client(server).chat([Message.user("привет")])

    assert response.text == "Ответ"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.cached_tokens == 7
    assert response.attempts == 1


def test_request_contains_model_limit_and_tools():
    server = Server(completion(tool_calls=[("found_city", '{"unit_id": 42}')], finish_reason="tool_calls"))
    make_client(server).chat([Message.user("x")], [TOOL])

    body = server.body()
    assert body["model"] == "m"
    assert body["max_tokens"] == 64
    assert body["tools"][0]["function"]["name"] == "found_city"


def test_tool_call_arguments_are_parsed():
    server = Server(completion("", tool_calls=[("found_city", '{"unit_id": 42}')], finish_reason="tool_calls"))
    response = make_client(server).chat([Message.user("x")], [TOOL], require_tool=True)

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "found_city"
    assert response.tool_calls[0].arguments == {"unit_id": 42}


def test_history_with_tool_call_is_serialized():
    server = Server(completion())
    history = [
        Message.user("основать город"),
        Message.assistant("", [ToolCall(id="c1", name="found_city", arguments={"unit_id": 42})]),
        Message.tool("c1", "Город основан"),
    ]
    make_client(server).chat(history)

    messages = server.body()["messages"]
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"unit_id": 42}
    assert messages[2] == {"role": "tool", "content": "Город основан", "tool_call_id": "c1"}


# --- Ретраи -------------------------------------------------------------------


def test_server_error_is_retried():
    server = Server(500, completion("ok"))
    response = make_client(server).chat([Message.user("x")])

    assert response.text == "ok"
    assert response.attempts == 2
    assert len(server.requests) == 2


def test_bad_request_is_not_retried():
    server = Server(404)
    with pytest.raises(LLMBadRequest):
        make_client(server).chat([Message.user("x")])
    assert len(server.requests) == 1


def test_persistent_failure_gives_up_after_max_attempts():
    server = Server(500, 500, 500)
    with pytest.raises(LLMUnavailable):
        make_client(server, max_attempts=3).chat([Message.user("x")])
    assert len(server.requests) == 3


def test_timeout_is_retried_then_raised():
    timeout = httpx.ReadTimeout("timeout")
    server = Server(timeout, timeout, timeout)
    with pytest.raises(LLMTimeout):
        make_client(server).chat([Message.user("x")])
    assert len(server.requests) == 3


def test_connection_error_is_unavailable():
    error = httpx.ConnectError("refused")
    server = Server(error, error, error)
    with pytest.raises(LLMUnavailable):
        make_client(server).chat([Message.user("x")])


def test_expired_deadline_prevents_request():
    server = Server(completion())
    with pytest.raises(LLMTimeout):
        make_client(server).chat([Message.user("x")], deadline=0.0)  # момент в прошлом
    assert server.requests == []


# --- Ошибки содержания ----------------------------------------------------------


def test_truncated_response():
    server = Server(completion("обрыв", finish_reason="length"))
    with pytest.raises(LLMTruncated) as info:
        make_client(server).chat([Message.user("x")])
    assert info.value.response.text == "обрыв"


def test_missing_tool_call_when_required():
    server = Server(completion("Я бы основал город"))
    with pytest.raises(LLMNoToolCall) as info:
        make_client(server).chat([Message.user("x")], [TOOL], require_tool=True)
    assert info.value.response.text == "Я бы основал город"


def test_text_is_fine_when_tool_not_required():
    server = Server(completion("Стратегия: расширение"))
    assert make_client(server).chat([Message.user("x")], [TOOL]).text == "Стратегия: расширение"


def test_invalid_tool_arguments():
    server = Server(completion("", tool_calls=[("found_city", "{unit_id: 42")], finish_reason="tool_calls"))
    with pytest.raises(LLMInvalidToolArguments):
        make_client(server).chat([Message.user("x")], [TOOL])


def test_content_errors_are_not_retried():
    server = Server(completion("обрыв", finish_reason="length"))
    with pytest.raises(LLMTruncated):
        make_client(server).chat([Message.user("x")])
    assert len(server.requests) == 1


# --- Контекст -------------------------------------------------------------------


def test_warns_when_prompt_may_not_fit(caplog):
    server = Server(completion())
    with caplog.at_level(logging.WARNING):
        make_client(server).chat([Message.user("x" * 5000)])  # ~2270 токенов при лимите ~936
    assert "может быть обрезан" in caplog.text


def test_warns_when_prompt_filled_context(caplog):
    server = Server(completion(prompt_tokens=990))
    with caplog.at_level(logging.WARNING):
        make_client(server).chat([Message.user("x")])
    assert "обрезано" in caplog.text


# --- Наблюдатель ----------------------------------------------------------------


def test_observer_gets_successful_and_failed_calls():
    records = []
    server = Server(completion("ok"), 404)
    client = make_client(server, observer=records.append)

    client.chat([Message.user("1")])
    with pytest.raises(LLMBadRequest):
        client.chat([Message.user("2")])

    assert [r.ok for r in records] == [True, False]
    assert records[0].response.text == "ok"
    assert isinstance(records[1].error, LLMBadRequest)


def test_broken_observer_does_not_break_call():
    def broken(record):
        raise RuntimeError("журнал сломался")

    server = Server(completion("ok"))
    assert make_client(server, observer=broken).chat([Message.user("x")]).text == "ok"


# --- health / warmup ------------------------------------------------------------


def test_health_finds_model():
    server = Server({"object": "list", "data": [{"id": "m", "object": "model", "created": 0, "owned_by": "x"}]})
    assert make_client(server).health() is True


def test_health_reports_missing_model():
    server = Server({"object": "list", "data": [{"id": "other", "object": "model", "created": 0, "owned_by": "x"}]})
    with pytest.raises(LLMBadRequest, match="other"):
        make_client(server).health()


def test_warmup_returns_elapsed_time():
    server = Server(completion("ok"))
    assert make_client(server).warmup() >= 0
    assert server.body()["max_tokens"] == 5