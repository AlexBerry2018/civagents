"""Очистка ответа модели и FakeLLM — без сети."""

import pytest

from civ_agents.llm.errors import LLMInvalidToolArguments, LLMNoToolCall, LLMTimeout
from civ_agents.llm.fake import FakeLLM
from civ_agents.llm.postprocess import estimate_tokens, normalize_finish_reason, parse_arguments, strip_thinking
from civ_agents.llm.types import FinishReason, Message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ответ", "Ответ"),
        ("<think>рассуждение</think>\n\nОтвет", "Ответ"),
        ("<think>строка 1\nстрока 2</think>Ответ", "Ответ"),  # многострочное рассуждение
        ("Okay, let's see...\n</think>\n\nОтвет", "Ответ"),  # thinking-2507: только закрывающий тег
        ("Начало <think>оборванное рассуждение", "Начало"),  # ответ обрезан посреди рассуждения
        ("", ""),
        (None, ""),
    ],
)
def test_strip_thinking(raw, expected):
    assert strip_thinking(raw) == expected


def test_parse_arguments_valid():
    assert parse_arguments('{"unit_id": 42, "x": 12, "y": 7}') == {"unit_id": 42, "x": 12, "y": 7}


@pytest.mark.parametrize("raw", [None, ""])
def test_parse_arguments_empty(raw):
    assert parse_arguments(raw) == {}


def test_parse_arguments_already_dict():
    assert parse_arguments({"x": 1}) == {"x": 1}


@pytest.mark.parametrize("raw", ["{unit_id: 42", "not json", "[1, 2]", '"строка"'])
def test_parse_arguments_invalid(raw):
    with pytest.raises(LLMInvalidToolArguments):
        parse_arguments(raw)


@pytest.mark.parametrize(
    ("raw", "has_tool_calls", "expected"),
    [
        ("stop", False, FinishReason.STOP),
        ("tool_calls", True, FinishReason.TOOL_CALLS),
        ("stop", True, FinishReason.TOOL_CALLS),  # вызов определяется по наличию tool_calls
        ("length", True, FinishReason.LENGTH),  # обрезка важнее вызова
        ("length", False, FinishReason.LENGTH),
        (None, False, FinishReason.STOP),
    ],
)
def test_normalize_finish_reason(raw, has_tool_calls, expected):
    assert normalize_finish_reason(raw, has_tool_calls) is expected


def test_estimate_tokens_matches_benchmark():
    # Замер: 8000 символов пакетов Freeciv = 3648 токенов.
    assert 3400 <= estimate_tokens("x" * 8000) <= 3900


# --- FakeLLM -----------------------------------------------------------------


def test_fake_llm_follows_script_and_records_calls():
    llm = FakeLLM([FakeLLM.tool("found_city", unit_id=42, x=12, y=7), FakeLLM.text("Расширяемся")])

    first = llm.chat([Message.user("ход 1")], require_tool=True)
    second = llm.chat([Message.user("ход 2")])

    assert first.tool_calls[0].arguments == {"unit_id": 42, "x": 12, "y": 7}
    assert second.text == "Расширяемся"
    assert [c.messages[0].content for c in llm.calls] == ["ход 1", "ход 2"]
    assert llm.calls[0].require_tool is True


def test_fake_llm_raises_scripted_error():
    llm = FakeLLM([LLMTimeout("сервер молчит")])
    with pytest.raises(LLMTimeout):
        llm.chat([Message.user("x")])


def test_fake_llm_requires_tool_like_real_client():
    llm = FakeLLM([FakeLLM.text("Я бы основал город")])
    with pytest.raises(LLMNoToolCall) as info:
        llm.chat([Message.user("x")], require_tool=True)
    assert info.value.response.text == "Я бы основал город"


def test_fake_llm_empty_script_fails_loudly():
    with pytest.raises(AssertionError, match="очередь ответов пуста"):
        FakeLLM().chat([Message.user("x")])