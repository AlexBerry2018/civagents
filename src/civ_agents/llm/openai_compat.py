"""Клиент для OpenAI-совместимого сервера модели (Ollama, LM Studio, llama.cpp server)."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import httpx
import openai
from openai import OpenAI

from civ_agents.llm.base import CallRecord, Observer
from civ_agents.llm.errors import (
    LLMBadRequest,
    LLMError,
    LLMNoToolCall,
    LLMTimeout,
    LLMTransportError,
    LLMTruncated,
    LLMUnavailable,
)
from civ_agents.llm.postprocess import estimate_tokens, normalize_finish_reason, parse_arguments, strip_thinking
from civ_agents.llm.types import FinishReason, LLMResponse, Message, ToolCall, ToolSpec, Usage

if TYPE_CHECKING:
    from civ_agents.config import LLMProfile

logger = logging.getLogger(__name__)


class OpenAICompatClient:
    """Реализация LLMClient поверх OpenAI SDK.

    Ретраи свои (во встроенных ретраях SDK нет дедлайна хода), поэтому max_retries=0.
    """

    def __init__(
        self,
        profile: LLMProfile,
        *,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 4.0,
        observer: Observer | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._profile = profile
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._observer = observer
        self._sleep = sleep  # подменяется в тестах, чтобы не ждать
        self._client = OpenAI(
            base_url=str(profile.base_url),
            api_key="ollama",  # Ollama ключ не проверяет, но SDK требует строку
            timeout=profile.timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    @property
    def model(self) -> str:
        return self._profile.model

    # --- Публичные методы ---------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        require_tool: bool = False,
        deadline: float | None = None,
    ) -> LLMResponse:
        messages, tools = list(messages), list(tools or [])
        warnings = self._check_prompt_size(messages, tools)
        request = self._build_request(messages, tools)

        started = time.monotonic()
        attempts = 0
        response: LLMResponse | None = None
        error: LLMError | None = None
        try:
            raw, attempts = self._send_with_retries(request, deadline)
            response = self._parse(raw, latency_s=time.monotonic() - started, attempts=attempts)
            warnings += self._check_context_usage(response)
            self._check_content(response, require_tool)
            return response
        except LLMError as exc:
            error = exc
            raise
        finally:
            self._notify(
                CallRecord(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    latency_s=time.monotonic() - started,
                    attempts=attempts,
                    response=response,
                    error=error,
                    warnings=warnings,
                )
            )

    def health(self) -> bool:
        """Сервер отвечает и нужная модель есть в списке."""
        try:
            available = {m.id for m in self._client.models.list()}
        except openai.APIConnectionError as exc:
            raise LLMUnavailable(f"Сервер модели недоступен: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMUnavailable(f"Сервер модели вернул HTTP {exc.status_code}") from exc
        if self.model not in available:
            raise LLMBadRequest(f"Модель '{self.model}' не найдена на сервере. Есть: {sorted(available)}")
        return True

    def warmup(self, timeout_s: float = 180.0) -> float:
        """Загрузить модель в память до первого хода. Возвращает время в секундах.

        Холодная загрузка (25–50 с) может превышать обычный timeout_s профиля,
        поэтому у прогрева свой, больший таймаут и нет ретраев.
        """
        started = time.monotonic()
        try:
            self._client.with_options(timeout=timeout_s).chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with one word: ok"}],
                max_tokens=5,
            )
        except openai.APIError as exc:
            error = _translate_error(exc)
            raise error from exc
        elapsed = time.monotonic() - started
        logger.info("Модель %s прогрета за %.1f с", self.model, elapsed)
        return elapsed

    # --- Запрос -------------------------------------------------------------

    def _build_request(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "max_tokens": self._profile.max_output_tokens,
        }
        if tools:
            request["tools"] = [_to_openai_tool(t) for t in tools]
        return request

    def _send_with_retries(self, request: dict[str, Any], deadline: float | None) -> tuple[Any, int]:
        for attempt in range(1, self._max_attempts + 1):
            timeout = self._profile.timeout_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMTimeout("Дедлайн исчерпан до отправки запроса")
                timeout = min(timeout, remaining)
            try:
                raw = self._client.with_options(timeout=timeout).chat.completions.create(**request)
                return raw, attempt
            except openai.APIError as exc:
                error = _translate_error(exc)
                error.__cause__ = exc

            if not isinstance(error, LLMTransportError):
                raise error  # 4xx: повторять бессмысленно
            if attempt == self._max_attempts:
                raise error
            delay = min(self._backoff_max_s, self._backoff_base_s * 2 ** (attempt - 1)) * random.uniform(0.5, 1.0)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise error
            logger.warning("Вызов модели не удался (%s), попытка %d/%d, повтор через %.1f с",
                           error, attempt, self._max_attempts, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    # --- Ответ --------------------------------------------------------------

    def _parse(self, raw: Any, *, latency_s: float, attempts: int) -> LLMResponse:
        choice = raw.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=call.id or f"call_{index}",
                name=call.function.name,
                arguments=parse_arguments(call.function.arguments),
            )
            for index, call in enumerate(message.tool_calls or [])
        ]
        return LLMResponse(
            text=strip_thinking(message.content),
            tool_calls=tool_calls,
            finish_reason=normalize_finish_reason(choice.finish_reason, bool(tool_calls)),
            usage=_parse_usage(raw.usage),
            latency_s=latency_s,
            model=raw.model or self.model,
            attempts=attempts,
        )

    @staticmethod
    def _check_content(response: LLMResponse, require_tool: bool) -> None:
        if response.finish_reason is FinishReason.LENGTH:
            raise LLMTruncated("Ответ обрезан лимитом max_output_tokens", response=response)
        if require_tool and not response.tool_calls:
            raise LLMNoToolCall("Ожидался вызов инструмента, модель ответила текстом", response=response)

    # --- Контекст -----------------------------------------------------------

    def _check_prompt_size(self, messages: list[Message], tools: list[ToolSpec]) -> list[str]:
        """До вызова: грубо оценить, влезет ли промпт (Ollama обрезает молча)."""
        text = "".join(m.content for m in messages) + json.dumps([t.model_dump() for t in tools], ensure_ascii=False)
        estimated = estimate_tokens(text)
        limit = self._profile.context_window - self._profile.max_output_tokens
        if estimated > limit:
            message = f"Промпт ~{estimated} токенов, лимит ~{limit}: может быть обрезан"
            logger.warning(message)
            return [message]
        return []

    def _check_context_usage(self, response: LLMResponse) -> list[str]:
        """После вызова: промпт почти во весь контекст — вероятно, его начало обрезано."""
        limit = self._profile.context_window - self._profile.max_output_tokens
        if response.usage.prompt_tokens >= limit:
            message = (f"prompt_tokens={response.usage.prompt_tokens} при context_window="
                       f"{self._profile.context_window}: начало промпта, вероятно, обрезано")
            logger.warning(message)
            return [message]
        return []

    # --- Наблюдатель --------------------------------------------------------

    def _notify(self, record: CallRecord) -> None:
        if self._observer is None:
            return
        try:
            self._observer(record)
        except Exception:  # noqa: BLE001 — сбой журнала не должен ронять вызов модели
            logger.exception("Наблюдатель вызова модели упал")


# --- Перевод форматов --------------------------------------------------------


def _to_openai_message(message: Message) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in message.tool_calls
        ]
    if message.role == "tool":
        result["tool_call_id"] = message.tool_call_id
    return result


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _parse_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    details = getattr(usage, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    )


def _translate_error(exc: openai.APIError) -> LLMError:
    """Ошибки OpenAI SDK → наши исключения. Транспортные будут повторены."""
    if isinstance(exc, openai.APITimeoutError):  # подкласс APIConnectionError — проверяем первым
        return LLMTimeout(f"Сервер модели не ответил вовремя: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return LLMUnavailable(f"Нет соединения с сервером модели: {exc}")
    if isinstance(exc, (openai.InternalServerError, openai.RateLimitError)):
        return LLMUnavailable(f"Сервер модели вернул HTTP {exc.status_code}")
    if isinstance(exc, openai.APIStatusError):
        return LLMBadRequest(f"Сервер модели отклонил запрос, HTTP {exc.status_code}: {exc.message}")
    return LLMUnavailable(f"Ошибка обращения к серверу модели: {exc}")