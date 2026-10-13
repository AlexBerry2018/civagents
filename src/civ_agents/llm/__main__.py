"""Ручная проверка слоя модели.

    uv run python -m civ_agents.llm "Вопрос модели"
    uv run python -m civ_agents.llm --tool "Основать город в (12, 7) поселенцем 42"
    uv run python -m civ_agents.llm --profile local_thinking "Вопрос"
"""

from __future__ import annotations

import argparse
import logging
import sys

from civ_agents.config import load_settings
from civ_agents.llm.errors import LLMContentError, LLMError
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.llm.types import Message, ToolSpec

DEMO_TOOL = ToolSpec(
    name="found_city",
    description="Found a city with the selected settler",
    parameters={
        "type": "object",
        "properties": {
            "unit_id": {"type": "integer", "description": "Settler unit id"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["unit_id", "x", "y"],
    },
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ручная проверка вызова модели")
    parser.add_argument("prompt", help="текст сообщения пользователя")
    parser.add_argument("--tool", action="store_true", help="передать тестовый инструмент found_city и требовать вызов")
    parser.add_argument("--profile", help="имя профиля модели вместо active_profile")
    parser.add_argument("--system", help="системный промпт")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings()
    name = args.profile or settings.llm.active_profile
    if name not in settings.llm.profiles:
        print(f"Профиль '{name}' не найден, есть: {sorted(settings.llm.profiles)}", file=sys.stderr)
        return 1
    client = OpenAICompatClient(settings.llm.profiles[name])

    messages = [Message.system(args.system)] if args.system else []
    messages.append(Message.user(args.prompt))
    tools = [DEMO_TOOL] if args.tool else None

    print(f"Профиль: {name} ({client.model})")
    try:
        client.health()
        response = client.chat(messages, tools, require_tool=args.tool)
    except LLMContentError as error:
        print(f"Негодный ответ ({type(error).__name__}): {error}", file=sys.stderr)
        if error.response is not None:
            print(f"Текст ответа: {error.response.text!r}", file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"Ошибка ({type(error).__name__}): {error}", file=sys.stderr)
        return 1

    if response.text:
        print(f"\nОтвет:\n{response.text}\n")
    for call in response.tool_calls:
        print(f"Вызов: {call.name}({call.arguments})")
    usage = response.usage
    print(
        f"finish={response.finish_reason.value}  "
        f"токены: промпт {usage.prompt_tokens} (из кэша {usage.cached_tokens}), ответ {usage.completion_tokens}  "
        f"время {response.latency_s:.2f} с  попыток {response.attempts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())