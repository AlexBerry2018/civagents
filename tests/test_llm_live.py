"""Проверка на живой Ollama. Не запускается по умолчанию: uv run pytest -m live"""

from pathlib import Path

import pytest

from civ_agents.config import load_settings
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.llm.types import FinishReason, Message, ToolSpec

pytestmark = pytest.mark.live

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "settings.yaml"
TOOL = ToolSpec(
    name="found_city",
    description="Found a city with the selected settler",
    parameters={
        "type": "object",
        "properties": {"unit_id": {"type": "integer"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
        "required": ["unit_id", "x", "y"],
    },
)


@pytest.fixture(scope="module")
def client():
    client = OpenAICompatClient(load_settings(CONFIG).llm.active)
    client.health()
    client.warmup()
    return client


def test_live_text(client):
    response = client.chat([Message.user("Reply with exactly one word: ready")])
    assert response.text
    assert "<think>" not in response.text


def test_live_tool_call(client):
    response = client.chat(
        [Message.user("Основать город в клетке (12, 7) поселенцем с id 42")], [TOOL], require_tool=True
    )
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "found_city"
    assert response.tool_calls[0].arguments == {"unit_id": 42, "x": 12, "y": 7}