"""Рефлексия после партии: модель разбирает события и итоги и пишет уроки в базу знаний.

Уроки попадают в knowledge/lessons.md и в следующих партиях находятся семантическим поиском.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from civ_agents.llm.base import LLMClient
from civ_agents.llm.errors import LLMError
from civ_agents.llm.types import Message
from civ_agents.memory.episodic import Event
from civ_agents.observability.context import call_context

PROMPT = (
    "You analyse a finished Freeciv game played by an AI government. "
    "Based on the events and final numbers, write exactly 3 short, concrete lessons for the next game "
    "(one line each, starting with '- '). Focus on what to do differently in the first 50 turns."
)


def reflect(llm: LLMClient, events: list[Event], outcome: str, lessons_path: str | Path, run_id: str) -> str:
    history = "\n".join(f"T{e.turn}: {e.text}" for e in events[-40:]) or "no notable events"
    with call_context(agent="reflection"):
        try:
            response = llm.chat([Message.system(PROMPT), Message.user(f"Events:\n{history}\n\nOutcome:\n{outcome}")])
        except LLMError:
            return ""
    lessons = "\n".join(line for line in response.text.splitlines() if line.strip().startswith("-"))
    if lessons:
        path = Path(lessons_path)
        header = f"\n## Lessons from game {run_id} ({datetime.now():%Y-%m-%d})\n"
        with path.open("a", encoding="utf-8") as file:
            file.write(header + lessons + "\n")
    return lessons
