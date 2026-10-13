"""Эпизодическая память: значимые события партии, выведенные из разницы состояний между ходами.

Пишутся не все ходы, а события: основан/потерян город, вырос город, появился/пропал юнит,
изучена технология, установлен контакт. Хранятся в JSONL в папке прогона.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from civ_agents.env.state import GameState


class Event(BaseModel):
    turn: int
    kind: str
    text: str


class EpisodicMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[Event] = []
        self._previous: GameState | None = None

    def observe(self, state: GameState) -> list[Event]:
        """Сравнить с прошлым ходом и записать новые события."""
        previous, self._previous = self._previous, state
        if previous is None or previous.turn == state.turn:
            return []
        new = diff(previous, state)
        self.events.extend(new)
        if self.path and new:
            with self.path.open("a", encoding="utf-8") as file:
                for event in new:
                    file.write(event.model_dump_json() + "\n")
        return new

    def recent(self, limit: int = 8) -> str:
        """Последние события строками — для промпта."""
        if not self.events:
            return ""
        lines = [f"T{e.turn}: {e.text}" for e in self.events[-limit:]]
        return "Recent events:\n" + "\n".join(lines)

    def search(self, kind: str | None = None, since_turn: int = 0) -> list[Event]:
        return [e for e in self.events if (kind is None or e.kind == kind) and e.turn >= since_turn]


def diff(old: GameState, new: GameState) -> list[Event]:
    turn = new.turn
    events: list[Event] = []
    old_cities = {c.id: c for c in old.cities}
    new_cities = {c.id: c for c in new.cities}
    for city_id, city in new_cities.items():
        if city_id not in old_cities:
            events.append(Event(turn=turn, kind="city_founded", text=f"city {city.name} founded at {(city.x, city.y)}"))
        elif city.size > old_cities[city_id].size:
            events.append(Event(turn=turn, kind="city_grew", text=f"{city.name} grew to size {city.size}"))
    for city_id, city in old_cities.items():
        if city_id not in new_cities:
            events.append(Event(turn=turn, kind="city_lost", text=f"LOST city {city.name}"))

    old_units = {u.id: u for u in old.units}
    new_units = {u.id: u for u in new.units}
    for unit_id, unit in new_units.items():
        if unit_id not in old_units:
            events.append(Event(turn=turn, kind="unit_built", text=f"new {unit.type} {unit_id}"))
    founded = sum(1 for e in events if e.kind == "city_founded")
    for unit_id, unit in old_units.items():
        if unit_id not in new_units:
            if unit.type == "Settlers" and founded:
                founded -= 1  # поселенец «исчез», потому что основал город
                continue
            events.append(Event(turn=turn, kind="unit_lost", text=f"lost {unit.type} {unit_id} at {(unit.x, unit.y)}"))

    for tech in sorted(set(new.known_techs) - set(old.known_techs)):
        events.append(Event(turn=turn, kind="tech_learned", text=f"learned {tech}"))
    old_contacts = {o.id: o.diplomacy for o in old.opponents}
    for opponent in new.opponents:
        if old_contacts.get(opponent.id) != opponent.diplomacy and opponent.diplomacy not in ("no contact", "unknown"):
            events.append(Event(turn=turn, kind="contact", text=f"{opponent.name}: {opponent.diplomacy}"))
    if new.me.government != old.me.government and new.me.government:
        events.append(Event(turn=turn, kind="government", text=f"government is now {new.me.government}"))
    return events
