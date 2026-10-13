"""Действия: внутренние имена CivRealm ↔ читаемые имена для модели, проверка допустимости.

Источник правды — info['available_actions'] от CivRealm:
    available_actions[тип][id][внутреннее_имя] -> True / False
Каталог строится из него на каждом шаге, поэтому модель может выбрать только то,
что игра действительно разрешает прямо сейчас.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Направления Freeciv (проверено: 'hut enter 5' ведёт в деревню на юго-западе).
DIRECTIONS = ["NorthWest", "North", "NorthEast", "West", "East", "SouthWest", "South", "SouthEast"]
SHORT_DIRECTIONS = {"NW": "NorthWest", "N": "North", "NE": "NorthEast", "W": "West",
                    "E": "East", "SW": "SouthWest", "S": "South", "SE": "SouthEast"}
SHORT_BY_FULL = {full: short for short, full in SHORT_DIRECTIONS.items()}


def expand_direction(name: str) -> str:
    """'move NE' → 'move NorthEast': модель может ответить короткой формой из сводки."""
    head, _, last = name.strip().rpartition(" ")
    full = SHORT_DIRECTIONS.get(last.upper())
    return f"{head} {full}" if head and full else name

_TECH = re.compile(r"^(research_tech|set_tech_goal)_(.+)_(\d+)$")
_TAX = re.compile(r"^set_sci_lux_tax_(\d+)_(\d+)_(\d+)$")
_DIRECTIONAL = re.compile(r"^(.+)_([0-7])$")


class InvalidAction(ValueError):
    """Действие недопустимо сейчас. Сообщение рассчитано на модель: в нём перечислены допустимые варианты."""


def readable_name(ctrl: str, name: str) -> str:
    """Внутреннее имя CivRealm → читаемое имя для модели."""
    if ctrl == "tech" and (match := _TECH.match(name)):
        kind, tech, _ = match.groups()
        return f"research {tech}" if kind == "research_tech" else f"set research goal {tech}"
    if ctrl == "gov":
        if match := _TAX.match(name):
            science, luxury, tax = match.groups()
            return f"set rates science {science}% luxury {luxury}% tax {tax}%"
        if name.startswith("change_gov_"):
            return f"change government to {name.removeprefix('change_gov_')}"
    if ctrl == "unit":
        if name.startswith("goto_") and name[5:].isdigit():
            return f"move {DIRECTIONS[int(name[5:])]}"
        if match := _DIRECTIONAL.match(name):
            action, direction = match.groups()
            return f"{action.replace('_', ' ')} {DIRECTIONS[int(direction)]}"
    return name.replace("_", " ")


@dataclass(frozen=True)
class Action:
    """Одно действие в формате, который понимает CivRealm."""

    ctrl: str  # 'unit', 'city', 'tech', 'gov', 'dipl', 'player'
    actor_id: Any  # id юнита/города, 'cur_player' у tech, 0 у gov
    name: str  # внутреннее имя: 'goto_5', 'research_tech_Bronze Working_9'

    @property
    def readable(self) -> str:
        return readable_name(self.ctrl, self.name)

    def to_civrealm(self) -> tuple[str, Any, str]:
        return (self.ctrl, self.actor_id, self.name)

    def __str__(self) -> str:
        return f"{self.ctrl} {self.actor_id}: {self.readable}"


class ActionCatalog:
    """Все допустимые сейчас действия с читаемыми именами."""

    def __init__(self, available_actions: dict[str, dict[Any, dict[str, bool]]]) -> None:
        # ctrl -> actor_id -> {читаемое имя: Action}
        self._actions: dict[str, dict[Any, dict[str, Action]]] = {}
        for ctrl, actors in available_actions.items():
            for actor_id, action_dict in actors.items():
                options = {}
                for name, allowed in action_dict.items():
                    if allowed:
                        action = Action(ctrl, actor_id, name)
                        options[action.readable] = action
                if options:
                    self._actions.setdefault(ctrl, {})[actor_id] = options

    def actors(self, ctrl: str) -> list[Any]:
        """Кто сейчас может действовать (только с хотя бы одним допустимым действием)."""
        return list(self._actions.get(ctrl, {}))

    def options(self, ctrl: str, actor_id: Any) -> list[str]:
        """Читаемые имена допустимых действий."""
        return list(self._find_actor(ctrl, actor_id)[1]) if self._has_actor(ctrl, actor_id) else []

    def resolve(self, ctrl: str, actor_id: Any, name: str) -> Action:
        """Имя от модели (читаемое или внутреннее) → проверенное Action. Иначе InvalidAction."""
        if not self._has_actor(ctrl, actor_id):
            known = ", ".join(map(str, self.actors(ctrl))) or "none"
            raise InvalidAction(f"{ctrl} {actor_id} cannot act now. Actors that can act: {known}")
        real_id, options = self._find_actor(ctrl, actor_id)
        wanted = expand_direction(name.strip())
        if wanted in options:
            return options[wanted]
        for action in options.values():  # допускаем внутреннее имя и другой регистр
            if wanted == action.name or wanted.lower() == action.readable.lower():
                return action
        raise InvalidAction(
            f"'{name}' is not available for {ctrl} {real_id}. Available: {', '.join(options)}"
        )

    def is_valid(self, action: Action) -> bool:
        try:
            return self.resolve(action.ctrl, action.actor_id, action.name) == action
        except InvalidAction:
            return False

    def __len__(self) -> int:
        return sum(len(options) for actors in self._actions.values() for options in actors.values())

    # Модель пришлёт id строкой ("103"), а у CivRealm он может быть числом — сравниваем как строки.
    def _has_actor(self, ctrl: str, actor_id: Any) -> bool:
        return any(str(a) == str(actor_id) for a in self._actions.get(ctrl, {}))

    def _find_actor(self, ctrl: str, actor_id: Any) -> tuple[Any, dict[str, Action]]:
        for real_id, options in self._actions[ctrl].items():
            if str(real_id) == str(actor_id):
                return real_id, options
        raise KeyError(actor_id)  # pragma: no cover — вызывается только после _has_actor