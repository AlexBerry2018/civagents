"""Скиллы: параметризованные процедуры поверх состояния игры.

Скилл — детерминированный код, который отвечает на типовой вопрос агента
(«где основать город?», «что исследовать?», «есть ли угроза?») и даёт подсказку
в промпт. Описание каждого скилла — в skills/<name>.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from civ_agents.env.session import TurnView

TERRAIN_SCORE = {"Grassland": 3, "Plains": 3, "Hills": 2, "Forest": 1, "Jungle": 0, "Desert": 0,
                 "Tundra": 0, "Swamp": 0, "Mountains": 0, "Ocean": 0, "Lake": 1}
BONUS = {"River": 2, "Wheat": 2, "Pheasant": 1, "Wine": 1, "Silk": 1, "Resources": 1, "Oasis": 1,
         "Buffalo": 1, "Fish": 2, "Whales": 1, "Coal": 1, "Gold": 1, "Spices": 1, "Fruit": 1}
NEIGHBORS = ["here", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIRECTION_NAME = {"N": "North", "NE": "NorthEast", "E": "East", "SE": "SouthEast",
                  "S": "South", "SW": "SouthWest", "W": "West", "NW": "NorthWest"}
TECH_ORDER = ["Bronze Working", "Alphabet", "Ceremonial Burial", "Pottery", "Writing", "Code of Laws",
              "Monarchy", "Masonry", "Currency", "Horseback Riding", "Warrior Code", "The Wheel"]


def _tiles(area: str) -> dict[str, list[str]]:
    tiles = {}
    for line in area.splitlines():
        label, _, rest = line.partition(":")
        if label in NEIGHBORS:
            tiles[label] = [p.strip() for p in rest.split(",")]
    return tiles


def tile_score(features: list[str]) -> int:
    if any(f.startswith("[units: player") for f in features):
        return -5  # чужие юниты — не место для города
    return sum(TERRAIN_SCORE.get(f, 0) + BONUS.get(f, 0) for f in features)


def city_site(view: TurnView) -> str:
    """Для каждого поселенца: оценка текущей клетки и лучшее соседнее направление."""
    hints = []
    for unit in view.state.units:
        if unit.type != "Settlers":
            continue
        area = view.areas.get((unit.x, unit.y))
        if not area:
            continue
        tiles = _tiles(area)
        here = tile_score(tiles.get("here", []))
        best = max((d for d in tiles if d != "here"), key=lambda d: tile_score(tiles[d]), default=None)
        can_build = "build city" in view.catalog.options("unit", unit.id)
        if can_build and here >= 3:  # приемлемое место — строим, а не ищем идеальное (частая ошибка)
            hints.append(f"Settlers {unit.id}: current tile score {here} is good -> 'build city'")
        elif best and tile_score(tiles[best]) > here + 1:
            hints.append(f"Settlers {unit.id}: current tile score {here}; better site {best} "
                         f"({', '.join(tiles[best])}) -> 'move {DIRECTION_NAME[best]}'")
        elif can_build:
            hints.append(f"Settlers {unit.id}: current tile score {here} is good -> 'build city'")
        else:
            hints.append(f"Settlers {unit.id}: cannot build here (too close to a city?) -> move 2-3 tiles away")
    return "\n".join(hints)


def research_plan(view: TurnView) -> str:
    """Следующая технология по проверенному порядку ранней игры."""
    options = view.state.research_options
    if view.state.me.researching in TECH_ORDER[:7]:
        return f"Keep researching {view.state.me.researching}."
    for tech in TECH_ORDER:
        if tech in options:
            return f"Suggested next research: {tech} -> 'research {tech}'"
    return ""


def assess_threat(view: TurnView) -> str:
    """Чужие юниты рядом с нашими юнитами/городами (по описанию окрестностей)."""
    threats = []
    for (x, y), area in view.areas.items():
        for line in area.splitlines():
            if "[units: player" in line:
                threats.append(f"foreign units near {(x, y)}: {line.strip()}")
    if not view.state.cities:
        return "\n".join(threats)
    undefended = [c.name for c in view.state.cities
                  if not any((u.x, u.y) == (c.x, c.y) and not u.is_worker for u in view.state.units)]
    if undefended:
        threats.append(f"cities without a defender: {', '.join(undefended)}")
    return "\n".join(threats) or "No visible threats."


@dataclass
class Skill:
    name: str
    roles: tuple[str, ...]
    run: Callable[[TurnView], str]


SKILLS = [
    Skill("city_site", ("economist", "single"), city_site),
    Skill("research_plan", ("science", "single"), research_plan),
    Skill("assess_threat", ("military", "ruler", "single"), assess_threat),
]


def skill_hints(role: str, view: TurnView) -> str:
    parts = [f"[{s.name}] {text}" for s in SKILLS if role in s.roles and (text := s.run(view))]
    return "Skill hints:\n" + "\n".join(parts) if parts else ""
