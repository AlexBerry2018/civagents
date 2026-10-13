"""Состояние игры в наших типах: только то, что нужно агентам, без карты и правил.

Извлечение из observations CivRealm сделано защитно (dict.get): поля, которых
нет в наблюдении, получают значения по умолчанию, а не роняют партию.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# Состояния технологии во Freeciv (поле inv_state).
TECH_UNKNOWN, TECH_REACHABLE, TECH_KNOWN = 0, 1, 2

# Дипломатические состояния Freeciv (поле diplomatic_state).
DIPLOMATIC_STATES = {
    0: "armistice",
    1: "war",
    2: "ceasefire",
    3: "peace",
    4: "alliance",
    5: "no contact",
    6: "team",
}


class PlayerState(BaseModel):
    id: int
    name: str
    score: int = 0
    gold: int = 0
    government: str = ""
    tax: int = 0
    science: int = 0
    luxury: int = 0
    researching: str | None = None  # имя технологии
    research_bulbs: int = 0  # накоплено «колб» науки
    research_cost: int = 0  # сколько нужно для текущей технологии
    techs_known: int = 0


class UnitState(BaseModel):
    id: int
    type: str
    x: int
    y: int
    health: int = 0
    max_health: int = 0
    moves_left: int = 0
    attack: int = 0
    defense: int = 0
    is_worker: bool = False
    home_city: int | None = None


class CityState(BaseModel):
    id: int
    name: str = ""
    size: int = 0
    x: int | None = None
    y: int | None = None
    # Рост
    food_stock: int = 0
    granary_size: int = 0
    food_surplus: int = 0
    growth_turns: int | None = None  # через сколько ходов вырастет; None — не растёт
    # Производство
    shield_stock: int = 0
    shield_surplus: int = 0
    turns_to_complete: int | None = None
    production: str | None = None  # что строится; заполнится, когда научимся называть
    # Торговля и наука
    trade_surplus: int = 0
    bulbs: int = 0
    # Настроение жителей
    happy: int = 0
    content: int = 0
    unhappy: int = 0
    angry: int = 0
    details: dict[str, Any] = Field(default_factory=dict)  # прочие простые поля — для отладки


class Opponent(BaseModel):
    id: int
    name: str
    score: int = 0
    is_alive: bool = True
    diplomacy: str = "unknown"


class GameState(BaseModel):
    turn: int
    me: PlayerState
    units: list[UnitState] = Field(default_factory=list)
    cities: list[CityState] = Field(default_factory=list)
    research_options: list[str] = Field(default_factory=list)  # можно начать исследовать сейчас
    known_techs: list[str] = Field(default_factory=list)
    opponents: list[Opponent] = Field(default_factory=list)

    @classmethod
    def from_observation(cls, observations: dict[str, Any], info: dict[str, Any], my_player_id: int = 0) -> GameState:
        """observations и info от CivRealm → GameState."""
        players = observations.get("player", {})
        techs = observations.get("tech", {})
        me_raw = _by_id(players, my_player_id) or {}

        me = _player(me_raw, my_player_id, techs)
        return cls(
            turn=int(info.get("turn", 0)),
            me=me,
            units=[
                _unit(unit_id, raw)
                for unit_id, raw in observations.get("unit", {}).items()
                if raw.get("owner", my_player_id) == my_player_id
            ],
            cities=[
                _city(city_id, raw)
                for city_id, raw in observations.get("city", {}).items()
                if raw.get("owner", my_player_id) == my_player_id
            ],
            # Текущее исследование не предлагаем: выбрать его снова — потраченное впустую действие.
            research_options=sorted(
                t["name"] for t in techs.values()
                if t.get("inv_state") == TECH_REACHABLE and t.get("name") != me.researching
            ),
            known_techs=sorted(t["name"] for t in techs.values() if t.get("inv_state") == TECH_KNOWN),
            opponents=[
                _opponent(player_id, raw, observations.get("dipl", {}))
                for player_id, raw in players.items()
                if int(player_id) != my_player_id
            ],
        )


# --- Извлечение -----------------------------------------------------------------


def _by_id(mapping: dict[Any, Any], key: int) -> Any:
    """Ключи в наблюдении бывают и числами, и строками."""
    return mapping.get(key, mapping.get(str(key)))


def _player(raw: dict[str, Any], player_id: int, techs: dict[Any, Any]) -> PlayerState:
    researching_id = raw.get("researching")
    researching = None
    if researching_id is not None and (tech := _by_id(techs, researching_id)):
        researching = tech.get("name")
    return PlayerState(
        id=player_id,
        name=raw.get("name", ""),
        score=max(0, raw.get("score") or 0),
        gold=raw.get("gold") or 0,
        government=raw.get("government_name") or "",
        tax=raw.get("tax") or 0,
        science=raw.get("science") or 0,
        luxury=raw.get("luxury") or 0,
        researching=researching,
        research_bulbs=raw.get("bulbs_researched") or 0,
        research_cost=raw.get("researching_cost") or 0,
        techs_known=raw.get("techs_researched") or 0,
    )


def _unit(unit_id: Any, raw: dict[str, Any]) -> UnitState:
    return UnitState(
        id=int(unit_id),
        type=raw.get("type_rule_name", "Unknown"),
        x=raw.get("x", -1),
        y=raw.get("y", -1),
        health=raw.get("health") or 0,
        max_health=raw.get("type_hp") or 0,
        moves_left=raw.get("moves_left") or 0,
        attack=raw.get("type_attack_strength") or 0,
        defense=raw.get("type_defense_strength") or 0,
        is_worker=bool(raw.get("type_worker")),
        home_city=raw.get("home_city") or None,
    )


_CITY_FIELDS = {
    # наше поле: поле CivRealm
    "food_stock": "food_stock",
    "granary_size": "granary_size",
    "food_surplus": "surplus_food",
    "shield_stock": "shield_stock",
    "shield_surplus": "surplus_shield",
    "turns_to_complete": "turns_to_prod_complete",
    "trade_surplus": "surplus_trade",
    "bulbs": "bulbs",
    "happy": "ppl_happy",
    "content": "ppl_content",
    "unhappy": "ppl_unhappy",
    "angry": "ppl_angry",
}
_CITY_USED = set(_CITY_FIELDS.values()) | {"id", "name", "size", "x", "y", "owner", "growth_in"}


def _turns(text: Any) -> int | None:
    """'17 turns' → 17. Всё, что не начинается с числа (например, 'never'), → None."""
    match = re.match(r"\s*(-?\d+)", str(text or ""))
    return int(match.group(1)) if match and int(match.group(1)) >= 0 else None


def _city(city_id: Any, raw: dict[str, Any]) -> CityState:
    values = {ours: raw.get(theirs) for ours, theirs in _CITY_FIELDS.items() if raw.get(theirs) is not None}
    return CityState(
        id=int(city_id),
        name=raw.get("name", ""),
        size=raw.get("size") or 0,
        x=raw.get("x"),
        y=raw.get("y"),
        growth_turns=_turns(raw.get("growth_in")),
        **values,
        details={
            k: v for k, v in raw.items() if k not in _CITY_USED and isinstance(v, (int, float, str, bool))
        },
    )


def _opponent(player_id: Any, raw: dict[str, Any], dipl: dict[Any, Any]) -> Opponent:
    state = (_by_id(dipl, int(player_id)) or {}).get("diplomatic_state")
    return Opponent(
        id=int(player_id),
        name=raw.get("name", ""),
        score=max(0, raw.get("score") or 0),
        is_alive=bool(raw.get("is_alive", True)),
        diplomacy=DIPLOMATIC_STATES.get(state, "unknown"),
    )