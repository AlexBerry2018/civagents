"""Сводка хода для модели: компактный текст с бюджетом токенов и срезами по ролям.

Разделы собираются по приоритету роли. Если раздел не помещается в бюджет,
он заменяется одной строкой-пометкой: модель знает, что информация была, но опущена.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal

from civ_agents.env.actions import SHORT_BY_FULL, ActionCatalog
from civ_agents.env.neighborhood import describe_area
from civ_agents.env.state import CityState, GameState, UnitState
from civ_agents.llm.postprocess import estimate_tokens

Role = Literal["all", "ruler", "economist", "military", "science"]
Tile = tuple[int, int]

# Какие разделы видит каждая роль и в каком порядке (порядок = приоритет при нехватке бюджета).
ROLE_SECTIONS: dict[str, list[str]] = {
    "all": ["overview", "research", "cities", "units", "government"],
    "ruler": ["overview", "cities", "research", "government", "units_brief"],
    "economist": ["overview", "cities", "workers"],
    "military": ["overview", "military_units", "cities_brief"],
    "science": ["overview", "research"],
}

# Типовые налоги: показываем их, а не все 36 сочетаний из каталога.
RATE_PRESETS = ["set rates science 60% luxury 0% tax 40%",
                "set rates science 40% luxury 0% tax 60%",
                "set rates science 30% luxury 10% tax 60%"]


def build_summary(
    state: GameState,
    catalog: ActionCatalog,
    areas: dict[Tile, str] | None = None,
    *,
    role: Role = "all",
    max_turns: int | None = None,
    max_tokens: int = 1500,
) -> str:
    """Сводка хода для роли в пределах бюджета токенов."""
    areas = areas or {}
    builders: dict[str, Callable[[], str]] = {
        "overview": lambda: overview(state, max_turns),
        "research": lambda: research(state, catalog),
        "cities": lambda: cities(state, catalog),
        "cities_brief": lambda: cities(state, catalog, brief=True),
        "units": lambda: units(state, catalog, areas),
        "units_brief": lambda: units(state, catalog, areas, brief=True),
        "workers": lambda: units(state, catalog, areas, only=lambda u: u.is_worker),
        "military_units": lambda: units(state, catalog, areas, only=lambda u: not u.is_worker),
        "government": lambda: government(state, catalog),
    }
    parts: list[str] = []
    used = 0
    for name in ROLE_SECTIONS[role]:
        text = builders[name]()
        if not text:
            continue
        cost = estimate_tokens(text)
        if used + cost > max_tokens:
            parts.append(f"[{name}: omitted, token budget]")
            continue
        parts.append(text)
        used += cost
    return "\n\n".join(parts)


# --- Разделы --------------------------------------------------------------------


def overview(state: GameState, max_turns: int | None = None) -> str:
    me = state.me
    turn = f"Turn {state.turn}" + (f" of {max_turns}" if max_turns else "")
    lines = [
        f"{turn}. Government: {me.government}. Gold: {me.gold}. Score: {me.score}.",
        f"Rates: science {me.science}%, tax {me.tax}%, luxury {me.luxury}%.",
        f"Cities: {len(state.cities)}. Units: {len(state.units)} ({_count_types(state.units)}).",
    ]
    contacts = [o for o in state.opponents if o.diplomacy not in ("no contact", "unknown")]
    if contacts:
        lines.append("Contacts: " + "; ".join(f"{o.name} ({o.diplomacy}, score {o.score})" for o in contacts))
    else:
        lines.append(f"No contact with {len(state.opponents)} other civilizations yet.")
    return "\n".join(lines)


def research(state: GameState, catalog: ActionCatalog) -> str:
    me = state.me
    lines = ["RESEARCH"]
    if me.researching:
        lines.append(f"Researching {me.researching}: {me.research_bulbs}/{me.research_cost} bulbs.")
    else:
        lines.append("Not researching anything.")
    current = f"research {me.researching}" if me.researching else None
    actions = [a for a in catalog.options("tech", "cur_player") if a.startswith("research ") and a != current]
    if actions:
        lines.append("Can research now: " + ", ".join(a.removeprefix("research ") for a in actions))
        lines.append('Actions (actor tech cur_player): "research <tech>", "set research goal <tech>"')
    return "\n".join(lines)


def cities(state: GameState, catalog: ActionCatalog, *, brief: bool = False) -> str:
    if not state.cities:
        return "CITIES\nNo cities yet."
    lines = ["CITIES"]
    for city in state.cities:
        lines.append(_city_line(city, brief))
        if not brief:
            options = catalog.options("city", city.id)
            if options:
                lines.append(f"  actions: {_limit(options, 12)}")
    return "\n".join(lines)


def units(
    state: GameState,
    catalog: ActionCatalog,
    areas: dict[Tile, str],
    *,
    only: Callable[[UnitState], bool] | None = None,
    brief: bool = False,
) -> str:
    selected = [u for u in state.units if only is None or only(u)]
    if not selected:
        return ""
    by_tile: dict[Tile, list[UnitState]] = defaultdict(list)
    for unit in selected:
        by_tile[(unit.x, unit.y)].append(unit)

    lines = ["UNITS"]
    for tile, tile_units in by_tile.items():
        lines.append(f"Tile {tile}:")
        for unit in tile_units:
            lines.append(f"  {unit.type} {unit.id} (hp {unit.health}/{unit.max_health}, moves {unit.moves_left})")
            if not brief:
                options = catalog.options("unit", unit.id)
                lines.append(f"    actions: {_compact_actions(options)}" if options else "    no actions left")
        if not brief and tile in areas:
            lines.append("  surroundings:")
            lines.extend(f"    {line}" for line in areas[tile].splitlines())
    return "\n".join(lines)


def government(state: GameState, catalog: ActionCatalog) -> str:
    options = set(catalog.options("gov", 0))
    if not options:
        return ""
    presets = [p for p in RATE_PRESETS if p in options]
    changes = sorted(o for o in options if o.startswith("change government") and not o.endswith("Anarchy"))
    lines = ["GOVERNMENT (actor gov 0)", f"Current: {state.me.government}."]
    if presets:
        lines.append("Rate options: " + "; ".join(f'"{p}"' for p in presets))
    if changes:
        lines.append("Government options: " + "; ".join(f'"{c}"' for c in changes))
    return "\n".join(lines)


# --- Окрестности ----------------------------------------------------------------


def areas_from_llm_info(llm_info: dict[str, Any], state: GameState) -> dict[Tile, str]:
    """Описания окрестностей из info['llm_info'] LLMWrapper — по одному на клетку, а не на юнит."""
    positions = {str(u.id): (u.x, u.y) for u in state.units}
    areas: dict[Tile, str] = {}
    for unit_id, unit_info in llm_info.get("unit", {}).items():
        tile = positions.get(str(unit_id))
        if tile is None or tile in areas:
            continue  # юниты на одной клетке видят одно и то же
        observations = unit_info.get("observations")
        if observations:
            areas[tile] = describe_area(observations)
    return areas


# --- Вспомогательное ------------------------------------------------------------


def _city_line(city: CityState, brief: bool) -> str:
    growth = f"grows in {city.growth_turns}" if city.growth_turns is not None else "not growing"
    production = city.production or "production"
    line = f"{city.name} (id {city.id}, size {city.size}, at {(city.x, city.y)}): {growth}"
    if brief:
        return line
    line += (f"; {production} ready in {city.turns_to_complete}" if city.turns_to_complete is not None else "")
    line += f"; surplus food {city.food_surplus}, shields {city.shield_surplus}, trade {city.trade_surplus}"
    if city.unhappy or city.angry:
        line += f"; UNHAPPY {city.unhappy}, angry {city.angry}, happy {city.happy}"
    return line


def _compact_actions(options: list[str]) -> str:
    """'move North', 'move NorthEast', ... → 'move N/NE/...'; остальные — как есть."""
    moves = [SHORT_BY_FULL[o.split(" ", 1)[1]] for o in options if o.startswith("move ") and o.split(" ", 1)[1] in SHORT_BY_FULL]
    others = [o for o in options if not (o.startswith("move ") and o.split(" ", 1)[1] in SHORT_BY_FULL)]
    return ", ".join(others + ([f"move {'/'.join(moves)}"] if moves else []))


def _count_types(units: list[UnitState]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for unit in units:
        counts[unit.type] += 1
    return ", ".join(f"{n} {t}" for t, n in counts.items()) or "none"


def _limit(options: list[str], limit: int) -> str:
    shown = ", ".join(options[:limit])
    return shown + (f" … and {len(options) - limit} more" if len(options) > limit else "")