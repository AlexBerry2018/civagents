"""Окрестности клетки: описание от LLMWrapper → компактный текст для модели.

LLMWrapper описывает окрестности каждого юнита JSON-ом примерно на 900 токенов:
    {"minimap": {"current_tile": [...], "tile_north_1_east_2": [...], ...},
     "upper_map": {"current_block": [...], "block_north_1": [...], ...}}
Здесь он превращается в несколько строк текста. Модуль — единственное место,
которое знает формат LLMWrapper: если формат изменится, правится только он.
"""

from __future__ import annotations

import re
from typing import Any

_TILE_KEY = re.compile(r"^(?:tile|block)((?:_(?:north|south|east|west)_\d+)*)$")
_PART = re.compile(r"_(north|south|east|west)_(\d+)")
_OWNERS = re.compile(r"^unit owners are:\s*(.+)$")

# Порядок обхода: сначала ближние клетки, по часовой стрелке от севера.
_COMPASS = {(-1, 0): "N", (-1, 1): "NE", (0, 1): "E", (1, 1): "SE", (1, 0): "S", (1, -1): "SW", (0, -1): "W", (-1, -1): "NW"}


def offset(key: str) -> tuple[int, int] | None:
    """'tile_north_2_east_1' → (-2, 1): (строки, столбцы), север — минус. current_* → (0, 0)."""
    if key in ("current_tile", "current_block"):
        return (0, 0)
    match = _TILE_KEY.match(key)
    if not match:
        return None
    dy = dx = 0
    for direction, distance in _PART.findall(match.group(1)):
        d = int(distance)
        if direction == "north":
            dy -= d
        elif direction == "south":
            dy += d
        elif direction == "east":
            dx += d
        else:
            dx -= d
    return (dy, dx)


def label(dy: int, dx: int) -> str:
    """(−1, 1) → 'NE'; (−2, 1) → '2N1E'. Соседние клетки — сторонами света, дальние — шагами."""
    if (dy, dx) in _COMPASS:
        return _COMPASS[(dy, dx)]
    parts = []
    if dy:
        parts.append(f"{abs(dy)}{'N' if dy < 0 else 'S'}")
    if dx:
        parts.append(f"{abs(dx)}{'E' if dx > 0 else 'W'}")
    return "".join(parts) or "here"


def clean(items: list[str]) -> list[str]:
    """['1 Hills', '1 Wine', '2 Settlers', 'unit owners are: myself player_0'] → ['Hills', 'Wine', '2 Settlers (own)']."""
    result: list[str] = []
    owners = None
    for item in items:
        text = str(item).strip()
        if match := _OWNERS.match(text):
            owners = match.group(1)
            continue
        if text.startswith("1 "):
            text = text[2:]
        result.append(text)
    if owners:
        mark = "own" if owners.startswith("myself") else owners.replace("player_", "player ")
        result.append(f"[units: {mark}]")
    return result


def _distance(key_offset: tuple[int, int]) -> int:
    return max(abs(key_offset[0]), abs(key_offset[1]))


def _order(item: tuple[tuple[int, int], Any]) -> tuple[int, int, int]:
    (dy, dx), _ = item
    return (_distance((dy, dx)), dy, dx)


def describe_minimap(minimap: dict[str, list[str]]) -> list[str]:
    """Клетки вокруг: одна строка на клетку, неразведанные — одной строкой в конце."""
    tiles = sorted(((offset(k), v) for k, v in minimap.items() if offset(k) is not None), key=_order)
    lines, unexplored = [], []
    for (dy, dx), items in tiles:
        parts = clean(items)
        if parts == ["tiles unexplored"]:
            unexplored.append(label(dy, dx))
            continue
        lines.append(f"{label(dy, dx)}: {', '.join(parts)}")
    if unexplored:
        lines.append(f"unexplored: {' '.join(unexplored)}")
    return lines


def describe_upper_map(upper_map: dict[str, list[str]]) -> list[str]:
    """Дальние области 5×5: какие стороны не разведаны и что заметного есть в разведанных."""
    unexplored, notable = [], []
    for key, items in upper_map.items():
        position = offset(key)
        if position is None or position == (0, 0):
            continue  # текущая область уже описана клетками
        parts = clean(items)
        if len(parts) == 1 and parts[0].endswith("tiles unexplored"):
            unexplored.append(label(*position))
        else:
            notable.append(f"area {label(*position)}: {', '.join(p for p in parts if 'unexplored' not in p)}")
    lines = notable
    if unexplored:
        lines.append(f"unexplored areas: {' '.join(unexplored)}")
    return lines


def describe_area(observations: dict[str, Any]) -> str:
    """observations юнита или города из LLMWrapper → компактный текст."""
    lines = describe_minimap(observations.get("minimap", {}))
    lines += describe_upper_map(observations.get("upper_map", {}))
    return "\n".join(lines)