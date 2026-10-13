"""Обёртка над средой: действия, состояние, окрестности, сводка, ожидание сервера. Без игры."""

import copy
import json
from pathlib import Path

import pytest

from civ_agents.env.actions import Action, ActionCatalog, InvalidAction, expand_direction, readable_name
from civ_agents.env.neighborhood import clean, describe_area, label, offset
from civ_agents.env.server import ServerNotReady, wait_for_server
from civ_agents.env.state import GameState, _turns
from civ_agents.env.summary import areas_from_llm_info, build_summary

FIXTURE = Path(__file__).parent / "fixtures" / "observation_turn5.json"

MOVES = {f"goto_{i}": True for i in range(8)}
AVAILABLE = {
    "unit": {
        101: {"build_city": True, "keep_activity": True, "fortify": False, **MOVES},
        102: {"cultivate": True, **MOVES},
        103: {"fortify": True, **MOVES},
    },
    "tech": {"cur_player": {"research_tech_Alphabet_2": True, "research_tech_Bronze Working_9": True,
                            "set_tech_goal_Monarchy_53": True}},
    "gov": {0: {"set_sci_lux_tax_60_0_40": True, "set_sci_lux_tax_40_0_60": True, "change_gov_Anarchy": True}},
    "dipl": {1: {}},
}

OBSERVATION = {
    "player": {
        0: {"name": "Myagent", "score": 1, "gold": 50, "government_name": "Despotism", "tax": 40,
            "science": 60, "luxury": 0, "researching": 2, "bulbs_researched": 3, "researching_cost": 28},
        1: {"name": "Pelasgos", "score": -1, "is_alive": True},
    },
    "unit": {
        101: {"owner": 0, "x": 10, "y": 5, "type_rule_name": "Settlers", "health": 20, "type_hp": 20,
              "moves_left": 3, "type_worker": True},
        102: {"owner": 0, "x": 10, "y": 5, "type_rule_name": "Workers", "health": 10, "type_hp": 10,
              "moves_left": 3, "type_worker": True},
        103: {"owner": 0, "x": 12, "y": 7, "type_rule_name": "Warriors", "health": 10, "type_hp": 10,
              "moves_left": 1, "type_attack_strength": 1, "type_defense_strength": 1},
        900: {"owner": 1, "x": 1, "y": 1, "type_rule_name": "Warriors"},
    },
    "city": {
        121: {"owner": 0, "name": "Ateste", "size": 1, "x": 11, "y": 5, "surplus_food": 1, "surplus_shield": 3,
              "surplus_trade": 1, "growth_in": "17 turns", "turns_to_prod_complete": 1, "ppl_content": 1},
    },
    "tech": {
        2: {"name": "Alphabet", "inv_state": 1},
        9: {"name": "Bronze Working", "inv_state": 1},
        53: {"name": "Monarchy", "inv_state": 0},
        88: {"name": "Pottery", "inv_state": 2},
    },
    "dipl": {"1": {"diplomatic_state": 5}},
}

MINIMAP = {
    "minimap": {
        "current_tile": ["1 Grassland", "1 River", "2 Settlers", "unit owners are: myself player_0"],
        "tile_north_1": ["1 Forest"],
        "tile_south_1_west_1": ["1 Grassland", "1 Minor Tribe Village"],
        "tile_north_2_east_1": ["1 Hills"],
        "tile_north_2_east_2": ["1 tiles unexplored"],
    },
    "upper_map": {
        "current_block": ["3 Forest"],
        "block_north_1": ["25 tiles unexplored"],
        "block_east_1": ["20 tiles unexplored", "5 Ocean"],
    },
}


def state() -> GameState:
    return GameState.from_observation(OBSERVATION, {"turn": 5})


# --- Действия ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ctrl", "name", "expected"),
    [
        ("unit", "goto_5", "move SouthWest"),
        ("unit", "goto_1", "move North"),
        ("unit", "hut_enter_5", "hut enter SouthWest"),
        ("unit", "build_city", "build city"),
        ("tech", "research_tech_Bronze Working_9", "research Bronze Working"),
        ("tech", "set_tech_goal_Monarchy_53", "set research goal Monarchy"),
        ("gov", "set_sci_lux_tax_60_0_40", "set rates science 60% luxury 0% tax 40%"),
        ("gov", "change_gov_Monarchy", "change government to Monarchy"),
        ("city", "change_build_Warriors", "change build Warriors"),
    ],
)
def test_readable_name(ctrl, name, expected):
    assert readable_name(ctrl, name) == expected


def test_expand_direction():
    assert expand_direction("move NE") == "move NorthEast"
    assert expand_direction("move ne") == "move NorthEast"
    assert expand_direction("build city") == "build city"


def test_catalog_lists_only_allowed_actions():
    catalog = ActionCatalog(AVAILABLE)
    assert catalog.actors("unit") == [101, 102, 103]
    assert catalog.actors("dipl") == []  # без допустимых действий — нет в каталоге
    assert "fortify" not in catalog.options("unit", 101)
    assert "build city" in catalog.options("unit", 101)


@pytest.mark.parametrize("name", ["move SouthWest", "move SW", "MOVE southwest", "goto_5"])
def test_resolve_accepts_variants(name):
    action = ActionCatalog(AVAILABLE).resolve("unit", "101", name)  # id строкой, как пришлёт модель
    assert action.to_civrealm() == ("unit", 101, "goto_5")


def test_resolve_tech_and_gov():
    catalog = ActionCatalog(AVAILABLE)
    assert catalog.resolve("tech", "cur_player", "research Alphabet").name == "research_tech_Alphabet_2"
    assert catalog.resolve("gov", "0", "set rates science 40% luxury 0% tax 60%").name == "set_sci_lux_tax_40_0_60"


def test_resolve_rejects_unavailable_with_options():
    with pytest.raises(InvalidAction, match="build city"):
        ActionCatalog(AVAILABLE).resolve("unit", 101, "fortify")


def test_resolve_rejects_unknown_actor():
    with pytest.raises(InvalidAction, match="101"):
        ActionCatalog(AVAILABLE).resolve("unit", 999, "move North")


def test_is_valid():
    catalog = ActionCatalog(AVAILABLE)
    assert catalog.is_valid(Action("unit", 101, "build_city"))
    assert not catalog.is_valid(Action("unit", 101, "fortify"))
    assert not catalog.is_valid(Action("unit", 102, "build_city"))


# --- Состояние --------------------------------------------------------------------


def test_state_player():
    me = state().me
    assert (me.name, me.gold, me.government) == ("Myagent", 50, "Despotism")
    assert me.researching == "Alphabet"  # id 2 → имя
    assert (me.research_bulbs, me.research_cost) == (3, 28)


def test_state_only_own_units():
    assert [u.id for u in state().units] == [101, 102, 103]


def test_state_research_excludes_current():
    s = state()
    assert s.research_options == ["Bronze Working"]  # Alphabet уже исследуется, Monarchy недоступна
    assert s.known_techs == ["Pottery"]


def test_state_city():
    [city] = state().cities
    assert (city.name, city.growth_turns, city.shield_surplus, city.turns_to_complete) == ("Ateste", 17, 3, 1)


def test_state_opponents():
    [opponent] = state().opponents
    assert opponent.diplomacy == "no contact"
    assert opponent.score == 0  # -1 «неизвестно» → 0


def test_state_tolerates_missing_fields():
    s = GameState.from_observation({}, {})
    assert s.turn == 0 and s.units == [] and s.me.name == ""


@pytest.mark.parametrize(("text", "expected"), [("17 turns", 17), ("1 turn", 1), ("never", None), (None, None)])
def test_turns_parsing(text, expected):
    assert _turns(text) == expected


@pytest.mark.skipif(not FIXTURE.exists(), reason="нет записанного наблюдения")
def test_state_from_real_observation():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    s = GameState.from_observation(data["observations"], data["info"])
    assert s.turn == data["info"]["turn"]
    assert s.me.name
    assert s.units, "у игрока должны быть юниты"
    assert all(u.type != "Unknown" for u in s.units)
    assert s.me.researching not in s.research_options
    assert len(s.opponents) == 3


# --- Окрестности ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected"),
    [("current_tile", (0, 0)), ("tile_north_1", (-1, 0)), ("tile_south_1_west_1", (1, -1)),
     ("tile_north_2_east_1", (-2, 1)), ("block_east_1", (0, 1)), ("something_else", None)],
)
def test_offset(key, expected):
    assert offset(key) == expected


@pytest.mark.parametrize(("dy", "dx", "expected"), [(-1, 0, "N"), (1, -1, "SW"), (-2, 1, "2N1E"), (0, 2, "2E")])
def test_label(dy, dx, expected):
    assert label(dy, dx) == expected


def test_clean():
    assert clean(["1 Hills", "2 Settlers", "unit owners are: myself player_0"]) == ["Hills", "2 Settlers", "[units: own]"]
    assert clean(["1 Warriors", "unit owners are: player_2"]) == ["Warriors", "[units: player 2]"]


def test_describe_area():
    text = describe_area(MINIMAP)
    lines = text.splitlines()
    assert lines[0] == "here: Grassland, River, 2 Settlers, [units: own]"  # текущая клетка первой
    assert "SW: Grassland, Minor Tribe Village" in lines
    assert "unexplored: 2N2E" in lines
    assert "area E: 5 Ocean" in lines
    assert "unexplored areas: N" in lines


# --- Сводка -----------------------------------------------------------------------


def test_summary_contains_sections_and_actions():
    text = build_summary(state(), ActionCatalog(AVAILABLE), max_turns=50)
    assert text.startswith("Turn 5 of 50.")
    assert "Can research now: Bronze Working" in text
    assert "research Alphabet" not in text.split("Can research now:")[1].splitlines()[0]
    assert "Ateste" in text
    assert "move NW/N/NE/W/E/SW/S/SE" in text  # движение сжато
    assert "Anarchy" not in text  # анархию не предлагаем


def test_summary_groups_units_by_tile_with_area():
    catalog = ActionCatalog(AVAILABLE)
    areas = {(10, 5): describe_area(MINIMAP)}
    text = build_summary(state(), catalog, areas)
    assert text.count("Tile (10, 5)") == 1
    assert text.count("here: Grassland") == 1  # окрестности клетки — один раз на два юнита


def test_summary_roles_select_sections():
    catalog = ActionCatalog(AVAILABLE)
    science = build_summary(state(), catalog, role="science")
    military = build_summary(state(), catalog, role="military")
    assert "RESEARCH" in science and "UNITS" not in science
    assert "Warriors 103" in military and "Settlers 101" not in military


def test_summary_respects_token_budget():
    text = build_summary(state(), ActionCatalog(AVAILABLE), max_tokens=120)
    assert "omitted, token budget" in text
    assert text.startswith("Turn 5")  # обзор — первым, он помещается


def test_summary_shows_unhappiness():
    observation = copy.deepcopy(OBSERVATION)
    observation["city"][121].update({"ppl_unhappy": 2, "ppl_angry": 1})
    s = GameState.from_observation(observation, {"turn": 5})
    assert "UNHAPPY 2, angry 1" in build_summary(s, ActionCatalog({}))


def test_areas_from_llm_info_one_per_tile():
    llm_info = {"unit": {101: {"observations": MINIMAP}, 102: {"observations": MINIMAP}}}
    areas = areas_from_llm_info(llm_info, state())
    assert list(areas) == [(10, 5)]


# --- Сервер -----------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_wait_for_server_until_ready():
    clock = FakeClock()
    answers = iter([None, 502, 502, 200])
    waited = wait_for_server(fetch=lambda url: next(answers), sleep=clock.sleep, clock=clock, interval_s=5)
    assert waited == 15


def test_wait_for_server_gives_up():
    clock = FakeClock()
    with pytest.raises(ServerNotReady, match="502"):
        wait_for_server(timeout_s=20, interval_s=5, fetch=lambda url: 502, sleep=clock.sleep, clock=clock)