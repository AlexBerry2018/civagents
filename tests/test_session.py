"""GameSession на поддельной среде: действия, конец хода, горизонт, запись прогона."""

import json
from types import SimpleNamespace

import pytest

from civ_agents.env.actions import Action, InvalidAction
from civ_agents.env.session import GameSession
from civ_agents.observability.run import Run


class FakeEnv:
    """Поддельная игра: один поселенец, наука; ход растёт на step(None)."""

    def __init__(self, truncate_on_turn=None):
        self.turn = 1
        self.sent = []
        self.closed = False
        self.truncate_on_turn = truncate_on_turn

    def observation(self):
        observations = {
            "player": {0: {"name": "Myagent", "gold": 50, "researching": 2}},
            "unit": {101: {"owner": 0, "x": 1, "y": 1, "type_rule_name": "Settlers", "type_worker": True}},
            "tech": {2: {"name": "Alphabet", "inv_state": 1}, 9: {"name": "Bronze Working", "inv_state": 1}},
        }
        info = {
            "turn": self.turn,
            "available_actions": {
                "unit": {101: {"build_city": True, "goto_1": True}},
                "tech": {"cur_player": {"research_tech_Alphabet_2": True, "research_tech_Bronze Working_9": True}},
            },
            "llm_info": {"unit": {101: {"observations": {"minimap": {"current_tile": ["1 Grassland"]}}}}},
        }
        return observations, info

    def reset(self):
        return self.observation()

    def step(self, action):
        self.sent.append(action)
        if action is None:
            self.turn += 1
        if self.truncate_on_turn and self.turn >= self.truncate_on_turn:
            return {}, 0, False, True, {}  # так CivRealm сообщает об обрыве
        observations, info = self.observation()
        return observations, 0, False, False, info

    def llm_info(self, observations, info):
        return {}

    def results(self):
        return {"0": {"score": 3}} if self.closed else {}  # CivRealm считает результаты при закрытии

    def close(self):
        self.closed = True


def settings(max_turns=3):
    return SimpleNamespace(game=SimpleNamespace(max_turns=max_turns))


@pytest.fixture
def run(tmp_path):
    return Run.create(tmp_path, name="test")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_start_builds_view(run):
    session = GameSession(settings(), run, env=FakeEnv())
    view = session.start()
    assert view.turn == 1
    assert view.catalog.options("unit", 101) == ["build city", "move North"]
    assert view.areas == {(1, 1): "here: Grassland"}
    assert "Settlers 101" in view.summary()


def test_valid_action_is_sent(run):
    env = FakeEnv()
    session = GameSession(settings(), run, env=env)
    view = session.start()
    session.act(view.catalog.resolve("unit", "101", "build city"))
    assert env.sent == [("unit", 101, "build_city")]


def test_invalid_action_never_reaches_game(run):
    env = FakeEnv()
    session = GameSession(settings(), run, env=env)
    session.start()
    with pytest.raises(InvalidAction):
        session.act(Action("unit", 101, "fortify"))
    assert env.sent == []
    assert session.stats.actions_rejected == 1


def test_repeating_current_research_is_skipped(run):
    env = FakeEnv()
    session = GameSession(settings(), run, env=env)
    session.start()
    session.act(Action("tech", "cur_player", "research_tech_Alphabet_2"))  # уже исследуется
    session.act(Action("tech", "cur_player", "research_tech_Bronze Working_9"))
    assert env.sent == [("tech", "cur_player", "research_tech_Bronze Working_9")]
    assert session.stats.actions_skipped == 1


def test_horizon_ends_game(run):
    env = FakeEnv()
    session = GameSession(settings(max_turns=3), run, env=env)
    session.start()
    while not session.done:
        session.end_turn()
    assert session.view.turn == 4
    assert session.stats.turns_played == 3
    assert env.sent == [None, None, None]


def test_truncation_keeps_last_state(run):
    session = GameSession(settings(max_turns=10), run, env=FakeEnv(truncate_on_turn=2))
    session.start()
    session.end_turn()
    assert session.done
    assert session.stats.truncated
    assert session.view.turn == 1  # пустое наблюдение при обрыве не затёрло состояние


def test_close_saves_results_after_env_close(run):
    session = GameSession(settings(max_turns=1), run, env=FakeEnv())
    session.start()
    session.end_turn()
    summary = session.close()
    assert summary["game_results"] == {"0": {"score": 3}}
    saved = json.loads((run.dir / "game_results.json").read_text(encoding="utf-8"))
    assert saved["turns_played"] == 1


def test_run_records_turns_and_actions(run):
    session = GameSession(settings(max_turns=2), run, env=FakeEnv())
    view = session.start()
    session.act(view.catalog.resolve("unit", 101, "build city"))
    with pytest.raises(InvalidAction):
        session.act(Action("unit", 101, "fortify"))
    while not session.done:
        session.end_turn()

    turns = read_lines(run.dir / "turns.jsonl")
    actions = read_lines(run.dir / "actions.jsonl")
    assert [t["turn"] for t in turns] == [1, 2, 3]
    assert turns[0]["state"]["me"]["name"] == "Myagent"
    assert [a["accepted"] for a in actions] == [True, False]


def test_view_before_start_fails():
    with pytest.raises(RuntimeError, match="start"):
        GameSession(settings(), env=FakeEnv()).view