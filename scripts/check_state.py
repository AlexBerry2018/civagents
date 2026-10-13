"""Проверка GameState на реальной игре и запись наблюдения для тестов."""

import json
from pathlib import Path

from civ_agents.env.server import wait_for_server

wait_for_server()

import gymnasium  # noqa: E402
import civrealm  # noqa: E402, F401
from civrealm.agents import RandomAgent  # noqa: E402
from civrealm.configs import fc_args  # noqa: E402

from civ_agents.env.state import GameState  # noqa: E402

TARGET_TURN = 5
FIXTURE = Path("tests/fixtures/observation_turn5.json")


def jsonable(obj):
    """Для записи наблюдения: numpy → списки, объекты без JSON-формы → строки."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


env = gymnasium.make("civrealm/FreecivBase-v0")
agent = RandomAgent()
observations, info = env.reset(client_port=fc_args["client_port"])
while info["turn"] < TARGET_TURN:
    observations, _, terminated, truncated, info = env.step(agent.act(observations, info))
    if terminated or truncated:
        break

state = GameState.from_observation(observations, info)
print(state.model_dump_json(indent=2))

cities = observations.get("city", {})
if cities:
    first = next(iter(cities.values()))
    print("\nПоля города:", sorted(first.keys()))

# Для тестов: всё, кроме карты и правил (они огромные и состоянию не нужны).
# У технологий оставляем только то, что читает state.py.
fixture = {k: v for k, v in observations.items() if k not in ("map", "rules")}
fixture["tech"] = {k: {"name": t.get("name"), "inv_state": t.get("inv_state")} for k, t in observations["tech"].items()}
FIXTURE.parent.mkdir(parents=True, exist_ok=True)
FIXTURE.write_text(json.dumps({"observations": jsonable(fixture), "info": {"turn": info["turn"]}}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nСохранено для тестов: {FIXTURE} ({FIXTURE.stat().st_size // 1024} КБ)")
env.close()