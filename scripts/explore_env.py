"""Разведка CivRealm: что среда отдаёт агенту.

    uv run python scripts/explore_env.py           # базовая среда
    uv run python scripts/explore_env.py --llm     # с LLMWrapper

Сохраняет в runs/explore-*/ описание observations, info и доступных действий
на первом и втором ходу.
"""

import json
import sys

import gymnasium

import civrealm  # noqa: F401 — регистрирует среды в gymnasium
from civrealm.agents import RandomAgent, RandomLLMAgent
from civrealm.configs import fc_args
from civrealm.envs.freeciv_wrapper import LLMWrapper

from civ_agents.observability.run import Run


def describe(obj, depth=0, max_depth=4, max_items=5):
    """Компактное описание структуры: типы, размеры и несколько примеров."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):  # numpy-массив
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype),
                "sample": obj.flatten()[:max_items].tolist()}
    if isinstance(obj, dict):
        if depth >= max_depth:
            return {"type": "dict", "len": len(obj), "keys": [str(k) for k in list(obj)[:max_items]]}
        items = list(obj.items())
        result = {str(k): describe(v, depth + 1, max_depth, max_items) for k, v in items[:max_items]}
        if len(items) > max_items:
            result["..."] = f"ещё {len(items) - max_items} ключей, всего {len(items)}"
        return result
    if isinstance(obj, (list, tuple)):
        return {"type": type(obj).__name__, "len": len(obj),
                "sample": [describe(v, depth + 1, max_depth, max_items) for v in obj[:max_items]]}
    if isinstance(obj, str) and len(obj) > 300:
        return obj[:300] + f"... (всего {len(obj)} символов)"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return {"type": type(obj).__name__, "repr": repr(obj)[:200]}


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  сохранено: {path} ({path.stat().st_size // 1024} КБ)")


def dump_turn(run, label, observations, info):
    print(f"Ход {info.get('turn')}: {label}")
    save(run.dir / f"{label}_observations.json", describe(observations))
    save(run.dir / f"{label}_info.json", describe({k: v for k, v in info.items() if k != "available_actions"}))

    actions = info.get("available_actions", {})
    summary = {}
    for ctrl_type, actors in actions.items():
        valid_examples = {}
        for actor_id, action_dict in list(actors.items())[:3]:
            valid_examples[str(actor_id)] = sorted(name for name, ok in action_dict.items() if ok)
        summary[ctrl_type] = {"actors": len(actors), "valid_actions_of_first_actors": valid_examples}
    save(run.dir / f"{label}_available_actions.json", summary)


def main():
    use_llm = "--llm" in sys.argv
    run = Run.create("runs", name="explore-llm" if use_llm else "explore")

    env = gymnasium.make("civrealm/FreecivBase-v0")
    if use_llm:
        env = LLMWrapper(env)
    agent = RandomLLMAgent() if use_llm else RandomAgent()

    observations, info = env.reset(client_port=fc_args["client_port"])
    dump_turn(run, "turn1", observations, info)

    start_turn = info["turn"]
    actions_log = []
    for step in range(300):
        action = agent.act(observations, info)
        actions_log.append({"step": step, "turn": info["turn"], "action": repr(action)})
        observations, reward, terminated, truncated, info = env.step(action)
        if info.get("turn") != start_turn or terminated or truncated:
            break
    save(run.dir / "actions_turn1.json", actions_log)

    if info:
        dump_turn(run, "turn2", observations, info)
    env.close()
    print(f"Готово: {run.dir}")


if __name__ == "__main__":
    main()