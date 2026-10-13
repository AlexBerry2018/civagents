"""Разведка, часть 2: полные записи юнита, игрока и список всех ключей наблюдения."""

import json

import gymnasium

import civrealm  # noqa: F401
from civrealm.configs import fc_args

from civ_agents.observability.run import Run


def jsonable(obj, max_list=20):
    """Перевести в JSON: массивы — в описание формы, длинные списки — обрезать."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        if obj.size <= max_list:
            return obj.tolist()
        return f"ndarray shape={list(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, dict):
        return {str(k): jsonable(v, max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, max_list) for v in obj[:max_list]]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)[:200]


def main():
    run = Run.create("runs", name="explore-details")
    env = gymnasium.make("civrealm/FreecivBase-v0")
    observations, info = env.reset(client_port=fc_args["client_port"])

    details = {
        "top_level_keys": {k: type(v).__name__ for k, v in observations.items()},
        "map_keys": {k: f"{getattr(v, 'shape', '?')} {getattr(v, 'dtype', '')}" for k, v in observations["map"].items()},
        "game": jsonable(observations["game"]),
        "my_player": jsonable(observations["player"][0]),
        "units": {str(k): jsonable(v) for k, v in list(observations.get("unit", {}).items())[:5]},
        "tech_sample": jsonable(dict(list(observations.get("tech", {}).items())[:3])),
        "other_keys": {
            k: jsonable(dict(list(v.items())[:2])) if isinstance(v, dict) else jsonable(v)
            for k, v in observations.items()
            if k not in ("game", "rules", "map", "player", "unit", "tech", "city")
        },
    }
    path = run.dir / "details.json"
    path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {path}")
    env.close()


if __name__ == "__main__":
    main()