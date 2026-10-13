"""Разведка, часть 3: полное описание одного юнита от LLMWrapper и его вес в токенах."""

import json

import gymnasium

import civrealm  # noqa: F401
from civrealm.configs import fc_args
from civrealm.envs.freeciv_wrapper import LLMWrapper

from civ_agents.llm.postprocess import estimate_tokens
from civ_agents.observability.run import Run


def jsonable(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def main():
    run = Run.create("runs", name="explore-llm-unit")
    env = LLMWrapper(gymnasium.make("civrealm/FreecivBase-v0"))
    observations, info = env.reset(client_port=fc_args["client_port"])

    units = info["llm_info"]["unit"]
    report = {}
    for unit_id, unit in list(units.items())[:2]:
        full = jsonable(unit)
        text = json.dumps(full, ensure_ascii=False)
        report[str(unit_id)] = {
            "chars": len(text),
            "estimated_tokens": estimate_tokens(text),
            "minimap_chars": len(json.dumps(full["observations"]["minimap"], ensure_ascii=False)),
            "upper_map_chars": len(json.dumps(full["observations"]["upper_map"], ensure_ascii=False)),
            "full": full,
        }
        print(f"{unit['name']}: ~{report[str(unit_id)]['estimated_tokens']} токенов")

    path = run.dir / "llm_units.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {path}")
    env.close()


if __name__ == "__main__":
    main()