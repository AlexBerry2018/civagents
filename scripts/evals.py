"""Evals уровня системы: несколько вариантов × несколько партий → таблица в docs/evals_results.md.

    uv run python scripts/evals.py --turns 15 --games 2
    uv run python scripts/evals.py --turns 15 --games 1 --variants random hierarchy
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

from civ_agents.config import load_settings
from civ_agents.runner import play_game

VARIANTS = {
    "random": dict(mode="random"),
    "single": dict(mode="single"),
    "hierarchy": dict(mode="hierarchy"),
    "hierarchy-nomem": dict(mode="hierarchy", memory=False),
    "hierarchy-noskills": dict(mode="hierarchy", skills=False),
}


def mean_sd(values):
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["random", "single", "hierarchy", "hierarchy-nomem"])
    parser.add_argument("--collect", action="store_true",
                        help="не играть, а собрать таблицу из уже сыгранных партий runs/*-eval-*/metrics.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if not args.collect:
        for variant in args.variants:
            for game in range(args.games):
                settings = load_settings()
                settings.game.max_turns = args.turns
                print(f"[{variant}] партия {game + 1}/{args.games}…", flush=True)
                metrics = play_game(settings, name=f"eval-{variant}", reflection=False, **VARIANTS[variant])
                print(f"  {metrics['outcome']}", flush=True)

    # Таблица всегда строится из файлов: так в неё попадают и партии прошлых запусков.
    rows: dict[str, list] = {}
    for path in sorted(Path("runs").glob("*-eval-*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.setdefault(metrics["variant"].removeprefix("eval-"), []).append(metrics)
    turns_seen = sorted({m["turns"] for results in rows.values() for m in results})
    args.games = max((len(r) for r in rows.values()), default=0)

    lines = [
        f"# Результаты evals (ходов в партии: {', '.join(map(str, turns_seen))}; до {args.games} партий на вариант)\n",
        "| Вариант | Партий | Счёт | Города | Население | Юниты | Выжил | Время хода, с (ср / макс) | Вызовов LLM на ход | "
        "Ошибки формата | Недопустимые действия | Токенов на ход |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, results in rows.items():
        turn_avg = [m["government"]["turn_seconds_avg"] for m in results if "government" in m]
        turn_max = [m["government"]["turn_seconds_max"] for m in results if "government" in m]
        calls = parse = invalid = valid = tokens = turns = 0
        for m in results:
            turns += m["turns"]
            for agent in m.get("government", {}).get("agents", {}).values():
                calls += agent["calls"]
                parse += agent["parse_errors"]
                invalid += agent["invalid_actions"]
                valid += agent["valid_actions"]
            tokens += m.get("llm", {}).get("prompt_tokens", 0) + m.get("llm", {}).get("completion_tokens", 0)
        llm_cols = (
            f"{calls / turns:.1f} | {parse / calls:.0%} | {invalid / max(1, invalid + valid):.0%} | {tokens / turns:.0f}"
            if calls else "— | — | — | —"
        )
        lines.append(
            f"| {variant} | {len(results)} | {mean_sd([m['score'] for m in results])} | {mean_sd([m['cities'] for m in results])} | "
            f"{mean_sd([m['population'] for m in results])} | {mean_sd([m['units'] for m in results])} | "
            f"{sum(m['survived'] for m in results)}/{len(results)} | "
            f"{mean_sd(turn_avg)} / {max(turn_max, default=0):.1f} | {llm_cols} |"
        )
    lines.append("\nОшибки формата — доля вызовов, где JSON не разобран; недопустимые — доля действий, "
                 "отклонённых проверкой по каталогу (до отправки в игру).")
    out = Path("docs/evals_results.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("runs/evals_raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()