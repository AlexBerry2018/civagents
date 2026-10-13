"""Сыграть партию агентами (через runner.play_game — с телеметрией и алертами).

    uv run python scripts/play.py --turns 20 --mode hierarchy
    uv run python scripts/play.py --turns 20 --mode single --no-memory --no-skills
    CIV_OTEL=1 uv run python scripts/play.py --turns 8      # + трейсы/метрики/логи в Grafana
"""

import argparse
import logging

from civ_agents.config import load_settings
from civ_agents.runner import play_game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--mode", choices=["hierarchy", "single", "random"], default="hierarchy")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = load_settings()
    if args.turns:
        settings.game.max_turns = args.turns

    def on_turn(report) -> None:
        print(f"Ход {report.turn}: {report.seconds:.1f} с, агенты {report.agents_called}, "
              f"действий {report.actions_sent} (отклонено {report.actions_failed})", flush=True)

    metrics = play_game(
        settings,
        mode=args.mode,
        memory=not args.no_memory,
        skills=not args.no_skills,
        reflection=not args.no_reflection,
        name=args.name,
        on_turn=on_turn,
    )
    print(f"\nИтог: {metrics['outcome']}")
    if metrics.get("alerts"):
        print(f"Алерты: {[a['name'] for a in metrics['alerts']]}")
    print(f"Метрики: runs/{metrics['run_id']}/metrics.json")


if __name__ == "__main__":
    main()
