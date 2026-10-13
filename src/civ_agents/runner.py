"""Одна партия целиком — общая точка входа для scripts/play.py и scripts/evals.py."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from civ_agents.agents.government import Government
from civ_agents.config import Settings, save_snapshot
from civ_agents.env.session import GameSession
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.memory.episodic import EpisodicMemory
from civ_agents.memory.reflection import reflect
from civ_agents.memory.semantic import KnowledgeBase, make_knowledge
from civ_agents.observability import otel
from civ_agents.observability.alerts import AlertMonitor
from civ_agents.observability.call_log import CallLog
from civ_agents.observability.run import Run
from civ_agents.observability.summary import load_calls, summarize

logger = logging.getLogger(__name__)
KNOWLEDGE_DIR = Path("knowledge")


def play_game(
    settings: Settings,
    *,
    mode: str = "hierarchy",
    memory: bool = True,
    skills: bool = True,
    reflection: bool = True,
    name: str | None = None,
    on_turn=None,
) -> dict[str, Any]:
    """mode: hierarchy | single | random. Возвращает метрики партии (и пишет metrics.json)."""
    variant = name or mode + ("" if memory else "-nomem") + ("" if skills else "-noskills")
    run = Run.create(settings.paths.runs_dir, name=variant)
    save_snapshot(settings, run.config_path)

    otel.setup(run_id=run.run_id)
    llm = government = episodic = alerts = None
    if mode != "random":
        call_log = CallLog.for_run(run, save_prompts=settings.logging.save_prompts)
        llm = OpenAICompatClient(settings.llm.active, observer=otel.chain(call_log, otel.llm_observer))
        llm.health()
        llm.warmup()
        episodic = EpisodicMemory(run.dir / "events.jsonl") if memory else None
        knowledge = make_knowledge(KnowledgeBase.from_dir(KNOWLEDGE_DIR)) if memory else None
        government = Government(llm, mode=mode, memory=episodic, knowledge=knowledge, use_skills=skills,
                                turn_budget_s=settings.limits.turn_time_budget_s)
        alerts = AlertMonitor(government, settings.limits.turn_time_budget_s, run.dir / "alerts.jsonl")

    session = GameSession(settings, run)
    session.start()
    turn_seconds: list[float] = []
    try:
        while not session.done:
            if government:
                report = government.play_turn(session)
                turn_seconds.append(report.seconds)
                alerts.check(report)
                state = session.view.state
                otel.set_game_state(turn=state.turn, cities=len(state.cities), units=len(state.units),
                                    score=state.me.score, gold=state.me.gold)
                if on_turn:
                    on_turn(report)
            else:
                _random_turn(session)
    finally:
        results = session.close()

    state = session.view.state
    outcome = (f"turns {results['turns_played']}, cities {len(state.cities)}, units {len(state.units)}, "
               f"score {state.me.score}, techs known {len(state.known_techs)}, researching {state.me.researching}")
    metrics: dict[str, Any] = {
        "variant": variant,
        "run_id": run.run_id,
        "turns": results["turns_played"],
        "survived": not results["terminated"] or bool(state.cities),
        "cities": len(state.cities),
        "population": sum(c.size for c in state.cities),
        "units": len(state.units),
        "score": state.me.score,
        "game": results,
        "outcome": outcome,
    }
    if government:
        metrics["government"] = government.stats()
        metrics["alerts"] = [vars(a) for a in alerts.alerts]
        metrics["llm"] = summarize(load_calls(run.dir))
        if episodic and reflection:
            metrics["lessons"] = reflect(llm, episodic.events, outcome, KNOWLEDGE_DIR / "lessons.md", run.run_id)
    (run.dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics


def _random_turn(session: GameSession) -> None:
    """Базовая линия: как RandomAgent CivRealm — поселенцы строят (80%), остальные ходят случайно."""
    view = session.view
    for unit_id in view.catalog.actors("unit"):
        options = session.view.catalog.options("unit", unit_id)
        if not options:
            continue
        choice = "build city" if "build city" in options and random.random() < 0.8 else random.choice(
            [o for o in options if o.startswith("move ")] or options)
        try:
            session.act(session.view.catalog.resolve("unit", unit_id, choice))
        except Exception:  # noqa: BLE001 — юнит мог исчезнуть после предыдущего действия
            continue
    session.end_turn()
