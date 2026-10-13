"""Правительство: оркестрация агентов на ходу.

Режимы (для сравнения архитектур в evals):
- hierarchy — правитель (раз в N ходов: стратегия и налоги) → советники
  (наука по необходимости, экономика, военный) → исполнитель (GameSession: проверка и отправка);
- single — один агент решает всё одним вызовом.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from civ_agents.agents.advisor import Advisor
from civ_agents.agents.skills import skill_hints
from civ_agents.env.actions import Action, InvalidAction
from civ_agents.env.session import GameSession, TurnView
from civ_agents.llm.base import LLMClient
from civ_agents.memory.episodic import EpisodicMemory
from civ_agents.observability import otel
from civ_agents.observability.context import call_context

logger = logging.getLogger(__name__)


@dataclass
class TurnReport:
    turn: int
    seconds: float
    agents_called: list[str] = field(default_factory=list)
    actions_sent: int = 0
    actions_failed: int = 0


class Government:
    def __init__(
        self,
        llm: LLMClient,
        *,
        mode: str = "hierarchy",
        memory: EpisodicMemory | None = None,
        knowledge=None,
        use_skills: bool = True,
        strategy_every: int = 10,
        turn_budget_s: float = 60.0,
    ) -> None:
        self.mode = mode
        self.memory = memory
        self.use_skills = use_skills
        self.strategy_every = strategy_every
        self.turn_budget_s = turn_budget_s
        self.strategy = "Expand: found new cities quickly on good land, explore, research basic techs."
        roles = ["single"] if mode == "single" else ["ruler", "science", "economist", "military"]
        self.advisors = {role: Advisor(role, llm, knowledge=knowledge) for role in roles}
        self.reports: list[TurnReport] = []

    def play_turn(self, session: GameSession) -> TurnReport:
        started = time.monotonic()
        deadline = started + self.turn_budget_s
        view = session.view
        if self.memory:
            self.memory.observe(view.state)
        report = TurnReport(turn=view.turn, seconds=0.0)

        with call_context(turn=view.turn), otel.span(f"turn {view.turn}", turn=view.turn, mode=self.mode):
            for role in self._roles_for(view):
                if time.monotonic() >= deadline:
                    logger.warning("Ход %s: бюджет времени исчерпан, %s пропущен", view.turn, role)
                    break
                with otel.span(f"agent {role}", agent=role, turn=view.turn):
                    advice = self.advisors[role].decide(view, self._context(role, view), deadline)
                report.agents_called.append(role)
                if advice.strategy:
                    self.strategy = advice.strategy
                with otel.span("executor", actions=len(advice.actions)):
                    view = self._execute(session, advice.actions, report)
            session.end_turn()

        report.seconds = round(time.monotonic() - started, 2)
        otel.record_turn(report.seconds, report.actions_sent, report.actions_failed)
        self.reports.append(report)
        return report

    def _roles_for(self, view: TurnView) -> list[str]:
        if self.mode == "single":
            return ["single"]
        roles = []
        if view.turn == 1 or view.turn % self.strategy_every == 0:
            roles.append("ruler")
        if view.state.research_options and (view.turn == 1 or view.state.me.researching is None):
            roles.append("science")
        workers = {u.id for u in view.state.units if u.is_worker}
        if view.catalog.actors("city") or any(a in workers for a in view.catalog.actors("unit")):
            roles.append("economist")
        if any(a not in workers for a in view.catalog.actors("unit")):
            roles.append("military")
        return roles

    def _context(self, role: str, view: TurnView) -> str:
        parts = [f"Current strategy: {self.strategy}"]
        if self.memory and (recent := self.memory.recent()):
            parts.append(recent)
        if self.use_skills and (hints := skill_hints(role, view)):
            parts.append(hints)
        return "\n\n".join(parts)

    @staticmethod
    def _execute(session: GameSession, actions: list[Action], report: TurnReport) -> TurnView:
        for action in actions:
            try:
                session.act(action)
                report.actions_sent += 1
            except InvalidAction as error:  # стало недопустимым после предыдущих действий хода
                report.actions_failed += 1
                logger.info("Действие отклонено исполнителем: %s", error)
        return session.view

    def stats(self) -> dict:
        per_agent = {role: vars(a.stats) for role, a in self.advisors.items()}
        times = [r.seconds for r in self.reports]
        return {
            "mode": self.mode,
            "turns": len(self.reports),
            "turn_seconds_avg": round(sum(times) / len(times), 2) if times else 0,
            "turn_seconds_max": max(times, default=0),
            "actions_sent": sum(r.actions_sent for r in self.reports),
            "actions_failed": sum(r.actions_failed for r in self.reports),
            "agents": per_agent,
        }
