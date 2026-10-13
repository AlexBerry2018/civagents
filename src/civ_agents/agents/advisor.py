"""Советник: сводка своей роли → модель → JSON-план → проверенные действия.

Ошибки разбора и недопустимые действия возвращаются модели текстом (один переспрос).
Каждый советник командует только «своими» действующими лицами.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from civ_agents.agents.prompts import system_prompt
from civ_agents.agents.schema import Plan, PlanError, parse_plan
from civ_agents.env.actions import Action, InvalidAction
from civ_agents.env.session import TurnView
from civ_agents.llm.base import LLMClient
from civ_agents.llm.errors import LLMContentError, LLMError
from civ_agents.llm.types import Message
from civ_agents.observability.context import call_context

logger = logging.getLogger(__name__)

SUMMARY_ROLE = {"single": "all"}  # роль агента → раздел сводки


def _allowed(role: str, view: TurnView, decision_type: str, actor_id: str) -> bool:
    """Кем может командовать роль."""
    workers = {str(u.id) for u in view.state.units if u.is_worker}
    if role == "economist":
        return decision_type == "city" or (decision_type == "unit" and actor_id in workers)
    if role == "military":
        return decision_type == "unit" and actor_id not in workers
    if role == "science":
        return decision_type == "tech"
    if role == "ruler":
        return decision_type == "gov"
    return True


@dataclass
class AdviceStats:
    calls: int = 0
    parse_errors: int = 0
    invalid_actions: int = 0
    valid_actions: int = 0
    llm_errors: int = 0


@dataclass
class Advice:
    actions: list[Action] = field(default_factory=list)
    strategy: str | None = None


class Advisor:
    def __init__(
        self,
        role: str,
        llm: LLMClient,
        *,
        knowledge: Callable[[str, TurnView], str] | None = None,
        max_summary_tokens: int = 1500,
        retries: int = 1,
    ) -> None:
        self.role = role
        self.llm = llm
        self.knowledge = knowledge  # (роль, ход) → заметки из семантической памяти
        self.max_summary_tokens = max_summary_tokens
        self.retries = retries
        self.stats = AdviceStats()

    def decide(self, view: TurnView, context: str = "", deadline: float | None = None) -> Advice:
        notes = self.knowledge(self.role, view) if self.knowledge else ""
        messages = [
            Message.system(system_prompt(self.role, notes)),
            Message.user((context + "\n\n" if context else "") + view.summary(SUMMARY_ROLE.get(self.role, self.role), self.max_summary_tokens)),
        ]
        with call_context(agent=self.role):
            for attempt in range(self.retries + 1):
                try:
                    self.stats.calls += 1
                    response = self.llm.chat(messages, deadline=deadline)
                except LLMContentError as error:
                    self.stats.llm_errors += 1
                    text = error.response.text if error.response else ""
                    problem = f"Your answer was cut or malformed ({type(error).__name__})."
                except LLMError as error:
                    self.stats.llm_errors += 1
                    logger.warning("%s: модель недоступна: %s", self.role, error)
                    return Advice()
                else:
                    text = response.text
                    try:
                        plan = parse_plan(text)
                    except PlanError as error:
                        self.stats.parse_errors += 1
                        problem = str(error)
                    else:
                        advice, rejected = self._validate(plan, view)
                        if advice.actions or advice.strategy or not rejected:
                            return advice
                        problem = "None of your actions are valid: " + " | ".join(rejected[:4])
                if attempt < self.retries:
                    messages += [Message.assistant(text), Message.user(f"Error: {problem} Try again.")]
        return Advice()

    def _validate(self, plan: Plan, view: TurnView) -> tuple[Advice, list[str]]:
        advice = Advice(strategy=plan.strategy if self.role == "ruler" else None)
        rejected: list[str] = []
        seen: set[tuple[str, str]] = set()
        for decision in plan.actions:
            actor = (decision.type, str(decision.id))
            if actor in seen:
                continue  # одно действие на действующее лицо за раз
            if not _allowed(self.role, view, decision.type, str(decision.id)):
                rejected.append(f"{decision.type} {decision.id} is not yours to command")
                self.stats.invalid_actions += 1
                continue
            try:
                action = view.catalog.resolve(decision.type, decision.id, decision.action)
            except InvalidAction as error:
                rejected.append(str(error))
                self.stats.invalid_actions += 1
                continue
            seen.add(actor)
            advice.actions.append(action)
            self.stats.valid_actions += 1
        return advice, rejected
