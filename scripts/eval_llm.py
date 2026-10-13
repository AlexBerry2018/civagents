"""Evals уровня LLM: вопросы по правилам Freeciv — без памяти и с семантической памятью (RAG).

    uv run python scripts/eval_llm.py
"""

import json
from pathlib import Path

from civ_agents.config import load_settings
from civ_agents.llm.openai_compat import OpenAICompatClient
from civ_agents.llm.types import Message
from civ_agents.memory.semantic import KnowledgeBase

# Вопрос → слова, хотя бы одно из которых должно быть в ответе (проверка кодом, без LLM-судьи).
QUESTIONS = [
    ("Which unit founds new cities in Freeciv?", ["settler"]),
    ("How far apart should new cities be kept, in tiles?", ["3", "three"]),
    ("Which terrain gives 2 food: grassland, mountains or desert?", ["grassland"]),
    ("Which terrain gives 2 shields: forest or grassland?", ["forest"]),
    ("What does a river on a tile add?", ["trade"]),
    ("What happens if you re-select the technology you are already researching?", ["lose", "lost", "reset", "bulbs"]),
    ("Which government does the game start in?", ["despotism"]),
    ("Which technologies are required for Monarchy?", ["ceremonial", "code of laws"]),
    ("What happens to research during Anarchy?", ["stop", "no research", "halt"]),
    ("What can a Minor Tribe Village (hut) give?", ["gold", "technology", "unit"]),
    ("Which unit is the cheapest early defender?", ["warrior"]),
    ("What causes civil disorder in a city?", ["unhappy"]),
]

SYSTEM = "Answer the Freeciv question in one short sentence."


def main() -> None:
    settings = load_settings()
    llm = OpenAICompatClient(settings.llm.active)
    kb = KnowledgeBase.from_dir("knowledge")
    rows, scores = [], {"no_memory": 0, "rag": 0}
    for question, expected in QUESTIONS:
        answers = {}
        for variant in scores:
            system = SYSTEM
            if variant == "rag":
                system += "\nReference notes:\n" + kb.notes(question, k=2, max_chars=700)
            text = llm.chat([Message.system(system), Message.user(question)]).text
            correct = any(word in text.lower() for word in expected)
            scores[variant] += correct
            answers[variant] = {"answer": text, "correct": correct}
        rows.append({"question": question, **answers})
        print(f"{'+' if answers['no_memory']['correct'] else '-'}{'+' if answers['rag']['correct'] else '-'} {question}")

    n = len(QUESTIONS)
    lines = ["# Evals уровня LLM: вопросы по правилам\n",
             f"Модель: {settings.llm.active.model}. Вопросов: {n}. Проверка — по ключевым словам.\n",
             "| Вариант | Верных | Доля |", "|---|---|---|",
             f"| Без памяти | {scores['no_memory']} | {scores['no_memory'] / n:.0%} |",
             f"| С семантической памятью (BM25 по knowledge/) | {scores['rag']} | {scores['rag'] / n:.0%} |"]
    Path("docs/evals_llm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path("runs/eval_llm_raw.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
