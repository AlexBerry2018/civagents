"""Семантическая память: поиск по базе знаний (.md) — BM25 без внешних зависимостей.

Документы режутся на фрагменты по заголовкам. Запрос строится из роли и ситуации
на ходу, в промпт попадают несколько лучших фрагментов в пределах лимита символов.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-zA-Zа-яА-Я]{3,}")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


@dataclass
class Chunk:
    source: str
    title: str
    text: str


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(c.title + " " + c.text)) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths) if chunks else 0.0
        df = Counter(term for doc in self._docs for term in doc)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_dir(cls, directory: str | Path) -> KnowledgeBase:
        chunks: list[Chunk] = []
        for path in sorted(Path(directory).glob("*.md")):
            title, lines = path.stem, []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    if "".join(lines).strip():
                        chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
                    title, lines = line.lstrip("# ").strip(), []
                else:
                    lines.append(line)
            if "".join(lines).strip():
                chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
        return cls(chunks)

    def search(self, query: str, k: int = 2) -> list[Chunk]:
        terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg)
                    score += self._idf[term] * tf * (self.k1 + 1) / norm
            scores.append((score, i))
        best = sorted((s for s in scores if s[0] > 0), reverse=True)[:k]
        return [self.chunks[i] for _, i in best]

    def notes(self, query: str, k: int = 2, max_chars: int = 900) -> str:
        parts, used = [], 0
        for chunk in self.search(query, k):
            text = f"[{chunk.title}] {chunk.text}"
            if used + len(text) > max_chars:
                text = text[: max(0, max_chars - used)]
            parts.append(text)
            used += len(text)
            if used >= max_chars:
                break
        return "\n".join(parts)


ROLE_QUERIES = {
    "economist": "settlers found city site terrain food workers improve irrigation city build production",
    "military": "explorer explore hut defense warriors fortify threat enemy",
    "science": "research technology order bronze alphabet monarchy",
    "ruler": "strategy expansion rates science tax government monarchy lessons",
    "single": "strategy settlers city research explore lessons",
}


def make_knowledge(kb: KnowledgeBase, k: int = 2, max_chars: int = 900):
    """Функция для Advisor: (роль, ход) → заметки из базы знаний."""

    def knowledge(role: str, view) -> str:
        query = ROLE_QUERIES.get(role, role)
        if any(c.unhappy or c.angry for c in view.state.cities):
            query += " unhappy luxury disorder"
        if not view.state.cities:
            query += " first city capital"
        return kb.notes(query, k, max_chars)

    return knowledge
