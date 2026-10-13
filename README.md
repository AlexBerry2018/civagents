# civ-agents — мультиагентный «правитель» Freeciv на локальной LLM

Мультиагентная система на open-source модели (`qwen3:4b-instruct` через Ollama) играет во Freeciv через [CivRealm](https://github.com/bigai-ai/civrealm): правитель задаёт стратегию, советники (наука, экономика, армия) предлагают действия, исполнитель-код проверяет каждое действие по списку допустимых от игры и только потом отправляет его.

**Полный отчёт:** [`docs/report.md`](docs/report.md) — ТЗ, изоляция, модель, архитектура, промпты, скиллы, память, evals, observability, ограничения.

## Что где

| Пункт задания | Где |
|---|---|
| ТЗ | [`docs/tz.md`](docs/tz.md) |
| Изоляция | `Dockerfile`, `compose.yaml`, отчёт §3 |
| LLM | `src/civ_agents/llm/`, `configs/settings.yaml`, [`docs/benchmarks.md`](docs/benchmarks.md) |
| Архитектура | [`docs/architecture.md`](docs/architecture.md), `src/civ_agents/agents/` |
| Системные промпты | `src/civ_agents/agents/prompts.py` |
| Скиллы | `src/civ_agents/agents/skills.py`, `skills/*.md` |
| База знаний (.md) | `knowledge/*.md` |
| Память | `src/civ_agents/memory/` (эпизодическая, семантическая BM25, рефлексия) |
| Evals | `scripts/evals.py`, `scripts/eval_llm.py`, [`docs/evals_results.md`](docs/evals_results.md), [`docs/evals_llm.md`](docs/evals_llm.md) |
| Observability | `src/civ_agents/observability/`, скриншоты в `docs/img/` |
| Решения (ADR) | `docs/decisions/` |

## Запуск

Требуется: WSL2/Linux, Docker с GPU (NVIDIA), [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run start_freeciv_web_service             # сервер игры (freeciv-web)
docker compose up -d ollama otel-lgtm        # модель + Grafana (http://localhost:3000)
docker compose exec ollama ollama pull qwen3:4b-instruct

# партия на хосте
CIV_OTEL=1 uv run python scripts/play.py --turns 10

# партия в контейнере (нужен Enable host networking в Docker Desktop)
docker compose --profile agent build agent
docker compose --profile agent run --rm agent --turns 10

# evals и тесты
uv run python scripts/evals.py --turns 15 --games 2
uv run python scripts/eval_llm.py
uv run pytest -q
```

Результаты партии — в `runs/<id>/`: `metrics.json`, журнал вызовов модели, действия, события, алерты.

## Ключевые результаты

- Иерархия против одного агента (15 ходов): недопустимых действий 14–21% против 48%, ошибок формата 0–3% против 13%, самый долгий ход 11–15 с против 40 с.
- По счёту на 15 ходах варианты не различаются (2–4 очка, random — 3): горизонт короткий, партий мало.
- Вопросы по правилам: без памяти 25%, с RAG по базе знаний 83%.
