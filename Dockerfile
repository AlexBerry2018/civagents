# Контейнер агентов: Python 3.10 (требование CivRealm), без GPU, непривилегированный пользователь.
FROM python:3.10-slim

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Сначала только зависимости — этот слой кэшируется, пока не меняется uv.lock.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# Затем код проекта.
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY knowledge ./knowledge
COPY skills ./skills
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 agent && chown -R agent:agent /app
USER agent

# Запуск без uv: окружение уже собрано, ничего не скачивается и не пишется в .venv.
ENTRYPOINT ["/app/.venv/bin/python", "scripts/play.py"]
CMD ["--turns", "10"]
