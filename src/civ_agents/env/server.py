"""Готовность игрового сервера freeciv-web.

CivRealm опрашивает сервер прямо при импорте своих модулей и падает, если тот
ещё не поднялся (после старта контейнера это занимает до нескольких минут).
Поэтому перед импортом CivRealm вызываем wait_for_server().
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable

logger = logging.getLogger(__name__)

STATUS_PATH = "/pubstatus"  # тот же адрес, который опрашивает CivRealm


class ServerNotReady(RuntimeError):
    """Сервер игры не стал доступен за отведённое время."""


def fetch_status(url: str, timeout_s: float = 5.0) -> int | None:
    """HTTP-код ответа или None, если ответа нет вовсе (сервер не слушает, таймаут)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 — адрес из настроек
            return response.status
    except urllib.error.HTTPError as error:
        return error.code  # 502 и т. п.: nginx в контейнере жив, сервис за ним — ещё нет
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def wait_for_server(
    base_url: str = "http://localhost:8080",
    *,
    timeout_s: float = 300.0,
    interval_s: float = 5.0,
    fetch: Callable[[str], int | None] = fetch_status,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """Ждать, пока сервер ответит 200. Возвращает время ожидания в секундах.

    fetch, sleep и clock подменяются в тестах.
    """
    url = base_url.rstrip("/") + STATUS_PATH
    started = clock()
    last_status: object = object()  # заведомо не равно ни одному ответу
    while True:
        status = fetch(url)
        if status == 200:
            waited = clock() - started
            if waited > 0:
                logger.info("Сервер игры готов через %.0f с", waited)
            return waited

        if status != last_status:
            reason = "нет ответа" if status is None else f"HTTP {status}"
            logger.info("Сервер игры не готов (%s), ждём…", reason)
            last_status = status

        if clock() - started + interval_s > timeout_s:
            reason = "нет ответа" if status is None else f"HTTP {status}"
            raise ServerNotReady(
                f"Сервер игры {base_url} не готов за {timeout_s:.0f} с (последний ответ: {reason}). "
                "Проверьте контейнер: docker ps, docker logs freeciv-web"
            )
        sleep(interval_s)