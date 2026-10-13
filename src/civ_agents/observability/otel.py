"""OpenTelemetry: трейсы ходов и агентов, метрики, логи → OTLP (grafana/otel-lgtm).

Включается переменной CIV_OTEL=1. Без неё (или без пакетов opentelemetry) всё работает
как no-op: код агентов не зависит от того, включена ли телеметрия.

Структура трейса одного хода:
    turn 7
    ├── agent economist
    │   └── llm.chat          (модель, токены, время, ошибка)
    └── agent military
        └── llm.chat
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from civ_agents.llm.base import CallRecord
from civ_agents.observability.context import current_context

logger = logging.getLogger(__name__)

_tracer = None
_instruments: dict[str, Any] = {}
_gauges: dict[str, float] = {}


def enabled() -> bool:
    return _tracer is not None


def setup(service_name: str = "civ-agents", run_id: str = "") -> bool:
    """Настроить экспорт в OTLP (по умолчанию http://localhost:4318). True — если включено."""
    global _tracer
    if _tracer is not None:
        return True  # уже настроено (evals играют несколько партий в одном процессе)
    if os.environ.get("CIV_OTEL") != "1":
        return False
    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("CIV_OTEL=1, но пакеты opentelemetry не установлены — телеметрия выключена")
        return False

    resource = Resource.create({"service.name": service_name, "run.id": run_id})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("civ_agents")

    reader = PeriodicExportingMetricReader(OTLPMetricExporter(), export_interval_millis=5000)
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
    meter = metrics.get_meter("civ_agents")
    _instruments.update(
        llm_calls=meter.create_counter("civ_llm_calls", description="Вызовы модели"),
        llm_latency=meter.create_histogram("civ_llm_latency", unit="s", description="Время вызова модели"),
        llm_tokens=meter.create_counter("civ_llm_tokens", description="Токены"),
        turn_duration=meter.create_histogram("civ_turn_duration", unit="s", description="Время хода"),
        actions=meter.create_counter("civ_actions", description="Действия агентов"),
        alerts=meter.create_counter("civ_alerts", description="Сработавшие алерты"),
    )
    meter.create_observable_gauge(
        "civ_game_state", callbacks=[_observe_game_state], description="Города, юниты, счёт, ход"
    )

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    logging.getLogger("civ_agents").addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))

    logger.info("OpenTelemetry включён: трейсы, метрики и логи → OTLP")
    return True


def _observe_game_state(options):
    from opentelemetry.metrics import Observation

    return [Observation(value, {"metric": name}) for name, value in _gauges.items()]


def set_game_state(**values: float) -> None:
    """Текущие показатели партии для gauge (города, юниты, счёт, ход)."""
    _gauges.update(values)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Спан трейса; без OTel — ничего не делает."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name, attributes={k: v for k, v in attributes.items() if v is not None}):
        yield


def record_turn(seconds: float, actions_sent: int, actions_failed: int) -> None:
    if not _instruments:
        return
    _instruments["turn_duration"].record(seconds)
    _instruments["actions"].add(actions_sent, {"accepted": True})
    _instruments["actions"].add(actions_failed, {"accepted": False})


def record_alert(name: str) -> None:
    if _instruments:
        _instruments["alerts"].add(1, {"alert": name})


def llm_observer(record: CallRecord) -> None:
    """Наблюдатель клиента модели: спан llm.chat внутри текущего спана агента + метрики."""
    if _tracer is None:
        return
    context = current_context()
    agent = str(context.get("agent", "none"))
    error_type = type(record.error).__name__ if record.error else ""
    response = record.response or getattr(record.error, "response", None)

    start_ns = time.time_ns() - int(record.latency_s * 1e9)
    llm_span = _tracer.start_span("llm.chat", start_time=start_ns)
    llm_span.set_attribute("llm.model", record.model)
    llm_span.set_attribute("agent", agent)
    llm_span.set_attribute("llm.attempts", record.attempts)
    if response:
        llm_span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
        llm_span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
        llm_span.set_attribute("llm.cached_tokens", response.usage.cached_tokens)
    if error_type:
        llm_span.set_attribute("error.type", error_type)
    llm_span.end()

    labels = {"agent": agent, "ok": record.ok, "error_type": error_type}
    _instruments["llm_calls"].add(1, labels)
    _instruments["llm_latency"].record(record.latency_s, {"agent": agent})
    if response:
        _instruments["llm_tokens"].add(response.usage.prompt_tokens, {"kind": "prompt", "agent": agent})
        _instruments["llm_tokens"].add(response.usage.completion_tokens, {"kind": "completion", "agent": agent})


def chain(*observers):
    """Несколько наблюдателей как один (журнал + OTel)."""

    def observe(record: CallRecord) -> None:
        for observer in observers:
            if observer is not None:
                observer(record)

    return observe
