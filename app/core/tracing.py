"""OpenTelemetry setup. No-ops cleanly when no OTLP endpoint is configured."""

from __future__ import annotations

import functools
from collections.abc import Callable

from opentelemetry import trace

_tracer = trace.get_tracer("documind")


def setup_tracing(app, otlp_endpoint: str | None) -> None:
    if not otlp_endpoint:
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "documind-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


def traced(span_name: str) -> Callable:
    """Wrap an agent node in a span so each LangGraph step shows up in traces."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _tracer.start_as_current_span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return deco
