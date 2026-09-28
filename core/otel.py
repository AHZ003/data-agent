"""Optional OpenTelemetry export of DataAgent traces (Langfuse, Jaeger, ...).

Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set (and the `otel` extra is
installed). Then every tracing.span is mirrored as an OTel span under a
per-run root span, with GenAI semantic-convention attributes:

    gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
    dataagent.cost_usd, dataagent.<metadata key>

Langfuse ingests OTLP directly:
    OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
    OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(public_key:secret_key)>"

The JSONL trace file stays the source of truth; OTel is a mirror, so a
collector outage never breaks a request.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

_lock = threading.Lock()
_tracer = None
_initialized = False


def tracer():
    """The OTel tracer, or None when export is not configured."""
    global _tracer, _initialized
    if _initialized:
        return _tracer
    with _lock:
        if _initialized:
            return _tracer
        _initialized = True
        if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            return None
        provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "dataagent")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("dataagent")
        return _tracer


def use_tracer(t) -> None:
    """Install a tracer directly (tests, custom exporters)."""
    global _tracer, _initialized
    with _lock:
        _tracer, _initialized = t, True


def attrs(model: Optional[str], tokens_in: int, tokens_out: int, cost: float, meta: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"gen_ai.usage.input_tokens": tokens_in, "gen_ai.usage.output_tokens": tokens_out,
                           "dataagent.cost_usd": cost}
    if model:
        out["gen_ai.request.model"] = model
    for k, v in meta.items():
        out[f"dataagent.{k}"] = v if isinstance(v, (bool, int, float)) else str(v)[:500]
    return out
