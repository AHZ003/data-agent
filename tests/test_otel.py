"""OpenTelemetry mirror of tracing spans."""

import pytest

from core import otel, tracing


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setenv("DATAAGENT_TRACE_PATH", str(tmp_path / "t.jsonl"))
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    otel.use_tracer(provider.get_tracer("test"))
    yield exp
    otel.use_tracer(None)


class _Resp:
    class usage_metadata:
        prompt_token_count, candidates_token_count, thoughts_token_count = 100, 20, 0


def test_spans_mirror_to_otel_with_genai_attributes(exporter):
    tracing.new_run("q", dataset="chinook")
    with tracing.span("coder", model="gemini-2.5-flash") as s:
        tracing.record_usage(_Resp(), "gemini-2.5-flash")
        s.add_metadata(sql="SELECT 1", success=True)
    tracing.end_run()
    spans = {sp.name: sp for sp in exporter.get_finished_spans()}
    coder, root = spans["coder"], spans["dataagent.run"]
    assert coder.parent.span_id == root.context.span_id
    assert coder.attributes["gen_ai.usage.input_tokens"] == 100
    assert coder.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert coder.attributes["dataagent.sql"] == "SELECT 1"
    assert root.attributes["dataagent.dataset"] == "chinook"


def test_disabled_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    otel._initialized, otel._tracer = False, None
    assert otel.tracer() is None
