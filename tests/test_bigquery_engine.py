"""BigQuery engine against a fake client: cost guard, introspection, coder loop."""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from core.bigquery_engine import COST_SENTINEL, BigQueryEngine, estimate

GiB = 2**30
TABLE = "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2022"


class FakeClient:
    def __init__(self, bytes_for):
        self.bytes_for = bytes_for      # sql -> bytes the dry run reports
        self.executed = []

    def query(self, sql, job_config):
        if job_config.get("dry_run"):
            return SimpleNamespace(total_bytes_processed=self.bytes_for(sql))
        self.executed.append((sql, job_config))
        df = pd.DataFrame({"n": [42]})
        return SimpleNamespace(result=lambda timeout, max_results: SimpleNamespace(to_dataframe=lambda: df))

    def get_table(self, table_id):
        f = lambda name, t, mode="NULLABLE", d=None: SimpleNamespace(name=name, field_type=t, mode=mode, description=d)  # noqa: E731
        return SimpleNamespace(
            schema=[f("pickup_datetime", "TIMESTAMP"), f("fare_amount", "NUMERIC"),
                    f("payment_type", "STRING", d="1=card, 2=cash")],
            time_partitioning=SimpleNamespace(field="pickup_datetime"),
            clustering_fields=["payment_type"], num_rows=36_000_000,
        )

    def list_rows(self, meta, max_results):
        return [SimpleNamespace(items=lambda: {"pickup_datetime": "2022-01-01 00:10:00",
                                                "fare_amount": 9.5, "payment_type": "1"}.items())]


@pytest.fixture
def engine():
    def bytes_for(sql):
        return 2 * GiB if "WHERE" in sql.upper() else 60 * GiB
    eng = BigQueryEngine([TABLE], client=FakeClient(bytes_for), confirm_bytes=10 * GiB, max_bytes=50 * GiB)
    eng._job_config = lambda **kw: kw
    return eng


def test_estimate_uses_on_demand_price():
    assert estimate(2**40).usd == pytest.approx(6.25)


def test_cheap_query_runs_with_billing_cap(engine):
    df, err = engine.execute_query(f"SELECT COUNT(*) AS n FROM `{TABLE}` WHERE DATE(pickup_datetime) = '2022-01-01'")
    assert err is None and df["n"][0] == 42
    assert engine.client.executed[0][1] == {"maximum_bytes_billed": 50 * GiB}


def test_over_limit_query_is_refused_with_actionable_error(engine):
    df, err = engine.execute_query(f"SELECT fare_amount FROM `{TABLE}`")
    assert df is None and err.startswith("CostGuard") and "partition" in err
    assert engine.client.executed == []


def test_confirmation_threshold(engine):
    engine.max_bytes = 100 * GiB
    df, err = engine.execute_query(f"SELECT fare_amount FROM `{TABLE}`")
    assert err.startswith(COST_SENTINEL)
    df, err = engine.execute_query(f"SELECT fare_amount FROM `{TABLE}`", confirm_cost=True)
    assert err is None


def test_guard_runs_before_any_dry_run(engine):
    df, err = engine.execute_query(f"DELETE FROM `{TABLE}` WHERE TRUE")
    assert "SQLGuardError" in err


def test_introspection_flags_partition_and_clustering(engine):
    ds = engine.datasource()
    t = ds.tables[0]
    assert t.row_count == 36_000_000
    assert "PARTITION COLUMN" in t.column("pickup_datetime").description
    assert "clustering" in t.column("payment_type").description
    assert "1=card" in t.column("payment_type").description
    prompt = ds.to_prompt()
    assert f"Table `{TABLE}`" in prompt and "`fare_amount` NUMERIC" in prompt


def test_coder_repairs_after_cost_guard_and_stops_for_confirmation(engine):
    from agents import coder_agent
    replies = iter([f"SELECT fare_amount FROM `{TABLE}`",
                    f"SELECT COUNT(*) AS n FROM `{TABLE}` WHERE DATE(pickup_datetime) = '2022-01-01'"])
    seen_errors = []

    def fake(q, ds, error_context="", model=None, context=None):
        seen_errors.append(error_context)
        return next(replies)

    with patch.object(coder_agent, "_generate_sql", side_effect=fake):
        df, res = coder_agent.generate_and_execute("trips on new year's day", engine)
    assert res.success and res.attempts == 2
    assert "CostGuard" in seen_errors[1]

    engine.max_bytes = 100 * GiB
    with patch.object(coder_agent, "_generate_sql", return_value=f"SELECT fare_amount FROM `{TABLE}`"):
        df, res = coder_agent.generate_and_execute("all fares", engine)
    assert res.error.startswith(COST_SENTINEL) and res.attempts == 1
    assert coder_agent.is_terminal_error(res.error)
