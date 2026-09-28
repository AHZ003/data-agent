"""Public-demo limits and sample workspace."""

from core.demo import SAMPLE_QUESTIONS, DemoLimits, load_sample_workspace


def test_limits_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    limits = DemoLimits.from_env()
    assert limits.question_allowed(10_000) == (True, "")
    assert limits.upload_allowed(10 ** 10) == (True, "")


def test_question_cap_and_own_key_bypass(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_QUESTIONS_PER_SESSION", "3")
    limits = DemoLimits.from_env()
    assert limits.question_allowed(2)[0] is True
    allowed, msg = limits.question_allowed(3)
    assert allowed is False and "own Gemini API key" in msg
    assert limits.question_allowed(3, user_key="AIza-user")[0] is True


def test_upload_cap(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_UPLOAD_MB", "10")
    limits = DemoLimits.from_env()
    assert limits.upload_allowed(10 * 1024 * 1024)[0] is True
    assert limits.upload_allowed(10 * 1024 * 1024 + 1)[0] is False


def test_sample_workspace_is_multi_table_and_read_only():
    db = load_sample_workspace()
    assert len(db.datasource().tables) == 11
    result, err = db.execute_query('SELECT COUNT(*) AS n FROM "Invoice"')
    assert err is None and int(result["n"].iloc[0]) == 412
    assert len(SAMPLE_QUESTIONS) == 5
