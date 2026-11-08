from geoai.config import Settings


def test_cors_origins_accepts_csv_and_json(monkeypatch):
    monkeypatch.setenv("GEOAI_CORS_ORIGINS", "https://a.example, https://b.example")
    assert Settings().cors_origins == ["https://a.example", "https://b.example"]
    monkeypatch.setenv("GEOAI_CORS_ORIGINS", '["https://c.example"]')
    assert Settings().cors_origins == ["https://c.example"]


def test_provider_resolution(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEOAI_ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert Settings(_env_file=None).resolved_provider == "demo"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    assert s.resolved_provider == "anthropic" and "sk-test" not in repr(s)  # secret is masked
    assert Settings(_env_file=None, provider="demo").resolved_provider == "demo"
    assert Settings(_env_file=None, api_keys="k1:a,k2").tenants == {"k1": "a", "k2": "default"}
