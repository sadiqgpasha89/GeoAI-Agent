import json

import pytest
from fastapi.testclient import TestClient

from geoai.api.app import create_app
from geoai.config import Settings
from tests.conftest import png_bytes

Q = {"question": "How many buildings are flooded?", "domain": "flood", "asset_id": "sample:flood-riverbend:after", "before_asset_id": "sample:flood-riverbend:before"}


def test_health_meta_samples(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    meta = client.get("/api/v1/meta").json()
    assert meta["default_mode"] == "demo" and not meta["live_available"] and {t["name"] for t in meta["tools"]} >= {"assess_flood", "ground_object"}
    s = client.get("/api/v1/samples").json()
    assert {x["domain"] for x in s} == {"flood", "wildfire", "earthquake", "urban", "agriculture"}
    assert client.headers.get("x-request-id") is None and client.get("/healthz").headers["x-request-id"]


def test_analysis_roundtrip_exports_and_review(client):
    r = client.post("/api/v1/analyses", json=Q)
    assert r.status_code == 200
    res = r.json()
    aid = res["id"]
    assert res["needs_review"] and res["review_status"] == "pending"
    assert client.get("/api/v1/analyses?needs_review=true").json()[0]["id"] == aid
    gj = client.get(f"/api/v1/analyses/{aid}/export?format=geojson").json()
    assert gj["type"] == "FeatureCollection" and any(f["geometry"] for f in gj["features"])
    assert "id,label,kind" in client.get(f"/api/v1/analyses/{aid}/export?format=csv").text
    html = client.get(f"/api/v1/analyses/{aid}/export?format=html")
    assert "Not an official" in html.text or "not an official" in html.text.lower()
    assert "default-src 'none'" in html.headers["content-security-policy"]
    assert client.get(f"/api/v1/analyses/{aid}/export?format=xml").status_code == 422
    rv = client.post(f"/api/v1/analyses/{aid}/review", json={"decision": "corrected", "reviewer": "sam", "comment": "b05 is dry", "corrections": [{"id": "x"}]})
    assert rv.json()["review_status"] == "corrected"
    assert client.get(f"/api/v1/analyses/{aid}").json()["review_status"] == "corrected"
    assert client.get(f"/api/v1/analyses/{aid}/reviews").json()[0]["reviewer"] == "sam"
    assert client.post("/api/v1/analyses/nope/review", json={"decision": "approved"}).status_code == 404


def test_validation_and_errors(client):
    assert client.post("/api/v1/analyses", json={**Q, "question": "x"}).status_code == 422
    assert client.post("/api/v1/analyses", json={**Q, "extra": 1}).status_code == 422
    assert client.post("/api/v1/analyses", json={**Q, "asset_id": "sample:nope:after"}).status_code == 404
    assert client.post("/api/v1/analyses", json={**Q, "asset_id": "sample:../../etc:after"}).status_code == 404
    live = client.post("/api/v1/analyses", json={**Q, "mode": "live"})
    assert live.status_code == 400 and "model provider" in live.json()["detail"]
    assert client.get("/api/v1/assets/sample:flood-riverbend:after/preview?max_side=256").headers["content-type"] == "image/png"


def test_upload_validation(client):
    ok = client.post("/api/v1/assets", files={"file": ("../../evil name.png", png_bytes(), "image/png")})
    assert ok.status_code == 201
    info = ok.json()
    assert info["georeferenced"] is False and info["filename"] == "evil name.png" and info["width"] == 200
    assert client.post("/api/v1/assets", files={"file": ("a.png", b"GIF89a....", "image/png")}).status_code == 415  # magic bytes, not header
    assert client.post("/api/v1/assets", files={"file": ("a.png", b"", "image/png")}).status_code == 400
    assert client.post("/api/v1/assets", files={"file": ("a.png", png_bytes(16, 16), "image/png")}).status_code == 422
    assert client.post("/api/v1/assets", files={"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"x" * 50, "image/png")}).status_code == 422
    r = client.post("/api/v1/analyses", json={"question": "Describe this scene in detail", "domain": "flood", "asset_id": info["id"]})
    assert r.status_code == 200 and r.json()["sources"][0]["georeferenced"] is False


def test_upload_size_and_pixel_limits(data_dir, tmp_path):
    s = Settings(data_dir=data_dir, env="test", database_url=f"sqlite:///{tmp_path/'x.db'}", max_upload_mb=1, max_image_pixels=10_000, rate_limit_per_minute=0)
    with TestClient(create_app(s)) as c:
        assert c.post("/api/v1/assets", files={"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * (1_100_000), "image/png")}).status_code == 413
        assert c.post("/api/v1/assets", files={"file": ("a.png", png_bytes(200, 150), "image/png")}).status_code == 422  # 30k px > 10k


def test_geotiff_upload_keeps_georeferencing(client, data_dir):
    tif = (data_dir / "samples/agri-valley/after.tif").read_bytes()
    info = client.post("/api/v1/assets", files={"file": ("f.tif", tif, "image/tiff")}).json()
    assert info["georeferenced"] and info["crs"] == "EPSG:32611" and "nir" in info["bands"] and info["bounds_wgs84"]
    r = client.post("/api/v1/analyses", json={"question": "Which fields show crop stress?", "domain": "agriculture", "asset_id": info["id"]}).json()
    assert r["metrics"]["stressed_area"]["unit"] == "ha"  # georeferenced -> real units
    assert any("field-boundary" in l for l in r["limitations"])  # no layer for uploads -> honest limitation


def test_sse_stream(client):
    with client.stream("POST", "/api/v1/analyses/stream", json=Q) as st:
        assert st.headers["content-type"].startswith("text/event-stream")
        text = "".join(st.iter_text())
    events = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "start" and events[-1] == "result" and "tool_start" in events
    last = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")][-1]
    assert last["result"]["observations"]
    saved = client.get(f"/api/v1/analyses/{last['result']['id']}")
    assert saved.status_code == 200


def test_auth_and_tenant_isolation(data_dir, tmp_path):
    s = Settings(data_dir=data_dir, env="test", database_url=f"sqlite:///{tmp_path/'a.db'}", api_keys="k-alpha:alpha,k-beta:beta", rate_limit_per_minute=0)
    with TestClient(create_app(s)) as c:
        assert c.get("/api/v1/meta").status_code == 401
        assert c.get("/api/v1/meta", headers={"X-API-Key": "wrong"}).status_code == 401
        a, b = {"X-API-Key": "k-alpha"}, {"X-API-Key": "k-beta"}
        asset = c.post("/api/v1/assets", files={"file": ("a.png", png_bytes(), "image/png")}, headers=a).json()["id"]
        assert c.get(f"/api/v1/assets/{asset}", headers=a).status_code == 200
        assert c.get(f"/api/v1/assets/{asset}", headers=b).status_code == 404            # other tenant cannot see it
        assert c.post("/api/v1/analyses", json={**Q, "asset_id": asset}, headers=b).status_code == 404
        res = c.post("/api/v1/analyses", json=Q, headers=a).json()
        assert c.get(f"/api/v1/analyses/{res['id']}", headers=b).status_code == 404
        assert c.get("/api/v1/analyses", headers=b).json() == []
        assert c.post(f"/api/v1/analyses/{res['id']}/review", json={"decision": "approved"}, headers=b).status_code == 404


def test_rate_limit(data_dir, tmp_path):
    s = Settings(data_dir=data_dir, env="test", database_url=f"sqlite:///{tmp_path/'r.db'}", rate_limit_per_minute=3)
    with TestClient(create_app(s)) as c:
        codes = [c.get("/api/v1/meta").status_code for _ in range(5)]
        assert codes[:3] == [200] * 3 and codes[3] == 429
        assert c.get("/api/v1/meta").headers["retry-after"]


def test_production_requires_api_keys(data_dir, tmp_path):
    with pytest.raises(RuntimeError, match="API_KEYS"):
        create_app(Settings(data_dir=data_dir, env="production", database_url=f"sqlite:///{tmp_path/'p.db'}"))


def test_csv_export_neutralises_formulas():
    from geoai.exports import _cell

    assert _cell("=HYPERLINK(...)").startswith("'") and _cell("safe") == "safe"


def test_completed_analysis_is_immutable(service):
    from tests.conftest import sample_req

    res = service.run(sample_req("flood-riverbend", "flood", "How many buildings are flooded?"), "t")
    res.answer = "tampered"
    with pytest.raises(ValueError, match="immutable"):
        service.db.save_analysis(res, "t")
    assert service.db.get_analysis(res.id, "t").answer != "tampered"
