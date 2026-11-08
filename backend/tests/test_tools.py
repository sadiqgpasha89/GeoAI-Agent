"""Deterministic tools are checked against the generator's known ground truth."""
import pytest

from geoai.llm.demo import DemoGateway
from geoai.schemas import Domain
from geoai.storage.assets import AssetService
from geoai.storage.db import Database
from geoai.tools import build_registry
from geoai.tools.base import ToolContext


@pytest.fixture
def ctx_for(settings):
    assets = AssetService(settings, Database(settings.db_url))

    def make(scene, domain, with_before=True):
        return ToolContext(assets.resolve(f"sample:{scene}:after", "t"), assets.resolve(f"sample:{scene}:before", "t") if with_before else None,
                           DemoGateway(), settings, "q", Domain(domain))

    return make


def run(tool, ctx):
    return build_registry().run(tool, {}, ctx)[0]


def test_flood_exposure_and_critical_facilities(ctx_for):
    out = run("assess_flood", ctx_for("flood-riverbend", "flood"))
    m = {k: v.value for k, v in out.metrics.items()}
    assert m["buildings_total"] == 10 and m["buildings_flooded"] >= 2
    assert m["new_flood_area"] > 4 and m["road_sections_cut"] >= 1 and m["critical_facilities_flooded"] >= 1
    sev = {o.severity for o in out.observations if o.label == "flooded_road_segment"}
    assert "critical" in sev
    assert out.review_reasons  # high-consequence findings require review
    geo = next(o for o in out.observations if o.label == "flooded_building")
    assert geo.geometry["type"] == "Polygon" and geo.pixel_polygon and geo.area_m2 > 0


def test_flood_without_before_asks_for_it_and_reports_no_change(ctx_for):
    out = run("assess_flood", ctx_for("flood-riverbend", "flood", with_before=False))
    assert "new_flood_area" not in out.metrics
    assert any("pre-flood" in a for a in out.next_actions)


def test_earthquake_matches_ground_truth(ctx_for):
    out = run("assess_earthquake", ctx_for("quake-northpoint", "earthquake"))
    by = {}
    for o in out.observations:
        if o.label.endswith("_building"):
            by.setdefault(o.label, set()).add(o.attributes["feature_id"])
    assert by["collapsed_building"] == {"b03", "b04", "b07"}
    assert by["damaged_building"] == {"b02", "b09", "b10", "b12", "b14"}
    assert out.metrics["roads_blocked_sections"].value == 3


def test_wildfire_burn_fire_smoke_and_exposure(ctx_for):
    out = run("assess_wildfire", ctx_for("wildfire-pineridge", "wildfire"))
    m = {k: v.value for k, v in out.metrics.items()}
    assert 90 < m["burned_area"] < 160 and m["active_fire_area"] > 1 and m["smoke_area"] > 1
    assert m["buildings_within_500m"] + m["buildings_within_1km"] + m["buildings_in_scar"] > 10
    assert any(f.kind == "inferred" and "Wind" in f.text for f in out.findings)  # inference is labelled
    assert out.review_reasons


def test_urban_change_zoning_and_access(ctx_for):
    out = run("assess_urban", ctx_for("urban-eastdistrict", "urban"))
    assert out.metrics["new_development_area"].value > 10 and out.metrics["zoning_conflicts"].value == 1
    assert out.metrics["park_access_pct"].value > 50


def test_crop_health_flags_known_problem_fields(ctx_for):
    out = run("assess_crop_health", ctx_for("agri-valley", "agriculture"))
    cls = {o.attributes["feature_id"]: o.label for o in out.observations if o.label.startswith("field_")}
    assert cls["f03"] == "field_severe" and cls["f07"] == "field_stressed" and cls["f06"] == "field_stressed"
    assert cls["f10"] == "field_fallow" and cls["f01"] == "field_healthy"
    assert out.metrics["fields_declining"].value >= 3


def test_crop_visible_only_is_downgraded(ctx_for, data_dir, settings):
    # a plain RGB image has no NIR: the tool must say so and lower its confidence
    from geoai.storage.assets import AssetService
    from geoai.tools.base import ToolContext

    assets = AssetService(settings, Database(settings.db_url))
    info = assets.register_upload((data_dir / "samples/agri-valley/after.png").read_bytes(), "agri.png", "t")
    ctx = ToolContext(assets.resolve(info.id, "t"), None, DemoGateway(), settings, "q", Domain.agriculture)
    out = build_registry().run("assess_crop_health", {}, ctx)[0]
    assert any("No NIR band" in l for l in out.limitations) and out.confidence < 0.7
    assert any("not georeferenced" in a for a in out.assumptions)


def test_registry_rejects_unknown_tool_and_bad_args(ctx_for):
    reg = build_registry()
    with pytest.raises(Exception, match="Unknown tool"):
        reg.validate_call("rm_rf", {})
    with pytest.raises(Exception, match="Invalid arguments"):
        reg.validate_call("ground_object", {})
