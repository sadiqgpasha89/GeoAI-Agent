"""Pre-compute agent results for the sample scenes so the web app can run as a static site (GitHub Pages).

Every result is produced by the *real* pipeline (same graph, tools and schemas as the API) running in
demo mode; the static client replays the recorded agent trace. Run:

    geoai-export-static --out ../frontend/public/demo
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from geoai import __version__
from geoai.agent.service import AnalysisService
from geoai.config import Settings
from geoai.llm.demo import DemoGateway
from geoai.samples.generate import generate_all
from geoai.schemas import AnalysisRequest
from geoai.storage.assets import AssetService
from geoai.storage.db import Database

EXTRA_QUESTIONS = {
    "flood": ["Which roads are cut by the flood?", "How much has the flood extent grown since the earlier image?", "Give a bounding box for the water"],
    "wildfire": ["Is the wind pushing the fire towards the town?", "Locate the active fire fronts", "How many buildings are within 500 m of the fire?"],
    "earthquake": ["How many buildings are collapsed?", "Locate the blocked roads"],
    "urban": ["How much green space was lost?", "Locate the new development"],
    "agriculture": ["Which field should I scout first?", "Locate the stressed crop areas"],
}


def export(samples: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_dir=samples.parent, env="test", database_url="sqlite:///:memory:")
    db = Database("sqlite:///:memory:")
    assets = AssetService(settings, db)
    demo = DemoGateway()
    svc = AnalysisService(settings, db, assets, demo, None, demo)
    scenes = []
    for sid in assets.sample_ids():
        meta = assets.sample_meta(sid)
        sdir = out / "scenes" / sid
        sdir.mkdir(parents=True, exist_ok=True)
        for which in ("after", "before"):
            b = assets.resolve(f"sample:{sid}:{which}", "export")
            (sdir / f"{which}.jpg").write_bytes(b.raster.to_png_bytes(fmt="JPEG"))
        shutil.copyfile(samples / sid / "layers.geojson", sdir / "layers.geojson")
        questions = []
        texts = list(meta["suggested_questions"]) + EXTRA_QUESTIONS.get(meta["domain"], [])
        (out / "results" / sid).mkdir(parents=True, exist_ok=True)
        for n, text in enumerate(texts, start=1):
            res = svc.run(AnalysisRequest(question=text, domain=meta["domain"], asset_id=f"sample:{sid}:after",
                                          before_asset_id=f"sample:{sid}:before", mode="demo"), "export", analysis_id=f"static_{sid}_{n:02d}")
            (out / "results" / sid / f"{n:02d}.json").write_text(res.model_dump_json())
            questions.append({"id": f"{sid}-{n:02d}", "text": text, "result": f"results/{sid}/{n:02d}.json"})
        scenes.append({**{k: meta[k] for k in ("id", "domain", "title", "location", "description", "sensor", "before_at", "after_at",
                                               "width", "height", "crs", "gsd_m", "bounds_wgs84")},
                       "images": {"after": f"scenes/{sid}/after.jpg", "before": f"scenes/{sid}/before.jpg"},
                       "layers": f"scenes/{sid}/layers.geojson", "questions": questions})
    manifest = {"version": __version__, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "mode": "static-demo", "scenes": scenes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Export static demo data for the web app")
    ap.add_argument("--samples", type=Path, default=Path("data/samples"))
    ap.add_argument("--out", type=Path, default=Path("../frontend/public/demo"))
    a = ap.parse_args()
    if not (a.samples / "flood-riverbend" / "scene.json").exists():
        generate_all(a.samples)
    m = export(a.samples, a.out)
    n = sum(len(s["questions"]) for s in m["scenes"])
    print(f"Exported {len(m['scenes'])} scenes / {n} results to {a.out}")


if __name__ == "__main__":
    main()
