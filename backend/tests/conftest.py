from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geoai.agent.service import AnalysisService
from geoai.api.app import create_app
from geoai.config import Settings
from geoai.llm.demo import DemoGateway
from geoai.samples.generate import generate_all
from geoai.storage.assets import AssetService
from geoai.storage.db import Database


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate_all(d / "samples")
    return d


@pytest.fixture
def settings(data_dir, tmp_path):
    return Settings(data_dir=data_dir, env="test", database_url=f"sqlite:///{tmp_path / 't.db'}", rate_limit_per_minute=0)


@pytest.fixture
def service(settings):
    db = Database(settings.db_url)
    demo = DemoGateway()
    return AnalysisService(settings, db, AssetService(settings, db), demo, None, demo)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def png_bytes(w=200, h=150, color=(90, 120, 80), noise=True) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.zeros((h, w, 3), np.uint8) + np.array(color, np.uint8)
    if noise:
        arr = np.clip(arr + rng.integers(-8, 8, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def sample_req(scene: str, domain: str, question: str, **kw):
    from geoai.schemas import AnalysisRequest

    return AnalysisRequest(question=question, domain=domain, asset_id=f"sample:{scene}:after", before_asset_id=f"sample:{scene}:before", **kw)
