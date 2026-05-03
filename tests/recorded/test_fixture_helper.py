"""Self-test for the `respx_cassette` fixture.

Uses a per-test override of the `cassette_dir` fixture (a clean pytest
pattern, no monkey-patching) to point the helper at a temp directory
holding a synthetic cassette. This keeps the harness honest before any
real HuggingFace cassettes land.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

pytestmark = pytest.mark.recorded


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    cassette = [
        {
            "method": "GET",
            "url": "https://example.invalid/api/ping",
            "params": {"q": "hello"},
            "status": 200,
            "json": {"pong": True, "echo": "hello"},
        }
    ]
    (tmp_path / "selftest.yaml").write_text(
        yaml.safe_dump(cassette, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize("respx_cassette", ["selftest"], indirect=True)
def test_respx_cassette_serves_recorded_response(respx_cassette):
    assert respx_cassette is not None
    r = httpx.get("https://example.invalid/api/ping", params={"q": "hello"})
    assert r.status_code == 200
    assert r.json() == {"pong": True, "echo": "hello"}
