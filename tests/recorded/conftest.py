"""Shared fixtures for recorded-HTTP tests.

The `respx_cassette` fixture loads a YAML cassette from
`tests/recorded/cassettes/<name>.yaml` into a `respx.MockRouter`,
yields the router so the test can add ad-hoc routes, and finally
asserts every cassette-defined route was actually called (so dead
entries surface as test failures rather than rotting in place).

Cassette schema (YAML, list at the top level):

```yaml
- method: GET
  url: https://huggingface.co/api/datasets
  params: {filter: text-classification, limit: 50}    # optional
  status: 200
  json: {data: [...]}                                 # one of:
  # text: "raw body"
  # bytes_b64: "..."
  headers: {content-type: application/json}           # optional
```

Usage:

```python
@pytest.mark.parametrize("respx_cassette", ["hf_search_basic"], indirect=True)
def test_search_basic(respx_cassette):
    r = httpx.get("https://huggingface.co/api/datasets",
                  params={"filter": "text-classification", "limit": 50})
    assert r.status_code == 200
```

Tests can override the `cassette_dir` fixture to point at a different
directory (used by the helper's self-test).
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
import yaml

CASSETTE_DIR = Path(__file__).parent / "cassettes"


@pytest.fixture
def cassette_dir() -> Path:
    """Directory holding cassette YAML files. Override per-test if needed."""
    return CASSETTE_DIR


def _load_cassette(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise ValueError(
            f"Cassette {path} must be a YAML list of records, got {type(data).__name__}."
        )
    return data


def _apply_record(router: respx.MockRouter, record: dict[str, Any]) -> None:
    method = str(record.get("method", "GET")).upper()
    url = record["url"]
    params = record.get("params")
    status = int(record.get("status", 200))
    headers = record.get("headers")

    route = router.request(method, url, params=params)

    if "json" in record:
        route.respond(status_code=status, json=record["json"], headers=headers)
    elif "text" in record:
        route.respond(status_code=status, text=record["text"], headers=headers)
    elif "bytes_b64" in record:
        body = base64.b64decode(record["bytes_b64"])
        route.respond(status_code=status, content=body, headers=headers)
    else:
        route.respond(status_code=status, headers=headers)


@pytest.fixture
def respx_cassette(
    request: pytest.FixtureRequest,
    cassette_dir: Path,
) -> Iterator[respx.MockRouter]:
    name = getattr(request, "param", None)
    if not isinstance(name, str) or not name:
        raise RuntimeError(
            "respx_cassette must be parametrized indirectly with a cassette name, "
            'e.g. @pytest.mark.parametrize("respx_cassette", ["hf_search"], indirect=True)'
        )

    path = cassette_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Cassette not found: {path}")

    records = _load_cassette(path)

    with respx.mock(assert_all_called=True) as router:
        for record in records:
            _apply_record(router, record)
        yield router
