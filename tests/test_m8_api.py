"""M8: service. All request texts here are invented, not customer text.

The tests use models/router.joblib, the file the service loads. It must exist: run the train step.
"""
import io
import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from types import SimpleNamespace

import pandas as pd
import pytest

KEYS = {"team", "confidence", "top3", "needs_triage", "reasons", "clarifying_question", "model_version"}
MARK = "zqxjk"  # an invented word that no fixed message contains
UNKNOWN_PRODUCT = "product not recognised; routed without the product"
VAGUE = "please call me about my purifier"
PAID = "I paid for installation but technician never came"
RECORDS = [
    {"request_text": "My mixer grinder stopped working and makes a burning smell"},
    {"request_text": PAID, "channel": "email"},
    {"request_text": VAGUE, "product_family": "Water Purifier"},
    {"request_text": "heater is not heating", "channel": "ivr", "warranty_status": "in_warranty"},
    {
        "request_text": "I was charged twice for the same order",
        "channel": "whatsapp",
        "product_family": "Air Fryer",
        "warranty_status": "out_of_warranty",
    },
]


@pytest.fixture(scope="module")
def main():
    from app import main

    return main


@pytest.fixture(scope="module")
def test_client():
    from fastapi.testclient import TestClient

    return TestClient


@pytest.fixture(scope="module")
def router(config):
    from src.router import Router

    if not config.ROUTER_JOBLIB.exists():
        pytest.fail("models/router.joblib is missing: run the train step")
    return Router.load(config.ROUTER_JOBLIB)


@pytest.fixture(scope="module")
def client(main, test_client, router):
    with test_client(main.create_app()) as c:
        yield c


@pytest.fixture
def model_copy(config, router, tmp_path):
    for path in (config.ROUTER_JOBLIB, config.ROUTER_META_JSON):
        shutil.copy(path, tmp_path / path.name)
    return tmp_path


def edit_meta(folder, change):
    path = folder / "router_meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    change(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")


# --- Criterion 1: health ---------------------------------------------------------------------------
def test_health_loaded(client, router):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": router.meta["model_version"],
        "training_window": "2025-04-01 to 2026-06-30",
    }


# --- Criterion 2: same answer as the router -----------------------------------------------------------
@pytest.mark.parametrize("record", RECORDS)
def test_predict_equals_router(client, router, record):
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == KEYS
    assert body == router.predict([record])[0]
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_optional_fields_mean_not_given(client, router, blank):
    record = {"request_text": "heater is not heating"}
    body = {**record, "channel": blank, "product_family": blank, "warranty_status": blank}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert response.json() == router.predict([record])[0]


def test_extra_keys_are_ignored(client, router):
    record = {"request_text": "heater is not heating"}
    response = client.post("/predict", json={**record, "team_label": "Billing"})
    assert response.json() == router.predict([record])[0]


def test_model_is_loaded_once(main, test_client, monkeypatch):
    calls = []
    load = main.load_final
    monkeypatch.setattr(main, "load_final", lambda *a, **k: calls.append(1) or load(*a, **k))
    with test_client(main.create_app()) as c:
        for record in RECORDS:
            c.post("/predict", json=record)
    assert calls == [1]


# --- Unknown product: accepted, with a reason -------------------------------------------------------------
@pytest.mark.parametrize("product", [f"{MARK} toaster", "water purifier"])
def test_unknown_product_is_lenient(client, router, product):
    record = {"request_text": "it stopped working yesterday", "channel": "chat", "product_family": product}
    response = client.post("/predict", json=record)
    assert response.status_code == 200
    body, expected = response.json(), router.predict([record])[0]
    assert body["reasons"] == [*expected["reasons"], UNKNOWN_PRODUCT]
    assert {**body, "reasons": None} == {**expected, "reasons": None}
    # An unknown product counts as no product: same team and probabilities as leaving it out.
    without = router.predict([{"request_text": record["request_text"], "channel": "chat"}])[0]
    assert (body["team"], body["top3"]) == (without["team"], without["top3"])
    assert MARK not in response.text


def test_known_or_missing_product_has_no_such_reason(client, config):
    for product in (*config.PRODUCT_FAMILIES, None):
        body = client.post("/predict", json={"request_text": "it stopped working", "product_family": product}).json()
        assert UNKNOWN_PRODUCT not in body["reasons"]


# --- Criterion 3: validation -------------------------------------------------------------------------------
BAD = {
    "missing": ({"channel": "chat", "note": MARK}, "request_text is required"),
    "null": ({"request_text": None}, "request_text"),
    "empty": ({"request_text": ""}, "must not be empty"),
    "whitespace": ({"request_text": " \t\n "}, "must not be empty"),
    "too long": ({"request_text": MARK * 401}, "at most 2,000 characters"),
    "channel": ({"request_text": f"{MARK} fan", "channel": f"{MARK}fax"}, "channel must be one of: chat, email"),
    "warranty": ({"request_text": f"{MARK} fan", "warranty_status": f"{MARK}x"}, "warranty_status must be one of:"),
    "number": ({"request_text": 48213377}, "request_text must be text"),
    "list": ({"request_text": [MARK]}, "request_text must be text"),
    "channel type": ({"request_text": f"{MARK} fan", "channel": 48213377}, "channel must be one of:"),
    "product type": ({"request_text": f"{MARK} fan", "product_family": [MARK]}, "product_family must be text"),
    "not an object": ([MARK], "JSON"),
    "string body": (MARK, "JSON"),
}


@pytest.mark.parametrize("name", BAD)
def test_422_readable_no_echo(client, name):
    body, message = BAD[name]
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert message in response.json()["detail"]
    assert MARK not in response.text and "48213377" not in response.text
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "content, kind",
    [
        (f'{{"request_text": "{MARK}', "application/json"),
        ("", "application/json"),
        (f"request_text={MARK}", "application/x-www-form-urlencoded"),
        (MARK, "text/plain"),
        (f'{{"request_text": "{MARK} \\ud800"}}', "application/json"),
    ],
)
def test_unreadable_bodies(client, content, kind):
    response = client.post("/predict", content=content, headers={"content-type": kind})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str) and len(response.json()["detail"]) > 10
    assert MARK not in response.text


def test_all_problems_are_listed(client):
    detail = client.post("/predict", json={"request_text": " ", "channel": MARK, "warranty_status": MARK}).json()["detail"]
    for part in ("request_text", "channel", "warranty_status"):
        assert part in detail


def test_length_limit_is_exact(client):
    assert client.post("/predict", json={"request_text": "a" * 2000}).status_code == 200
    assert client.post("/predict", json={"request_text": "a" * 2001}).status_code == 422
    assert client.post("/predict", json={"request_text": "fan" + " " * 1998}).status_code == 422  # before trimming


@pytest.mark.parametrize("text", ["???", "​​"])
def test_text_without_words_is_triaged(client, text):
    response = client.post("/predict", json={"request_text": text})
    assert response.status_code == 200
    assert response.json()["needs_triage"] is True


# --- Criterion 4: no usable model ----------------------------------------------------------------------------
def assert_unavailable(main, test_client, folder):
    with test_client(main.create_app(folder)) as c:
        assert c.get("/health").json() == {
            "status": "ok",
            "model_loaded": False,
            "model_version": None,
            "training_window": None,
        }
        response = c.post("/predict", json={"request_text": f"{MARK} fan"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "Sorry" in detail and "train step" in detail and "README" in detail
        assert MARK not in response.text and str(folder) not in response.text
        assert c.get("/").status_code == 200
        assert c.post("/predict", json={"request_text": ""}).status_code == 422


def test_no_model_503(main, test_client, tmp_path):
    assert_unavailable(main, test_client, tmp_path)


def test_missing_meta_503(main, test_client, model_copy):
    (model_copy / "router_meta.json").unlink()
    assert_unavailable(main, test_client, model_copy)


def test_version_mismatch_503(main, test_client, model_copy):
    edit_meta(model_copy, lambda meta: meta.update(sklearn_version="0.0.1"))
    assert_unavailable(main, test_client, model_copy)


def test_wrong_window_503(main, test_client, model_copy):
    edit_meta(model_copy, lambda meta: meta["training_window"].update(last_day="2026-03-31"))
    assert_unavailable(main, test_client, model_copy)


def test_corrupt_file_503(main, test_client, model_copy):
    (model_copy / "router.joblib").write_bytes(b"not a model")
    assert_unavailable(main, test_client, model_copy)


# --- Criterion 5: the page ---------------------------------------------------------------------------------------
class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags, self.examples, self.scripts, self._script = [], [], {}, None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "script":
            self._script = attrs.get("id", "main")
            self.scripts[self._script] = ""
        if tag == "button" and "data-text" in attrs:
            self.examples.append(attrs["data-text"])

    def handle_endtag(self, tag):
        if tag == "script":
            self._script = None

    def handle_data(self, data):
        if self._script:
            self.scripts[self._script] += data


@pytest.fixture(scope="module")
def html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    return response.text


@pytest.fixture(scope="module")
def page(html):
    return Page(html)


def test_page(page, html, config):
    names = [tag for tag, _ in page.tags]
    assert names.count("form") == 1 and names.count("textarea") == 1
    assert {a.get("id") for tag, a in page.tags if tag == "select"} == {"channel", "product_family", "warranty_status"}
    assert len(page.examples) == len(set(page.examples)) == 5
    assert PAID in page.examples and VAGUE in page.examples
    for part in ("Submit", "Needs a clarifying question", "First guess"):
        assert part in html
    options = json.loads(page.scripts["options"])
    assert options == {
        "channel": list(config.CHANNELS),
        "product_family": list(config.PRODUCT_FAMILIES),
        "warranty_status": list(config.WARRANTY_STATUSES),
    }


def test_page_is_safe_and_self_contained(page, html, root_dir):
    script = page.scripts["main"]
    assert "textContent" in script
    for name in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "DOMParser", "srcdoc"):
        assert name not in html
    assert not re.search(r"(?:https?:)?//[a-z0-9-]+\.", html)  # no external address
    assert not [a for tag, a in page.tags if tag in ("script", "link", "img", "iframe") and (a.get("src") or a.get("href"))]
    assert "@import" not in html and "url(" not in html
    assert not [k for _, a in page.tags for k in a if k.startswith("on")]  # no inline event handlers
    source = (root_dir / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert source.count("__OPTIONS__") == 1 and "__OPTIONS__" not in html


def test_options_cannot_close_the_script(main):
    out = main.options_json({"channel": ["</script><b>"]})
    assert "<" not in out and json.loads(out) == {"channel": ["</script><b>"]}


def test_options_equal_the_model_categories(router, config):
    encoder = router.model.named_steps["features"].named_transformers_["category"]
    expected = {name: list(values) for name, values in zip(encoder.feature_names_in_, encoder.categories_)}
    assert expected == {
        "channel": list(config.CHANNELS),
        "product_family": list(config.PRODUCT_FAMILIES),
        "warranty_status": list(config.WARRANTY_STATUSES),
    }


def test_examples_show_what_they_claim(page, router):
    repair, install, paid, vague, close = [router.predict([{"request_text": t}])[0] for t in page.examples]
    assert repair["team"] == "Repairs" and not repair["needs_triage"]
    assert install["team"] == "Installs & Demo" and not install["needs_triage"]
    assert paid["team"] != "Billing"
    assert vague["needs_triage"]
    assert any(reason.startswith("close call between") for reason in close["reasons"])


def test_clear_repair_reasons(client, page):
    body = client.post("/predict", json={"request_text": page.examples[0]}).json()
    assert body["reasons"][:2] == ["mentions 'grinder stopped working'", "mentions 'smell'"]
    assert "mentions 'working'" not in body["reasons"]


def test_examples_are_invented(page, processed_dir):
    """Counts only. VAGUE is exempt: the task spec gives it."""
    from src.clean import clean_text

    texts = set()
    for name in ("train_clean.parquet", "test_clean.parquet"):
        texts |= set(pd.read_parquet(processed_dir / name, columns=["text_clean"])["text_clean"])
    assert sum(clean_text(t) in texts for t in page.examples if t != VAGUE) == 0


def test_no_cdn_docs_pages(client):
    assert client.get("/docs").status_code == client.get("/redoc").status_code == 404


# --- Criterion 6: logs ------------------------------------------------------------------------------------------------
def test_no_text_in_logs(main, test_client, router, capfd):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    def crash(records):
        raise ValueError(records[0]["request_text"])

    try:
        with test_client(main.create_app()) as c:
            assert c.post("/predict", json={"request_text": f"{MARK} fan is broken"}).status_code == 200
            bad = {"request_text": f"{MARK} fan", "channel": f"{MARK}fax"}
            assert c.post("/predict", json=bad).status_code == 422
            c.app.state.router = SimpleNamespace(predict=crash, meta=router.meta)
            response = c.post("/predict", json={"request_text": f"{MARK} fan is broken"})
            assert response.status_code == 500
            assert MARK not in response.text and len(response.json()["detail"]) > 10
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    captured = capfd.readouterr()
    logged = stream.getvalue() + captured.out + captured.err
    assert "ValueError" in logged  # the failure itself is logged, by class only
    assert MARK not in logged


def test_real_server(root_dir, router):
    """uvicorn as the run target starts it: local address only, and nothing of the request in its output."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]
    process = subprocess.Popen(command, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                break
            except (urllib.error.URLError, OSError):
                assert process.poll() is None, "the server stopped"
                time.sleep(0.25)
        data = json.dumps({"request_text": f"{MARK} fan is broken"}).encode()
        request = urllib.request.Request(f"{base}/predict", data, {"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(request, timeout=10))["team"]
    finally:
        process.terminate()
        output = process.communicate(timeout=20)[0].decode("utf-8", "replace")
    assert f"http://127.0.0.1:{port}" in output and "POST /predict" in output
    assert MARK not in output


def test_run_target_is_local(root_dir):
    script = (root_dir / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    assert 'serve) "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ;;' in script


# --- Criterion 7: speed -------------------------------------------------------------------------------------------------
def test_20_calls_under_2s(client):
    start = time.perf_counter()
    for i in range(20):
        assert client.post("/predict", json=RECORDS[i % len(RECORDS)]).status_code == 200
    assert time.perf_counter() - start < 2


# --- Criterion 8: mojibake ------------------------------------------------------------------------------------------------
def test_mojibake(client):
    damaged = client.post("/predict", json={"request_text": "thÃ© mcb trips when the heater is on"})
    repaired = client.post("/predict", json={"request_text": "the mcb trips when the heater is on"})
    assert damaged.status_code == 200
    assert damaged.json() == repaired.json()
    assert "Ã" not in damaged.text
    assert client.post("/predict", json={"request_text": "thÃ© mcb"}).status_code == 200


# --- Dependencies and README ---------------------------------------------------------------------------------------------
def test_dependencies_are_pinned(root_dir):
    from importlib.metadata import version

    lines = (root_dir / "requirements.txt").read_text(encoding="utf-8").split()
    for name in ("fastapi", "uvicorn", "httpx"):
        assert f"{name}=={version(name)}" in lines


def test_readme_has_the_run_section(root_dir):
    text = (root_dir / "README.md").read_text(encoding="utf-8")
    section = text.split("## Run the service")[1]
    for part in ("PowerShell", "macOS", "Linux", "src.predictions train", "uvicorn app.main:app", "http://127.0.0.1:8000"):
        assert part in section
