"""Management dashboard. Expected numbers are recomputed here from reports/*.json, not with src.dashboard helpers.

Number rule: every number a reader can see (text, tooltips, labels) is a formatted JSON value, or one of the derived numbers
computed in expected() (per month, share of a row, over/under-count), or an axis tick. Failures never print customer text.
"""
import json
import re
from html.parser import HTMLParser

import pandas as pd
import pytest

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
TICKS = {"50%", "60%", "70%", "80%", "90%", "100%"}
VOID = {"meta", "br", "link", "img", "input"}


@pytest.fixture(scope="module")
def mod():
    from src import dashboard

    return dashboard


@pytest.fixture(scope="module")
def html(mod, tmp_path_factory):
    """Built into a temp folder: tests never touch reports/."""
    return mod.build(tmp_path_factory.mktemp("dashboard") / "dashboard.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reports(reports_dir):
    names = ("evaluation", "audit_metrics", "predictions_summary")
    return {n: json.loads((reports_dir / f"{n}.json").read_text(encoding="utf-8")) for n in names}


class Page(HTMLParser):
    """Visible text by element id and class, tooltip and label attributes, tables, and raw tags."""

    def __init__(self, html):
        super().__init__()
        self.stack, self.tags, self.text, self.ticks, self.attributes, self.scripts = [], [], [], [], [], {}
        self.by_id, self.rows, self._row, self.tips = {}, [], None, []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        self.attributes += [attrs[k] for k in ("data-value", "data-label", "aria-label", "title", "alt") if attrs.get(k)]
        if "data-value" in attrs:
            self.tips.append((attrs["data-value"], attrs["data-label"]))
        if tag == "tr":
            self._row = []
        if tag in ("td", "th") and self._row is not None:
            self._row.append("")
        if tag not in VOID:
            self.stack.append((tag, attrs))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.stack.pop()

    def handle_endtag(self, tag):
        assert self.stack and self.stack[-1][0] == tag, f"badly nested </{tag}>"
        self.stack.pop()
        if tag == "tr":
            self.rows.append((self._table(), self._row))
            self._row = None

    def _table(self):
        return next((a.get("id") for t, a in reversed(self.stack) if t in ("section", "div") and a.get("id")), None)

    def handle_data(self, data):
        if self.stack and self.stack[-1][0] == "script":
            self.scripts[self.stack[-1][1].get("id", "main")] = data
            return
        if self.stack and self.stack[-1][0] == "style":
            return
        for _, attrs in self.stack:
            if attrs.get("id"):
                self.by_id[attrs["id"]] = self.by_id.get(attrs["id"], "") + " " + data
        if any(t in ("td", "th") for t, _ in self.stack) and self._row:
            self._row[-1] += data
        (self.ticks if self.stack and self.stack[-1][1].get("class") == "tick" else self.text).append(data)


@pytest.fixture(scope="module")
def page(html):
    return Page(html)


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def rs(x):
    return f"Rs {x:,.0f}"


def signed(x):
    return ("+" if x >= 0 else "−") + f"{abs(100 * x):.0f}%"


@pytest.fixture(scope="module")
def expected(reports):
    """The numbers each panel must show."""
    ev, audit, pred = reports["evaluation"], reports["audit_metrics"], reports["predictions_summary"]
    months, teams = audit["months_covered"]["count"], ev["meta"]["teams"]
    closed, labels = audit["final_volume_by_team"], audit["bot_label_volume_by_team"]
    month = ev["costs"]["per_month_rs"]
    policy = ev["coverage_curve"]["policy_point"]
    points = {(p["coverage"], p["accuracy_on_covered"]) for p in ev["coverage_curve"]["points"]}
    points.add((policy["coverage"], policy["accuracy_on_covered"]))
    return {
        "teams": teams,
        "recall": {t: [pct(ev["per_team"][t][s]["recall"]) for s in ("router", "bot")] for t in teams},
        "headcount": {
            t: [f"{closed[t] / months:.1f}", f"{labels[t] / months:.1f}", signed(labels[t] / closed[t] - 1)] for t in teams
        },
        "costs": {
            "Bot": rs(month["bot"]),
            "Router, no question asked": rs(month["router_without_triage"]),
            "Router, question always works": rs(month["router_clarification_always_works"]),
            "Router, question never helps": rs(month["router_clarification_fixes_nothing"]),
        },
        "coverage": {(pct(c), pct(a)) for c, a in points},
        "confusion": ev["confusion"],
        "kpi": {
            "kpi_accuracy": [
                pct(ev["headline"]["router"]["accuracy"]),
                pct(ev["headline"]["bot"]["accuracy"]),
                f"{ev['headline']['paired_accuracy_points_router_minus_bot']['point']:.1f}",
                pct(ev["meta"]["ci_level"], 0),
                pct(ev["headline"]["router"]["accuracy_ci95"]["low"]),
                pct(ev["headline"]["router"]["accuracy_ci95"]["high"]),
            ],
            "kpi_ready": [pct(policy["coverage"]), pct(ev["triage"]["accuracy_sent"])],
            "kpi_question": [pct(ev["triage"]["rate"]), pct(ev["triage"]["accuracy_triaged"])],
            "kpi_cost": [
                rs(month[k])[3:] for k in ("bot", "router_without_triage", "router_clarification_always_works")
            ],
            "kpi_licence": [f"{ev['costs']['inputs']['bot_licence_per_year_rs'] / 100_000:.1f}"],
        },
        "break_even": pct(ev["costs"]["break_even_clarification_rate"]),
        "other": {
            f"{ev['costs']['inputs']['cost_clarify_rs']}",
            f"{ev['meta']['validation']['rows']:,}",
            f"{ev['meta']['training_window']['rows']:,}",
            f"{pred['model']['training_window']['rows']:,}",
            f"{months}",
            pct(ev["expected_score"]["final_team"]["low"]),
            pct(ev["expected_score"]["final_team"]["high"]),
            pct(ev["headline"]["bot"]["accuracy"]),
            pct(policy["coverage"]),
            pct(policy["accuracy_on_covered"]),
        },
        "versions": [ev["meta"]["model_version"], pred["model"]["model_version"]],
        "years": {d[:4] for d in (ev["meta"]["training_window"]["first_day"], ev["meta"]["validation"]["last_day"])},
    }


def heat_numbers(expected, side):
    out = set()
    for row in expected["confusion"][side]:
        for count in row:
            out |= {pct(count / sum(row)), pct(count / sum(row), 0), f"{count:,}", f"{sum(row):,}"}
    return out


def allowed(expected):
    numbers = set(expected["other"]) | expected["years"] | {expected["break_even"]}
    for values in (*expected["recall"].values(), *expected["kpi"].values()):
        numbers |= set(values)
    for values in expected["headcount"].values():
        numbers |= {v.lstrip("+−") for v in values}
    numbers |= {value[3:] for value in expected["costs"].values()}
    numbers |= {value for pair in expected["coverage"] for value in pair}
    return numbers | heat_numbers(expected, "router") | heat_numbers(expected, "bot")


def unsourced(page, expected):
    text = " \n ".join([*page.text, *page.attributes])
    for version in expected["versions"]:
        text = text.replace(version, " ")
    return [n for n in (m.rstrip(",") for m in NUMBER.findall(text)) if n not in allowed(expected)]


# --- Build ------------------------------------------------------------------------------------------
def test_builds_and_is_deterministic(mod, tmp_path):
    first, second = mod.build(tmp_path / "a.html"), mod.build(tmp_path / "b.html")
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size > 10_000 and b"\r" not in first.read_bytes()


def test_committed_dashboard_is_current(mod, html, page, reports_dir):
    """Tolerance as in M6: another BLAS or CPU may move model floats slightly."""

    def leaves(node):
        if isinstance(node, dict):
            node = [node[k] for k in sorted(node)]
        return [leaf for child in node for leaf in leaves(child)] if isinstance(node, list) else [node]

    committed = Page((reports_dir / mod.HTML_NAME).read_text(encoding="utf-8"))
    stored, fresh = (leaves(json.loads(p.scripts["data"])) for p in (committed, page))
    assert len(stored) == len(fresh)
    for a, b in zip(stored, fresh):
        same = a == pytest.approx(b, rel=1e-4, abs=1e-4) if isinstance(a, float) else a == b
        assert same, "reports/dashboard.html is stale: run bash scripts/run_all.sh dashboard"


def test_embedded_numbers_are_the_reports_own(page, reports):
    def floats(node):
        if isinstance(node, dict):
            node = list(node.values())
        if isinstance(node, list):
            return [leaf for child in node for leaf in floats(child)]
        return [node] if isinstance(node, (int, float)) and not isinstance(node, bool) else []

    known = set(floats(reports))
    data = floats(json.loads(page.scripts["data"]))
    assert len(data) > 100 and [x for x in data if x not in known] == []


# --- Numbers ----------------------------------------------------------------------------------------
def test_every_number_shown_has_a_source(page, expected):
    assert unsourced(page, expected) == []


def test_only_known_axis_ticks(page):
    assert {n for text in page.ticks for n in NUMBER.findall(text)} == TICKS


def test_kpi_tiles(page, expected):
    for tile, numbers in expected["kpi"].items():
        assert NUMBER.findall(page.by_id[tile]) == numbers, tile


def rows(page, panel):
    return {row[0]: row[1:] for owner, row in page.rows if owner == panel and len(row) > 1}


def tips(page, label):
    return [value for value, text in page.tips if text == label]


def test_recall_panel(page, expected):
    shown = rows(page, "recall")
    for team, values in expected["recall"].items():
        assert shown[team] == values
        assert tips(page, f"{team}, router") == [values[0]] and tips(page, f"{team}, bot") == [values[1]]


def test_headcount_panel(page, expected):
    shown = rows(page, "headcount")
    for team, values in expected["headcount"].items():
        assert shown[team] == values
        assert tips(page, f"{team}, closed per month") == [values[0]]
        assert tips(page, f"{team}, bot labels per month") == [values[1]]
        word = "over" if values[2].startswith("+") else "under"
        assert f"{team} Bot {word} by {values[2].lstrip('+−')}" in re.sub(r"\s+", " ", page.by_id["headcount"])


def test_cost_panel(page, expected):
    shown = rows(page, "costs")
    for label, value in expected["costs"].items():
        assert shown[label] == [value] and tips(page, f"{label}, per month") == [value]
    assert page.by_id["break_even"].strip() == expected["break_even"]


def test_coverage_panel(page, expected):
    shown = {(row[0], row[1]) for owner, row in page.rows if owner == "coverage" and row[0] != "Share sent straight on"}
    assert shown == expected["coverage"]
    stops = {(m.group(1), value) for value, text in page.tips if (m := re.match(r"right, with (\S+) of requests", text))}
    assert stops == expected["coverage"]


@pytest.mark.parametrize("side", ["router", "bot"])
def test_heatmap_cells(page, expected, side):
    teams, counts = expected["teams"], expected["confusion"][side]
    shown = rows(page, f"matrix_{side}")
    heading = re.sub(r"\s+", " ", page.by_id[f"matrix_{side}"]).strip()
    assert heading.startswith(f"{side.title()} Rows: the team that closed the request. Columns: where the {side} sent it.")
    for i, team in enumerate(teams):
        total = sum(counts[i])
        assert shown[team] == [pct(c / total, 0) for c in counts[i]]
        for j, other in enumerate(teams):
            label = f"of the requests {team} closed went to {other}: {counts[i][j]:,} of {total:,}"
            assert pct(counts[i][j] / total) in tips(page, label)


def test_heatmap_text_contrast(html):
    def luminance(colour):
        c = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    cells = re.findall(r'<td class="\w*" style="background:(#[0-9A-F]{6})"', html)
    assert len(cells) == 98
    ink = luminance("#1F2937")
    assert min((luminance(c) + 0.05) / (ink + 0.05) for c in cells) >= 4.5


def test_mark_lengths_follow_the_values(mod, reports, html):
    """Bar widths are shares of the longest bar; a wrong width would mislead although the label is right."""
    ev, audit = reports["evaluation"], reports["audit_metrics"]
    found = [float(w) for w in re.findall(r"--w:([\d.]+)", html)]
    recall = [ev["per_team"][t][s]["recall"] for t in ev["meta"]["teams"] for s in ("router", "bot")]
    closed, labels = audit["final_volume_by_team"], audit["bot_label_volume_by_team"]
    longest = max([*closed.values(), *labels.values()])
    order = sorted(ev["meta"]["teams"], key=lambda t: -closed[t])
    headcount = [v / longest for t in order for v in (closed[t], labels[t])]
    month = ev["costs"]["per_month_rs"]
    keys = ("bot", "router_without_triage", "router_clarification_always_works", "router_clarification_fixes_nothing")
    costs = [month[k] / max(month[k] for k in keys) for k in keys]
    assert found == pytest.approx([*recall, *headcount, *costs], abs=1e-4)


def test_curve_is_drawn_to_its_own_scale(page, html, expected, reports):
    """The scale comes from the axis ticks on the page; points, line, marker and bot line must sit on it."""
    texts = re.findall(r'<text class="tick" x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)">(\d+)%</text>', html)
    xs = {int(v): float(x) for x, _, anchor, v in texts if anchor == "middle"}
    ys = {int(v): float(y) - 4 for _, y, anchor, v in texts if anchor == "end"}
    assert xs[50] < xs[100] and ys[100] < ys[70]  # right is more, up is more (SVG y grows downwards)

    def place(share, scale):
        low, high = min(scale), max(scale)
        return scale[low] + (100 * share - low) / (high - low) * (scale[high] - scale[low])

    ev = reports["evaluation"]
    policy = ev["coverage_curve"]["policy_point"]
    points = {(p["coverage"], p["accuracy_on_covered"]) for p in ev["coverage_curve"]["points"]}
    where = sorted((place(c, xs), place(a, ys)) for c, a in points)
    path = re.search(r'class="router_line" d="([^"]+)"', html).group(1)
    drawn = [(float(x), float(y)) for x, y in re.findall(r"[ML]([\d.]+) ([\d.]+)", path)]
    assert [v for point in drawn for v in point] == pytest.approx([v for point in where for v in point], abs=0.2)
    marker = re.search(r'class="policy" cx="([\d.]+)" cy="([\d.]+)"', html).groups()
    target = (place(policy["coverage"], xs), place(policy["accuracy_on_covered"], ys))
    assert [float(v) for v in marker] == pytest.approx(target, abs=0.2)
    bot = re.search(r'class="bot_line" x1="[\d.]+" x2="[\d.]+" y1="([\d.]+)" y2="([\d.]+)"', html).groups()
    assert [float(v) for v in bot] == pytest.approx([place(ev["headline"]["bot"]["accuracy"], ys)] * 2, abs=0.2)
    stops = {(float(x), float(y)) for x, y in re.findall(r'class="stop" cx="([\d.]+)" cy="([\d.]+)"', html)}
    assert len(stops) == len(expected["coverage"])


def test_number_rule_catches_a_changed_number(html, expected):
    for old, new in ((expected["break_even"], "12.34%"), (expected["kpi"]["kpi_cost"][0], "999,999")):
        assert old in html
        assert unsourced(Page(html.replace(old, new)), expected) != []


def test_footer_windows_and_models(page, expected):
    text = re.sub(r"\s+", " ", " ".join(page.text))
    assert "Numbers from validation Apr–Jun 2026" in text and "Apr 2025–Jun 2026" in text
    assert all(version in text for version in expected["versions"])


# --- Self-contained and safe --------------------------------------------------------------------------
def test_no_external_address(html, page):
    assert not re.search(r"https?:|(?<!:)//[a-z0-9-]+", html, re.IGNORECASE)
    assert "url(" not in html and "@import" not in html
    assert not [a for _, a in page.tags for k in a if k in ("src", "href", "srcset", "xlink:href")]
    assert not [a for tag, a in page.tags if tag in ("link", "img", "iframe", "object", "embed", "base")]
    assert not [k for _, a in page.tags for k in a if k.startswith("on")]
    for name in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "fetch(", "localStorage"):
        assert name not in html


def test_one_tab_stop_per_chart(page):
    assert sum(a.get("tabindex") == "0" for _, a in page.tags if "data-value" in a) == sum(
        "data-roving" in a for _, a in page.tags
    )


def four_word_runs(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + 4]) for i in range(len(words) - 3)}


def test_no_customer_text(page, config):
    """Counts only: a failure never prints the text."""
    customer = set()
    for path in (config.TRAIN_CLEAN_PARQUET, config.TEST_CLEAN_PARQUET):
        for text in pd.read_parquet(path, columns=["text_clean"])["text_clean"]:
            customer |= four_word_runs(text)
    assert len(customer) > 1000
    shown = [*page.text, *page.ticks, *page.attributes, page.scripts["data"]]
    runs = set().union(*(four_word_runs(text) for text in shown)) | four_word_runs(" ".join(page.text))
    overlap = len(runs & customer)
    assert overlap == 0


def test_no_old_team_names(html, config):
    for old in config.TEAM_NAME_MAP:
        assert not re.search(rf"{old}(?!\w)", html.replace("Filters & Consumables", "").replace("Filters &amp; Consumables", ""))


def test_run_target_and_readme(root_dir):
    script = (root_dir / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    assert 'dashboard) "$PY" -m src.dashboard ;;' in script
    assert "open reports/dashboard.html" in (root_dir / "README.md").read_text(encoding="utf-8")
