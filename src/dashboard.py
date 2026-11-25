"""Management dashboard: reports/*.json -> reports/dashboard.html. Run: python -m src.dashboard

One self-contained page: inline SVG and CSS, no library, no external address. Only the keys picked in load_data() are used, so no
failure phrase or example message can reach the page. Derived numbers (per month, shares, over/under-count) are computed here only.
"""
import json
from datetime import date
from html import escape
from pathlib import Path

from src import config

HTML_NAME = "dashboard.html"
SOURCES = ("evaluation", "audit_metrics", "predictions_summary")
COST_CASES = (  # key in evaluation costs per_month_rs, label, series, explanation
    ("bot", "Bot", "bot", "What the bot's wrong sends cost now, on the same requests: transfers plus extra customer contacts."),
    (
        "router_without_triage",
        "Router, no question asked",
        "router",
        "Every request goes straight to the router's first guess. The figure to plan on.",
    ),
    (
        "router_clarification_always_works",
        "Router, question always works",
        "router",
        "Unsure requests get one question, and the answer always finds the right team. Best case.",
    ),
    (
        "router_clarification_fixes_nothing",
        "Router, question never helps",
        "router",
        "The question is paid for but never changes the outcome. Worst case.",
    ),
)


# --- Numbers --------------------------------------------------------------------------------------
def load_data(reports_dir: Path = config.REPORTS_DIR) -> dict:
    """The numbers on the page, copied unchanged from the reports."""
    r = {name: json.loads((reports_dir / f"{name}.json").read_text(encoding="utf-8")) for name in SOURCES}
    ev, audit, pred = r["evaluation"], r["audit_metrics"], r["predictions_summary"]
    return {
        "teams": ev["meta"]["teams"],
        "ci_level": ev["meta"]["ci_level"],
        "validation": ev["meta"]["validation"],
        "validation_model": {"version": ev["meta"]["model_version"], "window": ev["meta"]["training_window"]},
        "live_model": {"version": pred["model"]["model_version"], "window": pred["model"]["training_window"]},
        "accuracy": {
            "router": ev["headline"]["router"]["accuracy"],
            "bot": ev["headline"]["bot"]["accuracy"],
            "router_ci": ev["headline"]["router"]["accuracy_ci95"],
            "lead_points": ev["headline"]["paired_accuracy_points_router_minus_bot"]["point"],
        },
        "expected": {key: ev["expected_score"]["final_team"][key] for key in ("low", "high")},
        "triage": {key: ev["triage"][key] for key in ("rate", "accuracy_sent", "accuracy_triaged")},
        "coverage": {"points": ev["coverage_curve"]["points"], "policy_point": ev["coverage_curve"]["policy_point"]},
        "recall": {
            team: {side: ev["per_team"][team][side]["recall"] for side in ("router", "bot")} for team in ev["meta"]["teams"]
        },
        "confusion": ev["confusion"],
        "headcount": {
            "months": audit["months_covered"],
            "closed": audit["final_volume_by_team"],
            "bot_labels": audit["bot_label_volume_by_team"],
        },
        "costs": {
            "per_month_rs": {key: ev["costs"]["per_month_rs"][key] for key, *_ in COST_CASES},
            "break_even_clarification_rate": ev["costs"]["break_even_clarification_rate"],
            "cost_clarify_rs": ev["costs"]["inputs"]["cost_clarify_rs"],
            "bot_licence_per_year_rs": ev["costs"]["inputs"]["bot_licence_per_year_rs"],
        },
    }


def pct(share: float, digits: int = 1) -> str:
    return f"{100 * share:.{digits}f}%"


def rs(amount: float) -> str:
    return f"Rs {amount:,.0f}"


def per_month(count: int, months: int) -> float:
    return count / months


def over_count(labels: int, closed: int) -> float:
    """Share by which the bot's labels exceed (or, negative, fall short of) the requests the team closed."""
    return labels / closed - 1


def signed_pct(share: float) -> str:
    return f"{'+' if share >= 0 else '−'}{abs(100 * share):.0f}%"


def month_range(first: str, last: str) -> str:
    """'2026-04-01', '2026-06-30' -> 'Apr–Jun 2026'; years are repeated only when they differ."""
    start, end = (date.fromisoformat(d if len(d) == 10 else f"{d}-01") for d in (first, last))
    left = f"{start:%b}" if start.year == end.year else f"{start:%b %Y}"
    return f"{left}–{end:%b %Y}"


# --- Small HTML pieces --------------------------------------------------------------------------------
def tip(value: str, label: str) -> str:
    return f'data-value="{escape(value)}" data-label="{escape(label)}"'


def legend(*items: tuple[str, str]) -> str:
    return (
        '<ul class="legend">'
        + "".join(f'<li><span class="key {series}"></span>{escape(text)}</li>' for series, text in items)
        + "</ul>"
    )


def table(caption: str, head: list[str], rows: list[list[str]]) -> str:
    """The table view that goes with every chart."""
    header = "".join(f'<th scope="col">{escape(h)}</th>' for h in head)
    body = "".join(
        f'<tr><th scope="row">{escape(row[0])}</th>' + "".join(f"<td>{escape(cell)}</td>" for cell in row[1:]) + "</tr>"
        for row in rows
    )
    return (
        f'<details><summary>Show table</summary><div class="scroll"><table class="plain"><caption>{escape(caption)}</caption>'
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div></details>"
    )


def bar(series: str, share: float, text: str, label: str, first: bool) -> str:
    """One horizontal bar from zero, with its value at the tip. share is of the longest bar in the chart."""
    return (
        f'<div class="bar" role="img" tabindex="{0 if first else -1}" aria-label="{escape(label)}: {escape(text)}" '
        f'{tip(text, label)}><span class="mark {series}" style="--w:{share:.4f}"></span>'
        f'<span class="tip_value" aria-hidden="true">{escape(text)}</span></div>'
    )


def panel(panel_id: str, title: str, lead: str, body: str, wide: bool = False) -> str:
    return (
        f'<section class="card{" wide" if wide else ""}" id="{panel_id}"><h2>{escape(title)}</h2>'
        f'<p class="lead">{escape(lead)}</p>{body}</section>'
    )


# --- Panels ----------------------------------------------------------------------------------------------
def kpis(d: dict) -> str:
    acc, tri, costs = d["accuracy"], d["triage"], d["costs"]
    month = costs["per_month_rs"]
    tiles = [
        (
            "kpi_accuracy",
            "Router sends to the right team",
            pct(acc["router"]),
            f"Bot {pct(acc['bot'])} · router ahead by {acc['lead_points']:.1f} points · "
            f"router's {pct(d['ci_level'], 0)} range {pct(acc['router_ci']['low'])} to {pct(acc['router_ci']['high'])}",
        ),
        (
            "kpi_ready",
            "Ready to route",
            pct(d["coverage"]["policy_point"]["coverage"]),
            f"of requests go straight on; {pct(tri['accuracy_sent'])} of them to the right team",
        ),
        (
            "kpi_question",
            "Needs one question",
            pct(tri["rate"]),
            f"of requests; the first guess alone is right for {pct(tri['accuracy_triaged'])} of them",
        ),
        (
            "kpi_cost",
            "Bot misroute cost per month",
            rs(month["bot"]),
            f"Router {rs(month['router_without_triage'])} with no question asked, "
            f"{rs(month['router_clarification_always_works'])} at best",
        ),
        (
            "kpi_licence",
            "Bot licence per year",
            f"Rs {costs['bot_licence_per_year_rs'] / 100_000:.1f} lakh",
            "Saved if the bot is switched off. Not part of the misroute costs.",
        ),
    ]
    return '<section class="kpis" aria-label="Headline numbers">' + "".join(
        f'<div class="card kpi" id="{tile_id}"><h2>{escape(label)}</h2><p class="value">{escape(value)}</p>'
        f'<p class="sub">{escape(sub)}</p></div>'
        for tile_id, label, value, sub in tiles
    ) + "</section>"


def coverage_panel(d: dict) -> str:
    left, right, top, bottom, width, height = 50, 14, 14, 52, 480, 320
    x_low, y_low = 0.45, 0.70

    def x(share: float) -> float:
        return left + (share - x_low) / (1 - x_low) * (width - left - right)

    def y(share: float) -> float:
        return top + (1 - share) / (1 - y_low) * (height - top - bottom)

    policy, bot = d["coverage"]["policy_point"], d["accuracy"]["bot"]
    points = sorted({(p["coverage"], p["accuracy_on_covered"]) for p in d["coverage"]["points"]})
    grid = "".join(
        f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(t / 100):.1f}" y2="{y(t / 100):.1f}"/>'
        f'<text class="tick" x="{left - 8}" y="{y(t / 100) + 4:.1f}" text-anchor="end">{t}%</text>'
        for t in (70, 80, 90, 100)
    ) + "".join(
        f'<text class="tick" x="{x(t / 100):.1f}" y="{height - bottom + 20}" text-anchor="middle">{t}%</text>'
        for t in (50, 60, 70, 80, 90, 100)
    )
    path = " ".join(f"{'L' if i else 'M'}{x(c):.1f} {y(a):.1f}" for i, (c, a) in enumerate(points))

    def stop(coverage: float, accuracy: float, i: int) -> str:
        label = f"right, with {pct(coverage)} of requests sent straight on (bot: {pct(bot)} of all requests)"
        return (
            f'<circle class="stop" cx="{x(coverage):.1f}" cy="{y(accuracy):.1f}" r="5" role="img" '
            f'tabindex="{0 if i == 0 else -1}" aria-label="{escape(pct(accuracy) + " " + label)}" {tip(pct(accuracy), label)}/>'
        )

    stops = sorted([*points, (policy["coverage"], policy["accuracy_on_covered"])])
    px, py = x(policy["coverage"]), y(policy["accuracy_on_covered"])
    svg = (
        f'<svg class="curve" viewBox="0 0 {width} {height}" role="group" aria-label="Accuracy against share sent straight on" '
        f'data-left="{left}" data-right="{width - right}">{grid}'
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}"/>'
        f'<line class="bot_line" x1="{left}" x2="{width - right}" y1="{y(bot):.1f}" y2="{y(bot):.1f}"/>'
        f'<text class="note" x="{left + 8}" y="{y(bot) + 18:.1f}">Bot, all requests: {pct(bot)}</text>'
        f'<path class="router_line" d="{path}"/>'
        f'<line class="cross" id="cross" x1="0" x2="0" y1="{top}" y2="{height - bottom}"/>'
        f'<rect class="hit" x="{left}" y="{top}" width="{width - left - right}" height="{height - top - bottom}"/>'
        f'<g class="stops" data-roving="1">{"".join(stop(c, a, i) for i, (c, a) in enumerate(stops))}</g>'
        f'<circle class="policy" cx="{px:.1f}" cy="{py:.1f}" r="7"/>'
        f'<text class="note strong" x="{px - 12:.1f}" y="{py + 30:.1f}" text-anchor="end">Today\'s rule: {pct(policy["coverage"])} sent,</text>'
        f'<text class="note strong" x="{px - 12:.1f}" y="{py + 50:.1f}" text-anchor="end">'
        f'{pct(policy["accuracy_on_covered"])} right</text>'
        f'<text class="tick" x="{(left + width - right) / 2}" y="{height - 8}" text-anchor="middle">'
        "Share of requests sent straight on</text></svg>"
    )
    rows = [[pct(c), pct(a)] for c, a in sorted(stops, reverse=True)]
    body = (
        legend(("router line", "Router, right when sent straight on"), ("bot line", "Bot, right on all requests"))
        + svg
        + table("Router accuracy by share sent straight on", ["Share sent straight on", "Right team"], rows)
    )
    lead = "Asking about the unsure requests makes the rest more reliable. The vertical scale starts above zero."
    return panel("coverage", "Accuracy against share sent straight on", lead, body)


def recall_panel(d: dict) -> str:
    rows, cells = [], []
    for i, team in enumerate(d["teams"]):
        r = d["recall"][team]
        rows.append(
            f'<div class="row"><span class="name">{escape(team)}</span><div class="pair">'
            + bar("router", r["router"], pct(r["router"]), f"{team}, router", i == 0)
            + bar("bot", r["bot"], pct(r["bot"]), f"{team}, bot", False)
            + "</div></div>"
        )
        cells.append([team, pct(r["router"]), pct(r["bot"])])
    body = (
        legend(("router", "Router"), ("bot", "Bot"))
        + f'<div class="bars" data-roving="1">{"".join(rows)}</div>'
        + table("Share of each team's requests sent to it", ["Team", "Router", "Bot"], cells)
    )
    lead = "Of the requests each team closed, the share that was sent to it first time."
    return panel("recall", "Recall per team: requests that reached the right team", lead, body)


def mix(share: float) -> str:
    """Cell colour: one hue, light to dark, light enough for dark text everywhere."""
    low, high = (0xF0, 0xF7, 0xF6), (0x5F, 0xB0, 0xA7)
    return "#{:02X}{:02X}{:02X}".format(*(round(a + (b - a) * share) for a, b in zip(low, high)))


def matrix(d: dict, side: str, title: str) -> str:
    teams, counts = d["teams"], d["confusion"][side]
    head = "".join(f'<th scope="col"><span>{escape(team)}</span></th>' for team in teams)
    body = []
    for i, team in enumerate(teams):
        total = sum(counts[i])
        cells = []
        for j, other in enumerate(teams):
            share = counts[i][j] / total
            label = f"of the requests {team} closed went to {other}: {counts[i][j]:,} of {total:,}"
            cells.append(
                f'<td class="{"right" if i == j else ""}" style="background:{mix(share)}" '
                f'tabindex="{0 if i == j == 0 else -1}" {tip(pct(share), label)}>{pct(share, 0)}</td>'
            )
        body.append(f'<tr><th scope="row">{escape(team)}</th>{"".join(cells)}</tr>')
    return (
        f'<div class="matrix" id="matrix_{side}"><h3>{title}</h3><table class="heat"><caption>Rows: the team that closed the request. '
        f'Columns: where the {title.lower()} sent it.</caption><thead><tr><td></td>{head}</tr></thead>'
        f'<tbody data-roving="{len(teams)}">{"".join(body)}</tbody></table></div>'
    )


def heatmap_panel(d: dict) -> str:
    body = (
        '<div class="toggle" role="group" aria-label="Show router or bot">'
        '<button type="button" data-show="router" aria-pressed="true">Router</button>'
        '<button type="button" data-show="bot" aria-pressed="false">Bot</button></div>'
        f'<div class="matrices">{matrix(d, "router", "Router")}{matrix(d, "bot", "Bot")}</div>'
    )
    lead = "Each row adds up to one team's requests. The outlined diagonal is right first time; everything else is a misroute."
    return panel("heatmap", "Where each team's requests were sent", lead, body, wide=True)


def headcount_panel(d: dict) -> str:
    h = d["headcount"]
    months = h["months"]["count"]
    closed = {t: per_month(h["closed"][t], months) for t in d["teams"]}
    labels = {t: per_month(h["bot_labels"][t], months) for t in d["teams"]}
    longest = max([*closed.values(), *labels.values()])
    rows, cells = [], []
    for i, team in enumerate(sorted(d["teams"], key=lambda t: -closed[t])):
        gap = over_count(h["bot_labels"][team], h["closed"][team])
        word = "over" if gap > 0 else "under"
        rows.append(
            f'<div class="row"><span class="name">{escape(team)}<small>Bot {word} by {abs(gap) * 100:.0f}%</small></span>'
            '<div class="pair">'
            + bar("closed", closed[team] / longest, f"{closed[team]:.1f}", f"{team}, closed per month", i == 0)
            + bar("bot", labels[team] / longest, f"{labels[team]:.1f}", f"{team}, bot labels per month", False)
            + "</div></div>"
        )
        cells.append([team, f"{closed[team]:.1f}", f"{labels[team]:.1f}", signed_pct(gap)])
    body = (
        legend(("closed", "Closed by the team"), ("bot", "Labelled by the bot"))
        + f'<div class="bars" data-roving="1">{"".join(rows)}</div>'
        + '<p class="foot">Requests, not people: how long each one takes is not in the data.</p>'
        + table("Requests per month", ["Team", "Closed", "Bot labels", "Bot over or under"], cells)
    )
    lead = (
        f"Requests per month, averaged over {months} months ({month_range(h['months']['first'], h['months']['last'])}). "
        "Staffing from the bot's labels would over-staff the teams marked over and under-staff the rest."
    )
    return panel("headcount", "Workload: what teams closed against what the bot labelled", lead, body)


def cost_panel(d: dict) -> str:
    costs = d["costs"]
    month = costs["per_month_rs"]
    longest = max(month.values())
    rows, cells = [], []
    for i, (key, label, series, why) in enumerate(COST_CASES):
        rows.append(
            f'<div class="case"><span class="name">{escape(label)}</span>'
            + bar(series, month[key] / longest, rs(month[key]), f"{label}, per month", i == 0)
            + f'<p class="why">{escape(why)}</p></div>'
        )
        cells.append([label, rs(month[key])])
    rate = pct(costs["break_even_clarification_rate"])
    body = (
        legend(("router", "Router"), ("bot", "Bot"))
        + f'<div class="bars cases" data-roving="1">{"".join(rows)}</div>'
        + f'<div class="break_even"><strong id="break_even">{rate}</strong><p>Break-even: each question costs '
        + f"{rs(costs['cost_clarify_rs'])}, so asking pays only if it puts right more than this share of the flagged requests "
        + "whose first guess was wrong. How often it does is not in the data yet.</p></div>"
        + table("Misroute cost per month", ["Case", "Cost per month"], cells)
    )
    return panel("costs", "Misroute cost per month", "Licence not included. Same requests for the bot and the router.", body)


def footer(d: dict) -> str:
    val, live, scored, h = d["validation"], d["live_model"], d["validation_model"], d["headcount"]["months"]
    window = month_range(val["first_day"], val["last_day"])
    lines = [
        f"Numbers from validation {window} ({val['rows']:,} requests), except the workload panel: all {h['count']} labelled months, "
        f"{month_range(h['first'], h['last'])}.",
        f"Scored model {scored['version']}, trained on {scored['window']['rows']:,} requests from "
        f"{month_range(scored['window']['first_day'], scored['window']['last_day'])}.",
        f"Live model {live['version']}, trained on all {live['window']['rows']:,} labelled requests "
        f"({month_range(live['window']['first_day'], live['window']['last_day'])}). It has no score of its own.",
        f"Expected on new requests: {pct(d['expected']['low'])} to {pct(d['expected']['high'])} sent to the right team.",
        "The right team is the one that closed the request, not the bot's label.",
    ]
    return "<footer>" + "".join(f"<p>{escape(line)}</p>" for line in lines) + "</footer>"


STYLE = """
  :root {
    color-scheme: light;
    --bg: #F6F7F9; --card: #FFFFFF; --line: #E4E7EC; --primary: #0F766E; --text: #1F2937; --muted: #6B7280; --bot: #8A93A0;
    --shadow: 0 1px 2px rgba(16, 24, 40, .05), 0 4px 12px rgba(16, 24, 40, .05);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.5 "Segoe UI", system-ui, sans-serif; }
  h1, h2, h3, p, ul { margin: 0; padding: 0; }
  h1 { font-size: 1.25rem; line-height: 1.3; }
  h2 { font-size: 1.125rem; line-height: 1.35; font-weight: 600; }
  h3 { font-size: 1rem; font-weight: 600; }
  :focus-visible { outline: 3px solid var(--primary); outline-offset: 2px; }
  header { background: var(--card); border-top: 4px solid var(--primary); border-bottom: 1px solid var(--line); }
  .wrap { max-width: 1272px; margin: 0 auto; padding: 16px; }
  header p { color: var(--muted); font-size: .875rem; }
  main { display: grid; gap: 24px; padding-top: 24px !important; grid-template-columns: minmax(0, 1fr); }
  @media (min-width: 900px) { main { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); align-items: start; } }
  .wide, .kpis, footer { grid-column: 1 / -1; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 24px;
    min-width: 0; }
  .lead { color: var(--muted); font-size: .9375rem; margin: 4px 0 16px; }
  .kpis { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(212px, 1fr)); }
  .kpi { padding: 20px; }
  .kpi h2 { font-size: .9375rem; }
  .kpi .value { font-size: 2rem; line-height: 1.25; font-weight: 700; margin: 4px 0; }
  .kpi .sub { color: var(--muted); font-size: .875rem; }
  #kpi_accuracy .value, #kpi_ready .value { color: var(--primary); }

  .legend { list-style: none; display: flex; flex-wrap: wrap; gap: 4px 20px; font-size: .875rem; margin-bottom: 12px; }
  .legend li { display: flex; gap: 8px; align-items: center; }
  .key { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .key.line { width: 20px; height: 3px; border-radius: 2px; }
  .router { background: var(--primary); }
  .bot { background: var(--bot); }
  .closed { background: #374151; }

  .bars { display: grid; gap: 14px; }
  .row { display: grid; grid-template-columns: 168px minmax(0, 1fr); gap: 12px; align-items: center; }
  .name { font-size: .9375rem; line-height: 1.35; }
  .name small { display: block; color: var(--muted); font-size: .8125rem; }
  .pair { display: grid; gap: 2px; border-left: 1px solid var(--line); }
  .bar { display: flex; align-items: center; gap: 8px; min-height: 20px; border-radius: 4px; }
  .mark { height: 16px; border-radius: 0 4px 4px 0; width: calc(var(--w) * (100% - 92px)); min-width: 2px; flex: none; }
  .bar:hover .mark, .bar:focus .mark { filter: brightness(1.15); }
  .tip_value { font-size: .8125rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cases { gap: 18px; }
  .case .name { font-weight: 600; }
  .case .bar { border-left: 1px solid var(--line); border-radius: 0; margin: 2px 0; }
  .case .mark { height: 20px; }
  .why, .foot { color: var(--muted); font-size: .875rem; }
  .foot { margin-top: 12px; }
  .break_even { display: flex; gap: 16px; align-items: center; background: var(--bg); border-left: 4px solid var(--primary);
    border-radius: 8px; padding: 14px 16px; margin-top: 20px; font-size: .9375rem; }
  .break_even strong { font-size: 1.5rem; }

  .curve { display: block; width: 100%; height: auto; overflow: visible; }
  .curve text { font: 11px "Segoe UI", system-ui, sans-serif; fill: var(--text); }
  .curve .tick { fill: var(--muted); }
  .curve .note { paint-order: stroke; stroke: var(--card); stroke-width: 5px; stroke-linejoin: round; }
  .curve .strong { font-weight: 600; }
  .grid, .axis { stroke: var(--line); stroke-width: 1; }
  .router_line { fill: none; stroke: var(--primary); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .bot_line { stroke: var(--bot); stroke-width: 2; }
  .cross { stroke: var(--muted); stroke-width: 1; opacity: 0; }
  .hit { fill: transparent; }
  .stop { fill: var(--primary); stroke: var(--card); stroke-width: 2; opacity: 0; pointer-events: none; }
  .stop.on, .stop:focus { opacity: 1; }
  .policy { fill: var(--primary); stroke: var(--card); stroke-width: 2; pointer-events: none; }

  .toggle { display: none; gap: 8px; margin-bottom: 12px; }
  .js .toggle { display: flex; }
  button { font: inherit; font-size: .9375rem; font-weight: 600; min-height: 44px; padding: 6px 20px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--card); color: var(--text); cursor: pointer; }
  button:hover { border-color: var(--primary); }
  button[aria-pressed="true"] { background: var(--primary); border-color: var(--primary); color: #FFFFFF; }
  .matrices { display: grid; gap: 24px; }
  .js .matrix.off { display: none; }
  .heat { width: 100%; border-collapse: separate; border-spacing: 2px; table-layout: fixed; font-size: .875rem; }
  .heat caption { text-align: left; color: var(--muted); font-size: .875rem; padding: 2px 0 8px; }
  .heat th { font-weight: 400; text-align: left; line-height: 1.25; padding: 2px 4px; overflow-wrap: break-word; }
  .heat thead th { vertical-align: bottom; text-align: center; font-size: .8125rem; }
  .heat tbody td { text-align: center; height: 40px; padding: 0; border-radius: 4px; font-variant-numeric: tabular-nums; }
  .heat td.right { outline: 2px solid var(--text); outline-offset: -2px; font-weight: 700; }
  .heat td:focus-visible { outline: 3px solid var(--primary); outline-offset: 1px; }
  .heat tr > :first-child { width: 176px; }

  details { margin-top: 16px; font-size: .875rem; }
  summary { cursor: pointer; color: var(--primary); font-weight: 600; min-height: 32px; display: list-item; }
  .scroll { overflow-x: auto; }
  .plain { border-collapse: collapse; width: 100%; }
  .plain caption { text-align: left; color: var(--muted); padding: 4px 0; }
  .plain th, .plain td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
  .plain tr > :first-child { text-align: left; font-weight: 400; }
  .plain thead th { font-weight: 600; }
  footer { color: var(--muted); font-size: .875rem; overflow-wrap: anywhere; padding-bottom: 24px; }

  #tooltip { position: fixed; z-index: 5; max-width: 260px; padding: 8px 12px; border-radius: 8px; background: var(--text); color: #FFFFFF;
    font-size: .8125rem; line-height: 1.4; pointer-events: none; }
  #tooltip[hidden] { display: none; }
  #tooltip strong { display: block; font-size: 1rem; }

  @media (max-width: 480px) {
    .card { padding: 16px; }
    main { gap: 16px; }
    .row { grid-template-columns: minmax(0, 1fr); gap: 4px; }
    .name small { display: inline; margin-left: 8px; }
    .curve text { font-size: 16px; }
    .heat { font-size: .6875rem; border-spacing: 1px; }
    .heat tr > :first-child { width: 88px; }
    .heat th { font-size: .6875rem; padding: 2px 2px 2px 0; }
    .heat thead th span { writing-mode: vertical-rl; transform: rotate(180deg); display: inline-block; font-size: .6875rem; }
    .heat tbody td { height: 34px; }
    .break_even { align-items: flex-start; }
  }
  @media print {
    @page { margin: 12mm; }
    * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    body { background: #FFFFFF; font-size: 12px; }
    .wrap { padding: 8px 0; max-width: none; }
    main { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; padding-top: 8px !important; align-items: start; }
    .card { box-shadow: none; break-inside: avoid; padding: 12px; }
    .kpis { grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }
    .kpi .value { font-size: 1.125rem; }
    .kpi h2, .kpi .sub, .lead, .legend, .why, .foot, .name, .break_even, footer { font-size: .6875rem; }
    h2 { font-size: .875rem; }
    .row { grid-template-columns: 96px minmax(0, 1fr); gap: 8px; }
    .name small, .tip_value { font-size: .625rem; }
    .mark { width: calc(var(--w) * (100% - 64px)); }
    .curve text { font-size: 15px; }
    .heat { font-size: .6875rem; }
    .heat thead th { font-size: .5625rem; padding: 2px 0; }
    .heat tr > :first-child { width: 120px; }
    .heat tbody td { height: 30px; }
    .toggle, details, #tooltip { display: none !important; }
    .js .matrix.off { display: block; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

SCRIPT = """
"use strict";
document.documentElement.classList.add("js");
const tooltip = document.getElementById("tooltip");
const cross = document.getElementById("cross");
const curve = document.querySelector(".curve");
const stops = [...curve.querySelectorAll(".stop")];

function showTip(node, x, y) {
  tooltip.querySelector("strong").textContent = node.dataset.value;
  tooltip.querySelector("span").textContent = node.dataset.label;
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  const left = Math.min(Math.max(x + 14, 8), window.innerWidth - box.width - 8);
  const top = y + box.height + 22 > window.innerHeight ? y - box.height - 12 : y + 14;
  tooltip.style.left = left + "px";
  tooltip.style.top = Math.max(top, 8) + "px";
}

function hideTip() {
  tooltip.hidden = true;
  cross.style.opacity = 0;
  stops.forEach((stop) => stop.classList.remove("on"));
}

function markStop(stop) {
  stops.forEach((other) => other.classList.toggle("on", other === stop));
  cross.setAttribute("x1", stop.getAttribute("cx"));
  cross.setAttribute("x2", stop.getAttribute("cx"));
  cross.style.opacity = 1;
}

document.addEventListener("pointermove", (event) => {
  if (event.target.closest(".curve")) {
    const box = curve.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width * curve.viewBox.baseVal.width;
    if (x < Number(curve.dataset.left) - 8 || x > Number(curve.dataset.right) + 8) return hideTip();
    const distance = (stop) => Math.abs(Number(stop.getAttribute("cx")) - x);
    const nearest = stops.reduce((best, stop) => (distance(stop) < distance(best) ? stop : best));
    markStop(nearest);
    return showTip(nearest, event.clientX, event.clientY);
  }
  const node = event.target.closest("[data-value]");
  if (node) showTip(node, event.clientX, event.clientY);
  else hideTip();
});
function showFocused() {
  const node = document.activeElement.closest("[data-value]");
  if (!node) return hideTip();
  const box = node.getBoundingClientRect();
  if (node.classList.contains("stop")) markStop(node);
  showTip(node, box.left + box.width / 2, box.bottom);
}
document.addEventListener("focusin", showFocused);
document.addEventListener("focusout", hideTip);
document.addEventListener("pointerleave", hideTip);
window.addEventListener("scroll", showFocused, { passive: true });
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideTip();
});

/* One tab stop per chart: the arrow keys move inside it. */
for (const group of document.querySelectorAll("[data-roving]")) {
  const items = [...group.querySelectorAll("[data-value]")];
  const columns = Number(group.dataset.roving);
  group.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns }[event.key];
    const next = items[items.indexOf(event.target) + step];
    if (!step || !next) return;
    event.preventDefault();
    event.target.setAttribute("tabindex", "-1");
    next.setAttribute("tabindex", "0");
    next.focus();
  });
}

function showMatrix(side) {
  for (const button of document.querySelectorAll("[data-show]")) {
    button.setAttribute("aria-pressed", String(button.dataset.show === side));
    document.getElementById("matrix_" + button.dataset.show).classList.toggle("off", button.dataset.show !== side);
  }
}
for (const button of document.querySelectorAll("[data-show]")) {
  button.addEventListener("click", () => showMatrix(button.dataset.show));
}
showMatrix("router");
"""


def render(d: dict) -> str:
    val = d["validation"]
    data = json.dumps(d, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")
    subtitle = f"Summary for management · validation {month_range(val['first_day'], val['last_day'])}, {val['rows']:,} requests"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Kestrel Home routing dashboard</title>",
            f"<style>{STYLE}</style>",
            "</head>",
            "<body>",
            f'<header><div class="wrap"><h1>Kestrel Home · Routing dashboard</h1><p>{escape(subtitle)}</p></div></header>',
            '<main class="wrap">',
            kpis(d),
            coverage_panel(d),
            recall_panel(d),
            heatmap_panel(d),
            headcount_panel(d),
            cost_panel(d),
            footer(d),
            "</main>",
            '<div id="tooltip" role="status" hidden><strong></strong><span></span></div>',
            f'<script type="application/json" id="data">{data}</script>',
            f"<script>{SCRIPT}</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def build(out_path: Path = config.REPORTS_DIR / HTML_NAME, reports_dir: Path = config.REPORTS_DIR) -> Path:
    """Write the dashboard to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(load_data(reports_dir)), encoding="utf-8", newline="\n")
    return out_path


if __name__ == "__main__":
    print(f"wrote {build()}")
