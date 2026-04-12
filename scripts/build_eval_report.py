#!/usr/bin/env python3
"""Build a side-by-side comparison sheet (HTML + CSV) from eval_results.json.

Joins eval results with gold_100.jsonl by ticket_id and renders:
  - Top: metrics summary (base vs fine-tuned)
  - Bottom: per-ticket detail with color-coded cells

Usage:
    python scripts/build_eval_report.py \
        --eval-file eval_results.json \
        --gold-file data/sft-data/gold/gold_100.jsonl \
        --out-dir reports/
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tid = t.get("ticket_id")
            if tid:
                out[tid] = t
    return out


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _norm(v) -> str:
    if v is None or v == "":
        return "—"
    return str(v).strip()


def _eq(a, b) -> bool:
    def n(v):
        if v is None or v == "":
            return ""
        return str(v).strip().lower()
    return n(a) == n(b)


def pred(result: dict, field: str):
    p = result.get("parsed") or {}
    return p.get(field)


def pred_tools(result: dict) -> str:
    names = result.get("tool_call_names") or []
    return ", ".join(names) if names else "—"


def gold_tool_names(ticket: dict) -> str:
    calls = ticket.get("gold_tool_calls") or []
    names = [tc.get("name", "") for tc in calls if isinstance(tc, dict)]
    return ", ".join(n for n in names if n) or "—"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kiki SLM — Base vs Fine-tuned Eval</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #222; }
  h1 { margin-bottom: 4px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 13px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 12px; }
  th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }
  th { background: #f4f4f4; font-weight: 600; position: sticky; top: 0; }
  .metrics td, .metrics th { text-align: right; }
  .metrics td:first-child, .metrics th:first-child { text-align: left; font-weight: 600; }
  .good { background: #d4edda; color: #155724; }
  .bad { background: #f8d7da; color: #721c24; }
  .neutral { background: #fff; }
  .muted { color: #888; font-style: italic; }
  .msg { max-width: 420px; font-size: 11px; color: #555; max-height: 80px; overflow: hidden; }
  .tid { font-family: "SF Mono", Menlo, monospace; white-space: nowrap; font-size: 11px; }
  .diff-pos { color: #155724; font-weight: 600; }
  .diff-neg { color: #721c24; font-weight: 600; }
  .section { font-size: 18px; margin: 24px 0 8px 0; font-weight: 600; }
</style>
</head>
<body>
<h1>Kiki SLM — Base vs Fine-tuned</h1>
<div class="subtitle">__SUBTITLE__</div>
"""


def render_metrics_table(base: dict, ft: dict) -> str:
    rows = [
        ("Intent accuracy", "intent_accuracy", "pct"),
        ("Urgency accuracy", "urgency_accuracy", "pct"),
        ("is_valid accuracy", "is_valid_accuracy", "pct"),
        ("Rejection-type accuracy", "rejection_accuracy", "pct"),
        ("Resolution-type accuracy", "resolution_accuracy", "pct"),
        ("Team accuracy", "team_accuracy", "pct"),
        ("Tool-name F1", "tool_name_f1", "pct"),
        ("JSON parse rate", "json_parse_rate", "pct"),
        ("Avg latency (s)", "avg_latency_s", "num"),
        ("Avg turns", "avg_turns", "num"),
    ]
    html_rows = []
    for label, key, fmt in rows:
        bv = base.get(key, 0) or 0
        fv = ft.get(key, 0) or 0
        diff = fv - bv
        if fmt == "pct":
            bs = f"{bv * 100:.1f}%"
            fs = f"{fv * 100:.1f}%"
            ds = f"{diff * 100:+.1f} pp"
        else:
            bs = f"{bv:.2f}"
            fs = f"{fv:.2f}"
            ds = f"{diff:+.2f}"
        diff_cls = "diff-pos" if diff > 0 else ("diff-neg" if diff < 0 else "")
        # For latency lower = better, flip color
        if key == "avg_latency_s":
            diff_cls = "diff-pos" if diff < 0 else ("diff-neg" if diff > 0 else "")
        html_rows.append(
            f"<tr><td>{label}</td><td>{bs}</td><td>{fs}</td>"
            f'<td class="{diff_cls}">{ds}</td></tr>'
        )
    body = "\n".join(html_rows)
    return f"""
<div class="section">Summary metrics ({base.get('total', 0)} gold tickets)</div>
<table class="metrics">
  <thead><tr><th>Metric</th><th>Base Qwen3</th><th>Fine-tuned</th><th>Δ</th></tr></thead>
  <tbody>
    {body}
  </tbody>
</table>
"""


def cell(value, gold_value, highlight=True) -> str:
    text = html.escape(_norm(value))
    if not highlight or gold_value is None:
        return f'<td>{text}</td>'
    if _eq(value, gold_value):
        return f'<td class="good">{text}</td>'
    return f'<td class="bad">{text}</td>'


def render_detail_table(base_results: list, ft_results: list, gold: dict) -> str:
    headers = [
        "Ticket", "Customer message",
        "Gold intent", "Base", "FT",
        "Gold urgency", "Base", "FT",
        "Gold is_valid", "Base", "FT",
        "Gold rejection_type", "Base", "FT",
        "Gold resolution_type", "Base", "FT",
        "Gold team", "Base", "FT",
        "Gold tools", "Base tools", "FT tools",
        "Base turns", "FT turns",
        "Base parse", "FT parse",
    ]
    th = "".join(f"<th>{h}</th>" for h in headers)

    rows = []
    for br, fr in zip(base_results, ft_results):
        tid = br.get("ticket_id", "?")
        t = gold.get(tid, {})
        msg = (t.get("customer_message") or "")[:320]
        if len(t.get("customer_message") or "") > 320:
            msg += "…"

        bp = br.get("parsed") or {}
        fp = fr.get("parsed") or {}
        b_parsed_ok = br.get("parsed") is not None
        f_parsed_ok = fr.get("parsed") is not None

        r = []
        r.append(f'<td class="tid">{html.escape(tid)}</td>')
        r.append(f'<td class="msg">{html.escape(msg)}</td>')

        # Intent
        gi = t.get("gold_intent")
        r.append(f'<td>{html.escape(_norm(gi))}</td>')
        r.append(cell(bp.get("intent") if b_parsed_ok else None, gi))
        r.append(cell(fp.get("intent") if f_parsed_ok else None, gi))

        # Urgency
        gu = t.get("gold_urgency")
        r.append(f'<td>{html.escape(_norm(gu))}</td>')
        r.append(cell(bp.get("urgency") if b_parsed_ok else None, gu))
        r.append(cell(fp.get("urgency") if f_parsed_ok else None, gu))

        # is_valid
        gv = t.get("gold_is_valid")
        r.append(f'<td>{html.escape(_norm(gv))}</td>')
        r.append(cell(bp.get("is_valid") if b_parsed_ok else None, gv))
        r.append(cell(fp.get("is_valid") if f_parsed_ok else None, gv))

        # rejection_type
        grj = t.get("gold_rejection_type")
        r.append(f'<td>{html.escape(_norm(grj))}</td>')
        r.append(cell(bp.get("rejection_type") if b_parsed_ok else None, grj))
        r.append(cell(fp.get("rejection_type") if f_parsed_ok else None, grj))

        # resolution_type
        gres = t.get("gold_resolution_type")
        r.append(f'<td>{html.escape(_norm(gres))}</td>')
        r.append(cell(bp.get("resolution_type") if b_parsed_ok else None, gres))
        r.append(cell(fp.get("resolution_type") if f_parsed_ok else None, gres))

        # team
        gt = t.get("gold_team")
        r.append(f'<td>{html.escape(_norm(gt))}</td>')
        r.append(cell(bp.get("team") if b_parsed_ok else None, gt))
        r.append(cell(fp.get("team") if f_parsed_ok else None, gt))

        # tools
        r.append(f'<td>{html.escape(gold_tool_names(t))}</td>')
        r.append(f'<td>{html.escape(pred_tools(br))}</td>')
        r.append(f'<td>{html.escape(pred_tools(fr))}</td>')

        # turns
        r.append(f'<td>{br.get("turns", 0)}</td>')
        r.append(f'<td>{fr.get("turns", 0)}</td>')

        # parse status
        r.append(
            '<td class="good">ok</td>' if b_parsed_ok
            else '<td class="bad">fail</td>'
        )
        r.append(
            '<td class="good">ok</td>' if f_parsed_ok
            else '<td class="bad">fail</td>'
        )

        rows.append("<tr>" + "".join(r) + "</tr>")

    body = "\n".join(rows)
    return f"""
<div class="section">Per-ticket comparison (green = matches gold, red = mismatch)</div>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>
    {body}
  </tbody>
</table>
"""


# ---------------------------------------------------------------------------
# CSV rendering
# ---------------------------------------------------------------------------

CSV_HEADERS = [
    "ticket_id", "customer_message",
    "gold_intent", "base_intent", "ft_intent", "base_intent_ok", "ft_intent_ok",
    "gold_urgency", "base_urgency", "ft_urgency", "base_urgency_ok", "ft_urgency_ok",
    "gold_is_valid", "base_is_valid", "ft_is_valid", "base_is_valid_ok", "ft_is_valid_ok",
    "gold_rejection_type", "base_rejection_type", "ft_rejection_type",
    "gold_resolution_type", "base_resolution_type", "ft_resolution_type",
    "base_resolution_ok", "ft_resolution_ok",
    "gold_team", "base_team", "ft_team", "base_team_ok", "ft_team_ok",
    "gold_tools", "base_tools", "ft_tools",
    "base_turns", "ft_turns",
    "base_latency_s", "ft_latency_s",
    "base_parse_ok", "ft_parse_ok",
]


def write_csv(path: Path, base_results: list, ft_results: list, gold: dict) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        for br, fr in zip(base_results, ft_results):
            tid = br.get("ticket_id", "?")
            t = gold.get(tid, {})
            bp = br.get("parsed") or {}
            fp = fr.get("parsed") or {}
            b_ok = br.get("parsed") is not None
            f_ok = fr.get("parsed") is not None

            row = [
                tid,
                (t.get("customer_message") or "")[:500],
                _norm(t.get("gold_intent")),
                _norm(bp.get("intent") if b_ok else None),
                _norm(fp.get("intent") if f_ok else None),
                _eq(bp.get("intent"), t.get("gold_intent")) if b_ok else False,
                _eq(fp.get("intent"), t.get("gold_intent")) if f_ok else False,
                _norm(t.get("gold_urgency")),
                _norm(bp.get("urgency") if b_ok else None),
                _norm(fp.get("urgency") if f_ok else None),
                _eq(bp.get("urgency"), t.get("gold_urgency")) if b_ok else False,
                _eq(fp.get("urgency"), t.get("gold_urgency")) if f_ok else False,
                _norm(t.get("gold_is_valid")),
                _norm(bp.get("is_valid") if b_ok else None),
                _norm(fp.get("is_valid") if f_ok else None),
                (bp.get("is_valid") == t.get("gold_is_valid")) if b_ok else False,
                (fp.get("is_valid") == t.get("gold_is_valid")) if f_ok else False,
                _norm(t.get("gold_rejection_type")),
                _norm(bp.get("rejection_type") if b_ok else None),
                _norm(fp.get("rejection_type") if f_ok else None),
                _norm(t.get("gold_resolution_type")),
                _norm(bp.get("resolution_type") if b_ok else None),
                _norm(fp.get("resolution_type") if f_ok else None),
                _eq(bp.get("resolution_type"), t.get("gold_resolution_type")) if b_ok else False,
                _eq(fp.get("resolution_type"), t.get("gold_resolution_type")) if f_ok else False,
                _norm(t.get("gold_team")),
                _norm(bp.get("team") if b_ok else None),
                _norm(fp.get("team") if f_ok else None),
                _eq(bp.get("team"), t.get("gold_team")) if b_ok else False,
                _eq(fp.get("team"), t.get("gold_team")) if f_ok else False,
                gold_tool_names(t),
                pred_tools(br),
                pred_tools(fr),
                br.get("turns", 0),
                fr.get("turns", 0),
                f"{br.get('latency', 0):.3f}",
                f"{fr.get('latency', 0):.3f}",
                b_ok,
                f_ok,
            ]
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="eval_results.json")
    parser.add_argument("--gold-file", default="data/sft-data/gold/gold_100.jsonl")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    with open(args.eval_file) as f:
        data = json.load(f)

    gold = load_gold(args.gold_file)
    base_results = data.get("base_results", [])
    ft_results = data.get("ft_results", [])
    base_metrics = data.get("base_metrics", {})
    ft_metrics = data.get("ft_metrics", {})
    cfg = data.get("config", {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # HTML
    subtitle = (
        f"Adapter: {cfg.get('adapter_path', '?')}"
        f" | Base: {cfg.get('base_model', '?')}"
        f" | max_turns={cfg.get('max_turns', '?')}"
    )
    html_out = HTML_HEAD.replace("__SUBTITLE__", html.escape(subtitle))
    html_out += render_metrics_table(base_metrics, ft_metrics)
    html_out += render_detail_table(base_results, ft_results, gold)
    html_out += "</body></html>"

    html_path = out_dir / "eval_comparison.html"
    html_path.write_text(html_out)

    # CSV
    csv_path = out_dir / "eval_comparison.csv"
    write_csv(csv_path, base_results, ft_results, gold)

    print(f"HTML: {html_path}")
    print(f"CSV:  {csv_path}")
    print(f"  {len(base_results)} tickets joined against {len(gold)} gold entries")


if __name__ == "__main__":
    main()
