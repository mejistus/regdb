#!/usr/bin/env python3
"""Build a self-contained RegDB hyperparameter search report."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def status_for(row: dict[str, Any]) -> str:
    return "complete" if row["complete_trials"] == row["expected_trials"] else "incomplete"


def trial_status(stage: dict[str, Any]) -> str:
    if stage.get("complete"):
        return "complete"
    if stage.get("exists"):
        return "started"
    return "missing"


def stage_metric(stage: dict[str, Any], key: str) -> float | None:
    value = stage.get(key)
    return float(value) if isinstance(value, int | float) else None


def flatten_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trial_rows: list[dict[str, Any]] = []
    for candidate in rows:
        for trial in candidate["trials"]:
            stage1 = trial["stage1"]
            stage2 = trial["stage2"]
            trial_rows.append(
                {
                    "tag": candidate["tag"],
                    "samples": candidate["samples"],
                    "dropout": candidate["dropout"],
                    "source": candidate["source"],
                    "candidate_status": status_for(candidate),
                    "trial": trial["trial"],
                    "stage1_status": trial_status(stage1),
                    "stage1_best_rank1": stage_metric(stage1, "best_rank1"),
                    "stage1_best_map": stage_metric(stage1, "best_map"),
                    "stage1_best_epoch": stage1.get("best_epoch"),
                    "stage1_log_path": stage1.get("log_path"),
                    "stage2_status": trial_status(stage2),
                    "stage2_best_rank1": stage_metric(stage2, "best_rank1"),
                    "stage2_best_map": stage_metric(stage2, "best_map"),
                    "stage2_best_epoch": stage2.get("best_epoch"),
                    "stage2_log_path": stage2.get("log_path"),
                }
            )
    return trial_rows


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete_rows = [
        row
        for row in rows
        if row["complete_trials"] == row["expected_trials"] and row["mean_best_rank1"] is not None
    ]
    complete_sorted = sorted(
        complete_rows,
        key=lambda row: (row["mean_best_rank1"], row["mean_best_map"] or -1),
        reverse=True,
    )
    rank_by_tag = {row["tag"]: index for index, row in enumerate(complete_sorted, 1)}
    for row in rows:
        stage2_scores = [
            trial["stage2"].get("best_rank1")
            for trial in row["trials"]
            if trial["stage2"].get("complete") and trial["stage2"].get("best_rank1") is not None
        ]
        row["status"] = status_for(row)
        row["official_rank"] = rank_by_tag.get(row["tag"])
        row["std_best_rank1"] = pstdev(stage2_scores) if len(stage2_scores) > 1 else None
        row["min_best_rank1"] = min(stage2_scores) if stage2_scores else None
        row["trial_best_rank1_values"] = stage2_scores
    return rows


def build_payload(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = enrich_rows(rows)
    trial_rows = flatten_trials(rows)
    complete_rows = [
        row
        for row in rows
        if row["status"] == "complete" and row["mean_best_rank1"] is not None
    ]
    best = (
        max(complete_rows, key=lambda row: (row["mean_best_rank1"], row["mean_best_map"] or -1))
        if complete_rows
        else None
    )
    complete_trial_count = sum(row["complete_trials"] for row in rows)
    expected_trial_count = sum(row["expected_trials"] for row in rows)
    missing_stage2 = sum(1 for row in trial_rows if row["stage2_status"] == "missing")
    started_stage2 = sum(1 for row in trial_rows if row["stage2_status"] == "started")
    all_complete = all(row["status"] == "complete" for row in rows)
    incomplete = [row["tag"] for row in rows if row["status"] != "complete"]
    validation = {
        "candidate_count": len(rows),
        "complete_candidate_count": len(complete_rows),
        "incomplete_candidate_count": len(rows) - len(complete_rows),
        "expected_trial_count": expected_trial_count,
        "complete_trial_count": complete_trial_count,
        "missing_stage2_count": missing_stage2,
        "started_stage2_count": started_stage2,
        "all_candidates_complete": all_complete,
        "incomplete_candidates": incomplete,
        "best_scope": "all planned candidates" if all_complete else "completed 10-trial candidates only",
    }
    if complete_rows:
        validation["complete_mean_rank1_mean"] = mean(row["mean_best_rank1"] for row in complete_rows)
    generated_at = args.generated_at or datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return {
        "manifest": {
            "title": "RegDB PCLHD+MDUE+CGCF Hyperparameter Search",
            "dataset": "RegDB visible-to-thermal setting",
            "method": "PCLHD + MDUE + CGCF",
            "metric_policy": "Official score is the arithmetic mean of stage2 best Rank-1 across 10 RegDB trials. Mean mAP is used as the tie-breaker.",
            "script": "tools/build_regdb_hp_search_report.py",
            "summary_json": str(args.summary),
            "generated_at": generated_at,
            "status_note": args.status_note,
            "failure_note": args.failure_note,
        },
        "best": best,
        "validation": validation,
        "summary_rows": rows,
        "trial_rows": trial_rows,
    }


def html_template(payload: dict[str, Any]) -> str:
    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(payload["manifest"]["title"])
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d9dee7;
      --text: #1f2937;
      --muted: #64748b;
      --strong: #0f172a;
      --accent: #2563eb;
      --warn: #b45309;
      --bad: #b91c1c;
      --good: #047857;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1440px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: var(--strong);
      font-size: 26px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      color: var(--strong);
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; }}
    .subhead {{
      color: var(--muted);
      max-width: 980px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .card {{
      min-height: 104px;
      padding: 14px;
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .card .value {{
      margin-top: 8px;
      color: var(--strong);
      font-size: 24px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }}
    .card .detail {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .panel {{
      margin-top: 16px;
      padding: 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1.4fr repeat(3, minmax(150px, .7fr)) auto auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 12px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    input, select, button {{
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 13px;
    }}
    input {{ padding: 0 10px; }}
    select {{ padding: 4px 8px; }}
    select[multiple] {{ min-height: 70px; }}
    button {{
      padding: 0 12px;
      cursor: pointer;
      font-weight: 650;
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    .note {{
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid #f1d195;
      border-radius: 8px;
      background: #fff8ea;
      color: #7c4a03;
      font-size: 13px;
    }}
    .note.strong {{
      border-color: #f0aaaa;
      background: #fff1f1;
      color: #7f1d1d;
    }}
    .table-wrap {{
      width: 100%;
      max-height: 68vh;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1040px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e7ebf1;
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      color: #334155;
      font-size: 12px;
      font-weight: 760;
      cursor: pointer;
      user-select: none;
    }}
    tbody tr:hover td {{ background: #f9fbff; }}
    td.num, th.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #f8fafc;
      font-size: 12px;
      font-weight: 720;
    }}
    .pill.complete {{ color: var(--good); border-color: #a7f3d0; background: #ecfdf5; }}
    .pill.incomplete, .pill.started {{ color: var(--warn); border-color: #fed7aa; background: #fff7ed; }}
    .pill.missing {{ color: var(--bad); border-color: #fecaca; background: #fef2f2; }}
    .best-row td {{
      box-shadow: inset 3px 0 0 var(--accent);
      background: #f8fbff;
    }}
    .methodology {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .methodology div {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .methodology h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--strong);
    }}
    .methodology ul {{
      margin: 0;
      padding-left: 18px;
      color: #475569;
    }}
    .methodology li {{ margin: 5px 0; }}
    @media (max-width: 980px) {{
      main {{ width: min(100vw - 20px, 1440px); }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
      .methodology {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 640px) {{
      .grid, .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>RegDB PCLHD+MDUE+CGCF 超参数搜索报告</h1>
  <p class="subhead">以 PCLHD + MDUE + CGCF 为 ours，按 RegDB 10-trial 的 stage2 最佳 Rank-1 均值排序，mAP 均值作为并列时的辅助指标。未完成候选不进入官方最佳排序。</p>

  <section class="grid" id="cards"></section>

  <section class="panel">
    <h2>结论与方法约束</h2>
    <div id="statusNotes"></div>
    <div class="methodology">
      <div>
        <h3>评分规则</h3>
        <ul>
          <li>每个候选必须完成 10 个 RegDB trial，才进入正式排名。</li>
          <li>每个 trial 使用 stage2 日志中的 best Rank-1 和 best mAP。</li>
          <li>主指标为 10-trial mean Rank-1；mAP 只作为并列时的 tie-break。</li>
        </ul>
      </div>
      <div>
        <h3>候选含义</h3>
        <ul>
          <li><span class="mono">S</span> 是 MDUE 的 MC-Dropout 采样次数。</li>
          <li><span class="mono">p</span> 是 MDUE 的 dropout 概率。</li>
          <li>CGCF 在 stage2 中进行 confidence-guided cross-modal center fusion。</li>
        </ul>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>候选配置汇总</h2>
    <div class="controls">
      <label>Search <input id="summarySearch" type="search" placeholder="tag, source, status"></label>
      <label>Status <select id="statusFilter" multiple></select></label>
      <label>Source <select id="sourceFilter" multiple></select></label>
      <label>S <select id="sampleFilter" multiple></select></label>
      <button id="clearFilters">Clear</button>
      <button class="primary" id="downloadSummary">CSV</button>
    </div>
    <div class="table-wrap">
      <table id="summaryTable"></table>
    </div>
  </section>

  <section class="panel">
    <h2>Trial 级明细</h2>
    <div class="controls">
      <label>Search <input id="trialSearch" type="search" placeholder="tag, trial, log path"></label>
      <label>Stage2 status <select id="trialStatusFilter" multiple></select></label>
      <label>Candidate <select id="tagFilter" multiple></select></label>
      <label>Source <select id="trialSourceFilter" multiple></select></label>
      <button id="clearTrialFilters">Clear</button>
      <button class="primary" id="downloadTrials">CSV</button>
    </div>
    <div class="table-wrap">
      <table id="trialTable"></table>
    </div>
  </section>
</main>

<script>
const PAYLOAD = __PAYLOAD__;
const summaryColumns = [
  ["official_rank", "Rank", "num"],
  ["tag", "Tag", ""],
  ["status", "Status", ""],
  ["samples", "S", "num"],
  ["dropout", "p", "num"],
  ["complete_trials", "Done", "num"],
  ["mean_best_rank1", "Mean Rank-1", "metric"],
  ["mean_best_map", "Mean mAP", "metric"],
  ["std_best_rank1", "Std Rank-1", "num"],
  ["min_best_rank1", "Min Rank-1", "metric"],
  ["max_best_rank1", "Max Rank-1", "metric"],
  ["source", "Source", ""],
];
const trialColumns = [
  ["tag", "Tag", ""],
  ["trial", "Trial", "num"],
  ["samples", "S", "num"],
  ["dropout", "p", "num"],
  ["stage1_status", "Stage1", ""],
  ["stage1_best_rank1", "S1 R1", "metric"],
  ["stage1_best_map", "S1 mAP", "metric"],
  ["stage2_status", "Stage2", ""],
  ["stage2_best_rank1", "S2 R1", "metric"],
  ["stage2_best_map", "S2 mAP", "metric"],
  ["stage2_best_epoch", "S2 best epoch", "num"],
  ["source", "Source", ""],
  ["stage2_log_path", "Stage2 log", "mono"],
];
let summarySort = {key: "official_rank", dir: "asc"};
let trialSort = {key: "tag", dir: "asc"};

function pct(value) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "-" : Number(value).toFixed(2);
}
function text(value) {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}
function rank(value) {
  return value === null || value === undefined ? "not ranked" : String(value);
}
function metricStyle(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  const t = Math.max(0, Math.min(1, Number(value) / 100));
  const hue = 8 + 132 * t;
  return `background:hsl(${hue} 78% 92%); color:#172033;`;
}
function escapeHtml(value) {
  return text(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
}
function selectedValues(id) {
  return Array.from(document.getElementById(id).selectedOptions).map(o => o.value);
}
function setOptions(id, values, labels = {}) {
  const node = document.getElementById(id);
  node.innerHTML = values.map(value => `<option value="${escapeHtml(value)}">${escapeHtml(labels[value] || value)}</option>`).join("");
}
function compare(a, b) {
  if (a === null || a === undefined) return 1;
  if (b === null || b === undefined) return -1;
  const an = Number(a);
  const bn = Number(b);
  if (!Number.isNaN(an) && !Number.isNaN(bn)) return an - bn;
  return String(a).localeCompare(String(b));
}
function sortRows(rows, state) {
  return [...rows].sort((a, b) => {
    const result = compare(a[state.key], b[state.key]);
    return state.dir === "asc" ? result : -result;
  });
}
function statusPill(value) {
  return `<span class="pill ${escapeHtml(value)}">${escapeHtml(value)}</span>`;
}
function cellHtml(row, column) {
  const [key, , kind] = column;
  const value = row[key];
  if (key === "official_rank") return rank(value);
  if (key === "complete_trials") return `${row.complete_trials}/${row.expected_trials}`;
  if (key.endsWith("_status") || key === "status") return statusPill(value);
  if (kind === "metric") return `<span>${pct(value)}</span>`;
  if (kind === "num") return escapeHtml(pct(value));
  if (kind === "mono") return `<span class="mono" title="${escapeHtml(value)}">${escapeHtml(value)}</span>`;
  return escapeHtml(value);
}
function renderTable(id, columns, rows, state, onSort, rowClass) {
  const table = document.getElementById(id);
  const headers = columns.map(([key, label, kind]) => {
    const arrow = state.key === key ? (state.dir === "asc" ? " ▲" : " ▼") : "";
    return `<th class="${kind === "num" || kind === "metric" ? "num" : ""}" data-key="${key}">${label}${arrow}</th>`;
  }).join("");
  const body = rows.map(row => {
    const cells = columns.map(column => {
      const kind = column[2];
      const style = kind === "metric" ? metricStyle(row[column[0]]) : "";
      const cls = kind === "num" || kind === "metric" ? "num" : "";
      return `<td class="${cls}" style="${style}">${cellHtml(row, column)}</td>`;
    }).join("");
    return `<tr class="${rowClass ? rowClass(row) : ""}">${cells}</tr>`;
  }).join("");
  table.innerHTML = `<thead><tr>${headers}</tr></thead><tbody>${body || `<tr><td colspan="${columns.length}">No rows match the current filters.</td></tr>`}</tbody>`;
  table.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => onSort(th.dataset.key));
  });
}
function includesAny(value, selected) {
  return selected.length === 0 || selected.includes(String(value));
}
function renderCards() {
  const v = PAYLOAD.validation;
  const best = PAYLOAD.best;
  const cards = [
    ["Best completed config", best ? `${best.tag}` : "-", best ? `S=${best.samples}, p=${Number(best.dropout).toFixed(2)}; mean R1 ${pct(best.mean_best_rank1)} / mAP ${pct(best.mean_best_map)}` : "No completed 10-trial candidate"],
    ["Completed candidates", `${v.complete_candidate_count}/${v.candidate_count}`, `${v.complete_trial_count}/${v.expected_trial_count} stage2 trials complete`],
    ["Ranking scope", v.best_scope, v.all_candidates_complete ? "All planned candidates completed" : `Incomplete: ${v.incomplete_candidates.join(", ")}`],
    ["Generated", PAYLOAD.manifest.generated_at, PAYLOAD.manifest.dataset],
  ];
  document.getElementById("cards").innerHTML = cards.map(([label, value, detail]) => `
    <article class="card">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(value)}</div>
      <div class="detail">${escapeHtml(detail)}</div>
    </article>`).join("");
}
function renderNotes() {
  const notes = [];
  if (PAYLOAD.manifest.status_note) notes.push(`<div class="note">${escapeHtml(PAYLOAD.manifest.status_note)}</div>`);
  if (PAYLOAD.manifest.failure_note) notes.push(`<div class="note strong">${escapeHtml(PAYLOAD.manifest.failure_note)}</div>`);
  if (!PAYLOAD.validation.all_candidates_complete) {
    notes.push(`<div class="note strong">Search is not fully complete. Official best is selected only from candidates with 10/10 completed trials.</div>`);
  }
  document.getElementById("statusNotes").innerHTML = notes.join("");
}
function filteredSummaryRows() {
  const query = document.getElementById("summarySearch").value.trim().toLowerCase();
  const statuses = selectedValues("statusFilter");
  const sources = selectedValues("sourceFilter");
  const samples = selectedValues("sampleFilter");
  return PAYLOAD.summary_rows.filter(row => {
    const haystack = [row.tag, row.source, row.status, row.samples, row.dropout].join(" ").toLowerCase();
    return (!query || haystack.includes(query))
      && includesAny(row.status, statuses)
      && includesAny(row.source, sources)
      && includesAny(row.samples, samples);
  });
}
function filteredTrialRows() {
  const query = document.getElementById("trialSearch").value.trim().toLowerCase();
  const statuses = selectedValues("trialStatusFilter");
  const tags = selectedValues("tagFilter");
  const sources = selectedValues("trialSourceFilter");
  return PAYLOAD.trial_rows.filter(row => {
    const haystack = [row.tag, row.trial, row.stage2_status, row.source, row.stage2_log_path].join(" ").toLowerCase();
    return (!query || haystack.includes(query))
      && includesAny(row.stage2_status, statuses)
      && includesAny(row.tag, tags)
      && includesAny(row.source, sources);
  });
}
function renderSummary() {
  renderTable("summaryTable", summaryColumns, sortRows(filteredSummaryRows(), summarySort), summarySort, key => {
    summarySort = {key, dir: summarySort.key === key && summarySort.dir === "asc" ? "desc" : "asc"};
    renderSummary();
  }, row => row.official_rank === 1 ? "best-row" : "");
}
function renderTrials() {
  renderTable("trialTable", trialColumns, sortRows(filteredTrialRows(), trialSort), trialSort, key => {
    trialSort = {key, dir: trialSort.key === key && trialSort.dir === "asc" ? "desc" : "asc"};
    renderTrials();
  });
}
function csvEscape(value) {
  const s = text(value);
  return /[",\\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}
function downloadCsv(name, columns, rows) {
  const lines = [columns.map(c => csvEscape(c[1])).join(",")];
  rows.forEach(row => lines.push(columns.map(c => csvEscape(row[c[0]])).join(",")));
  const blob = new Blob([lines.join("\\n")], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function init() {
  const summaryRows = PAYLOAD.summary_rows;
  const trialRows = PAYLOAD.trial_rows;
  setOptions("statusFilter", [...new Set(summaryRows.map(r => r.status))]);
  setOptions("sourceFilter", [...new Set(summaryRows.map(r => r.source))]);
  setOptions("sampleFilter", [...new Set(summaryRows.map(r => String(r.samples)))]);
  setOptions("trialStatusFilter", [...new Set(trialRows.map(r => r.stage2_status))]);
  setOptions("tagFilter", [...new Set(trialRows.map(r => r.tag))]);
  setOptions("trialSourceFilter", [...new Set(trialRows.map(r => r.source))]);
  ["summarySearch", "statusFilter", "sourceFilter", "sampleFilter"].forEach(id => document.getElementById(id).addEventListener("input", renderSummary));
  ["trialSearch", "trialStatusFilter", "tagFilter", "trialSourceFilter"].forEach(id => document.getElementById(id).addEventListener("input", renderTrials));
  document.getElementById("clearFilters").addEventListener("click", () => {
    document.getElementById("summarySearch").value = "";
    ["statusFilter", "sourceFilter", "sampleFilter"].forEach(id => Array.from(document.getElementById(id).options).forEach(o => o.selected = false));
    renderSummary();
  });
  document.getElementById("clearTrialFilters").addEventListener("click", () => {
    document.getElementById("trialSearch").value = "";
    ["trialStatusFilter", "tagFilter", "trialSourceFilter"].forEach(id => Array.from(document.getElementById(id).options).forEach(o => o.selected = false));
    renderTrials();
  });
  document.getElementById("downloadSummary").addEventListener("click", () => downloadCsv("regdb_hp_search_summary.csv", summaryColumns, filteredSummaryRows()));
  document.getElementById("downloadTrials").addEventListener("click", () => downloadCsv("regdb_hp_search_trials.csv", trialColumns, filteredTrialRows()));
  renderCards();
  renderNotes();
  renderSummary();
  renderTrials();
}
init();
</script>
</body>
</html>
"""
    return template.replace("__TITLE__", title).replace("__PAYLOAD__", embedded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("logs/regdb_ours_hp10_search_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("htmls/hp_search.html"))
    parser.add_argument("--generated-at", default="")
    parser.add_argument("--status-note", default="")
    parser.add_argument("--failure-note", default="")
    args = parser.parse_args()

    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    payload = build_payload(rows, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_template(payload), encoding="utf-8")
    best = payload["best"]
    if best:
        print(
            "best_completed="
            f"{best['tag']} S={best['samples']} p={best['dropout']:.2f} "
            f"mean_R1={fmt(best['mean_best_rank1'])} mean_mAP={fmt(best['mean_best_map'])}"
        )
    print(f"wrote {args.out}")
    validation = payload["validation"]
    print(
        "validation "
        f"candidates={validation['candidate_count']} "
        f"complete_candidates={validation['complete_candidate_count']} "
        f"stage2_trials={validation['complete_trial_count']}/{validation['expected_trial_count']} "
        f"all_complete={validation['all_candidates_complete']}"
    )


if __name__ == "__main__":
    main()
