"""Render a self-contained paper-results dashboard from a long CSV ledger."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path


REQUIRED_COLUMNS = {
    "experiment_id", "task", "cohort", "split", "corpus", "model_id",
    "conditioning", "representation", "metric_name", "value", "status",
    "priority", "selection_role", "higher_is_better", "report_path",
}


def _read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def _html(rows: list[dict[str, str]], title: str) -> str:
    payload = json.dumps(rows, separators=(",", ":"))
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{safe_title}</title>
<style>
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 24px; color:#182025; background:#f8fafb }}
h1 {{ margin:0 0 4px }} .sub {{ color:#58636d; margin:0 0 18px }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 20px }}
label {{ font-weight:600 }} select {{ margin-left:5px; padding:4px }}
.section {{ margin:24px 0 34px; overflow-x:auto; background:white; border:1px solid #dbe2e7 }}
table {{ border-collapse:collapse; width:100%; min-width:840px }} th,td {{ border:1px solid #dbe2e7; padding:7px 8px; text-align:right; white-space:nowrap }}
th {{ background:#edf2f5; position:sticky; top:0; z-index:1 }} td:first-child,td:nth-child(2),td:nth-child(3) {{ text-align:left }}
.complete.primary {{ background:#d9f3df }} .complete.secondary,.complete.sensitivity {{ background:#dceefa }}
.pending,.running {{ background:#fff0c8; color:#6a4b00 }} .deferred,.not_applicable {{ background:#edf0f2; color:#66727a }}
.blocked,.invalidated {{ background:#f9d9d7; color:#861f18 }} .best {{ font-weight:700; box-shadow:inset 0 -3px #18794e }} .empty {{ background:#fff }} tr.not-selected td {{ opacity:.48; background:#f1f3f4 }} tr.leakage-concern td {{ opacity:.58; background:#eceff1 }} .selection-tag {{ display:inline-block; margin-left:6px; padding:1px 5px; border:1px solid #aeb8bf; color:#53616a; font-size:11px; font-weight:600 }} .selection-tag.selected {{ border-color:#18794e; color:#176b45; background:#e2f3e7 }} .selection-tag.leakage-concern {{ border-color:#8c969d; color:#5b666d; background:#eef0f2 }}
.legend span {{ display:inline-block; padding:4px 7px; margin-right:7px; border:1px solid #dbe2e7 }}
.completion {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 }} .completion-card {{ min-width:260px; padding:10px 12px; border:1px solid #cfd8de; background:white }} .completion-card strong {{ display:block; font-size:16px }} .completion-card .muted {{ color:#58636d; font-size:12px }}
a {{ color:#075e8c }}
</style></head><body>
<h1>{safe_title}</h1><p class="sub">Generated from <code>metrics_long.csv</code>. Values link to immutable raw reports when available.</p>
<div class="legend"><span class="complete primary">complete primary</span><span class="complete secondary">complete secondary</span><span class="pending">pending/running</span><span class="deferred">deferred/N.A.</span><span class="invalidated">blocked/invalidated</span><span class="selection-tag selected">provisional selected checkpoint</span><span class="selection-tag">not selected checkpoint</span><span class="selection-tag leakage-concern">leakage concern</span></div>
<div id="completion" class="completion"></div>
<div class="controls"><label><input id="show_test_results" type="checkbox" checked> Show test results</label><label>Split <select id="split"></select></label><label>Task <select id="task"></select></label><label>Cohort <select id="cohort"></select></label><label>Status <select id="status"></select></label></div>
<main id="content"></main>
<script>
const rows={payload};
const labels={{"retrieval/broad_map":"Broad mAP","retrieval/same_charge_10ppm_map":"Strict mAP","pair/same_charge_10ppm/roc_auc":"Strict AUROC","pair/same_charge_10ppm/average_precision":"Strict AP","classification/roc_auc":"Validation AUROC","classification/matched_backbone_roc_auc":"Matched-backbone AUROC","regression/mae":"MAE (lower)","regression/r2":"R²","regression/pearson":"Pearson r","regression/spearman":"Spearman rho","regression/delta_t95":"Delta-t95 (lower)","denovo/peptide_precision":"Peptide precision","denovo/aa_precision":"AA precision","denovo/aa_recall":"AA recall","label_free/emitted_fraction":"Coverage","label_free/proteome_mapped_fraction_emitted":"Proteome mapped @ emitted","label_free/proteome_mapped_fraction_all":"Proteome mapped @ all","label_free/proteome_mapped_fraction_tryptic_emitted":"Tryptic mapped @ emitted","label_free/proteome_mapped_fraction_reversed_decoy":"Reversed-decoy mapped","label_free/proteome_mapped_fraction_shuffled_decoy":"Shuffled-decoy mapped","label_free/modified_call_fraction":"Modified-call fraction","counterfactual/a_condition_source_accuracy":"model(A+B, prec_A) -> A","counterfactual/b_condition_source_accuracy":"model(A+B, prec_B) -> B","counterfactual/both_conditions_source_accuracy":"Both source selections correct","counterfactual/mean_source_accuracy":"Mean source accuracy","counterfactual/a_condition_positive_negative_margin":"A positive-negative margin","counterfactual/b_condition_positive_negative_margin":"B positive-negative margin","counterfactual/mean_positive_negative_margin":"Mean positive-negative margin","null_local/view_1_source_accuracy":"local view 1(A, NULL) -> A","null_local/view_2_source_accuracy":"local view 2(A, NULL) -> A","null_local/both_views_source_accuracy":"Both local views select A","null_local/mean_source_accuracy":"Mean local-view accuracy","null_local/view_1_positive_negative_margin":"View 1 positive-negative margin","null_local/view_2_positive_negative_margin":"View 2 positive-negative margin","null_local/mean_positive_negative_margin":"Mean local-view margin"}};
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":"&quot;");
function options(id, values) {{ const s=document.getElementById(id), old=s.value; s.innerHTML='<option value="all">All</option>'+[...values].sort().map(v=>`<option value="${{esc(v)}}">${{esc(v)}}</option>`).join(''); if ([...s.options].some(o=>o.value===old)) s.value=old; }}
function refreshOptions() {{ for (const [id,key] of [["split","split"],["task","task"],["cohort","cohort"],["status","status"]]) options(id,new Set(rows.map(r=>r[key]))); }}
function renderCompletion() {{ const excluded=new Set(["deferred","not_applicable","invalidated","blocked","obsolete"]); const format=items=>{{ const complete=items.filter(r=>r.status==="complete").length; const pct=items.length ? 100*complete/items.length : 0; return `${{complete}} / ${{items.length}} (${{pct.toFixed(1)}}%)`; }}; const cards=["validation","test"].map(split=>{{ const tracked=rows.filter(r=>r.split===split&&!excluded.has(r.status)); const nonDenovo=tracked.filter(r=>r.task!=="denovo"); const label=split==="test"?"Held-out test":"Validation"; return `<div class="completion-card"><strong>${{esc(label)}}: ${{format(tracked)}}</strong><span class="muted">Non-de novo: ${{format(nonDenovo)}}. All-ledger totals include development-only rows, so their denominators need not match.</span></div>`; }}); const normalizedKey=r=>[r.task,r.cohort,r.corpus,r.model_id,r.conditioning,r.representation,r.metric_name].join("|"); const expectedTest=rows.filter(r=>r.split==="test"&&!excluded.has(r.status)); const validationByKey=new Map(rows.filter(r=>r.split==="validation"&&!excluded.has(r.status)).map(r=>[normalizedKey(r),r])); const matchingValidation=expectedTest.map(r=>validationByKey.get(normalizedKey(r))).filter(Boolean); const validationComplete=matchingValidation.filter(r=>r.status==="complete").length; const testComplete=expectedTest.filter(r=>r.status==="complete").length; cards.push(`<div class="completion-card"><strong>Frozen paper test matrix</strong><span class="muted">Validation counterparts: ${{validationComplete}} / ${{expectedTest.length}} (${{expectedTest.length ? (100*validationComplete/expectedTest.length).toFixed(1) : "0.0"}}%). Held-out test: ${{testComplete}} / ${{expectedTest.length}} (${{expectedTest.length ? (100*testComplete/expectedTest.length).toFixed(1) : "0.0"}}%). Includes de novo and all selected external baselines.</span></div>`); document.getElementById("completion").innerHTML=cards.join(""); }}
function filtered() {{ const showLockedTest=document.getElementById("show_test_results").checked; const status=document.getElementById("status").value; return rows.filter(r=>{{ if (r.split==="test"&&!showLockedTest) return false; if (r.status==="obsolete" && status!=="obsolete") return false; return ["split","task","cohort","status"].every(k=>{{const v=document.getElementById(k).value;return v==="all"||r[k]===v;}}); }}); }}
function render() {{ const data=filtered(), out=document.getElementById('content'); const groups={{}}; for (const r of data) {{ const key=[r.task,r.cohort,r.split].join('|'); (groups[key]??=[]).push(r); }} out.innerHTML=''; for (const [key,rs] of Object.entries(groups).sort()) {{ const metrics=[...new Set(rs.map(r=>r.metric_name))].sort(); const best={{}}; for (const r of rs) {{ if (r.status!=="complete" || r.value==="") continue; const n=Number(r.value); if (!Number.isFinite(n)) continue; const metricKey=[r.corpus,r.metric_name].join('|'); const higher=r.higher_is_better!=="false"; const initial=higher?-Infinity:Infinity; const current=best[metricKey]??initial; best[metricKey]=higher?Math.max(current,n):Math.min(current,n); }} const cells={{}}; for (const r of rs) {{ const row=[r.model_id,r.conditioning,r.representation].join('|'); cells[row]??={{model_id:r.model_id,conditioning:r.conditioning,representation:r.representation,selection_roles:new Set(),values:{{}}}}; cells[row].selection_roles.add(r.selection_role); cells[row].values[[r.corpus,r.metric_name].join('|')]=r; }} const corpora=[...new Set(rs.map(r=>r.corpus))].sort(); let h=`<section class="section"><h2>${{esc(key.replaceAll('|',' / '))}}</h2><table><thead><tr><th>Model</th><th>Conditioning</th><th>Representation</th>${{corpora.flatMap(c=>metrics.map(m=>`<th>${{esc(c)}}<br>${{esc(labels[m]||m)}}</th>`)).join('')}}</tr></thead><tbody>`; for (const row of Object.values(cells).sort((a,b)=>a.model_id.localeCompare(b.model_id))) {{ const selected=row.selection_roles.has("selected"); const provisional=row.selection_roles.has("provisional_selected"); const rejected=row.selection_roles.has("not_selected"); const leakage=row.selection_roles.has("leakage_concern"); const tag=leakage?'<span class="selection-tag leakage-concern">leakage concern</span>':selected?'<span class="selection-tag selected">selected</span>':provisional?'<span class="selection-tag selected">provisional selected</span>':rejected?'<span class="selection-tag">not selected</span>':''; h+=`<tr class="${{leakage?'leakage-concern':rejected?'not-selected':''}}"><td>${{esc(row.model_id)}}${{tag}}</td><td>${{esc(row.conditioning)}}</td><td>${{esc(row.representation)}}</td>`; for (const c of corpora) for (const m of metrics) {{ const r=row.values[[c,m].join('|')]; if (!r) {{h+='<td class="empty">-</td>';continue;}} const val=r.value===''?esc(r.status):Number(r.value).toFixed(4); const content=r.report_href?`<a href="${{esc(r.report_href)}}">${{val}}</a>`:val; const metricKey=[c,m].join('|'); const isBest=r.status==="complete" && r.value!=="" && Number(r.value)===best[metricKey]; h+=`<td class="${{esc(r.status)}} ${{esc(r.priority)}}${{isBest?' best':''}}" title="${{esc(r.selection_role)}}">${{content}}</td>`; }} h+='</tr>'; }} h+='</tbody></table></section>'; out.insertAdjacentHTML('beforeend',h); }} if (!data.length) out.innerHTML='<p>No ledger rows match the selected filters.</p>'; }}
refreshOptions(); renderCompletion(); for (const id of ["split","task","cohort","status","show_test_results"]) document.getElementById(id).onchange=render; document.getElementById('split').value='test'; render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("results/paper/metrics_long.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/paper/paper_result_dashboard.html"))
    parser.add_argument("--title", default="dIon Paper Results Dashboard")
    args = parser.parse_args()
    rows = _read_ledger(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        report_path = row.get("report_path", "")
        if report_path:
            row["report_href"] = os.path.relpath(report_path, args.output.parent)
    args.output.write_text(_html(rows, args.title))
    print(f"Wrote dashboard with {len(rows)} metric rows: {args.output}")


if __name__ == "__main__":
    main()
