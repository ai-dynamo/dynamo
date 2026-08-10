# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render perf-classification.md into an interactive review page.

Run from the repo root:  python3 notes/env-vars/render_classification.py

perf-classification.md is the authored source; this script only presents it. Each
proposed verdict is shown next to the setting's CLI flag, default, and component
scope (pulled from generate.py's record set) so a reviewer can judge it without
cross-referencing the catalogue. Decisions are captured in the browser and exported
as a ready-to-paste PERF dict; nothing is applied to
dynamo-launch-env-vars.html until that dict is copied into generate.py.
"""
import html
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
SRC = "notes/env-vars/perf-classification.md"
OUT = "notes/env-vars/perf-classification-review.html"


def load_catalogue():
    """Import generate.py for its record set without rewriting the catalogue page."""
    ns = {
        "__file__": os.path.abspath("notes/env-vars/generate.py"),
        "__name__": "catalogue",
    }
    exec(open("notes/env-vars/generate.py", encoding="utf-8").read(), ns)  # noqa: S102
    return {(r["env"] or r["flag"]): r for r in ns["records"].values()}, ns["ALL4"]


CATALOGUE, ALL4 = load_catalogue()
SCOPE_LABEL = {"frontend": "FE", "vllm": "vLLM", "trtllm": "TRT", "sglang": "SGL"}


def esc(x):
    return html.escape("" if x is None else str(x))


def inline(text):
    """Bold, inline code, and links — the only inline markdown this file uses."""
    out = esc(text)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def prose(lines):
    """Convert a block of plain markdown (paragraphs, lists, tables) to HTML."""
    out, buf, table, lst = [], [], [], []

    def flush_para():
        if buf:
            out.append(f"<p>{inline(' '.join(buf))}</p>")
            buf.clear()

    def flush_list():
        if lst:
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in lst) + "</ol>")
            lst.clear()

    def flush_table():
        if not table:
            return
        head, *rest = table
        body = [r for r in rest if not set(r.replace("|", "").strip()) <= set("-: ")]

        def cells(row):
            return [c.strip() for c in row.strip().strip("|").split("|")]

        out.append(
            "<table><thead><tr>"
            + "".join(f"<th>{inline(c)}</th>" for c in cells(head))
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r)) + "</tr>"
                for r in body
            )
            + "</tbody></table>"
        )
        table.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|"):
            flush_para()
            flush_list()
            table.append(line)
        elif re.match(r"^\d+\. ", line):
            flush_para()
            flush_table()
            lst.append(re.sub(r"^\d+\. ", "", line))
        elif line.startswith("- "):
            flush_para()
            flush_table()
            lst.append(line[2:])
        elif not line.strip():
            flush_para()
            flush_table()
            flush_list()
        elif line.startswith("#"):
            flush_para()
            flush_table()
            flush_list()
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level + 1}>{inline(line.lstrip('# '))}</h{level + 1}>")
        else:
            flush_table()
            flush_list()
            buf.append(line.strip())
    flush_para()
    flush_table()
    flush_list()
    return "\n".join(out)


def parse():
    """Split the source into the preamble, the numbered categories, and the result."""
    lines = open(SRC, encoding="utf-8").read().split("\n")
    preamble, cats, result = [], [], []
    cur, mode = None, "pre"
    for line in lines:
        m = re.match(r"^## (\d+)\. (.+?) — (.+?) \((\d+)\)$", line)
        if m:
            mode = "cat"
            cur = {
                "n": int(m.group(1)),
                "group": m.group(2),
                "name": m.group(3),
                "count": int(m.group(4)),
                "intro": [],
                "rows": [],
                "summary": "",
            }
            cats.append(cur)
            continue
        if line.startswith("## Result"):
            mode = "result"
            continue
        if mode == "pre":
            preamble.append(line)
        elif mode == "result":
            result.append(line)
        else:
            if line.startswith("|") and not re.match(r"^\|\s*#\s*\|", line):
                if set(line.replace("|", "").strip()) <= set("-: "):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) == 7 and cells[0].split("-")[0].strip().isdigit():
                    cur["rows"].append(cells)
                    continue
            if line.startswith("|"):
                continue
            if line.startswith("**Summary:**"):
                cur["summary"] = line
                continue
            if line.strip() in ("---", ""):
                if line.strip() == "" and cur["intro"]:
                    cur["intro"].append("")
                continue
            cur["intro"].append(line)
    return preamble, cats, result


PREAMBLE, CATS, RESULT = parse()

# Rows whose "Setting" cell covers a run of identical entries (e.g. "17-27").
EXPANDED = {
    "`DYN_HTTP_SVC_*_PATH` (11)": [
        "DYN_HTTP_SVC_CHAT_PATH",
        "DYN_HTTP_SVC_CMP_PATH",
        "DYN_HTTP_SVC_EMB_PATH",
        "DYN_HTTP_SVC_RESPONSES_PATH",
        "DYN_HTTP_SVC_ANTHROPIC_PATH",
        "DYN_HTTP_SVC_MODELS_PATH",
        "DYN_HTTP_SVC_FILES_PATH",
        "DYN_HTTP_SVC_BATCHES_PATH",
        "DYN_HTTP_SVC_METRICS_PATH",
        "DYN_HTTP_SVC_HEALTH_PATH",
        "DYN_HTTP_SVC_LIVE_PATH",
    ]
}

VERDICT_SLUG = {"impact": "impact", "no impact": "noimpact"}


def names_for(cell):
    if cell in EXPANDED:
        return EXPANDED[cell]
    return [cell.strip().strip("`")]


def fmt_default(d):
    if d is None:
        return '<span class="none">unset</span>'
    if isinstance(d, bool):
        return f"<code>{str(d).lower()}</code>"
    if d == "":
        return '<code>""</code>'
    if isinstance(d, list):
        return f'<code>{esc(", ".join(map(str, d)) or "[]")}</code>'
    return f"<code>{esc(d)}</code>"


def render_rows(cat):
    out = []
    for cells in cat["rows"]:
        idx, setting, L, M, score, verdict, why = cells
        for name in names_for(setting):
            rec = CATALOGUE.get(name, {})
            scope = rec.get("scope", [])
            chips = "".join(
                f'<span class="chip s-{s}{"" if s in scope else " off"}">'
                f"{SCOPE_LABEL[s]}</span>"
                for s in ALL4
            )
            flags = ([rec["flag"]] if rec.get("flag") else []) + list(
                rec.get("aliases") or []
            )
            if name.startswith("--"):
                flags = [f for f in flags if f != name]
            flag_html = " ".join(f'<code class="flag">{esc(f)}</code>' for f in flags)
            slug = VERDICT_SLUG[verdict]
            out.append(
                f'<tr class="r" data-name="{esc(name)}" data-verdict="{slug}" '
                f'data-score="{esc(score)}" '
                f'data-text="{esc((name + " " + why).lower())}">'
                f'<td class="c-idx">{esc(idx)}</td>'
                f'<td class="c-name"><code class="env">{esc(name)}</code>{chips}'
                f'<div class="meta">{flag_html}'
                f"<span class=\"def\">default {fmt_default(rec.get('default'))}</span>"
                f"</div></td>"
                f'<td class="c-lm"><span class="lm">L{esc(L)}</span>'
                f'<span class="lm">M{esc(M)}</span>'
                f'<span class="score">{esc(score)}</span></td>'
                f'<td class="c-verdict"><span class="chip v v-{slug}">{esc(verdict)}'
                f"</span></td>"
                f'<td class="c-why">{inline(why)}</td>'
                f'<td class="c-act"><div class="acts">'
                f'<button class="act agree" data-act="agree" title="Accept this verdict">'
                f"agree</button>"
                f'<button class="act flip" data-act="flip" '
                f'title="Record the opposite verdict">flip</button>'
                f'<button class="act clear" data-act="clear" title="Undecided">'
                f"&times;</button></div></td>"
                f"</tr>"
            )
    return "\n".join(out)


total_rows = sum(len(names_for(c[1])) for cat in CATS for c in cat["rows"])
n_impact = sum(
    len(names_for(c[1])) for cat in CATS for c in cat["rows"] if c[5] == "impact"
)

nav, panels = [], []
for cat in CATS:
    key = f"c{cat['n']}"
    nav.append(
        f'<button class="tab" data-tab="{key}"><span class="tn">{cat["n"]}</span>'
        f'{esc(cat["group"])} <span class="ts">{esc(cat["name"])}</span>'
        f'<span class="n">{cat["count"]}</span></button>'
    )
    panels.append(
        f'<div class="panel" id="p-{key}">'
        f'<h2>{cat["n"]}. {esc(cat["group"])} <span class="sep">&mdash;</span> '
        f'{esc(cat["name"])} <span class="count">{cat["count"]}</span></h2>'
        f'<div class="intro">{prose(cat["intro"])}</div>'
        '<table class="rows"><thead><tr><th>#</th><th>Setting</th>'
        "<th>L&nbsp;/&nbsp;M</th><th>Proposed</th><th>Why</th><th>Decision</th>"
        f"</tr></thead><tbody>{render_rows(cat)}</tbody></table>"
        f'<p class="summary">{inline(cat["summary"].replace("**Summary:**", "Summary:"))}'
        f"</p></div>"
    )

nav.insert(
    0,
    '<button class="tab" data-tab="about"><span class="tn">i</span>About & method</button>',
)
panels.insert(
    0,
    f'<div class="panel" id="p-about"><div class="intro doc">{prose(PREAMBLE)}'
    f"<h2>Result</h2>{prose(RESULT)}</div></div>",
)

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#12151a;--muted:#5d6672;--line:#e3e7ed;
--accent:#3d7dd8;--code:#f2f4f7;--fe:#3d7dd8;--vllm:#2f9e6e;--trt:#c26a1f;--sgl:#8355c9;
--impact:#c0392b;--noimpact:#2f9e6e;--warn:#b8860b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);padding:22px 28px 0;
position:sticky;top:0;z-index:20}
.wrap{max-width:1500px;margin:0 auto}
h1{margin:0 0 6px;font-size:22px;letter-spacing:-.01em}
.banner{background:#fff8ec;border:1px solid #f0e0c2;border-radius:9px;padding:10px 14px;
margin:0 0 14px;font-size:13px;color:#5a4a2e;max-width:1000px}
.banner b{color:#4a3a1e}
.tools{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
input[type=search]{flex:1;min-width:220px;max-width:360px;padding:8px 11px;
border:1px solid var(--line);border-radius:8px;font-size:14px;background:var(--card)}
input[type=search]:focus,select:focus{outline:2px solid rgba(61,125,216,.25);
border-color:var(--accent)}
select{padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:var(--card);
font:inherit;font-size:13px;cursor:pointer}
label.tog{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:13px}
.prog{margin-left:auto;font-size:13px;color:var(--muted);display:flex;gap:10px;
align-items:center}
.prog b{color:var(--ink)}
button.primary{background:var(--accent);color:#fff;border:none;padding:8px 14px;
border-radius:8px;font:600 13px inherit;cursor:pointer}
button.primary:hover{filter:brightness(1.07)}
button.ghost{background:var(--card);color:var(--muted);border:1px solid var(--line);
padding:8px 12px;border-radius:8px;font:600 13px inherit;cursor:pointer}
nav{display:flex;gap:3px;flex-wrap:wrap;padding-bottom:0;max-height:132px;overflow-y:auto}
.tab{appearance:none;background:transparent;border:1px solid transparent;border-bottom:none;
padding:7px 11px;border-radius:8px 8px 0 0;font:600 12.5px/1 inherit;color:var(--muted);
cursor:pointer;display:flex;gap:6px;align-items:center}
.tab:hover{color:var(--ink);background:#f0f2f6}
.tab.on{background:var(--bg);border-color:var(--line);color:var(--ink)}
.tab .tn{background:var(--code);border-radius:5px;padding:2px 6px;font-size:10.5px}
.tab.on .tn{background:var(--accent);color:#fff}
.tab .ts{color:var(--muted);font-weight:500}
.tab .n{color:var(--muted);font-size:10.5px}
.tab.done .tn{background:var(--noimpact);color:#fff}
main{max-width:1500px;margin:0 auto;padding:22px 28px 90px}
.panel{display:none}.panel.on{display:block}
h2{font-size:18px;margin:0 0 12px;display:flex;gap:9px;align-items:center}
.sep{color:var(--muted);font-weight:400}
.count{background:var(--code);color:var(--muted);border-radius:20px;padding:1px 9px;
font-size:11px;font-weight:600}
.intro{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:4px 20px;margin-bottom:16px;font-size:14px}
.intro p{color:#2b3038}
.intro.doc{padding:8px 26px 20px}
.intro h2,.intro h3{font-size:15px;margin-top:22px}
.intro table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}
.intro th{text-align:left;border-bottom:1px solid var(--line);padding:7px 9px;
color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.intro td{border-bottom:1px solid var(--line);padding:7px 9px}
table.rows{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:12px;overflow:hidden}
table.rows thead th{text-align:left;font-size:11px;text-transform:uppercase;
letter-spacing:.07em;color:var(--muted);padding:9px 11px;border-bottom:1px solid var(--line);
font-weight:600;background:#fbfcfd}
table.rows td{padding:11px;border-bottom:1px solid var(--line);vertical-align:top}
table.rows tr:last-child td{border-bottom:none}
tr.r:hover{background:#fafbfd}
tr.r.agreed{background:#f2faf5}tr.r.flipped{background:#fdf4f2}
.c-idx{width:32px;color:var(--muted);font-size:12px}
.c-name{width:22%}.c-lm{width:92px;white-space:nowrap}.c-verdict{width:96px}
.c-act{width:132px}
code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:var(--code);padding:1.5px 5px;border-radius:4px}
code.env{background:transparent;padding:0;font-weight:600;word-break:break-all}
code.flag{color:#1f5fae;background:#eef4fc;font-size:11.5px}
.meta{margin-top:5px;font-size:11px;color:var(--muted);display:flex;gap:7px;
flex-wrap:wrap;align-items:center}
.meta code{font-size:11px}
.none{color:#a8b0bb}
.chip{display:inline-block;margin:5px 3px 0 0;font:600 9.5px/1.4 ui-monospace,monospace;
padding:2px 5px;border-radius:4px;color:#fff}
.s-frontend{background:var(--fe)}.s-vllm{background:var(--vllm)}
.s-trtllm{background:var(--trt)}.s-sglang{background:var(--sgl)}
.chip.off{background:#eceff3;color:#c2c8d0}
.chip.v{margin:0;font-size:10.5px;padding:3px 7px;text-transform:lowercase}
.v-impact{background:var(--impact)}.v-noimpact{background:var(--noimpact)}
.lm{display:inline-block;font:600 10.5px ui-monospace,monospace;background:var(--code);
color:var(--muted);border-radius:4px;padding:2px 5px;margin-right:3px}
.score{display:inline-block;font:700 11px ui-monospace,monospace;color:var(--ink)}
.c-why{color:#2b3038;font-size:13.5px}
.acts{display:flex;gap:4px}
.act{border:1px solid var(--line);background:var(--card);border-radius:7px;
padding:5px 9px;font:600 11.5px inherit;color:var(--muted);cursor:pointer}
.act:hover{border-color:var(--accent);color:var(--accent)}
tr.agreed .act.agree{background:var(--noimpact);border-color:var(--noimpact);color:#fff}
tr.flipped .act.flip{background:var(--impact);border-color:var(--impact);color:#fff}
.summary{color:var(--muted);font-size:13px;margin:12px 2px 0}
dialog{border:none;border-radius:14px;padding:0;max-width:820px;width:92%;
box-shadow:0 24px 60px rgba(0,0,0,.22)}
dialog::backdrop{background:rgba(15,20,28,.45)}
.dlg{padding:22px 24px}
.dlg h3{margin:0 0 6px;font-size:17px}
.dlg p{color:var(--muted);font-size:13px;margin:0 0 14px}
textarea{width:100%;height:340px;font:12px/1.5 ui-monospace,monospace;border:1px solid
var(--line);border-radius:9px;padding:12px;background:#fbfcfd;resize:vertical}
.dlgbtns{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
footer{max-width:1500px;margin:0 auto;padding:0 28px 60px;color:var(--muted);font-size:12.5px}
"""

JS = """
const KEY='dyn-perf-classification-v1';
let dec=JSON.parse(localStorage.getItem(KEY)||'{}');
const rows=[...document.querySelectorAll('tr.r')];
const tabs=[...document.querySelectorAll('.tab')];
const panels=[...document.querySelectorAll('.panel')];
const q=document.getElementById('q'),fv=document.getElementById('fv'),
      fd=document.getElementById('fd');

function paint(){
  rows.forEach(r=>{const d=dec[r.dataset.name];
    r.classList.toggle('agreed',d==='agree');
    r.classList.toggle('flipped',d==='flip');});
  const n=rows.filter(r=>dec[r.dataset.name]).length;
  document.getElementById('done').textContent=n;
  document.getElementById('left').textContent=rows.length-n;
  tabs.forEach(t=>{const p=document.getElementById('p-'+t.dataset.tab);
    if(!p)return;const rs=[...p.querySelectorAll('tr.r')];
    t.classList.toggle('done',rs.length>0&&rs.every(r=>dec[r.dataset.name]));});
}
function final(r){const d=dec[r.dataset.name];if(!d)return null;
  const v=r.dataset.verdict;
  return d==='agree'?v:(v==='impact'?'noimpact':'impact');}
function filter(){
  const s=q.value.trim().toLowerCase(),v=fv.value,st=fd.value;
  document.querySelectorAll('.panel.on table.rows tr.r').forEach(r=>{
    const d=dec[r.dataset.name]||'';
    const ok=(!s||r.dataset.text.includes(s))&&(!v||r.dataset.verdict===v)&&
      (!st||(st==='undecided'?!d:d===st));
    r.style.display=ok?'':'none';});
}
function show(k){tabs.forEach(t=>t.classList.toggle('on',t.dataset.tab===k));
  panels.forEach(p=>p.classList.toggle('on',p.id==='p-'+k));
  location.hash=k;filter();window.scrollTo(0,0);}
tabs.forEach(t=>t.onclick=()=>show(t.dataset.tab));
q.oninput=filter;fv.onchange=filter;fd.onchange=filter;
document.addEventListener('click',e=>{
  const b=e.target.closest('button.act');if(!b)return;
  const r=b.closest('tr.r'),a=b.dataset.act;
  if(a==='clear')delete dec[r.dataset.name];else dec[r.dataset.name]=a;
  localStorage.setItem(KEY,JSON.stringify(dec));paint();filter();});
document.getElementById('export').onclick=()=>{
  const out=['    # Verdicts validated by review; paste into PERF in generate.py.'];
  let n=0;
  rows.forEach(r=>{const f=final(r);if(!f)return;n++;
    out.push('    "'+r.dataset.name+'": "'+(f==='impact'?'impact':'no impact')+'",');});
  document.getElementById('outbox').value=n?out.join('\\n')
    :'// No decisions recorded yet — agree or flip at least one row.';
  document.getElementById('outnote').textContent=
    n+' of '+rows.length+' settings decided. Undecided settings are omitted, so they stay unexamined.';
  document.getElementById('dlg').showModal();};
document.getElementById('copy').onclick=()=>{
  const t=document.getElementById('outbox');t.select();
  navigator.clipboard.writeText(t.value);
  document.getElementById('copy').textContent='copied';
  setTimeout(()=>document.getElementById('copy').textContent='copy',1200);};
document.getElementById('close').onclick=()=>document.getElementById('dlg').close();
document.getElementById('reset').onclick=()=>{
  if(!confirm('Discard all recorded decisions?'))return;
  dec={};localStorage.removeItem(KEY);paint();filter();};
show((location.hash||'#about').slice(1));paint();
"""

doc = f"""<!doctype html>
<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo settings — proposed performance classification (review)</title>
<style>{CSS}</style></head>
<body>
<header><div class="wrap">
<h1>Proposed performance classification &mdash; for review</h1>
<p class="banner"><b>Nothing here is applied.</b> These are proposed verdicts for all
{total_rows} Dynamo launch settings ({n_impact} <code>impact</code>,
{total_rows - n_impact} <code>no impact</code>). The catalogue page
&mdash; <a href="dynamo-launch-env-vars.html">dynamo-launch-env-vars.html</a> &mdash; still
reports every setting as <code>unexamined</code>, and stays that way until these are
validated. Work through the categories, <b>agree</b> or <b>flip</b> each verdict, then
export the result and paste it into <code>PERF</code> in <code>generate.py</code>.</p>
<div class="tools">
  <input type="search" id="q" placeholder="Filter by name or rationale&hellip;" autocomplete="off">
  <label class="tog">Proposed
    <select id="fv"><option value="">any</option><option value="impact">impact</option>
    <option value="noimpact">no impact</option></select></label>
  <label class="tog">Decision
    <select id="fd"><option value="">any</option><option value="undecided">undecided</option>
    <option value="agree">agreed</option><option value="flip">flipped</option></select></label>
  <span class="prog"><span><b id="done">0</b> decided</span>
    <span><b id="left">{total_rows}</b> left</span>
    <button class="ghost" id="reset">reset</button>
    <button class="primary" id="export">Export decisions</button></span>
</div>
<nav>{''.join(nav)}</nav>
</div></header>
<main>{''.join(panels)}</main>
<dialog id="dlg"><div class="dlg">
  <h3>Validated verdicts</h3>
  <p id="outnote"></p>
  <textarea id="outbox" spellcheck="false"></textarea>
  <div class="dlgbtns"><button class="ghost" id="copy">copy</button>
    <button class="primary" id="close">done</button></div>
</div></dialog>
<footer>Generated from <code>notes/env-vars/perf-classification.md</code> by
<code>notes/env-vars/render_classification.py</code>. Setting flags, defaults, and component
scope come from the catalogue's own record set in <code>notes/env-vars/generate.py</code>.
Decisions are stored in this browser only.</footer>
<script>{JS}</script></body></html>"""

if globals().get("__name__") == "__main__":
    open(OUT, "w", encoding="utf-8").write(doc)
    print(f"wrote {OUT}: {len(CATS)} categories, {total_rows} settings")
