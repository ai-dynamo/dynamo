/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * "Get Dynamo" install-command selector.
 *
 * Server component (no "use client"): it expands the generated data
 * (install-selector-data.ts) into hidden radios + <label> chips + [data-*]
 * command blocks, and a generated :has() stylesheet that reveals exactly one
 * block per selection. The selection is pure CSS — no bundled or hydrated JS;
 * the only script is the copy button's inline onclick handler (which a strict
 * CSP without unsafe-inline would block). Delivered as a page-level <style> +
 * injected markup, mirroring the CSS-only pattern in RecipeStyles.tsx (which
 * survives the shared NVIDIA global theme).
 *
 * Register via docs.yml `experimental.mdx-components: ./components` and IMPORT it
 * on the page (ambient use renders "Unsupported JSX tag"):
 *   import { InstallSelector } from "@/components/InstallSelector";
 *   <InstallSelector />
 */
import { INSTALL_DATA } from "./install-selector-data";

type Cmds = { container: string; wheel?: string };
type Entry = {
  backend_version: string;
  dynamo?: string;
  latest?: boolean;
  pin_date?: string;
  note?: string;
  commands: Cmds;
};
type Framework = { label: string; wheel: boolean; stable: Entry[]; nightly: Entry[] };
type Data = Record<string, Framework>;

const CHANNELS: Array<["stable" | "nightly", string, string]> = [
  ["stable", "Stable release", "QA-validated"],
  ["nightly", "Nightly", "latest features"],
];
const FORMS: Array<["container" | "wheel", string]> = [
  ["container", "Container"],
  ["wheel", "Wheel"],
];

function esc(s: string): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function radio(name: string, id: string, checked: boolean): string {
  return `<input type="radio" name="${name}" id="${id}"${checked ? " checked" : ""} />`;
}

function chip(id: string, text: string, sub?: string): string {
  const s = sub ? `<span class="is-avail">${esc(sub)}</span>` : "";
  return `<label for="${id}" class="is-chip">${esc(text)}${s}</label>`;
}

function block(data: Data, fw: string, ch: "stable" | "nightly", e: Entry, i: number, form: "container" | "wheel"): string {
  const label = data[fw].label;
  const isStable = ch === "stable";
  const badge = isStable ? "Stable" : "Nightly";
  const title = isStable
    ? `Dynamo ${e.dynamo}`
    : e.latest
      ? "Latest nightly"
      : `Nightly · ${e.pin_date}`;
  const role = isStable
    ? "Latest stable release that supports this version"
    : e.latest
      ? "Latest nightly build"
      : "Latest nightly in this version's range";
  const ships = `Supports ${label} ${e.backend_version}`;

  const cmd = (e.commands as Record<string, string>)[form];
  const hint = !isStable && e.note ? `<p class="is-hint">${esc(e.note)}</p>` : "";
  const body = cmd
    ? `<div class="is-cmd"><button type="button" class="is-copy" onclick="navigator.clipboard&&navigator.clipboard.writeText(this.closest('.is-cmd').querySelector('pre').innerText);this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},1200)">Copy</button><pre>${esc(cmd)}</pre>${hint}</div>`
    : `<div class="is-cmd is-note"><pre>${esc(label)} has no PyPI wheel — use the Container option above.</pre></div>`;

  return (
    `<div class="is-block" data-block data-fw="${esc(fw)}" data-ch="${esc(ch)}" data-ver="${i}" data-form="${esc(form)}">` +
    `<div class="is-rec is-${ch}"><div class="is-eyebrow">${esc(role)}</div>` +
    `<div class="is-headline"><span class="is-badge">${esc(badge)}</span>${esc(title)}</div>` +
    `<div class="is-ships">${esc(ships)}</div></div>${body}</div>`
  );
}

function buildHtml(data: Data): string {
  const fws = Object.keys(data);

  const fwRow = fws
    .map((fw, i) => radio("is-fw", `is-fw-${fw}`, i === 0) + chip(`is-fw-${fw}`, data[fw].label))
    .join("");

  const chRow = CHANNELS.map(([ch, lbl, sub], i) =>
    radio("is-ch", `is-ch-${ch}`, i === 0) + chip(`is-ch-${ch}`, lbl, sub),
  ).join("");

  const verRow = fws
    .map((fw) =>
      CHANNELS.map(([ch]) => {
        const chips = data[fw][ch]
          .map((e, i) => {
            const sub = ch === "stable" ? `Dynamo ${e.dynamo}` : e.latest ? "latest nightly" : "";
            return radio(`is-ver-${fw}-${ch}`, `is-ver-${fw}-${ch}-${i}`, i === 0) +
              chip(`is-ver-${fw}-${ch}-${i}`, e.backend_version, sub);
          })
          .join("");
        return `<div class="is-vergroup" data-vg="${fw}-${ch}">${chips}</div>`;
      }).join(""),
    )
    .join("");

  const formRow = FORMS.map(([f, lbl], i) =>
    radio("is-form", `is-form-${f}`, i === 0) + chip(`is-form-${f}`, lbl),
  ).join("");

  // Per-framework version row heading ("vLLM version" …); only the selected one shows.
  const verLabel = fws
    .map((fw) => `<span class="is-rl" data-fwlabel="${fw}">${esc(data[fw].label)} version</span>`)
    .join("");

  let blocks = "";
  for (const fw of fws)
    for (const [ch] of CHANNELS) {
      const entries = data[fw][ch];
      if (!entries.length) {
        // No versions for this framework/channel — show a note instead of a blank area.
        blocks +=
          `<div class="is-block" data-block data-fw="${esc(fw)}" data-ch="${esc(ch)}">` +
          `<div class="is-cmd is-note"><pre>No ${esc(ch)} builds available for ${esc(data[fw].label)} right now.</pre></div></div>`;
        continue;
      }
      entries.forEach((e, i) => {
        for (const [form] of FORMS) blocks += block(data, fw, ch, e, i, form);
      });
    }

  return (
    `<div class="is-head"><h3>Choose your build</h3></div>` +
    `<div class="is-row"><span class="is-rl">Backend</span><div class="is-chips">${fwRow}</div></div>` +
    `<div class="is-row"><span class="is-rl">Dynamo build</span><div class="is-chips">${chRow}</div></div>` +
    `<div class="is-row">${verLabel}<div class="is-chips">${verRow}</div></div>` +
    `<div class="is-row"><span class="is-rl">Install form</span><div class="is-chips">${formRow}</div></div>` +
    `<div class="is-out">${blocks}</div>`
  );
}

function buildCss(data: Data): string {
  const fws = Object.keys(data);
  const hide: string[] = [];
  for (const fw of fws) hide.push(`.is-sel:has(#is-fw-${fw}:checked) [data-block][data-fw]:not([data-fw="${fw}"])`);
  for (const [ch] of CHANNELS) hide.push(`.is-sel:has(#is-ch-${ch}:checked) [data-block][data-ch]:not([data-ch="${ch}"])`);
  for (const [f] of FORMS) hide.push(`.is-sel:has(#is-form-${f}:checked) [data-block][data-form]:not([data-form="${f}"])`);
  for (const fw of fws)
    for (const [ch] of CHANNELS)
      data[fw][ch].forEach((_e, i) =>
        hide.push(
          `.is-sel:has(#is-ver-${fw}-${ch}-${i}:checked) [data-block][data-fw="${fw}"][data-ch="${ch}"][data-ver]:not([data-ver="${i}"])`,
        ),
      );

  const vg: string[] = [];
  for (const fw of fws)
    for (const [ch] of CHANNELS) {
      vg.push(`.is-sel:not(:has(#is-fw-${fw}:checked)) [data-vg="${fw}-${ch}"]`);
      vg.push(`.is-sel:not(:has(#is-ch-${ch}:checked)) [data-vg="${fw}-${ch}"]`);
    }
  for (const fw of fws) vg.push(`.is-sel:not(:has(#is-fw-${fw}:checked)) [data-fwlabel="${fw}"]`);

  return `${STATIC_CSS}\n${hide.join(",\n")}{display:none}\n${vg.join(",\n")}{display:none}`;
}

const STATIC_CSS = `
.is-sel {
  --is-green:#76B900; --is-green-2:#5a8f00;
  --is-bg:#fff; --is-surface:#f6f7f5; --is-text:#1a1a1a; --is-muted:#6a6f66;
  --is-border:#e2e4df; --is-code:#f4f5f2; --is-stable:#eef7dd; --is-nightly:#eef1f6;
  border:1px solid var(--is-border); border-radius:14px; overflow:hidden;
  background:var(--is-surface); margin:24px 0; color:var(--is-text);
}
.dark .is-sel {
  --is-bg:#0a0b09; --is-surface:#151713; --is-text:#ececec; --is-muted:#9aa094;
  --is-border:#2a2e27; --is-code:#14160f; --is-stable:#1d2a10; --is-nightly:#141821;
}
.is-sel .is-head { padding:15px 18px 4px; }
.is-sel .is-head h3 { margin:0; font-size:15px; }
.is-sel .is-head p { margin:3px 0 0; font-size:12.5px; color:var(--is-muted); }
.is-sel .is-row { display:flex; align-items:flex-start; flex-wrap:wrap; gap:8px; padding:10px 18px; }
.is-sel .is-row + .is-row { border-top:1px solid var(--is-border); }
.is-sel .is-rl { flex:0 0 120px; padding-top:8px; font-size:11px; font-weight:700;
  letter-spacing:.07em; text-transform:uppercase; color:var(--is-muted); }
.is-sel .is-chips { display:flex; flex-wrap:wrap; gap:8px; flex:1 1 auto; min-width:0; }
.is-sel .is-vergroup { display:flex; flex-wrap:wrap; gap:8px; }
.is-sel input[type="radio"] { position:absolute; opacity:0; pointer-events:none; }
.is-sel .is-chip { display:inline-flex; flex-direction:column; align-items:flex-start; gap:2px;
  min-height:34px; padding:5px 14px; border:1px solid var(--is-border); border-radius:8px;
  background:var(--is-bg); color:var(--is-text); cursor:pointer; font-size:13.5px; line-height:1.15; }
.is-sel input:checked + .is-chip { border-color:var(--is-green); box-shadow:0 0 0 1px var(--is-green); font-weight:700; }
.is-sel input:focus-visible + .is-chip { outline:2px solid var(--is-green); outline-offset:2px; }
.is-sel .is-avail { font-size:10px; font-weight:600; color:var(--is-muted); }
.is-sel input:checked + .is-chip .is-avail { color:var(--is-green-2); }
.dark .is-sel input:checked + .is-chip .is-avail { color:var(--is-green); }
.is-sel .is-out { border-top:1px solid var(--is-border); }
.is-sel .is-rec { padding:13px 18px 12px; }
.is-sel .is-rec.is-stable { background:var(--is-stable); }
.is-sel .is-rec.is-nightly { background:var(--is-nightly); }
.is-sel .is-eyebrow { font-size:10.5px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--is-muted); }
.is-sel .is-headline { margin-top:5px; font-size:15px; }
.is-sel .is-badge { display:inline-block; margin-right:8px; padding:2px 9px; border-radius:999px;
  font-size:11px; font-weight:800; letter-spacing:.03em; vertical-align:1px; }
.is-sel .is-rec.is-stable .is-badge { background:var(--is-green); color:#0a0b09; }
.is-sel .is-rec.is-nightly .is-badge { background:#5b6472; color:#fff; }
.is-sel .is-ships { margin-top:4px; font-size:12.5px; color:var(--is-muted); }
.is-sel .is-cmd { position:relative; padding:15px 18px; background:var(--is-code); }
.is-sel .is-cmd pre { margin:0; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--is-text);
  font:13px/1.5 "Roboto Mono",ui-monospace,SFMono-Regular,Menlo,monospace; }
.is-sel .is-cmd:not(.is-note) pre { padding-right:62px; }
.is-sel .is-copy { position:absolute; top:11px; right:12px; cursor:pointer; font-size:12px;
  padding:4px 10px; border-radius:7px; border:1px solid var(--is-border); background:var(--is-surface); color:var(--is-text); }
.is-sel .is-copy:hover { border-color:var(--is-green); }
.is-sel .is-hint { margin:9px 0 0; font-size:11.5px; color:var(--is-muted); }
.is-sel .is-cmd.is-note pre { color:var(--is-muted); }
`;

export function InstallSelector() {
  const data = INSTALL_DATA as unknown as Data;
  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: buildCss(data) }} />
      <div className="is-sel" dangerouslySetInnerHTML={{ __html: buildHtml(data) }} />
    </>
  );
}

export default InstallSelector;
