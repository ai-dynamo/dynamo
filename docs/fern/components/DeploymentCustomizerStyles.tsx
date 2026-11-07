/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Styles for DeploymentCustomizer: the option panel, which builds on the
 * local selector's lqs-* classes, and the highlighted manifest viewer.
 *
 * Named *Styles.tsx deliberately: check_style_components.py guards CSS
 * template literals by that filename. Keep backticks and dollar-brace
 * sequences out of the CSS below, comments included.
 */
export const DEPLOYMENT_CUSTOMIZER_CSS = `
.dc {
  --dc-green: #76b900;
  --dc-green-strong: #4d7a00;
  --dc-bg: #ffffff;
  --dc-surface: #f7f8f6;
  --dc-code-bg: #fbfcfa;
  --dc-text: #1a1a1a;
  --dc-muted: #676c63;
  --dc-border: #dfe2dc;
  --dc-add-bg: rgba(118, 185, 0, 0.12);
  --dc-del-bg: rgba(207, 34, 46, 0.08);
  --dc-del: #cf222e;
  --dc-flash: rgba(118, 185, 0, 0.32);
  --dc-key: #0550ae;
  --dc-str: #0a3069;
  --dc-num: #953800;
  --dc-bool: #8250df;
  --dc-cmt: #6e7781;
  --dc-punct: #57606a;
  --dc-flag: #116329;
  --dc-var: #8250df;
  --dc-kind: #6639ba;
  --dc-cmd: #cf222e;
  --dc-url: #0969da;
  --dc-ph: #7d4e00;
  --dc-ph-bg: #fff3b0;
  --dc-ph-line: #d4a72c;
  --dc-model-bg: rgba(118, 185, 0, 0.16);
  display: grid;
  gap: 16px;
  min-width: 0;
  margin: 4px 0 24px;
  color: var(--dc-text);
}
.dark .dc {
  --dc-green-strong: #9fd332;
  --dc-bg: #0f100e;
  --dc-surface: #171916;
  --dc-code-bg: #11130e;
  --dc-text: #ededed;
  --dc-muted: #9ba095;
  --dc-border: #2c3029;
  --dc-add-bg: rgba(118, 185, 0, 0.14);
  --dc-del-bg: rgba(248, 81, 73, 0.12);
  --dc-del: #ff7b72;
  --dc-flash: rgba(118, 185, 0, 0.34);
  --dc-key: #79c0ff;
  --dc-str: #a5d6ff;
  --dc-num: #ffa657;
  --dc-bool: #d2a8ff;
  --dc-cmt: #8b949e;
  --dc-punct: #8b949e;
  --dc-flag: #7ee787;
  --dc-var: #d2a8ff;
  --dc-kind: #d2a8ff;
  --dc-cmd: #ff7b72;
  --dc-url: #79c0ff;
  --dc-ph: #f2cc60;
  --dc-ph-bg: rgba(187, 128, 9, 0.22);
  --dc-ph-line: #bb8009;
  --dc-model-bg: rgba(118, 185, 0, 0.2);
}

.dc-panel.lqs-panel { margin: 0; }
.dc [hidden] { display: none !important; }
.dc-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 18px;
}
.dc-head-text { min-width: 0; }
.dc-head h3 { margin: 0; font-size: 15px; font-weight: 700; color: var(--dc-text); }
.dc-head p { margin: 3px 0 0; font-size: 12.5px; color: var(--dc-muted); }
.dc-head-actions { display: flex; flex-shrink: 0; gap: 8px; }
.dc .dc-source {
  color: var(--dc-text);
  font-weight: 600;
  text-decoration: underline;
  text-decoration-color: var(--dc-green);
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
  white-space: nowrap;
}
.dc .dc-source:hover { color: var(--dc-green-strong); }
.dc .dc-source:focus-visible { outline: 2px solid var(--dc-green); outline-offset: 2px; border-radius: 3px; }
.dc .dc-source code { all: unset; font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.dc .dc-source svg { display: inline-block; width: 12px; height: 12px; margin-left: 3px; vertical-align: -1px; }
.dc-sep { margin: 0 7px; }

.dc .dc-row-toggle {
  display: inline-flex;
  align-items: center;
  align-self: start;
  gap: 5px;
  margin: 0;
  padding: 8px 0 0;
  border: 0;
  background: none;
  font-family: inherit;
  text-align: left;
  cursor: pointer;
}
.dc .dc-row-toggle:hover { color: var(--dc-text); }
.dc .dc-row-toggle:focus-visible { outline: 2px solid var(--dc-green); outline-offset: 2px; border-radius: 3px; }
.dc-row-toggle svg, .dc-more-toggle svg { flex-shrink: 0; transition: transform 0.15s ease; }
.dc-row-toggle svg { width: 12px; height: 12px; }
.dc-row[data-open="false"] .dc-row-toggle svg, .dc-more[data-open="false"] .dc-more-toggle svg { transform: rotate(-90deg); }
.dc-row-content { display: grid; min-width: 0; }
.dc-row-summary {
  margin: 0;
  padding-top: 6px;
  overflow: hidden;
  color: var(--dc-muted);
  font-size: 12.5px;
  line-height: 1.5;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}
.dc-row-summary:hover { color: var(--dc-text); }
.dc-row-summary code { font-size: 11.5px; }

.dc-more { border-top: 1px solid var(--dc-border); }
.dc-more-toggle {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-height: 44px;
  margin: 0;
  padding: 10px 18px;
  border: 0;
  background: transparent;
  color: var(--dc-text);
  font: inherit;
  font-size: 13px;
  text-align: left;
  cursor: pointer;
}
.dc-more-toggle:hover { background: var(--dc-bg); }
.dc-more-toggle:focus-visible { outline: 2px solid var(--dc-green); outline-offset: -2px; }
.dc-more-title { flex-shrink: 0; font-weight: 650; }
.dc-more-list { flex: 1 1 auto; min-width: 0; overflow: hidden; color: var(--dc-muted); font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.dc-more-toggle .dc-badge { flex-shrink: 0; }
.dc-more-toggle svg { width: 14px; height: 14px; color: var(--dc-muted); }

.dc-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 32px;
  padding: 5px 11px;
  border: 1px solid var(--dc-border);
  border-radius: 8px;
  background: var(--dc-bg);
  color: var(--dc-text);
  font: inherit;
  font-size: 12.5px;
  font-weight: 600;
  line-height: 1.2;
  white-space: nowrap;
  cursor: pointer;
}
.dc-button:hover:not(:disabled) { border-color: var(--dc-green); }
.dc-button:disabled { cursor: not-allowed; opacity: 0.5; }
.dc-button[aria-pressed="true"] { border-color: var(--dc-green); box-shadow: 0 0 0 1px var(--dc-green); }
.dc-button svg { width: 14px; height: 14px; flex-shrink: 0; }
.dc-button:focus-visible, .dc-icon-button:focus-visible, .dc-tab:focus-visible, .dc-option:focus-visible,
.dc-select:focus-visible, .dc-check input:focus-visible, .dc-scroll:focus-visible {
  outline: 2px solid var(--dc-green);
  outline-offset: 2px;
}

.dc-field-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 10px 12px; min-width: 0; }
.dc-field { display: grid; gap: 4px; min-width: 0; flex: 1 1 180px; }
.dc-field--wide { flex: 1 1 100%; }
.dc-field--narrow { flex: 0 1 172px; }
.dc-field-label { color: var(--dc-muted); font-size: 11.5px; font-weight: 600; }
.dc .lqs-input { font-family: inherit; }
.dc-mono, .dc .lqs-input.dc-mono { font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.dc-select {
  width: 100%;
  min-height: 36px;
  box-sizing: border-box;
  padding: 7px 30px 7px 10px;
  border: 1px solid var(--dc-border);
  border-radius: 7px;
  background-color: var(--dc-bg);
  color: var(--dc-text);
  font: inherit;
  font-size: 13px;
}
.dc-select:disabled { background-color: var(--dc-surface); color: var(--dc-muted); cursor: default; opacity: 1; }
.dc-hint { margin: 2px 0 0; color: var(--dc-muted); font-size: 11.5px; line-height: 1.45; }
.dc-hint--warn { color: #9a6700; }
.dark .dc-hint--warn { color: #e3b341; }
.dc-hint code, .dc-note code { font-size: 11px; }
.dc-check { display: inline-flex; align-items: center; gap: 7px; color: var(--dc-text); font-size: 12.5px; cursor: pointer; }
.dc-check input { width: 15px; height: 15px; margin: 0; accent-color: var(--dc-green); }
.dc-check[data-disabled] { color: var(--dc-muted); cursor: not-allowed; }
.dc-checks { display: flex; flex-wrap: wrap; gap: 8px 18px; }
.dc-stack { display: grid; gap: 10px; min-width: 0; }

.dc-combobox { position: relative; min-width: 0; }
.dc-combobox .lqs-input { padding-right: 68px; }
.dc-combobox-actions { position: absolute; top: 4px; right: 4px; display: flex; gap: 2px; }
.dc-icon-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--dc-muted);
  cursor: pointer;
}
.dc-icon-button:hover:not(:disabled) { background: var(--dc-surface); color: var(--dc-text); }
.dc-icon-button:disabled { cursor: not-allowed; opacity: 0.4; }
.dc-icon-button svg { width: 15px; height: 15px; }
.dc-icon-button--remove:hover:not(:disabled) { color: var(--dc-del); }
.dc-listbox {
  position: absolute;
  z-index: 30;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  max-height: 320px;
  margin: 0;
  padding: 6px;
  overflow-y: auto;
  border: 1px solid var(--dc-border);
  border-radius: 10px;
  background: var(--dc-bg);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
  list-style: none;
}
.dark .dc-listbox { box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5); }
.dc-listbox-group {
  padding: 8px 8px 4px;
  color: var(--dc-muted);
  font-size: 10.5px;
  font-weight: 750;
  letter-spacing: 0.065em;
  text-transform: uppercase;
}
.dc-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 7px 9px;
  border-radius: 7px;
  color: var(--dc-text);
  font-size: 13px;
  cursor: pointer;
}
.dc-option[aria-selected="true"], .dc-option:hover { background: var(--dc-surface); }
.dc-option[data-active="true"] { background: var(--dc-add-bg); box-shadow: inset 2px 0 0 var(--dc-green); }
.dc-option-id { min-width: 0; overflow-wrap: anywhere; font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.dc-option-meta { display: flex; flex-shrink: 0; gap: 6px; color: var(--dc-muted); font-size: 11px; }
.dc-badge {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--dc-surface);
  color: var(--dc-muted);
  font-size: 10.5px;
  font-weight: 700;
}
.dc-badge--warn { background: var(--dc-ph-bg); color: var(--dc-ph); }
.dc-badge--ok { background: var(--dc-add-bg); color: var(--dc-green-strong); }
.dc-listbox-empty { padding: 8px 9px; color: var(--dc-muted); font-size: 12.5px; }

.dc-table-wrap { min-width: 0; overflow-x: auto; }
.dc-table { width: 100%; margin: 0; border-collapse: collapse; font-size: 13px; }
.dc-table th {
  padding: 0 10px 6px 0;
  color: var(--dc-muted);
  font-size: 11px;
  font-weight: 650;
  text-align: left;
  white-space: nowrap;
}
.dc-table td { padding: 5px 10px 5px 0; border-top: 1px solid var(--dc-border); vertical-align: middle; }
.dc-table tr:first-child td { border-top: 0; }
.dc-component { font-weight: 650; white-space: nowrap; }
.dc-component small { display: block; color: var(--dc-muted); font-size: 11px; font-weight: 500; }
.dc-stepper {
  display: inline-flex;
  align-items: stretch;
  overflow: hidden;
  border: 1px solid var(--dc-border);
  border-radius: 8px;
  background: var(--dc-bg);
}
.dc-stepper button {
  width: 28px;
  border: 0;
  background: transparent;
  color: var(--dc-muted);
  font: inherit;
  font-size: 15px;
  cursor: pointer;
}
.dc-stepper button:hover:not(:disabled) { background: var(--dc-surface); color: var(--dc-text); }
.dc-stepper button:disabled { cursor: not-allowed; opacity: 0.35; }
.dc-stepper button:focus-visible { outline: 2px solid var(--dc-green); outline-offset: -2px; }
.dc-stepper input {
  width: 44px;
  padding: 5px 0;
  border: 0;
  border-inline: 1px solid var(--dc-border);
  background: transparent;
  color: var(--dc-text);
  font: inherit;
  font-variant-numeric: tabular-nums;
  text-align: center;
  appearance: textfield;
  -moz-appearance: textfield;
}
.dc-stepper input::-webkit-outer-spin-button, .dc-stepper input::-webkit-inner-spin-button { margin: 0; -webkit-appearance: none; }
.dc-stepper input:focus-visible { outline: 2px solid var(--dc-green); outline-offset: -2px; }
.dc-stepper[data-changed="true"] { border-color: var(--dc-green); }
.dc-muted { color: var(--dc-muted); font-size: 12px; }
.dc-total { margin-top: 8px; color: var(--dc-muted); font-size: 12px; }
.dc-total strong { color: var(--dc-text); }

.dc-env { display: grid; gap: 6px; min-width: 0; }
.dc-env-row {
  display: grid;
  grid-template-columns: minmax(110px, 0.8fr) minmax(120px, 1fr) minmax(120px, 1.3fr) 30px;
  gap: 6px;
  align-items: center;
  min-width: 0;
}
.dc-env-head { color: var(--dc-muted); font-size: 11px; font-weight: 650; }
.dc-env-row .lqs-input, .dc-env-row .dc-select { min-height: 32px; padding-top: 5px; padding-bottom: 5px; font-size: 12.5px; }
.dc-env-from {
  min-width: 0;
  padding: 6px 9px;
  overflow: hidden;
  border: 1px dashed var(--dc-border);
  border-radius: 7px;
  color: var(--dc-muted);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dc-env-row[data-new="true"] .lqs-input { border-color: var(--dc-green); }
.dc-env-add { justify-self: start; }

.dc-view {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--dc-border);
  border-radius: 12px;
  background: var(--dc-code-bg);
}
.dc-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 6px 8px 6px 6px;
  border-bottom: 1px solid var(--dc-border);
  background: var(--dc-surface);
}
.dc-tabs { display: flex; flex-wrap: wrap; gap: 2px; min-width: 0; }
.dc-tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--dc-muted);
  font: inherit;
  font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}
.dc-tab:hover { color: var(--dc-text); background: var(--dc-bg); }
.dc-tab[aria-selected="true"] { background: var(--dc-bg); color: var(--dc-text); box-shadow: inset 0 -2px 0 var(--dc-green), 0 0 0 1px var(--dc-border); }
.dc-tab-count {
  display: inline-flex;
  gap: 4px;
  padding: 0 6px;
  border-radius: 999px;
  background: var(--dc-surface);
  font-family: inherit;
  font-size: 10.5px;
}
.dc-tab-count-add { color: var(--dc-green-strong); }
.dc-tab-count-del { color: var(--dc-del); }
.dc-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 6px; margin-left: auto; }
.dc-notes { display: grid; gap: 4px; margin: 0; padding: 8px 14px; border-bottom: 1px solid var(--dc-border); list-style: none; }
.dc-note { color: var(--dc-muted); font-size: 12px; }
.dc-note::before { content: "i"; display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; margin-right: 7px; border-radius: 50%; background: var(--dc-surface); color: var(--dc-muted); font-size: 10px; font-style: italic; font-weight: 800; vertical-align: 1px; }

.dc-scroll { position: relative; max-height: min(72vh, 860px); overflow: auto; overscroll-behavior: contain; }
.dc-view[data-expanded="true"] .dc-scroll { max-height: none; }
.dc-pre {
  display: inline-block;
  min-width: 100%;
  box-sizing: border-box;
  margin: 0;
  padding: 10px 0;
  background: transparent;
  color: var(--dc-text);
  font: 13px/1.62 "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  tab-size: 2;
  white-space: pre;
}
.dc .dc-pre { max-width: none; overflow: visible; }
.dc-pre code { display: block; padding: 0; background: transparent; color: inherit; font: inherit; white-space: inherit; border: 0; }
.dc-line { display: block; min-height: 1.62em; padding: 0 18px 0 0; }
.dc-line::before {
  content: attr(data-ln);
  display: inline-block;
  width: 4.2ch;
  margin-right: 2.4ch;
  padding-left: 1.2ch;
  color: var(--dc-muted);
  opacity: 0.7;
  text-align: right;
  user-select: none;
  -webkit-user-select: none;
}
.dc-line[data-mark]::before { content: attr(data-mark) " " attr(data-ln); }
.dc-line--add { background: var(--dc-add-bg); box-shadow: inset 3px 0 0 var(--dc-green); }
.dc-line--del { background: var(--dc-del-bg); box-shadow: inset 3px 0 0 var(--dc-del); }
.dc-line--del::before { color: var(--dc-del); }
.dc-line--add[data-mark]::before { color: var(--dc-green-strong); opacity: 1; }
.dc-line--flash { animation: dc-flash 1.4s ease-out; }
@keyframes dc-flash { from { background-color: var(--dc-flash); } to { background-color: var(--dc-add-bg); } }

.dc-t-key { color: var(--dc-key); }
.dc-t-str { color: var(--dc-str); }
.dc-t-num { color: var(--dc-num); }
.dc-t-bool, .dc-t-null { color: var(--dc-bool); }
.dc-t-cmt { color: var(--dc-cmt); font-style: italic; }
.dc-t-punct { color: var(--dc-punct); }
.dc-t-flag { color: var(--dc-flag); }
.dc-t-var { color: var(--dc-var); font-weight: 600; }
.dc-t-kind { color: var(--dc-kind); font-weight: 700; }
.dc-t-cmd { color: var(--dc-cmd); font-weight: 600; }
.dc-t-url { color: var(--dc-url); }
.dc-t-anchor { color: var(--dc-bool); font-style: italic; }
.dc-t-ph {
  border-radius: 3px;
  background: var(--dc-ph-bg);
  box-shadow: inset 0 -1px 0 var(--dc-ph-line);
  color: var(--dc-ph);
}
.dc-t-model { border-radius: 3px; background: var(--dc-model-bg); color: var(--dc-str); font-weight: 600; }
.dc-t-img-reg { color: var(--dc-muted); }
.dc-t-img-repo { color: var(--dc-str); font-weight: 600; }
.dc-t-img-tag { color: var(--dc-flag); font-weight: 700; }
.dc-t-img-digest { color: var(--dc-muted); }

.dc-sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 720px) {
  .dc-env-row { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 30px; }
  .dc-env-row > :first-child { grid-column: 1 / -1; }
  .dc-env-head { display: none; }
  .dc-head { flex-wrap: wrap; }
}
@media (max-width: 640px) {
  .dc-toolbar { align-items: stretch; flex-direction: column; }
  .dc-actions { justify-content: flex-end; }
  .dc-pre { font-size: 12px; }
  .dc .dc-row-toggle { padding-top: 0; }
  .dc-row-summary { padding-top: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .dc-line--flash { animation: none; }
  .dc-row-toggle svg, .dc-more-toggle svg { transition: none; }
}
`;

export function DeploymentCustomizerStyles() {
  return <style dangerouslySetInnerHTML={{ __html: DEPLOYMENT_CUSTOMIZER_CSS }} />;
}

export default DeploymentCustomizerStyles;
