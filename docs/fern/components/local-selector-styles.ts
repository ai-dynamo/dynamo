/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

export const LOCAL_SELECTOR_CSS = `
.lqs-panel {
  --lqs-green:#76B900; --lqs-green-dark:#527d00;
  --lqs-bg:#fff; --lqs-surface:#f7f8f6; --lqs-code:#f4f5f2;
  --lqs-text:#1a1a1a; --lqs-muted:#676c63; --lqs-border:#dfe2dc;
  --lqs-stable:#eef7dd; --lqs-nightly:#eef1f6; --lqs-source:#f3effa;
  margin:20px 0; overflow:hidden; border:1px solid var(--lqs-border);
  border-radius:12px; background:var(--lqs-surface); color:var(--lqs-text);
}
.dark .lqs-panel {
  --lqs-bg:#0f100e; --lqs-surface:#171916; --lqs-code:#11130e;
  --lqs-text:#ededed; --lqs-muted:#9ba095; --lqs-border:#2c3029;
  --lqs-stable:#1d2a10; --lqs-nightly:#171c26; --lqs-source:#211b2b;
}
.lqs-head { padding:16px 18px 8px; }
.lqs-head h3 { margin:0; color:var(--lqs-text); font-size:15px; font-weight:700; }
.lqs-head p { margin:4px 0 0; color:var(--lqs-muted); font-size:12.5px; }
.lqs-row { display:grid; grid-template-columns:120px minmax(0,1fr); gap:12px; align-items:start; padding:11px 18px; border-top:1px solid var(--lqs-border); }
.lqs-label { padding-top:8px; color:var(--lqs-muted); font-size:11px; font-weight:750; letter-spacing:.065em; text-transform:uppercase; }
.lqs-options { display:flex; flex-wrap:wrap; gap:8px; min-width:0; }
.lqs-chip { display:inline-flex; min-height:35px; flex-direction:column; align-items:flex-start; justify-content:center; gap:2px; padding:6px 13px; border:1px solid var(--lqs-border); border-radius:8px; background:var(--lqs-bg); color:var(--lqs-text); font:inherit; font-size:13px; line-height:1.1; cursor:pointer; }
.lqs-chip:hover:not(:disabled) { border-color:var(--lqs-green); }
.lqs-chip[aria-pressed="true"] { border-color:var(--lqs-green); box-shadow:0 0 0 1px var(--lqs-green); font-weight:750; }
.lqs-chip:focus-visible { outline:2px solid var(--lqs-green); outline-offset:2px; }
.lqs-chip:disabled { cursor:not-allowed; opacity:.42; background:transparent; color:var(--lqs-muted); text-decoration:none; }
.lqs-chip-sub { color:var(--lqs-muted); font-size:10px; font-weight:600; }
.lqs-chip[aria-pressed="true"] .lqs-chip-sub { color:var(--lqs-green-dark); }
.dark .lqs-chip[aria-pressed="true"] .lqs-chip-sub { color:var(--lqs-green); }
.lqs-field { max-width:420px; min-width:0; }
.lqs-input { width:100%; min-height:36px; box-sizing:border-box; padding:7px 10px; border:1px solid var(--lqs-border); border-radius:7px; background:var(--lqs-bg); color:var(--lqs-text); font:inherit; font-size:13px; }
.lqs-input:focus-visible { outline:2px solid var(--lqs-green); outline-offset:2px; }
.lqs-input[aria-invalid="true"] { border-color:#b54708; }
.lqs-output { border-top:1px solid var(--lqs-border); }
.lqs-rec { padding:14px 18px 12px; }
.lqs-rec--stable { background:var(--lqs-stable); }
.lqs-rec--nightly { background:var(--lqs-nightly); }
.lqs-rec--source { background:var(--lqs-source); }
.lqs-eyebrow { color:var(--lqs-muted); font-size:10.5px; font-weight:750; letter-spacing:.075em; text-transform:uppercase; }
.lqs-title { margin-top:5px; color:var(--lqs-text); font-size:15px; }
.lqs-badge { display:inline-block; margin-right:8px; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:800; vertical-align:1px; }
.lqs-rec--stable .lqs-badge { background:var(--lqs-green); color:#0a0b09; }
.lqs-rec--nightly .lqs-badge { background:#5b6472; color:#fff; }
.lqs-rec--source .lqs-badge { background:#75539b; color:#fff; }
.lqs-support { margin-top:4px; color:var(--lqs-muted); font-size:12.5px; }
.lqs-command { position:relative; padding:16px 18px; background:var(--lqs-code); }
.lqs-command pre { margin:0; padding-right:66px; overflow:auto; color:var(--lqs-text); font:13px/1.55 "Roboto Mono",ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; overflow-wrap:anywhere; }
.lqs-copy { position:absolute; top:12px; right:12px; padding:5px 10px; border:1px solid var(--lqs-border); border-radius:7px; background:var(--lqs-surface); color:var(--lqs-text); font-size:12px; cursor:pointer; }
.lqs-copy:hover { border-color:var(--lqs-green); }
.lqs-copy:disabled { cursor:not-allowed; opacity:.58; }
.lqs-hint { margin:10px 0 0; color:var(--lqs-muted); font-size:11.5px; }
.lqs-hint--error { margin-top:6px; color:#b54708; }
@media (max-width:640px) {
  .lqs-row { grid-template-columns:1fr; gap:6px; }
  .lqs-label { padding-top:0; }
  .lqs-command pre { padding-right:0; padding-top:34px; }
}
`;
