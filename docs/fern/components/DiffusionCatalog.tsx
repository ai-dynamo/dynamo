/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Server-rendered catalog styling with a search/filter/sort controller. Card
 * content, images, and links stay in MDX for Fern's asset and URL rewriting.
 */
import type { ReactNode } from "react";

import { DiffusionCatalogControls } from "./DiffusionCatalogControls";
import { EXAMPLES_LAYOUT_CSS } from "./examples-layout";

const CSS = EXAMPLES_LAYOUT_CSS + `
.dynamo-diffusion {
  --diffusion-text: #18212f;
  --diffusion-muted: #5b6473;
  --diffusion-surface: #ffffff;
  --diffusion-border: #e1e5eb;
  --diffusion-neutral: #f1f3f6;
  --diffusion-shadow: 0 3px 12px #17203306;
  margin-block: 24px 20px;
}
.dynamo-diffusion *, .dynamo-diffusion *::before, .dynamo-diffusion *::after { box-sizing: border-box; }
.dynamo-diffusion-controls {
  padding: 20px; border: 1px solid var(--diffusion-border); border-radius: 14px;
  color: var(--diffusion-text); background: var(--diffusion-surface); box-shadow: var(--diffusion-shadow);
}
.dynamo-diffusion-controls label { display: flex; flex-direction: column; gap: 7px; min-width: 0; font-size: 12px; font-weight: 600; }
.dynamo-diffusion-controls input, .dynamo-diffusion-controls select {
  width: 100%; min-width: 0; min-height: 42px; padding: 9px 11px;
  border: 1px solid var(--diffusion-border); border-radius: 8px;
  color: var(--diffusion-text); background: var(--diffusion-surface); font: inherit; font-size: 13px; font-weight: 400;
}
.dynamo-diffusion-controls input::placeholder { color: var(--diffusion-muted); opacity: 1; }
.dynamo-diffusion-controls input:focus-visible, .dynamo-diffusion-controls select:focus-visible,
.dynamo-diffusion-reset:focus-visible { outline: 2px solid #76b900; outline-offset: 3px; }
.dynamo-diffusion-toolbar { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 230px); gap: 16px; }
.dynamo-diffusion-search { position: relative; display: flex; align-items: center; }
.dynamo-diffusion-search svg { position: absolute; left: 12px; color: var(--diffusion-muted); pointer-events: none; }
.dynamo-diffusion-search input { padding-left: 38px; }
.dynamo-diffusion-filters { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
.dynamo-diffusion-results { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 14px 0; color: var(--diffusion-muted); font-size: 12px; }
.dynamo-diffusion-reset {
  border: 1px solid var(--diffusion-border); border-radius: 7px; padding: 7px 10px;
  color: var(--diffusion-text); background: var(--diffusion-surface); font: inherit; cursor: pointer;
}
.dynamo-diffusion-reset:hover:not(:disabled) { border-color: #76b900; }
.dynamo-diffusion-reset:disabled { opacity: .5; cursor: default; }
.dynamo-diffusion-empty { padding: 36px 20px; border: 1px dashed var(--diffusion-border); border-radius: 14px; text-align: center; color: var(--diffusion-text); }
.dynamo-diffusion .dynamo-diffusion-empty p { margin: 8px 0 0 !important; font-size: 13px; color: var(--diffusion-muted); }
.dynamo-diffusion-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
.dynamo-diffusion-card {
  --diffusion-accent: #7c3aed;
  --diffusion-tint: #7c3aed0c;
  --diffusion-chip-bg: #ede9fe;
  --diffusion-chip-text: #5b21b6;
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
  padding: 22px;
  border: 1px solid var(--diffusion-border);
  border-top: 3px solid var(--diffusion-accent);
  border-radius: 16px;
  color: var(--diffusion-text);
  background: linear-gradient(145deg, var(--diffusion-tint), transparent 65%), var(--diffusion-surface);
  box-shadow: var(--diffusion-shadow);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.dynamo-diffusion-card[data-case="text-to-text"] {
  --diffusion-accent: #16803d; --diffusion-tint: #16803d0c;
  --diffusion-chip-bg: #dcfce7; --diffusion-chip-text: #166534;
}
.dynamo-diffusion-card[data-case="text-to-audio"] {
  --diffusion-accent: #c2410c; --diffusion-tint: #ea580c0b;
  --diffusion-chip-bg: #ffedd5; --diffusion-chip-text: #9a3412;
}
.dynamo-diffusion-card[data-case="text-to-video"] {
  --diffusion-accent: #0e7490; --diffusion-tint: #0891b20c;
  --diffusion-chip-bg: #cffafe; --diffusion-chip-text: #155e75;
}
.dynamo-diffusion-card[data-case="image-to-video"] {
  --diffusion-accent: #2563eb; --diffusion-tint: #2563eb0c;
  --diffusion-chip-bg: #dbeafe; --diffusion-chip-text: #1e40af;
}
.dynamo-diffusion-card[hidden] { display: none !important; }
.dynamo-diffusion-card:hover {
  transform: translateY(-2px);
  border-color: var(--diffusion-accent);
  box-shadow: 0 10px 24px #17203312;
}
.dynamo-diffusion-card-top { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.dynamo-diffusion .dynamo-diffusion-logo {
  display: block; width: 48px; height: 48px; flex: 0 0 48px;
  margin: 0 !important; border: 1px solid #e5e7eb; border-radius: 12px;
  object-fit: contain; background: #fff;
}
.dynamo-diffusion-card-heading { flex: 1; min-width: 130px; }
.dynamo-diffusion .dynamo-diffusion-provider {
  margin: 0 0 4px !important; font-size: 10px; line-height: 1.4;
  font-weight: 650; letter-spacing: .08em; text-transform: uppercase; color: var(--diffusion-muted);
}
.dynamo-diffusion .dynamo-diffusion-card h3 {
  margin: 0 !important; font-size: 18px !important; line-height: 1.3 !important;
  font-weight: 650; letter-spacing: -.02em; color: var(--diffusion-text); overflow-wrap: anywhere;
}
.dynamo-diffusion-experimental {
  padding: 4px 7px; border: 1px solid #fed7aa; border-radius: 6px;
  background: #fff7ed; color: #9a3412; font-size: 10px; line-height: 1.4; font-weight: 600;
}
.dynamo-diffusion .dynamo-diffusion-model {
  margin: 0 !important; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--diffusion-muted); overflow-wrap: anywhere;
}
.dynamo-diffusion .dynamo-diffusion-description {
  margin: 0 !important; font-size: 13px; line-height: 1.6; color: var(--diffusion-muted);
}
.dynamo-diffusion-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; }
.dynamo-diffusion-chip {
  display: inline-flex; align-items: center; padding: 5px 8px; border-radius: 6px;
  font-size: 11px; font-weight: 600; line-height: 1.4; white-space: nowrap;
}
.dynamo-diffusion-modality { color: var(--diffusion-chip-text); background: var(--diffusion-chip-bg); }
.dynamo-diffusion-backend { color: #92400e; background: #fef3c7; }
.dynamo-diffusion-card[data-backend="sglang"] .dynamo-diffusion-backend { color: #3f6212; background: #ecfccb; }
.dynamo-diffusion-card[data-backend="trtllm"] .dynamo-diffusion-backend { color: #1e40af; background: #dbeafe; }
.dynamo-diffusion-size { color: var(--diffusion-muted); background: var(--diffusion-neutral); }
.dynamo-diffusion-card-footer {
  display: flex; justify-content: flex-end; align-items: center; padding-top: 13px;
  border-top: 1px solid var(--diffusion-border); font-size: 11px; font-weight: 600; color: var(--diffusion-accent);
}
.dynamo-diffusion-card-footer > :last-child { font-size: 17px; line-height: 1; }
/* One native, Fern-resolved link covers the entire card. No nested links or controls. */
.dynamo-diffusion .dynamo-diffusion-card-link {
  position: absolute; inset: 0; z-index: 1; border-radius: inherit;
  font-size: 0; color: transparent; text-decoration: none;
}
.dynamo-diffusion .dynamo-diffusion-card-link:focus-visible { outline: 3px solid var(--diffusion-accent); outline-offset: 4px; }
.dark .dynamo-diffusion, [data-theme="dark"] .dynamo-diffusion {
  color-scheme: dark;
  --diffusion-text: #edf1f7; --diffusion-muted: #a6b0bf;
  --diffusion-surface: #191d25; --diffusion-border: #343b48;
  --diffusion-neutral: #2a303b; --diffusion-shadow: 0 3px 12px #0002;
}
.dark .dynamo-diffusion-card, [data-theme="dark"] .dynamo-diffusion-card {
  --diffusion-accent: #c4b5fd; --diffusion-tint: #a78bfa0c;
  --diffusion-chip-bg: #352554; --diffusion-chip-text: #ddd6fe;
}
.dark .dynamo-diffusion-card[data-case="text-to-text"], [data-theme="dark"] .dynamo-diffusion-card[data-case="text-to-text"] {
  --diffusion-accent: #86efac; --diffusion-tint: #4ade800a;
  --diffusion-chip-bg: #163b2c; --diffusion-chip-text: #bbf7d0;
}
.dark .dynamo-diffusion-card[data-case="text-to-audio"], [data-theme="dark"] .dynamo-diffusion-card[data-case="text-to-audio"] {
  --diffusion-accent: #fdba74; --diffusion-tint: #fb923c0a;
  --diffusion-chip-bg: #4b2d19; --diffusion-chip-text: #fed7aa;
}
.dark .dynamo-diffusion-card[data-case="text-to-video"], [data-theme="dark"] .dynamo-diffusion-card[data-case="text-to-video"] {
  --diffusion-accent: #67e8f9; --diffusion-tint: #22d3ee0a;
  --diffusion-chip-bg: #16404a; --diffusion-chip-text: #a5f3fc;
}
.dark .dynamo-diffusion-card[data-case="image-to-video"], [data-theme="dark"] .dynamo-diffusion-card[data-case="image-to-video"] {
  --diffusion-accent: #93c5fd; --diffusion-tint: #60a5fa0a;
  --diffusion-chip-bg: #203857; --diffusion-chip-text: #bfdbfe;
}
.dark .dynamo-diffusion-experimental, [data-theme="dark"] .dynamo-diffusion-experimental { color: #fed7aa; background: #412e1d; border-color: #6b4725; }
.dark .dynamo-diffusion-backend, [data-theme="dark"] .dynamo-diffusion-backend { color: #fde68a; background: #493817; }
.dark .dynamo-diffusion-card[data-backend="sglang"] .dynamo-diffusion-backend,
[data-theme="dark"] .dynamo-diffusion-card[data-backend="sglang"] .dynamo-diffusion-backend { color: #d9f99d; background: #303d1d; }
.dark .dynamo-diffusion-card[data-backend="trtllm"] .dynamo-diffusion-backend,
[data-theme="dark"] .dynamo-diffusion-card[data-backend="trtllm"] .dynamo-diffusion-backend { color: #bfdbfe; background: #203857; }
@media (max-width: 1100px) {
  .dynamo-diffusion-filters { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 680px) {
  .dynamo-diffusion-toolbar { grid-template-columns: minmax(0, 1fr); }
  .dynamo-diffusion-filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .dynamo-diffusion-controls { padding: 16px; }
  .dynamo-diffusion-grid { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  .dynamo-diffusion-card { padding: 18px; }
}
@media (max-width: 380px) {
  .dynamo-diffusion-filters { grid-template-columns: minmax(0, 1fr); }
}
@media (prefers-reduced-motion: reduce) {
  .dynamo-diffusion-card { transition: none; }
  .dynamo-diffusion-card:hover { transform: none; }
}
`;

export function DiffusionCatalog({ children }: { children: ReactNode }) {
  return (
    <section className="dynamo-diffusion" aria-label="Diffusion examples">
      <style dangerouslySetInnerHTML={{ __html: CSS }} />
      <DiffusionCatalogControls>{children}</DiffusionCatalogControls>
    </section>
  );
}
