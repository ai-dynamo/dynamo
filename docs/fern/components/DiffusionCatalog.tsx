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
/* Surfaces, fields, and chips mirror the shared site palette: the Reference
   panels and badges (ReferenceStyles), the CompatibilityHero select, and the
   main.css .dynamo-chip-* tints. Values stay literal because the published
   theme drops main.css. The modality band carries each card's color, backend
   chips carry per-backend tints, and green remains the interactive accent. */
.dynamo-diffusion {
  --diffusion-text: #1a1a1a;
  --diffusion-muted: #666666;
  --diffusion-panel: #f7f7f7;
  --diffusion-card: #ffffff;
  --diffusion-field: #ffffff;
  --diffusion-border: var(--grayscale-a5, #dfe1df);
  --diffusion-field-border: var(--grayscale-a5, #dfe1df);
  --diffusion-field-hover: rgba(118, 185, 0, 0.55);
  --diffusion-green: #76b900;
  --diffusion-violet-fg: #6d28d9;
  --diffusion-violet-bg: rgba(124, 58, 237, 0.12);
  --diffusion-violet-border: rgba(124, 58, 237, 0.3);
  --diffusion-green-fg: #4d7c0f;
  --diffusion-green-bg: rgba(118, 185, 0, 0.16);
  --diffusion-green-border: rgba(118, 185, 0, 0.38);
  --diffusion-orange-fg: #c7401c;
  --diffusion-orange-bg: rgba(233, 84, 32, 0.12);
  --diffusion-orange-border: rgba(233, 84, 32, 0.3);
  --diffusion-teal-fg: #0e7490;
  --diffusion-teal-bg: rgba(8, 145, 178, 0.12);
  --diffusion-teal-border: rgba(8, 145, 178, 0.3);
  --diffusion-blue-fg: #1d4ed8;
  --diffusion-blue-bg: rgba(37, 99, 235, 0.1);
  --diffusion-blue-border: rgba(37, 99, 235, 0.28);
  --diffusion-sky-fg: #0369a1;
  --diffusion-sky-bg: rgba(2, 132, 199, 0.1);
  --diffusion-sky-border: rgba(2, 132, 199, 0.28);
  --diffusion-pink-fg: #be185d;
  --diffusion-pink-bg: rgba(219, 39, 119, 0.1);
  --diffusion-pink-border: rgba(219, 39, 119, 0.28);
  --diffusion-gray-fg: #5f5e5a;
  --diffusion-gray-bg: rgba(120, 120, 120, 0.1);
  --diffusion-gray-border: rgba(120, 120, 120, 0.28);
  --diffusion-amber-fg: #854f0b;
  --diffusion-amber-dash: rgba(185, 122, 23, 0.7);
  margin-block: 24px 20px;
}
.dark .dynamo-diffusion, [data-theme="dark"] .dynamo-diffusion {
  color-scheme: dark;
  --diffusion-text: #eeeeee;
  --diffusion-muted: #999999;
  --diffusion-panel: #161616;
  --diffusion-card: #161616;
  --diffusion-field: #1d1d1d;
  --diffusion-border: #2b2b2b;
  --diffusion-field-border: #3a3a3a;
  --diffusion-violet-fg: #c4b5fd;
  --diffusion-violet-bg: rgba(139, 92, 246, 0.2);
  --diffusion-violet-border: rgba(139, 92, 246, 0.42);
  --diffusion-green-fg: #a3e635;
  --diffusion-green-bg: rgba(118, 185, 0, 0.22);
  --diffusion-green-border: rgba(118, 185, 0, 0.46);
  --diffusion-orange-fg: #ff9068;
  --diffusion-orange-bg: rgba(233, 84, 32, 0.2);
  --diffusion-orange-border: rgba(233, 84, 32, 0.42);
  --diffusion-teal-fg: #67e8f9;
  --diffusion-teal-bg: rgba(34, 211, 238, 0.16);
  --diffusion-teal-border: rgba(34, 211, 238, 0.4);
  --diffusion-blue-fg: #93c5fd;
  --diffusion-blue-bg: rgba(59, 130, 246, 0.18);
  --diffusion-blue-border: rgba(59, 130, 246, 0.42);
  --diffusion-sky-fg: #7dd3fc;
  --diffusion-sky-bg: rgba(56, 189, 248, 0.16);
  --diffusion-sky-border: rgba(56, 189, 248, 0.4);
  --diffusion-pink-fg: #f9a8d4;
  --diffusion-pink-bg: rgba(236, 72, 153, 0.18);
  --diffusion-pink-border: rgba(236, 72, 153, 0.4);
  --diffusion-gray-fg: #a8a8a8;
  --diffusion-gray-bg: #242424;
  --diffusion-gray-border: #383838;
  --diffusion-amber-fg: #fac775;
  --diffusion-amber-dash: rgba(239, 159, 39, 0.55);
}
.dynamo-diffusion *, .dynamo-diffusion *::before, .dynamo-diffusion *::after { box-sizing: border-box; }
.dynamo-diffusion-controls {
  padding: 20px 22px; border: 1px solid var(--diffusion-border); border-radius: 12px;
  color: var(--diffusion-text); background: var(--diffusion-panel);
}
.dynamo-diffusion-controls label {
  display: flex; flex-direction: column; gap: 7px; min-width: 0;
  color: var(--diffusion-muted); font-size: 11px; font-weight: 700; line-height: 1.3;
  letter-spacing: .08em; text-transform: uppercase;
}
.dynamo-diffusion-controls input, .dynamo-diffusion-controls select {
  width: 100%; min-width: 0; min-height: 38px; padding: 7px 12px;
  border: 1px solid var(--diffusion-field-border); border-radius: 8px;
  color: var(--diffusion-text); background: var(--diffusion-field);
  font: inherit; font-size: 13.5px; font-weight: 400; letter-spacing: normal; text-transform: none;
  transition: border-color .15s ease;
}
.dynamo-diffusion-controls select { padding-right: 34px; cursor: pointer; }
.dynamo-diffusion-controls input:hover:not(:disabled),
.dynamo-diffusion-controls select:hover:not(:disabled) { border-color: var(--diffusion-field-hover); }
.dynamo-diffusion-controls input::placeholder { color: var(--diffusion-muted); opacity: 1; }
.dynamo-diffusion-controls input:focus-visible, .dynamo-diffusion-controls select:focus-visible,
.dynamo-diffusion-reset:focus-visible { outline: 2px solid var(--diffusion-green); outline-offset: 2px; }
.dynamo-diffusion-toolbar { display: grid; grid-template-columns: minmax(0, 1fr) minmax(180px, 230px); gap: 16px; }
.dynamo-diffusion-search { position: relative; display: flex; align-items: center; }
.dynamo-diffusion-search svg { position: absolute; left: 12px; color: var(--diffusion-muted); pointer-events: none; }
.dynamo-diffusion-search input { padding-left: 38px; }
.dynamo-diffusion-filters { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
.dynamo-diffusion-results {
  display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 16px 0 14px;
  color: var(--diffusion-muted); font-size: 12.5px; font-variant-numeric: tabular-nums;
}
.dynamo-diffusion-reset {
  display: inline-flex; align-items: center; min-height: 28px; padding: 5px 12px;
  border: 1px solid var(--diffusion-border); border-radius: 999px;
  color: var(--diffusion-muted); background: var(--diffusion-panel);
  font: inherit; font-size: 12px; font-weight: 700; line-height: 1; cursor: pointer;
  transition: border-color .15s ease, color .15s ease;
}
.dynamo-diffusion-reset:hover:not(:disabled) { border-color: var(--diffusion-green); color: var(--diffusion-text); }
.dynamo-diffusion-reset:disabled { opacity: .5; cursor: default; }
.dynamo-diffusion-empty {
  padding: 32px 20px; border: 1px dashed var(--diffusion-field-border); border-radius: 12px;
  text-align: center; color: var(--diffusion-text);
}
.dynamo-diffusion .dynamo-diffusion-empty p { margin: 8px 0 0 !important; font-size: 13px; color: var(--diffusion-muted); }
.dynamo-diffusion-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.dynamo-diffusion-card {
  --diffusion-accent-fg: var(--diffusion-violet-fg);
  --diffusion-accent-bg: var(--diffusion-violet-bg);
  --diffusion-accent-border: var(--diffusion-violet-border);
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  padding: 20px;
  border: 1px solid var(--diffusion-border);
  border-radius: 12px;
  color: var(--diffusion-text);
  background-color: var(--diffusion-card);
  /* Modality glow behind the logo, after the green glow on Home and Mermaid canvases. */
  background-image: radial-gradient(240px circle at 44px 44px, color-mix(in srgb, var(--diffusion-accent-bg) 60%, transparent), transparent 70%);
  transition: border-color .18s ease, box-shadow .18s ease;
}
.dynamo-diffusion-card[data-case="text-to-text"] {
  --diffusion-accent-fg: var(--diffusion-green-fg);
  --diffusion-accent-bg: var(--diffusion-green-bg);
  --diffusion-accent-border: var(--diffusion-green-border);
}
.dynamo-diffusion-card[data-case="text-to-audio"] {
  --diffusion-accent-fg: var(--diffusion-orange-fg);
  --diffusion-accent-bg: var(--diffusion-orange-bg);
  --diffusion-accent-border: var(--diffusion-orange-border);
}
.dynamo-diffusion-card[data-case="text-to-video"] {
  --diffusion-accent-fg: var(--diffusion-teal-fg);
  --diffusion-accent-bg: var(--diffusion-teal-bg);
  --diffusion-accent-border: var(--diffusion-teal-border);
}
.dynamo-diffusion-card[data-case="image-to-video"] {
  --diffusion-accent-fg: var(--diffusion-blue-fg);
  --diffusion-accent-bg: var(--diffusion-blue-bg);
  --diffusion-accent-border: var(--diffusion-blue-border);
}
.dynamo-diffusion-card[hidden] { display: none !important; }
.dynamo-diffusion-card:hover { border-color: var(--diffusion-green); box-shadow: 0 12px 28px rgba(0, 0, 0, .09); }
.dark .dynamo-diffusion-card:hover, [data-theme="dark"] .dynamo-diffusion-card:hover { box-shadow: 0 14px 32px rgba(0, 0, 0, .32); }
.dynamo-diffusion-card-top { display: flex; align-items: center; gap: 14px; }
.dynamo-diffusion .dynamo-diffusion-logo {
  display: block; width: 48px; height: 48px; flex: 0 0 48px;
  margin: 0 !important; border: 1px solid var(--diffusion-border); border-radius: 12px;
  object-fit: contain; background: #fff;
}
.dynamo-diffusion-card-heading { flex: 1; min-width: 0; }
.dynamo-diffusion-card-arrow {
  display: grid; place-items: center; flex: 0 0 auto; align-self: flex-start;
  width: 30px; height: 30px; border: 1px solid var(--diffusion-accent-border); border-radius: 50%;
  color: var(--diffusion-accent-fg); background: var(--diffusion-accent-bg);
  font-size: 15px; line-height: 1;
}
.dynamo-diffusion .dynamo-diffusion-provider {
  margin: 0 0 3px !important; color: var(--diffusion-muted);
  font-size: 11px; font-weight: 700; line-height: 1.4; letter-spacing: .08em; text-transform: uppercase;
}
.dynamo-diffusion .dynamo-diffusion-card h3 {
  margin: 0 !important; font-size: 18px !important; line-height: 1.3 !important;
  font-weight: 650; letter-spacing: -.02em; color: var(--diffusion-text); overflow-wrap: anywhere;
}
/* The inset accent bar follows the site convention for recommended rows and the active sidebar link. */
.dynamo-diffusion .dynamo-diffusion-modality {
  display: flex; align-items: center; min-height: 34px;
  margin: 0 !important; padding: 6px 12px 6px 14px;
  border: 1px solid var(--diffusion-accent-border); border-radius: 8px;
  color: var(--diffusion-accent-fg); background: var(--diffusion-accent-bg);
  box-shadow: inset 3px 0 0 var(--diffusion-accent-fg);
  font-size: 12px; font-weight: 700; line-height: 1.3; letter-spacing: .08em; text-transform: uppercase;
}
.dynamo-diffusion .dynamo-diffusion-description {
  margin: 0 !important; font-size: 13.5px; line-height: 1.6; color: var(--diffusion-muted);
}
.dynamo-diffusion-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: auto; padding-top: 2px; }
.dynamo-diffusion-chip {
  display: inline-flex; align-items: center; min-height: 22px; padding: 2px 9px;
  border: 1px solid transparent; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; line-height: 1.2; white-space: nowrap;
}
.dynamo-diffusion-backend {
  color: var(--diffusion-gray-fg); background: var(--diffusion-gray-bg); border-color: var(--diffusion-gray-border);
}
/* TensorRT-LLM uses NVIDIA green. vLLM-Omni avoids the amber of the experimental badge beside it. */
.dynamo-diffusion-card[data-backend="sglang"] .dynamo-diffusion-backend {
  color: var(--diffusion-sky-fg); background: var(--diffusion-sky-bg); border-color: var(--diffusion-sky-border);
}
.dynamo-diffusion-card[data-backend="trtllm"] .dynamo-diffusion-backend {
  color: var(--diffusion-green-fg); background: var(--diffusion-green-bg); border-color: var(--diffusion-green-border);
}
.dynamo-diffusion-card[data-backend="vllm"] .dynamo-diffusion-backend {
  color: var(--diffusion-pink-fg); background: var(--diffusion-pink-bg); border-color: var(--diffusion-pink-border);
}
.dynamo-diffusion-size {
  color: var(--diffusion-muted); background: transparent; border-color: var(--diffusion-gray-border);
  font-variant-numeric: tabular-nums;
}
/* Dashed amber marks experimental entries, as in the Reference badges and heatmap. */
.dynamo-diffusion-experimental {
  color: var(--diffusion-amber-fg); background: transparent; border: 1.5px dashed var(--diffusion-amber-dash);
}
/* One native, Fern-resolved link covers the entire card. No nested links or controls. */
.dynamo-diffusion .dynamo-diffusion-card-link {
  position: absolute; inset: 0; z-index: 1; border-radius: inherit;
  font-size: 0; color: transparent; text-decoration: none;
}
.dynamo-diffusion .dynamo-diffusion-card-link:focus-visible { outline: 2px solid var(--diffusion-green); outline-offset: 2px; }
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
  .dynamo-diffusion-card, .dynamo-diffusion-reset,
  .dynamo-diffusion-controls input, .dynamo-diffusion-controls select { transition: none; }
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
