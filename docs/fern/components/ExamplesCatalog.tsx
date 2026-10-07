/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The cards live in MDX so Fern resolves their links for each version and locale.
 * Filtering uses actual platform:backend pairs, not independent support lists.
 * All cards remain readable without JavaScript.
 */
"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { EXAMPLE_BACKENDS as BACKENDS, EXAMPLE_PLATFORMS as PLATFORMS } from "./example-options";
import { EXAMPLES_LAYOUT_CSS } from "./examples-layout";
import { PlatformGlyph } from "./ExampleTargets";

const CSS = EXAMPLES_LAYOUT_CSS + `
/* Panels, fields, cards, and pill chips match DiffusionCatalog, which mirrors
   the shared site palette: the Reference panels and badges (ReferenceStyles)
   and the main.css .dynamo-chip-* tints. Values stay literal because the
   published theme drops main.css. Each topic carries one accent on the icon
   tile, topic label, arrow, and glow; backend chips use the diffusion
   catalog's per-backend tints; green remains the interactive accent. */
.dynamo-examples {
  --example-text: #1a1a1a;
  --example-muted: #666666;
  --example-panel: #f7f7f7;
  --example-card: #ffffff;
  --example-field: #ffffff;
  --example-border: var(--grayscale-a5, #dfe1df);
  --example-field-border: var(--grayscale-a5, #dfe1df);
  --example-field-hover: rgba(118, 185, 0, 0.55);
  --example-green: #76b900;
  --example-green-fg: #4d7c0f;
  --example-green-bg: rgba(118, 185, 0, 0.16);
  --example-green-border: rgba(118, 185, 0, 0.38);
  --example-blue-fg: #1d4ed8;
  --example-blue-bg: rgba(37, 99, 235, 0.1);
  --example-blue-border: rgba(37, 99, 235, 0.28);
  --example-violet-fg: #6d28d9;
  --example-violet-bg: rgba(124, 58, 237, 0.12);
  --example-violet-border: rgba(124, 58, 237, 0.3);
  --example-teal-fg: #0e7490;
  --example-teal-bg: rgba(8, 145, 178, 0.12);
  --example-teal-border: rgba(8, 145, 178, 0.3);
  --example-orange-fg: #c7401c;
  --example-orange-bg: rgba(233, 84, 32, 0.12);
  --example-orange-border: rgba(233, 84, 32, 0.3);
  --example-pink-fg: #be185d;
  --example-pink-bg: rgba(219, 39, 119, 0.1);
  --example-pink-border: rgba(219, 39, 119, 0.28);
  --example-emerald-fg: #047857;
  --example-emerald-bg: rgba(16, 185, 129, 0.12);
  --example-emerald-border: rgba(16, 185, 129, 0.3);
  --example-purple-fg: #7e22ce;
  --example-purple-bg: rgba(147, 51, 234, 0.1);
  --example-purple-border: rgba(147, 51, 234, 0.3);
  --example-sky-fg: #0369a1;
  --example-sky-bg: rgba(2, 132, 199, 0.1);
  --example-sky-border: rgba(2, 132, 199, 0.28);
  --example-gray-fg: #5f5e5a;
  --example-gray-bg: rgba(120, 120, 120, 0.1);
  --example-gray-border: rgba(120, 120, 120, 0.28);
  margin-block: 24px 20px;
}
.dark .dynamo-examples, [data-theme="dark"] .dynamo-examples {
  color-scheme: dark;
  --example-text: #eeeeee;
  --example-muted: #999999;
  --example-panel: #161616;
  --example-card: #161616;
  --example-field: #1d1d1d;
  --example-border: #2b2b2b;
  --example-field-border: #3a3a3a;
  --example-green-fg: #a3e635;
  --example-green-bg: rgba(118, 185, 0, 0.22);
  --example-green-border: rgba(118, 185, 0, 0.46);
  --example-blue-fg: #93c5fd;
  --example-blue-bg: rgba(59, 130, 246, 0.18);
  --example-blue-border: rgba(59, 130, 246, 0.42);
  --example-violet-fg: #c4b5fd;
  --example-violet-bg: rgba(139, 92, 246, 0.2);
  --example-violet-border: rgba(139, 92, 246, 0.42);
  --example-teal-fg: #67e8f9;
  --example-teal-bg: rgba(34, 211, 238, 0.16);
  --example-teal-border: rgba(34, 211, 238, 0.4);
  --example-orange-fg: #ff9068;
  --example-orange-bg: rgba(233, 84, 32, 0.2);
  --example-orange-border: rgba(233, 84, 32, 0.42);
  --example-pink-fg: #f9a8d4;
  --example-pink-bg: rgba(236, 72, 153, 0.18);
  --example-pink-border: rgba(236, 72, 153, 0.4);
  --example-emerald-fg: #6ee7b7;
  --example-emerald-bg: rgba(16, 185, 129, 0.18);
  --example-emerald-border: rgba(16, 185, 129, 0.42);
  --example-purple-fg: #d8b4fe;
  --example-purple-bg: rgba(168, 85, 247, 0.2);
  --example-purple-border: rgba(168, 85, 247, 0.46);
  --example-sky-fg: #7dd3fc;
  --example-sky-bg: rgba(56, 189, 248, 0.16);
  --example-sky-border: rgba(56, 189, 248, 0.4);
  --example-gray-fg: #a8a8a8;
  --example-gray-bg: #242424;
  --example-gray-border: #383838;
}
.dynamo-examples *, .dynamo-examples *::before, .dynamo-examples *::after { box-sizing: border-box; }
.dynamo-example-controls {
  padding: 20px 22px; border: 1px solid var(--example-border); border-radius: 12px;
  color: var(--example-text); background: var(--example-panel);
}
.dynamo-example-controls label {
  display: flex; flex-direction: column; gap: 7px; min-width: 0;
  color: var(--example-muted); font-size: 11px; font-weight: 700; line-height: 1.3;
  letter-spacing: .08em; text-transform: uppercase;
}
.dynamo-example-controls input, .dynamo-example-controls select {
  width: 100%; min-width: 0; min-height: 38px; padding: 7px 12px;
  border: 1px solid var(--example-field-border); border-radius: 8px;
  color: var(--example-text); background: var(--example-field);
  font: inherit; font-size: 13.5px; font-weight: 400; letter-spacing: normal; text-transform: none;
  transition: border-color .15s ease;
}
.dynamo-example-controls select { padding-right: 34px; cursor: pointer; }
.dynamo-example-controls input:hover, .dynamo-example-controls select:hover { border-color: var(--example-field-hover); }
.dynamo-example-controls input::placeholder { color: var(--example-muted); opacity: 1; }
.dynamo-example-controls input:focus-visible, .dynamo-example-controls select:focus-visible,
.dynamo-example-reset:focus-visible { outline: 2px solid var(--example-green); outline-offset: 2px; }
.dynamo-example-search { position: relative; display: flex; align-items: center; }
.dynamo-example-search svg { position: absolute; left: 12px; color: var(--example-muted); pointer-events: none; }
.dynamo-example-search input { padding-left: 38px; }
.dynamo-example-filters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
.dynamo-example-filters[data-columns="2"] { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.dynamo-example-results {
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 16px 0 14px;
  color: var(--example-muted); font-size: 12.5px; font-variant-numeric: tabular-nums;
}
.dynamo-example-legend { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin-left: auto; }
.dynamo-example-legend > span { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.dynamo-example-legend .dynamo-example-platform { width: 14px; height: 14px; flex-basis: 14px; color: var(--example-text); }
.dynamo-example-reset {
  display: inline-flex; align-items: center; min-height: 28px; padding: 5px 12px;
  border: 1px solid var(--example-border); border-radius: 999px;
  color: var(--example-muted); background: var(--example-panel);
  font: inherit; font-size: 12px; font-weight: 700; line-height: 1; cursor: pointer;
  transition: border-color .15s ease, color .15s ease;
}
.dynamo-example-reset:hover:not(:disabled) { border-color: var(--example-green); color: var(--example-text); }
.dynamo-example-reset:disabled { opacity: .5; cursor: default; }
.dynamo-example-empty {
  padding: 32px 20px; border: 1px dashed var(--example-field-border); border-radius: 12px;
  text-align: center; color: var(--example-text);
}
.dynamo-examples .dynamo-example-empty p { margin: 8px 0 0 !important; font-size: 13px; color: var(--example-muted); }
.dynamo-example-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
/* The card is a grid so the MDX stays flat: icon, optional topic, title, and
   arrow on top, then the description, then the chips pinned to the bottom. */
.dynamo-example-card {
  --example-accent-fg: var(--example-gray-fg);
  --example-accent-bg: var(--example-gray-bg);
  --example-accent-border: var(--example-gray-border);
  position: relative;
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr) 30px;
  grid-template-rows: auto auto 1fr;
  grid-template-areas: "icon title arrow" "desc desc desc" "chips chips chips";
  column-gap: 14px;
  min-width: 0;
  padding: 20px;
  border: 1px solid var(--example-border);
  border-radius: 12px;
  color: var(--example-text);
  background-color: var(--example-card);
  /* Topic glow behind the icon, after the modality glow in DiffusionCatalog. */
  background-image: radial-gradient(240px circle at 44px 44px, color-mix(in srgb, var(--example-accent-bg) 60%, transparent), transparent 70%);
  transition: border-color .18s ease, box-shadow .18s ease;
}
.dynamo-example-card:has(> .dynamo-example-topic) {
  grid-template-rows: auto auto auto 1fr;
  grid-template-areas: "icon topic arrow" "icon title arrow" "desc desc desc" "chips chips chips";
}
.dynamo-example-card[data-topic="basic-serving"] {
  --example-accent-fg: var(--example-green-fg); --example-accent-bg: var(--example-green-bg); --example-accent-border: var(--example-green-border);
}
.dynamo-example-card[data-topic="routing"] {
  --example-accent-fg: var(--example-blue-fg); --example-accent-bg: var(--example-blue-bg); --example-accent-border: var(--example-blue-border);
}
.dynamo-example-card[data-topic="kv-cache-offloading"] {
  --example-accent-fg: var(--example-violet-fg); --example-accent-bg: var(--example-violet-bg); --example-accent-border: var(--example-violet-border);
}
.dynamo-example-card[data-topic="multimodal-serving"] {
  --example-accent-fg: var(--example-teal-fg); --example-accent-bg: var(--example-teal-bg); --example-accent-border: var(--example-teal-border);
}
.dynamo-example-card[data-topic="workloads"] {
  --example-accent-fg: var(--example-orange-fg); --example-accent-bg: var(--example-orange-bg); --example-accent-border: var(--example-orange-border);
}
.dynamo-example-card[data-topic="diffusion"] {
  --example-accent-fg: var(--example-pink-fg); --example-accent-bg: var(--example-pink-bg); --example-accent-border: var(--example-pink-border);
}
.dynamo-example-card[data-topic="autoscaling-and-budgets"] {
  --example-accent-fg: var(--example-emerald-fg); --example-accent-bg: var(--example-emerald-bg); --example-accent-border: var(--example-emerald-border);
}
.dynamo-example-card[data-topic="observability-and-recovery"] {
  --example-accent-fg: var(--example-purple-fg); --example-accent-bg: var(--example-purple-bg); --example-accent-border: var(--example-purple-border);
}
.dynamo-example-card[data-topic="cloud-and-integrations"] {
  --example-accent-fg: var(--example-sky-fg); --example-accent-bg: var(--example-sky-bg); --example-accent-border: var(--example-sky-border);
}
.dynamo-example-card[hidden] { display: none !important; }
.dynamo-example-card:hover { border-color: var(--example-green); box-shadow: 0 12px 28px rgba(0, 0, 0, .09); }
.dark .dynamo-example-card:hover, [data-theme="dark"] .dynamo-example-card:hover { box-shadow: 0 14px 32px rgba(0, 0, 0, .32); }
.dynamo-example-icon {
  grid-area: icon; align-self: center;
  display: grid; place-items: center; width: 48px; height: 48px;
  border: 1px solid var(--example-accent-border); border-radius: 12px;
  color: var(--example-accent-fg); background: var(--example-accent-bg);
}
/* Fern's Icon sets its size inline and its gray color in a cascade layer. */
.dynamo-example-icon svg { display: block; width: 22px !important; height: 22px !important; color: inherit; }
.dynamo-examples .dynamo-example-topic {
  grid-area: topic; align-self: end;
  margin: 0 0 3px !important; color: var(--example-accent-fg);
  font-size: 11px; font-weight: 700; line-height: 1.4; letter-spacing: .08em; text-transform: uppercase;
}
.dynamo-examples .dynamo-example-card h3 {
  grid-area: title; align-self: center;
  margin: 0 !important; font-size: 18px !important; line-height: 1.3 !important;
  font-weight: 650; letter-spacing: -.02em; color: var(--example-text); overflow-wrap: anywhere;
}
.dynamo-examples .dynamo-example-card:has(> .dynamo-example-topic) h3 { align-self: start; }
/* One native, Fern-resolved link covers the card. It stays outside the
   heading, whose Fern click handler copies the anchor instead of navigating. */
.dynamo-examples .dynamo-example-card-link {
  position: absolute; inset: 0; z-index: 1; border-radius: inherit;
  font-size: 0; color: transparent; text-decoration: none;
}
.dynamo-examples .dynamo-example-card-link:focus-visible { outline: 2px solid var(--example-green); outline-offset: 2px; }
.dynamo-example-card::after {
  content: "\\2197"; content: "\\2197" / "";
  grid-area: arrow; align-self: start;
  display: grid; place-items: center; width: 30px; height: 30px;
  border: 1px solid var(--example-accent-border); border-radius: 50%;
  color: var(--example-accent-fg); background: var(--example-accent-bg);
  font-size: 15px; line-height: 1;
}
.dynamo-examples .dynamo-example-card > p:not(.dynamo-example-topic) {
  grid-area: desc; margin: 12px 0 0 !important;
  font-size: 13.5px; line-height: 1.6; color: var(--example-muted);
}
.dynamo-example-chips { grid-area: chips; align-self: end; display: flex; flex-wrap: wrap; gap: 6px; padding-top: 14px; }
.dynamo-example-chip {
  --example-chip-fg: var(--example-gray-fg);
  --example-chip-bg: var(--example-gray-bg);
  --example-chip-border: var(--example-gray-border);
  display: inline-flex; align-items: center; gap: 6px; min-height: 24px; padding: 2px 8px 2px 10px;
  border: 1px solid var(--example-chip-border); border-radius: 999px;
  color: var(--example-chip-fg); background: var(--example-chip-bg);
  font-size: 11.5px; font-weight: 600; line-height: 1.2; white-space: nowrap;
}
/* TensorRT-LLM uses NVIDIA green. Mocker, sample, and custom backends stay neutral. */
.dynamo-example-chip[data-backend="sglang"] {
  --example-chip-fg: var(--example-sky-fg); --example-chip-bg: var(--example-sky-bg); --example-chip-border: var(--example-sky-border);
}
.dynamo-example-chip[data-backend="trtllm"] {
  --example-chip-fg: var(--example-green-fg); --example-chip-bg: var(--example-green-bg); --example-chip-border: var(--example-green-border);
}
.dynamo-example-chip[data-backend="vllm"] {
  --example-chip-fg: var(--example-pink-fg); --example-chip-bg: var(--example-pink-bg); --example-chip-border: var(--example-pink-border);
}
.dynamo-example-chip[data-backend="fastvideo"] {
  --example-chip-fg: var(--example-orange-fg); --example-chip-bg: var(--example-orange-bg); --example-chip-border: var(--example-orange-border);
}
.dynamo-example-chip[data-backend="tritonserver"] {
  --example-chip-fg: var(--example-violet-fg); --example-chip-bg: var(--example-violet-bg); --example-chip-border: var(--example-violet-border);
}
.dynamo-example-chip-platforms {
  display: inline-flex; align-items: center; gap: 4px; padding-left: 6px;
  border-left: 1px solid var(--example-chip-border);
}
.dynamo-example-platform { display: block; flex: 0 0 13px; width: 13px; height: 13px; }
.dynamo-example-sr {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; border: 0;
  overflow: hidden; clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap;
}
@media (max-width: 720px) {
  .dynamo-example-filters, .dynamo-example-filters[data-columns="2"], .dynamo-example-grid { grid-template-columns: minmax(0, 1fr); }
  .dynamo-example-controls { padding: 16px; }
  .dynamo-example-grid { gap: 14px; }
  .dynamo-example-card { padding: 18px; }
}
@media (prefers-reduced-motion: reduce) {
  .dynamo-example-card, .dynamo-example-reset,
  .dynamo-example-controls input, .dynamo-example-controls select { transition: none; }
}
`;

export function ExamplesCatalog({ children, topicFilter = true }: { children: ReactNode; topicFilter?: boolean }) {
  const id = useId();
  const root = useRef<HTMLElement>(null);
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState("");
  const [platform, setPlatform] = useState("");
  const [backend, setBackend] = useState("");
  const [topics, setTopics] = useState<[string, string][]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [total, setTotal] = useState(0);
  const changed = Boolean(query || topic || platform || backend);

  useEffect(() => {
    const cards = Array.from(root.current?.querySelectorAll<HTMLElement>("[data-example]") ?? []);
    const labels = new Map<string, string>();
    cards.forEach((card) => {
      labels.set(card.dataset.topic ?? "", card.dataset.topicLabel ?? "");
    });
    setTopics(Array.from(labels.entries()));
    setTotal(cards.length);
  }, [children]);

  useEffect(() => {
    const cards = Array.from(root.current?.querySelectorAll<HTMLElement>("[data-example]") ?? []);
    const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    let visible = 0;
    cards.forEach((card) => {
      // Subsection pages show no topic label, so search reads it from the metadata.
      const text = `${card.textContent ?? ""} ${card.dataset.topicLabel ?? ""} ${card.dataset.keywords ?? ""}`.toLowerCase();
      const matchesTarget = (card.dataset.targets ?? "").split(/\s+/).some((target) => {
        const [targetPlatform, targetBackend] = target.split(":");
        return (!platform || platform === targetPlatform) && (!backend || backend === targetBackend);
      });
      const matches = (!topic || topic === card.dataset.topic) && matchesTarget && words.every((word) => text.includes(word));
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    setCount(visible);
  }, [children, query, topic, platform, backend]);

  function reset() {
    setQuery("");
    setTopic("");
    setPlatform("");
    setBackend("");
  }

  return (
    <section className="dynamo-examples" ref={root} aria-label="Examples catalog">
      <style dangerouslySetInnerHTML={{ __html: CSS }} />
      <div className="dynamo-example-controls" role="search" aria-label="Filter examples">
        <label htmlFor={`${id}-search`}>
          Search examples
          <span className="dynamo-example-search">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
              <circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4.5 4.5" />
            </svg>
            <input id={`${id}-search`} type="search" placeholder={topicFilter ? "Topic, feature, backend, or filename…" : "Feature, backend, or filename…"} value={query} onChange={(event) => setQuery(event.target.value)} />
          </span>
        </label>
        <div className="dynamo-example-filters" data-columns={topicFilter ? 3 : 2}>
          {topicFilter && (
            <label htmlFor={`${id}-topic`}>
              Topic
              <select id={`${id}-topic`} value={topic} onChange={(event) => setTopic(event.target.value)}>
                <option value="">All topics</option>
                {topics.map(([key, label]) => <option key={key} value={key}>{label}</option>)}
              </select>
            </label>
          )}
          <label htmlFor={`${id}-platform`}>
            Platform
            <select id={`${id}-platform`} value={platform} onChange={(event) => setPlatform(event.target.value)}>
              <option value="">All platforms</option>
              {Object.entries(PLATFORMS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
            </select>
          </label>
          <label htmlFor={`${id}-backend`}>
            Backend
            <select id={`${id}-backend`} value={backend} onChange={(event) => setBackend(event.target.value)}>
              <option value="">All backends</option>
              {Object.entries(BACKENDS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
            </select>
          </label>
        </div>
      </div>
      <div className="dynamo-example-results">
        <span role="status" aria-live="polite" aria-atomic="true">{count === null ? "Examples" : `${count} of ${total} examples`}</span>
        {/* Visual key for the platform glyphs; each chip carries its own screen-reader text. */}
        <span className="dynamo-example-legend" aria-hidden="true">
          {Object.entries(PLATFORMS).map(([key, label]) => (
            <span key={key}><PlatformGlyph platform={key} />{label}</span>
          ))}
        </span>
        <button className="dynamo-example-reset" type="button" disabled={!changed} onClick={reset}>Reset filters</button>
      </div>
      <div className="dynamo-example-grid">{children}</div>
      {count === 0 && (
        <div className="dynamo-example-empty">
          <strong>No matching examples</strong>
          <p>Try another platform or backend, or reset the filters.</p>
        </div>
      )}
    </section>
  );
}
