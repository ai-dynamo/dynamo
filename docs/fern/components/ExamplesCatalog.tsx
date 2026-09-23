/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The cards live in MDX so Fern resolves their links for each version and locale.
 * Filtering uses actual platform:backend pairs, not independent support lists.
 * All cards remain readable without JavaScript.
 */
"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { EXAMPLE_BACKENDS as BACKENDS, EXAMPLE_PLATFORMS as PLATFORMS } from "./example-options";
import { EXAMPLES_LAYOUT_CSS } from "./examples-layout";

const CSS = EXAMPLES_LAYOUT_CSS + `
.dynamo-examples { --example-border: var(--grayscale-a5, #8884); margin-top: 24px; }
.dynamo-examples *, .dynamo-examples *::before, .dynamo-examples *::after { box-sizing: border-box; }
.dynamo-example-controls { padding: 20px; border: 1px solid var(--example-border); border-radius: 12px; background: var(--grayscale-a2, #88808); }
.dynamo-example-controls label { display: flex; flex-direction: column; gap: 7px; min-width: 0; font-size: 12px; font-weight: 600; }
.dynamo-example-controls input, .dynamo-example-controls select { width: 100%; min-width: 0; border: 1px solid var(--example-border); border-radius: 7px; padding: 9px 12px; font: inherit; font-size: 14px; font-weight: 400; color: inherit; background: var(--background, var(--grayscale-a1, transparent)); }
.dynamo-example-controls input:focus-visible, .dynamo-example-controls select:focus-visible, .dynamo-example-reset:focus-visible, .dynamo-example-card a:focus-visible { outline: 2px solid #76b900; outline-offset: 3px; }
.dynamo-example-filters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
.dynamo-example-results { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 16px 0; font-size: 13px; color: var(--pst-color-text-muted, inherit); }
.dynamo-example-reset { border: 1px solid var(--example-border); border-radius: 7px; padding: 6px 10px; background: transparent; color: inherit; font: inherit; cursor: pointer; }
.dynamo-example-reset:hover { border-color: #76b900; }
.dynamo-example-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.dynamo-example-card { min-width: 0; padding: 20px; border: 1px solid var(--example-border); border-radius: 12px; background: var(--grayscale-a1, transparent); transition: border-color .15s; }
.dynamo-example-card[hidden] { display: none !important; }
.dynamo-example-card:hover { border-color: #76b900; }
.dynamo-example-eyebrow { display: block; font-size: 10px; line-height: 1.5; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--pst-color-text-muted, inherit); }
.dynamo-example-card h3 { margin: 10px 0 !important; font-size: 17px !important; line-height: 1.35 !important; }
.dynamo-example-card h3 a { color: inherit; text-decoration: none; }
.dynamo-example-card h3 a:hover { color: #76b900; }
.dynamo-example-card p { margin: 0 0 14px !important; font-size: 13px; line-height: 1.6; color: var(--pst-color-text-muted, inherit); }
.dynamo-example-badges { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }
.dynamo-example-badges span { display: inline-block; max-width: 100%; border-radius: 5px; padding: 4px 7px; background: var(--grayscale-a3, #8881); font-size: 11px; line-height: 1.5; }
.dynamo-example-empty { padding: 28px 20px; border: 1px dashed var(--example-border); border-radius: 10px; text-align: center; }
@media (max-width: 720px) {
  .dynamo-example-filters, .dynamo-example-grid { grid-template-columns: minmax(0, 1fr); }
  .dynamo-example-controls, .dynamo-example-card { padding: 16px; }
}
`;

export function ExamplesCatalog({ children }: { children: ReactNode }) {
  const root = useRef<HTMLElement>(null);
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState("");
  const [platform, setPlatform] = useState("");
  const [backend, setBackend] = useState("");
  const [topics, setTopics] = useState<[string, string][]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [total, setTotal] = useState(0);

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
      const text = `${card.textContent ?? ""} ${card.dataset.keywords ?? ""}`.toLowerCase();
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
      <div className="dynamo-example-controls">
        <label htmlFor="example-search">
          Search examples
          <input id="example-search" type="search" placeholder="Find a topic, feature, or filename…" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="dynamo-example-filters">
          <label htmlFor="example-topic">
            Topic
            <select id="example-topic" value={topic} onChange={(event) => setTopic(event.target.value)}>
              <option value="">All topics</option>
              {topics.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
          <label htmlFor="example-platform">
            Platform
            <select id="example-platform" value={platform} onChange={(event) => setPlatform(event.target.value)}>
              <option value="">All platforms</option>
              {Object.entries(PLATFORMS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
          <label htmlFor="example-backend">
            Backend
            <select id="example-backend" value={backend} onChange={(event) => setBackend(event.target.value)}>
              <option value="">All backends</option>
              {Object.entries(BACKENDS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
        </div>
      </div>
      <div className="dynamo-example-results">
        <span role="status" aria-live="polite">{count === null ? "Examples" : `${count} of ${total} examples`}</span>
        <button className="dynamo-example-reset" type="button" onClick={reset}>Reset filters</button>
      </div>
      <div className="dynamo-example-grid">{children}</div>
      {count === 0 && <p className="dynamo-example-empty">No examples match this combination. Try another platform or backend, or reset the filters.</p>}
    </section>
  );
}
