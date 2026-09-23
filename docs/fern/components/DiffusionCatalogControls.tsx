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
 */
"use client";

import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";

type FilterKey = "type" | "backend" | "provider" | "size" | "status";
type Option = { value: string; label: string };
const EMPTY_FILTERS: Record<FilterKey, string> = { type: "", backend: "", provider: "", size: "", status: "" };
const SORT_OPTIONS = {
  default: "Catalog order",
  "name-asc": "Model name: A–Z",
  "name-desc": "Model name: Z–A",
  type: "Type",
  backend: "Backend",
  provider: "Provider",
  "size-asc": "Weights: smallest first",
  "size-desc": "Weights: largest first",
  status: "Experimental last",
};
type SortOrder = keyof typeof SORT_OPTIONS;
const collator = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

function readCards(grid: HTMLElement) {
  return Array.from(grid.querySelectorAll<HTMLElement>(":scope > .dynamo-diffusion-card"), (node, index) => {
    const text = (selector: string) => node.querySelector(selector)?.textContent?.trim() ?? "";
    const { model = "", source = "", case: type = "", backend = "", sizeGb = "", experimental } = node.dataset;
    return {
      node,
      index,
      name: text(".dynamo-diffusion-name") || model,
      type,
      typeLabel: text(".dynamo-diffusion-modality"),
      backend,
      backendLabel: text(".dynamo-diffusion-backend"),
      provider: text(".dynamo-diffusion-provider"),
      size: sizeGb,
      sizeGB: Number(sizeGb),
      status: experimental === "true" ? "experimental" : "non-experimental",
      // Include the machine-readable type so "text-to-image" and visible chip text both work.
      searchText: `${node.textContent ?? ""} ${model} ${type} ${backend} ${source}`.toLowerCase(),
    };
  });
}

type Card = ReturnType<typeof readCards>[number];

function choices(cards: Card[], value: (card: Card) => string, label = value): Option[] {
  const options = new Map(cards.map((card) => [value(card), label(card)]));
  return Array.from(options, ([value, label]) => ({ value, label }))
    .sort((a, b) => collator.compare(a.label, b.label));
}

function compareCards(a: Card, b: Card, sort: SortOrder): number {
  const byName = collator.compare(a.name, b.name) || a.index - b.index;
  switch (sort) {
    case "name-asc": return byName;
    case "name-desc": return collator.compare(b.name, a.name) || a.index - b.index;
    case "type": return collator.compare(a.typeLabel, b.typeLabel) || byName;
    case "backend": return collator.compare(a.backendLabel, b.backendLabel) || byName;
    case "provider": return collator.compare(a.provider, b.provider) || byName;
    case "size-asc": return a.sizeGB - b.sizeGB || byName;
    case "size-desc": return b.sizeGB - a.sizeGB || byName;
    case "status": return Number(a.status === "experimental") - Number(b.status === "experimental") || byName;
    default: return a.index - b.index;
  }
}

export function DiffusionCatalogControls({ children }: { children: ReactNode }) {
  const id = useId();
  const grid = useRef<HTMLDivElement>(null);
  const [cards, setCards] = useState<Card[] | null>(null);
  const entries = cards ?? [];
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [sort, setSort] = useState<SortOrder>("default");
  const visible = useMemo(() => {
    const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    return (cards ?? []).filter((card) =>
      Object.entries(filters).every(([key, value]) => !value || card[key as FilterKey] === value)
      && words.every((word) => card.searchText.includes(word))
    );
  }, [cards, filters, query]);
  const fields: { key: FilterKey; label: string; all: string; options: Option[] }[] = [
    { key: "type", label: "Type", all: "All types", options: choices(entries, (card) => card.type, (card) => card.typeLabel) },
    { key: "backend", label: "Backend", all: "All backends", options: choices(entries, (card) => card.backend, (card) => card.backendLabel) },
    { key: "provider", label: "Provider", all: "All providers", options: choices(entries, (card) => card.provider) },
    {
      key: "size", label: "Weight size", all: "All sizes",
      options: choices(entries, (card) => card.size, (card) => `${card.size} GB`)
        .sort((a, b) => Number(a.value) - Number(b.value)),
    },
    {
      key: "status", label: "Status", all: "All statuses", options: [
        { value: "non-experimental", label: "Non-experimental" },
        { value: "experimental", label: "Experimental" },
      ],
    },
  ];
  const changed = Boolean(query || Object.values(filters).some(Boolean) || sort !== "default");

  // Fern supplies opaque MDX children. Read the resolved native cards rather
  // than rebuilding their content, image URLs, or version-aware links.
  useEffect(() => {
    if (grid.current) setCards(readCards(grid.current));
  }, [children]);

  useEffect(() => {
    if (!cards) return;
    const matches = new Set(visible);
    cards.forEach((card) => {
      card.node.hidden = !matches.has(card);
    });
  }, [cards, visible]);

  useEffect(() => {
    if (!cards || !grid.current) return;
    // Move existing siblings, not copies or CSS-only positions: keyboard and
    // screen-reader order must agree with the selected visual order.
    const sorted = [...cards].sort((a, b) => compareCards(a, b, sort));
    if (sorted.some(({ node }, index) => grid.current?.children[index] !== node)) {
      sorted.forEach(({ node }) => grid.current?.appendChild(node));
    }
  }, [cards, sort]);

  function reset() {
    setQuery("");
    setFilters(EMPTY_FILTERS);
    setSort("default");
  }

  return (
    <>
      <div className="dynamo-diffusion-controls" role="search" aria-label="Filter diffusion examples">
        <div className="dynamo-diffusion-toolbar">
          <label htmlFor={`${id}-search`}>
            Search examples
            <span className="dynamo-diffusion-search">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
                <circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4.5 4.5" />
              </svg>
              <input id={`${id}-search`} type="search" disabled={cards === null} placeholder="Model, provider, type, backend…" value={query} onChange={(event) => setQuery(event.target.value)} />
            </span>
          </label>
          <label htmlFor={`${id}-sort`}>
            Sort by
            <select id={`${id}-sort`} disabled={cards === null} value={sort} onChange={(event) => setSort(event.target.value as SortOrder)}>
              {Object.entries(SORT_OPTIONS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
        </div>
        <div className="dynamo-diffusion-filters">
          {fields.map(({ key, label, all, options }) => (
            <label key={key} htmlFor={`${id}-${key}`}>
              {label}
              <select id={`${id}-${key}`} disabled={cards === null} value={filters[key]} onChange={(event) => setFilters((current) => ({ ...current, [key]: event.target.value }))}>
                <option value="">{all}</option>
                {options.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
          ))}
        </div>
      </div>
      <div className="dynamo-diffusion-results">
        <span role="status" aria-live="polite" aria-atomic="true">{cards === null ? "Examples" : `${visible.length} of ${cards.length} examples`}</span>
        <button className="dynamo-diffusion-reset" type="button" disabled={!changed} onClick={reset}>Reset filters</button>
      </div>
      <div className="dynamo-diffusion-grid" ref={grid}>{children}</div>
      {cards !== null && visible.length === 0 && (
        <div className="dynamo-diffusion-empty">
          <strong>No matching examples</strong>
          <p>Try another search or reset the filters.</p>
        </div>
      )}
    </>
  );
}
