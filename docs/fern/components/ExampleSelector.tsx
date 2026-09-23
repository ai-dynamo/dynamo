/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * One selected source bundle, with platform, backend, and optional example rows.
 * Source blocks remain native MDX so Fern embeds files and rewrites their links.
 */
"use client";

import { useState, type ReactNode } from "react";

import { EXAMPLE_BACKENDS, EXAMPLE_PLATFORMS, type ExampleVariant } from "./example-options";
import { EXAMPLES_LAYOUT_CSS } from "./examples-layout";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

const CSS = `
.dynamo-example-viewer { margin-block: 20px; }
.dynamo-example-selector.lqs-panel { margin: 0 0 20px; }
.dynamo-example-selector .lqs-row { grid-template-columns: 92px minmax(0, 1fr); }
.dynamo-example-selector .lqs-row:first-child { border-top: 0; }
.dynamo-example-selector .lqs-chip { max-width: 100%; line-height: 1.3; overflow-wrap: anywhere; }
.dynamo-example-selector .lqs-chip[aria-pressed="true"] { background-color: var(--lqs-stable); }
.dynamo-example-output, .dynamo-example-variant { min-width: 0; max-width: 100%; }
.dynamo-example-variant { display: none; }
.dynamo-example-output pre { max-width: 100%; overflow-x: auto; }
.dynamo-example-output p { overflow-wrap: anywhere; }
.dynamo-example-description { margin: 0 0 14px; font-size: 13px; color: var(--grayscale-a11, inherit); }
.dynamo-example-variant-title { margin: 0 0 12px; font-size: 15px; font-weight: 600; }
@media (max-width: 640px) {
  .dynamo-example-selector .lqs-row { grid-template-columns: minmax(0, 1fr); }
}
`;

type Choice = { id: string; label: string; description?: string; disabled?: boolean };

function ChoiceRow({ label, choices, selected, onSelect }: {
  label: string;
  choices: Choice[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="lqs-row">
      <span className="lqs-label">{label}</span>
      <div className="lqs-options" role="group" aria-label={label}>
        {choices.map((choice) => (
          <button
            key={choice.id}
            type="button"
            className="lqs-chip"
            aria-pressed={selected === choice.id}
            disabled={choice.disabled}
            title={choice.description}
            onClick={() => onSelect(choice.id)}
          >
            {choice.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ExampleSelector({ variants, children }: { variants: ExampleVariant[]; children: ReactNode }) {
  const platforms = Object.keys(EXAMPLE_PLATFORMS).filter((platform) => variants.some((variant) => variant.target.startsWith(`${platform}:`)));
  const backends = Object.keys(EXAMPLE_BACKENDS).filter((backend) => variants.some((variant) => variant.target.endsWith(`:${backend}`)));
  const initial = variants.find((variant) => variant.target === `${platforms[0]}:${backends[0]}`) ?? variants[0];
  const [selectedId, setSelectedId] = useState(initial?.id ?? "");
  const selected = variants.find((variant) => variant.id === selectedId) ?? initial;

  if (!selected) return <div className="dynamo-example-output">{children}</div>;

  const [platform, backend] = selected.target.split(":");
  const choices = variants.filter((variant) => variant.target === selected.target);

  function choosePlatform(next: string) {
    if (next === platform) return;
    const matching = variants.filter((variant) => variant.target.startsWith(`${next}:`));
    const candidate = matching.find((variant) => variant.target.endsWith(`:${backend}`)) ?? matching[0];
    if (candidate) setSelectedId(candidate.id);
  }

  function chooseBackend(next: string) {
    if (next === backend) return;
    const candidate = variants.find((variant) => variant.target === `${platform}:${next}`);
    if (candidate) setSelectedId(candidate.id);
  }

  // These rules render with the server HTML: exactly one bundle is open before
  // hydration, without a flash of every code block or a DOM-filtering effect.
  const selectionCSS = variants.map(({ id }) =>
    `.dynamo-example-viewer[data-selected-example=${JSON.stringify(id)}] .dynamo-example-variant[data-example-variant=${JSON.stringify(id)}] { display: block; }`
  ).join("\n");

  return (
    <div className="dynamo-example-viewer" data-selected-example={selected.id}>
      <style dangerouslySetInnerHTML={{ __html: EXAMPLES_LAYOUT_CSS + LOCAL_SELECTOR_CSS + CSS + selectionCSS }} />
      <div className="lqs-panel dynamo-example-selector">
        <ChoiceRow label="Platform" selected={platform} onSelect={choosePlatform}
          choices={platforms.map((id) => ({ id, label: EXAMPLE_PLATFORMS[id] }))} />
        <ChoiceRow label="Backend" selected={backend} onSelect={chooseBackend}
          choices={backends.map((id) => ({
            id,
            label: EXAMPLE_BACKENDS[id],
            disabled: !variants.some((variant) => variant.target === `${platform}:${id}`),
            description: variants.some((variant) => variant.target === `${platform}:${id}`)
              ? undefined : `No ${EXAMPLE_PLATFORMS[platform]} example for ${EXAMPLE_BACKENDS[id]}.`,
          }))} />
        {choices.length > 1 && (
          <ChoiceRow label="Example" selected={selected.id} onSelect={setSelectedId}
            choices={choices.map(({ id, label, description }) => ({ id, label, description }))} />
        )}
      </div>
      <p className="dynamo-example-description" role="status" aria-live="polite">{selected.description}</p>
      <div className="dynamo-example-output">{children}</div>
    </div>
  );
}
