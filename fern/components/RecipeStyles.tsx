/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Recipe & Feature Benchmark component styles.
 *
 * Delivered as a page-level <style> block (NOT via the docs.yml `css:` field)
 * so it survives the shared NVIDIA global theme, which replaces project `css`
 * at publish. Mirrors the prod-proven pattern in NVIDIA-NeMo/DataDesigner
 * (fern/components/BlogCard.tsx) — same global-theme: nvidia, same no-css
 * constraint, CSS injected this exact way.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. IMPORT it (ambient use is
 * unsupported — renders "Unsupported JSX tag"); the @/ prefix resolves to the
 * fern/ root and is rewritten to a relative path at publish time:
 *   import { RecipeStyles } from "@/components/RecipeStyles";
 * Then place <RecipeStyles /> once, right after the frontmatter, on every
 * recipe/benchmark page (and the two landing READMEs).
 */
const RECIPE_CSS = `
/* Dark-mode variable re-bind.
   The shared NVIDIA theme defines the dark values of --pst-color-text-base,
   --pst-color-text-muted, and --pst-color-surface only under
   html[data-theme="dark"], but Fern's theme toggle flips dark mode with the
   .dark *class* and does NOT set data-theme. So in real dark mode these three
   resolve to their LIGHT values (#1a1a1a text, #666 muted, #f7f7f7 surface),
   which our components use for text/panels — rendering dark-on-dark. Re-bind
   them under .dark so they track the class. !important is required: the
   theme's light-default selector outranks a bare .dark. Scoped safely because
   this stylesheet only loads on recipe/benchmark pages. (--nv-color-bg-default
   never flips even with data-theme; those surfaces keep their own .dark
   background overrides below.) */
.dark {
    --pst-color-text-base: #eee !important;
    --pst-color-text-muted: #999 !important;
    --pst-color-surface: #1a1a1a !important;
}

/* Recipe catalog */
.dynamo-recipe-selector {
    margin: 24px 0;
    padding: 20px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dynamo-recipe-selector-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 16px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-recipe-selector-header h3 {
    margin: 0 0 6px;
}

.dynamo-recipe-selector-header p {
    margin: 0;
    color: var(--pst-color-text-muted);
}

.dynamo-recipe-selector-header a,
.dynamo-recipe-actions a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 36px;
    padding: 8px 12px;
    border: 1px solid var(--nv-color-green);
    border-radius: var(--rounded);
    color: var(--pst-color-text-base);
    font-weight: 600;
    line-height: 1;
    text-decoration: none;
    white-space: nowrap;
}

.dynamo-recipe-eyebrow {
    margin-bottom: 8px !important;
    color: var(--nv-color-green);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-recipe-row {
    display: grid;
    grid-template-columns: 112px 1fr;
    align-items: center;
    gap: 10px 14px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-recipe-row:last-of-type {
    border-bottom: 0;
}

.dynamo-recipe-label {
    color: var(--pst-color-text-muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-recipe-options {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.dynamo-recipe-chip {
    display: inline-flex;
    align-items: center;
    min-height: 32px;
    padding: 7px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded);
    background: var(--nv-color-bg-default);
    color: var(--pst-color-text-base);
    cursor: pointer;
    font: inherit;
    line-height: 1;
}

.dark .dynamo-recipe-chip {
    background: var(--nv-dark-grey-2);
}

.dynamo-recipe-chip.active {
    border-color: var(--nv-color-green);
    box-shadow: 0 0 0 1px var(--nv-color-green);
    font-weight: 700;
}

#provider-all:checked ~ .dynamo-recipe-browser label[for="provider-all"],
#provider-nvidia:checked ~ .dynamo-recipe-browser label[for="provider-nvidia"],
#provider-qwen:checked ~ .dynamo-recipe-browser label[for="provider-qwen"],
#provider-deepseek:checked ~ .dynamo-recipe-browser label[for="provider-deepseek"],
#provider-moonshot:checked ~ .dynamo-recipe-browser label[for="provider-moonshot"],
#provider-meta:checked ~ .dynamo-recipe-browser label[for="provider-meta"],
#provider-openai:checked ~ .dynamo-recipe-browser label[for="provider-openai"],
#provider-zai:checked ~ .dynamo-recipe-browser label[for="provider-zai"],
#runtime-all:checked ~ .dynamo-recipe-browser label[for="runtime-all"],
#runtime-vllm:checked ~ .dynamo-recipe-browser label[for="runtime-vllm"],
#runtime-trtllm:checked ~ .dynamo-recipe-browser label[for="runtime-trtllm"],
#runtime-sglang:checked ~ .dynamo-recipe-browser label[for="runtime-sglang"],
#runtime-dynamo:checked ~ .dynamo-recipe-browser label[for="runtime-dynamo"],
#hardware-all:checked ~ .dynamo-recipe-browser label[for="hardware-all"],
#hardware-h100:checked ~ .dynamo-recipe-browser label[for="hardware-h100"],
#hardware-h200:checked ~ .dynamo-recipe-browser label[for="hardware-h200"],
#hardware-gb200:checked ~ .dynamo-recipe-browser label[for="hardware-gb200"],
#hardware-b200:checked ~ .dynamo-recipe-browser label[for="hardware-b200"],
#technique-all:checked ~ .dynamo-recipe-browser label[for="technique-all"],
#technique-aggregated:checked ~ .dynamo-recipe-browser label[for="technique-aggregated"],
#technique-disaggregated:checked ~ .dynamo-recipe-browser label[for="technique-disaggregated"],
#technique-kv-routing:checked ~ .dynamo-recipe-browser label[for="technique-kv-routing"],
#technique-expert-parallel:checked ~ .dynamo-recipe-browser label[for="technique-expert-parallel"],
#technique-spec-decoding:checked ~ .dynamo-recipe-browser label[for="technique-spec-decoding"],
#technique-embedding-cache:checked ~ .dynamo-recipe-browser label[for="technique-embedding-cache"],
#technique-frontend-decoding:checked ~ .dynamo-recipe-browser label[for="technique-frontend-decoding"],
#workload-all:checked ~ .dynamo-recipe-browser label[for="workload-all"],
#workload-agentic-coding:checked ~ .dynamo-recipe-browser label[for="workload-agentic-coding"],
#workload-long-context-reuse:checked ~ .dynamo-recipe-browser label[for="workload-long-context-reuse"],
#workload-multimodal-reuse:checked ~ .dynamo-recipe-browser label[for="workload-multimodal-reuse"],
#workload-static-generation:checked ~ .dynamo-recipe-browser label[for="workload-static-generation"],
#workload-long-output:checked ~ .dynamo-recipe-browser label[for="workload-long-output"],
#workload-deployment-only:checked ~ .dynamo-recipe-browser label[for="workload-deployment-only"] {
    border-color: var(--nv-color-green);
    box-shadow: 0 0 0 1px var(--nv-color-green);
    font-weight: 700;
}

.dynamo-recipe-chip.disabled {
    opacity: 0.45;
    text-decoration: line-through;
}

.dynamo-recipe-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.dynamo-recipe-matrix {
    display: grid;
    margin: 24px 0;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    overflow: hidden;
}

.dynamo-recipe-matrix-row {
    display: grid;
    grid-template-columns: 1.2fr 1.1fr 0.95fr 0.8fr 1.8fr;
    gap: 16px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-recipe-matrix-row:last-child {
    border-bottom: 0;
}

.dynamo-recipe-matrix-row.header {
    background: var(--pst-color-surface);
    font-weight: 700;
}

.dynamo-recipe-matrix-row span,
.dynamo-recipe-matrix-row a {
    min-width: 0;
}

.dynamo-recipe-browser {
    margin: 24px 0;
    padding: 18px 20px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dynamo-recipe-browser-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
    gap: 20px;
    align-items: center;
    padding-bottom: 16px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-recipe-browser-header h3 {
    margin: 0 0 6px;
}

.dynamo-recipe-browser-header p {
    margin: 0;
    color: var(--pst-color-text-muted);
}

.dynamo-filter-input {
    /* display:none (not visually-hidden positioning): the inputs sit at the
       top of the form, so if they can receive focus, clicking any filter
       label scrolls the page back to the top. Label activation still toggles
       undisplayed inputs, and the :checked sibling selectors still match. */
    display: none;
}

.dynamo-search-box {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    align-items: center;
    min-height: 42px;
    padding: 0 14px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded);
    background: var(--nv-color-bg-default);
    color: var(--pst-color-text-muted);
}

.dynamo-search-box span {
    margin-right: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-search-box input {
    min-width: 0;
    border: 0;
    outline: 0;
    background: transparent;
    color: var(--pst-color-text-base);
    font: inherit;
}

.dynamo-search-box input::placeholder {
    color: var(--pst-color-text-muted);
}

.dark .dynamo-search-box {
    background: var(--nv-dark-grey-2);
}

.dynamo-catalog-facts {
    display: grid;
    justify-items: end;
    gap: 4px;
    min-width: 160px;
    color: var(--pst-color-text-muted);
}

.dynamo-catalog-facts strong {
    color: var(--pst-color-text-base);
    font-size: 28px;
    line-height: 1;
}

.dynamo-catalog-facts span {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-filter-reset {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 5px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded);
    background: var(--pst-color-surface);
    color: var(--pst-color-text-muted);
    font: inherit;
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
    cursor: pointer;
}

.dynamo-filter-reset:hover {
    border-color: var(--nv-color-green);
    color: var(--pst-color-text-base);
}

.dynamo-config-value {
    display: inline-flex;
    align-items: center;
    min-height: 0;
    padding: 2px 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    color: var(--pst-color-text-base);
    font-weight: 700;
    line-height: 1.35;
    cursor: default;
}

.dynamo-config-value:not(:last-child)::after {
    content: ",";
    margin-right: 2px;
    color: var(--pst-color-text-muted);
    font-weight: 400;
}

.dynamo-filter-row {
    display: grid;
    grid-template-columns: 96px 1fr;
    gap: 10px 12px;
    align-items: center;
    padding: 8px 0;
}

.dynamo-filter-options {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.dynamo-traffic-facts {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}

.dynamo-traffic-facts div {
    min-height: 84px;
    padding: 10px 12px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--nv-color-bg-default);
}

.dark .dynamo-traffic-facts div {
    background: var(--nv-dark-grey-2);
}

.dynamo-traffic-facts span {
    display: block;
    margin-bottom: 4px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.2;
    text-transform: uppercase;
}

.dynamo-traffic-facts strong {
    display: block;
    color: var(--pst-color-text-base);
    font-size: 15px;
    line-height: 1.25;
}

.dynamo-traffic-facts p {
    margin: 6px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.35;
}

.dynamo-model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(184px, 1fr));
    gap: 10px;
    margin: 18px 0 28px;
}

.dynamo-model-card {
    position: relative;
    display: flex;
    flex-direction: column;
    min-height: 138px;
    padding: 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--nv-color-bg-default);
    color: var(--pst-color-text-base);
    text-decoration: none;
}

/* --nv-color-bg-default does not flip in the dark theme (stays #FFFFFF), so
   in dark mode the card would be a white tile with light text. Flip the
   surface to match the other dark-aware components (chip, search box, etc.). */
.dark .dynamo-model-card {
    background: var(--nv-dark-grey-2);
}

.dynamo-model-card:hover {
    border-color: var(--nv-color-green);
    text-decoration: none;
}

/* Stretched-link: the card is a <div> (so its data-* still drive the CSS
   filter and the rich markup is never reprocessed); this single inner link
   covers the whole tile and is the one Fern-resolved, version-safe href. */
.dynamo-card-link {
    position: absolute;
    inset: 0;
    z-index: 1;
    font-size: 0;
    color: transparent;
}

.dynamo-card-link:focus-visible {
    outline: 2px solid var(--nv-color-green);
    outline-offset: 2px;
}

.dynamo-model-card-link {
    display: contents;
    color: inherit;
    text-decoration: none;
}

.dynamo-model-card.active {
    border-color: var(--nv-color-green);
    box-shadow: 0 0 0 1px var(--nv-color-green);
}

.dynamo-model-card > svg {
    display: none;
}

.dynamo-model-card-top {
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    gap: 8px;
    align-items: start;
}

.dynamo-model-card-top > div:nth-child(2) {
    min-width: 0;
}

.dynamo-model-card-top h3 {
    margin: 0;
    font-size: 14px;
    line-height: 1.25;
    overflow-wrap: break-word;
}

.dynamo-model-card-top p {
    margin: 2px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.25;
}

.dynamo-model-logo,
.dynamo-model-mark {
    width: 28px;
    height: 28px;
    border-radius: 8px;
}

.dynamo-model-logo {
    display: block;
    object-fit: cover;
    margin: 0 !important;
    border: 1px solid var(--border, var(--grayscale-a5));
    background: #fff;
}

.dynamo-model-logo.large {
    width: 56px;
    height: 56px;
    min-width: 56px;
    flex: 0 0 56px;
    border-radius: 12px;
}

.dynamo-model-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--nv-color-green);
    color: var(--nv-color-black);
    font-weight: 800;
}

.dynamo-recipe-status {
    display: inline-flex;
    grid-column: 2;
    justify-self: start;
    align-items: center;
    min-height: 20px;
    padding: 3px 6px;
    border-radius: var(--rounded);
    font-size: 11px;
    font-weight: 700;
    line-height: 1;
    white-space: nowrap;
}

.dynamo-recipe-status.verified {
    background: rgba(118, 185, 0, 0.16);
    color: var(--nv-color-green-2);
}

.dark .dynamo-recipe-status.verified {
    color: var(--nv-color-green);
}

.dynamo-recipe-status.preview {
    background: rgba(255, 184, 77, 0.2);
    color: #8a5200;
}

.dark .dynamo-recipe-status.preview {
    color: #ffbd5a;
}

.dynamo-recipe-status.deployment {
    background: rgba(127, 127, 127, 0.16);
    color: var(--pst-color-text-muted);
}

.dynamo-model-summary {
    display: -webkit-box;
    margin: 8px 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.4;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    overflow: hidden;
}

.dynamo-model-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: auto;
}

.dynamo-model-tags span {
    display: inline-flex;
    align-items: center;
    min-height: 20px;
    padding: 3px 6px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded);
    color: var(--pst-color-text-muted);
    font-size: 10.5px;
    line-height: 1;
}

.dynamo-model-card[hidden],
.dynamo-recipe-empty[hidden] {
    display: none !important;
}

#provider-nvidia:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="nvidia"]),
#provider-qwen:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="qwen"]),
#provider-deepseek:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="deepseek"]),
#provider-moonshot:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="moonshot"]),
#provider-meta:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="meta"]),
#provider-openai:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="openai"]),
#provider-zai:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-provider~="zai"]),
#runtime-vllm:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-runtime~="vllm"]),
#runtime-trtllm:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-runtime~="trtllm"]),
#runtime-sglang:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-runtime~="sglang"]),
#runtime-dynamo:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-runtime~="dynamo"]),
#hardware-h100:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-hardware~="h100"]),
#hardware-h200:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-hardware~="h200"]),
#hardware-gb200:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-hardware~="gb200"]),
#hardware-b200:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-hardware~="b200"]),
#technique-aggregated:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="aggregated"]),
#technique-disaggregated:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="disaggregated"]),
#technique-kv-routing:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="kv-routing"]),
#technique-expert-parallel:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="expert-parallel"]),
#technique-spec-decoding:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="spec-decoding"]),
#technique-embedding-cache:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="embedding-cache"]),
#technique-frontend-decoding:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-technique~="frontend-decoding"]),
#workload-agentic-coding:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="agentic-coding"]),
#workload-long-context-reuse:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="long-context-reuse"]),
#workload-multimodal-reuse:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="multimodal-reuse"]),
#workload-static-generation:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="static-generation"]),
#workload-long-output:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="long-output"]),
#workload-deployment-only:checked ~ .dynamo-model-grid [data-recipe-card]:not([data-workload~="deployment-only"]) {
    display: none;
}

.dynamo-recipe-empty {
    margin: 20px 0 32px;
    padding: 18px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    color: var(--pst-color-text-muted);
    text-align: center;
}

.dynamo-selected-model {
    margin: 20px 0 28px;
    padding: 20px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dynamo-selected-model-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 16px;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-selected-model-header h3 {
    margin: 0 0 6px;
}

.dynamo-selected-model-header p {
    margin: 0;
    color: var(--pst-color-text-muted);
}

.dynamo-selected-model-header a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 36px;
    padding: 8px 12px;
    border: 1px solid var(--nv-color-green);
    border-radius: var(--rounded);
    color: var(--pst-color-text-base);
    font-weight: 600;
    line-height: 1;
    text-decoration: none;
    white-space: nowrap;
}

.dynamo-selected-facts {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.dynamo-selected-facts div {
    padding: 12px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--nv-color-bg-default);
}

.dark .dynamo-selected-facts div {
    background: var(--nv-dark-grey-2);
}

.dynamo-selected-facts span,
.dynamo-evidence-grid span {
    display: block;
    margin-bottom: 4px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-selected-facts strong,
.dynamo-evidence-grid strong {
    display: block;
}

.dynamo-variant-table-wrap {
    margin: 18px 0 24px;
    overflow-x: auto;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 10px;
    background: var(--nv-color-bg-default);
}

.dark .dynamo-variant-table-wrap {
    background: var(--nv-dark-grey-2);
}

.dynamo-variant-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.dynamo-variant-table th,
.dynamo-variant-table td {
    padding: 12px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    text-align: left;
    vertical-align: top;
}

.dynamo-variant-table tbody tr:last-child td {
    border-bottom: 0;
}

.dynamo-variant-table th {
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-variant-table td strong {
    display: block;
    color: var(--pst-color-text-base);
}

.dynamo-variant-table td span {
    display: block;
    margin-top: 4px;
    color: var(--pst-color-text-muted);
}

.dynamo-variant-table td em {
    display: inline-flex;
    align-items: center;
    min-height: 20px;
    margin: 6px 0 0;
    padding: 2px 7px;
    border: 1px solid color-mix(in srgb, var(--nv-color-green) 65%, var(--border, var(--grayscale-a5)));
    border-radius: var(--rounded);
    color: var(--pst-color-text-base);
    font-size: 11px;
    font-style: normal;
    font-weight: 700;
    line-height: 1;
}

.dynamo-variant-table td em.role {
    margin: 0;
    border-color: var(--border, var(--grayscale-a5));
    background: var(--pst-color-surface);
    color: var(--pst-color-text-muted);
}

.dynamo-variant-table td em.recommended,
.dynamo-variant-table td em.recipe,
.dynamo-variant-table td em.benchmark-backed {
    border-color: color-mix(in srgb, var(--nv-color-green) 65%, var(--border, var(--grayscale-a5)));
    color: var(--pst-color-text-base);
}

.dynamo-variant-table td em.control,
.dynamo-variant-table td em.baseline {
    border-color: color-mix(in srgb, #7c8a99 55%, var(--border, var(--grayscale-a5)));
}

.dynamo-variant-table td em.deploy-only {
    border-style: dashed;
}

.dynamo-variant-table td em.winner {
    border-color: color-mix(in srgb, var(--nv-color-green) 80%, var(--border, var(--grayscale-a5)));
    color: var(--pst-color-text-base);
    font-weight: 600;
}

.dynamo-variant-table td em.treatment {
    border-color: color-mix(in srgb, #5b8def 55%, var(--border, var(--grayscale-a5)));
    color: var(--pst-color-text-base);
}

.dynamo-variant-table td mark {
    padding: 2px 6px;
    border-radius: var(--rounded);
    background: var(--pst-color-surface);
    color: var(--pst-color-text-base);
    font-size: 12px;
    white-space: nowrap;
}

.dynamo-variant-table tr.recommended {
    box-shadow: inset 3px 0 0 var(--nv-color-green);
}

.dynamo-variant-table tr.deploy-only {
    box-shadow: inset 3px 0 0 var(--pst-color-text-muted);
}

.dynamo-variant-table tr.recommended td:first-child strong {
    color: var(--pst-color-text-base);
}

.dynamo-variant-table a {
    color: var(--pst-color-text-base);
    font-weight: 700;
    text-decoration: none;
}

.dynamo-variant-table a:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green);
}

.dynamo-evidence-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 20px 0;
}

.dynamo-comparison-brief {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 12px;
    margin: 16px 0 20px;
}

.dynamo-comparison-brief div,
.dynamo-placeholder-note {
    padding: 14px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--pst-color-surface);
}

.dark .dynamo-comparison-brief div,
.dark .dynamo-placeholder-note {
    background: var(--nv-dark-grey-2);
}

.dynamo-comparison-brief span {
    display: block;
    margin-bottom: 6px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-comparison-brief strong {
    display: block;
    color: var(--pst-color-text-base);
    font-size: 15px;
    line-height: 1.45;
}

.dynamo-placeholder-note {
    margin: 16px 0;
    box-shadow: inset 3px 0 0 var(--pst-color-text-muted);
}

.dynamo-placeholder-note strong {
    display: block;
    margin-bottom: 6px;
}

.dynamo-placeholder-note p {
    margin: 0;
    color: var(--pst-color-text-muted);
}

.dynamo-evidence-grid div {
    padding: 14px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
}

.dynamo-evidence-grid p {
    margin: 8px 0 0;
    color: var(--pst-color-text-muted);
}

.dynamo-model-detail-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    gap: 20px;
    align-items: start;
    margin: 24px 0 32px;
    padding: 0;
}

.dynamo-model-detail-hero h2 {
    margin: 0 0 10px;
}

.dynamo-model-detail-hero p {
    color: var(--pst-color-text-muted);
}

.dynamo-model-hero-copy,
.dynamo-recommendation-card {
    padding: 22px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dynamo-model-hero-copy {
    display: flex;
    flex-direction: column;
}

.dynamo-model-title-lockup {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 12px;
}

.dynamo-model-title-lockup .dynamo-recipe-eyebrow {
    margin: 0 0 4px !important;
}

.dynamo-model-title-lockup h2 {
    margin-bottom: 0;
}

.dynamo-hero-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 18px;
}

.dynamo-hero-actions a {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 36px;
    padding: 8px 12px;
    border: 1px solid var(--nv-color-green);
    border-radius: var(--rounded);
    color: var(--pst-color-text-base);
    font-weight: 600;
    line-height: 1;
    text-decoration: none;
    white-space: nowrap;
}

.dynamo-recommendation-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.dark .dynamo-model-hero-copy,
.dark .dynamo-recommendation-card {
    background: var(--nv-dark-grey-2);
}

.dynamo-recommendation-card > span,
.dynamo-prereq-grid span {
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-recommendation-card > strong {
    font-size: 22px;
}

.dynamo-recommendation-card p {
    margin: 0;
}

.dynamo-recommendation-facts {
    display: grid;
    gap: 8px;
    margin-top: 8px;
    padding-top: 12px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-recommendation-facts div {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
}

.dark .dynamo-recommendation-facts div {
    background: transparent;
}

.dynamo-recommendation-facts span {
    display: block;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-recommendation-facts strong {
    text-align: right;
}

.dynamo-unvalidated-note {
    margin: 6px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 13px;
}

.dynamo-prereq-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 20px 0;
}

.dynamo-prereq-grid div {
    padding: 14px;
    border: 1px solid var(--nv-color-green);
    border-radius: 8px;
}

.dynamo-prereq-grid strong {
    display: block;
}

.dynamo-recipe-target-table {
    min-width: 0;
    table-layout: fixed;
}

.dynamo-recipe-target-table th:nth-child(1),
.dynamo-recipe-target-table td:nth-child(1) {
    width: 26%;
}

.dynamo-recipe-target-table th:nth-child(2),
.dynamo-recipe-target-table td:nth-child(2) {
    width: 12%;
}

.dynamo-recipe-target-table th:nth-child(3),
.dynamo-recipe-target-table td:nth-child(3) {
    width: 14%;
}

.dynamo-recipe-target-table th:nth-child(4),
.dynamo-recipe-target-table td:nth-child(4) {
    width: 18%;
}

.dynamo-recipe-target-table th:nth-child(5),
.dynamo-recipe-target-table td:nth-child(5) {
    width: 16%;
}

.dynamo-recipe-target-table th:nth-child(6),
.dynamo-recipe-target-table td:nth-child(6) {
    width: 14%;
    overflow-wrap: normal;
}

.dynamo-benchmark-index {
    display: grid;
    gap: 10px;
    margin: 24px 0 28px;
}

.dynamo-benchmark-lane,
.dynamo-benchmark-row,
.dynamo-benchmark-card,
.dynamo-benchmark-hero,
.dynamo-benchmark-facts div {
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--pst-color-surface);
}

.dark .dynamo-benchmark-lane,
.dark .dynamo-benchmark-row,
.dark .dynamo-benchmark-card,
.dark .dynamo-benchmark-hero,
.dark .dynamo-benchmark-facts div {
    background: var(--nv-dark-grey-2);
}

.dynamo-benchmark-lane {
    padding: 18px;
}

.dynamo-benchmark-lane h2 {
    margin: 0 0 8px;
}

.dynamo-benchmark-lane p:last-child {
    margin-bottom: 0;
    color: var(--pst-color-text-muted);
}

.dynamo-benchmark-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin: 24px 0 28px;
}

.dynamo-benchmark-index-header,
.dynamo-benchmark-row {
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(128px, 0.75fr) minmax(130px, 0.8fr) minmax(150px, 0.95fr) 34px;
    gap: 10px;
    align-items: center;
}

.dynamo-benchmark-index-header {
    padding: 0 14px 2px;
    color: var(--pst-color-text-muted);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-benchmark-row {
    position: relative;
    min-height: 108px;
    padding: 13px 14px;
    color: var(--pst-color-text-base);
    text-decoration: none;
}

.dynamo-benchmark-row:hover,
.dynamo-benchmark-row:focus-visible {
    border-color: var(--nv-color-green);
    text-decoration: none;
}

.dynamo-benchmark-row:focus-visible {
    outline: 2px solid color-mix(in srgb, var(--nv-color-green) 72%, transparent);
    outline-offset: 2px;
}

.dynamo-benchmark-main h2 {
    margin: 0 0 4px;
    color: var(--pst-color-text-base);
    font-size: 16px;
    line-height: 1.22;
}

.dynamo-benchmark-main p {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.35;
}

.dynamo-technique-guide {
    margin: 24px 0 26px;
    padding: 0;
    border: 0;
    background: transparent;
}

.dark .dynamo-technique-guide {
    background: transparent;
}

.dynamo-technique-guide-header {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-technique-guide-header h2 {
    margin: 0;
    font-size: 22px;
    line-height: 1.2;
}

.dynamo-technique-guide-header p {
    max-width: 520px;
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 14px;
    line-height: 1.45;
}

.dynamo-technique-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    column-gap: 28px;
    row-gap: 0;
}

.dynamo-technique-card {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 12px;
    align-items: start;
    min-width: 0;
    padding: 12px 0;
    border: 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    cursor: default;
}

.dark .dynamo-technique-card {
    background: transparent;
}

.dynamo-technique-card:nth-last-child(-n + 2) {
    border-bottom: 0;
}

.dynamo-technique-card .dynamo-feature-symbol {
    flex: 0 0 auto;
    width: 34px;
    height: 34px;
    border-color: transparent;
    border-radius: 0;
    background: transparent;
}

.dark .dynamo-technique-card .dynamo-feature-symbol {
    background: transparent;
}

.dynamo-technique-card .dynamo-feature-icon {
    width: 28px;
    height: 28px;
}

.dynamo-technique-card strong {
    display: block;
    margin: 0 0 3px;
    color: var(--pst-color-text-base);
    font-size: 14px;
    line-height: 1.25;
}

.dynamo-technique-card p {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 11.5px;
    line-height: 1.45;
}

.dynamo-benchmark-card {
    display: flex;
    flex-direction: column;
    min-height: 340px;
    padding: 16px;
    color: var(--pst-color-text-base);
    text-decoration: none;
}

.dynamo-benchmark-card:hover {
    border-color: var(--nv-color-green);
    text-decoration: none;
}

.dynamo-benchmark-kind,
.dynamo-benchmark-facts span {
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-benchmark-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    min-height: 0;
    margin-bottom: 14px;
}

.dynamo-feature-sprite {
    position: absolute;
    width: 0;
    height: 0;
    overflow: hidden;
}

.dynamo-feature-symbol-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    min-width: 0;
}

.dynamo-feature-symbol {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    border: 1px solid color-mix(in srgb, var(--nv-color-green) 55%, var(--border, var(--grayscale-a5)));
    border-radius: 12px;
    background:
        linear-gradient(135deg, color-mix(in srgb, var(--nv-color-green) 12%, transparent), transparent 58%),
        var(--nv-color-bg-default);
    color: var(--pst-color-text-base);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.02em;
    overflow: hidden;
}

.dark .dynamo-feature-symbol {
    background:
        linear-gradient(135deg, color-mix(in srgb, var(--nv-color-green) 16%, transparent), transparent 58%),
        var(--nv-dark-grey-1);
}

.dynamo-feature-symbol::before,
.dynamo-feature-symbol::after {
    content: "";
    position: absolute;
    opacity: 0.55;
    display: none;
}

.dynamo-feature-icon {
    position: relative;
    z-index: 1;
    width: 34px;
    height: 34px;
    color: var(--pst-color-text-base);
    overflow: visible;
}

.dark .dynamo-feature-icon {
    color: var(--pst-color-text-base);
}

.dynamo-feature-symbol.feature-kv::before {
    width: 30px;
    height: 2px;
    background: var(--nv-color-green);
    transform: rotate(-28deg);
}

.dynamo-feature-symbol.feature-kv::after {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    box-shadow: -14px 8px 0 var(--nv-color-green), 12px -8px 0 var(--nv-color-green), 12px 10px 0 var(--nv-color-green);
    background: var(--nv-color-green);
}

.dynamo-feature-symbol.feature-pd::before {
    width: 28px;
    height: 26px;
    border-left: 2px solid var(--nv-color-green);
    border-right: 2px solid var(--nv-color-green);
    border-radius: 8px;
}

.dynamo-feature-symbol.feature-pd::after {
    width: 30px;
    height: 2px;
    background: var(--nv-color-green);
}

.dynamo-feature-symbol.feature-ep::before {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    box-shadow: -11px -9px 0 var(--nv-color-green), 0 -9px 0 var(--nv-color-green), 11px -9px 0 var(--nv-color-green), -11px 9px 0 var(--nv-color-green), 0 9px 0 var(--nv-color-green), 11px 9px 0 var(--nv-color-green);
    background: var(--nv-color-green);
}

.dynamo-feature-symbol.feature-sp::before {
    width: 13px;
    height: 28px;
    background: var(--nv-color-green);
    clip-path: polygon(58% 0, 100% 0, 58% 42%, 100% 42%, 28% 100%, 42% 55%, 0 55%);
}

.dynamo-feature-symbol.feature-ec::before {
    width: 26px;
    height: 18px;
    border: 2px solid var(--nv-color-green);
    border-radius: 5px;
    box-shadow: 5px 5px 0 color-mix(in srgb, var(--nv-color-green) 45%, transparent);
}

.dynamo-feature-symbol.feature-of::before {
    width: 28px;
    height: 16px;
    border: 2px solid var(--nv-color-green);
    border-top: 0;
    border-radius: 0 0 7px 7px;
}

.dynamo-feature-symbol.feature-of::after {
    width: 18px;
    height: 2px;
    background: var(--nv-color-green);
    transform: translateY(-11px);
}

.dynamo-feature-symbol.feature-fd::before {
    width: 26px;
    height: 18px;
    border: 2px solid var(--nv-color-green);
    border-radius: 9px;
}

.dynamo-feature-symbol.feature-fd::after {
    width: 20px;
    height: 2px;
    background: var(--nv-color-green);
    box-shadow: 0 -6px 0 var(--nv-color-green), 0 6px 0 var(--nv-color-green);
}

.dynamo-feature-symbol.feature-mn::before {
    width: 26px;
    height: 22px;
    border-top: 2px solid var(--nv-color-green);
    border-bottom: 2px solid var(--nv-color-green);
    transform: rotate(35deg);
}

.dynamo-feature-symbol.feature-mn::after {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--nv-color-green);
    box-shadow: -13px -10px 0 var(--nv-color-green), 13px -10px 0 var(--nv-color-green), -13px 10px 0 var(--nv-color-green), 13px 10px 0 var(--nv-color-green);
}

.dynamo-benchmark-model {
    display: grid;
    grid-template-columns: 38px minmax(0, 1fr);
    gap: 8px;
    align-items: center;
    min-width: 0;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
}

.dark .dynamo-benchmark-model {
    background: transparent;
}

.dynamo-benchmark-model .dynamo-model-logo {
    width: 38px;
    height: 38px;
    border-radius: 9px;
}

.dynamo-benchmark-model strong {
    display: block;
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 13px;
    line-height: 1.2;
    overflow-wrap: break-word;
}

.dynamo-benchmark-kind {
    display: block;
    margin: 0 0 4px;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
}

.dark .dynamo-benchmark-kind {
    background: transparent;
}

.dynamo-benchmark-card h2 {
    margin: 0 0 6px;
    font-size: 18px;
    line-height: 1.25;
}

.dynamo-benchmark-card p {
    margin: 0 0 10px;
    color: var(--pst-color-text-muted);
    font-size: 14px;
    line-height: 1.45;
}

.dynamo-benchmark-card small {
    margin-top: auto;
    color: var(--pst-color-text-muted);
}

.dynamo-benchmark-features {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 0;
    padding: 0;
}

.dynamo-feature-pair {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    min-height: 24px;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    color: var(--pst-color-text-base);
    font-size: 11.5px;
    font-weight: 700;
    line-height: 1.2;
    pointer-events: none;
}

.dark .dynamo-feature-pair {
    background: transparent;
}

.dynamo-feature-pair .dynamo-feature-symbol {
    flex: 0 0 auto;
    width: 20px;
    height: 20px;
    border: 0;
    border-radius: 999px;
    background: color-mix(in srgb, var(--nv-color-green) 18%, var(--nv-color-bg-default));
}

.dark .dynamo-feature-pair .dynamo-feature-symbol {
    background: color-mix(in srgb, var(--nv-color-green) 24%, var(--nv-dark-grey-1));
}

.dynamo-feature-pair .dynamo-feature-icon {
    width: 17px;
    height: 17px;
}

.dynamo-benchmark-target {
    display: grid;
    gap: 6px;
    min-width: 0;
}

.dynamo-benchmark-target span {
    display: grid;
    gap: 1px;
    min-width: 0;
}

.dynamo-benchmark-target b {
    color: var(--pst-color-text-muted);
    font-size: 10px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-benchmark-target strong {
    color: var(--pst-color-text-base);
    font-size: 12.5px;
    line-height: 1.2;
    overflow-wrap: break-word;
}

.dynamo-benchmark-open {
    justify-self: end;
    color: color-mix(in srgb, var(--nv-color-green) 72%, var(--pst-color-text-base));
    font-size: 12.5px;
    font-weight: 800;
    white-space: nowrap;
}

.dynamo-benchmark-open::after {
    content: " ->";
}

.dynamo-benchmark-targets {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
}

.dynamo-benchmark-targets span {
    display: grid;
    gap: 3px;
    min-width: 0;
    padding: 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--nv-color-bg-default);
}

.dark .dynamo-benchmark-targets span {
    background: var(--nv-dark-grey-1);
}

.dynamo-benchmark-targets b {
    color: var(--pst-color-text-muted);
    font-size: 10px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dynamo-benchmark-targets strong {
    color: var(--pst-color-text-base);
    font-size: 14px;
    line-height: 1.25;
    overflow-wrap: break-word;
}

.dynamo-benchmark-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 18px;
    align-items: start;
    margin: 24px 0 30px;
    padding: 22px;
}

.dynamo-benchmark-hero h2 {
    margin: 0 0 10px;
}

.dynamo-benchmark-facts {
    display: grid;
    gap: 10px;
}

.dynamo-benchmark-facts div {
    padding: 12px;
}

.dynamo-benchmark-facts strong {
    display: block;
    margin-top: 4px;
}

/* Mobile Styles */
@media (max-width: 768px) {
    .hero-section {
        padding: 2rem 1.5rem;
    }

    .hero-title-section {
        margin-bottom: 2rem;
    }

    .hero-heading {
        font-size: 32px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

    .hero-content-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
    }

    .hero-column-title {
        font-size: 20px;
    }

    .hero-column-subtitle {
        font-size: 14px;
    }

    .hero-card-content {
        flex-direction: column;
        align-items: flex-start;
    }

    .hero-card-button-wrapper {
        align-self: flex-start;
    }

    .hero-column .card-icon,
    .hero-column .card-icon svg,
    .hero-column .card-icon i {
        font-size: 40px !important;
        width: 40px !important;
        height: 40px !important;
    }

    .hero-column .fern-card-title {
        font-size: 14px;
    }

    .hero-column .fern-card p {
        font-size: 11px;
    }

    .body-section {
        padding: 2rem 1.5rem;
    }

    .fern-selection-item-icon.use-icon {
        display: none !important;
    }

    .dynamo-recipe-selector {
        padding: 16px;
    }

    .dynamo-recipe-selector-header {
        flex-direction: column;
    }

    .dynamo-recipe-row {
        grid-template-columns: 1fr;
    }

    .dynamo-recipe-matrix {
        border: 0;
        gap: 12px;
        overflow: visible;
    }

    .dynamo-recipe-matrix-row,
    .dynamo-recipe-matrix-row.header {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        padding: 16px;
        border: 1px solid var(--border, var(--grayscale-a5));
        border-radius: 8px;
        background: transparent;
    }

    .dynamo-recipe-matrix-row.header {
        display: none;
    }

    .dynamo-recipe-matrix-row span::before {
        content: attr(data-label);
        display: block;
        margin-bottom: 2px;
        color: var(--pst-color-text-muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .dynamo-recipe-browser {
        padding: 16px;
    }

    .dynamo-catalog-facts {
        justify-items: start;
    }

    .dynamo-recipe-browser-header,
    .dynamo-filter-row,
    .dynamo-traffic-facts,
    .dynamo-model-grid,
    .dynamo-selected-facts,
    .dynamo-comparison-brief,
    .dynamo-evidence-grid,
    .dynamo-technique-grid,
    .dynamo-benchmark-grid,
    .dynamo-benchmark-index,
    .dynamo-benchmark-hero {
        grid-template-columns: 1fr;
    }

    .dynamo-variant-table {
        min-width: 720px;
    }

    .dynamo-model-card {
        min-height: 0;
    }

    .dynamo-model-card-top {
        grid-template-columns: 40px minmax(0, 1fr);
    }

    .dynamo-recipe-status {
        grid-column: 2;
        justify-self: start;
    }

    .dynamo-selected-model {
        padding: 16px;
    }

    .dynamo-selected-model-header {
        grid-template-columns: 1fr;
        flex-direction: column;
    }

    .dynamo-model-detail-hero {
        grid-template-columns: 1fr;
    }

    .dynamo-model-hero-copy,
    .dynamo-recommendation-card {
        padding: 16px;
    }

    .dynamo-prereq-grid {
        grid-template-columns: 1fr;
    }

    .dynamo-benchmark-hero {
        padding: 16px;
    }

    .dynamo-technique-guide {
        padding: 0;
    }

    .dynamo-technique-guide-header {
        display: grid;
        gap: 6px;
    }

    .dynamo-technique-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        column-gap: 14px;
        row-gap: 0;
    }

    .dynamo-technique-card {
        grid-template-columns: 22px minmax(0, 1fr);
        gap: 7px;
        padding: 8px 0;
    }

    .dynamo-technique-card .dynamo-feature-symbol {
        width: 22px;
        height: 22px;
    }

    .dynamo-technique-card .dynamo-feature-icon {
        width: 20px;
        height: 20px;
    }

    .dynamo-technique-card strong {
        font-size: 12.5px;
    }

    .dynamo-technique-card p {
        display: none !important;
    }

    .dynamo-technique-card:nth-last-child(-n + 2) {
        border-bottom: 1px solid var(--border, var(--grayscale-a5));
    }

    .dynamo-technique-card:last-child {
        border-bottom: 0;
    }

    .dynamo-benchmark-card {
        min-height: 0;
        gap: 10px;
    }

    .dynamo-benchmark-card-head {
        flex-direction: column;
        min-height: 0;
    }

    .dynamo-benchmark-index-header {
        display: none;
    }

    .dynamo-benchmark-row {
        grid-template-columns: 1fr;
        align-items: start;
        gap: 10px;
        min-height: 0;
        padding: 14px;
    }

    .dynamo-feature-symbol-row,
    .dynamo-benchmark-model,
    .dynamo-benchmark-kind,
    .dynamo-benchmark-card h2,
    .dynamo-benchmark-card p,
    .dynamo-benchmark-features,
    .dynamo-benchmark-target,
    .dynamo-benchmark-targets,
    .dynamo-benchmark-open {
        grid-column: 1;
        grid-row: auto;
    }

    .dynamo-benchmark-model {
        flex-basis: auto;
        width: 100%;
        max-width: 100%;
    }

    .dynamo-benchmark-targets {
        grid-template-columns: 1fr;
    }

    .dynamo-benchmark-open {
        justify-self: start;
    }
}

/* ---------------------------------------------------------------------------
   Recipe target picker (variant selector on multi-target recipe pages).
   Pages opt in by rendering hidden radio inputs named "recipe-sku" /
   "recipe-usecase" with an adjacent label, and tagging variant-scoped blocks
   with data-sku / data-usecase (space-separated values allowed). Content is
   hidden via body:has(), so browsers without :has() degrade to showing all
   variants. Pages without a picker are unaffected.
--------------------------------------------------------------------------- */

.dynamo-target-picker {
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 14px;
    padding: 18px 20px;
    margin: 1.25rem 0 1.75rem;
    background: var(--nv-color-bg-alt);
}

.dark .dynamo-target-picker {
    background: var(--nv-dark-grey-2);
}

.dynamo-target-picker-title {
    margin: 0 0 12px;
    font-size: 0.95rem;
    font-weight: 700;
}

.dynamo-target-picker-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.dynamo-target-picker-row + .dynamo-target-picker-row {
    margin-top: 10px;
}

.dynamo-target-picker-dim {
    flex: 0 0 84px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--grayscale-a9, #777);
}

.dynamo-target-picker input[type="radio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
}

.dynamo-target-picker input[type="radio"] + label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 34px;
    padding: 7px 16px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded);
    background: var(--nv-color-bg-default);
    color: var(--pst-color-text-base);
    cursor: pointer;
    font-size: 13.5px;
    line-height: 1;
    user-select: none;
}

.dark .dynamo-target-picker input[type="radio"] + label {
    background: var(--nv-dark-grey-3);
}

.dynamo-target-picker input[type="radio"]:checked + label {
    border-color: var(--nv-color-green);
    box-shadow: 0 0 0 1px var(--nv-color-green);
    font-weight: 700;
}

.dynamo-target-picker input[type="radio"]:focus-visible + label {
    outline: 2px solid var(--nv-color-green);
    outline-offset: 2px;
}

.dynamo-target-picker-hint {
    font-size: 11px;
    font-weight: 600;
    color: var(--nv-color-green-2);
    background: color-mix(in srgb, var(--nv-color-green) 18%, transparent);
    border-radius: var(--rounded);
    padding: 2px 8px;
}

.dark .dynamo-target-picker-hint {
    color: var(--nv-color-green);
}

/* Selected-target summary strip inside the picker */
.dynamo-target-picker-summary {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
    /* Stacked label-over-value cells in a responsive grid: each fact reads as
       its own labeled cell with aligned columns, instead of an inline run-on
       where adjacent label/value pairs blur together. */
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px 28px;
}

.dynamo-target-picker-summary span {
    display: flex;
    flex-direction: column;
    gap: 3px;
    /* the value (text node after the label) — prominent, dark */
    font-size: 13.5px;
    line-height: 1.35;
    color: var(--pst-color-text-base);
}

.dynamo-target-picker-summary b {
    /* the label — small, uppercase, muted, sits above the value */
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--grayscale-a9, #777);
}

/* Variant visibility: hide blocks that do not match the checked sku/usecase */
body:has(input[name="recipe-sku"][value="b200"]:checked) [data-sku]:not([data-sku~="b200"]),
body:has(input[name="recipe-sku"][value="h200"]:checked) [data-sku]:not([data-sku~="h200"]),
body:has(input[name="recipe-sku"][value="h100"]:checked) [data-sku]:not([data-sku~="h100"]),
body:has(input[name="recipe-sku"][value="gb200"]:checked) [data-sku]:not([data-sku~="gb200"]),
body:has(input[name="recipe-usecase"][value="chat"]:checked) [data-usecase]:not([data-usecase~="chat"]),
body:has(input[name="recipe-usecase"][value="agentic"]:checked) [data-usecase]:not([data-usecase~="agentic"]) {
    display: none;
}

/* Highlight the matching row in an expected-performance or comparison table */
body:has(input[name="recipe-sku"]:checked) tr[data-sku][data-usecase] {
    opacity: 0.55;
}

body:has(input[name="recipe-sku"][value="b200"]:checked):has(input[name="recipe-usecase"][value="chat"]:checked) tr[data-sku~="b200"][data-usecase~="chat"],
body:has(input[name="recipe-sku"][value="b200"]:checked):has(input[name="recipe-usecase"][value="agentic"]:checked) tr[data-sku~="b200"][data-usecase~="agentic"],
body:has(input[name="recipe-sku"][value="h200"]:checked):has(input[name="recipe-usecase"][value="chat"]:checked) tr[data-sku~="h200"][data-usecase~="chat"],
body:has(input[name="recipe-sku"][value="h200"]:checked):has(input[name="recipe-usecase"][value="agentic"]:checked) tr[data-sku~="h200"][data-usecase~="agentic"] {
    opacity: 1;
    font-weight: 600;
}

/* Picker dimension extensions: hopper/blackwell SKUs and a generic
   recipe-variant axis for single-dimension pickers (topology, build, etc.) */
body:has(input[name="recipe-sku"][value="hopper"]:checked) [data-sku]:not([data-sku~="hopper"]),
body:has(input[name="recipe-sku"][value="blackwell"]:checked) [data-sku]:not([data-sku~="blackwell"]),
body:has(input[name="recipe-variant"][value="agg"]:checked) [data-variant]:not([data-variant~="agg"]),
body:has(input[name="recipe-variant"][value="disagg"]:checked) [data-variant]:not([data-variant~="disagg"]),
body:has(input[name="recipe-variant"][value="disagg-single-node"]:checked) [data-variant]:not([data-variant~="disagg-single-node"]),
body:has(input[name="recipe-variant"][value="disagg-multi-node"]:checked) [data-variant]:not([data-variant~="disagg-multi-node"]),
body:has(input[name="recipe-variant"][value="trtllm-agg"]:checked) [data-variant]:not([data-variant~="trtllm-agg"]),
body:has(input[name="recipe-variant"][value="trtllm-disagg"]:checked) [data-variant]:not([data-variant~="trtllm-disagg"]),
body:has(input[name="recipe-variant"][value="vllm-disagg"]:checked) [data-variant]:not([data-variant~="vllm-disagg"]),
body:has(input[name="recipe-variant"][value="standard"]:checked) [data-variant]:not([data-variant~="standard"]),
body:has(input[name="recipe-variant"][value="efa"]:checked) [data-variant]:not([data-variant~="efa"]),
body:has(input[name="recipe-variant"][value="kvbm"]:checked) [data-variant]:not([data-variant~="kvbm"]) {
    display: none;
}

/* Row highlighting for the recipe-variant axis */
body:has(input[name="recipe-variant"]:checked) tr[data-variant] {
    opacity: 0.55;
}

body:has(input[name="recipe-variant"][value="agg"]:checked) tr[data-variant~="agg"],
body:has(input[name="recipe-variant"][value="disagg"]:checked) tr[data-variant~="disagg"],
body:has(input[name="recipe-variant"][value="disagg-single-node"]:checked) tr[data-variant~="disagg-single-node"],
body:has(input[name="recipe-variant"][value="disagg-multi-node"]:checked) tr[data-variant~="disagg-multi-node"],
body:has(input[name="recipe-variant"][value="trtllm-agg"]:checked) tr[data-variant~="trtllm-agg"],
body:has(input[name="recipe-variant"][value="trtllm-disagg"]:checked) tr[data-variant~="trtllm-disagg"],
body:has(input[name="recipe-variant"][value="vllm-disagg"]:checked) tr[data-variant~="vllm-disagg"],
body:has(input[name="recipe-variant"][value="standard"]:checked) tr[data-variant~="standard"],
body:has(input[name="recipe-variant"][value="efa"]:checked) tr[data-variant~="efa"],
body:has(input[name="recipe-variant"][value="kvbm"]:checked) tr[data-variant~="kvbm"] {
    opacity: 1;
    font-weight: 600;
}

/* Combined sku x variant row highlighting (Hopper/Blackwell matrix pages) */
body:has(input[name="recipe-sku"]:checked):has(input[name="recipe-variant"]:checked) tr[data-sku][data-variant] {
    opacity: 0.55;
    font-weight: 400;
}

body:has(input[name="recipe-sku"][value="hopper"]:checked):has(input[name="recipe-variant"][value="agg"]:checked) tr[data-sku~="hopper"][data-variant~="agg"],
body:has(input[name="recipe-sku"][value="hopper"]:checked):has(input[name="recipe-variant"][value="disagg"]:checked) tr[data-sku~="hopper"][data-variant~="disagg"],
body:has(input[name="recipe-sku"][value="blackwell"]:checked):has(input[name="recipe-variant"][value="agg"]:checked) tr[data-sku~="blackwell"][data-variant~="agg"],
body:has(input[name="recipe-sku"][value="blackwell"]:checked):has(input[name="recipe-variant"][value="disagg"]:checked) tr[data-sku~="blackwell"][data-variant~="disagg"] {
    opacity: 1;
    font-weight: 600;
}

/* Static single-target summary panel (degenerate, no-picker form) */
.dynamo-target-picker.static .dynamo-target-picker-summary {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}

/* Filter empty-state without JS: the message is painted underneath the card
   grid (z-index 0); any visible card sits above it (z-index 1) and covers it,
   so it only becomes visible when every card is filtered out. */
.dynamo-model-grid {
    position: relative;
    min-height: 150px;
}

.dynamo-model-grid::before {
    content: "No recipes match the selected filters. Clear a filter or reset to see the full catalog.";
    position: absolute;
    top: 0;
    left: 0;
    max-width: 230px;
    padding: 12px 2px;
    font-size: 13.5px;
    line-height: 1.45;
    color: var(--grayscale-a9, #777);
    z-index: 0;
}

.dynamo-model-grid > [data-recipe-card] {
    position: relative;
    z-index: 1;
}
`;

export function RecipeStyles() {
  return <style dangerouslySetInnerHTML={{ __html: RECIPE_CSS }} />;
}
