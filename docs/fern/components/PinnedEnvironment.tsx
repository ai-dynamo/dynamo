/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PinnedEnvironment — one copy-paste block that pins every install path
 * (backend runtime container, frontend + operator images, Helm chart, and
 * wheel) to the current release. Every command string is assembled from the
 * ARTIFACTS clipboard payloads and CURRENT_* consts in releases.data.ts —
 * no registry or version literals live here.
 *
 * The backend switch is CSS-only: three hidden radios sit first inside the
 * panel and :checked general-sibling selectors toggle which pre-rendered
 * script block AND which "Copy all" button shows — one of each per backend,
 * so the copy payload always matches the visible script exactly (same
 * mechanism as ArtifactBrowser's filter rail). TensorRT-LLM ships via the
 * NGC container, so its variant omits the wheel line and carries a comment
 * instead.
 *
 * Server component; shared vocabulary (panel, eyebrow, copy buttons) comes
 * from ReferenceStyles — place <ReferenceStyles /> on the page alongside
 * this component. Only the .dynref-pe-* layout classes are defined here.
 */

import { ARTIFACTS, CURRENT_VERSION, CURRENT_WHEEL, CURRENT_TAG } from "./releases.data";

const PE_CSS = `
.dynref-pe-input {
    /* display:none, not visually-hidden positioning — focusable inputs would
       scroll the page on pill click; label activation still toggles them.
       Same rationale as .dynref-ab-filter in ArtifactBrowser. */
    display: none;
}

.dynref-pe-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 12px;
}

.dynref-pe-pill {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 6px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded, 6px);
    background: transparent;
    color: var(--pst-color-text-base);
    font-size: 12.5px;
    line-height: 1;
    cursor: pointer;
}

.dynref-pe-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

#pe-sglang:checked ~ .dynref-pe-rail label[for="pe-sglang"],
#pe-trtllm:checked ~ .dynref-pe-rail label[for="pe-trtllm"],
#pe-vllm:checked ~ .dynref-pe-rail label[for="pe-vllm"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

/* Pre-rendered script blocks and Copy-all buttons: hidden by default, the
   checked backend's pair shows. Payload and visible text come from the same
   string, so they always match. */
.dynref-pe-script,
.dynref-pe-copyall {
    display: none;
}

#pe-sglang:checked ~ .dynref-pe-script[data-backend="sglang"],
#pe-trtllm:checked ~ .dynref-pe-script[data-backend="trtllm"],
#pe-vllm:checked ~ .dynref-pe-script[data-backend="vllm"] {
    display: block;
}

#pe-sglang:checked ~ .dynref-panel-header .dynref-pe-copyall[data-backend="sglang"],
#pe-trtllm:checked ~ .dynref-panel-header .dynref-pe-copyall[data-backend="trtllm"],
#pe-vllm:checked ~ .dynref-panel-header .dynref-pe-copyall[data-backend="vllm"] {
    display: inline-flex;
}

/* Prominent green-tinted Copy all — composes with .dynref-copy for the
   click-to-copy binder, glyph, and copied-state feedback. */
.dynref-pe-copyall {
    align-items: center;
    padding: 8px 14px;
    border: 1px solid var(--dynref-green-border);
    border-radius: 8px;
    background: var(--dynref-green-bg);
    color: var(--dynref-green-fg);
    font-size: 12.5px;
    font-weight: 700;
}

.dynref-pe-copyall:hover {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
}

.dynref-pe-script {
    margin: 0;
    padding: 14px 16px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: #fcfcfc;
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
    line-height: 1.7;
    white-space: pre;
    overflow-x: auto;
}

.dark .dynref-pe-script {
    background: #0f0f0f;
    border-color: #2b2b2b;
}
`;

interface Backend {
  id: "sglang" | "trtllm" | "vllm";
  label: string;
  runtime: string;
  /** ai-dynamo wheel extra; null = ships via container only (comment line instead). */
  extra: string | null;
}

/* Rail order per the reference pages' backend convention; vLLM is the
   default selection. */
const BACKENDS: Backend[] = [
  { id: "sglang", label: "SGLang", runtime: "sglang-runtime", extra: "sglang" },
  { id: "trtllm", label: "TensorRT-LLM", runtime: "tensorrtllm-runtime", extra: null },
  { id: "vllm", label: "vLLM", runtime: "vllm-runtime", extra: "vllm" },
];

/** docker pull line for a container artifact's CURRENT_TAG image reference. */
function containerPull(name: string): string | null {
  const artifact = ARTIFACTS.find((a) => a.category === "container" && a.name === name);
  const tag = artifact?.tags.find((t) => t.label === CURRENT_TAG);
  return tag ? `docker pull ${tag.clipboard}` : null;
}

/** Full multi-line pinned-install script for one backend. */
function buildScript(backend: Backend): string {
  const helm = ARTIFACTS.find((a) => a.category === "helm" && a.name === "dynamo-platform");
  const lines: (string | null)[] = [
    containerPull(backend.runtime),
    containerPull("dynamo-frontend"),
    containerPull("kubernetes-operator"),
    helm?.tags[0]?.clipboard ?? null,
    backend.extra
      ? `uv pip install "ai-dynamo[${backend.extra}]==${CURRENT_WHEEL}"`
      : "# TensorRT-LLM ships via the NGC container",
  ];
  return lines.filter((line): line is string => line !== null).join("\n");
}

export function PinnedEnvironment() {
  const scripts = BACKENDS.map((backend) => ({ backend, script: buildScript(backend) }));

  return (
    <>
      <style>{PE_CSS}</style>
      <section className="dynref-panel">
        <input className="dynref-pe-input" type="radio" id="pe-sglang" name="dynref-pe-backend" />
        <input className="dynref-pe-input" type="radio" id="pe-trtllm" name="dynref-pe-backend" />
        <input className="dynref-pe-input" type="radio" id="pe-vllm" name="dynref-pe-backend" defaultChecked />

        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">Pinned environment</p>
            <h3 className="dynref-h">Everything pinned to {CURRENT_VERSION}</h3>
          </div>
          {scripts.map(({ backend, script }) => (
            <button
              key={backend.id}
              className="dynref-copy dynref-pe-copyall"
              type="button"
              data-backend={backend.id}
              data-dynref-copy={script}
              aria-label={`Copy the full ${backend.label} pinned-install script`}
              title={script}
            >
              Copy all
            </button>
          ))}
        </div>

        <div className="dynref-pe-rail">
          {BACKENDS.map((backend) => (
            <label key={backend.id} className="dynref-pe-pill" htmlFor={`pe-${backend.id}`}>
              {backend.label}
            </label>
          ))}
        </div>

        {scripts.map(({ backend, script }) => (
          <pre key={backend.id} className="dynref-pe-script" data-backend={backend.id}>
            {script}
          </pre>
        ))}

        <p className="dynref-grid-note">Assembled from the current release&apos;s artifact inventory.</p>
      </section>
    </>
  );
}
