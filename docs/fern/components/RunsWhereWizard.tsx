/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RunsWhereWizard — "what runs where" picker for the Compatibility page.
 * Reader states two facts (backend + CUDA driver generation) and gets the
 * list of stable/patch Dynamo releases that ship a matching container, with
 * the CUDA toolkit and driver floor per release — and the click-to-copy pull
 * command for the current release only (older tags are not restated here).
 *
 * Everything is derived from releases.data.ts: the CUDA options come from
 * CUDA_HISTORY's distinct toolkit majors, a release qualifies for a
 * (backend, major) pair when CUDA_HISTORY has a row for it, and pull commands
 * come from ARTIFACTS (which tracks the current release only).
 *
 * Filtering is CSS-only, same mechanism as ArtifactBrowser: two hidden radio
 * groups sit first inside the panel; each group's :checked sibling selector
 * hides the rows whose data-backend / data-cuda does not match, so only rows
 * matching BOTH selections stay visible. Combinations with no qualifying
 * release pre-render a muted empty-state row driven by the same selectors.
 *
 * Server component; shared vocabulary (panel, eyebrow, chips, badges, copy
 * buttons) comes from ReferenceStyles — place <ReferenceStyles /> on the page.
 * Only the .dynref-ww-* layout classes are defined here.
 */

import { ARTIFACTS, CUDA_HISTORY, CURRENT_TAG, RELEASES, type CudaRow } from "./releases.data";

interface BackendOption {
  id: string;
  label: string;
  backend: CudaRow["backend"];
  runtime: string;
}

const BACKENDS: BackendOption[] = [
  { id: "sglang", label: "SGLang", backend: "SGLang", runtime: "sglang-runtime" },
  { id: "trtllm", label: "TensorRT-LLM", backend: "TensorRT-LLM", runtime: "tensorrtllm-runtime" },
  { id: "vllm", label: "vLLM", backend: "vLLM", runtime: "vllm-runtime" },
];

interface CudaOption {
  major: string;
  /** Driver floor for the major's current toolkits, e.g. "580.xx+". */
  floor: string;
}

/** Distinct CUDA toolkit majors in the history, newest first, each with the
 *  highest driver floor seen for that major (the floor a reader on that
 *  driver generation should expect for recent releases). */
function cudaOptions(): CudaOption[] {
  const floors = new Map<string, string>();
  for (const row of CUDA_HISTORY) {
    const major = row.toolkit.split(".")[0];
    const prev = floors.get(major);
    if (prev === undefined || row.minDriver > prev) floors.set(major, row.minDriver);
  }
  return [...floors.entries()]
    .map(([major, floor]) => ({ major, floor }))
    .sort((a, b) => Number(b.major) - Number(a.major));
}

interface WizardRow {
  backendId: string;
  major: string;
  version: string;
  href: string;
  toolkits: string[];
  minDriver: string;
  experimental: boolean;
  isCurrent: boolean;
  /** Full image reference to copy — current release only. */
  pull?: string;
}

function pullCommandFor(runtime: string): string | undefined {
  const artifact = ARTIFACTS.find((a) => a.category === "container" && a.name === runtime);
  const tag = artifact?.tags.find((t) => t.label === CURRENT_TAG);
  return tag?.clipboard;
}

/** Stable + patch releases (newest first, RELEASES order) that CUDA_HISTORY
 *  lists for the given backend on the given toolkit major. */
function rowsFor(backend: BackendOption, major: string): WizardRow[] {
  const rows: WizardRow[] = [];
  for (const release of RELEASES) {
    if (release.kind !== "stable" && release.kind !== "patch") continue;
    const bare = release.version.replace(/^v/, "");
    const matches = CUDA_HISTORY.filter(
      (h) => h.version === bare && h.backend === backend.backend && h.toolkit.split(".")[0] === major,
    );
    if (matches.length === 0) continue;
    const isCurrent = bare === CURRENT_TAG;
    rows.push({
      backendId: backend.id,
      major,
      version: release.version,
      href: release.notesHref ?? release.github ?? "https://github.com/ai-dynamo/dynamo/releases",
      toolkits: [...new Set(matches.map((h) => h.toolkit))],
      minDriver: matches[0].minDriver,
      experimental: matches.every((h) => h.note === "Experimental"),
      isCurrent,
      pull: isCurrent ? pullCommandFor(backend.runtime) : undefined,
    });
  }
  return rows;
}

const WW_CSS = `
.dynref-ww-filter {
    /* display:none, not visually-hidden positioning — focusable inputs would
       scroll the page back to them on pill click; label activation still
       toggles undisplayed inputs (same rationale as .dynref-ab-filter). */
    display: none;
}

.dynref-ww-rails {
    display: flex;
    flex-wrap: wrap;
    gap: 16px 32px;
    margin: 0 0 6px;
}

.dynref-ww-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.dynref-ww-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.dynref-ww-pill {
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

.dynref-ww-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

.dynref-ww-row {
    display: grid;
    grid-template-columns: 96px max-content minmax(0, 1fr) max-content;
    gap: 10px;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-ww-row:last-child {
    border-bottom: 0;
}

.dynref-ww-version {
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}

.dynref-ww-link {
    color: var(--pst-color-text-base);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
}

.dynref-ww-link:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-ww-driver {
    color: var(--pst-color-text-base);
    font-size: 12.5px;
}

.dynref-ww-notecol {
    margin-left: 6px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

.dynref-ww-pullslot {
    display: flex;
    justify-content: flex-end;
}

.dynref-ww-empty {
    padding: 12px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    color: var(--pst-color-text-muted);
    font-size: 13px;
}

.dynref-ww-empty:last-child {
    border-bottom: 0;
}

@media (max-width: 640px) {
    .dynref-ww-row {
        grid-template-columns: minmax(0, 1fr);
        gap: 6px;
    }

    .dynref-ww-pullslot {
        justify-content: flex-start;
    }
}
`;

/** :checked rules are generated from the option lists so the selectors always
 *  match the rendered inputs — active-pill highlight per input, plus the two
 *  independent hide rules whose intersection leaves only rows matching both
 *  the checked backend and the checked CUDA major. */
function filterCss(cuda: CudaOption[]): string {
  const inputs = [
    ...BACKENDS.map((b) => ({ id: `ww-b-${b.id}`, attr: "data-backend", value: b.id })),
    ...cuda.map((c) => ({ id: `ww-c-${c.major}`, attr: "data-cuda", value: c.major })),
  ];
  const pillRules = inputs
    .map((i) => `#${i.id}:checked ~ .dynref-ww-rails label[for="${i.id}"]`)
    .join(",\n");
  const hideRules = inputs
    .map(
      (i) =>
        `#${i.id}:checked ~ .dynref-ww-list [${i.attr}]:not([${i.attr}="${i.value}"]) { display: none; }`,
    )
    .join("\n");
  return `
${pillRules} {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}
${hideRules}
`;
}

function WizardDataRow({ row, backendLabel }: { row: WizardRow; backendLabel: string }) {
  return (
    <div className="dynref-ww-row" data-backend={row.backendId} data-cuda={row.major}>
      <span className="dynref-ww-version">
        <a className="dynref-mono dynref-ww-link" href={row.href}>
          {row.version}
        </a>
        {row.isCurrent && <span className="dynref-badge dynref-badge--green">Current</span>}
      </span>
      <span>
        {row.toolkits.map((toolkit) => (
          <span
            key={toolkit}
            className={`dynref-chip dynref-chip--cuda${row.experimental ? " dynref-chip--exp" : ""}`}
          >
            CUDA {toolkit}
          </span>
        ))}
        {row.experimental && <span className="dynref-ww-notecol">Experimental image</span>}
      </span>
      <span className="dynref-ww-driver">
        driver <span className="dynref-mono">{row.minDriver}</span>
      </span>
      <span className="dynref-ww-pullslot">
        {row.pull && (
          <button
            className="dynref-copy dynref-badge dynref-badge--blue"
            type="button"
            data-dynref-copy={row.pull}
            aria-label={`Copy ${backendLabel} runtime image reference for ${row.version}`}
            title={row.pull}
          >
            {row.pull.replace(/^.*\//, "")}
          </button>
        )}
      </span>
    </div>
  );
}

export function RunsWhereWizard() {
  const cuda = cudaOptions();
  const defaultBackend = BACKENDS[0];
  const defaultCuda = cuda[0];

  const combos = BACKENDS.flatMap((backend) =>
    cuda.map((option) => ({ backend, option, rows: rowsFor(backend, option.major) })),
  );

  return (
    <>
      <style>{WW_CSS + filterCss(cuda)}</style>
      <section className="dynref-panel">
        {BACKENDS.map((backend) => (
          <input
            key={backend.id}
            className="dynref-ww-filter"
            type="radio"
            id={`ww-b-${backend.id}`}
            name="dynref-ww-backend"
            defaultChecked={backend.id === defaultBackend.id}
          />
        ))}
        {cuda.map((option) => (
          <input
            key={option.major}
            className="dynref-ww-filter"
            type="radio"
            id={`ww-c-${option.major}`}
            name="dynref-ww-cuda"
            defaultChecked={option.major === defaultCuda.major}
          />
        ))}

        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">What runs where</p>
            <h3 className="dynref-h">Releases that match your backend and driver</h3>
          </div>
        </div>

        <div className="dynref-ww-rails">
          <div className="dynref-ww-group">
            <span className="dynref-label">Backend</span>
            <div className="dynref-ww-rail">
              {BACKENDS.map((backend) => (
                <label key={backend.id} className="dynref-ww-pill" htmlFor={`ww-b-${backend.id}`}>
                  {backend.label}
                </label>
              ))}
            </div>
          </div>
          <div className="dynref-ww-group">
            <span className="dynref-label">CUDA driver situation</span>
            <div className="dynref-ww-rail">
              {cuda.map((option) => (
                <label key={option.major} className="dynref-ww-pill" htmlFor={`ww-c-${option.major}`}>
                  CUDA {option.major} driver ({option.floor})
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="dynref-ww-list">
          {combos.map(({ backend, option, rows }) =>
            rows.length > 0 ? (
              rows.map((row) => (
                <WizardDataRow
                  key={`${backend.id}-${option.major}-${row.version}`}
                  row={row}
                  backendLabel={backend.label}
                />
              ))
            ) : (
              <div
                key={`${backend.id}-${option.major}-empty`}
                className="dynref-ww-empty"
                data-backend={backend.id}
                data-cuda={option.major}
              >
                No release ships {backend.label} for a CUDA {option.major} driver.
              </div>
            ),
          )}
        </div>

        <p className="dynref-grid-note">
          Driver floors from the CUDA & driver history; pull commands shown for the current release
          only.
        </p>
      </section>
    </>
  );
}
