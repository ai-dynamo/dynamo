/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ArtifactBrowser — filterable list of every published artifact for the
 * current release (containers, wheels, Helm charts, crates) driven entirely
 * by releases.data.ts.
 *
 * Filtering is CSS-only: five hidden radio inputs sit first inside the panel,
 * and :checked general-sibling selectors hide the rows (and their group
 * headers) whose data-cat does not match — same pattern as the recipe catalog
 * in recipes/README.mdx. Server component; shared vocabulary (panel, eyebrow,
 * badges, copy buttons) comes from ReferenceStyles — place <ReferenceStyles />
 * on the page alongside this component. Only the .dynref-ab-* layout classes
 * are defined here.
 */

import {
  ARTIFACTS,
  CURRENT_TAG,
  CURRENT_VERSION,
  CURRENT_WHEEL,
  type Artifact,
  type ArtifactCategory,
} from "./releases.data";

const AB_CSS = `
.dynref-ab-filter {
    /* display:none (not visually-hidden positioning): if the inputs can
       receive focus, clicking a filter pill scrolls the page back to the
       inputs. Label activation still toggles undisplayed inputs, and the
       :checked sibling selectors still match — same rationale as
       .dynamo-filter-input in RecipeStyles. */
    display: none;
}

.dynref-ab-note {
    margin: 0;
}

.dynref-ab-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 4px;
}

.dynref-ab-pill {
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

.dynref-ab-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

#ab-all:checked ~ .dynref-ab-rail label[for="ab-all"],
#ab-container:checked ~ .dynref-ab-rail label[for="ab-container"],
#ab-wheel:checked ~ .dynref-ab-rail label[for="ab-wheel"],
#ab-helm:checked ~ .dynref-ab-rail label[for="ab-helm"],
#ab-crate:checked ~ .dynref-ab-rail label[for="ab-crate"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

#ab-container:checked ~ .dynref-ab-list > [data-cat]:not([data-cat="container"]),
#ab-wheel:checked ~ .dynref-ab-list > [data-cat]:not([data-cat="wheel"]),
#ab-helm:checked ~ .dynref-ab-list > [data-cat]:not([data-cat="helm"]),
#ab-crate:checked ~ .dynref-ab-list > [data-cat]:not([data-cat="crate"]) {
    display: none;
}

/* Heading recount: the filter is CSS-only, so the heading pre-renders one
   span per filter and the same :checked sibling selectors toggle which one
   shows. Default (All checked, or nothing checked) shows the All span. */
.dynref-ab-hspan {
    display: none;
}

.dynref-ab-hspan--all {
    display: inline;
}

#ab-container:checked ~ .dynref-panel-header .dynref-ab-hspan--all,
#ab-wheel:checked ~ .dynref-panel-header .dynref-ab-hspan--all,
#ab-helm:checked ~ .dynref-panel-header .dynref-ab-hspan--all,
#ab-crate:checked ~ .dynref-panel-header .dynref-ab-hspan--all {
    display: none;
}

#ab-container:checked ~ .dynref-panel-header .dynref-ab-hspan--container,
#ab-wheel:checked ~ .dynref-panel-header .dynref-ab-hspan--wheel,
#ab-helm:checked ~ .dynref-panel-header .dynref-ab-hspan--helm,
#ab-crate:checked ~ .dynref-panel-header .dynref-ab-hspan--crate {
    display: inline;
}

.dynref-ab-group {
    margin: 0;
    padding: 16px 0 6px;
}

.dynref-ab-row {
    /* Fixed first column so names, descriptions, and tag groups align into
       clean columns across rows. 230px fits the longest name
       (tensorrtllm-runtime + glyph ≈ 171px at 13px mono). */
    display: grid;
    grid-template-columns: 230px minmax(0, 1fr) max-content;
    gap: 10px;
    align-items: center;
    padding: 11px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-ab-row:last-child {
    border-bottom: 0;
}

.dynref-ab-name {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    flex-wrap: wrap;
}

.dynref-ab-glyph {
    flex: 0 0 auto;
}

.dynref-ab-glyph--container {
    color: #76B900;
}

.dynref-ab-glyph--wheel {
    color: #0F766E;
}

.dark .dynref-ab-glyph--wheel {
    color: #5EEAD4;
}

.dynref-ab-glyph--helm {
    color: #1D4ED8;
}

.dark .dynref-ab-glyph--helm {
    color: #93C5FD;
}

.dynref-ab-glyph--crate {
    color: #6D28D9;
}

.dark .dynref-ab-glyph--crate {
    color: #C4B5FD;
}

.dynref-ab-link {
    color: var(--pst-color-text-base);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.dynref-ab-link:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-ab-desc {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.4;
}

.dynref-ab-tags {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
}

.dynref-ab-tags .dynref-copy {
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 11px;
}

/* Narrow widths: stack each row (name / description / tags in column flow).
   Last in the sheet so these override the base rules above. */
@media (max-width: 640px) {
    .dynref-ab-row {
        grid-template-columns: minmax(0, 1fr);
        gap: 6px;
    }

    .dynref-ab-tags {
        justify-content: flex-start;
    }
}
`;

/* Simple 15px stroke glyphs per artifact category: box (container),
   wheel (Python wheel), anchor (Helm), gear (crate). */
function Glyph({ category }: { category: ArtifactCategory }) {
  const shared = {
    className: `dynref-ab-glyph dynref-ab-glyph--${category}`,
    width: 15,
    height: 15,
    viewBox: "0 0 16 16",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.4,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  if (category === "container") {
    return (
      <svg {...shared}>
        <path d="M2 5l6-3 6 3v6l-6 3-6-3z" />
        <path d="M2 5l6 3 6-3M8 8v6" />
      </svg>
    );
  }
  if (category === "wheel") {
    return (
      <svg {...shared}>
        <circle cx="8" cy="8" r="6" />
        <circle cx="8" cy="8" r="1.6" />
        <path d="M8 6.4V2M9.4 8.8l3.4 2.6M6.6 8.8l-3.4 2.6" />
      </svg>
    );
  }
  if (category === "helm") {
    return (
      <svg {...shared}>
        <circle cx="8" cy="3.6" r="1.6" />
        <path d="M8 5.2V14M3 9.5c0 3 2.2 4.5 5 4.5s5-1.5 5-4.5M3 9.5h2.2M13 9.5h-2.2" />
      </svg>
    );
  }
  return (
    <svg {...shared}>
      <circle cx="8" cy="8" r="2.2" />
      <path d="M8 1.8v2.1M8 12.1v2.1M1.8 8h2.1M12.1 8h2.1M3.7 3.7l1.5 1.5M10.8 10.8l1.5 1.5M12.3 3.7l-1.5 1.5M5.2 10.8l-1.5 1.5" />
    </svg>
  );
}

/* Short accessible name for a copy button; the full clipboard payload goes
   in the title attribute. */
function copyAriaLabel(artifact: Artifact, tagLabel: string): string {
  if (artifact.category === "container") return `Copy ${artifact.name}:${tagLabel} image reference`;
  if (artifact.category === "wheel") return `Copy pip install command for ${artifact.name}`;
  if (artifact.category === "helm") return `Copy helm install command for ${artifact.name}`;
  return `Copy cargo add command for ${artifact.name}`;
}

function ArtifactRow({ artifact }: { artifact: Artifact }) {
  const badgeVariant = artifact.badge === "Deprecated" ? "red" : "amber";
  return (
    <div className="dynref-ab-row" data-cat={artifact.category}>
      <div className="dynref-ab-name">
        <Glyph category={artifact.category} />
        <a className="dynref-mono dynref-ab-link" href={artifact.href}>
          {artifact.name}
        </a>
        {artifact.badge && (
          <span className={`dynref-badge dynref-badge--${badgeVariant}`}>{artifact.badge}</span>
        )}
      </div>
      <p className="dynref-ab-desc">
        {artifact.description}
        {artifact.meta ? ` · ${artifact.meta}` : ""}
      </p>
      <div className="dynref-ab-tags">
        {artifact.tags.map((tag) => (
          <button
            key={tag.clipboard}
            className={`dynref-copy dynref-badge dynref-badge--${
              tag.variant === "experimental" ? "amber" : "blue"
            }`}
            type="button"
            data-dynref-copy={tag.clipboard}
            aria-label={copyAriaLabel(artifact, tag.label)}
            title={tag.clipboard}
          >
            {tag.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function GroupHeader({ label, cat }: { label: string; cat: ArtifactCategory }) {
  return (
    <p className="dynref-label dynref-ab-group" data-cat={cat}>
      {label}
    </p>
  );
}

export function ArtifactBrowser() {
  const counts: Record<ArtifactCategory, number> = { container: 0, wheel: 0, helm: 0, crate: 0 };
  for (const artifact of ARTIFACTS) counts[artifact.category] += 1;

  const runtimeContainers = ARTIFACTS.filter((a) => a.category === "container" && a.group === "runtime");
  const componentContainers = ARTIFACTS.filter((a) => a.category === "container" && a.group !== "runtime");
  const wheels = ARTIFACTS.filter((a) => a.category === "wheel");
  const helmCharts = ARTIFACTS.filter((a) => a.category === "helm");
  const crates = ARTIFACTS.filter((a) => a.category === "crate");

  return (
    <>
      <style>{AB_CSS}</style>
      <section className="dynref-panel">
        <input className="dynref-ab-filter" type="radio" id="ab-all" name="dynref-ab-filter" defaultChecked />
        <input className="dynref-ab-filter" type="radio" id="ab-container" name="dynref-ab-filter" />
        <input className="dynref-ab-filter" type="radio" id="ab-wheel" name="dynref-ab-filter" />
        <input className="dynref-ab-filter" type="radio" id="ab-helm" name="dynref-ab-filter" />
        <input className="dynref-ab-filter" type="radio" id="ab-crate" name="dynref-ab-filter" />

        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">Release artifacts</p>
            <h3 className="dynref-h">
              <span className="dynref-ab-hspan dynref-ab-hspan--all">
                {ARTIFACTS.length} artifacts for {CURRENT_VERSION}
              </span>
              <span className="dynref-ab-hspan dynref-ab-hspan--container">
                {counts.container} container images for {CURRENT_VERSION}
              </span>
              <span className="dynref-ab-hspan dynref-ab-hspan--wheel">
                {counts.wheel} Python wheels for {CURRENT_VERSION}
              </span>
              <span className="dynref-ab-hspan dynref-ab-hspan--helm">
                {counts.helm} Helm charts for {CURRENT_VERSION}
              </span>
              <span className="dynref-ab-hspan dynref-ab-hspan--crate">
                {counts.crate} Rust crates for {CURRENT_VERSION}
              </span>
            </h3>
          </div>
          <p className="dynref-muted dynref-ab-note">
            Wheels ship as <span className="dynref-mono">{CURRENT_WHEEL}</span> · containers stay{" "}
            <span className="dynref-mono">:{CURRENT_TAG}</span>
          </p>
        </div>

        <div className="dynref-ab-rail">
          <label className="dynref-ab-pill" htmlFor="ab-all">
            All · {ARTIFACTS.length}
          </label>
          <label className="dynref-ab-pill" htmlFor="ab-container">
            Containers · {counts.container}
          </label>
          <label className="dynref-ab-pill" htmlFor="ab-wheel">
            Wheels · {counts.wheel}
          </label>
          <label className="dynref-ab-pill" htmlFor="ab-helm">
            Helm · {counts.helm}
          </label>
          <label className="dynref-ab-pill" htmlFor="ab-crate">
            Crates · {counts.crate}
          </label>
        </div>

        <div className="dynref-ab-list">
          <GroupHeader label="Runtime containers" cat="container" />
          {runtimeContainers.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
          <GroupHeader label="Component containers" cat="container" />
          {componentContainers.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
          <GroupHeader label="Python wheels" cat="wheel" />
          {wheels.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
          <GroupHeader label="Helm charts" cat="helm" />
          {helmCharts.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
          <GroupHeader label="Rust crates" cat="crate" />
          {crates.map((a) => (
            <ArtifactRow key={a.name} artifact={a} />
          ))}
        </div>

        <p className="dynref-grid-note">
          Every tag and command is click-to-copy. For TensorRT-LLM use the NGC container — not the{" "}
          <span className="dynref-mono">ai-dynamo[trtllm]</span> wheel.
        </p>
      </section>
    </>
  );
}
