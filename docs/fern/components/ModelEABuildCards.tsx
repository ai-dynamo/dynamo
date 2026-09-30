/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelEABuildCards — card grid of model early-access builds from
 * releases.data.ts, grouped by release line (or filtered to one line via the
 * `line` prop). Each card shows the GA-path badge, a click-to-copy docker
 * pull for the primary runtime, runtime/ship metadata, the build status
 * line, an Images/Wheels/Helm/Crates coverage-dots row, and the recipe link.
 *
 * Also exports EaCoverageDots — a standalone inline coverage-dots row for a
 * platform-preview version (reads PLATFORM_PREVIEW_COVERAGE), for use next
 * to platform-preview mentions on the release-artifacts page.
 *
 * Server component; shared vocabulary (badges, copy buttons, labels, mono)
 * comes from ReferenceStyles — place <ReferenceStyles /> on the page
 * alongside this component. Only the .dynref-ea-* layout classes are
 * defined here.
 */

import {
  MODEL_EA_BUILDS,
  PLATFORM_PREVIEW_COVERAGE,
  type Coverage,
  type GaPath,
  type ModelEaBuild,
} from "./releases.data";

/* Coverage-dots rules are shared by ModelEABuildCards and the standalone
   EaCoverageDots, so each component ships them in its own <style> block
   (duplicate <style> tags on one page are harmless). */
const EA_DOTS_CSS = `
.dynref-ea-dots {
    /* One line, always: nowrap + tight gaps sized so the four
       Images/Wheels/Helm/Crates pairs never orphan-wrap. */
    display: flex;
    flex-wrap: nowrap;
    gap: 10px;
}

.dynref-ea-dots-inline {
    display: inline-flex;
    vertical-align: middle;
}

.dynref-ea-dot-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--pst-color-text-base);
    font-size: 11px;
    line-height: 1.2;
    white-space: nowrap;
}

.dynref-ea-dot-item--off {
    color: var(--pst-color-text-muted);
    opacity: 0.55;
}

.dynref-ea-dot {
    box-sizing: border-box; /* 7px means 7 rendered px, border included */
    flex: 0 0 auto;
    width: 7px;
    height: 7px;
    border: 1.5px solid var(--pst-color-text-muted);
    border-radius: 50%;
    background: transparent;
}

.dynref-ea-dot--on {
    border-color: #76B900;
    background: #76B900;
}
`;

const EA_CSS = `${EA_DOTS_CSS}
.dynref-ea-grouphead {
    margin: 20px 0 0;
}

.dynref-ea-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 12px;
    margin: 12px 0 20px;
}

.dynref-ea-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 16px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
    /* Inline-size container so the coverage-dots row can tighten on narrow
       cards (see @container rule below). */
    container-type: inline-size;
}

/* Narrow cards (two-up grids near the 210px track minimum): the four dot
   pairs total ~204px at base sizes, so tighten gaps and label size to keep
   them on one line down to the 210px card (178px inner) worst case. */
@container (max-width: 250px) {
    .dynref-ea-dots {
        gap: 5px;
    }

    .dynref-ea-dot-item {
        gap: 4px;
        font-size: 10px;
    }
}

/* Final tier for the 210px track minimum (178px inner): the NVIDIA face is
   wide, so step down once more to keep all four pairs on one line. */
@container (max-width: 227px) {
    .dynref-ea-dots {
        gap: 4px;
    }

    .dynref-ea-dot-item {
        gap: 3px;
        font-size: 9px;
    }
}

.dark .dynref-ea-card {
    background: #161616;
    border-color: #2b2b2b;
}

.dynref-ea-head {
    /* Identical anatomy on every card: model name on its own line, the
       GA-path badge always on a dedicated line below (fit-content via
       align-items: flex-start) — no conditional inline/wrapped layout. */
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
}

.dynref-ea-model {
    color: var(--pst-color-text-base);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.3;
}

.dynref-ea-subtitle {
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.3;
}

.dynref-ea-tagrow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}

.dynref-ea-meta,
.dynref-ea-status {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.45;
    overflow-wrap: anywhere;
}

.dynref-ea-link {
    margin-top: auto;
    padding-top: 2px;
    color: var(--nv-color-green-2, #538300);
    font-size: 12.5px;
    font-weight: 600;
    text-decoration: none;
}

.dark .dynref-ea-link {
    color: var(--nv-color-green, #76B900);
}

.dynref-ea-link:hover {
    text-decoration: underline;
}
`;

const GA_BADGE_VARIANT: Record<GaPath, string> = {
  promoted: "green",
  "recipe-in-ga": "green",
  "dev-only": "amber",
  superseded: "gray",
};

/* Display names for runtime images, used in the disambiguating subtitle. */
const RUNTIME_LABELS: Record<string, string> = {
  "vllm-runtime": "vLLM",
  "sglang-runtime": "SGLang",
  "tensorrtllm-runtime": "TensorRT-LLM",
};

/* Models that appear on more than one build (e.g. the three v1.2.0
   "DeepSeek-V4 preview" cards) get a subtitle so the cards stay
   distinguishable. Computed from data — nothing hardcoded. */
const DUPLICATE_MODELS = new Set<string>();
{
  const seen = new Set<string>();
  for (const build of MODEL_EA_BUILDS) {
    if (seen.has(build.model)) DUPLICATE_MODELS.add(build.model);
    seen.add(build.model);
  }
}

/* Derive "dev.N · <runtimes> · <hardware>" strictly from data: the dev
   suffix from the tag, runtime display names from the runtimes list, and a
   Blackwell/B200 qualifier only when statusLine already says so. */
function buildSubtitle(build: ModelEaBuild): string | null {
  const dev = /dev\.\d+$/.exec(build.tag);
  if (!dev) return null;
  const parts = [dev[0], build.runtimes.map((r) => RUNTIME_LABELS[r] ?? r).join(" + ")];
  if (build.statusLine.includes("Blackwell")) {
    parts.push("Blackwell");
  } else if (build.statusLine.includes("B200")) {
    parts.push("B200");
  }
  return parts.join(" · ");
}

function CoverageDots({ coverage }: { coverage: Coverage }) {
  const cells: [string, boolean][] = [
    ["Images", coverage.images],
    ["Wheels", coverage.wheels],
    ["Helm", coverage.helm],
    ["Crates", coverage.crates],
  ];
  return (
    <span className="dynref-ea-dots">
      {cells.map(([label, shipped]) => (
        <span
          key={label}
          className={shipped ? "dynref-ea-dot-item" : "dynref-ea-dot-item dynref-ea-dot-item--off"}
        >
          <span className={shipped ? "dynref-ea-dot dynref-ea-dot--on" : "dynref-ea-dot"} />
          {label}
        </span>
      ))}
    </span>
  );
}

function BuildCard({ build }: { build: ModelEaBuild }) {
  const primaryRuntime = build.runtimes[0];
  const pullCommand = `docker pull nvcr.io/nvidia/ai-dynamo/${primaryRuntime}:${build.tag}`;
  const subtitle = DUPLICATE_MODELS.has(build.model) ? buildSubtitle(build) : null;
  return (
    <div className="dynref-ea-card">
      <div className="dynref-ea-head">
        <span className="dynref-ea-model">{build.model}</span>
        {subtitle && <span className="dynref-ea-subtitle">{subtitle}</span>}
        <span
          className={`dynref-badge dynref-badge--${GA_BADGE_VARIANT[build.gaPath]}${build.gaPath === "recipe-in-ga" ? " dynref-badge--outline" : ""}`}
        >
          {build.gaLabel}
        </span>
      </div>
      <div className="dynref-ea-tagrow">
        <button
          className="dynref-copy dynref-badge dynref-badge--blue dynref-mono"
          type="button"
          data-dynref-copy={pullCommand}
          aria-label={`Copy docker pull command for ${build.tag}`}
          title={pullCommand}
        >
          {build.tag}
        </button>
      </div>
      <p className="dynref-ea-meta">
        {build.runtimes.join(" + ")} · {build.shipped}
      </p>
      <p className="dynref-ea-status">{build.statusLine}</p>
      <CoverageDots coverage={build.coverage} />
      {build.recipeLabel && build.recipeHref && (
        <a className="dynref-ea-link" href={build.recipeHref}>
          {build.recipeLabel}
        </a>
      )}
    </div>
  );
}

export function ModelEABuildCards({ line }: { line?: string }) {
  if (line) {
    const builds = MODEL_EA_BUILDS.filter((b) => b.releaseLine === line);
    return (
      <>
        <style>{EA_CSS}</style>
        <div className="dynref-ea-grid">
          {builds.map((build) => (
            <BuildCard key={build.tag} build={build} />
          ))}
        </div>
      </>
    );
  }

  // Group by release line, preserving first-seen data order.
  const groups: { line: string; builds: ModelEaBuild[] }[] = [];
  for (const build of MODEL_EA_BUILDS) {
    const group = groups.find((g) => g.line === build.releaseLine);
    if (group) {
      group.builds.push(build);
    } else {
      groups.push({ line: build.releaseLine, builds: [build] });
    }
  }

  return (
    <>
      <style>{EA_CSS}</style>
      {groups.map((group) => (
        <div key={group.line}>
          <p className="dynref-label dynref-ea-grouphead">{group.line} release</p>
          <div className="dynref-ea-grid">
            {group.builds.map((build) => (
              <BuildCard key={build.tag} build={build} />
            ))}
          </div>
        </div>
      ))}
    </>
  );
}

/* Standalone inline coverage-dots row for a platform-preview version.
   Returns null when the version is not in PLATFORM_PREVIEW_COVERAGE. */
export function EaCoverageDots({ version }: { version: string }) {
  const coverage = PLATFORM_PREVIEW_COVERAGE[version];
  if (!coverage) return null;
  return (
    <span className="dynref-ea-dots-inline">
      <style>{EA_DOTS_CSS}</style>
      <CoverageDots coverage={coverage} />
    </span>
  );
}
