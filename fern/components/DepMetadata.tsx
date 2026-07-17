/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DepMetadata — a styled header card that separates a DEP's metadata (front
 * matter) from the proposal body: DEP number + title, a color-coded status
 * pill (Draft / Under Review / Accepted / ...), a labeled field grid, and an
 * action row of links (View PR, Tracking Issue).
 *
 * Server component (no "use client"), mirroring RecipeStyles.tsx. Registered
 * via docs.yml `experimental.mdx-components: ./components`;
 * import per page: import { DepMetadata } from "@/components/DepMetadata";
 * The CSS ships as a page-level <style> (not docs.yml `css:`) so it survives the
 * shared NVIDIA global theme at publish, per the RecipeStyles constraint.
 */

// Type-only import (erased at build): lets the lifecycle stepper set the
// `--dep-accent` CSS custom property inline without pulling React into runtime
// scope, matching the other components' zero-runtime-import convention.
import type { CSSProperties } from "react";

interface DepMetadataProps {
  /**
   * DEP number, e.g. "0000". Optional; when omitted, the card leads with the
   * eyebrow + status pill (the page-title H1 from the front-matter already
   * conveys the DEP number).
   */
  dep?: string | number;
  /**
   * Proposal title. Rarely needed: Fern renders the front-matter `title` as
   * the page H1 above the card. Only pass this if you deliberately want the
   * title repeated inside the card as an H2 (usually you don't).
   */
  title?: string;
  /** Lifecycle status: Draft | Under Review | Accepted | Rejected | Deferred | Implemented | Replaced. */
  status: string;
  category?: string;
  /**
   * Owning SIG (Special Interest Group). Mirrors the Kubernetes KEP
   * `owning-sig` metadata. Per DEP-0001, a SIG owns a DEP, not a code repo.
   */
  owningSig?: string;
  /** Other SIGs involved or impacted. Mirrors Kubernetes KEP `participating-sigs`. */
  participatingSigs?: string;
  authors?: string;
  sponsor?: string;
  requiredReviewers?: string;
  reviewDate?: string;
  replaces?: string;
  replacedBy?: string;
  /** PR number; renders a "View PR #N" action using owner/repo. */
  pr?: number;
  owner?: string;
  repo?: string;
  /** URL to the tracking issue. */
  trackingIssue?: string;
}

/**
 * Map a free-text lifecycle status to a pill variant (color).
 *
 * Kept as plain JavaScript syntax (no TS annotations in the signature or body)
 * so the companion test file at fern/components/test_dep_metadata.mjs can
 * regex-extract and eval it under node — the same extraction pattern used for
 * parseLinkedItems below. The regex branches MUST stay byte-for-byte identical
 * to variant() in fern/js/dep-status-pills.js so the on-page card pill and the
 * sidebar pill always agree for the same DEP.
 *
 * Buckets: accepted (green) covers Accepted + Implemented + legacy synonyms;
 * proposed (blue) covers Under Review + legacy Proposed; rejected (red);
 * muted (gray) covers Deferred + Replaced; draft (amber) is the fallback.
 *
 * @param {string | null | undefined} status
 * @returns {string}
 */
function statusVariant(status) {
  const s = (status || "").toLowerCase();
  if (/accept|approv|implement|final|active|ratif/.test(s)) return "accepted";
  if (/propos|review/.test(s)) return "proposed";
  if (/reject|withdraw/.test(s)) return "rejected";
  if (/replac|supersed|deferr|defer/.test(s)) return "muted";
  return "draft";
}

/** Forward lifecycle stages shown in the stepper, in order. Off-track terminal
 * states (Rejected / Deferred / Replaced) are handled separately — see
 * DepLifecycle below. */
const LIFECYCLE_STAGES = ["Draft", "Under Review", "Accepted", "Implemented"];

/** Accent color per pill variant. Drives both the card's status-tinted accent
 * (border + corner wash) and the stepper fill, so a DEP reads in one color.
 * Values match ACCENT in fern/js/dep-index.js. */
const STATUS_ACCENT = {
  draft: "#e0a800",
  proposed: "#5b8def",
  accepted: "#76b900",
  rejected: "#dc4848",
  muted: "#9a9a9a",
};

/**
 * Index into LIFECYCLE_STAGES for a forward-moving DEP. Rejected / Deferred /
 * Replaced are terminal off-track states and never reach here (DepLifecycle
 * branches on statusVariant first) — this only ranks the happy path.
 *
 * Kept as plain JavaScript syntax (no TS annotations) so the companion test at
 * fern/components/test_dep_metadata.mjs can regex-extract and eval it, matching
 * the parseLinkedItems / statusVariant extraction pattern.
 *
 * @param {string | null | undefined} status
 * @returns {number}
 */
function lifecycleStage(status) {
  const s = (status || "").toLowerCase();
  if (/implement|final/.test(s)) return 3;
  if (/accept|approv|active|ratif/.test(s)) return 2;
  if (/propos|review/.test(s)) return 1;
  return 0;
}

/**
 * Parse an Authors / Required Reviewers field value into a list of
 * `{ label, href }` entries. Each comma-separated entry is treated as either
 * a markdown link `[label](https://...)` — the canonical shape emitted by
 * `fern/scripts/sync_deps.py` when it pulls DEP bodies from
 * ai-dynamo/enhancements — or as a plain-text fallback (href = null).
 *
 * Kept as plain JavaScript syntax (no TS annotations inside the body) so the
 * companion test file at `fern/components/test_dep_metadata.mjs` can regex-
 * extract and eval it under node, matching the extraction pattern used by
 * `fern/js/test_dep_pr_comments.mjs`.
 *
 * Only `http` / `https` URLs are accepted — anything else (e.g. a
 * `javascript:` URI in a malformed manifest entry) is dropped to a plain-text
 * entry, so the component never renders an unsafe anchor href.
 *
 * @param {string | null | undefined} value
 * @returns {Array<{ label: string, href: string | null }>}
 */
function parseLinkedItems(value) {
  if (!value) return [];
  const raw = String(value);
  const pieces = raw.split(",");
  const items = [];
  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i].trim();
    if (piece.length === 0) continue;
    const m = piece.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (m) {
      items.push({ label: m[1], href: m[2] });
    } else {
      items.push({ label: piece, href: null });
    }
  }
  return items;
}

/** Labels whose values are parsed as a comma-separated list of author/reviewer
 * handles (see parseLinkedItems). Other fields (Category, Sponsor, ...) render
 * as plain text. */
const LINKED_FIELDS = new Set(["Authors", "Required Reviewers"]);

/** Turn a parsed author/reviewer value into a React fragment — anchor tags
 * for entries with an href, plain spans for plain-text fallbacks, joined by
 * ", " separators. Falls back to the raw value string if nothing parsed (an
 * empty span already suppresses render, so this only matters if a caller
 * bypasses the `useful_fields`-style filter). */
function renderLinkedList(value: string) {
  const items = parseLinkedItems(value);
  if (items.length === 0) return value;
  return (
    <>
      {items.map((item, i) => (
        <span key={`item-${i}`}>
          {i > 0 ? ", " : ""}
          {item.href ? (
            <a
              className="dep-meta-link"
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {item.label}
            </a>
          ) : (
            item.label
          )}
        </span>
      ))}
    </>
  );
}

/* Theme-aware via Fern's --pst-color-* vars (defined for both light and dark in
 * fern/main.css). Every background is paired with an explicit foreground so the
 * card never inherits an adjacent callout's color. */
const DEP_META_CSS = `
.dep-meta{position:relative;overflow:hidden;margin:0 0 2rem;padding:20px 22px;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:14px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);}
/* Status accent: tint the card border and lay a faint status-colored wash in
   the top-right corner (echoes the index card's ghosted watermark). --dep-accent
   is set inline per DEP; no hard left rail. */
.dep-meta::after{content:"";position:absolute;top:-40%;right:-10%;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,var(--dep-accent,transparent) 0%,transparent 70%);opacity:.06;pointer-events:none;}
.dep-meta--draft{border-color:rgba(224,168,0,.38);}
.dep-meta--proposed{border-color:rgba(91,141,239,.38);}
.dep-meta--accepted{border-color:rgba(118,185,0,.48);}
.dep-meta--rejected{border-color:rgba(220,72,72,.36);}
.dep-meta--muted{border-color:rgba(127,127,127,.32);}
.dep-meta-top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.dep-meta-eyebrow{margin:0 0 4px;color:var(--nv-color-green,#76b900);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
.dep-meta-title{margin:0;font-size:1.35rem;line-height:1.25;color:var(--pst-color-heading,inherit);}
.dep-meta-pill{display:inline-flex;align-items:center;gap:6px;min-height:26px;padding:3px 12px;border-radius:999px;font-size:12px;font-weight:800;letter-spacing:.02em;white-space:nowrap;}
.dep-meta-pill::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;opacity:.9;}
.dep-meta-pill--draft{background:rgba(224,168,0,.18);color:#8a6100;}
.dark .dep-meta-pill--draft{color:#ffcf5a;}
.dep-meta-pill--proposed{background:rgba(91,141,239,.18);color:#2f5fd0;}
.dark .dep-meta-pill--proposed{color:#8fb0ff;}
.dep-meta-pill--accepted{background:rgba(118,185,0,.20);color:#4c7a00;}
.dark .dep-meta-pill--accepted{color:var(--nv-color-green,#76b900);}
.dep-meta-pill--rejected{background:rgba(220,72,72,.16);color:#b23636;}
.dark .dep-meta-pill--rejected{color:#ff8a8a;}
.dep-meta-pill--muted{background:rgba(127,127,127,.16);color:var(--pst-color-text-muted,#6b6b6b);}
.dep-meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px 24px;margin-top:16px;padding-top:14px;border-top:1px solid var(--border,var(--grayscale-a5,#e2e2e2));}
.dep-meta-field{display:flex;flex-direction:column;gap:3px;min-width:0;}
.dep-meta-field b{font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--pst-color-text-muted,#777);}
.dep-meta-field span{font-size:13.5px;line-height:1.35;color:var(--pst-color-text-base,#1a1a1a);overflow-wrap:break-word;}
.dep-meta-link{color:#4c7a00;text-decoration:none;font-weight:600;}
.dark .dep-meta-link{color:var(--nv-color-green,#76b900);}
.dep-meta-link:hover{text-decoration:underline;}
.dep-meta-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.dep-meta-action{display:inline-flex;align-items:center;gap:6px;min-height:34px;padding:6px 14px;border:1px solid var(--nv-color-green,#76b900);border-radius:8px;color:var(--pst-color-text-base,#1a1a1a);font-size:13px;font-weight:600;line-height:1;text-decoration:none;white-space:nowrap;}
.dep-meta-action:hover{background:rgba(118,185,0,.12);text-decoration:none;}
/* Lifecycle stepper: Draft -> Under Review -> Accepted -> Implemented, filled to
   the current stage. Off-track terminal states branch (see DepLifecycle):
   Draft -> Under Review -> [Rejected|Deferred|...]. --dep-accent is set per step. */
.dep-meta-steps{display:flex;align-items:flex-start;margin-top:16px;padding-top:16px;border-top:1px solid var(--border,var(--grayscale-a5,#e2e2e2));position:relative;z-index:1;}
.dep-meta-step{flex:1;display:flex;flex-direction:column;align-items:center;gap:7px;position:relative;text-align:center;}
.dep-meta-step::before{content:"";position:absolute;top:6px;left:-50%;width:100%;height:2px;background:var(--border,var(--grayscale-a5,#dcdcdc));z-index:0;}
.dep-meta-step:first-child::before{display:none;}
.dep-meta-step.is-done::before,.dep-meta-step.is-current::before,.dep-meta-step.is-terminal::before{background:var(--dep-accent,var(--nv-color-green,#76b900));}
.dep-meta-dot{position:relative;z-index:1;width:14px;height:14px;border-radius:50%;background:var(--pst-color-surface,#fff);border:2px solid var(--border,var(--grayscale-a5,#ccc));box-sizing:border-box;}
.dep-meta-step.is-done .dep-meta-dot,.dep-meta-step.is-terminal .dep-meta-dot{background:var(--dep-accent,var(--nv-color-green,#76b900));border-color:var(--dep-accent,var(--nv-color-green,#76b900));}
.dep-meta-step.is-current .dep-meta-dot{background:var(--dep-accent,var(--nv-color-green,#76b900));border-color:var(--dep-accent,var(--nv-color-green,#76b900));box-shadow:0 0 0 4px color-mix(in srgb,var(--dep-accent,#76b900) 22%,transparent);}
.dep-meta-steplabel{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;line-height:1.2;color:var(--pst-color-text-muted,#888);overflow-wrap:normal;word-break:keep-all;hyphens:none;}
.dep-meta-step.is-done .dep-meta-steplabel,.dep-meta-step.is-current .dep-meta-steplabel,.dep-meta-step.is-terminal .dep-meta-steplabel{color:var(--pst-color-text-base,#1a1a1a);}
`;

/** A step in the lifecycle stepper. `state` drives the dot/connector fill via
 * the is-* CSS classes; `accent` is the per-step color (green for reached
 * happy-path stages, the terminal color for an off-track end node). */
type LifecycleStep = {
  label: string;
  state: "done" | "current" | "todo" | "terminal";
  accent: string;
};

/**
 * Render the lifecycle stepper for a DEP.
 *
 * Happy path: the four LIFECYCLE_STAGES, filled up to the current stage. Off-
 * track terminal states (Rejected / Deferred / Replaced — statusVariant
 * "rejected" or "muted") instead show Draft -> Under Review -> <terminal>, so
 * the card conveys "this reached review, then ended" rather than implying it
 * marched to Implemented.
 */
function DepLifecycle({ status }: { status: string }) {
  const variant = statusVariant(status);
  const terminalOff = variant === "rejected" || variant === "muted";

  let steps: LifecycleStep[];
  if (terminalOff) {
    steps = [
      { label: "Draft", state: "done", accent: STATUS_ACCENT.accepted },
      { label: "Under Review", state: "done", accent: STATUS_ACCENT.accepted },
      { label: status, state: "terminal", accent: STATUS_ACCENT[variant] },
    ];
  } else {
    const current = lifecycleStage(status);
    const accent = STATUS_ACCENT[variant];
    steps = LIFECYCLE_STAGES.map((label, i) => ({
      label,
      state: i < current ? "done" : i === current ? "current" : "todo",
      accent,
    }));
  }

  return (
    <div className="dep-meta-steps" role="img" aria-label={`Lifecycle: ${status}`}>
      {steps.map((step, i) => (
        <div
          className={`dep-meta-step is-${step.state}`}
          key={`${step.label}-${i}`}
          style={{ "--dep-accent": step.accent } as CSSProperties}
        >
          <span className="dep-meta-dot" />
          <span className="dep-meta-steplabel">{step.label}</span>
        </div>
      ))}
    </div>
  );
}

export function DepMetadata({
  dep,
  title,
  status,
  category,
  owningSig,
  participatingSigs,
  authors,
  sponsor,
  requiredReviewers,
  reviewDate,
  replaces,
  replacedBy,
  pr,
  owner = "ai-dynamo",
  repo = "dynamo",
  trackingIssue,
}: DepMetadataProps) {
  const variant = statusVariant(status);
  const prUrl = pr ? `https://github.com/${owner}/${repo}/pull/${pr}` : undefined;

  const fields: Array<{ label: string; value: string }> = [];
  const push = (label: string, value?: string) => {
    if (value && value.trim()) fields.push({ label, value });
  };
  push("Category", category);
  push("Owning SIG", owningSig);
  push("Participating SIGs", participatingSigs);
  push("Authors", authors);
  push("Sponsor", sponsor);
  push("Required Reviewers", requiredReviewers);
  push("Review Date", reviewDate);
  push("Replaces", replaces);
  push("Replaced By", replacedBy);

  return (
    <div
      className={`dep-meta dep-meta--${variant}`}
      style={{ "--dep-accent": STATUS_ACCENT[variant] } as CSSProperties}
    >
      <style dangerouslySetInnerHTML={{ __html: DEP_META_CSS }} />
      <div className="dep-meta-top">
        <div>
          <p className="dep-meta-eyebrow">
            Dynamo Enhancement Proposal{dep ? ` \u00B7 DEP-${dep}` : ""}
          </p>
          {title ? <h2 className="dep-meta-title">{title}</h2> : null}
        </div>
        <span className={`dep-meta-pill dep-meta-pill--${variant}`}>{status}</span>
      </div>

      {fields.length > 0 && (
        <div className="dep-meta-grid">
          {fields.map((f) => (
            <div className="dep-meta-field" key={f.label}>
              <b>{f.label}</b>
              <span>
                {LINKED_FIELDS.has(f.label) ? renderLinkedList(f.value) : f.value}
              </span>
            </div>
          ))}
        </div>
      )}

      {status && status.trim() ? <DepLifecycle status={status} /> : null}

      {(prUrl || trackingIssue) && (
        <div className="dep-meta-actions">
          {prUrl && (
            <a className="dep-meta-action" href={prUrl} target="_blank" rel="noopener noreferrer">
              View PR #{pr} &rarr;
            </a>
          )}
          {trackingIssue && (
            <a className="dep-meta-action" href={trackingIssue} target="_blank" rel="noopener noreferrer">
              Tracking issue &rarr;
            </a>
          )}
        </div>
      )}
    </div>
  );
}
