/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DepMetadata — a styled header card that separates a DEP's metadata (front
 * matter) from the proposal body: DEP number + title, a color-coded status
 * pill (Draft / Proposed / Accepted / ...), a labeled field grid, and an action
 * row of links (View PR, Tracking Issue, Discussion).
 *
 * Server component (no "use client"), mirroring RecipeStyles.tsx / GiscusComments
 * .tsx. Registered via docs.yml `experimental.mdx-components: ./components`;
 * import per page: import { DepMetadata } from "@/components/DepMetadata";
 * The CSS ships as a page-level <style> (not docs.yml `css:`) so it survives the
 * shared NVIDIA global theme at publish, per the RecipeStyles constraint.
 */

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
  /** Lifecycle status: Draft | Proposed | Under Review | Accepted | Replaced | Deferred | Rejected. */
  status: string;
  category?: string;
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
  /** URL to the GitHub Discussion for this DEP. */
  discussionsTo?: string;
}

/** Map a free-text status to a pill variant (color). */
function statusVariant(status: string): string {
  const s = (status || "").toLowerCase();
  if (/accept|approv|final|active|ratif/.test(s)) return "accepted";
  if (/propos|review/.test(s)) return "proposed";
  if (/reject|withdraw/.test(s)) return "rejected";
  if (/replac|supersed|deferr|defer/.test(s)) return "muted";
  return "draft";
}

/* Theme-aware via Fern's --pst-color-* vars (defined for both light and dark in
 * fern/main.css). Every background is paired with an explicit foreground so the
 * card never inherits an adjacent callout's color. */
const DEP_META_CSS = `
.dep-meta{margin:0 0 2rem;padding:20px 22px;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:14px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);}
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
.dep-meta-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.dep-meta-action{display:inline-flex;align-items:center;gap:6px;min-height:34px;padding:6px 14px;border:1px solid var(--nv-color-green,#76b900);border-radius:8px;color:var(--pst-color-text-base,#1a1a1a);font-size:13px;font-weight:600;line-height:1;text-decoration:none;white-space:nowrap;}
.dep-meta-action:hover{background:rgba(118,185,0,.12);text-decoration:none;}
`;

export function DepMetadata({
  dep,
  title,
  status,
  category,
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
  discussionsTo,
}: DepMetadataProps) {
  const variant = statusVariant(status);
  const prUrl = pr ? `https://github.com/${owner}/${repo}/pull/${pr}` : undefined;

  const fields: Array<{ label: string; value: string }> = [];
  const push = (label: string, value?: string) => {
    if (value && value.trim()) fields.push({ label, value });
  };
  push("Category", category);
  push("Authors", authors);
  push("Sponsor", sponsor);
  push("Required Reviewers", requiredReviewers);
  push("Review Date", reviewDate);
  push("Replaces", replaces);
  push("Replaced By", replacedBy);

  return (
    <div className="dep-meta">
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
              <span>{f.value}</span>
            </div>
          ))}
        </div>
      )}

      {(prUrl || trackingIssue || discussionsTo) && (
        <div className="dep-meta-actions">
          {prUrl && (
            <a className="dep-meta-action" href={prUrl} target="_blank" rel="noopener">
              View PR #{pr} &rarr;
            </a>
          )}
          {trackingIssue && (
            <a className="dep-meta-action" href={trackingIssue} target="_blank" rel="noopener">
              Tracking issue &rarr;
            </a>
          )}
          {discussionsTo && (
            <a className="dep-meta-action" href={discussionsTo} target="_blank" rel="noopener">
              Discussion &rarr;
            </a>
          )}
        </div>
      )}
    </div>
  );
}
