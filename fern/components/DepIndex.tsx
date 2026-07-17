/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DepIndex — SSR mount point for the Proposals index "registry" grid.
 *
 * Renders ONLY a static <div id="dep-index"> plus a page-level <style>. The
 * runtime lives in `fern/js/dep-index.js` (injected site-wide via docs.yml
 * `js:`); it reads the build-time dataset `window.__DEP_INDEX` (emitted by
 * fern/scripts/sync_deps.py into fern/js/dep-index-data.js) and renders a
 * filterable/sortable card grid of every DEP into this mount. Each card links
 * to the DEP's stable /proposals/<slug> page.
 *
 * This mirrors the PrInlineComments SSR-mount + external-runtime pattern:
 * SSR of a plain div is reliable, and decoupling the runtime from MDX keeps it
 * resilient to Fern-pipeline changes. The CSS ships as a page-level <style>
 * (not docs.yml `css:`) so it survives the shared NVIDIA global theme at
 * publish — the same constraint DepMetadata/RecipeStyles follow.
 *
 * Card design (approved): squared card + rounded status pill + mono tags +
 * ghosted DEP-number watermark + Fern-native green-ring hover, with the
 * submitter's GitHub @handle in the footer. The pill colours + status->variant
 * bucketing match DepMetadata and the sidebar pill so a DEP reads the same
 * everywhere.
 *
 * Registered via docs.yml `experimental.mdx-components: ./components`. Import
 * per page: `import { DepIndex } from "@/components/DepIndex";`
 */

/* Theme-aware via Fern's --pst-color-* / --nv-color-green vars (defined for
 * both light and dark; .dark overrides mirror DepMetadata.tsx). */
const DEP_INDEX_CSS = `
.dep-index{--dep-mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"RobotoMono",monospace;margin:0 0 2rem;}
/* Slim result-count line. No title/heading here — registry.mdx frontmatter
   already renders the "Dynamo Enhancement Proposals" H1 + subtitle above. */
.dep-index-head{display:flex;align-items:baseline;gap:16px;margin:0 0 14px;}
.dep-index-count{font-family:var(--dep-mono);font-size:12px;font-weight:600;letter-spacing:.06em;color:var(--pst-color-text-muted,#6b6b6b);text-transform:uppercase;white-space:nowrap;}
.dep-index-filters{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:18px;}
.dep-index-chip{font-family:var(--dep-mono);font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));background:transparent;color:var(--pst-color-text-muted,#6b6b6b);padding:5px 11px;border-radius:2px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;line-height:1;}
.dep-index-chip .dep-index-sw{width:8px;height:8px;border-radius:0;}
.dep-index-chip.is-active{border-color:var(--nv-color-green,#76b900);color:var(--pst-color-heading,#111);background:rgba(118,185,0,.08);}
.dep-index-select{font-family:var(--dep-mono);font-size:11px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));background:var(--pst-color-surface,#fbfbfa);color:var(--pst-color-text-base,#1a1a1a);padding:6px 9px;border-radius:2px;cursor:pointer;}
.dep-index-select--sig{margin-left:auto;}
.dep-index-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}
.dep-index-card{position:relative;display:block;background:var(--pst-color-surface,#fbfbfa);border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:2px;padding:13px 16px;overflow:hidden;text-decoration:none;color:var(--pst-color-text-base,#1a1a1a);transition:box-shadow .2s ease,background .15s ease;}
.dep-index-card:hover{background:var(--pst-color-surface-hover,rgba(118,185,0,.04));box-shadow:0 0 0 1px var(--nv-color-green,#76b900);text-decoration:none;}
.dep-index-card::after{content:attr(data-num);position:absolute;right:6px;bottom:-18px;font-family:var(--dep-mono);font-size:70px;font-weight:800;letter-spacing:-4px;line-height:1;color:var(--dep-accent,var(--nv-color-green,#76b900));opacity:.08;pointer-events:none;user-select:none;}
.dep-index-top{display:flex;align-items:center;justify-content:space-between;gap:10px;position:relative;z-index:1;}
.dep-index-num{font-family:var(--dep-mono);font-size:11px;font-weight:700;letter-spacing:.12em;color:#4c7a00;}
.dark .dep-index-num{color:var(--nv-color-green,#76b900);}
.dep-index-cardtitle{position:relative;z-index:1;margin:8px 0 10px;font-size:14.5px;font-weight:750;line-height:1.3;color:var(--pst-color-heading,inherit);}
.dep-index-tags{display:flex;gap:6px;flex-wrap:wrap;position:relative;z-index:1;margin-bottom:11px;}
.dep-index-tag{font-family:var(--dep-mono);font-size:9.5px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--pst-color-text-muted,#6b6b6b);border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:2px;padding:2px 7px;}
.dep-index-tag--sig{color:#4c7a00;border-color:rgba(118,185,0,.4);}
.dark .dep-index-tag--sig{color:var(--nv-color-green,#76b900);}
.dep-index-foot{display:flex;align-items:center;justify-content:space-between;gap:8px;position:relative;z-index:1;}
.dep-index-who{display:flex;align-items:center;gap:8px;min-width:0;}
.dep-index-authors{display:flex;align-items:center;flex:none;}
.dep-index-authors img{width:20px;height:20px;border-radius:50%;border:2px solid var(--pst-color-surface,#fbfbfa);margin-left:-6px;background:var(--pst-color-surface-hover,#eee);}
.dep-index-authors img:first-child{margin-left:0;}
.dep-index-submitter{font-family:var(--dep-mono);font-size:11px;color:var(--pst-color-text-base,#1a1a1a);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dep-index-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:800;letter-spacing:.02em;white-space:nowrap;}
.dep-index-pill::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.9;}
.dep-index-pill--draft{background:rgba(224,168,0,.18);color:#8a6100;}
.dark .dep-index-pill--draft{color:#ffcf5a;}
.dep-index-pill--proposed{background:rgba(91,141,239,.18);color:#2f5fd0;}
.dark .dep-index-pill--proposed{color:#8fb0ff;}
.dep-index-pill--accepted{background:rgba(118,185,0,.20);color:#4c7a00;}
.dark .dep-index-pill--accepted{color:var(--nv-color-green,#76b900);}
.dep-index-pill--rejected{background:rgba(220,72,72,.16);color:#b23636;}
.dark .dep-index-pill--rejected{color:#ff8a8a;}
.dep-index-pill--muted{background:rgba(127,127,127,.16);color:var(--pst-color-text-muted,#6b6b6b);}
.dark .dep-index-pill--muted{color:#b0b0b0;}
.dep-index-empty{grid-column:1/-1;color:var(--pst-color-text-muted,#6b6b6b);font-size:13px;padding:26px;text-align:center;border:1px dashed var(--border,var(--grayscale-a5,#dcdcdc));border-radius:2px;font-family:var(--dep-mono);}
`;

export function DepIndex() {
  return (
    <div id="dep-index" className="dep-index">
      <style dangerouslySetInnerHTML={{ __html: DEP_INDEX_CSS }} />
      {/* Runtime (fern/js/dep-index.js) renders the head + filters + grid here
          from window.__DEP_INDEX. This noscript fallback keeps the page useful
          without JS: the Proposals sidebar still lists every DEP. */}
      <noscript>
        <p>
          Enable JavaScript to browse the Dynamo Enhancement Proposals registry,
          or use the Proposals sidebar to open a specific DEP.
        </p>
      </noscript>
    </div>
  );
}
