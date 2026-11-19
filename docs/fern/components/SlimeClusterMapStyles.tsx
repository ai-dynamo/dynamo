/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Styles for SlimeClusterMap. SlimeClusterMap renders this itself, so a page
 * only renders <SlimeClusterMap />.
 *
 * Named *Styles.tsx deliberately: check_style_components.py guards CSS
 * template literals by that filename, so a stray backtick or interpolation
 * here fails the hook instead of the docs build.
 *
 * The palette matches SlimeIntegrationMapStyles: NVIDIA green for Dynamo and
 * inference, amber for training, and slate for KubeRay. Kubernetes terms add
 * a blue highlight, so the three groups of terms stay apart. The selected
 * term's group sets --slclu-lens on the root. Fern sets the dark theme as a
 * .dark class on the root element.
 */
const SLIME_CLUSTER_MAP_CSS = `
.slclu {
  --slclu-text: #1a1a1a;
  --slclu-muted: #5a6572;
  --slclu-faint: #9ca3af;
  --slclu-rule: rgba(0, 0, 0, 0.12);
  --slclu-surface: #ffffff;
  --slclu-panel: #fafafa;
  --slclu-dot: rgba(0, 0, 0, 0.1);
  --slclu-wire: #b8bfc8;
  --slclu-green: #76b900;
  --slclu-green-ink: #3f6600;
  --slclu-green-tint: #f4f9e8;
  --slclu-amber: #d97706;
  --slclu-amber-ink: #92400e;
  --slclu-amber-tint: #fef5e7;
  --slclu-slate: #64748b;
  --slclu-slate-ink: #374151;
  --slclu-slate-tint: #f2f4f7;
  --slclu-blue: #2563eb;
  --slclu-blue-ink: #1d4ed8;
  --slclu-blue-tint: #eff4ff;
  --slclu-lens: var(--slclu-blue);
  --slclu-lens-ink: var(--slclu-blue-ink);
  --slclu-lens-tint: var(--slclu-blue-tint);
  container-type: inline-size;
  margin: 1.5rem 0 2rem;
  color: var(--slclu-text);
  font-size: 15px;
  line-height: 1.55;
}

.dark .slclu {
  --slclu-text: #eeeeee;
  --slclu-muted: #a3a9b1;
  --slclu-faint: #6b7280;
  --slclu-rule: rgba(255, 255, 255, 0.14);
  --slclu-surface: #141414;
  --slclu-panel: #0a0a0a;
  --slclu-dot: rgba(255, 255, 255, 0.09);
  --slclu-wire: #4b5159;
  --slclu-green-ink: #a6dc4f;
  --slclu-green-tint: #121a07;
  --slclu-amber: #f59e0b;
  --slclu-amber-ink: #fbbf24;
  --slclu-amber-tint: #1f1405;
  --slclu-slate: #94a3b8;
  --slclu-slate-ink: #cbd5e1;
  --slclu-slate-tint: #111418;
  --slclu-blue: #60a5fa;
  --slclu-blue-ink: #93c5fd;
  --slclu-blue-tint: #0b1526;
}

.slclu *,
.slclu *::before,
.slclu *::after {
  box-sizing: border-box;
}

.slclu--dynamo {
  --slclu-lens: var(--slclu-green);
  --slclu-lens-ink: var(--slclu-green-ink);
  --slclu-lens-tint: var(--slclu-green-tint);
}

.slclu--ray {
  --slclu-lens: var(--slclu-slate);
  --slclu-lens-ink: var(--slclu-slate-ink);
  --slclu-lens-tint: var(--slclu-slate-tint);
}

/* ---- Term buttons ---- */

.slclu-terms {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin: 0 0 12px;
}

.slclu-terms__group {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.slclu-terms__group--kubernetes {
  --slclu-group: var(--slclu-blue);
  --slclu-group-ink: var(--slclu-blue-ink);
  --slclu-group-tint: var(--slclu-blue-tint);
}

.slclu-terms__group--dynamo {
  --slclu-group: var(--slclu-green);
  --slclu-group-ink: var(--slclu-green-ink);
  --slclu-group-tint: var(--slclu-green-tint);
}

.slclu-terms__group--ray {
  --slclu-group: var(--slclu-slate);
  --slclu-group-ink: var(--slclu-slate-ink);
  --slclu-group-tint: var(--slclu-slate-tint);
}

.slclu-terms__label {
  margin-right: 2px;
  color: var(--slclu-group-ink);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slclu-term {
  margin: 0;
  padding: 4px 11px;
  border: 1px solid var(--slclu-rule);
  border-radius: 999px;
  background: var(--slclu-surface);
  color: var(--slclu-text);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease;
}

.slclu-term:hover {
  border-color: var(--slclu-group);
}

.slclu-term[aria-pressed="true"] {
  border-color: var(--slclu-group);
  background: var(--slclu-group-tint);
  color: var(--slclu-group-ink);
  box-shadow: inset 0 0 0 1px var(--slclu-group);
}

.slclu-term:focus-visible {
  outline: 2px solid var(--slclu-group);
  outline-offset: 2px;
}

/* ---- Diagram ---- */

.slclu-stage {
  border: 1px solid var(--slclu-rule);
  border-radius: 12px;
  background-color: var(--slclu-panel);
  background-image: radial-gradient(var(--slclu-dot) 0.9px, transparent 0.9px);
  background-size: 14px 14px;
  overflow-x: auto;
}

.slclu-svg {
  display: block;
  width: 100%;
  min-width: 600px;
  height: auto;
  font-family: inherit;
}

.slclu-annot text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 11px;
  font-weight: 700;
}

.slclu-annot path {
  fill: none;
  stroke: var(--slclu-wire);
  stroke-width: 1.75;
}

.slclu-annot polygon {
  fill: var(--slclu-wire);
}

.slclu-frame {
  fill: var(--slclu-surface);
  fill-opacity: 0.55;
  stroke: var(--slclu-faint);
  stroke-width: 1.5;
  transition: stroke 0.2s ease, stroke-width 0.2s ease;
}

.slclu-zone-label {
  fill: var(--slclu-muted);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
}

.slclu-box__bg {
  fill: var(--slclu-surface);
  stroke: var(--slclu-rule);
  stroke-width: 1.25;
  transition: stroke 0.2s ease, stroke-width 0.2s ease, fill 0.2s ease;
}

.slclu-box--control .slclu-box__bg,
.slclu-box--addons .slclu-box__bg {
  fill: var(--slclu-panel);
  stroke-dasharray: 5 4;
  stroke: var(--slclu-faint);
}

.slclu-box--volume .slclu-box__bg {
  fill: var(--slclu-surface);
  stroke: var(--slclu-faint);
  stroke-dasharray: 5 4;
}

.slclu-box__title {
  fill: var(--slclu-text);
  font-size: 14px;
  font-weight: 700;
}

.slclu-frame.is-lit,
.slclu-box.is-lit .slclu-box__bg {
  stroke: var(--slclu-lens);
  stroke-width: 2.75;
  stroke-dasharray: none;
}

.slclu-box.is-lit .slclu-box__bg {
  fill: var(--slclu-lens-tint);
}

.slclu-object rect {
  fill: var(--slclu-surface);
  stroke: var(--slclu-faint);
  stroke-width: 1;
  transition: fill 0.2s ease, stroke 0.2s ease, opacity 0.2s ease;
}

.slclu-object text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
  font-weight: 700;
}

.slclu-object.is-lit rect {
  fill: var(--slclu-lens-tint);
  stroke: var(--slclu-lens);
  stroke-width: 1.75;
}

.slclu-object.is-lit text {
  fill: var(--slclu-lens-ink);
}

.slclu-object.is-dim {
  opacity: 0.35;
}

.slclu-watch path {
  fill: none;
  stroke: var(--slclu-faint);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}

.slclu-watch polygon {
  fill: var(--slclu-faint);
}

.slclu-watch text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
  font-weight: 700;
}

.slclu-watch.is-lit path {
  stroke: var(--slclu-lens);
  stroke-width: 2;
}

.slclu-watch.is-lit polygon {
  fill: var(--slclu-lens);
}

.slclu-watch.is-lit text {
  fill: var(--slclu-lens-ink);
}

.slclu-gpus__label {
  fill: var(--slclu-muted);
  font-size: 11px;
  font-weight: 600;
}

.slclu-gpu {
  stroke-width: 1.25;
}

.slclu-gpu--training {
  fill: var(--slclu-amber);
  stroke: var(--slclu-amber);
}

.slclu-gpu--inference {
  fill: var(--slclu-green);
  stroke: var(--slclu-green);
}

.slclu-gpu--free {
  fill: var(--slclu-surface);
  stroke: var(--slclu-faint);
}

.slclu-mounts path {
  fill: none;
  stroke: var(--slclu-wire);
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
}

.slclu-mounts polygon {
  fill: var(--slclu-wire);
}

.slclu-mounts text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
  font-weight: 700;
}

.slclu-file rect {
  fill: var(--slclu-panel);
  stroke: var(--slclu-rule);
  stroke-width: 1;
}

.slclu-file text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
}

.slclu-pod {
  transition: opacity 0.2s ease;
}

.slclu-pod--dynamo {
  --slclu-owner: var(--slclu-green);
}

.slclu-pod--kuberay {
  --slclu-owner: var(--slclu-slate);
}

.slclu-pod__box {
  fill: var(--slclu-surface);
  stroke: var(--slclu-owner);
  stroke-width: 1.5;
  transition: stroke 0.2s ease, stroke-width 0.2s ease;
}

.slclu-pod__halo {
  fill: none;
  stroke: var(--slclu-lens);
  stroke-width: 9;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.slclu-pod.is-lit .slclu-pod__box {
  stroke: var(--slclu-lens);
  stroke-width: 2.5;
}

.slclu-pod.is-lit .slclu-pod__halo {
  opacity: 0.22;
}

.slclu-pod.is-dim {
  opacity: 0.3;
}

.slclu-pod__name {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10px;
  font-weight: 700;
}

.slclu-proc__lit {
  fill: var(--slclu-lens-tint);
  stroke: var(--slclu-lens);
  stroke-width: 1.25;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.slclu-proc.is-lit .slclu-proc__lit {
  opacity: 1;
}

.slclu-proc__name {
  fill: var(--slclu-text);
  font-size: 12px;
  font-weight: 600;
}

.slclu-chip rect {
  fill: var(--slclu-panel);
  stroke: var(--slclu-faint);
  stroke-width: 1;
}

.slclu-chip text {
  fill: var(--slclu-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.slclu-chip--training rect {
  fill: var(--slclu-amber);
  stroke: var(--slclu-amber);
}

.slclu-chip--inference rect {
  fill: var(--slclu-green);
  stroke: var(--slclu-green);
}

.slclu-chip--training text,
.slclu-chip--inference text {
  fill: #000000;
}

/* ---- Status and legend ---- */

.slclu-status {
  min-height: 92px;
  padding: 12px 2px 4px;
}

.slclu-status__title {
  margin: 0;
  color: var(--slclu-text);
  font-size: 15px;
  font-weight: 700;
}

.slclu-status__text {
  margin: 4px 0 0;
  color: var(--slclu-text);
  font-size: 14.5px;
}

.slclu-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  padding: 6px 2px 0;
  color: var(--slclu-muted);
  font-size: 12.5px;
}

.slclu-legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.slclu-key {
  display: inline-block;
  width: 14px;
  height: 10px;
  border-radius: 3px;
}

.slclu-key--dynamo {
  border: 1.5px solid var(--slclu-green);
}

.slclu-key--kuberay {
  border: 1.5px solid var(--slclu-slate);
}

.slclu-key--gpu {
  background: linear-gradient(90deg, var(--slclu-amber) 50%, var(--slclu-green) 50%);
}

@media (prefers-reduced-motion: reduce) {
  .slclu-term,
  .slclu-frame,
  .slclu-box__bg,
  .slclu-object rect,
  .slclu-pod,
  .slclu-pod__box,
  .slclu-pod__halo,
  .slclu-proc__lit {
    transition: none;
  }
}
`;

export function SlimeClusterMapStyles() {
  return <style dangerouslySetInnerHTML={{ __html: SLIME_CLUSTER_MAP_CSS }} />;
}

export default SlimeClusterMapStyles;
