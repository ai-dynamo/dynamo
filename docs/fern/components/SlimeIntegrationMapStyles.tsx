/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Styles for SlimeIntegrationMap. SlimeIntegrationMap renders this itself,
 * so a page only renders <SlimeIntegrationMap />.
 *
 * Named *Styles.tsx deliberately: check_style_components.py guards CSS
 * template literals by that filename, so a stray backtick or interpolation
 * here fails the hook instead of the docs build.
 *
 * The palette follows the RL overview diagram (rl-dynamo-components.svg):
 * NVIDIA green for Dynamo, amber for the training framework, and slate for
 * KubeRay. Fern sets the dark theme as a .dark class on the root element.
 */
const SLIME_INTEGRATION_MAP_CSS = `
.slmap {
  --slmap-text: #1a1a1a;
  --slmap-muted: #5a6572;
  --slmap-faint: #9ca3af;
  --slmap-rule: rgba(0, 0, 0, 0.12);
  --slmap-surface: #ffffff;
  --slmap-panel: #fafafa;
  --slmap-dot: rgba(0, 0, 0, 0.1);
  --slmap-wire: #cdd2d8;
  --slmap-green: #76b900;
  --slmap-green-ink: #3f6600;
  --slmap-green-tint: #f4f9e8;
  --slmap-amber: #d97706;
  --slmap-amber-ink: #92400e;
  --slmap-amber-tint: #fef5e7;
  --slmap-slate: #64748b;
  --slmap-slate-ink: #374151;
  --slmap-slate-tint: #f2f4f7;
  container-type: inline-size;
  margin: 1.5rem 0 2rem;
  color: var(--slmap-text);
  font-size: 15px;
  line-height: 1.55;
}

.dark .slmap {
  --slmap-text: #eeeeee;
  --slmap-muted: #a3a9b1;
  --slmap-faint: #6b7280;
  --slmap-rule: rgba(255, 255, 255, 0.14);
  --slmap-surface: #141414;
  --slmap-panel: #0a0a0a;
  --slmap-dot: rgba(255, 255, 255, 0.09);
  --slmap-wire: #3a3f46;
  --slmap-green-ink: #a6dc4f;
  --slmap-green-tint: #121a07;
  --slmap-amber: #f59e0b;
  --slmap-amber-ink: #fbbf24;
  --slmap-amber-tint: #1f1405;
  --slmap-slate: #94a3b8;
  --slmap-slate-ink: #cbd5e1;
  --slmap-slate-tint: #111418;
}

.slmap *,
.slmap *::before,
.slmap *::after {
  box-sizing: border-box;
}

/* ---- Owner cards ---- */

.slmap-owners {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.slmap-owner {
  display: flex;
  flex-direction: column;
  margin: 0;
  padding: 16px 16px 14px;
  border: 1px solid var(--slmap-rule);
  border-top: 3px solid var(--slmap-owner);
  border-radius: 12px;
  background: var(--slmap-surface);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.slmap-owner:hover {
  border-color: var(--slmap-owner);
  box-shadow: 0 0 0 1px var(--slmap-owner);
}

.slmap-owner--dynamo {
  --slmap-owner: var(--slmap-green);
  --slmap-owner-ink: var(--slmap-green-ink);
  --slmap-owner-tint: var(--slmap-green-tint);
}

.slmap-owner--slime {
  --slmap-owner: var(--slmap-amber);
  --slmap-owner-ink: var(--slmap-amber-ink);
  --slmap-owner-tint: var(--slmap-amber-tint);
}

.slmap-owner--kuberay {
  --slmap-owner: var(--slmap-slate);
  --slmap-owner-ink: var(--slmap-slate-ink);
  --slmap-owner-tint: var(--slmap-slate-tint);
}

.slmap-owner__eyebrow {
  margin: 0;
  color: var(--slmap-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slmap-owner__num {
  margin-right: 6px;
  color: var(--slmap-owner-ink);
}

.slmap-owner__name {
  margin: 6px 0 6px;
  color: var(--slmap-text);
  font-size: 19px;
  font-weight: 700;
  line-height: 1.25;
}

.slmap-owner__text {
  flex: 1;
  margin: 0;
  color: var(--slmap-text);
  font-size: 14px;
  line-height: 1.55;
}

.slmap-flow {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 14px;
}

.slmap-flow span {
  padding: 3px 10px;
  border: 1px solid var(--slmap-owner);
  border-radius: 999px;
  background: var(--slmap-owner-tint);
  color: var(--slmap-owner-ink);
  font-size: 12.5px;
  font-weight: 600;
  white-space: nowrap;
}

.slmap-flow b {
  color: var(--slmap-owner);
  font-size: 15px;
  line-height: 1;
}

.slmap-caption {
  margin: 12px 0 0;
  color: var(--slmap-muted);
  font-size: 14px;
  text-align: center;
}

/* ---- One training round ---- */

.slmap-world {
  margin-top: 16px;
  border: 1px solid var(--slmap-rule);
  border-radius: 12px;
  background: var(--slmap-surface);
  overflow: hidden;
}

.slmap-world__head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px 16px;
  padding: 14px 16px 10px;
}

.slmap-eyebrow {
  margin: 0;
  color: var(--slmap-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slmap-round {
  margin: 2px 0 0;
  color: var(--slmap-text);
  font-size: 15px;
  font-weight: 700;
}

.slmap-play {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  padding: 7px 16px;
  border: 1px solid var(--slmap-green);
  border-radius: 999px;
  background: var(--slmap-green);
  color: #000000;
  font: inherit;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.2;
  cursor: pointer;
  transition: filter 0.15s ease;
}

.slmap-play:hover {
  filter: brightness(1.08);
}

.slmap-play__icon {
  display: inline-block;
  width: 1em;
  text-align: center;
}

.slmap-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0;
  padding: 0 16px 12px;
  list-style: none;
}

.slmap-steps li {
  margin: 0;
  padding: 0;
}

.slmap-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  padding: 4px 11px 4px 5px;
  border: 1px solid var(--slmap-rule);
  border-radius: 999px;
  background: var(--slmap-surface);
  color: var(--slmap-muted);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease;
}

.slmap-step:hover {
  border-color: var(--slmap-green);
  color: var(--slmap-text);
}

.slmap-step__num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: var(--slmap-panel);
  color: var(--slmap-muted);
  font-size: 11.5px;
  font-weight: 700;
}

.slmap-step.is-done .slmap-step__num {
  background: var(--slmap-green-tint);
  color: var(--slmap-green-ink);
}

.slmap-step[aria-current="step"] {
  border-color: var(--slmap-green);
  background: var(--slmap-green-tint);
  color: var(--slmap-green-ink);
}

.slmap-step[aria-current="step"] .slmap-step__num {
  background: var(--slmap-green);
  color: #000000;
}

.slmap-play:focus-visible,
.slmap-step:focus-visible {
  outline: 2px solid var(--slmap-green);
  outline-offset: 2px;
}

.slmap-stage {
  border-top: 1px solid var(--slmap-rule);
  border-bottom: 1px solid var(--slmap-rule);
  background-color: var(--slmap-panel);
  background-image: radial-gradient(var(--slmap-dot) 0.9px, transparent 0.9px);
  background-size: 14px 14px;
  overflow-x: auto;
}

.slmap-svg {
  display: block;
  width: 100%;
  min-width: 600px;
  height: auto;
  font-family: inherit;
}

.slmap-status {
  min-height: 104px;
  padding: 12px 16px 4px;
}

.slmap-status__title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 8px;
  margin: 0;
  color: var(--slmap-text);
  font-size: 15px;
  font-weight: 700;
}

.slmap-status__text {
  margin: 4px 0 0;
  color: var(--slmap-text);
  font-size: 14.5px;
}

.slmap-status code {
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--slmap-panel);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.86em;
}

.slmap-tag {
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.slmap-tag--dynamo {
  background: var(--slmap-green-tint);
  color: var(--slmap-green-ink);
  box-shadow: inset 0 0 0 1px var(--slmap-green);
}

.slmap-tag--slime {
  background: var(--slmap-amber-tint);
  color: var(--slmap-amber-ink);
  box-shadow: inset 0 0 0 1px var(--slmap-amber);
}

.slmap-tag--kuberay {
  background: var(--slmap-slate-tint);
  color: var(--slmap-slate-ink);
  box-shadow: inset 0 0 0 1px var(--slmap-slate);
}

.slmap-footnote {
  margin: 0;
  padding: 0 16px 14px;
  color: var(--slmap-muted);
  font-size: 12.5px;
}

/* ---- Diagram ---- */

.slmap-zone {
  stroke-width: 1.25;
  transition: stroke-width 0.25s ease, opacity 0.25s ease;
}

.slmap-zone--ray {
  fill: var(--slmap-slate-tint);
  stroke: var(--slmap-slate);
}

.slmap-zone--dgd {
  fill: var(--slmap-green-tint);
  stroke: var(--slmap-green);
}

.slmap-zone--engines {
  fill: none;
  stroke: var(--slmap-green);
  stroke-dasharray: 5 4;
}

.slmap-zone--volume {
  fill: var(--slmap-surface);
  stroke: var(--slmap-faint);
  stroke-dasharray: 5 4;
}

.slmap-zone.is-focus {
  stroke-width: 2.75;
}

.slmap-zone--volume.is-active {
  stroke: var(--slmap-amber);
  stroke-width: 1.75;
}

.slmap-zone-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
}

.slmap-zone-label--ray {
  fill: var(--slmap-slate-ink);
}

.slmap-zone-label--dgd,
.slmap-zone-label--engines {
  fill: var(--slmap-green-ink);
}

.slmap-zone-label--volume {
  fill: var(--slmap-muted);
}

.slmap-zone-owner {
  fill: var(--slmap-muted);
  font-size: 11.5px;
  font-style: italic;
}

.slmap-node {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.slmap-node.is-active {
  transform: translateY(-2px);
}

.slmap-node.is-dim {
  opacity: 0.38;
}

.slmap-node__box {
  fill: var(--slmap-surface);
  stroke-width: 1.5;
  transition: stroke-width 0.25s ease;
}

.slmap-node__halo {
  fill: none;
  stroke-width: 8;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.slmap-node[data-owner="slime"] .slmap-node__box,
.slmap-node[data-owner="slime"] .slmap-node__halo {
  stroke: var(--slmap-amber);
}

.slmap-node[data-owner="dynamo"] .slmap-node__box,
.slmap-node[data-owner="dynamo"] .slmap-node__halo {
  stroke: var(--slmap-green);
}

.slmap-node.is-active .slmap-node__box,
.slmap-node.is-focus .slmap-node__box {
  stroke-width: 2.25;
}

.slmap-node.is-active .slmap-node__halo,
.slmap-node.is-focus .slmap-node__halo {
  opacity: 0.2;
}

.slmap-node__title {
  fill: var(--slmap-text);
  font-size: 14.5px;
  font-weight: 700;
}

.slmap-node__sub {
  fill: var(--slmap-muted);
  font-size: 11.5px;
}

.slmap-wire {
  fill: none;
  stroke: var(--slmap-wire);
  stroke-width: 1.75;
  stroke-linecap: round;
  stroke-linejoin: round;
  transition: stroke 0.3s ease, opacity 0.3s ease;
}

.slmap-wire--control {
  stroke-dasharray: 6 5;
}

.slmap-head {
  fill: var(--slmap-wire);
  transition: fill 0.3s ease, opacity 0.3s ease;
}

.slmap-wire.is-active.slmap-tone--green {
  stroke: var(--slmap-green);
  stroke-width: 2.25;
}

.slmap-wire.is-active.slmap-tone--amber {
  stroke: var(--slmap-amber);
  stroke-width: 2.25;
}

.slmap-head.is-active.slmap-tone--green {
  fill: var(--slmap-green);
}

.slmap-head.is-active.slmap-tone--amber {
  fill: var(--slmap-amber);
}

.slmap-wire.is-quiet,
.slmap-head.is-quiet {
  opacity: 0.45;
}

.slmap-wire-label {
  fill: var(--slmap-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 11px;
  paint-order: stroke;
  stroke: var(--slmap-panel);
  stroke-width: 4px;
  stroke-linejoin: round;
  transition: fill 0.3s ease, opacity 0.3s ease;
}

.slmap-wire-label--on-dgd {
  stroke: var(--slmap-green-tint);
}

.slmap-wire-label--on-ray {
  stroke: var(--slmap-slate-tint);
}

.slmap-wire-label.is-active.slmap-tone--green {
  fill: var(--slmap-green-ink);
  font-weight: 700;
}

.slmap-wire-label.is-active.slmap-tone--amber {
  fill: var(--slmap-amber-ink);
  font-weight: 700;
}

.slmap-wire-label.is-quiet {
  opacity: 0.55;
}

.slmap-packet {
  stroke: var(--slmap-surface);
  stroke-width: 1.5;
}

.slmap-packet--green {
  fill: var(--slmap-green);
}

.slmap-packet--amber {
  fill: var(--slmap-amber);
}

.slmap-packet--return {
  fill: var(--slmap-green);
  opacity: 0.5;
}

.slmap-reward {
  transform-box: fill-box;
  transform-origin: center;
  animation: slmap-pop 0.35s ease-out both;
}

.slmap-reward--hit {
  fill: var(--slmap-green);
}

.slmap-reward--miss {
  fill: var(--slmap-surface);
  stroke: var(--slmap-faint);
  stroke-width: 1.5;
}

.slmap-progress-track {
  fill: var(--slmap-amber-tint);
}

.slmap-progress {
  fill: var(--slmap-amber);
  transform-box: fill-box;
  transform-origin: left center;
  animation: slmap-grow 5s linear both;
}

.slmap-version rect {
  fill: var(--slmap-green-tint);
  stroke: var(--slmap-green);
  stroke-width: 1;
}

.slmap-version text {
  fill: var(--slmap-green-ink);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
  font-weight: 700;
}

.slmap-version.is-loading rect {
  animation: slmap-pulse 1s ease-in-out infinite;
}

.slmap-chip rect {
  fill: var(--slmap-panel);
  stroke: var(--slmap-rule);
  stroke-width: 1;
}

.slmap-chip text {
  fill: var(--slmap-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
}

.slmap-chip.is-new rect {
  fill: var(--slmap-amber-tint);
  stroke: var(--slmap-amber);
}

.slmap-chip.is-new text {
  fill: var(--slmap-amber-ink);
  font-weight: 700;
}

@keyframes slmap-pop {
  from {
    opacity: 0;
    transform: scale(0.3);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

@keyframes slmap-grow {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}

@keyframes slmap-pulse {
  50% {
    fill: var(--slmap-surface);
  }
}

@container (max-width: 640px) {
  .slmap-owners {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 640px) {
  .slmap-owners {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (prefers-reduced-motion: reduce) {
  .slmap-packet {
    display: none;
  }

  .slmap-node,
  .slmap-node__box,
  .slmap-node__halo,
  .slmap-zone,
  .slmap-wire,
  .slmap-head,
  .slmap-wire-label {
    transition: none;
  }

  .slmap-node.is-active {
    transform: none;
  }

  .slmap-reward,
  .slmap-progress,
  .slmap-version.is-loading rect {
    animation: none;
  }
}
`;

export function SlimeIntegrationMapStyles() {
  return <style dangerouslySetInnerHTML={{ __html: SLIME_INTEGRATION_MAP_CSS }} />;
}

export default SlimeIntegrationMapStyles;
