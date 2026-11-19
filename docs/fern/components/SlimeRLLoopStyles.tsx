/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Styles for SlimeRLLoop. SlimeRLLoop renders this itself, so a page only
 * renders <SlimeRLLoop />.
 *
 * Named *Styles.tsx deliberately: check_style_components.py guards CSS
 * template literals by that filename, so a stray backtick or interpolation
 * here fails the hook instead of the docs build.
 *
 * The palette matches SlimeIntegrationMapStyles: NVIDIA green for inference,
 * amber for training, slate for Ray and KubeRay, and gray for inputs. Each
 * slrl-tone--* class sets the three tone variables that the rules below read.
 * Fern sets the dark theme as a .dark class on the root element.
 */
const SLIME_RL_LOOP_CSS = `
.slrl {
  --slrl-text: #1a1a1a;
  --slrl-muted: #5a6572;
  --slrl-faint: #9ca3af;
  --slrl-rule: rgba(0, 0, 0, 0.12);
  --slrl-surface: #ffffff;
  --slrl-panel: #fafafa;
  --slrl-dot: rgba(0, 0, 0, 0.1);
  --slrl-green: #76b900;
  --slrl-green-ink: #3f6600;
  --slrl-green-tint: #f4f9e8;
  --slrl-amber: #d97706;
  --slrl-amber-ink: #92400e;
  --slrl-amber-tint: #fef5e7;
  --slrl-slate: #64748b;
  --slrl-slate-ink: #374151;
  --slrl-slate-tint: #f2f4f7;
  --slrl-gray: #9ca3af;
  --slrl-gray-ink: #3f4752;
  --slrl-gray-tint: #f4f5f7;
  --slrl-tip-bg: #1f2328;
  --slrl-tip-text: #f3f4f6;
  container-type: inline-size;
  margin: 1.5rem 0 2rem;
  color: var(--slrl-text);
  font-size: 15px;
  line-height: 1.55;
}

.dark .slrl {
  --slrl-text: #eeeeee;
  --slrl-muted: #a3a9b1;
  --slrl-faint: #6b7280;
  --slrl-rule: rgba(255, 255, 255, 0.14);
  --slrl-surface: #141414;
  --slrl-panel: #0a0a0a;
  --slrl-dot: rgba(255, 255, 255, 0.09);
  --slrl-green-ink: #a6dc4f;
  --slrl-green-tint: #121a07;
  --slrl-amber: #f59e0b;
  --slrl-amber-ink: #fbbf24;
  --slrl-amber-tint: #1f1405;
  --slrl-slate: #94a3b8;
  --slrl-slate-ink: #cbd5e1;
  --slrl-slate-tint: #111418;
  --slrl-gray: #6b7280;
  --slrl-gray-ink: #d1d5db;
  --slrl-gray-tint: #17191c;
  --slrl-tip-bg: #2a2f36;
  --slrl-tip-text: #f3f4f6;
}

.slrl *,
.slrl *::before,
.slrl *::after {
  box-sizing: border-box;
}

.slrl-tone--green {
  --slrl-tone: var(--slrl-green);
  --slrl-tone-ink: var(--slrl-green-ink);
  --slrl-tone-tint: var(--slrl-green-tint);
}

.slrl-tone--amber {
  --slrl-tone: var(--slrl-amber);
  --slrl-tone-ink: var(--slrl-amber-ink);
  --slrl-tone-tint: var(--slrl-amber-tint);
}

.slrl-tone--slate {
  --slrl-tone: var(--slrl-slate);
  --slrl-tone-ink: var(--slrl-slate-ink);
  --slrl-tone-tint: var(--slrl-slate-tint);
}

.slrl-tone--neutral {
  --slrl-tone: var(--slrl-gray);
  --slrl-tone-ink: var(--slrl-gray-ink);
  --slrl-tone-tint: var(--slrl-gray-tint);
}

.slrl-eyebrow {
  margin: 0;
  color: var(--slrl-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slrl > .slrl-eyebrow {
  margin-bottom: 8px;
}

/* ---- Parts tray ---- */

.slrl-tray {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.slrl-part {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 62px;
  margin: 0;
  padding: 10px 10px 10px 12px;
  border: 1px solid var(--slrl-rule);
  border-radius: 12px;
  background: var(--slrl-surface);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

.slrl-part.is-new {
  border-color: var(--slrl-tone);
  box-shadow: 0 0 0 1px var(--slrl-tone), 0 6px 16px -8px var(--slrl-tone);
}

.slrl-part.is-later {
  border-style: dashed;
}

.slrl-part__icon {
  display: inline-flex;
  flex: none;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 9px;
  background: var(--slrl-tone-tint);
  color: var(--slrl-tone-ink);
  transition: background-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
}

.slrl-part.is-later .slrl-part__icon {
  background: transparent;
  color: var(--slrl-faint);
  box-shadow: inset 0 0 0 1px var(--slrl-rule);
}

.slrl-part__icon svg {
  width: 20px;
  height: 20px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.slrl-part__text {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.slrl-part__name {
  color: var(--slrl-text);
  font-size: 14px;
  font-weight: 700;
  line-height: 1.3;
  white-space: nowrap;
}

.slrl-part__kind {
  color: var(--slrl-muted);
  font-size: 12.5px;
  line-height: 1.35;
}

.slrl-part__info {
  display: inline-flex;
  flex: none;
  align-self: flex-start;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  margin: 0;
  padding: 0;
  border: 1px solid var(--slrl-rule);
  border-radius: 999px;
  background: var(--slrl-surface);
  color: var(--slrl-muted);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 13px;
  font-style: italic;
  font-weight: 700;
  line-height: 1;
  cursor: help;
  transition: border-color 0.2s ease, color 0.2s ease, background-color 0.2s ease;
}

.slrl-part__info:hover,
.slrl-part.is-open .slrl-part__info {
  border-color: var(--slrl-tone);
  background: var(--slrl-tone-tint);
  color: var(--slrl-tone-ink);
}

.slrl-tip {
  position: absolute;
  z-index: 30;
  top: calc(100% + 6px);
  left: 0;
  width: max(100%, 250px);
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--slrl-tip-bg);
  color: var(--slrl-tip-text);
  font-size: 13px;
  font-weight: 400;
  line-height: 1.5;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
  opacity: 0;
  visibility: hidden;
  transform: translateY(-3px);
  transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s;
  pointer-events: none;
}

.dark .slrl-tip {
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.16), 0 10px 28px rgba(0, 0, 0, 0.5);
}

.slrl-part:nth-child(4n) .slrl-tip {
  right: 0;
  left: auto;
}

.slrl-part__info:hover + .slrl-tip,
.slrl-part__info:focus-visible + .slrl-tip,
.slrl-part.is-open .slrl-tip {
  opacity: 1;
  visibility: visible;
  transform: none;
}

.slrl-caption {
  margin: 10px 0 0;
  color: var(--slrl-muted);
  font-size: 13.5px;
  text-align: center;
}

.slrl-caption__i {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 17px;
  height: 17px;
  margin: 0 1px;
  border: 1px solid var(--slrl-rule);
  border-radius: 999px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 11px;
  font-style: italic;
  font-weight: 700;
  line-height: 1;
  vertical-align: 1px;
}

/* ---- Stepper ---- */

.slrl-world {
  margin-top: 16px;
  border: 1px solid var(--slrl-rule);
  border-radius: 12px;
  background: var(--slrl-surface);
  overflow: hidden;
}

.slrl-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 10px 16px;
  padding: 14px 16px 10px;
}

.slrl-title {
  margin: 2px 0 0;
  color: var(--slrl-text);
  font-size: 16px;
  font-weight: 700;
}

.slrl-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.slrl-nav,
.slrl-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  margin: 0;
  padding: 6px 13px;
  border: 1px solid var(--slrl-rule);
  border-radius: 999px;
  background: var(--slrl-surface);
  color: var(--slrl-text);
  font: inherit;
  font-size: 13.5px;
  font-weight: 600;
  line-height: 1.2;
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease, filter 0.15s ease;
}

.slrl-nav:hover:not(:disabled),
.slrl-toggle:hover {
  border-color: var(--slrl-green);
}

.slrl-nav:disabled {
  color: var(--slrl-faint);
  cursor: default;
}

.slrl-nav--next {
  border-color: var(--slrl-green);
  background: var(--slrl-green);
  color: #000000;
  font-weight: 700;
}

.slrl-nav--next:hover {
  filter: brightness(1.08);
}

.slrl-toggle__box {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border: 1.5px solid var(--slrl-faint);
  border-radius: 4px;
  color: #000000;
  font-size: 11px;
  line-height: 1;
}

.slrl-toggle[aria-pressed="true"] {
  border-color: var(--slrl-amber);
  background: var(--slrl-amber-tint);
  color: var(--slrl-amber-ink);
}

.slrl-toggle[aria-pressed="true"] .slrl-toggle__box {
  border-color: var(--slrl-amber);
  background: var(--slrl-amber);
}

.slrl-seg {
  display: inline-flex;
  padding: 2px;
  border: 1px solid var(--slrl-rule);
  border-radius: 999px;
  background: var(--slrl-panel);
}

.slrl-seg button {
  margin: 0;
  padding: 5px 12px;
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--slrl-muted);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.2;
  cursor: pointer;
}

.slrl-seg button[aria-pressed="true"] {
  background: var(--slrl-slate-tint);
  color: var(--slrl-slate-ink);
  box-shadow: inset 0 0 0 1px var(--slrl-slate);
}

.slrl-seg button:last-child[aria-pressed="true"] {
  background: var(--slrl-green-tint);
  color: var(--slrl-green-ink);
  box-shadow: inset 0 0 0 1px var(--slrl-green);
}

.slrl-steps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0;
  padding: 0 16px 12px;
  list-style: none;
}

.slrl-steps li {
  margin: 0;
  padding: 0;
}

.slrl-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  padding: 4px 11px 4px 5px;
  border: 1px solid var(--slrl-rule);
  border-radius: 999px;
  background: var(--slrl-surface);
  color: var(--slrl-muted);
  font: inherit;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.2s ease, color 0.2s ease;
}

.slrl-step:hover {
  border-color: var(--slrl-green);
  color: var(--slrl-text);
}

.slrl-step__num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: var(--slrl-panel);
  color: var(--slrl-muted);
  font-size: 11.5px;
  font-weight: 700;
}

.slrl-step.is-done .slrl-step__num {
  background: var(--slrl-green-tint);
  color: var(--slrl-green-ink);
}

.slrl-step[aria-current="step"] {
  border-color: var(--slrl-green);
  background: var(--slrl-green-tint);
  color: var(--slrl-green-ink);
}

.slrl-step[aria-current="step"] .slrl-step__num {
  background: var(--slrl-green);
  color: #000000;
}

.slrl-part__info:focus-visible,
.slrl-nav:focus-visible,
.slrl-toggle:focus-visible,
.slrl-seg button:focus-visible,
.slrl-step:focus-visible {
  outline: 2px solid var(--slrl-green);
  outline-offset: 2px;
}

.slrl-stage {
  border-top: 1px solid var(--slrl-rule);
  border-bottom: 1px solid var(--slrl-rule);
  background-color: var(--slrl-panel);
  background-image: radial-gradient(var(--slrl-dot) 0.9px, transparent 0.9px);
  background-size: 14px 14px;
  overflow-x: auto;
}

.slrl-svg {
  display: block;
  width: 100%;
  min-width: 600px;
  height: auto;
  font-family: inherit;
}

.slrl-status {
  min-height: 132px;
  padding: 12px 16px 14px;
}

.slrl-status p {
  margin: 0;
  color: var(--slrl-text);
  font-size: 14.5px;
}

.slrl-status p + p {
  margin-top: 8px;
}

.slrl-status code {
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--slrl-panel);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.86em;
}

.slrl-status .slrl-caveat {
  padding: 8px 12px;
  border-left: 3px solid var(--slrl-amber);
  border-radius: 0 8px 8px 0;
  background: var(--slrl-amber-tint);
  font-size: 14px;
}

.slrl-compare {
  width: 100%;
  margin: 12px 0 0;
  border-collapse: collapse;
  font-size: 13.5px;
  line-height: 1.45;
}

.slrl-compare th,
.slrl-compare td {
  padding: 6px 10px;
  border-top: 1px solid var(--slrl-rule);
  text-align: left;
  vertical-align: top;
}

.slrl-compare thead th {
  border-top: 0;
  color: var(--slrl-muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slrl-compare tbody th {
  color: var(--slrl-text);
  font-weight: 600;
}

.slrl-compare td {
  color: var(--slrl-muted);
  transition: background-color 0.25s ease, color 0.25s ease;
}

.slrl-compare td.is-current {
  color: var(--slrl-text);
}

.slrl-compare thead th.is-current:nth-child(2),
.slrl-compare td.is-current:nth-child(2) {
  background: var(--slrl-slate-tint);
}

.slrl-compare thead th.is-current:nth-child(3),
.slrl-compare td.is-current:nth-child(3) {
  background: var(--slrl-green-tint);
}

.slrl-compare thead th.is-current {
  color: var(--slrl-text);
}

.slrl-sr {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}

/* ---- Diagram ---- */

.slrl-zone {
  animation: slrl-in 0.45s ease both;
}

.slrl-zone rect {
  fill: var(--slrl-tone-tint);
  stroke: var(--slrl-tone);
  stroke-width: 1.25;
  transition: x 0.55s ease, y 0.55s ease, width 0.55s ease, height 0.55s ease;
}

.slrl-zone--inner rect {
  fill: none;
  stroke-dasharray: 5 4;
}

.slrl-zone.slrl-tone--neutral rect {
  fill: var(--slrl-surface);
  stroke: var(--slrl-faint);
  stroke-dasharray: 5 4;
}

.slrl-zone__label {
  fill: var(--slrl-tone-ink);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.12em;
}

.slrl-zone.slrl-tone--neutral .slrl-zone__label {
  fill: var(--slrl-muted);
}

.slrl-zone__owner {
  fill: var(--slrl-muted);
  font-size: 11.5px;
  font-style: italic;
}

.slrl-file rect {
  fill: var(--slrl-panel);
  stroke: var(--slrl-rule);
  stroke-width: 1;
}

.slrl-file text {
  fill: var(--slrl-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10.5px;
}

.slrl-node {
  transition: transform 0.55s cubic-bezier(0.2, 0.75, 0.25, 1);
  animation: slrl-in 0.45s ease both;
}

.slrl-node__box {
  fill: var(--slrl-surface);
  stroke: var(--slrl-tone);
  stroke-width: 1.5;
}

.slrl-node.slrl-tone--neutral .slrl-node__box {
  stroke: var(--slrl-faint);
}

.slrl-node__title {
  fill: var(--slrl-text);
  font-size: 14.5px;
  font-weight: 700;
}

.slrl-node__sub {
  fill: var(--slrl-muted);
  font-size: 11.5px;
}

.slrl-weights rect {
  fill: var(--slrl-tone-tint);
  stroke: var(--slrl-tone);
  stroke-width: 1;
}

.slrl-weights text {
  fill: var(--slrl-tone-ink);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10px;
  font-weight: 700;
}

.slrl-chip rect {
  fill: var(--slrl-panel);
  stroke: var(--slrl-faint);
  stroke-width: 1;
}

.slrl-chip text {
  fill: var(--slrl-muted);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.slrl-chip--gpu rect {
  fill: var(--slrl-tone);
  stroke: var(--slrl-tone);
}

.slrl-chip--gpu text {
  fill: #000000;
}

.slrl-num circle {
  fill: var(--slrl-surface);
  stroke: var(--slrl-tone);
  stroke-width: 1.5;
}

.slrl-num text {
  fill: var(--slrl-tone-ink);
  font-size: 12px;
  font-weight: 700;
}

.slrl-wire {
  animation: slrl-in 0.45s ease both;
}

.slrl-wire path {
  fill: none;
  stroke: var(--slrl-tone);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.slrl-wire polygon {
  fill: var(--slrl-tone);
}

.slrl-wire.slrl-tone--neutral path {
  stroke: var(--slrl-faint);
}

.slrl-wire.slrl-tone--neutral polygon {
  fill: var(--slrl-faint);
}

.slrl-wire--dashed path {
  stroke-dasharray: 6 5;
}

.slrl-wire__label {
  fill: var(--slrl-tone-ink);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 11px;
  font-weight: 700;
  paint-order: stroke;
  stroke: var(--slrl-panel);
  stroke-width: 4px;
  stroke-linejoin: round;
  animation: slrl-in 0.45s ease both;
}

.slrl-wire__label--on-slate {
  stroke: var(--slrl-slate-tint);
}

.slrl-wire__label--on-green {
  stroke: var(--slrl-green-tint);
}

.slrl-center {
  fill: var(--slrl-muted);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

@keyframes slrl-in {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@container (max-width: 700px) {
  .slrl-tray {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .slrl-part:nth-child(4n) .slrl-tip {
    right: auto;
    left: 0;
  }

  .slrl-part:nth-child(2n) .slrl-tip {
    right: 0;
    left: auto;
  }
}

@media (max-width: 700px) {
  .slrl-tray {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .slrl-part:nth-child(4n) .slrl-tip {
    right: auto;
    left: 0;
  }

  .slrl-part:nth-child(2n) .slrl-tip {
    right: 0;
    left: auto;
  }
}

@container (max-width: 480px) {
  .slrl-part__icon {
    display: none;
  }
}

@media (max-width: 480px) {
  .slrl-part__icon {
    display: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .slrl-part,
  .slrl-part__icon,
  .slrl-tip,
  .slrl-node,
  .slrl-zone rect,
  .slrl-compare td {
    transition: none;
  }

  .slrl-node,
  .slrl-zone,
  .slrl-wire,
  .slrl-wire__label {
    animation: none;
  }
}
`;

export function SlimeRLLoopStyles() {
  return <style dangerouslySetInnerHTML={{ __html: SLIME_RL_LOOP_CSS }} />;
}

export default SlimeRLLoopStyles;
