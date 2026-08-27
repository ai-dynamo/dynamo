/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Landing page component styles (Home and Community).
 *
 * Styles for WelcomeHero, WhyDynamo, EventsCalendar and CommunityLanding.
 * This block is their hosted-build delivery path. The shared NVIDIA global theme
 * replaces the project `css:` entry and custom footer at publish (#11952), so
 * production ships neither the main.css link nor CustomFooter's mirrored copy.
 *
 * main.css remains the stylesheet for local `fern docs dev` and other builds
 * that do not inject the global theme. Keep site-wide local-preview rules there,
 * but keep landing-page rules and variables in this component so the rendered
 * pages are self-contained. Same pattern as ReferenceStyles.tsx and
 * RecipeStyles.tsx.
 *
 * Injected via dangerouslySetInnerHTML, like RecipeStyles.tsx, not as a text
 * child like ReferenceStyles.tsx. A text child is escaped on render, which
 * turns every `>` child combinator into `&gt;` and silently drops the rule.
 * ReferenceStyles gets away with it because it has no child combinators;
 * this bundle has dozens.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. IMPORT it (ambient use is
 * unsupported -- renders "Unsupported JSX tag"); the @/ prefix resolves to the
 * fern/ root and is rewritten to a relative path at publish time:
 *
 *   import { LandingStyles } from `@/components/LandingStyles`;
 *
 * The backticks above stand in for the double quotes the real page uses, for
 * the reason spelled out in RecipeStyles.tsx: Fern's mdx-components bundler
 * regex-scans this file for imports without skipping comments, and a quoted
 * non-relative specifier makes it shell out to `npx rolldown` on every build.
 *
 * Then place <LandingStyles /> once, right after the imports, on
 * pages/home/index.mdx and pages/community/community.mdx.
 */
const LANDING_CSS = `
/* ===================== Welcome (Home) landing page ===================== */

/* The marker component scopes every override to the Home page. */
article:has(.dynamo-welcome) {
  position: relative;
  isolation: isolate;
  width: 100% !important;
  max-width: 1200px !important;
  margin-inline: auto;
  padding-inline: clamp(1rem, 3vw, 2.5rem);
}

article:has(.dynamo-welcome)::before {
  content: "";
  position: absolute;
  top: 0;
  left: 50%;
  width: min(100vw, 1440px);
  height: 760px;
  transform: translateX(-50%);
  pointer-events: none;
  z-index: -1;
  background:
    linear-gradient(rgba(118, 185, 0, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(118, 185, 0, 0.035) 1px, transparent 1px),
    radial-gradient(
      ellipse 82% 72% at 50% 8%,
      rgba(118, 185, 0, 0.24),
      transparent 76%
    );
  background-size:
    44px 44px,
    44px 44px,
    auto;
  /* Fade the grid and glow before the page-width boundary so the backdrop
     blends into the surrounding canvas instead of ending at a hard edge. */
  -webkit-mask-image:
    linear-gradient(to right, transparent 0%, black 12%, black 88%, transparent 100%),
    linear-gradient(to bottom, black 0%, black 62%, transparent 100%);
  -webkit-mask-composite: source-in;
  mask-image:
    linear-gradient(to right, transparent 0%, black 12%, black 88%, transparent 100%),
    linear-gradient(to bottom, black 0%, black 62%, transparent 100%);
  mask-composite: intersect;
}

.dark article:has(.dynamo-welcome)::before {
  background:
    linear-gradient(rgba(118, 185, 0, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(118, 185, 0, 0.055) 1px, transparent 1px),
    radial-gradient(
      ellipse 82% 72% at 50% 8%,
      rgba(118, 185, 0, 0.3),
      transparent 76%
    );
  background-size:
    44px 44px,
    44px 44px,
    auto;
}

/* Turn Fern's generated title and subtitle into the first half of the hero. */
article:has(.dynamo-welcome) > header {
  margin: 0;
  /* The extra top padding reserves the space the mark below occupies, since
     the mark is taken out of the flow to sit above the heading. */
  padding: calc(clamp(3rem, 6vh, 4.5rem) + 92px + 1.45rem) 1rem 0;
  text-align: center;
}

/* The Dynamo mark, rendered as an <img> from the page MDX. It cannot be a
   background-image here: Fern rewrites asset paths only in MDX and docs.yml,
   never inside a <style> string, so a url() reaches the browser verbatim and
   404s. That puts the mark in the prose, below the heading, so pull it back up
   over the heading; the header padding above holds its place. */
article:has(.dynamo-welcome) .dynamo-welcome__mark {
  position: absolute;
  top: clamp(3rem, 6vh, 4.5rem);
  left: 50%;
  transform: translateX(-50%);
  display: block;
  width: 92px;
  height: 92px;
  margin: 0;
  border: 1px solid rgba(118, 185, 0, 0.4);
  border-radius: 24px;
  background-color: #f3fbdc;
  object-fit: cover;
  box-shadow:
    0 18px 42px rgba(54, 86, 0, 0.24),
    inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

.dark article:has(.dynamo-welcome) .dynamo-welcome__mark {
  border-color: rgba(118, 185, 0, 0.4);
  background-color: #0c0d0b;
  box-shadow: 0 18px 46px rgba(0, 0, 0, 0.42);
}

article:has(.dynamo-welcome) > header .fern-breadcrumb {
  display: none;
}

article:has(.dynamo-welcome) > header .fern-page-heading {
  display: block;
  width: 100%;
  font-size: clamp(3.75rem, 8vw, 6.75rem);
  font-weight: 600;
  line-height: 0.9;
  letter-spacing: -0.055em;
}

article:has(.dynamo-welcome) > header div:has(.fern-page-heading) {
  width: 100%;
  justify-content: center;
}

article:has(.dynamo-welcome) > header .fern-page-subtitle {
  max-width: 900px;
  margin: clamp(1.5rem, 3vw, 2.25rem) auto 0;
  font-size: clamp(1.05rem, 2vw, 1.35rem);
  line-height: 1.55;
  color: var(--grayscale-a11);
  white-space: nowrap;
}

article:has(.dynamo-welcome) > header .fern-page-subtitle p {
  margin: 0;
}

.dynamo-welcome {
  position: relative;
  z-index: 1;
}

.dynamo-welcome__intro {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.75rem 1rem 0;
  text-align: center;
}

.dynamo-welcome__statement {
  min-height: 3.5rem;
  font-family:
    RobotoMono, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: clamp(1rem, 2vw, 1.25rem);
  color: var(--grayscale-a11);
}

.dynamo-welcome__statement p {
  margin: 0;
}

.dynamo-welcome__typed {
  color: #579000;
  font-weight: 600;
}

.dark .dynamo-welcome__typed {
  color: #8fd120;
}

.dynamo-welcome__cursor {
  display: inline-block;
  width: 0.55em;
  height: 1.05em;
  margin-left: 0.16em;
  vertical-align: -0.16em;
  background: #76b900;
  animation: dynamo-cursor-blink 0.9s steps(1, end) infinite;
}



@keyframes dynamo-cursor-blink {
  0%,
  45% {
    opacity: 1;
  }
  46%,
  100% {
    opacity: 0;
  }
}

.dynamo-welcome__sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.dynamo-welcome__actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.75rem;
  margin-top: 1.35rem;
}

.dynamo-welcome__cta {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
  min-height: 3.15rem;
  padding: 0.82rem 1.2rem 0.82rem 1.4rem;
  border: 1px solid #68a400;
  border-radius: 999px;
  background: #76b900;
  color: #0b0f08 !important;
  font-size: 0.98rem;
  font-weight: 680;
  line-height: 1;
  letter-spacing: -0.01em;
  text-decoration: none !important;
  box-shadow: 0 7px 18px rgba(54, 86, 0, 0.16);
  transition:
    transform 160ms ease,
    border-color 160ms ease,
    background 160ms ease,
    box-shadow 160ms ease;
}

.dynamo-welcome__cta:hover {
  transform: translateY(-1px);
  border-color: #5e9500;
  background: #69a700;
  box-shadow: 0 9px 22px rgba(54, 86, 0, 0.22);
}

.dynamo-welcome__cta:active {
  transform: translateY(0);
}

.dynamo-welcome__cta:focus-visible {
  outline: 3px solid rgba(118, 185, 0, 0.3);
  outline-offset: 4px;
}

.dynamo-welcome__cta svg {
  width: 1rem;
  height: 1rem;
  fill: none;
  stroke: #0b0f08;
  stroke-width: 2.35;
  stroke-linecap: round;
  stroke-linejoin: round;
  transition: transform 160ms ease;
}

.dynamo-welcome__cta--secondary {
  border-color: color-mix(in srgb, var(--grayscale-a12) 16%, transparent);
  background: var(--grayscale-a1);
  color: var(--grayscale-a12) !important;
  box-shadow: none;
}

.dynamo-welcome__cta--secondary:hover {
  border-color: rgba(118, 185, 0, 0.55);
  background: color-mix(in srgb, var(--grayscale-a2) 84%, rgba(118, 185, 0, 0.12));
}

.dynamo-welcome__cta--secondary svg {
  stroke: currentColor;
}

.dynamo-welcome__cta:hover svg {
  transform: translateX(2px);
}

/* The demo reveals as the visitor leaves the opening hero. */
.dynamo-welcome__terminal {
  position: relative;
  margin: clamp(7rem, 16vh, 10rem) auto 4.5rem;
}

/* A broad ambient caption appears only after the opening hero is left behind. */
.dynamo-welcome__demo-reveal {
  position: absolute;
  z-index: 2;
  top: clamp(-6.8rem, -10vh, -5.2rem);
  left: 50%;
  width: min(100%, 1040px);
  margin: 0;
  padding: 1rem clamp(1.25rem, 4vw, 3.5rem) 0.65rem;
  isolation: isolate;
  transform: translateX(-50%);
}

.dynamo-welcome__demo-reveal::before {
  content: "";
  position: absolute;
  z-index: -1;
  inset: -1.5rem 10% -1.25rem -2%;
  background: radial-gradient(
    ellipse at 22% 58%,
    rgba(118, 185, 0, 0.17),
    rgba(118, 185, 0, 0.055) 35%,
    transparent 70%
  );
  filter: blur(18px);
  opacity: 0;
  transform: scaleX(0.55);
  transform-origin: left;
  transition:
    opacity 520ms ease,
    transform 760ms cubic-bezier(0.22, 1, 0.36, 1);
}

.dynamo-welcome__terminal[data-visible="true"] .dynamo-welcome__demo-reveal::before {
  opacity: 1;
  transform: scaleX(1);
}

.dynamo-welcome__demo-reveal p,
.dynamo-welcome__demo-reveal h2 {
  opacity: 0;
  transform: translateX(-28px);
  filter: blur(5px);
  transition:
    opacity 240ms ease,
    transform 360ms cubic-bezier(0.22, 1, 0.36, 1),
    filter 300ms ease;
}

.dynamo-welcome__terminal[data-visible="true"] .dynamo-welcome__demo-reveal p,
.dynamo-welcome__terminal[data-visible="true"] .dynamo-welcome__demo-reveal h2 {
  opacity: 1;
  transform: none;
  filter: none;
  transition-delay: 180ms;
}

.dynamo-welcome__demo-reveal p {
  margin: 0 0 0.42rem;
  color: #579000;
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.68rem;
  font-weight: 750;
  letter-spacing: 0.11em;
  line-height: 1;
  text-transform: uppercase;
}

.dark .dynamo-welcome__demo-reveal p {
  color: #a4e344;
}

.dynamo-welcome__demo-reveal h2 {
  margin: 0;
  color: var(--grayscale-a12);
  font-size: clamp(1.8rem, 4vw, 3.15rem);
  font-weight: 610;
  letter-spacing: -0.05em;
  line-height: 1;
}

.dynamo-welcome__terminal-stage {
  position: relative;
  max-width: 1110px;
  margin: 0 auto;
  padding: clamp(1.35rem, 4.2vw, 3.5rem);
  overflow: hidden;
  border: 1px solid rgba(118, 185, 0, 0.48);
  border-radius: 30px;
  background:
    linear-gradient(rgba(118, 185, 0, 0.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(118, 185, 0, 0.055) 1px, transparent 1px),
    radial-gradient(
      ellipse 72% 105% at 50% 0%,
      rgba(118, 185, 0, 0.24),
      rgba(118, 185, 0, 0.055) 48%,
      transparent 76%
    ),
    linear-gradient(145deg, rgba(238, 248, 223, 0.16), rgba(220, 239, 194, 0.055));
  background-size: 34px 34px, 34px 34px, auto, auto;
  -webkit-backdrop-filter: blur(28px) saturate(135%);
  backdrop-filter: blur(28px) saturate(135%);
  box-shadow:
    inset 0 1px 0 rgba(215, 255, 169, 0.14),
    0 24px 70px rgba(0, 0, 0, 0.08),
    0 0 54px rgba(118, 185, 0, 0.08);
}

.dark .dynamo-welcome__terminal-stage {
  border-color: rgba(118, 185, 0, 0.5);
  background:
    linear-gradient(rgba(118, 185, 0, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(118, 185, 0, 0.06) 1px, transparent 1px),
    radial-gradient(
      ellipse 72% 105% at 50% 0%,
      rgba(118, 185, 0, 0.25),
      rgba(35, 61, 12, 0.1) 48%,
      transparent 76%
    ),
    linear-gradient(145deg, rgba(10, 18, 11, 0.38), rgba(2, 7, 3, 0.22));
  background-size: 34px 34px, 34px 34px, auto, auto;
  box-shadow:
    inset 0 1px 0 rgba(187, 244, 112, 0.08),
    0 30px 90px rgba(0, 0, 0, 0.3),
    0 0 60px rgba(118, 185, 0, 0.1);
}

.dynamo-welcome__terminal-stage::before {
  content: "";
  position: absolute;
  inset: -35% 12% auto;
  height: 70%;
  border-radius: 50%;
  background: radial-gradient(
    ellipse,
    rgba(118, 185, 0, 0.18),
    transparent 68%
  );
  filter: blur(28px);
  pointer-events: none;
}

.dynamo-welcome__terminal .dynamo-terminal-demo {
  position: relative;
  width: 100%;
  max-width: 1040px;
  margin: 0 auto;
  border: 1px solid rgba(142, 213, 43, 0.34);
  border-radius: 18px;
  background:
    radial-gradient(circle at 18% -5%, rgba(118, 185, 0, 0.24), transparent 46%),
    radial-gradient(circle at 82% 110%, rgba(42, 78, 18, 0.18), transparent 44%),
    linear-gradient(145deg, rgba(8, 18, 10, 0.9), rgba(0, 5, 2, 0.96));
  -webkit-backdrop-filter: blur(34px) saturate(145%);
  backdrop-filter: blur(34px) saturate(145%);
  box-shadow:
    0 0 0 1px rgba(219, 255, 180, 0.035) inset,
    0 24px 70px rgba(0, 0, 0, 0.32),
    0 0 60px rgba(118, 185, 0, 0.07);
}

@media (prefers-reduced-motion: reduce) {

  .dynamo-welcome__demo-reveal::before,
  .dynamo-welcome__demo-reveal p,
  .dynamo-welcome__demo-reveal h2,
  .dynamo-welcome__terminal-stage {
    transition: none;
  }
}

/* Scroll-driven feature story: copy advances on the left while one large visual stays pinned. */
.dynamo-story {
  display: grid;
  grid-template-columns: minmax(280px, 0.8fr) minmax(520px, 1.55fr);
  gap: clamp(2.5rem, 6vw, 6.5rem);
  margin: 3rem 0 7rem;
}

.dynamo-story__steps {
  min-width: 0;
}

.dynamo-story__step {
  display: flex;
  min-height: 72vh;
  align-items: center;
  margin: 0;
  opacity: 0.42;
  transition: opacity 260ms ease;
}

.dynamo-story__step[data-active="true"] {
  opacity: 1;
}

.dynamo-story__step-copy {
  padding: 1.5rem 0 1.5rem 1.2rem;
  border-left: 2px solid var(--grayscale-a4);
  transition:
    border-color 260ms ease,
    transform 260ms ease;
}

.dynamo-story__step[data-active="true"] .dynamo-story__step-copy {
  border-color: #76b900;
  transform: translateX(0.35rem);
}

.dynamo-story__eyebrow {
  margin: 0 0 0.9rem !important;
  color: #579000 !important;
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.72rem !important;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.dark .dynamo-story__eyebrow {
  color: #8fd120 !important;
}

.dynamo-story__step h3 {
  margin: 0 0 1rem;
  font-size: clamp(2rem, 3.4vw, 3.25rem);
  line-height: 1.03;
  letter-spacing: -0.045em;
}

.dynamo-story__step-copy > p:last-child {
  margin: 0;
  color: var(--grayscale-a11);
  font-size: 1.05rem;
  line-height: 1.65;
}

.dynamo-story__stage {
  position: sticky;
  top: 8rem;
  align-self: start;
  height: min(680px, calc(100vh - 10rem));
  min-height: 520px;
  overflow: hidden;
  border: 1px solid rgba(127, 127, 127, 0.2);
  border-radius: 30px;
  background:
    radial-gradient(circle at 18% 12%, rgba(118, 185, 0, 0.2), transparent 32%),
    linear-gradient(
      145deg,
      rgba(245, 249, 238, 0.96),
      rgba(235, 240, 229, 0.92)
    );
  box-shadow: 0 30px 80px rgba(23, 42, 0, 0.16);
}

.dark .dynamo-story__stage {
  background:
    radial-gradient(circle at 18% 12%, rgba(118, 185, 0, 0.2), transparent 32%),
    linear-gradient(145deg, rgba(16, 20, 12, 0.98), rgba(5, 7, 4, 0.98));
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.4);
}

.dynamo-story__stage::before {
  content: "DYNAMO / SYSTEM VIEW";
  position: absolute;
  top: 1.4rem;
  left: 1.6rem;
  z-index: 5;
  color: var(--grayscale-a9);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
}

.dynamo-story__stage-panel {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: clamp(2rem, 5vw, 4.5rem);
  opacity: 0;
  transform: translateY(24px);
  pointer-events: none;
  transition:
    opacity 320ms ease,
    transform 420ms ease;
}

.dynamo-story__stage-panel[data-active="true"] {
  opacity: 1;
  transform: none;
}

.dynamo-story__progress {
  position: absolute;
  right: 1.5rem;
  bottom: 1.5rem;
  left: 1.5rem;
  z-index: 5;
  display: flex;
  gap: 0.45rem;
}

.dynamo-story__progress span {
  flex: 1;
  height: 3px;
  overflow: hidden;
  border-radius: 99px;
  background: var(--grayscale-a4);
}

.dynamo-story__progress span[data-active="true"] {
  background: #76b900;
}

.dynamo-story__mobile-graphic {
  display: none;
}

.dynamo-story-graphic {
  position: relative;
  width: min(100%, 620px);
  min-height: 390px;
}

.dynamo-story-windowbar {
  position: relative;
  display: flex;
  height: 3rem;
  flex: none;
  align-items: center;
  gap: 0.48rem;
  padding: 0 1rem;
  border-bottom: 1px solid #d7d8d7;
  background: #f3f4f3;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.dark .dynamo-story-windowbar {
  border-bottom-color: #171817;
  background: #2b2c2b;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.dynamo-story-windowbar > span {
  width: 0.72rem;
  min-width: 0.72rem;
  max-width: 0.72rem;
  height: 0.72rem;
  min-height: 0.72rem;
  max-height: 0.72rem;
  flex: 0 0 0.72rem;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 50%;
  background: #ff5f57;
  box-sizing: border-box;
  box-shadow: inset 0 0 0 0.5px rgba(0, 0, 0, 0.08);
  animation: none;
  transform: none;
  transition: none;
}

.dynamo-story-windowbar > span:nth-child(2) {
  background: #febc2e;
}

.dynamo-story-windowbar > span:nth-child(3) {
  background: #28c840;
}

.dynamo-story-window-label {
  position: absolute;
  left: 50%;
  margin: 0;
  transform: translateX(-50%);
  color: #555755;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1;
  white-space: nowrap;
}

.dark .dynamo-story-window-label {
  color: #d4d5d4;
}

.dynamo-story-graphic--performance,
.dynamo-story-graphic--engines {
  display: flex;
  overflow: hidden;
  flex-direction: column;
  border: 1px solid #d7d8d7;
  border-radius: 18px;
  background: linear-gradient(#f3f4f3 0 3rem, #0d1117 3rem);
  color: #f5f5f5;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.32);
}

.dark .dynamo-story-graphic--performance,
.dark .dynamo-story-graphic--engines {
  border-color: rgba(255, 255, 255, 0.12);
  background: linear-gradient(#2b2c2b 0 3rem, #0d1117 3rem);
}

.dynamo-story-metrics {
  display: grid;
  gap: 1.15rem;
  padding: 2.2rem 1.8rem 1.5rem;
}

.dynamo-story-metrics > div {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 0.55rem 1rem;
  align-items: center;
}

.dynamo-story-metrics span {
  color: #c9d1d9;
  font-size: 0.85rem;
}

.dynamo-story-metrics strong {
  color: #8fd120;
  font-size: 1.1rem;
}

.dynamo-story-metrics i {
  grid-column: 1 / -1;
  height: 0.62rem;
  overflow: hidden;
  border-radius: 99px;
  background: #252b33;
}

.dynamo-story-metrics i::after {
  content: "";
  display: block;
  width: var(--metric);
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #4e7f00, #8fd120);
}

.dynamo-story-flow {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 0.75rem;
  align-items: center;
  margin: 0 1.8rem 1.8rem;
  padding: 1rem;
  border: 1px solid #30363d;
  border-radius: 12px;
  color: #c9d1d9;
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.72rem;
  text-align: center;
}

.dynamo-story-flow b {
  padding: 0.6rem 0.85rem;
  border-radius: 8px;
  background: rgba(118, 185, 0, 0.18);
  color: #a8e34b;
}

.dynamo-story-graphic--engines pre {
  min-height: 245px;
  margin: 0;
  padding: 2rem 1.8rem;
  background: transparent;
  color: #f0f3f6;
  font-size: clamp(0.78rem, 1.4vw, 0.95rem);
  line-height: 1.75;
}

.dynamo-story-graphic--engines .prompt,
.dynamo-story-graphic--engines .success {
  color: #8fd120;
}

.dynamo-story-graphic--engines .muted {
  color: #8b949e;
}

.dynamo-story-graphic--engines b {
  color: #67b7ff;
}

.dynamo-story-engine-tabs {
  display: flex;
  gap: 0.6rem;
  padding: 0 1.8rem 1.8rem;
}

.dynamo-story-engine-tabs span,
.dynamo-story-hardware span {
  padding: 0.55rem 0.75rem;
  border: 1px solid rgba(118, 185, 0, 0.34);
  border-radius: 999px;
  background: rgba(118, 185, 0, 0.12);
  color: #8fd120;
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.7rem;
}

.dynamo-story-graphic--infrastructure {
  display: grid;
  place-items: center;
}

.dynamo-story-orbit {
  position: relative;
  width: min(100%, 430px);
  aspect-ratio: 1;
  border: 1px dashed rgba(118, 185, 0, 0.38);
  border-radius: 50%;
}

.dynamo-story-orbit::before {
  content: "";
  position: absolute;
  inset: 18%;
  border: 1px solid rgba(118, 185, 0, 0.24);
  border-radius: 50%;
}

.dynamo-story-core,
.dynamo-story-orbit .node {
  position: absolute;
  display: grid;
  place-items: center;
  border: 1px solid rgba(118, 185, 0, 0.4);
  background: rgba(12, 18, 8, 0.94);
  color: #d8f4aa;
  box-shadow: 0 14px 30px rgba(18, 32, 0, 0.25);
  font-family: RobotoMono, ui-monospace, monospace;
}

/* Orbit node positions. The modifiers are unprefixed in the markup
   (class "node node--k8s"), so the dynamo-* extraction missed them. Left in
   main.css the production theme drops them and the three labels collapse to
   the container's static position. No backticks in this comment: it lives
   inside a template literal. */
.node--k8s {
  top: 3%;
  left: 50%;
  transform: translateX(-50%);
}
.node--slurm {
  right: -2%;
  bottom: 19%;
}
.node--local {
  bottom: 19%;
  left: -2%;
}

.dynamo-story-core {
  inset: 50% auto auto 50%;
  width: 7.2rem;
  height: 7.2rem;
  transform: translate(-50%, -50%);
  border-radius: 28px;
  color: white;
  font-size: 1.05rem;
  font-weight: 700;
}

.dynamo-story-orbit .node {
  min-width: 6.7rem;
  min-height: 3.1rem;
  padding: 0.5rem;
  border-radius: 14px;
  font-size: 0.72rem;
}

.dynamo-story-hardware {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.55rem;
}

.dynamo-story-graphic--modular {
  display: grid;
  align-content: center;
  gap: 2rem;
}

.dynamo-story-stack {
  display: grid;
  gap: 0.75rem;
  perspective: 700px;
}

.dynamo-story-stack span {
  display: flex;
  min-height: 3.6rem;
  align-items: center;
  padding: 0 1.2rem;
  border: 1px solid rgba(118, 185, 0, 0.34);
  border-radius: 12px;
  background: linear-gradient(
    90deg,
    rgba(23, 36, 12, 0.96),
    rgba(47, 70, 25, 0.9)
  );
  color: #e9f8d2;
  box-shadow: 0 10px 24px rgba(17, 30, 5, 0.18);
  font-family: RobotoMono, ui-monospace, monospace;
  font-size: 0.85rem;
  transform: rotateX(2deg) translateX(calc(var(--stack-index, 0) * 0.5rem));
}

.dynamo-story-stack span:nth-child(2) {
  margin-inline: 1rem;
}

.dynamo-story-stack span:nth-child(3) {
  margin-inline: 2rem;
}

.dynamo-story-stack span:nth-child(4) {
  margin-inline: 3rem;
}

.dynamo-story-stack span:nth-child(5) {
  margin-inline: 4rem;
}

.dynamo-story-stack-caption {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  color: var(--grayscale-a10);
  font-size: 0.75rem;
  text-transform: uppercase;
}

.dynamo-story-stack-caption i {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--grayscale-a5), #76b900);
}

/* macOS-style public community calendar preview. */
.dynamo-calendar {
  width: min(100%, 1040px);
  margin: 8rem auto 5rem;
  overflow: hidden;
  border: 1px solid rgba(127, 127, 127, 0.2);
  border-radius: 28px;
  background:
    radial-gradient(circle at 12% 0%, rgba(255, 255, 255, 0.82), transparent 34%),
    linear-gradient(145deg, rgba(247, 249, 244, 0.84), rgba(226, 232, 220, 0.72));
  box-shadow:
    0 32px 90px rgba(24, 39, 10, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(28px) saturate(1.35);
}

.dark .dynamo-calendar {
  border-color: rgba(255, 255, 255, 0.12);
  background:
    radial-gradient(circle at 12% 0%, rgba(255, 255, 255, 0.1), transparent 34%),
    linear-gradient(145deg, rgba(28, 32, 24, 0.88), rgba(8, 10, 7, 0.84));
  box-shadow:
    0 32px 90px rgba(0, 0, 0, 0.48),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.dynamo-calendar__chrome {
  position: relative;
  display: flex;
  height: 3.35rem;
  align-items: center;
  gap: 0.5rem;
  padding: 0 1rem;
  border-bottom: 1px solid #d7d8d7;
  background: #f3f4f3;
  color: #555755;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.dark .dynamo-calendar__chrome {
  border-bottom-color: #171817;
  background: #2b2c2b;
  color: #d4d5d4;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.dynamo-calendar__chrome > span {
  width: 0.72rem;
  height: 0.72rem;
  border-radius: 50%;
  background: #ff5f57;
}

.dynamo-calendar__chrome > span:nth-child(2) { background: #febc2e; }

.dynamo-calendar__chrome > span:nth-child(3) { background: #28c840; }

.dynamo-calendar__chrome p {
  position: absolute;
  left: 50%;
  margin: 0;
  color: inherit;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
  transform: translateX(-50%);
}

.dynamo-calendar__chrome a {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-left: auto;
  padding: 0.45rem 0.7rem;
  border: 1px solid rgba(31, 35, 31, 0.18);
  border-radius: 8px;
  color: #555755 !important;
  font-size: 0.7rem;
  text-decoration: none !important;
}

.dark .dynamo-calendar__chrome a {
  border-color: rgba(255, 255, 255, 0.18);
  color: #d4d5d4 !important;
}

.dynamo-calendar__chrome a svg {
  width: 0.72rem;
  height: 0.72rem;
  flex: 0 0 auto;
  fill: currentColor;
}

.dynamo-calendar__chrome a:hover,
.dynamo-calendar__chrome a:focus-visible {
  border-color: rgba(143, 209, 32, 0.7);
  background: rgba(118, 185, 0, 0.18);
}

.dynamo-calendar__body {
  display: grid;
  grid-template-columns: 270px minmax(0, 1fr);
  min-height: 560px;
}

.dynamo-calendar__sidebar {
  padding: 2rem 1.5rem;
  border-right: 1px solid rgba(127, 127, 127, 0.18);
  background: rgba(255, 255, 255, 0.28);
}

.dark .dynamo-calendar__sidebar {
  background: rgba(255, 255, 255, 0.035);
}

.dynamo-calendar__month-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 1.2rem;
}

.dynamo-calendar__month-heading strong {
  font-size: 1.1rem;
}

.dynamo-calendar__month-heading span {
  color: var(--grayscale-a9);
  font-size: 0.8rem;
}

.dynamo-calendar__weekdays,
.dynamo-calendar__month-grid {
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 0.32rem;
  text-align: center;
}

.dynamo-calendar__weekdays {
  margin-bottom: 0.35rem;
  color: var(--grayscale-a9);
  font-size: 0.66rem;
  font-weight: 700;
}

.dynamo-calendar__month-grid > span {
  display: grid;
  aspect-ratio: 1;
  place-items: center;
  border-radius: 50%;
  color: var(--grayscale-a11);
  font-size: 0.75rem;
}

.dynamo-calendar__month-grid > .is-selected {
  background: #ff3b30;
  color: white;
  font-weight: 750;
  box-shadow: 0 5px 14px rgba(255, 59, 48, 0.28);
}

/* Days with something on the community calendar. The highlight marks today,
   so the dot is what carries event information in the grid. */
.dynamo-calendar__month-grid > .has-event {
  position: relative;
  color: var(--grayscale-a12);
  font-weight: 700;
}

.dynamo-calendar__month-grid > .has-event::after {
  content: "";
  position: absolute;
  bottom: 0.1rem;
  left: 50%;
  width: 3px;
  height: 3px;
  transform: translateX(-50%);
  border-radius: 50%;
  background: #76b900;
}

/* Today and an event on the same cell: the dot goes white so it stays legible
   against the red fill. */
.dynamo-calendar__month-grid > .has-event.is-selected {
  color: white;
}

.dynamo-calendar__month-grid > .has-event.is-selected::after {
  background: white;
}

.dynamo-calendar__source {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-top: 2rem;
  color: var(--grayscale-a10);
  font-size: 0.76rem;
}

.dynamo-calendar__source-dot {
  width: 0.68rem;
  height: 0.68rem;
  border-radius: 50%;
  background: #76b900;
  box-shadow: 0 0 0 4px rgba(118, 185, 0, 0.12);
}

.dynamo-calendar__agenda {
  display: flex;
  min-width: 0;
  flex-direction: column;
  padding: clamp(2rem, 5vw, 3.5rem);
}

.dynamo-calendar__intro {
  margin-bottom: 2.2rem;
}

.dynamo-calendar__intro > p {
  margin: 0 0 0.6rem;
  color: #579000;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.dark .dynamo-calendar__intro > p {
  color: #8fd120;
}

.dynamo-calendar__intro h2 {
  margin: 0 0 0.7rem;
  font-size: clamp(2.3rem, 5vw, 4.4rem);
  line-height: 0.96;
  letter-spacing: -0.055em;
}

.dynamo-calendar__intro > span {
  color: var(--grayscale-a10);
  font-size: 0.92rem;
}

.dynamo-calendar__events {
  display: grid;
  gap: 0.85rem;
}

.dynamo-calendar__event {
  display: grid;
  grid-template-columns: 3.4rem minmax(0, 1fr) 2.25rem;
  gap: 1rem;
  align-items: center;
  margin: 0;
  padding: 1rem;
  border: 1px solid rgba(127, 127, 127, 0.18);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.46);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.66);
}

.dark .dynamo-calendar__event {
  border-color: rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.045);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

.dynamo-calendar__event-date {
  display: flex;
  min-height: 3.7rem;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  background: #ff3b30;
  color: white;
  box-shadow: 0 8px 20px rgba(255, 59, 48, 0.24);
}

.dynamo-calendar__event-date span {
  font-size: 0.62rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.dynamo-calendar__event-date strong {
  font-size: 1.35rem;
  line-height: 1.05;
}

.dynamo-calendar__event-copy {
  min-width: 0;
}

.dynamo-calendar__event-copy > p {
  margin: 0 0 0.2rem;
  color: var(--grayscale-a9);
  font-size: 0.67rem;
  font-weight: 650;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.dynamo-calendar__event-copy h3 {
  margin: 0 0 0.45rem;
  font-size: 1.05rem;
  line-height: 1.25;
}

.dynamo-calendar__event-copy h3 a {
  color: inherit !important;
  text-decoration: none !important;
}

.dynamo-calendar__event-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  color: var(--grayscale-a10);
  font-size: 0.74rem;
}

.dynamo-calendar__event-meta > span,
.dynamo-calendar__location {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}

.dynamo-calendar__location {
  color: inherit !important;
  text-decoration: none !important;
}

.dynamo-calendar__event-action {
  display: grid;
  width: 2.25rem;
  height: 2.25rem;
  place-items: center;
  border: 1px solid rgba(127, 127, 127, 0.22);
  border-radius: 50%;
  color: var(--grayscale-a11) !important;
  text-decoration: none !important;
}

.dynamo-calendar__event-action:hover,
.dynamo-calendar__event-action:focus-visible {
  border-color: #76b900;
  background: rgba(118, 185, 0, 0.12);
  color: #579000 !important;
}

.dynamo-calendar__past {
  margin-top: auto;
  padding-top: 1.5rem;
  color: var(--grayscale-a10);
  font-size: 0.78rem;
}

.dynamo-calendar__past summary {
  cursor: pointer;
  font-weight: 650;
}

.dynamo-calendar__past > div {
  display: grid;
  gap: 0.6rem;
  padding-top: 0.8rem;
}

.dynamo-calendar__past a {
  display: grid;
  grid-template-columns: 9rem 1fr;
  gap: 1rem;
  color: var(--grayscale-a11) !important;
  text-decoration: none !important;
}

.dynamo-calendar__past a span {
  color: var(--grayscale-a9);
}

.dynamo-calendar__empty {
  display: grid;
  min-height: 180px;
  place-items: center;
  align-content: center;
  gap: 0.7rem;
  border: 1px dashed var(--grayscale-a5);
  border-radius: 16px;
  color: var(--grayscale-a10);
  text-align: center;
}

.dynamo-calendar__empty-icon {
  display: grid;
  width: 2.8rem;
  height: 2.8rem;
  place-items: center;
  border-radius: 10px;
  background: #ff3b30;
  color: white;
  font-size: 0.7rem;
  font-weight: 800;
}

.dynamo-calendar__empty p {
  margin: 0;
}

@media (max-width: 960px) {

  .dynamo-story {
    display: block;
    margin-top: 2rem;
  }

  .dynamo-story__stage {
    display: none;
  }

  .dynamo-story__step {
    display: block;
    min-height: 0;
    margin-bottom: 4.5rem;
    opacity: 1;
  }

  .dynamo-story__step-copy,
  .dynamo-story__step[data-active="true"] .dynamo-story__step-copy {
    max-width: 680px;
    margin-bottom: 1.5rem;
    border-color: #76b900;
    transform: none;
  }

  .dynamo-story__mobile-graphic {
    display: grid;
    min-height: 480px;
    place-items: center;
    padding: clamp(1.25rem, 5vw, 3rem);
    overflow: hidden;
    border: 1px solid rgba(127, 127, 127, 0.2);
    border-radius: 24px;
    background:
      radial-gradient(
        circle at 18% 12%,
        rgba(118, 185, 0, 0.18),
        transparent 32%
      ),
      var(--grayscale-a2);
  }
}

@media (max-width: 640px) {

  article:has(.dynamo-welcome) {
    padding-inline: 0.75rem;
  }

  article:has(.dynamo-welcome) > header {
    padding-top: calc(2.75rem + 60px + 1rem);
  }

  article:has(.dynamo-welcome) .dynamo-welcome__mark {
    top: 2.75rem;
    width: 60px;
    height: 60px;
    border-radius: 14px;
  }

  article:has(.dynamo-welcome) > header .fern-page-heading {
    font-size: clamp(3.5rem, 18vw, 4.75rem);
  }

  article:has(.dynamo-welcome) > header .fern-page-subtitle {
    max-width: 340px;
    white-space: normal;
  }

  .dynamo-welcome__terminal {
    margin-top: 6.5rem;
  }

  .dynamo-welcome__demo-reveal {
    top: -5.25rem;
    width: 100%;
    margin: 0;
    padding: 0.5rem 0.55rem 0.2rem;
  }

  .dynamo-welcome__demo-reveal h2 {
    font-size: 1.7rem;
  }

  .dynamo-welcome__terminal-stage {
    padding: 0.6rem;
    border-radius: 21px;
  }

  .dynamo-welcome__statement {
    width: 100%;
    min-height: 6.5rem;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .dynamo-story__step {
    margin-bottom: 3.5rem;
  }

  .dynamo-story__step h3 {
    font-size: 2.15rem;
  }

  .dynamo-story__step-copy > p:last-child {
    font-size: 0.98rem;
  }

  .dynamo-story__mobile-graphic {
    min-height: 390px;
    padding: 1rem;
    border-radius: 20px;
  }

  .dynamo-story-graphic {
    min-height: 330px;
  }

  .dynamo-story-orbit {
    width: 300px;
  }

  .dynamo-story-core {
    width: 5.8rem;
    height: 5.8rem;
    border-radius: 22px;
  }

  .dynamo-story-orbit .node {
    min-width: 5.7rem;
    min-height: 2.7rem;
    font-size: 0.64rem;
  }

  .dynamo-story-stack span:nth-child(n) {
    margin-inline: 0;
  }
}

@media (max-width: 760px) {

  .dynamo-calendar {
    margin-top: 5rem;
    border-radius: 20px;
  }

  .dynamo-calendar__chrome p {
    display: none;
  }

  .dynamo-calendar__chrome a {
    font-size: 0;
  }

  .dynamo-calendar__chrome a svg {
    width: 0.75rem;
    height: 0.75rem;
  }

  .dynamo-calendar__body {
    display: block;
  }

  .dynamo-calendar__sidebar {
    border-right: 0;
    border-bottom: 1px solid rgba(127, 127, 127, 0.18);
  }

  .dynamo-calendar__agenda {
    padding: 2rem 1.25rem;
  }

  .dynamo-calendar__intro h2 {
    font-size: 3rem;
  }

  .dynamo-calendar__event {
    grid-template-columns: 3.2rem minmax(0, 1fr);
  }

  .dynamo-calendar__event-action {
    display: none;
  }

  .dynamo-calendar__past a {
    display: block;
  }

  .dynamo-calendar__past a span {
    display: block;
    margin-bottom: 0.15rem;
  }
}

@media (prefers-reduced-motion: reduce) {

  .dynamo-welcome__cursor {
    animation: none;
    opacity: 1;
  }

  .dynamo-welcome__cta,
  .dynamo-story__step,
  .dynamo-story__step-copy,
  .dynamo-story__stage-panel {
    transition: none;
  }
}

.dynamo-community-page {
  --dynamo-community-green: #76b900;
  --dynamo-community-green-bright: #8ed600;
  --dynamo-community-ink: var(--grayscale-a12);
  --dynamo-community-muted: var(--grayscale-a10);
  --dynamo-community-rule: color-mix(in srgb, var(--grayscale-a12) 14%, transparent);
  --dynamo-community-soft: #f3f4f3;
  --dynamo-community-titlebar-text: #555755;
  color: var(--dynamo-community-ink);
}

article:has(.dynamo-community-page) {
  width: 100% !important;
  max-width: 1200px !important;
  /* Fern centers the article in the viewport area left after the sidebar.
     On very wide screens that separates the content from its navigation.
     Anchor this custom layout to the sidebar-side content rail instead. */
  margin-inline: 0 auto;
}

.dark .dynamo-community-page {
  --dynamo-community-soft: #2b2c2b;
  --dynamo-community-titlebar-text: #d4d5d4;
}

.dynamo-community-page h2, .dynamo-community-page h3, .dynamo-community-page p { margin-top: 0; }

.dynamo-community-page h2 { margin-bottom: 0.55rem; font-size: clamp(1.55rem, 3vw, 2.15rem); font-weight: 650; line-height: 1.18; letter-spacing: -0.025em; }

.dynamo-community-eyebrow { margin-bottom: 0.55rem; color: var(--dynamo-community-green); font-size: 0.72rem; font-weight: 750; letter-spacing: 0.1em; text-transform: uppercase; }

.dynamo-community-section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1.5rem; margin-bottom: 1.2rem; }

.dynamo-community-section-heading > div > p:last-child { max-width: 670px; margin-bottom: 0; color: var(--dynamo-community-muted); line-height: 1.65; }

.dynamo-community-button { display: inline-flex; min-height: 2.55rem; flex: none; align-items: center; justify-content: center; gap: 0.5rem; padding: 0.62rem 0.9rem; border: 1px solid var(--dynamo-community-rule); border-radius: 8px; background: var(--grayscale-a1); color: var(--dynamo-community-ink) !important; font-size: 0.82rem; font-weight: 680; text-decoration: none !important; transition: border-color 150ms ease, background 150ms ease; }

.dynamo-community-button:hover { border-color: rgba(118, 185, 0, 0.55); background: rgba(118, 185, 0, 0.06); }

.dynamo-community-button.is-outline { background: transparent; }

.dynamo-community-button.is-primary { border-color: var(--dynamo-community-green); background: var(--dynamo-community-green); color: #0b1400 !important; }

.dynamo-community-button.is-primary:hover { background: var(--dynamo-community-green-bright); }

.dynamo-community-page svg { width: 0.95rem; height: 0.95rem; flex: none; fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }

.dynamo-community-meeting { display: grid; grid-template-columns: minmax(190px, 0.42fr) minmax(0, 1fr); overflow: hidden; margin: 1.5rem 0 3.5rem; border: 1px solid var(--dynamo-community-rule); border-radius: 14px; background: var(--grayscale-a1); }

.dynamo-community-meeting__cadence { display: flex; min-height: 240px; flex-direction: column; align-items: center; justify-content: center; padding: 1.5rem; border-right: 1px solid var(--dynamo-community-rule); background: linear-gradient(rgba(118,185,0,.045) 1px,transparent 1px), linear-gradient(90deg,rgba(118,185,0,.045) 1px,transparent 1px), rgba(118,185,0,.045); background-size: 24px 24px; text-align: center; }

.dynamo-community-meeting__cadence span { color: var(--dynamo-community-green); font-size: 0.7rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }

.dynamo-community-meeting__cadence strong { margin-top: 0.55rem; font-size: clamp(2.7rem, 6vw, 4rem); font-weight: 600; line-height: 1; letter-spacing: -0.055em; }

.dynamo-community-meeting__cadence small { margin-top: 0.4rem; color: var(--dynamo-community-muted); font-size: 0.72rem; }

.dynamo-community-meeting__copy { padding: clamp(1.5rem, 4vw, 2.5rem); }

.dynamo-community-meeting__copy > p:not(.dynamo-community-eyebrow) { max-width: 650px; margin-bottom: 0; color: var(--dynamo-community-muted); line-height: 1.65; }

.dynamo-community-meeting__actions { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; margin-top: 1.35rem; }

.dynamo-community-text-link { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.55rem 0.25rem; color: var(--dynamo-community-muted) !important; font-size: 0.79rem; font-weight: 650; text-decoration: none !important; }

.dynamo-community-text-link:hover { color: var(--dynamo-community-green) !important; }

.dynamo-community-calendar, .dynamo-community-channels { margin-top: 3.5rem; }

.dynamo-community-calendar__window {
  overflow: hidden;
  border: 1px solid var(--dynamo-community-rule);
  border-radius: 30px;
  background: var(--grayscale-a1);
  box-shadow: 0 32px 90px rgba(0, 0, 0, 0.1);
}

.dark .dynamo-community-calendar__window {
  background: #0c0d0b;
  box-shadow: 0 32px 100px rgba(0, 0, 0, 0.38);
}

.dynamo-community-channels__window {
  overflow: hidden;
  border: 1px solid var(--dynamo-community-rule);
  border-radius: 14px;
  background: var(--grayscale-a1);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.055);
}

.dark .dynamo-community-channels__window {
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.22);
}

.dynamo-community-calendar__chrome {
  position: relative;
  display: grid;
  min-height: 3.7rem;
  grid-template-columns: 0.76rem 0.76rem 0.76rem 1fr 2.28rem;
  align-items: center;
  gap: 0.5rem;
  padding: 0 1.15rem;
  border-bottom: 1px solid var(--dynamo-community-rule);
  background: var(--dynamo-community-soft);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.dynamo-community-calendar__chrome > span {
  width: 0.76rem;
  height: 0.76rem;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 50%;
  background: #ff5f57;
  box-shadow: inset 0 0 0 0.5px rgba(0, 0, 0, 0.08);
}

.dynamo-community-calendar__chrome > span:nth-child(2) { background: #febc2e; }

.dynamo-community-calendar__chrome > span:nth-child(3) { background: #28c840; }

.dynamo-community-calendar__chrome strong {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  color: var(--dynamo-community-titlebar-text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
}

.dynamo-community-window-bar {
  display: grid;
  min-height: 3.4rem;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  padding: 0 1rem;
  border-bottom: 1px solid var(--dynamo-community-rule);
  background: var(--dynamo-community-soft);
}

.dark .dynamo-community-calendar__chrome,
.dark .dynamo-community-window-bar {
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.dynamo-community-window-bar > strong {
  color: var(--dynamo-community-titlebar-text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
}

.dynamo-community-window-dots { display: flex; gap: 0.48rem; }

.dynamo-community-window-dots i { width: 0.72rem; height: 0.72rem; border: 1px solid rgba(0,0,0,.12); border-radius: 50%; background: #ff5f57; box-shadow: inset 0 0 0 .5px rgba(0,0,0,.08); }

.dynamo-community-window-dots i:nth-child(2) { background: #febc2e; }

.dynamo-community-window-dots i:nth-child(3) { background: #28c840; }

.dynamo-community-calendar__weekdays,
.dynamo-community-calendar__grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.dynamo-community-calendar__weekdays {
  border-bottom: 1px solid var(--dynamo-community-rule);
}

.dynamo-community-calendar__weekdays span {
  padding: 0.8rem 0.65rem;
  color: var(--dynamo-community-muted);
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-align: right;
  text-transform: uppercase;
}

.dynamo-community-calendar__day {
  position: relative;
  min-width: 0;
  min-height: 128px;
  padding: 0.6rem;
  border-right: 1px solid var(--dynamo-community-rule);
  border-bottom: 1px solid var(--dynamo-community-rule);
  background: color-mix(in srgb, var(--grayscale-a1) 95%, transparent);
}

.dynamo-community-calendar__day:nth-child(7n) { border-right: 0; }

.dynamo-community-calendar__day:nth-last-child(-n + 7) { border-bottom: 0; }

.dynamo-community-calendar__day.is-empty { background: color-mix(in srgb, var(--grayscale-a3) 32%, transparent); }

.dynamo-community-calendar__day.has-event { background: linear-gradient(145deg, rgba(118, 185, 0, 0.085), transparent 78%); }

.dynamo-community-calendar__number {
  display: block;
  margin-bottom: 0.55rem;
  color: var(--dynamo-community-muted);
  font-size: 0.73rem;
  font-weight: 700;
  text-align: right;
}

.dynamo-community-calendar__day.has-event .dynamo-community-calendar__number {
  color: var(--dynamo-community-ink);
}

.dynamo-community-calendar__day.is-today {
  background: color-mix(in srgb, var(--dynamo-community-green) 13%, var(--grayscale-a1));
}

.dynamo-community-calendar__day.is-today .dynamo-community-calendar__number {
  display: grid;
  width: 1.65rem;
  height: 1.65rem;
  margin: -0.15rem -0.15rem 0.3rem auto;
  place-items: center;
  border-radius: 50%;
  background: color-mix(in srgb, var(--dynamo-community-green) 22%, transparent);
  color: color-mix(in srgb, var(--dynamo-community-green) 72%, var(--dynamo-community-ink));
}

.dynamo-community-calendar__event-slot {
  position: relative;
  min-height: 3.5rem;
  margin: 0.32rem 0;
}

.dynamo-community-calendar__event-slot > a {
  position: absolute;
  inset: 0;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  align-items: start;
  gap: 0.08rem;
  overflow: hidden;
  padding: 0.42rem 0.48rem;
  border: 1px solid transparent;
  border-radius: 7px;
  background: rgba(118, 185, 0, 0.13);
  color: var(--dynamo-community-ink) !important;
  font-size: 0.68rem;
  font-weight: 700;
  line-height: 1.25;
  text-decoration: none !important;
}

.dynamo-community-calendar__event-slot time {
  color: color-mix(in srgb, var(--dynamo-community-green) 78%, var(--dynamo-community-ink));
  font-size: 0.58rem;
  font-variant-numeric: tabular-nums;
  font-weight: 800;
  line-height: 1.45;
  white-space: nowrap;
}

.dynamo-community-calendar__event-title {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.dynamo-community-calendar__event-slot > a:hover,
.dynamo-community-calendar__event-slot > a:focus-visible {
  z-index: 12;
  right: auto;
  width: max(100%, 230px);
  height: auto;
  min-height: 100%;
  overflow: visible;
  border-color: rgba(118, 185, 0, 0.38);
  background: color-mix(in srgb, var(--grayscale-a1) 88%, #76b900 12%);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
}

.dynamo-community-calendar__event-slot > a:focus-visible {
  outline: 2px solid var(--dynamo-community-green);
  outline-offset: 2px;
}

.dynamo-community-calendar__event-slot > a:hover .dynamo-community-calendar__event-title,
.dynamo-community-calendar__event-slot > a:focus-visible .dynamo-community-calendar__event-title {
  display: block;
  overflow: visible;
  -webkit-line-clamp: unset;
}

.dynamo-community-calendar__day:nth-child(7n) .dynamo-community-calendar__event-slot > a:hover,
.dynamo-community-calendar__day:nth-child(7n) .dynamo-community-calendar__event-slot > a:focus-visible,
.dynamo-community-calendar__day:nth-child(7n - 1) .dynamo-community-calendar__event-slot > a:hover,
.dynamo-community-calendar__day:nth-child(7n - 1) .dynamo-community-calendar__event-slot > a:focus-visible {
  right: 0;
  left: auto;
}

.dynamo-community-calendar__upcoming {
  display: grid;
  grid-template-columns: 130px minmax(0, 1fr);
  gap: 1.5rem;
  padding-top: 1.5rem;
}

.dynamo-community-calendar__upcoming > p {
  margin: 0;
  padding-top: 1rem;
  color: var(--dynamo-community-muted);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.11em;
  text-transform: uppercase;
}

.dynamo-community-calendar__upcoming > div { border-top: 1px solid var(--dynamo-community-rule); }

.dynamo-community-calendar__upcoming a {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr) 1.2rem;
  align-items: center;
  gap: 1rem;
  padding: 1.15rem 0.35rem;
  border-bottom: 1px solid var(--dynamo-community-rule);
  color: var(--dynamo-community-ink) !important;
  text-decoration: none !important;
  transition: padding 160ms ease, background 160ms ease;
}

.dynamo-community-calendar__upcoming a:hover {
  padding-inline: 0.75rem;
  background: rgba(118, 185, 0, 0.055);
}

.dynamo-community-event-date {
  display: flex;
  width: 3.4rem;
  height: 3.4rem;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--dynamo-community-rule);
  border-radius: 13px;
  background: var(--grayscale-a1);
  color: var(--dynamo-community-ink);
  font-size: 1.05rem;
  line-height: 1;
}

.dynamo-community-event-date strong {
  margin-bottom: 0.28rem;
  color: #e2332b;
  font-size: 0.58rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.dynamo-community-event-copy { display: flex; min-width: 0; flex-direction: column; gap: 0.25rem; }

.dynamo-community-event-copy > strong { font-size: 0.92rem; }

.dynamo-community-event-copy > small { overflow: hidden; color: var(--dynamo-community-muted); font-size: 0.74rem; text-overflow: ellipsis; white-space: nowrap; }

.dynamo-community-channels__grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); }

.dynamo-community-channels__grid > a { display: grid; min-width: 0; grid-template-columns: 42px minmax(0,1fr) auto; align-items: center; gap: 0.8rem; padding: 1rem; border-right: 1px solid var(--dynamo-community-rule); border-bottom: 1px solid var(--dynamo-community-rule); color: var(--dynamo-community-ink) !important; text-decoration: none !important; }

.dynamo-community-channels__grid > a:nth-child(2n) { border-right: 0; }

.dynamo-community-channels__grid > a:last-child:nth-child(odd) { grid-column: 1 / -1; border-right: 0; border-bottom: 0; }

.dynamo-community-channels__grid > a:hover { background: rgba(118,185,0,.045); }

.dynamo-community-channels__grid > a:hover > span:nth-child(2) > strong { color: var(--dynamo-community-green); }

.dynamo-community-channels__grid > a > span:nth-child(2) { min-width: 0; }

.dynamo-community-channels__grid strong, .dynamo-community-channels__grid small { display: block; }

.dynamo-community-channels__grid strong { font-size: 0.82rem; }

.dynamo-community-channels__grid small { overflow: hidden; margin-top: 0.18rem; color: var(--dynamo-community-muted); font-size: 0.68rem; text-overflow: ellipsis; white-space: nowrap; }

.dynamo-community-app { display: grid; width: 38px; height: 38px; place-items: center; border: 1px solid rgba(255,255,255,.22); border-radius: 9px; color: white; }

.dynamo-community-app svg { width: 1.15rem; height: 1.15rem; fill: currentColor; stroke: none; }

.dynamo-community-app--github { background: #24292f; }

.dynamo-community-app--discussions { background: #6e40c9; }

.dynamo-community-app--slack { background: #4a154b; }

.dynamo-community-app--youtube { background: #ff0033; }

.dynamo-community-contribute { display: grid; grid-template-columns: minmax(0,1fr) minmax(240px,.7fr); gap: 2rem; margin-top: 3.5rem; padding: 2rem 0; border-top: 1px solid var(--dynamo-community-rule); }

.dynamo-community-contribute > div > p:last-child { max-width: 620px; margin-bottom: 0; color: var(--dynamo-community-muted); line-height: 1.65; }

.dynamo-community-contribute ul { margin: 0; padding: 0; list-style: none; }

.dynamo-community-contribute li + li { border-top: 1px solid var(--dynamo-community-rule); }

.dynamo-community-contribute a { display: flex; align-items: center; justify-content: space-between; gap: 1rem; padding: 0.75rem 0; color: var(--dynamo-community-ink) !important; font-size: 0.8rem; font-weight: 650; text-decoration: none !important; }

.dynamo-community-contribute a:hover { color: var(--dynamo-community-green) !important; }

body:has(.dynamo-community-page) .fern-layout-footer,
.fern-layout-guide:has(.dynamo-community-page) > .grow { display: none; }

body:has(.dynamo-community-page) .fern-layout-guide,
article:has(.dynamo-community-page) { margin-bottom: 0; }

@media (max-width: 760px) {

  .dynamo-community-section-heading, .dynamo-community-meeting__actions { align-items: flex-start; }
  .dynamo-community-section-heading { flex-direction: column; }
  .dynamo-community-meeting, .dynamo-community-contribute { grid-template-columns: 1fr; }
  .dynamo-community-meeting__cadence { min-height: 180px; border-right: 0; border-bottom: 1px solid var(--dynamo-community-rule); }
  .dynamo-community-calendar__window { overflow-x: auto; border-radius: 22px; }
  .dynamo-community-calendar__chrome, .dynamo-community-calendar__weekdays, .dynamo-community-calendar__grid { min-width: 720px; }
  .dynamo-community-channels__grid { grid-template-columns: 1fr; }
  .dynamo-community-channels__grid > a, .dynamo-community-channels__grid > a:nth-child(2n), .dynamo-community-channels__grid > a:nth-last-child(-n + 2) { border-right: 0; border-bottom: 1px solid var(--dynamo-community-rule); }
  .dynamo-community-channels__grid > a:last-child { border-bottom: 0; }
  .dynamo-community-calendar__upcoming { grid-template-columns: 1fr; }
  .dynamo-community-calendar__upcoming > p { padding-top: 0; }
}

@media (max-width: 520px) {

  .dynamo-community-meeting__actions { flex-direction: column; }
  .dynamo-community-meeting__actions .dynamo-community-button, .dynamo-community-meeting__actions .dynamo-community-text-link { width: 100%; }
}


/* Community rail: the right-hand Slack and calendar links. Restored with
   the hero mark; the inline secondary buttons that briefly stood in for
   them are gone, so these selectors are the only place they live. */
.dynamo-welcome__community {
  position: absolute;
  top: -20rem;
  right: calc((100vw - 1200px) / -2 + 1rem);
  z-index: 30;
  display: flex;
  width: 242px;
  flex-direction: column;
  gap: 0.55rem;
}

.dynamo-welcome__notification {
  display: grid;
  grid-template-columns: 2.55rem 1fr;
  gap: 0.7rem;
  align-items: center;
  min-height: 4.3rem;
  padding: 0.65rem 0.75rem;
  border: 1px solid rgba(255, 255, 255, 0.58);
  border-radius: 18px;
  background:
    radial-gradient(circle at 14% 0%, rgba(255, 255, 255, 0.74), transparent 45%),
    linear-gradient(145deg, rgba(250, 250, 250, 0.72), rgba(231, 235, 226, 0.62));
  color: var(--grayscale-a12) !important;
  box-shadow:
    0 18px 46px rgba(28, 38, 18, 0.14),
    inset 0 1px 0 rgba(255, 255, 255, 0.88);
  text-decoration: none !important;
  backdrop-filter: blur(24px) saturate(1.4);
  transition:
    transform 160ms ease,
    border-color 160ms ease,
    box-shadow 160ms ease;
}

.dark .dynamo-welcome__notification {
  border-color: rgba(255, 255, 255, 0.12);
  background:
    radial-gradient(circle at 14% 0%, rgba(255, 255, 255, 0.12), transparent 45%),
    linear-gradient(145deg, rgba(35, 38, 31, 0.78), rgba(14, 16, 12, 0.7));
  box-shadow:
    0 18px 46px rgba(0, 0, 0, 0.42),
    inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.dynamo-welcome__notification:hover,
.dynamo-welcome__notification:focus-visible {
  transform: translateX(-4px);
  border-color: rgba(118, 185, 0, 0.55);
  box-shadow: 0 18px 42px rgba(54, 86, 0, 0.2);
}

.dynamo-welcome__notification:focus-visible {
  outline: 3px solid rgba(118, 185, 0, 0.34);
  outline-offset: 3px;
}

.dynamo-welcome__notification-icon {
  display: grid;
  width: 2.55rem;
  height: 2.55rem;
  place-items: center;
  border-radius: 11px;
  color: white;
  box-shadow:
    0 5px 14px rgba(0, 0, 0, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.dynamo-welcome__notification--slack .dynamo-welcome__notification-icon {
  background: #4a154b;
}

.dynamo-welcome__notification--calendar .dynamo-welcome__notification-icon {
  background: #ff3b30;
}

.dynamo-welcome__notification-icon svg {
  width: 1.42rem;
  height: 1.42rem;
  fill: currentColor;
}

.dynamo-welcome__calendar-app {
  display: grid;
  width: 1.55rem;
  height: 1.65rem;
  grid-template-rows: 0.55rem 1fr;
  overflow: hidden;
  border-radius: 5px;
  background: white;
  color: #171717;
  text-align: center;
}

.dynamo-welcome__calendar-app > span {
  display: grid;
  place-items: center;
  background: #ff3b30;
  color: white;
  font-size: 0.38rem;
  font-weight: 800;
  letter-spacing: 0.03em;
}

.dynamo-welcome__calendar-app strong {
  display: grid;
  place-items: center;
  font-size: 0.72rem;
  line-height: 1;
}

.dynamo-welcome__notification-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 0.18rem;
  color: var(--grayscale-a11);
  font-size: 0.76rem;
  line-height: 1.25;
}

.dynamo-welcome__notification-app {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
  color: var(--grayscale-a12);
  font-size: 0.78rem;
  font-weight: 700;
}

.dynamo-welcome__notification-app small {
  color: var(--grayscale-a9);
  font-size: 0.65rem;
  font-weight: 500;
  text-transform: uppercase;
}

@media (max-width: 1360px) {

  .dynamo-welcome__community {
    position: static;
    width: min(100%, 720px);
    margin: 2rem auto 0;
    flex-direction: row;
  }

  .dynamo-welcome__notification {
    flex: 1;
    min-width: 0;
  }
}

@media (max-width: 640px) {

  article:has(.dynamo-welcome) {
    padding-inline: 0.75rem;
  }

  article:has(.dynamo-welcome) > header {
    padding-top: 2.75rem;
  }

  article:has(.dynamo-welcome) > header::before {
    width: 60px;
    height: 60px;
    margin-bottom: 1rem;
    border-radius: 14px;
  }

  article:has(.dynamo-welcome) > header .fern-page-heading {
    font-size: clamp(3.5rem, 18vw, 4.75rem);
  }

  article:has(.dynamo-welcome) > header .fern-page-subtitle {
    max-width: 340px;
    white-space: normal;
  }

  .dynamo-welcome__terminal {
    margin-top: 6.5rem;
  }

  .dynamo-welcome__demo-reveal {
    top: -5.25rem;
    width: 100%;
    margin: 0;
    padding: 0.5rem 0.55rem 0.2rem;
  }

  .dynamo-welcome__demo-reveal h2 {
    font-size: 1.7rem;
  }

  .dynamo-welcome__terminal-stage {
    padding: 0.6rem;
    border-radius: 21px;
  }

  .dynamo-welcome__statement {
    width: 100%;
    min-height: 6.5rem;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .dynamo-welcome__community {
    width: min(100%, 350px);
    flex-direction: column;
  }

  .dynamo-welcome__notification {
    width: 100%;
  }
  .dynamo-story__step {
    margin-bottom: 3.5rem;
  }

  .dynamo-story__step h3 {
    font-size: 2.15rem;
  }

  .dynamo-story__step-copy > p:last-child {
    font-size: 0.98rem;
  }

  .dynamo-story__mobile-graphic {
    min-height: 390px;
    padding: 1rem;
    border-radius: 20px;
  }

  .dynamo-story-graphic {
    min-height: 330px;
  }

  .dynamo-story-orbit {
    width: 300px;
  }

  .dynamo-story-core {
    width: 5.8rem;
    height: 5.8rem;
    border-radius: 22px;
  }

  .dynamo-story-orbit .node {
    min-width: 5.7rem;
    min-height: 2.7rem;
    font-size: 0.64rem;
  }

  .dynamo-story-stack span:nth-child(n) {
    margin-inline: 0;
  }
}

@media (prefers-reduced-motion: reduce) {

  .dynamo-welcome__cursor {
    animation: none;
    opacity: 1;
  }

  .dynamo-welcome__cta,
  .dynamo-welcome__notification,
  .dynamo-story__step,
  .dynamo-story__step-copy,
  .dynamo-story__stage-panel {
    transition: none;
  }
}`;

export function LandingStyles() {
  return <style dangerouslySetInnerHTML={{ __html: LANDING_CSS }} />;
}
