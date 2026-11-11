/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TerminalDemo — hero player for an asciinema terminal recording (.cast).
 *
 * Renders a looping, autoplaying terminal demo for a landing/hero surface.
 * Real, selectable terminal text at any resolution for a few KB — sharper and
 * far lighter than an mp4 screen capture of small terminal type.
 *
 * WHY CDN-LOADED (not an npm import):
 *   This repo ships NO package.json alongside fern/, and neither existing
 *   custom component (CustomFooter, RecipeStyles) pulls an external npm dep —
 *   the build-safe pattern here is self-contained, dependency-free components.
 *   So instead of importing `asciinema-player` directly (which Fern's
 *   mdx-components bundling can't resolve without a declared dep), we inject
 *   the player's JS + CSS from jsDelivr at runtime and call the global
 *   AsciinemaPlayer.create().
 *   The site already loads third-party JS (adobedtm) and iframes (ghbtns), so
 *   an external <script>/<link> is consistent with the page's existing origins.
 *
 * WHY "use client":
 *   Unlike RecipeStyles (a server component), this touches window/document,
 *   injects <script>/<link>, and uses hooks — it must run on the client. All
 *   DOM access is inside useEffect and guarded, so SSR renders an empty frame
 *   and hydration wires up the player. dispose() runs on unmount.
 *
 * USAGE:
 *   The page-usage example lives in README.md, for the reason recorded there.
 *   Ambient JSX is unsupported, so the component must be imported. Props:
 *     startAt / endAt     play only a slice; endAt loops back to startAt
 *     idleTimeLimit       cap any idle gap in seconds; kills dead air
 *     speed               above 1 plays faster
 *     chrome              "dynamo" (landing-page product bar) or "macos"
 *     theme               "fern-code" matches the site's code blocks
 *
 * GETTING THE CAST (deferred — not wired to a page yet):
 *   The recording lives on asciinema.org, not in the repo. Pull it local:
 *     curl -L https://asciinema.org/a/941754.cast -o docs/assets/dynamo-demo.cast
 *   then confirm the served path (docs/assets/ → site asset URL) and pass it as
 *   `src`. `src` may also be a full https URL, but a same-origin local asset
 *   avoids CORS and a third-party runtime dependency for a hero.
 *
 * TRIMMING A LONG RECORDING (three independent knobs, no re-recording):
 *   - startAt / endAt — show only a slice; endAt loops back to startAt.
 *   - idleTimeLimit   — cap any idle gap (seconds); kills dead air.
 *   - speed           — >1 plays faster.
 *   For a permanent hard cut you can also trim the .cast JSON itself, but these
 *   props cover "the recording is long, only show the good part" without it.
 */
"use client";

import { type CSSProperties, useEffect, useRef, useState } from "react";

// Pin the player version for reproducible builds; bump deliberately.
const PLAYER_VERSION = "3.17.0";
const PLAYER_JS = `https://cdn.jsdelivr.net/npm/asciinema-player@${PLAYER_VERSION}/dist/bundle/asciinema-player.min.js`;
const PLAYER_CSS = `https://cdn.jsdelivr.net/npm/asciinema-player@${PLAYER_VERSION}/dist/bundle/asciinema-player.css`;
// Subresource-integrity pins for the exact 3.17.0 bundle files (recompute if PLAYER_VERSION changes).
const PLAYER_JS_SRI = "sha384-s55nTYAdrPwGWmKKQ1lCnoB8H9LbqmsXsqqqPAHK2+T5h9IfI2dTXTDXJcZnySJD";
const PLAYER_CSS_SRI = "sha384-05cmIVRzN7mR7nmqajPpGPUPqJ5VyTAGHL1xJuiGWfhpWDp5hEfBk50kr21f3ILM";

const CSS_ELEMENT_ID = "asciinema-player-css";
const JS_ELEMENT_ID = "asciinema-player-js";

// A terminal cell is about 0.6em wide in the player's monospace fonts, and the
// player sets each line 1.333em tall. Their ratio sizes the pre-load reserve.
const CELL_WIDTH_EM = 0.6;
const LINE_HEIGHT_EM = 4 / 3;

type FitMode = "width" | "height" | "both" | false;
type ChromeStyle = "dynamo" | "macos";
type ColorScheme = "light" | "dark";

/** Track the site's light or dark mode, which Fern sets as a class on <html>. */
function useSiteColorScheme(enabled: boolean): ColorScheme | null {
  const [scheme, setScheme] = useState<ColorScheme | null>(null);
  useEffect(() => {
    if (!enabled || typeof document === "undefined") return;
    const root = document.documentElement;
    const read = () => setScheme(root.classList.contains("dark") ? "dark" : "light");
    read();
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, [enabled]);
  return scheme;
}

export interface TerminalDemoProps {
  /**
   * Cast source: a path relative to the page that renders this component
   * (recommended), or an https URL.
   *
   * Relative is not a style preference. Fern rewrites relative asset paths to
   * the published URL and leaves site-absolute ones untouched, so an absolute
   * path reaches the browser verbatim and 404s. See components/README.md.
   */
  src: string;
  /** Start playback at this time (seconds). Default 0. */
  startAt?: number;
  /** Stop at this time (seconds) and loop back to startAt. Omit to play to the end. */
  endAt?: number;
  /** Cap idle gaps to this many seconds so pauses feel snappy. Default 2. */
  idleTimeLimit?: number;
  /** Playback speed multiplier. Default 1. */
  speed?: number;
  /** Loop the (optionally trimmed) segment. Default true. */
  loop?: boolean;
  /** Autoplay on load. Auto-disabled under prefers-reduced-motion. Default true. */
  autoPlay?: boolean;
  /** Show the control bar. Default false for a clean hero. */
  controls?: boolean;
  /**
   * Force a named player theme: one of the player's own (asciinema | dracula |
   * monokai | nord | solarized-dark | solarized-light | tango), or "fern-code",
   * defined below, which follows the site's light and dark mode. Omit to use
   * the cast's embedded term.theme palette.
   */
  theme?: string;
  /** Poster frame, e.g. "npt:0:04" or a data URL. */
  poster?: string;
  /**
   * Force terminal width/height in cells (otherwise taken from the cast).
   * With both set, the frame also reserves the terminal's height before the
   * cast loads, so pass the cast's own size for a cast that isn't 120x25.
   */
  cols?: number;
  rows?: number;
  /** How the player scales to its container. Default "width". */
  fit?: FitMode;
  /** Render the window title bar. Default true. */
  titleBar?: boolean;
  /**
   * Window style. "dynamo" is the landing page's product bar. "macos" is a
   * plain macOS window in the style of the community calendar, light or dark
   * with the site. Default "dynamo".
   */
  chrome?: ChromeStyle;
  /** Optional centered caption in the title bar (e.g. a command or hostname). */
  title?: string;
  /** Escape hatch: extra asciinema-player options merged last. */
  extraOptions?: Record<string, unknown>;
  /** Extra class on the outer frame. */
  className?: string;
}

/** Inject the player stylesheet once (idempotent, client-only). */
function ensureStylesheet(): void {
  if (typeof document === "undefined") return;
  if (document.getElementById(CSS_ELEMENT_ID)) return;
  const link = document.createElement("link");
  link.id = CSS_ELEMENT_ID;
  link.rel = "stylesheet";
  link.href = PLAYER_CSS;
  link.integrity = PLAYER_CSS_SRI;
  link.crossOrigin = "anonymous";
  document.head.appendChild(link);
}

// Module-level cache so concurrent instances share one script load.
let scriptPromise: Promise<void> | null = null;

/** Inject the player bundle once and resolve when the global is ready. */
function ensureScript(): Promise<void> {
  if (typeof window === "undefined") return Promise.resolve();
  if ((window as unknown as { AsciinemaPlayer?: unknown }).AsciinemaPlayer) {
    return Promise.resolve();
  }
  if (scriptPromise) return scriptPromise;

  scriptPromise = new Promise<void>((resolve, reject) => {
    const existing = document.getElementById(JS_ELEMENT_ID) as HTMLScriptElement | null;
    if (existing) {
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error("asciinema-player failed to load")));
      return;
    }
    const script = document.createElement("script");
    script.id = JS_ELEMENT_ID;
    script.src = PLAYER_JS;
    script.integrity = PLAYER_JS_SRI;
    script.crossOrigin = "anonymous";
    script.async = true;
    script.addEventListener("load", () => resolve());
    script.addEventListener("error", () => reject(new Error("asciinema-player failed to load")));
    document.head.appendChild(script);
  });
  return scriptPromise;
}

export function TerminalDemo({
  src,
  startAt = 0,
  endAt,
  idleTimeLimit = 2,
  speed = 1,
  loop = true,
  autoPlay = true,
  controls = false,
  theme,
  poster,
  cols,
  rows,
  fit = "width",
  titleBar = true,
  chrome = "dynamo",
  title,
  extraOptions,
  className,
}: TerminalDemoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showControls, setShowControls] = useState(controls);
  const [loaded, setLoaded] = useState(false);
  const isMacos = chrome === "macos";
  // The player reads a CSS theme's colors once, when it mounts. A named theme
  // can change with the site's light and dark mode, so the player is rebuilt
  // on each switch and resumes where the old one was.
  const scheme = useSiteColorScheme(Boolean(theme));
  const resumeRef = useRef<{ src: string; time: number } | null>(null);
  // Reserve the terminal's height from its size in cells when the page gives it.
  const reserveStyle =
    cols && rows
      ? ({
          "--dynamo-term-aspect": ((Number(cols) * CELL_WIDTH_EM) / (Number(rows) * LINE_HEIGHT_EM)).toFixed(3),
        } as CSSProperties)
      : undefined;

  useEffect(() => {
    // Mount a themed player only once the site's mode is known, so it never
    // starts with the other mode's palette.
    if (theme && scheme === null) return;

    let player: {
      dispose?: () => void;
      getCurrentTime?: () => number;
      seek?: (t: number) => void;
      play?: () => void;
      pause?: () => void;
    } | null = null;
    let disposed = false;
    let trimTimer: ReturnType<typeof setInterval> | null = null;
    let releaseTimer: ReturnType<typeof setTimeout> | null = null;

    const prefersReducedMotion =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const shouldAutoPlay = autoPlay && !prefersReducedMotion;

    // Resume at the old player's position when only the site's mode changed.
    const resume = resumeRef.current;
    resumeRef.current = null;
    const resumeAt = resume && resume.src === src ? resume.time : undefined;

    ensureStylesheet();
    ensureScript()
      .then(() => {
        if (disposed || !containerRef.current) return;
        const AsciinemaPlayer = (
          window as unknown as {
            AsciinemaPlayer?: { create: (s: string, el: HTMLElement, o?: Record<string, unknown>) => typeof player };
          }
        ).AsciinemaPlayer;
        if (!AsciinemaPlayer) return;

        const options: Record<string, unknown> = {
          autoPlay: shouldAutoPlay,
          // When endAt is set we loop manually (seek back), so let native loop off.
          loop: endAt == null ? loop : false,
          idleTimeLimit,
          speed,
          // Always build the control bar into the DOM; we reveal it via CSS on
          // toggle so switching never re-creates the player (no jump/reset).
          controls: true,
          fit,
          startAt: resumeAt ?? startAt,
          // Only force a named theme when explicitly asked. Otherwise omit it so
          // the player uses the cast's embedded Dynamo Glass term.theme header.
          // Passing theme="asciinema" would override that palette.
          ...(theme ? { theme } : {}),
          ...(poster ? { poster } : {}),
          ...(cols ? { cols } : {}),
          ...(rows ? { rows } : {}),
          ...(extraOptions ?? {}),
        };

        player = AsciinemaPlayer.create(src, containerRef.current, options);

        // Release the pre-load aspect-ratio reserve only AFTER the player has
        // fetched and fit the cast — the fitted terminal height then matches the
        // reserve, so the box doesn't jump. Releasing synchronously here (before
        // fit) let the unfitted, natural-font-size terminal briefly define the
        // height: an expand-then-shrink flash. We wait for the 'play' event
        // (fires post-mount/fit under autoplay) and keep a timeout fallback so
        // the reserve is never stuck if the event never arrives.
        const release = () => {
          if (!disposed) setLoaded(true);
        };
        let released = false;
        const releaseOnce = () => {
          if (released) return;
          released = true;
          release();
        };
        const playerWithEvents = player as typeof player & {
          addEventListener?: (event: string, handler: () => void) => void;
        };
        if (typeof playerWithEvents?.addEventListener === "function") {
          playerWithEvents.addEventListener("play", releaseOnce);
          playerWithEvents.addEventListener("playing", releaseOnce);
        }
        // Fallback: if autoplay is blocked (e.g. reduced motion) or no event
        // fires, release after a short delay so the reserve doesn't linger.
        releaseTimer = setTimeout(releaseOnce, shouldAutoPlay ? 1200 : 400);

        // Manual trim/loop: asciinema-player has no native "endAt", so poll the
        // clock and seek back (or pause) at the boundary. Fully defensive — if a
        // method is missing on some version, endAt simply no-ops.
        if (endAt != null && player && typeof player.getCurrentTime === "function") {
          trimTimer = setInterval(() => {
            if (!player) return;
            try {
              const t = player.getCurrentTime?.();
              if (typeof t !== "number" || t < endAt) return;
              if (loop && typeof player.seek === "function") {
                player.seek(startAt);
                if (shouldAutoPlay && typeof player.play === "function") player.play();
              } else if (typeof player.pause === "function") {
                player.pause();
                if (trimTimer) clearInterval(trimTimer);
              }
            } catch {
              /* ignore transient player state errors */
            }
          }, 250);
        }
      })
      .catch(() => {
        /* Leave the frame empty rather than throw during hydration. */
      });

    return () => {
      disposed = true;
      setLoaded(false);
      if (trimTimer) clearInterval(trimTimer);
      if (releaseTimer) clearTimeout(releaseTimer);
      try {
        const time = player?.getCurrentTime?.();
        if (typeof time === "number" && time > 0) resumeRef.current = { src, time };
      } catch {
        /* no position to keep */
      }
      try {
        player?.dispose?.();
      } catch {
        /* already torn down */
      }
    };
  }, [src, startAt, endAt, idleTimeLimit, speed, loop, autoPlay, theme, scheme, poster, cols, rows, fit, extraOptions]);

  return (
    <div
      className={
        (className ? `dynamo-terminal-demo ${className}` : "dynamo-terminal-demo") +
        (isMacos ? " dynamo-terminal-demo--macos" : "") +
        (loaded ? " dynamo-terminal-demo--loaded" : "") +
        (showControls ? " dynamo-terminal-demo--controls" : "")
      }
    >
      <style>{TERMINAL_DEMO_CSS}</style>
      {titleBar && (
        <div className="dynamo-terminal-demo__bar">
          <span className="dynamo-terminal-demo__window-controls" aria-hidden="true">
            <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--red" />
            <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--yellow" />
            <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--green" />
          </span>
          {!isMacos && <span className="dynamo-terminal-demo__product">Dynamo</span>}
          {title && <span className="dynamo-terminal-demo__title">{title}</span>}
          {!isMacos && (
            <span className="dynamo-terminal-demo__activity">
              <span aria-hidden="true" />
              Demo running
            </span>
          )}
          <span className="dynamo-terminal-demo__toggle-wrap">
            <button
              type="button"
              className={
                showControls
                  ? "dynamo-terminal-demo__toggle dynamo-terminal-demo__toggle--on"
                  : "dynamo-terminal-demo__toggle"
              }
              onClick={() => setShowControls((v) => !v)}
              aria-pressed={showControls}
              aria-label={showControls ? "Hide playback controls" : "Show playback controls"}
            >
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" focusable="false">
                <circle cx="7" cy="8" r="2.1" />
                <line x1="10" y1="8" x2="20" y2="8" />
                <line x1="4" y1="8" x2="5" y2="8" />
                <circle cx="15" cy="16" r="2.1" />
                <line x1="4" y1="16" x2="12" y2="16" />
                <line x1="18" y1="16" x2="20" y2="16" />
              </svg>
            </button>
            <span className="dynamo-terminal-demo__tooltip" role="tooltip">
              {showControls ? "Hide playback controls" : "Show playback controls"}
            </span>
          </span>
        </div>
      )}
      <div ref={containerRef} className="dynamo-terminal-demo__player" style={reserveStyle} />
    </div>
  );
}

export default TerminalDemo;

/*
 * Injected as a page-level <style> (same rationale as RecipeStyles): the shared
 * NVIDIA global theme replaces the docs.yml `css:` field at publish, so
 * component CSS must ride along with the component. Frames the player only —
 * the player supplies its own terminal colors via `theme`. rgba borders track
 * both light and dark without a theme hook.
 */
const TERMINAL_DEMO_CSS = `
.dynamo-terminal-demo {
  position: relative;
  max-width: 860px;
  margin: 24px auto;
  overflow: hidden;
  border-radius: 18px;
  background: #030804;
  box-shadow: 0 4px 16px rgb(0 0 0 / 25%);
}
/* Product chrome keeps the recording visually tied to the Dynamo landing page. */
.dynamo-terminal-demo__bar {
  display: grid;
  grid-template-columns: auto auto minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 12px;
  min-height: 46px;
  padding: 0 13px 0 16px;
  position: relative;
  z-index: 4;
  border-bottom: 1px solid rgb(118 185 0 / 24%);
  background: linear-gradient(180deg, rgb(15 25 17 / 62%), rgb(4 11 6 / 56%));
  -webkit-backdrop-filter: blur(20px) saturate(128%);
  backdrop-filter: blur(20px) saturate(128%);
  box-shadow: inset 0 1px 0 rgb(224 255 191 / 6%);
}
.dynamo-terminal-demo__window-controls {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding-right: 3px;
}
.dynamo-terminal-demo__dot {
  display: block;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  box-shadow: inset 0 0 0 0.5px rgb(0 0 0 / 22%);
}
.dynamo-terminal-demo__dot--red { background: #ff5f57; }
.dynamo-terminal-demo__dot--yellow { background: #febc2e; }
.dynamo-terminal-demo__dot--green { background: #28c840; }
.dynamo-terminal-demo__product,
.dynamo-terminal-demo__activity {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: #d7ded1;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.055em;
  text-transform: uppercase;
  white-space: nowrap;
}
.dynamo-terminal-demo__activity > span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #76b900;
  box-shadow: 0 0 0 3px rgb(118 185 0 / 12%), 0 0 12px rgb(118 185 0 / 45%);
}
.dynamo-terminal-demo__activity {
  color: #95a08e;
  font-size: 0.64rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.dynamo-terminal-demo__activity > span {
  animation: dynamo-terminal-pulse 2s ease-in-out infinite;
}
@keyframes dynamo-terminal-pulse {
  0%, 100% { opacity: 0.45; }
  50% { opacity: 1; }
}
.dynamo-terminal-demo__title {
  min-width: 0;
  overflow: hidden;
  color: #aeb8aa;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 0.78rem;
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* Toggle button (top-right of the window bar) reveals the player's controls. */
.dynamo-terminal-demo__toggle-wrap {
  position: relative;
  flex: 0 0 auto;
  margin-left: auto;
  display: inline-flex;
}
.dynamo-terminal-demo__toggle {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: #8b949e;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.dynamo-terminal-demo__toggle:hover {
  background: rgb(255 255 255 / 8%);
  color: #d4d4d8;
}
.dynamo-terminal-demo__toggle:focus-visible {
  outline: 2px solid #6ca4f8;
  outline-offset: 1px;
}
.dynamo-terminal-demo__toggle--on {
  background: rgb(108 164 248 / 18%);
  color: #6ca4f8;
}
.dynamo-terminal-demo__toggle svg {
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
}
.dynamo-terminal-demo__toggle svg circle {
  fill: currentColor;
  stroke: none;
}
/*
 * Reserve the terminal's height BEFORE the cast loads so the window opens at
 * full size instead of flashing a 0-height body under the title bar. The
 * asciinema player only sets the real terminal height in JS after the cast
 * fetches; until then this aspect-ratio holds the space (defaults to the
 * 120x25 hero cast; the cols and rows props override it through
 * --dynamo-term-aspect). Once the player
 * mounts we drop the fixed ratio (via --loaded) so the box is free to grow
 * when the controls slide out.
 */
.dynamo-terminal-demo__player {
  width: 100%;
  aspect-ratio: var(--dynamo-term-aspect, 2.16);
  position: relative;
  z-index: 1;
  background: #030804;
  /* Clip the pre-fit frame: the player mounts at its natural font size for one
     frame before fit="width" scales it down, briefly overflowing this reserved
     box. Hide that overflow so the terminal never flashes oversized on load. */
  overflow: hidden;
}
.dynamo-terminal-demo--loaded .dynamo-terminal-demo__player {
  aspect-ratio: auto;
  overflow: visible;
}
.dynamo-terminal-demo__player .ap-player {
  display: block;
  width: 100%;
}
/* Keep the terminal body solid and predictable; the title bar carries the gradient. */
.dynamo-terminal-demo .ap-player {
  --term-font-family: RobotoMono, "Cascadia Code", "Source Code Pro", Menlo, Consolas, "DejaVu Sans Mono", monospace, "Symbols Nerd Font" !important;
  --term-color-background: #030804 !important;
  background: #030804;
}
.dynamo-terminal-demo .ap-term {
  border-color: #030804;
  background: #030804;
}
/*
 * Playback-controls reveal: the player always builds its control bar (native
 * position: absolute; bottom: 0). We grow the player box downward with an
 * animated padding-bottom, opening a strip for the bar to slide into — so the
 * bottom of the window genuinely expands. No player reset, no hover-reveal
 * (only the toggle controls it).
 */
.dynamo-terminal-demo__player .ap-player {
  padding-bottom: 0;
  transition: padding-bottom 0.28s ease;
}
.dynamo-terminal-demo__player .ap-player .ap-control-bar {
  opacity: 0 !important;
  pointer-events: none;
  transition: opacity 0.2s ease 0.06s;
}
.dynamo-terminal-demo--controls .dynamo-terminal-demo__player .ap-player {
  padding-bottom: 32px;
}
.dynamo-terminal-demo--controls .dynamo-terminal-demo__player .ap-control-bar {
  opacity: 1 !important;
  pointer-events: auto;
}
/* Tooltip on the toggle, styled like Fern's copy-button tooltip. */
.dynamo-terminal-demo__tooltip {
  position: absolute;
  top: calc(100% + 6px);
  right: 0;
  z-index: 100;
  padding: 4px 8px;
  border-radius: 6px;
  background: #1c2128;
  color: #e6edf3;
  border: 1px solid rgb(255 255 255 / 10%);
  box-shadow: 0 4px 12px rgb(0 0 0 / 30%);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 0.72rem;
  line-height: 1;
  white-space: nowrap;
  opacity: 0;
  transform: translateY(-2px);
  pointer-events: none;
  transition: opacity 0.12s ease, transform 0.12s ease;
}
.dynamo-terminal-demo__toggle-wrap:hover .dynamo-terminal-demo__tooltip,
.dynamo-terminal-demo__toggle:focus-visible + .dynamo-terminal-demo__tooltip {
  opacity: 1;
  transform: translateY(0);
}
@media (prefers-reduced-motion: reduce) {
  .dynamo-terminal-demo__activity > span {
    animation: none;
    opacity: 1;
  }
}
@media (max-width: 640px) {
  .dynamo-terminal-demo__bar {
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: 9px;
  }
  .dynamo-terminal-demo__product,
  .dynamo-terminal-demo__activity {
    display: none;
  }
  .dynamo-terminal-demo {
    margin: 16px auto;
  }
}
/*
 * chrome="macos": a plain macOS window that matches the community calendar
 * windows on the home and community pages. The title bar follows the site's
 * light or dark mode, and the terminal body takes the player theme's colors.
 * It sits in a page's content column, so it fills that column instead of
 * stopping at the hero's 860px; the wider window means larger text.
 */
.dynamo-terminal-demo.dynamo-terminal-demo--macos {
  --dynamo-terminal-titlebar: #f3f4f3;
  --dynamo-terminal-titlebar-text: #555755;
  --dynamo-terminal-titlebar-rule: #d7d8d7;
  --dynamo-terminal-surface: #ffffff;
  max-width: none;
  border: 1px solid rgb(127 127 127 / 20%);
  border-radius: 14px;
  background: var(--dynamo-terminal-surface);
  box-shadow: 0 12px 32px rgb(0 0 0 / 8%);
}
.dark .dynamo-terminal-demo.dynamo-terminal-demo--macos {
  --dynamo-terminal-titlebar: #2b2c2b;
  --dynamo-terminal-titlebar-text: #d4d5d4;
  --dynamo-terminal-titlebar-rule: #171817;
  --dynamo-terminal-surface: #0c0d0b;
  border-color: rgb(255 255 255 / 12%);
  box-shadow: 0 14px 36px rgb(0 0 0 / 40%);
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__bar {
  grid-template-columns: minmax(4rem, 1fr) minmax(0, auto) minmax(4rem, 1fr);
  gap: 0.75rem;
  min-height: 3rem;
  padding: 0 0.75rem 0 1rem;
  border-bottom: 1px solid var(--dynamo-terminal-titlebar-rule);
  background: var(--dynamo-terminal-titlebar);
  -webkit-backdrop-filter: none;
  backdrop-filter: none;
  box-shadow: inset 0 1px 0 rgb(255 255 255 / 90%);
}
.dark .dynamo-terminal-demo--macos .dynamo-terminal-demo__bar {
  box-shadow: inset 0 1px 0 rgb(255 255 255 / 8%);
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__window-controls {
  gap: 0.5rem;
  padding-right: 0;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__dot {
  width: 0.76rem;
  height: 0.76rem;
  border: 1px solid rgb(0 0 0 / 12%);
  box-shadow: inset 0 0 0 0.5px rgb(0 0 0 / 8%);
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__title {
  grid-column: 2;
  color: var(--dynamo-terminal-titlebar-text);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
  font-size: 0.78rem;
  font-weight: 600;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle-wrap {
  grid-column: 3;
  justify-self: end;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle {
  color: #8a8c8a;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle:hover {
  background: rgb(0 0 0 / 6%);
  color: #3f413f;
}
.dark .dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle:hover {
  background: rgb(255 255 255 / 8%);
  color: #e4e5e4;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle--on,
.dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle--on:hover {
  background: rgb(118 185 0 / 16%);
  color: #4d8100;
}
.dark .dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle--on,
.dark .dynamo-terminal-demo--macos .dynamo-terminal-demo__toggle--on:hover {
  background: rgb(118 185 0 / 18%);
  color: #8cce37;
}
.dynamo-terminal-demo--macos .dynamo-terminal-demo__player {
  background: var(--dynamo-terminal-surface);
}
.dynamo-terminal-demo--macos .ap-player,
.dynamo-terminal-demo--macos .ap-term {
  border-color: var(--term-color-background);
  background: var(--term-color-background);
}
/*
 * theme="fern-code": the token colors of Fern's code blocks on this site,
 * min-light in light mode and Material Theme Darker in dark mode, so the
 * recording matches the code blocks around it. Comments are darker in light
 * mode and lighter in dark mode than Fern's, because the recording narrates
 * in comments, and the backgrounds match the macOS window body.
 *
 * Each slot is a syntax role rather than an ANSI hue: the two Fern themes give
 * one role different hues, such as purple command names in light mode and
 * yellow ones in dark mode. scripts/slime-demo/slime-demo.sh lists the role of
 * every slot. The !important beats the forced background above and the
 * inline colors the player sets from a cast's embedded theme.
 */
.dynamo-terminal-demo .ap-player.asciinema-player-theme-fern-code {
  --term-color-foreground: #24292e !important;
  --term-color-background: #ffffff !important;
  --term-color-0: #dddddd !important;  /* header rules */
  --term-color-1: #24292e !important;  /* attribute names */
  --term-color-2: #22863a !important;  /* strings, success */
  --term-color-3: #6f42c1 !important;  /* command names */
  --term-color-4: #6f42c1 !important;  /* builtins, function names */
  --term-color-5: #d32f2f !important;  /* declarations: export, def */
  --term-color-6: #d32f2f !important;  /* operators, control keywords */
  --term-color-7: #6a737d !important;  /* narration, comments */
  --term-color-8: #8c959f !important;  /* progress notes */
  --term-color-9: #1976d2 !important;  /* numbers, status */
  --term-color-10: #2b5581 !important; /* arguments, flags */
  --term-color-11: #ff9800 !important; /* parameters */
  --term-color-12: #212121 !important; /* call arguments */
  --term-color-13: #22863a !important; /* quote marks */
  --term-color-14: #212121 !important; /* punctuation */
  --term-color-15: #4d8100 !important; /* prompt: Fern accent text */
}
.dark .dynamo-terminal-demo .ap-player.asciinema-player-theme-fern-code {
  --term-color-foreground: #eeffff !important;
  --term-color-background: #0c0d0b !important;
  --term-color-0: #333333 !important;
  --term-color-1: #f07178 !important;
  --term-color-2: #c3e88d !important;
  --term-color-3: #ffcb6b !important;
  --term-color-4: #82aaff !important;
  --term-color-5: #c792ea !important;
  --term-color-6: #89ddff !important;
  --term-color-7: #8b949e !important;
  --term-color-8: #6e7681 !important;
  --term-color-9: #f78c6c !important;
  --term-color-10: #c3e88d !important;
  --term-color-11: #eeffff !important;
  --term-color-12: #82aaff !important;
  --term-color-13: #89ddff !important;
  --term-color-14: #89ddff !important;
  --term-color-15: #8cce37 !important;
}
`;
