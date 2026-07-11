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
 *   So instead of `import "asciinema-player"` (which Fern's mdx-components
 *   bundling can't resolve without a declared dep), we inject the player's JS +
 *   CSS from jsDelivr at runtime and call the global AsciinemaPlayer.create().
 *   The site already loads third-party JS (adobedtm) and iframes (ghbtns), so
 *   an external <script>/<link> is consistent with the page's existing origins.
 *
 * WHY "use client":
 *   Unlike RecipeStyles (a server component), this touches window/document,
 *   injects <script>/<link>, and uses hooks — it must run on the client. All
 *   DOM access is inside useEffect and guarded, so SSR renders an empty frame
 *   and hydration wires up the player. dispose() runs on unmount.
 *
 * USAGE (import — ambient JSX is unsupported, per the RecipeStyles note):
 *   import { TerminalDemo } from "@/components/TerminalDemo";
 *
 *   <TerminalDemo
 *     src="/dynamo/assets/dynamo-demo.cast"
 *     startAt={0}
 *     endAt={18}          // play only the first 18s of a long recording, then loop
 *     idleTimeLimit={2}   // compress dead air so pauses feel snappy
 *     speed={1.2}
 *   />
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

import { useEffect, useRef, useState } from "react";

// Pin the player version for reproducible builds; bump deliberately.
const PLAYER_VERSION = "3.17.0";
const PLAYER_JS = `https://cdn.jsdelivr.net/npm/asciinema-player@${PLAYER_VERSION}/dist/bundle/asciinema-player.min.js`;
const PLAYER_CSS = `https://cdn.jsdelivr.net/npm/asciinema-player@${PLAYER_VERSION}/dist/bundle/asciinema-player.css`;

const CSS_ELEMENT_ID = "asciinema-player-css";
const JS_ELEMENT_ID = "asciinema-player-js";

type FitMode = "width" | "height" | "both" | false;

export interface TerminalDemoProps {
  /** Cast source: a same-origin asset path (recommended) or an https URL. */
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
  /** Force a named player theme (asciinema | dracula | monokai | nord | solarized-dark | solarized-light | tango). Omit to use the cast's embedded term.theme palette. */
  theme?: string;
  /** Poster frame, e.g. "npt:0:04" or a data URL. */
  poster?: string;
  /** Force terminal width/height in cells (otherwise taken from the cast). */
  cols?: number;
  rows?: number;
  /** How the player scales to its container. Default "width". */
  fit?: FitMode;
  /** Render the macOS-style title bar with traffic-light dots. Default true. */
  titleBar?: boolean;
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
  title,
  extraOptions,
  className,
}: TerminalDemoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showControls, setShowControls] = useState(controls);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
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
          startAt,
          // Only force a named theme when explicitly asked. Otherwise omit it so
          // the player uses the cast's embedded term.theme header (our GitHub
          // Dark palette). Passing theme="asciinema" would override that header.
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
        player?.dispose?.();
      } catch {
        /* already torn down */
      }
    };
  }, [src, startAt, endAt, idleTimeLimit, speed, loop, autoPlay, theme, poster, cols, rows, fit, extraOptions]);

  return (
    <div
      className={
        (className ? `dynamo-terminal-demo ${className}` : "dynamo-terminal-demo") +
        (loaded ? " dynamo-terminal-demo--loaded" : "") +
        (showControls ? " dynamo-terminal-demo--controls" : "")
      }
    >
      <style>{TERMINAL_DEMO_CSS}</style>
      {titleBar && (
        <div className="dynamo-terminal-demo__bar">
          <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--red" />
          <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--yellow" />
          <span className="dynamo-terminal-demo__dot dynamo-terminal-demo__dot--green" />
          {title && <span className="dynamo-terminal-demo__title">{title}</span>}
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
      <div ref={containerRef} className="dynamo-terminal-demo__player" />
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
  max-width: 860px;
  margin: 24px auto;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgb(0 0 0 / 25%);
  background: #1a1a2e;
}
/* macOS-style title bar (colors match the OpenShell reference). */
.dynamo-terminal-demo__bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 13px;
  background: #252545;
}
.dynamo-terminal-demo__dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  display: inline-block;
}
.dynamo-terminal-demo__dot--red { background: #ff5f56; }
.dynamo-terminal-demo__dot--yellow { background: #ffbd2e; }
.dynamo-terminal-demo__dot--green { background: #27c93f; }
.dynamo-terminal-demo__title {
  flex: 1;
  text-align: center;
  margin-right: 0; /* balanced by the toggle button on the right */
  color: #c9ccd1;
  /* macOS window-title style: system UI sans-serif, semibold, tight tracking. */
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
 * 120x25 hero cast; overridable via --dynamo-term-aspect). Once the player
 * mounts we drop the fixed ratio (via --loaded) so the box is free to grow
 * when the controls slide out.
 */
.dynamo-terminal-demo__player {
  width: 100%;
  aspect-ratio: var(--dynamo-term-aspect, 2.16);
  background: #0d1117;
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
/* Match the player's terminal background to the cast's GitHub-Dark theme bg. */
.dynamo-terminal-demo .asciinema-player {
  background: #0d1117;
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
  z-index: 40;
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
@media (max-width: 640px) {
  .dynamo-terminal-demo {
    margin: 16px auto;
  }
}
`;
