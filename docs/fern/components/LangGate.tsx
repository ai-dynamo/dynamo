/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * LangGate — render children only for (or except for) a set of browser languages.
 *
 * Detects the visitor's language from `navigator.language` on the client and
 * conditionally renders its children. Use it to localize a small piece of a
 * page (a card, a callout) without standing up full Fern localization.
 *
 * Usage — swap one card between Slack (default) and WeChat (Chinese browsers):
 *
 *     <LangGate langs="zh" invert>   // everyone EXCEPT zh
 *       <Card title="Community Slack" ... />
 *     </LangGate>
 *     <LangGate langs="zh">          // zh only
 *       <Card title="Community WeChat" ... />
 *     </LangGate>
 *
 * WHY "use client":
 *   `navigator.language` only exists in the browser. The component renders the
 *   default (non-matching) branch during SSR / first paint, then re-evaluates
 *   after mount. For a matched visitor this means a brief flash of the default
 *   content before the localized branch appears — an accepted trade-off for a
 *   single card. For whole translated pages, prefer Fern's native localization.
 *
 * NO WRAPPER ELEMENT:
 *   Returns a Fragment (or null), never a <div>, so a gated <Card> stays a
 *   direct child of <CardGroup> and the grid layout is preserved.
 */
"use client";

import { useEffect, useState, type ReactNode } from "react";

export interface LangGateProps {
  /** Language prefix(es) to match against navigator.language, e.g. "zh" or ["zh", "ja"]. */
  langs: string | string[];
  /** When true, render for every language EXCEPT the ones listed. Default false. */
  invert?: boolean;
  children: ReactNode;
}

export function LangGate({ langs, invert = false, children }: LangGateProps) {
  const [mounted, setMounted] = useState(false);
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    setMounted(true);
    const list = (Array.isArray(langs) ? langs : [langs]).map((l) => l.toLowerCase());
    const preferred = [
      ...(navigator.languages ?? []),
      navigator.language,
    ]
      .filter(Boolean)
      .map((l) => l.toLowerCase());
    setMatches(
      preferred.some((nav) =>
        list.some((l) => nav === l || nav.startsWith(`${l}-`)),
      ),
    );
  }, [langs]);

  // Before mount, show the default (non-matching) branch: the invert gate is
  // visible, the direct gate is hidden. This keeps SSR output stable.
  const show = mounted ? (invert ? !matches : matches) : invert;
  return show ? <>{children}</> : null;
}
