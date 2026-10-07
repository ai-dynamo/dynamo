/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Backend chips for an Examples catalog card: one chip per backend, with a
 * glyph for each platform that backend runs on. The chips derive from the
 * card's platform:backend pairs, so they cannot advertise a combination the
 * card does not declare. ExamplesCatalog owns the chip styles.
 */
import type { ReactNode } from "react";

import { EXAMPLE_BACKENDS, EXAMPLE_PLATFORMS } from "./example-options";

// Original line drawings after the Kubernetes Guide and Local Guide tab icons:
// a heptagonal helm for Kubernetes and a laptop for Local / CLI.
const GLYPHS: Record<string, ReactNode> = {
  kubernetes: (
    <>
      <path d="M8 0.9L13.55 3.57L14.92 9.58L11.08 14.4L4.92 14.4L1.08 9.58L2.45 3.57Z" />
      <circle cx="8" cy="8" r="2.9" strokeWidth="1.2" />
      <circle cx="8" cy="8" r="1" fill="currentColor" stroke="none" />
      <path
        strokeWidth="1.15"
        d="M8 7L8 3.4M8.78 7.38L11.6 5.13M8.97 8.22L12.48 9.02M8.43 8.9L10 12.14M7.57 8.9L6 12.14M7.03 8.22L3.52 9.02M7.22 7.38L4.4 5.13"
      />
    </>
  ),
  local: (
    <>
      <rect x="2.75" y="3" width="10.5" height="7.5" rx="1.25" />
      <path d="M1 13h14" />
    </>
  ),
};

export function PlatformGlyph({ platform }: { platform: string }) {
  return (
    <svg
      className="dynamo-example-platform"
      data-platform={platform}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {GLYPHS[platform]}
    </svg>
  );
}

function rank(order: Record<string, string>, key: string): number {
  const index = Object.keys(order).indexOf(key);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

export function ExampleTargets({ targets }: { targets: string }) {
  const platforms = new Map<string, string[]>();
  for (const target of targets.split(/\s+/).filter(Boolean)) {
    const [platform, backend] = target.split(":");
    platforms.set(backend, [...(platforms.get(backend) ?? []), platform]);
  }
  const backends = Array.from(platforms.keys()).sort(
    (a, b) => rank(EXAMPLE_BACKENDS, a) - rank(EXAMPLE_BACKENDS, b)
  );

  return (
    <div className="dynamo-example-chips">
      {backends.map((backend) => {
        const supported = (platforms.get(backend) ?? []).sort(
          (a, b) => rank(EXAMPLE_PLATFORMS, a) - rank(EXAMPLE_PLATFORMS, b)
        );
        const names = supported.map((platform) => EXAMPLE_PLATFORMS[platform] ?? platform);
        return (
          <span key={backend} className="dynamo-example-chip" data-backend={backend}>
            {EXAMPLE_BACKENDS[backend] ?? backend}
            {/* Read aloud and searchable; the glyphs are decorative. */}
            <span className="dynamo-example-sr">{` (${names.join(", ")})`}</span>
            <span className="dynamo-example-chip-platforms">
              {supported.map((platform) => <PlatformGlyph key={platform} platform={platform} />)}
            </span>
          </span>
        );
      })}
    </div>
  );
}
