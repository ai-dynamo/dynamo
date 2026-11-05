/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * RecipeCount — the Recipes overview's model-family and configuration counts.
 *
 * Counts the page's model cards (`[data-recipe-card]`) in the browser: one card
 * per model family, and one configuration per id in each card's
 * `data-targets`, which scripts/gen_recipe_nav.py writes from the catalog.
 *
 * WHY COUNT IN THE BROWSER:
 *   A committed total is one line that every recipe PR changes. Two PRs that
 *   each add a recipe both turn 27 into 28, git merges that cleanly, and main
 *   ends up one short with the generator's --check failing. Each card's
 *   target list is its own line, so a clean merge of two recipe PRs is
 *   always correct, and the total follows from it.
 *
 * WHY "use client":
 *   The cards are only countable once the page is in the DOM. During SSR and
 *   first paint this renders an em dash, then the count after mount.
 */
"use client";

import { useEffect, useState } from "react";

export interface RecipeCountProps {
  /** "families" counts the cards; "configurations" counts their target ids. */
  of: "families" | "configurations";
}

export function RecipeCount({ of }: RecipeCountProps) {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    const cards = Array.from(
      document.querySelectorAll<HTMLElement>("[data-recipe-card]"),
    );
    setCount(
      of === "families"
        ? cards.length
        : cards.reduce(
            (total, card) =>
              total + (card.dataset.targets ?? "").split(/\s+/).filter(Boolean).length,
            0,
          ),
    );
  }, [of]);

  return <>{count ?? "—"}</>;
}
