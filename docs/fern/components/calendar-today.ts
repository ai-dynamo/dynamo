/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared "today" for the two calendar grids: EventsCalendar (Home) and
 * CommunityLanding's FullCalendar (Community).
 *
 * It lives here rather than in either component because both previously
 * derived the displayed month from `UPCOMING_EVENTS[0] ?? PAST_EVENTS[0]` --
 * the same defect, copy-pasted -- and pinning both to one helper is what stops
 * the next edit from fixing only one of them.
 */
import { GENERATED_ON } from "./events.generated";

export interface CalendarDate {
  year: number;
  /** Zero-based, matching Date#getMonth and the MONTHS lookups. */
  month: number;
  day: number;
}

/**
 * Today, as of the last generator run.
 *
 * GENERATED_ON is a Pacific YYYY-MM-DD refreshed on a six-hour cron by
 * community-events-refresh.yml, so a grid keyed off this tracks the real month
 * with no client-side date logic: no hydration mismatch, no post-hydration
 * flash, and the same answer on the server and in the browser.
 *
 * Parsed field-wise rather than through `new Date(string)`, which reads a bare
 * YYYY-MM-DD as UTC midnight and so reports the previous day anywhere west of
 * Greenwich -- Pacific included.
 */
export function resolveToday(): CalendarDate {
  const [year, month, day] = GENERATED_ON.split("-").map(Number);
  if (!year || !month || !day) {
    const fallback = new Date();
    return {
      year: fallback.getFullYear(),
      month: fallback.getMonth(),
      day: fallback.getDate(),
    };
  }
  return { year, month: month - 1, day };
}
