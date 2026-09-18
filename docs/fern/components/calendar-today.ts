/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared "today" and displayed-month resolution for the two calendar grids:
 * EventsCalendar (Home) and CommunityLanding's FullCalendar (Community).
 *
 * It lives here rather than in either component because both previously
 * derived the displayed month from `UPCOMING_EVENTS[0] ?? PAST_EVENTS[0]` --
 * the same defect, copy-pasted -- and pinning both to one helper is what stops
 * the next edit from fixing only one of them.
 */
import { GENERATED_ON, PAST_EVENTS, UPCOMING_EVENTS } from "./events.generated";

export interface CalendarDate {
  year: number;
  /** Zero-based, matching Date#getMonth and the MONTHS lookups. */
  month: number;
  day: number;
}

export type CalendarMonth = Pick<CalendarDate, "year" | "month">;

/**
 * Maps the abbreviated month names generate-events.js emits ("Jul") to an index.
 *
 * Exported so the grids filter their events with the same lookup
 * resolveCalendarMonth() below picks the month with. Keeping a second copy in a
 * component means a change to one -- a locale tweak, the generator switching to
 * full month names -- lands the grid on a month whose event filter matches
 * nothing, which renders empty while resolveCalendarMonth() still reports a
 * confident answer.
 */
export const MONTH_INDEX: Record<string, number> = {
  Jan: 0,
  Feb: 1,
  Mar: 2,
  Apr: 3,
  May: 4,
  Jun: 5,
  Jul: 6,
  Aug: 7,
  Sep: 8,
  Oct: 9,
  Nov: 10,
  Dec: 11,
};

/**
 * How far GENERATED_ON may lag before it stops counting as "today".
 *
 * The refresh cron writes it every six hours, so three days of slack absorbs a
 * transient ICS outage while still catching a genuinely stalled refresh (an
 * expired OPS_BOT_PAT, dynamo-ops losing its ruleset bypass). Past that we show
 * no marker at all: a confidently wrong "today" is worse than an absent one.
 */
const STALE_AFTER_DAYS = 3;

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * GENERATED_ON as a date, with no staleness judgement.
 *
 * Parsed field-wise rather than through `new Date(string)`, which reads a bare
 * YYYY-MM-DD as UTC midnight and so reports the previous day anywhere west of
 * Greenwich -- Pacific included.
 */
function parseGeneratedOn(): CalendarDate | null {
  const [year, month, day] = GENERATED_ON.split("-").map(Number);
  if (!year || !month || !day) return null;
  return { year, month: month - 1, day };
}

/**
 * Today, as of the last generator run, or null when that is unknowable.
 *
 * GENERATED_ON is a Pacific YYYY-MM-DD refreshed on a six-hour cron by
 * community-events-refresh.yml, so a grid keyed off this tracks the real month
 * with no client-side date logic: no hydration mismatch, no post-hydration
 * flash, and the same answer on the server and in the browser.
 *
 * Returns null when GENERATED_ON is malformed or has gone stale, so callers can
 * omit the highlight rather than mark an arbitrary day as today. `Date.now()`
 * here is evaluated once at build time (these are server components), which is
 * the right granularity for a check measured in days -- and because any docs
 * commit rebuilds the site, a stalled calendar refresh still gets caught by the
 * next unrelated build.
 */
export function resolveToday(): CalendarDate | null {
  const generated = parseGeneratedOn();
  if (!generated) return null;

  const generatedUtc = Date.UTC(generated.year, generated.month, generated.day);
  if (Date.now() - generatedUtc > STALE_AFTER_DAYS * DAY_MS) return null;

  return generated;
}

function monthOfEvent(event: {
  year: string;
  month: string;
}): CalendarMonth | null {
  const year = Number(event.year);
  const month = MONTH_INDEX[event.month];
  if (!Number.isFinite(year) || month === undefined) return null;
  return { year, month };
}

/**
 * The month the grids should display.
 *
 * Today's month wins whenever it actually has events, so the marker and the
 * grid agree. Otherwise the grid follows the nearest event -- the soonest
 * upcoming one, else the most recent past one -- because events are announced
 * weeks ahead, so "the next event is next month" is the ordinary case rather
 * than an edge case. Pinning hard to today's month would render an empty grid
 * for most of the year.
 */
export function resolveCalendarMonth(): CalendarMonth {
  const today = resolveToday();

  if (today) {
    const hasEventsThisMonth = [...UPCOMING_EVENTS, ...PAST_EVENTS].some(
      (event) => {
        const eventMonth = monthOfEvent(event);
        return (
          eventMonth?.year === today.year && eventMonth?.month === today.month
        );
      },
    );
    if (hasEventsThisMonth) return { year: today.year, month: today.month };
  }

  // UPCOMING_EVENTS is soonest-first and PAST_EVENTS most-recent-first, so the
  // first parseable entry in each is the nearest event in that direction.
  for (const event of [...UPCOMING_EVENTS, ...PAST_EVENTS]) {
    const eventMonth = monthOfEvent(event);
    if (eventMonth) return eventMonth;
  }

  // No events at all: fall back to today, then to GENERATED_ON even if stale --
  // a slightly old month still beats rendering nothing.
  if (today) return { year: today.year, month: today.month };
  const generated = parseGeneratedOn();
  if (generated) return { year: generated.year, month: generated.month };

  const now = new Date();
  return { year: now.getUTCFullYear(), month: now.getUTCMonth() };
}
