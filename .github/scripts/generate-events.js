// Generates fern/components/events.generated.ts from the Dynamo public Google Calendar.
// Run during Fern publishing and on its six-hour refresh schedule (or manually:
// node .github/scripts/generate-events.js).
//
// The upcoming/past split is baked at generation time. Scheduled regeneration
// keeps the boundary fresh; the EventsCalendar
// component renders the baked arrays as-is, so there is no client-side date logic
// and no hydration mismatch.
//
// GENERATED_ON is emitted for the same reason: the month grid needs a "today"
// to highlight, and reading the clock in the component would either be build
// time (arbitrarily old) or client time (hydration mismatch). Pinning it to the
// generation date ties the grid to the six-hour refresh instead.
//
// It is a Pacific calendar date, not a timestamp, and that is deliberate on two
// counts. The grid only resolves to a day, so a timestamp would be false
// precision -- and because this file is committed, a timestamp would differ on
// every six-hour run and republish the docs four times a day over a byte that
// renders identically. A date changes once a day, when the rendered grid
// actually moves. Pacific because ptParts() labels every event in Pacific.

const ical = require('node-ical');
const fs = require('fs');
const path = require('path');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';

const ALLOWED_ONLINE_HOSTS = ['meet.google.com', 'zoom.us', 'teams.microsoft.com', 'webex.com'];

const MAX_UPCOMING = 8;
const MAX_PAST = 6;

const OUT = path.join(__dirname, '..', '..', 'docs', 'fern', 'components', 'events.generated.ts');

function safeUrl(url) {
  return url.replace(/\)/g, '%29');
}

// Returns { label, url } for the location cell.
function formatLocation(location) {
  if (!location) return { label: null, url: null };
  try {
    const parsed = new URL(location);
    const host = parsed.hostname.replace(/^www\./, '');
    if (/^(lu\.ma|luma\.com)$/i.test(host)) return { label: 'Luma', url: safeUrl(location) };
    if (ALLOWED_ONLINE_HOSTS.some((h) => host === h || host.endsWith(`.${h}`))) {
      return { label: 'Online', url: safeUrl(location) };
    }
  } catch (_) {
    /* not a URL — treat as a physical address below */
  }
  const parts = location.split(',').map((s) => s.trim());
  // "City, State, Country" -> City; otherwise first component.
  const city = parts.length >= 3 ? parts[parts.length - 3] : parts[0];
  return { label: city || null, url: null };
}

function buildAddToCalendarURL(e) {
  const fmt = (d) => d.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
  const params = new URLSearchParams({
    action: 'TEMPLATE',
    text: e.summary || 'Event',
    dates: `${fmt(e.start)}/${fmt(e.end || e.start)}`,
    ...(e.location && { location: e.location }),
    ...(e.description && { details: e.description }),
  });
  return `https://calendar.google.com/calendar/render?${params.toString()}`;
}

function ptParts(d, isAllDay) {
  const tz = isAllDay ? 'UTC' : 'America/Los_Angeles';
  const fmt = (opts) => d.toLocaleDateString('en-US', { timeZone: tz, ...opts });
  return {
    month: fmt({ month: 'short' }),
    day: fmt({ day: 'numeric' }),
    year: fmt({ year: 'numeric' }),
    dateLabel: fmt({ weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' }),
    timeLabel: isAllDay
      ? null
      : d.toLocaleTimeString('en-US', {
          timeZone: tz,
          hour: 'numeric',
          minute: '2-digit',
        }),
  };
}

function toEvent(e, isPast) {
  const isAllDay = e.datetype === 'date';
  const loc = formatLocation(e.location);
  return {
    title: e.summary,
    start: e.start.toISOString(),
    ...ptParts(e.start, isAllDay),
    isPast,
    location: loc.label,
    locationUrl: loc.url,
    addUrl: buildAddToCalendarURL(e),
  };
}

/**
 * How many events the committed file currently holds.
 *
 * Counted by matching the one field every event object carries exactly once,
 * rather than importing the file, since it is TypeScript. A missing file (first
 * run, fresh checkout) counts as zero rather than failing.
 */
function countExistingEvents() {
  try {
    return (fs.readFileSync(OUT, 'utf8').match(/"addUrl":/g) || []).length;
  } catch (err) {
    if (err.code === 'ENOENT') return 0;
    throw err;
  }
}

async function main() {
  const events = await ical.async.fromURL(ICS_URL);
  const now = new Date();
  // Assemble YYYY-MM-DD from named parts. toLocaleDateString('en-CA') is not
  // guaranteed to stay YYYY-MM-DD across ICU/CLDR updates (see node#45945).
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/Los_Angeles',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    })
      .formatToParts(now)
      .map(({ type, value }) => [type, value]),
  );
  const generatedOn = `${parts.year}-${parts.month}-${parts.day}`;

  const all = Object.values(events)
    .filter((e) => e.type === 'VEVENT' && e.start && e.summary && String(e.status || '').toUpperCase() !== 'CANCELLED')
    .sort((a, b) => a.start - b.start);

  const upcoming = all
    .filter((e) => (e.end || e.start) >= now)
    .slice(0, MAX_UPCOMING)
    .map((e) => toEvent(e, false));

  const past = all
    .filter((e) => (e.end || e.start) < now)
    .slice(-MAX_PAST)
    .reverse()
    .map((e) => toEvent(e, true));

  const banner =
    '/*\n' +
    ' * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n' +
    ' * SPDX-License-Identifier: Apache-2.0\n' +
    ' *\n' +
    ' * AUTO-GENERATED by .github/scripts/generate-events.js — do not edit by hand.\n' +
    ' * Source: Dynamo public Google Calendar.\n' +
    ' * Refreshed by .github/workflows/community-events-refresh.yml and committed to main.\n' +
    ' */\n';

  const body =
    banner +
    '\n' +
    'export interface DynamoEvent {\n' +
    '  title: string;\n' +
    '  start: string;\n' +
    '  month: string;\n' +
    '  day: string;\n' +
    '  year: string;\n' +
    '  dateLabel: string;\n' +
    '  timeLabel: string | null;\n' +
    '  isPast: boolean;\n' +
    '  location: string | null;\n' +
    '  locationUrl: string | null;\n' +
    '  addUrl: string;\n' +
    '}\n\n' +
    '/** Generation date in Pacific, YYYY-MM-DD. The calendar grid treats this as "today". */\n' +
    `export const GENERATED_ON = ${JSON.stringify(generatedOn)};\n\n` +
    `export const UPCOMING_EVENTS: DynamoEvent[] = ${JSON.stringify(upcoming, null, 2)};\n\n` +
    `export const PAST_EVENTS: DynamoEvent[] = ${JSON.stringify(past, null, 2)};\n`;

  // Floor guard: never publish a calendar that has gone empty.
  //
  // node-ical turns a non-200 body into {} rather than throwing, so a calendar
  // that has been unshared or made private -- or a transient Google serve --
  // parses cleanly into zero events. That was self-correcting while this file
  // only ever existed inside the runner: one publish was wrong and the next
  // fixed it. Now that it is committed to main and is the source of truth, a
  // blank result blanks both grids on the live site and stays that way until
  // someone notices, behind a commit titled like every other refresh. Fail the
  // run instead and leave the last good calendar in place.
  const existingCount = countExistingEvents();
  if (upcoming.length + past.length === 0 && existingCount > 0) {
    const veventCount = Object.values(events).filter(
      (e) => e.type === 'VEVENT',
    ).length;
    throw new Error(
      `Calendar fetch yielded no usable events (${veventCount} VEVENT(s) parsed) ` +
        `while ${path.basename(OUT)} currently holds ${existingCount}. ` +
        'Refusing to overwrite it with an empty calendar.',
    );
  }

  fs.writeFileSync(OUT, body);
  console.log(`Wrote ${OUT}: ${upcoming.length} upcoming, ${past.length} past.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
