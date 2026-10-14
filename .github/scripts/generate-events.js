// Generates the docs-site views of the Dynamo public Google Calendar:
//
//   docs/fern/components/events.generated.ts      -- events for the Home and
//     Community calendars, plus the weekly community meeting's schedule
//   docs/fern/assets/dynamo-community-meeting.ics -- that meeting's invite
//
// community-events-refresh.yml runs this every six hours. To run it by hand
// from the repo root:
//
//   npm ci --prefix .github/scripts/events
//   NODE_PATH=.github/scripts/events/node_modules node .github/scripts/generate-events.js
//
// Recurring series are expanded into one event per occurrence, so the grids
// mark a weekly meeting in every week instead of only on the series' first
// date. The meeting card and the invite are read from that same series, so a
// rescheduled meeting reaches the page on the next refresh.
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
const { RRule } = require('rrule');
const fs = require('fs');
const path = require('path');

const ICS_URL = 'https://calendar.google.com/calendar/ical/c_c2448d2efb09eac2ddee1f34524124135bd3f4554868769059105e18e1b97e8f%40group.calendar.google.com/public/full.ics';

const ALLOWED_ONLINE_HOSTS = ['meet.google.com', 'zoom.us', 'teams.microsoft.com', 'webex.com'];

// Count caps for one-off events. Recurring occurrences are bounded by
// expansionWindow() instead: under a count cap, one weekly meeting would crowd
// every one-off event out of the file.
const MAX_UPCOMING = 8;
const MAX_PAST = 6;

const REPO_ROOT = path.join(__dirname, '..', '..');
const OUT = path.join(REPO_ROOT, 'docs', 'fern', 'components', 'events.generated.ts');
const INVITE_OUT = path.join(REPO_ROOT, 'docs', 'fern', 'assets', 'dynamo-community-meeting.ics');

// The site labels every date and time in Pacific (see ptParts), so the meeting
// card and its invite are Pacific too.
const PACIFIC = 'America/Los_Angeles';

const DAY_MS = 24 * 60 * 60 * 1000;
const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const ICAL_WEEKDAYS = ['SU', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA'];

// Ranks weekly series whose title reads like the community meeting first. It
// ranks rather than filters: a renamed meeting that is the only weekly series
// running still wins.
const MEETING_TITLE_HINT = /\b(community|sync)\b/i;

// The meeting card and invite when the feed parses but has no running weekly
// series to read them from -- the series ended, went biweekly, or was made
// private. Keep this on the current slot.
const FALLBACK_MEETING = {
  source: 'fallback',
  title: 'Dynamo community meeting',
  // Pacific wall clock of the first meeting at this slot: Tuesday, Oct 13, 2026, 9:00 AM.
  startWall: Date.UTC(2026, 9, 13, 9, 0),
  durationMs: 60 * 60 * 1000,
  until: null,
  count: null,
  joinUrl: 'https://meet.google.com/zyn-hfqk-asm',
  stamp: null,
};

// Stable across reschedules, so a calendar app that imported an earlier invite
// applies a fresh download as a revision of that event, not a second meeting.
const INVITE_UID = 'dynamo-community-weekly@ai-dynamo.github.io';
const INVITE_NOTES_URL = 'https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/edit';

// US Pacific rules since 2007, matching the block Google Calendar emits.
const PACIFIC_VTIMEZONE = [
  'BEGIN:VTIMEZONE',
  `TZID:${PACIFIC}`,
  `X-LIC-LOCATION:${PACIFIC}`,
  'BEGIN:DAYLIGHT',
  'TZOFFSETFROM:-0800',
  'TZOFFSETTO:-0700',
  'TZNAME:PDT',
  'DTSTART:19700308T020000',
  'RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU',
  'END:DAYLIGHT',
  'BEGIN:STANDARD',
  'TZOFFSETFROM:-0700',
  'TZOFFSETTO:-0800',
  'TZNAME:PST',
  'DTSTART:19701101T020000',
  'RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU',
  'END:STANDARD',
  'END:VTIMEZONE',
];

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
  // Google Meet links ride in X-GOOGLE-CONFERENCE, not LOCATION, so an
  // online-only meeting would otherwise show no location at all.
  const loc = formatLocation(e.location || e.conference);
  return {
    title: e.summary,
    start: e.start.toISOString(),
    ...ptParts(e.start, isAllDay),
    isPast,
    recurring: Boolean(e.series),
    location: loc.label,
    locationUrl: loc.url,
    addUrl: buildAddToCalendarURL(e),
  };
}

/** A property's text: node-ical parses a property that has parameters to { params, val }. */
function text(value) {
  return value && typeof value === 'object' && 'val' in value ? value.val : value;
}

function isCancelled(e) {
  return String(e.status || '').toUpperCase() === 'CANCELLED';
}

function isTimeZone(name) {
  if (typeof name !== 'string' || name === '') return false;
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: name });
    return true;
  } catch (_) {
    return false;
  }
}

const wallClockFormats = new Map();

/**
 * What a wall clock in `timeZone` reads at `date`, as the UTC timestamp with
 * the same fields -- "floating" time. Recurrence rules are defined on wall-clock
 * time (every Tuesday at 9:00 in Los Angeles), so expanding them on floating
 * timestamps keeps 9:00 at 9:00 across daylight-saving changes.
 */
function wallClockMs(date, timeZone) {
  if (!wallClockFormats.has(timeZone)) {
    wallClockFormats.set(
      timeZone,
      new Intl.DateTimeFormat('en-US', {
        timeZone,
        hourCycle: 'h23',
        year: 'numeric',
        month: 'numeric',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
      }),
    );
  }
  const parts = Object.fromEntries(
    wallClockFormats
      .get(timeZone)
      .formatToParts(date)
      .map(({ type, value }) => [type, Number(value)]),
  );
  return (
    Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second) +
    date.getUTCMilliseconds()
  );
}

/**
 * The instant a wall clock in `timeZone` reads `floatingMs`, resolved the way
 * RFC 5545 resolves recurrence instances: a time that occurs twice when the
 * clocks fall back means the first one, and a time skipped when they spring
 * forward takes the UTC offset from before the gap.
 */
function zonedInstant(floatingMs, timeZone) {
  const offsetAt = (ms) => wallClockMs(new Date(ms), timeZone) - ms;
  // A day to either side brackets any one DST change, so these are the only
  // two offsets the answer can carry.
  const before = offsetAt(floatingMs - DAY_MS);
  const after = offsetAt(floatingMs + DAY_MS);
  const matches = [before, after]
    .map((offset) => floatingMs - offset)
    .filter((ms) => wallClockMs(new Date(ms), timeZone) === floatingMs);
  return new Date(matches.length > 0 ? Math.min(...matches) : floatingMs - before);
}

// node-ical keys EXDATE and RECURRENCE-ID entries by the UTC date of the
// occurrence they refer to.
const dateKey = (date) => date.toISOString().slice(0, 10);

/**
 * One occurrence, in the shape of the node-ical VEVENT fields that toEvent()
 * and buildAddToCalendarURL() read. `series` is the recurring VEVENT it was
 * expanded from, or null for a one-off event.
 */
function instance(source, start, end, series) {
  return {
    summary: source.summary,
    start,
    end,
    datetype: source.datetype,
    location: source.location,
    description: source.description,
    conference: source['GOOGLE-CONFERENCE'],
    status: source.status,
    series,
  };
}

/**
 * A recurring VEVENT's rule, re-anchored on floating wall-clock time.
 *
 * node-ical attaches an RRule to each recurring VEVENT but never expands it,
 * and that RRule can't be expanded as-is: rrule.js re-zones TZID dates through
 * the host's own time zone, so its output depends on where this runs, and
 * node-ical rebuilds the rule without UNTIL's trailing Z. So expansion runs on
 * floating time -- what RFC 5545 defines recurrence over -- and each occurrence
 * is resolved back to an instant in the event's own zone.
 */
function seriesRule(e) {
  const allDay = e.datetype === 'date';
  const zone = isTimeZone(e.start.tz) ? e.start.tz : PACIFIC;
  // node-ical builds all-day dates at midnight in the generating host's zone.
  const toFloating = (date) =>
    allDay ? Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) : wallClockMs(date, zone);
  const toInstant = (floatingMs) => {
    if (!allDay) return zonedInstant(floatingMs, zone);
    const date = new Date(floatingMs);
    return new Date(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
  };
  const { until } = e.rrule.origOptions;
  const rule = new RRule({
    ...e.rrule.origOptions,
    tzid: null,
    dtstart: new Date(toFloating(e.start)),
    // RFC 5545 makes UNTIL a UTC instant next to a zoned DTSTART, and a plain
    // date next to an all-day one, which rrule.js already reads as floating.
    until: until ? new Date(allDay ? until.getTime() : wallClockMs(until, zone)) : null,
  });
  return { rule, toFloating, toInstant };
}

/** A recurring VEVENT's occurrences that start in [from, to), with its exceptions applied. */
function expandSeries(e, from, to) {
  const { rule, toFloating, toInstant } = seriesRule(e);
  const durationMs = e.end ? e.end - e.start : 0;
  const excluded = e.exdate || {};
  const overrides = e.recurrences || {};
  const occurrences = [];

  // Pad the search by a day: the window is in instants, the rule in floating time.
  const searchFrom = new Date(toFloating(from) - DAY_MS);
  const searchTo = new Date(toFloating(to) + DAY_MS);
  for (const floating of rule.between(searchFrom, searchTo, true)) {
    const start = toInstant(floating.getTime());
    const key = dateKey(start);
    if (excluded[key]) continue;
    // A RECURRENCE-ID override replaces its occurrence outright: it can move,
    // retitle, or cancel it.
    const override = overrides[key];
    const occurrence = override
      ? instance(override, override.start, override.end, e)
      : instance(e, start, new Date(start.getTime() + durationMs), e);
    if (isCancelled(occurrence)) continue;
    if (occurrence.start >= from && occurrence.start < to) occurrences.push(occurrence);
  }
  return occurrences;
}

/**
 * The span recurring series are expanded across: the start of last month
 * through the end of next month, in Pacific. The grids show one month --
 * today's, or the nearest event's when today's has none -- so this covers
 * whichever month they land on, and still holds a series' latest occurrence
 * for the recent-events list early in a month.
 */
function expansionWindow(now) {
  const today = new Date(wallClockMs(now, PACIFIC));
  const year = today.getUTCFullYear();
  const month = today.getUTCMonth();
  return {
    from: zonedInstant(Date.UTC(year, month - 1, 1), PACIFIC),
    to: zonedInstant(Date.UTC(year, month + 2, 1), PACIFIC),
  };
}

/** Every one-off event plus each recurring occurrence that starts in [from, to), soonest first. */
function calendarInstances(data, from, to) {
  const seen = new Set();
  return (
    Object.values(data)
      .filter((e) => e.type === 'VEVENT' && e.start && e.summary && !isCancelled(e))
      .flatMap((e) =>
        e.rrule && e.rrule.origOptions ? expandSeries(e, from, to) : [instance(e, e.start, e.end, null)],
      )
      .sort((a, b) => a.start - b.start)
      // One title at one instant is one meeting on the page, even when the
      // calendar carries it twice -- as it did while a copied weekly series ran
      // alongside its original.
      .filter((o) => {
        const key = `${text(o.summary)}\u0000${o.start.getTime()}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
  );
}

/**
 * The recurring VEVENT the weekly community meeting follows: among weekly
 * series that are still running, one whose title reads like the meeting,
 * then the newest.
 *
 * Neither title nor UID is stable enough to match on. The meeting has already
 * moved once by ending its Wednesday series and starting a Tuesday one under a
 * new UID and a new name ("Dynamo community meeting" became "Dynamo
 * Engineering Sync"). Newest-first follows a move like that even while the old
 * series still has weeks left to run.
 */
function findMeetingSeries(runningSeries) {
  const candidates = [...new Set(runningSeries)].filter((e) => {
    const { freq, interval, byweekday } = e.rrule.origOptions;
    // The card reads "Every <weekday>", which only fits a series that meets on
    // one weekday, every week.
    return (
      e.datetype !== 'date' &&
      freq === RRule.WEEKLY &&
      (interval || 1) === 1 &&
      [].concat(byweekday ?? []).length <= 1
    );
  });
  const hinted = (e) => Number(MEETING_TITLE_HINT.test(String(text(e.summary))));
  candidates.sort(
    (a, b) => hinted(b) - hinted(a) || b.start - a.start || String(a.uid).localeCompare(String(b.uid)),
  );
  return candidates[0] || null;
}

function isOnlineMeetingUrl(value) {
  try {
    const url = new URL(String(value));
    const host = url.hostname.replace(/^www\./, '');
    return url.protocol === 'https:' && ALLOWED_ONLINE_HOSTS.some((h) => host === h || host.endsWith(`.${h}`));
  } catch (_) {
    return false;
  }
}

/** The series' join link: its conference, else its location, else the first one in its description. */
function joinUrlOf(e) {
  const described = String(text(e.description) || '').match(/https?:\/\/[^\s<>"']+/g) || [];
  return [e['GOOGLE-CONFERENCE'], text(e.location), ...described].find(isOnlineMeetingUrl) || null;
}

/** The meeting's schedule, read from its series; null when the rule never fires. */
function meetingFromSeries(e) {
  const { rule, toInstant } = seriesRule(e);
  // Anchor on the rule's first occurrence. It stays put from one refresh to the
  // next, so the invite only changes when the series does, and unlike a
  // hand-entered DTSTART it always falls on a day the rule recurs on.
  const first = rule.after(rule.options.dtstart, true);
  if (!first) return null;
  const joinUrl = joinUrlOf(e);
  if (!joinUrl) {
    console.log(`::warning::"${text(e.summary)}" has no online meeting link; using the fallback join link.`);
  }
  const { until, count } = e.rrule.origOptions;
  return {
    source: 'calendar',
    title: String(text(e.summary)),
    startWall: wallClockMs(toInstant(first.getTime()), PACIFIC),
    durationMs: e.end ? e.end - e.start : FALLBACK_MEETING.durationMs,
    until: until || null,
    count: count || null,
    joinUrl: joinUrl || FALLBACK_MEETING.joinUrl,
    stamp: e.created || null,
  };
}

/** The meeting card's fields, as events.generated.ts exports them. */
function meetingCard(meeting) {
  const start = new Date(meeting.startWall);
  const hour = start.getUTCHours();
  return {
    source: meeting.source,
    title: meeting.title,
    weekday: WEEKDAYS[start.getUTCDay()],
    time: `${hour % 12 || 12}:${String(start.getUTCMinutes()).padStart(2, '0')}`,
    meridiem: hour < 12 ? 'AM' : 'PM',
    joinUrl: meeting.joinUrl,
  };
}

/**
 * The calendar as of `now`: the upcoming and past event lists, and the weekly
 * community meeting -- or FALLBACK_MEETING when no weekly series is running.
 */
function buildCalendar(data, now) {
  const { from, to } = expansionWindow(now);
  const instances = calendarInstances(data, from, to);
  const isUpcoming = (o) => (o.end || o.start) >= now;
  const byStart = (a, b) => a.start - b.start;

  const oneOffs = instances.filter((o) => !o.series);
  const occurrences = instances.filter((o) => o.series);

  const upcoming = [...oneOffs.filter(isUpcoming).slice(0, MAX_UPCOMING), ...occurrences.filter(isUpcoming)]
    .sort(byStart)
    .map((o) => toEvent(o, false));

  const past = [
    ...oneOffs.filter((o) => !isUpcoming(o)).slice(-MAX_PAST),
    ...occurrences.filter((o) => !isUpcoming(o)),
  ]
    .sort(byStart)
    .reverse()
    .map((o) => toEvent(o, true));

  const series = findMeetingSeries(occurrences.filter(isUpcoming).map((o) => o.series));
  const meeting = (series && meetingFromSeries(series)) || FALLBACK_MEETING;
  return { upcoming, past, meeting };
}

function renderEventsModule({ generatedOn, upcoming, past, meeting }) {
  const banner =
    '/*\n' +
    ' * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n' +
    ' * SPDX-License-Identifier: Apache-2.0\n' +
    ' *\n' +
    ' * AUTO-GENERATED by .github/scripts/generate-events.js — do not edit by hand.\n' +
    ' * Source: Dynamo public Google Calendar.\n' +
    ' * Refreshed by .github/workflows/community-events-refresh.yml and committed to main.\n' +
    ' */\n';

  return (
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
    '  /** One occurrence of a recurring series. Agenda lists show each series once. */\n' +
    '  recurring: boolean;\n' +
    '  location: string | null;\n' +
    '  locationUrl: string | null;\n' +
    '  addUrl: string;\n' +
    '}\n\n' +
    'export interface CommunityMeeting {\n' +
    '  /** "calendar" when read from the weekly series, "fallback" when none was running. */\n' +
    '  source: "calendar" | "fallback";\n' +
    '  /** Title of the series on the calendar. */\n' +
    '  title: string;\n' +
    '  /** Weekday the meeting recurs on, in Pacific. */\n' +
    '  weekday: string;\n' +
    '  /** Start time in Pacific, without the meridiem. */\n' +
    '  time: string;\n' +
    '  meridiem: "AM" | "PM";\n' +
    '  joinUrl: string;\n' +
    '}\n\n' +
    '/** Generation date in Pacific, YYYY-MM-DD. The calendar grid treats this as "today". */\n' +
    `export const GENERATED_ON = ${JSON.stringify(generatedOn)};\n\n` +
    '/** The weekly community meeting, read from its recurring series on the calendar. */\n' +
    `export const COMMUNITY_MEETING: CommunityMeeting = ${JSON.stringify(meetingCard(meeting), null, 2)};\n\n` +
    `export const UPCOMING_EVENTS: DynamoEvent[] = ${JSON.stringify(upcoming, null, 2)};\n\n` +
    `export const PAST_EVENTS: DynamoEvent[] = ${JSON.stringify(past, null, 2)};\n`
  );
}

/** RFC 5545 TEXT escaping. */
function escapeText(value) {
  return String(value)
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/\r?\n/g, '\\n');
}

// codespell's word pattern. Folding only between words keeps each word whole
// on one physical line, so the committed invite never gains a word fragment
// for codespell to flag, or a trailing space for the whitespace hook.
const WORD_CHAR = /[\w\-'’]/;

/**
 * Folds a content line at 75 octets (RFC 5545, section 3.1). RFC 5545 allows a
 * fold between any two characters; this picks the last one in budget that is
 * not right after whitespace and not inside a word, and splits mid-word only
 * when a single word overruns the whole line.
 */
function foldLine(line) {
  const chars = [...line];
  const physical = [];
  let start = 0;
  while (start < chars.length) {
    // A continuation line spends one octet on its leading space.
    const budget = physical.length === 0 ? 75 : 74;
    let end = start;
    for (let octets = 0; end < chars.length; end += 1) {
      octets += Buffer.byteLength(chars[end]);
      if (octets > budget) break;
    }
    if (end < chars.length) {
      const between = (cut) =>
        !/\s/.test(chars[cut - 1]) && !(WORD_CHAR.test(chars[cut - 1]) && WORD_CHAR.test(chars[cut]));
      let cut = end;
      while (cut > start + 1 && !between(cut)) cut -= 1;
      if (between(cut)) end = cut;
    }
    physical.push(chars.slice(start, end).join(''));
    start = end;
  }
  return physical.join('\n ');
}

const unfold = (ics) => ics.replace(/\r\n/g, '\n').replace(/\n[ \t]/g, '');
const utcStamp = (date) => date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
const floatingStamp = (floatingMs) => utcStamp(new Date(floatingMs)).replace(/Z$/, '');

/**
 * The community meeting's invite.
 *
 * `previousInvite` is the committed copy, read to carry its SEQUENCE forward:
 * unchanged while the meeting is, one higher once anything an attendee sees
 * changes, so calendar apps apply a fresh download as an update to the event
 * they already imported.
 */
function renderInvite(meeting, previousInvite) {
  const rule = ['FREQ=WEEKLY', `BYDAY=${ICAL_WEEKDAYS[new Date(meeting.startWall).getUTCDay()]}`];
  if (meeting.until) rule.push(`UNTIL=${utcStamp(meeting.until)}`);
  else if (meeting.count) rule.push(`COUNT=${meeting.count}`);

  const description =
    'Weekly NVIDIA Dynamo community meeting.\n\n' +
    `Join via Google Meet: ${meeting.joinUrl}\n\n` +
    `Meeting notes: ${INVITE_NOTES_URL}`;

  const render = (sequence) =>
    [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//NVIDIA Dynamo//Community Meeting//EN',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH',
      'X-WR-CALNAME:NVIDIA Dynamo Community Meeting',
      `X-WR-TIMEZONE:${PACIFIC}`,
      'X-DYNAMO-GENERATED-BY:.github/scripts/generate-events.js from the public Dynamo Google Calendar. Do not edit by hand.',
      ...PACIFIC_VTIMEZONE,
      'BEGIN:VEVENT',
      `UID:${INVITE_UID}`,
      `DTSTAMP:${utcStamp(meeting.stamp || zonedInstant(meeting.startWall, PACIFIC))}`,
      'SUMMARY:NVIDIA Dynamo Community Meeting',
      `DESCRIPTION:${escapeText(description)}`,
      `DTSTART;TZID=${PACIFIC}:${floatingStamp(meeting.startWall)}`,
      `DTEND;TZID=${PACIFIC}:${floatingStamp(meeting.startWall + meeting.durationMs)}`,
      `RRULE:${rule.join(';')}`,
      `LOCATION:${escapeText(meeting.joinUrl)}`,
      `URL:${meeting.joinUrl}`,
      'STATUS:CONFIRMED',
      `SEQUENCE:${sequence}`,
      'TRANSP:OPAQUE',
      'END:VEVENT',
      'END:VCALENDAR',
    ]
      .map(foldLine)
      .join('\n') + '\n';

  if (!previousInvite) return render(0);
  const previousSequence = Number(/^SEQUENCE:(\d+)$/m.exec(unfold(previousInvite))?.[1] ?? 0);
  // DTSTAMP and SEQUENCE are bookkeeping; any other difference is a revision.
  const substance = (ics) =>
    unfold(ics)
      .split('\n')
      .filter((line) => !/^(DTSTAMP|SEQUENCE)[:;]/.test(line))
      .join('\n');
  const unchanged = substance(render(previousSequence)) === substance(previousInvite);
  return render(unchanged ? previousSequence : previousSequence + 1);
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

function readIfExists(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') return null;
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

  const { upcoming, past, meeting } = buildCalendar(events, now);

  // Floor guard: never publish a calendar that has gone empty.
  //
  // node-ical turns a non-200 body into {} rather than throwing, so a calendar
  // that has been unshared or made private -- or a transient Google serve --
  // parses cleanly into zero events. That was self-correcting while this file
  // only ever existed inside the runner: one publish was wrong and the next
  // fixed it. Now that it is committed to main and is the source of truth, a
  // blank result blanks both grids on the live site and stays that way until
  // someone notices, behind a commit titled like every other refresh. Fail the
  // run instead and leave the last good calendar -- and invite -- in place.
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

  const { weekday, time, meridiem } = meetingCard(meeting);
  if (meeting.source === 'fallback') {
    console.log(
      `::warning::No weekly series is running on the calendar; the community meeting falls back to ${weekday}s at ${time} ${meridiem} Pacific.`,
    );
  }

  fs.writeFileSync(OUT, renderEventsModule({ generatedOn, upcoming, past, meeting }));
  fs.writeFileSync(INVITE_OUT, renderInvite(meeting, readIfExists(INVITE_OUT)));
  console.log(`Wrote ${OUT}: ${upcoming.length} upcoming, ${past.length} past.`);
  console.log(`Wrote ${INVITE_OUT}: ${weekday}s at ${time} ${meridiem} Pacific (${meeting.source}).`);
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

module.exports = { FALLBACK_MEETING, buildCalendar, meetingCard, renderInvite, wallClockMs, zonedInstant };
