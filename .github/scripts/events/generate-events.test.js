// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Tests for ../generate-events.js. From the repo root:
//
//   npm ci --prefix .github/scripts/events
//   npm test --prefix .github/scripts/events
//
// The fixtures mirror what the public Google Calendar feed publishes. Results
// must not depend on the host's time zone; run with TZ set to a few zones to
// check that.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const ical = require('node-ical');
const {
  FALLBACK_MEETING,
  buildCalendar,
  meetingCard,
  renderInvite,
  zonedInstant,
} = require('../generate-events.js');

// Wednesday, Oct 14, 2026, 10:43 AM Pacific.
const NOW = new Date('2026-10-14T17:43:53Z');

const VTIMEZONE = [
  'BEGIN:VTIMEZONE',
  'TZID:America/Los_Angeles',
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

function feed(...vevents) {
  return ical.sync.parseICS(
    [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Google Inc//Google Calendar 70.9054//EN',
      'X-WR-TIMEZONE:America/Los_Angeles',
      ...VTIMEZONE,
      ...vevents.flat(),
      'END:VCALENDAR',
    ].join('\r\n'),
  );
}

const vevent = (...lines) => ['BEGIN:VEVENT', ...lines, 'END:VEVENT'];

const weekly = ({ uid, summary, start, end, rule = 'FREQ=WEEKLY', extra = [] }) =>
  vevent(
    `DTSTART;TZID=America/Los_Angeles:${start}`,
    `DTEND;TZID=America/Los_Angeles:${end}`,
    `RRULE:${rule}`,
    `UID:${uid}`,
    'STATUS:CONFIRMED',
    `SUMMARY:${summary}`,
    ...extra,
  );

// The current slot, published the way Google Calendar publishes it.
const TUESDAY_SYNC = weekly({
  uid: 'tuesday@example.com',
  summary: 'Dynamo Engineering Sync',
  start: '20261013T090000',
  end: '20261013T100000',
  rule: 'FREQ=WEEKLY;BYDAY=TU',
  extra: ['X-GOOGLE-CONFERENCE:https://meet.google.com/aaa-bbbb-ccc', 'CREATED:20261012T150518Z'],
});

// The retired slot: ended by a UTC UNTIL (11:59 PM Pacific on Oct 13), with one
// week excluded.
const WEDNESDAY_MEETING = weekly({
  uid: 'wednesday@example.com',
  summary: 'Dynamo community meeting',
  start: '20260729T103000',
  end: '20260729T111500',
  rule: 'FREQ=WEEKLY;UNTIL=20261014T065959Z;BYDAY=WE',
  extra: ['EXDATE;TZID=America/Los_Angeles:20260923T103000', 'LOCATION:https://meet.google.com/old-slot-xyz'],
});

const MEETUP = vevent(
  'DTSTART:20261016T010000Z',
  'DTEND:20261016T040000Z',
  'UID:meetup@example.com',
  'LOCATION:San Francisco\\, CA',
  'STATUS:CONFIRMED',
  'SUMMARY:CoreWeave x NVIDIA Dynamo Meetup',
);

test('resolves Pacific wall-clock times across DST changes as RFC 5545 does', () => {
  const at = (month, day, hour, minute) =>
    zonedInstant(Date.UTC(2026, month - 1, day, hour, minute), 'America/Los_Angeles').toISOString();
  assert.equal(at(10, 27, 9, 0), '2026-10-27T16:00:00.000Z');
  assert.equal(at(11, 3, 9, 0), '2026-11-03T17:00:00.000Z');
  // 1:30 AM happens twice on Nov 1; RFC 5545 means the first, still in PDT.
  assert.equal(at(11, 1, 1, 30), '2026-11-01T08:30:00.000Z');
  // 2:30 AM never happens on Mar 8; RFC 5545 applies the offset from before the gap.
  assert.equal(at(3, 8, 2, 30), '2026-03-08T10:30:00.000Z');
});

test('expands a weekly series into every week, at 9:00 AM Pacific through the DST change', () => {
  const { upcoming, past } = buildCalendar(feed(TUESDAY_SYNC), NOW);

  // Through the end of next month, and no further.
  assert.deepEqual(
    upcoming.map((e) => e.start),
    [
      '2026-10-20T16:00:00.000Z',
      '2026-10-27T16:00:00.000Z',
      '2026-11-03T17:00:00.000Z',
      '2026-11-10T17:00:00.000Z',
      '2026-11-17T17:00:00.000Z',
      '2026-11-24T17:00:00.000Z',
    ],
  );
  assert.deepEqual(past.map((e) => e.dateLabel), ['Tue, Oct 13, 2026']);
  for (const event of [...upcoming, ...past]) {
    assert.equal(event.timeLabel, '9:00 AM');
    assert.equal(event.recurring, true);
    // The Meet link lives in X-GOOGLE-CONFERENCE, not LOCATION.
    assert.equal(event.location, 'Online');
    assert.equal(event.locationUrl, 'https://meet.google.com/aaa-bbbb-ccc');
  }
});

test('ends a series at its UTC UNTIL and skips its EXDATE weeks', () => {
  const { upcoming, past } = buildCalendar(feed(WEDNESDAY_MEETING), NOW);
  assert.deepEqual(upcoming, []);
  // From the start of last month; Sep 23 is excluded.
  assert.deepEqual(
    past.map((e) => e.dateLabel),
    ['Wed, Oct 7, 2026', 'Wed, Sep 30, 2026', 'Wed, Sep 16, 2026', 'Wed, Sep 9, 2026', 'Wed, Sep 2, 2026'],
  );

  // UNTIL is 5:00 AM Pacific on Oct 14. Read as wall-clock time (12:00) it
  // would wrongly keep that morning's 10:30 meeting, which is still running at NOW.
  const earlyUntil = weekly({
    uid: 'early-until@example.com',
    summary: 'Dynamo community meeting',
    start: '20260930T103000',
    end: '20260930T111500',
    rule: 'FREQ=WEEKLY;UNTIL=20261014T120000Z;BYDAY=WE',
  });
  assert.deepEqual(buildCalendar(feed(earlyUntil), NOW).upcoming, []);
});

test('applies RECURRENCE-ID overrides that move or cancel an occurrence', () => {
  const moved = vevent(
    'DTSTART;TZID=America/Los_Angeles:20261021T090000',
    'DTEND;TZID=America/Los_Angeles:20261021T100000',
    'RECURRENCE-ID;TZID=America/Los_Angeles:20261020T090000',
    'UID:tuesday@example.com',
    'STATUS:CONFIRMED',
    'SUMMARY:Dynamo Engineering Sync (moved)',
  );
  const cancelled = vevent(
    'DTSTART;TZID=America/Los_Angeles:20261027T090000',
    'DTEND;TZID=America/Los_Angeles:20261027T100000',
    'RECURRENCE-ID;TZID=America/Los_Angeles:20261027T090000',
    'UID:tuesday@example.com',
    'STATUS:CANCELLED',
    'SUMMARY:Dynamo Engineering Sync',
  );
  const { upcoming } = buildCalendar(feed(TUESDAY_SYNC, moved, cancelled), NOW);
  assert.deepEqual(
    upcoming.slice(0, 2).map((e) => [e.dateLabel, e.title]),
    [
      ['Wed, Oct 21, 2026', 'Dynamo Engineering Sync (moved)'],
      ['Tue, Nov 3, 2026', 'Dynamo Engineering Sync'],
    ],
  );
});

test('shows a meeting once when two series publish it at the same time', () => {
  const copy = weekly({
    uid: 'copy@example.com',
    summary: 'Dynamo community meeting',
    start: '20260923T103000',
    end: '20260923T111500',
    rule: 'FREQ=WEEKLY;UNTIL=20261014T065959Z;BYDAY=WE',
  });
  const { past } = buildCalendar(feed(WEDNESDAY_MEETING, copy), NOW);
  assert.deepEqual(
    past.map((e) => e.dateLabel),
    [
      'Wed, Oct 7, 2026',
      'Wed, Sep 30, 2026',
      'Wed, Sep 23, 2026',
      'Wed, Sep 16, 2026',
      'Wed, Sep 9, 2026',
      'Wed, Sep 2, 2026',
    ],
  );
});

test('keeps every one-off event when a weekly series fills the same weeks', () => {
  const meetups = Array.from({ length: 8 }, (_, i) =>
    vevent(
      `DTSTART:202610${String(15 + i).padStart(2, '0')}T010000Z`,
      `DTEND:202610${String(15 + i).padStart(2, '0')}T040000Z`,
      `UID:meetup-${i}@example.com`,
      'STATUS:CONFIRMED',
      `SUMMARY:Meetup ${i}`,
    ),
  );
  const { upcoming } = buildCalendar(feed(TUESDAY_SYNC, ...meetups), NOW);
  assert.equal(upcoming.filter((e) => !e.recurring).length, 8);
  assert.equal(upcoming.filter((e) => e.recurring).length, 6);
});

test('reads the meeting card from the running weekly series', () => {
  const { meeting } = buildCalendar(feed(WEDNESDAY_MEETING, TUESDAY_SYNC, MEETUP), NOW);
  assert.deepEqual(meetingCard(meeting), {
    source: 'calendar',
    title: 'Dynamo Engineering Sync',
    weekday: 'Tuesday',
    time: '9:00',
    meridiem: 'AM',
    joinUrl: 'https://meet.google.com/aaa-bbbb-ccc',
  });
});

test('prefers a meeting-like title, then the newest series', () => {
  const officeHours = weekly({
    uid: 'office-hours@example.com',
    summary: 'Office hours',
    start: '20261015T140000',
    end: '20261015T150000',
    rule: 'FREQ=WEEKLY;BYDAY=TH',
  });
  assert.equal(buildCalendar(feed(officeHours, TUESDAY_SYNC), NOW).meeting.title, 'Dynamo Engineering Sync');

  // An older series that never ended loses to the slot that replaced it.
  const unendedWednesday = weekly({
    uid: 'unended@example.com',
    summary: 'Dynamo community meeting',
    start: '20260729T103000',
    end: '20260729T111500',
    rule: 'FREQ=WEEKLY;BYDAY=WE',
  });
  assert.equal(buildCalendar(feed(unendedWednesday, TUESDAY_SYNC), NOW).meeting.title, 'Dynamo Engineering Sync');
});

test('takes the join link from the description when the series has no conference', () => {
  const zoom = weekly({
    uid: 'zoom@example.com',
    summary: 'Dynamo community meeting',
    start: '20261015T080000',
    end: '20261015T090000',
    rule: 'FREQ=WEEKLY;BYDAY=TH',
    extra: ['LOCATION:Santa Clara\\, CA', 'DESCRIPTION:Dial in: https://tel.example.com/1\\nJoin: https://zoom.us/j/123'],
  });
  const card = meetingCard(buildCalendar(feed(zoom), NOW).meeting);
  assert.equal(card.joinUrl, 'https://zoom.us/j/123');
  assert.equal(`${card.weekday} ${card.time} ${card.meridiem}`, 'Thursday 8:00 AM');
});

test('falls back to Tuesday 9:00 AM Pacific when no weekly series is running', () => {
  const biweekly = weekly({
    uid: 'biweekly@example.com',
    summary: 'Dynamo community meeting',
    start: '20261013T090000',
    end: '20261013T100000',
    rule: 'FREQ=WEEKLY;INTERVAL=2;BYDAY=TU',
  });
  for (const calendar of [feed(WEDNESDAY_MEETING, MEETUP), feed(biweekly), feed()]) {
    const { meeting } = buildCalendar(calendar, NOW);
    assert.equal(meeting, FALLBACK_MEETING);
  }
  assert.deepEqual(meetingCard(FALLBACK_MEETING), {
    source: 'fallback',
    title: 'Dynamo community meeting',
    weekday: 'Tuesday',
    time: '9:00',
    meridiem: 'AM',
    joinUrl: 'https://meet.google.com/zyn-hfqk-asm',
  });
});

test('renders an invite that calendar parsers read back as the same weekly meeting', () => {
  const { meeting } = buildCalendar(feed(TUESDAY_SYNC), NOW);
  const invite = renderInvite(meeting, null);

  assert.match(invite, /^DTSTART;TZID=America\/Los_Angeles:20261013T090000$/m);
  assert.match(invite, /^DTEND;TZID=America\/Los_Angeles:20261013T100000$/m);
  assert.match(invite, /^RRULE:FREQ=WEEKLY;BYDAY=TU$/m);
  assert.match(invite, /^DTSTAMP:20261012T150518Z$/m);
  assert.match(invite, /^SEQUENCE:0$/m);
  for (const line of invite.split('\n')) assert.ok(Buffer.byteLength(line) <= 75, line);

  const [event] = Object.values(ical.sync.parseICS(invite)).filter((e) => e.type === 'VEVENT');
  assert.equal(event.uid, 'dynamo-community-weekly@ai-dynamo.github.io');
  assert.equal(event.url, 'https://meet.google.com/aaa-bbbb-ccc');
  assert.match(event.description, /Join via Google Meet: https:\/\/meet\.google\.com\/aaa-bbbb-ccc/);
  assert.deepEqual(
    buildCalendar(ical.sync.parseICS(invite), NOW).upcoming.map((e) => `${e.dateLabel} ${e.timeLabel}`).slice(0, 3),
    ['Tue, Oct 20, 2026 9:00 AM', 'Tue, Oct 27, 2026 9:00 AM', 'Tue, Nov 3, 2026 9:00 AM'],
  );
});

test('folds the invite between words, with no trailing spaces, at any join link length', () => {
  const { meeting } = buildCalendar(feed(TUESDAY_SYNC), NOW);
  // codespell's word pattern: a word split across a fold reads as a misspelling.
  const words = (text) => text.match(/[\w\-'’]+/g) || [];
  for (let length = 20; length <= 400; length += 1) {
    const joinUrl = `https://zoom.us/j/${'1234567/'.repeat(50)}`.slice(0, length);
    const invite = renderInvite({ ...meeting, joinUrl }, null);
    const physical = invite.trimEnd().split('\n');
    for (const line of physical) {
      assert.ok(Buffer.byteLength(line) <= 75, `${length}: ${line}`);
      assert.doesNotMatch(line, /\s$/, `trailing whitespace with a ${length}-character join link`);
    }
    assert.deepEqual(physical.flatMap(words), words(invite.replace(/\n /g, '')), `split word at ${length}`);
    const [event] = Object.values(ical.sync.parseICS(invite)).filter((e) => e.type === 'VEVENT');
    assert.equal(event.url, joinUrl);
  }
});

test('bumps the invite SEQUENCE only when the meeting changes', () => {
  const { meeting } = buildCalendar(feed(TUESDAY_SYNC), NOW);
  const first = renderInvite(meeting, null);
  assert.equal(renderInvite(meeting, first), first);

  const moved = { ...meeting, startWall: Date.UTC(2026, 9, 14, 10, 30), stamp: new Date('2026-10-14T00:00:00Z') };
  const second = renderInvite(moved, first);
  assert.match(second, /^SEQUENCE:1$/m);
  assert.match(second, /^RRULE:FREQ=WEEKLY;BYDAY=WE$/m);
  assert.equal(renderInvite(moved, second), second);

  // A new DTSTAMP alone is bookkeeping, not a revision.
  assert.match(renderInvite({ ...moved, stamp: new Date('2026-10-15T00:00:00Z') }, second), /^SEQUENCE:1$/m);
});

test('carries a series end date into the invite', () => {
  const ending = weekly({
    uid: 'ending@example.com',
    summary: 'Dynamo community meeting',
    start: '20261013T090000',
    end: '20261013T100000',
    rule: 'FREQ=WEEKLY;UNTIL=20261216T075959Z;BYDAY=TU',
    extra: ['X-GOOGLE-CONFERENCE:https://meet.google.com/aaa-bbbb-ccc'],
  });
  const { meeting } = buildCalendar(feed(ending), NOW);
  assert.match(renderInvite(meeting, null), /^RRULE:FREQ=WEEKLY;BYDAY=TU;UNTIL=20261216T075959Z$/m);
});
