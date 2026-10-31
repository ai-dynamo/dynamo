/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The weekly community meeting, and its occurrences for the calendar grids.
 *
 * The calendar generator emits a recurring series only once, at its first
 * date, so the grids and agenda lists add the weekly meeting from here. When
 * the meeting moves, update COMMUNITY_MEETING and
 * docs/fern/assets/dynamo-community-meeting.ics together.
 */
import { GENERATED_ON, type DynamoEvent } from "./events.generated";

const PACIFIC = "America/Los_Angeles";
const WEEKDAYS = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
] as const;

export const COMMUNITY_MEETING = {
  /** Title of the series on the public Dynamo Google Calendar. */
  title: "Dynamo Engineering Sync",
  /** Day of the week, 0 = Sunday. */
  weekday: 2,
  /** Start time in Pacific, 24-hour. */
  hour: 9,
  minute: 0,
  durationMinutes: 60,
  /** First meeting on this schedule. Zero-based month. */
  firstMeeting: { year: 2026, month: 9, day: 13 },
  joinUrl: "https://meet.google.com/zyn-hfqk-asm",
  notesUrl:
    "https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view",
} as const;

/** The meeting's schedule as the cadence card shows it: "Tuesday", "9:00", "AM". */
export const MEETING_SCHEDULE = {
  weekday: WEEKDAYS[COMMUNITY_MEETING.weekday],
  time: `${COMMUNITY_MEETING.hour % 12 || 12}:${String(COMMUNITY_MEETING.minute).padStart(2, "0")}`,
  meridiem: COMMUNITY_MEETING.hour < 12 ? "AM" : "PM",
};

const DAY_MS = 24 * 60 * 60 * 1000;

/** The instant the meeting starts on a Pacific calendar date, across DST. */
function meetingStart(year: number, month: number, day: number): Date {
  const { hour, minute } = COMMUNITY_MEETING;
  const pacificHour = new Intl.DateTimeFormat("en-US", {
    timeZone: PACIFIC,
    hour: "numeric",
    hourCycle: "h23",
  });
  // Pacific is UTC-7 in daylight time and UTC-8 in standard time.
  for (const offset of [7, 8]) {
    const start = new Date(Date.UTC(year, month, day, hour + offset, minute));
    if (Number(pacificHour.format(start)) === hour) return start;
  }
  return new Date(Date.UTC(year, month, day, hour + 8, minute));
}

function pacific(date: Date, options: Intl.DateTimeFormatOptions): string {
  return date.toLocaleString("en-US", { timeZone: PACIFIC, ...options });
}

const compactUtc = (date: Date) =>
  date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");

/** "Today" as a UTC-midnight timestamp of GENERATED_ON, the Pacific date of the last refresh. */
function generatedOnMs(): number | null {
  const [year, month, day] = GENERATED_ON.split("-").map(Number);
  return year && month && day ? Date.UTC(year, month - 1, day) : null;
}

function occurrence(year: number, month: number, day: number): DynamoEvent {
  const start = meetingStart(year, month, day);
  const end = new Date(start.getTime() + COMMUNITY_MEETING.durationMinutes * 60 * 1000);
  const today = generatedOnMs();
  const params = new URLSearchParams({
    action: "TEMPLATE",
    text: COMMUNITY_MEETING.title,
    dates: `${compactUtc(start)}/${compactUtc(end)}`,
    ctz: PACIFIC,
    recur: `RRULE:FREQ=WEEKLY;BYDAY=${WEEKDAYS[COMMUNITY_MEETING.weekday].slice(0, 2).toUpperCase()}`,
    location: COMMUNITY_MEETING.joinUrl,
    details: `Weekly NVIDIA Dynamo community meeting.\n\nJoin via Google Meet: ${COMMUNITY_MEETING.joinUrl}\n\nMeeting notes: ${COMMUNITY_MEETING.notesUrl}`,
  });
  return {
    title: COMMUNITY_MEETING.title,
    start: start.toISOString(),
    month: pacific(start, { month: "short" }),
    day: String(day),
    year: String(year),
    dateLabel: pacific(start, {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    timeLabel: pacific(start, { hour: "numeric", minute: "2-digit" }),
    isPast: today !== null && Date.UTC(year, month, day) < today,
    location: "Online",
    locationUrl: COMMUNITY_MEETING.joinUrl,
    addUrl: `https://calendar.google.com/calendar/render?${params.toString()}`,
  };
}

/** Every meeting in a month (zero-based), from the first meeting on this schedule onward. */
export function meetingsInMonth(year: number, month: number): DynamoEvent[] {
  const { firstMeeting, weekday } = COMMUNITY_MEETING;
  const first = Date.UTC(firstMeeting.year, firstMeeting.month, firstMeeting.day);
  const meetings: DynamoEvent[] = [];
  const dayCount = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  for (let day = 1; day <= dayCount; day += 1) {
    const date = Date.UTC(year, month, day);
    if (date >= first && new Date(date).getUTCDay() === weekday) {
      meetings.push(occurrence(year, month, day));
    }
  }
  return meetings;
}

/** The next meeting on or after GENERATED_ON. */
export function nextMeeting(): DynamoEvent {
  const { firstMeeting, weekday } = COMMUNITY_MEETING;
  const first = Date.UTC(firstMeeting.year, firstMeeting.month, firstMeeting.day);
  let date = Math.max(generatedOnMs() ?? first, first);
  while (new Date(date).getUTCDay() !== weekday) date += DAY_MS;
  const day = new Date(date);
  return occurrence(day.getUTCFullYear(), day.getUTCMonth(), day.getUTCDate());
}

/** `events` plus `meetings`, without the calendar's own copy of a meeting, soonest first. */
function merge(events: DynamoEvent[], meetings: DynamoEvent[]): DynamoEvent[] {
  const listed = new Set(meetings.map((m) => `${m.title}@${Date.parse(m.start)}`));
  return [
    ...events.filter((e) => !listed.has(`${e.title}@${Date.parse(e.start)}`)),
    ...meetings,
  ].sort((a, b) => Date.parse(a.start) - Date.parse(b.start));
}

/** The calendar's events in a month (zero-based), with the weekly meeting on every week. */
export function eventsInMonth(
  events: DynamoEvent[],
  year: number,
  month: number,
  monthIndex: Record<string, number>,
): DynamoEvent[] {
  return merge(
    events.filter((e) => Number(e.year) === year && monthIndex[e.month] === month),
    meetingsInMonth(year, month),
  );
}

/** UPCOMING_EVENTS with the next weekly meeting in its place. */
export function upcomingWithMeeting(upcoming: DynamoEvent[]): DynamoEvent[] {
  return merge(upcoming, [nextMeeting()]);
}
