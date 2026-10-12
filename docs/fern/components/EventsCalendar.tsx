/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EventsCalendar — macOS-inspired preview of the public Dynamo calendar.
 *
 * The event data is fetched from the public Google Calendar during Fern
 * publishing and on a six-hour schedule. This component stays server-rendered and
 * uses CSS classes from main.css so it works without client JavaScript.
 */

import {
  UPCOMING_EVENTS,
  PAST_EVENTS,
  type DynamoEvent,
} from "./events.generated";
import {
  MONTH_INDEX,
  resolveCalendarMonth,
  resolveToday,
} from "./calendar-today";

const CALENDAR_URL =
  "https://calendar.google.com/calendar/u/0/r?cid=Y19jMjQ0OGQyZWZiMDllYWMyZGRlZTFmMzQ1MjQxMjQxMzViZDNmNDU1NDg2ODc2OTA1OTEwNWUxOGUxYjk3ZThmQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20";

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

function buildMonthDays(year: number, month: number) {
  const leadingBlanks = new Date(Date.UTC(year, month, 1)).getUTCDay();
  const dayCount = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  return [
    ...Array.from({ length: leadingBlanks }, () => null),
    ...Array.from({ length: dayCount }, (_, index) => index + 1),
  ];
}

function EventLocation({ event }: { event: DynamoEvent }) {
  if (!event.location) return null;

  return event.locationUrl ? (
    <a
      className="dynamo-calendar__location"
      href={event.locationUrl}
      target="_blank"
      rel="noreferrer"
    >
      {event.location}
    </a>
  ) : (
    <span className="dynamo-calendar__location">{event.location}</span>
  );
}

function UpcomingEvent({ event }: { event: DynamoEvent }) {
  return (
    <article className="dynamo-calendar__event">
      <div className="dynamo-calendar__event-date" aria-hidden="true">
        <span>{event.month}</span>
        <strong>{event.day}</strong>
      </div>
      <div className="dynamo-calendar__event-copy">
        <p>Upcoming event</p>
        <h3>
          <a href={event.addUrl} target="_blank" rel="noreferrer">
            {event.title}
          </a>
        </h3>
        <div className="dynamo-calendar__event-meta">
          <span>{event.dateLabel}</span>
          <EventLocation event={event} />
        </div>
      </div>
      <a
        className="dynamo-calendar__event-action"
        href={event.addUrl}
        target="_blank"
        rel="noreferrer"
        aria-label={`Add ${event.title} to Google Calendar`}
      >
        <span aria-hidden="true">+</span>
      </a>
    </article>
  );
}

export function EventsCalendar() {
  // Today's month when it has events, else the nearest event's month. Keying
  // the grid off UPCOMING_EVENTS[0] meant an empty calendar fell back to
  // PAST_EVENTS[0] and sat on a month that had already gone by; keying it hard
  // to today's month instead would empty the grid whenever the next event is
  // in another month, which is the ordinary case.
  const { year, month } = resolveCalendarMonth();
  const today = resolveToday();
  // Mark today only when the grid is actually showing today's month, so the
  // highlight can never land on the same-numbered day of some other month.
  const selectedDay =
    today && today.year === year && today.month === month ? today.day : null;
  const days = buildMonthDays(year, month);

  // Days in the displayed month that have something scheduled, so the grid
  // still carries event information now that the highlight marks today.
  const eventDays = new Set(
    [...UPCOMING_EVENTS, ...PAST_EVENTS]
      .filter(
        (event) =>
          Number(event.year) === year && MONTH_INDEX[event.month] === month,
      )
      .map((event) => Number(event.day)),
  );

  return (
    <section
      className="dynamo-calendar"
      aria-labelledby="dynamo-calendar-title"
    >
      <div className="dynamo-calendar__chrome">
        <span />
        <span />
        <span />
        <p>Community Calendar</p>
        <a href={CALENDAR_URL} target="_blank" rel="noreferrer">
          Open Google Calendar
          {/* Font Awesome Free: arrow-up-right-from-square (CC BY 4.0). */}
          <svg viewBox="0 0 512 512" aria-hidden="true">
            <path d="M320 32c0-17.7 14.3-32 32-32H480c17.7 0 32 14.3 32 32V160c0 17.7-14.3 32-32 32s-32-14.3-32-32V109.3L246.6 310.6c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L402.7 64H352c-17.7 0-32-14.3-32-32zM80 32H192c17.7 0 32 14.3 32 32S209.7 96 192 96H80c-8.8 0-16 7.2-16 16V432c0 8.8 7.2 16 16 16H400c8.8 0 16-7.2 16-16V320c0-17.7 14.3-32 32-32s32 14.3 32 32V432c0 44.2-35.8 80-80 80H80c-44.2 0-80-35.8-80-80V112c0-44.2 35.8-80 80-80z" />
          </svg>
        </a>
      </div>

      <div className="dynamo-calendar__body">
        <aside
          className="dynamo-calendar__sidebar"
          aria-label="Calendar month preview"
        >
          <div className="dynamo-calendar__month-heading">
            <strong>{MONTHS[month]}</strong>
            <span>{year}</span>
          </div>
          <div className="dynamo-calendar__weekdays" aria-hidden="true">
            {["S", "M", "T", "W", "T", "F", "S"].map((day, index) => (
              <span key={`${day}-${index}`}>{day}</span>
            ))}
          </div>
          <div className="dynamo-calendar__month-grid">
            {days.map((day, index) =>
              day === null ? (
                <span key={`blank-${index}`} />
              ) : (
                <span
                  key={day}
                  className={
                    [
                      day === selectedDay ? "is-selected" : "",
                      eventDays.has(day) ? "has-event" : "",
                    ]
                      .filter(Boolean)
                      .join(" ") || undefined
                  }
                  aria-current={day === selectedDay ? "date" : undefined}
                  aria-label={
                    eventDays.has(day)
                      ? `${MONTHS[month]} ${day}, has an event`
                      : undefined
                  }
                >
                  {day}
                </span>
              ),
            )}
          </div>
          <div className="dynamo-calendar__source">
            <span className="dynamo-calendar__source-dot" />
            Dynamo community
          </div>
        </aside>

        <div className="dynamo-calendar__agenda">
          <div className="dynamo-calendar__intro">
            <p>Meetups, talks, and community meetings</p>
            <h2 id="dynamo-calendar-title">Community events</h2>
            <span>Previewed from the public Dynamo Google Calendar.</span>
          </div>

          <div className="dynamo-calendar__events">
            {UPCOMING_EVENTS.length > 0 ? (
              UPCOMING_EVENTS.slice(0, 4).map((event) => (
                <UpcomingEvent
                  key={`${event.start}-${event.title}`}
                  event={event}
                />
              ))
            ) : (
              <div className="dynamo-calendar__empty">
                <div className="dynamo-calendar__empty-icon" aria-hidden="true">
                  CAL
                </div>
                <p>No upcoming events are scheduled.</p>
                <a href={CALENDAR_URL} target="_blank" rel="noreferrer">
                  Check the public calendar
                </a>
              </div>
            )}
          </div>

          {PAST_EVENTS.length > 0 && (
            <details className="dynamo-calendar__past">
              <summary>Recent events</summary>
              <div>
                {PAST_EVENTS.slice(0, 5).map((event) => (
                  <a
                    key={`${event.start}-${event.title}`}
                    href={event.addUrl}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <span>{event.dateLabel}</span>
                    {event.title}
                  </a>
                ))}
              </div>
            </details>
          )}
        </div>
      </div>
    </section>
  );
}
