/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EventsCalendar — macOS-inspired preview of the public Dynamo calendar.
 *
 * The event data is generated hourly from the public Google Calendar by
 * .github/scripts/generate-events.js. This component stays server-rendered and
 * uses CSS classes from main.css so it works without client JavaScript.
 */

import {
  UPCOMING_EVENTS,
  PAST_EVENTS,
  type DynamoEvent,
} from "./events.generated";

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

const MONTH_INDEX = Object.fromEntries(
  MONTHS.map((month, index) => [month.slice(0, 3), index]),
);

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
  const focusEvent = UPCOMING_EVENTS[0] ?? PAST_EVENTS[0];
  const year = Number(focusEvent?.year ?? new Date().getUTCFullYear());
  const month = MONTH_INDEX[focusEvent?.month ?? "Jan"] ?? 0;
  const selectedDay = Number(focusEvent?.day ?? 1);
  const days = buildMonthDays(year, month);

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
          <span aria-hidden="true">↗</span>
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
                  className={day === selectedDay ? "is-selected" : undefined}
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
