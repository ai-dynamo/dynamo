/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  UPCOMING_EVENTS,
  PAST_EVENTS,
  type DynamoEvent,
} from "./events.generated";

const CALENDAR_URL =
  "https://calendar.google.com/calendar/u/0/r?cid=Y19jMjQ0OGQyZWZiMDllYWMyZGRlZTFmMzQ1MjQxMjQxMzViZDNmNDU1NDg2ODc2OTA1OTEwNWUxOGUxYjk3ZThmQGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20";
const MEETING_URL = "https://meet.google.com/heb-demu-qok";
const NOTES_URL =
  "https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view";
const INVITE_URL =
  "https://github.com/ai-dynamo/dynamo/blob/main/docs/assets/dynamo-community-meeting.ics";

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
] as const;

const MONTH_INDEX = Object.fromEntries(
  MONTHS.map((month, index) => [month.slice(0, 3), index]),
);

const CHANNELS = [
  {
    name: "GitHub",
    label: "Source, issues, and pull requests",
    href: "https://github.com/ai-dynamo/dynamo",
    tone: "github",
    glyph: "GH",
  },
  {
    name: "Discussions",
    label: "Questions, ideas, and project help",
    href: "https://github.com/ai-dynamo/dynamo/discussions",
    tone: "discussions",
    glyph: "✦",
  },
  {
    name: "Discord",
    label: "Chat with builders in real time",
    href: "https://discord.gg/D92uqZRjCZ",
    tone: "discord",
    glyph: "D",
  },
  {
    name: "CNCF Slack",
    label: "Find us in #ai-dynamo",
    href: "https://communityinviter.com/apps/cloud-native/cncf",
    tone: "slack",
    glyph: "#",
  },
  {
    name: "YouTube",
    label: "Meetings, talks, and demos",
    href: "https://www.youtube.com/@ai-dynamo-community",
    tone: "youtube",
    glyph: "▶",
  },
  {
    name: "Calendar",
    label: "Meetups and community calls",
    href: CALENDAR_URL,
    tone: "calendar",
    glyph: "22",
  },
] as const;

const PARTICIPATION = [
  {
    index: "01",
    title: "Ask and answer",
    copy: "Bring deployment questions, share what worked, and help other builders move forward.",
    href: "https://github.com/ai-dynamo/dynamo/discussions",
    cta: "Open Discussions",
  },
  {
    index: "02",
    title: "Contribute",
    copy: "Improve code, docs, examples, integrations, and the day-to-day developer experience.",
    href: "/dynamo/dev/contributing/contribution-guide",
    cta: "Read the contribution guide",
  },
  {
    index: "03",
    title: "Shape the roadmap",
    copy: "Propose substantial changes through Dynamo Enhancement Proposals and review ideas in progress.",
    href: "https://github.com/ai-dynamo/enhancements",
    cta: "Explore enhancements",
  },
  {
    index: "04",
    title: "Share your work",
    copy: "Show the community what you are building, measuring, or learning with Dynamo.",
    href: NOTES_URL,
    cta: "Add a meeting topic",
  },
] as const;

function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      {diagonal ? (
        <path d="M6 14 14 6M8 6h6v6" />
      ) : (
        <path d="M4 10h11M11 5l5 5-5 5" />
      )}
    </svg>
  );
}

function buildMonthCells(year: number, month: number) {
  const leading = new Date(Date.UTC(year, month, 1)).getUTCDay();
  const count = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
  const cells: Array<number | null> = [
    ...Array.from({ length: leading }, () => null),
    ...Array.from({ length: count }, (_, index) => index + 1),
  ];

  while (cells.length % 7 !== 0 || cells.length < 42) cells.push(null);
  return cells;
}

function eventDay(event: DynamoEvent) {
  return Number(event.day);
}

function FullCalendar() {
  const focusEvent = UPCOMING_EVENTS[0] ?? PAST_EVENTS[0];
  const year = Number(focusEvent?.year ?? new Date().getUTCFullYear());
  const month = MONTH_INDEX[focusEvent?.month ?? "Jan"] ?? 0;
  const events = [...UPCOMING_EVENTS, ...PAST_EVENTS].filter(
    (event) => Number(event.year) === year && MONTH_INDEX[event.month] === month,
  );
  const eventsByDay = new Map<number, DynamoEvent[]>();

  events.forEach((event) => {
    const day = eventDay(event);
    eventsByDay.set(day, [...(eventsByDay.get(day) ?? []), event]);
  });

  return (
    <section className="dynamo-community-calendar" aria-labelledby="community-calendar-title">
      <div className="dynamo-community-calendar__toolbar">
        <div>
          <p className="dynamo-community-kicker">Community calendar</p>
          <h2 id="community-calendar-title">See where the community meets next</h2>
        </div>
        <a href={CALENDAR_URL} target="_blank" rel="noreferrer">
          Open calendar <Arrow diagonal />
        </a>
      </div>

      <div className="dynamo-community-calendar__window">
        <div className="dynamo-community-calendar__chrome" aria-hidden="true">
          <span /><span /><span />
          <strong>{MONTHS[month]} {year}</strong>
          <i />
        </div>
        <div className="dynamo-community-calendar__weekdays" aria-hidden="true">
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => <span key={day}>{day}</span>)}
        </div>
        <div className="dynamo-community-calendar__grid">
          {buildMonthCells(year, month).map((day, index) => {
            const dayEvents = day === null ? [] : eventsByDay.get(day) ?? [];
            return (
              <div
                className={`dynamo-community-calendar__day${day === null ? ' is-empty' : ''}${dayEvents.length ? ' has-event' : ''}`}
                key={`${day ?? 'empty'}-${index}`}
              >
                {day !== null && <span className="dynamo-community-calendar__number">{day}</span>}
                {dayEvents.slice(0, 2).map((event) => (
                  <a
                    href={event.addUrl}
                    target="_blank"
                    rel="noreferrer"
                    key={`${event.start}-${event.title}`}
                    title={`${event.title} — ${event.dateLabel}`}
                  >
                    <span />{event.title}
                  </a>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      <div className="dynamo-community-calendar__upcoming">
        <p>Up next</p>
        <div>
          {UPCOMING_EVENTS.length > 0 ? UPCOMING_EVENTS.slice(0, 3).map((event) => (
            <a href={event.addUrl} target="_blank" rel="noreferrer" key={`${event.start}-${event.title}`}>
              <span><strong>{event.month}</strong>{event.day}</span>
              <span><strong>{event.title}</strong><small>{event.dateLabel}{event.location ? ` · ${event.location}` : ''}</small></span>
              <Arrow diagonal />
            </a>
          )) : (
            <a href={CALENDAR_URL} target="_blank" rel="noreferrer">
              <span><strong>NEW</strong>+</span>
              <span><strong>More events are on the way</strong><small>Follow the public calendar for updates.</small></span>
              <Arrow diagonal />
            </a>
          )}
        </div>
      </div>
    </section>
  );
}

export function CommunityLanding() {
  return (
    <div className="dynamo-community-page">
      <section className="dynamo-community-hero">
        <div className="dynamo-community-hero__copy">
          <p className="dynamo-community-kicker"><span /> Open source, in the open</p>
          <h1>Build the future of inference, together.</h1>
          <p className="dynamo-community-hero__lede">
            Meet the people building Dynamo, trade hard-won lessons, and help shape an open inference platform.
          </p>
          <div className="dynamo-community-actions">
            <a className="is-primary" href="https://discord.gg/D92uqZRjCZ" target="_blank" rel="noreferrer">
              Join the community <Arrow />
            </a>
            <a href="https://github.com/ai-dynamo/dynamo" target="_blank" rel="noreferrer">
              Explore on GitHub <Arrow diagonal />
            </a>
          </div>
        </div>

        <div className="dynamo-community-launcher" aria-label="Community channels">
          <div className="dynamo-community-launcher__bar">
            <span /><span /><span />
            <p>Community</p>
          </div>
          <div className="dynamo-community-launcher__grid">
            {CHANNELS.map((channel) => (
              <a href={channel.href} target="_blank" rel="noreferrer" key={channel.name}>
                <span className={`dynamo-community-app dynamo-community-app--${channel.tone}`}>{channel.glyph}</span>
                <strong>{channel.name}</strong>
                <small>{channel.label}</small>
              </a>
            ))}
          </div>
        </div>
      </section>

      <div className="dynamo-community-marquee" aria-label="Ways to participate">
        <div>
          <span>Ask questions</span><i />
          <span>Review ideas</span><i />
          <span>Share benchmarks</span><i />
          <span>Build integrations</span><i />
          <span>Improve docs</span><i />
          <span>Meet maintainers</span><i />
          <span aria-hidden="true">Ask questions</span><i aria-hidden="true" />
          <span aria-hidden="true">Review ideas</span><i aria-hidden="true" />
          <span aria-hidden="true">Share benchmarks</span><i aria-hidden="true" />
        </div>
      </div>

      <FullCalendar />

      <section className="dynamo-community-participate" aria-labelledby="participate-title">
        <div className="dynamo-community-section-heading">
          <p className="dynamo-community-kicker">Choose your path in</p>
          <h2 id="participate-title">There is more than one way to contribute</h2>
          <p>Start with the part of the project and community that matches how you want to help.</p>
        </div>
        <div className="dynamo-community-participate__grid">
          {PARTICIPATION.map((item) => (
            <a href={item.href} target={item.href.startsWith('http') ? '_blank' : undefined} rel={item.href.startsWith('http') ? 'noreferrer' : undefined} key={item.index}>
              <span>{item.index}</span>
              <h3>{item.title}</h3>
              <p>{item.copy}</p>
              <strong>{item.cta} <Arrow /></strong>
            </a>
          ))}
        </div>
      </section>

      <section className="dynamo-community-meeting" aria-labelledby="meeting-title">
        <div className="dynamo-community-meeting__date" aria-hidden="true">
          <span>WED</span>
          <strong>10:30</strong>
          <small>AM PT</small>
        </div>
        <div className="dynamo-community-meeting__copy">
          <p className="dynamo-community-kicker">Weekly community meeting</p>
          <h2 id="meeting-title">Bring a question. Leave with context.</h2>
          <p>
            Join maintainers and contributors every Wednesday for project updates, design discussions, demos, and open Q&amp;A.
          </p>
          <div className="dynamo-community-meeting__links">
            <a href={MEETING_URL} target="_blank" rel="noreferrer">Join Google Meet <Arrow diagonal /></a>
            <a href={NOTES_URL} target="_blank" rel="noreferrer">Agenda and notes <Arrow diagonal /></a>
            <a href="https://www.youtube.com/@ai-dynamo-community" target="_blank" rel="noreferrer">Watch recordings <Arrow diagonal /></a>
            <a href={INVITE_URL} target="_blank" rel="noreferrer">Download .ics <Arrow diagonal /></a>
          </div>
        </div>
      </section>

      <section className="dynamo-community-cncf">
        <div className="dynamo-community-cncf__mark" aria-hidden="true">CNCF</div>
        <div>
          <p className="dynamo-community-kicker">Part of the cloud native ecosystem</p>
          <h2>Open source works best when the doors stay open.</h2>
          <p>
            Dynamo participates in the Cloud Native Computing Foundation ecosystem. Join the conversation, learn in public, and help build durable infrastructure for generative AI.
          </p>
        </div>
        <a href="https://www.cncf.io/about/join/" target="_blank" rel="noreferrer">Learn about CNCF <Arrow diagonal /></a>
      </section>
    </div>
  );
}

export default CommunityLanding;
