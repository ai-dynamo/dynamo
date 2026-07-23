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
    icon: "github",
  },
  {
    name: "Discussions",
    label: "Questions, ideas, and project help",
    href: "https://github.com/ai-dynamo/dynamo/discussions",
    tone: "discussions",
    icon: "comments",
  },
  {
    name: "Discord",
    label: "Chat with builders in real time",
    href: "https://discord.gg/D92uqZRjCZ",
    tone: "discord",
    icon: "discord",
  },
  {
    name: "CNCF Slack",
    label: "Find us in #ai-dynamo",
    href: "https://communityinviter.com/apps/cloud-native/cncf",
    tone: "slack",
    icon: "slack",
  },
  {
    name: "YouTube",
    label: "Meetings, talks, and demos",
    href: "https://www.youtube.com/@ai-dynamo-community",
    tone: "youtube",
    icon: "youtube",
  },
  {
    name: "Calendar",
    label: "Meetups and community calls",
    href: CALENDAR_URL,
    tone: "calendar",
    icon: "calendar",
  },
] as const;

// Font Awesome Free icons (CC BY 4.0), Copyright Fonticons, Inc.
const FONT_AWESOME_ICONS = {
  github: {
    viewBox: "0 0 512 512",
    path: "M216.5 362.5c-66-8-112.5-55.5-112.5-117 0-25 9-52 24-70-6.5-16.5-5.5-51.5 2-66 20-2.5 47 8 63 22.5 19-6 39-9 63.5-9s44.5 3 62.5 8.5c15.5-14 43-24.5 63-22 7 13.5 8 48.5 1.5 65.5 16 19 24.5 44.5 24.5 70.5 0 61.5-46.5 108-113.5 116.5 17 11 28.5 35 28.5 62.5l0 52C323 491.5 335.5 500 350.5 494 441 459.5 512 369 512 257 512 115.5 397 0 255.5 0S0 115.5 0 257c0 111 70.5 203 165.5 237.5 13.5 5 26.5-4 26.5-17.5l0-40c-7 3-16 5-24 5-33 0-52.5-18-66.5-51.5-5.5-13.5-11.5-21.5-23-23-6-.5-8-3-8-6 0-6 10-10.5 20-10.5 14.5 0 27 9 40 27.5 10 14.5 20.5 21 33 21s20.5-4.5 32-16c8.5-8.5 15-16 21-21z",
  },
  comments: {
    viewBox: "0 0 576 512",
    path: "M384 144c0 97.2-86 176-192 176-26.7 0-52.1-5-75.2-14L35.2 349.2c-9.3 4.9-20.7 3.2-28.2-4.2s-9.2-18.9-4.2-28.2l35.6-67.2C14.3 220.2 0 183.6 0 144 0 46.8 86-32 192-32S384 46.8 384 144zm0 368c-94.1 0-172.4-62.1-188.8-144 120-1.5 224.3-86.9 235.8-202.7 83.3 19.2 145 88.3 145 170.7 0 39.6-14.3 76.2-38.4 105.6l35.6 67.2c4.9 9.3 3.2 20.7-4.2 28.2s-18.9 9.2-28.2 4.2L459.2 498c-23.1 9-48.5 14-75.2 14z",
  },
  discord: {
    viewBox: "0 0 576 512",
    path: "M492.5 69.8c-.2-.3-.4-.6-.8-.7-38.1-17.5-78.4-30-119.7-37.1-.4-.1-.8 0-1.1 .1s-.6 .4-.8 .8c-5.5 9.9-10.5 20.2-14.9 30.6-44.6-6.8-89.9-6.8-134.4 0-4.5-10.5-9.5-20.7-15.1-30.6-.2-.3-.5-.6-.8-.8s-.7-.2-1.1-.2c-41.3 7.1-81.6 19.6-119.7 37.1-.3 .1-.6 .4-.8 .7-76.2 113.8-97.1 224.9-86.9 334.5 0 .3 .1 .5 .2 .8s.3 .4 .5 .6c44.4 32.9 94 58 146.8 74.2 .4 .1 .8 .1 1.1 0s.7-.4 .9-.7c11.3-15.4 21.4-31.8 30-48.8 .1-.2 .2-.5 .2-.8s0-.5-.1-.8-.2-.5-.4-.6-.4-.3-.7-.4c-15.8-6.1-31.2-13.4-45.9-21.9-.3-.2-.5-.4-.7-.6s-.3-.6-.3-.9 0-.6 .2-.9 .3-.5 .6-.7c3.1-2.3 6.2-4.7 9.1-7.1 .3-.2 .6-.4 .9-.4s.7 0 1 .1c96.2 43.9 200.4 43.9 295.5 0 .3-.1 .7-.2 1-.2s.7 .2 .9 .4c2.9 2.4 6 4.9 9.1 7.2 .2 .2 .4 .4 .6 .7s.2 .6 .2 .9-.1 .6-.3 .9-.4 .5-.6 .6c-14.7 8.6-30 15.9-45.9 21.8-.2 .1-.5 .2-.7 .4s-.3 .4-.4 .7-.1 .5-.1 .8 .1 .5 .2 .8c8.8 17 18.8 33.3 30 48.8 .2 .3 .6 .6 .9 .7s.8 .1 1.1 0c52.9-16.2 102.6-41.3 147.1-74.2 .2-.2 .4-.4 .5-.6s.2-.5 .2-.8c12.3-126.8-20.5-236.9-86.9-334.5zm-302 267.7c-29 0-52.8-26.6-52.8-59.2s23.4-59.2 52.8-59.2c29.7 0 53.3 26.8 52.8 59.2 0 32.7-23.4 59.2-52.8 59.2zm195.4 0c-29 0-52.8-26.6-52.8-59.2s23.4-59.2 52.8-59.2c29.7 0 53.3 26.8 52.8 59.2 0 32.7-23.2 59.2-52.8 59.2z",
  },
  slack: {
    viewBox: "0 0 448 512",
    path: "M94.1 315.1c0 25.9-21.2 47.1-47.1 47.1S0 341 0 315.1 21.2 268 47.1 268l47.1 0 0 47.1zm23.7 0c0-25.9 21.2-47.1 47.1-47.1S212 289.2 212 315.1l0 117.8c0 25.9-21.2 47.1-47.1 47.1s-47.1-21.2-47.1-47.1l0-117.8zm47.1-189c-25.9 0-47.1-21.2-47.1-47.1S139 32 164.9 32 212 53.2 212 79.1l0 47.1-47.1 0zm0 23.7c25.9 0 47.1 21.2 47.1 47.1S190.8 244 164.9 244L47.1 244C21.2 244 0 222.8 0 196.9s21.2-47.1 47.1-47.1l117.8 0zm189 47.1c0-25.9 21.2-47.1 47.1-47.1S448 171 448 196.9 426.8 244 400.9 244l-47.1 0 0-47.1zm-23.7 0c0 25.9-21.2 47.1-47.1 47.1S236 222.8 236 196.9l0-117.8C236 53.2 257.2 32 283.1 32s47.1 21.2 47.1 47.1l0 117.8zm-47.1 189c25.9 0 47.1 21.2 47.1 47.1S309 480 283.1 480 236 458.8 236 432.9l0-47.1 47.1 0zm0-23.7c-25.9 0-47.1-21.2-47.1-47.1S257.2 268 283.1 268l117.8 0c25.9 0 47.1 21.2 47.1 47.1s-21.2 47.1-47.1 47.1l-117.8 0z",
  },
  youtube: {
    viewBox: "0 0 576 512",
    path: "M549.7 124.1C543.5 100.4 524.9 81.8 501.4 75.5 458.9 64 288.1 64 288.1 64S117.3 64 74.7 75.5C51.2 81.8 32.7 100.4 26.4 124.1 15 167 15 256.4 15 256.4s0 89.4 11.4 132.3c6.3 23.6 24.8 41.5 48.3 47.8 42.6 11.5 213.4 11.5 213.4 11.5s170.8 0 213.4-11.5c23.5-6.3 42-24.2 48.3-47.8 11.4-42.9 11.4-132.3 11.4-132.3s0-89.4-11.4-132.3zM232.2 337.6l0-162.4 142.7 81.2-142.7 81.2z",
  },
  calendar: {
    viewBox: "0 0 448 512",
    path: "M128 0c17.7 0 32 14.3 32 32l0 32 128 0 0-32c0-17.7 14.3-32 32-32s32 14.3 32 32l0 32 32 0c35.3 0 64 28.7 64 64l0 288c0 35.3-28.7 64-64 64L64 480c-35.3 0-64-28.7-64-64L0 128C0 92.7 28.7 64 64 64l32 0 0-32c0-17.7 14.3-32 32-32zM64 240l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16zm128 0l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16zm144-16c-8.8 0-16 7.2-16 16l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0zM64 368l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16zm144-16c-8.8 0-16 7.2-16 16l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0zm112 16l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16z",
  },
} as const;

function FontAwesomeIcon({ name }: { name: keyof typeof FONT_AWESOME_ICONS }) {
  const icon = FONT_AWESOME_ICONS[name];
  return (
    <svg viewBox={icon.viewBox} aria-hidden="true">
      <path d={icon.path} />
    </svg>
  );
}

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

  while (cells.length % 7 !== 0) cells.push(null);
  return cells;
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
    const day = Number(event.day);
    eventsByDay.set(day, [...(eventsByDay.get(day) ?? []), event]);
  });

  return (
    <section className="dynamo-community-calendar" aria-labelledby="community-calendar-title">
      <div className="dynamo-community-section-heading">
        <div>
          <p className="dynamo-community-eyebrow">Community calendar</p>
          <h2 id="community-calendar-title">Upcoming events</h2>
          <p>Meetups, community calls, and other opportunities to connect with Dynamo contributors.</p>
        </div>
        <a className="dynamo-community-button" href={CALENDAR_URL} target="_blank" rel="noreferrer">
          Open full calendar <Arrow diagonal />
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
              <span className="dynamo-community-event-date"><strong>{event.month}</strong>{event.day}</span>
              <span className="dynamo-community-event-copy"><strong>{event.title}</strong><small>{event.dateLabel}{event.location ? ` · ${event.location}` : ''}</small></span>
              <Arrow diagonal />
            </a>
          )) : (
            <a href={CALENDAR_URL} target="_blank" rel="noreferrer">
              <span className="dynamo-community-event-date"><strong>New</strong>+</span>
              <span className="dynamo-community-event-copy"><strong>More events are on the way</strong><small>Follow the public calendar for updates.</small></span>
              <Arrow diagonal />
            </a>
          )}
        </div>
      </div>
    </section>
  );
}

function CommunityChannels() {
  return (
    <section className="dynamo-community-channels" aria-labelledby="community-channels-title">
      <div className="dynamo-community-section-heading">
        <div>
          <p className="dynamo-community-eyebrow">Community links</p>
          <h2 id="community-channels-title">Find the right channel</h2>
          <p>Follow project work, ask questions, join live conversations, or catch up on recordings.</p>
        </div>
      </div>

      <div className="dynamo-community-channels__window">
        <div className="dynamo-community-window-bar" aria-hidden="true">
          <span className="dynamo-community-window-dots"><i /><i /><i /></span>
          <strong>Community</strong>
          <span />
        </div>
        <div className="dynamo-community-channels__grid">
          {CHANNELS.map((channel) => (
            <a href={channel.href} target="_blank" rel="noreferrer" key={channel.name}>
              <span className={`dynamo-community-app dynamo-community-app--${channel.tone}`}>
                <FontAwesomeIcon name={channel.icon} />
              </span>
              <span><strong>{channel.name}</strong><small>{channel.label}</small></span>
              <Arrow diagonal />
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}

export function CommunityLanding() {
  return (
    <div className="dynamo-community-page">
      <section className="dynamo-community-meeting" aria-labelledby="community-meeting-title">
        <div className="dynamo-community-meeting__cadence" aria-hidden="true">
          <span>Every Wednesday</span>
          <strong>10:30</strong>
          <small>AM Pacific Time</small>
        </div>
        <div className="dynamo-community-meeting__copy">
          <p className="dynamo-community-eyebrow">Weekly community meeting</p>
          <h2 id="community-meeting-title">Join the project conversation</h2>
          <p>Meet maintainers and contributors for project updates, design discussions, demos, and open Q&amp;A.</p>
          <div className="dynamo-community-meeting__actions">
            <a className="dynamo-community-button is-primary" href={MEETING_URL} target="_blank" rel="noreferrer">
              Join Google Meet <Arrow diagonal />
            </a>
            <a className="dynamo-community-button" href={NOTES_URL} target="_blank" rel="noreferrer">
              Agenda and notes <Arrow diagonal />
            </a>
            <a className="dynamo-community-text-link" href={INVITE_URL} target="_blank" rel="noreferrer">
              Download calendar invite <Arrow diagonal />
            </a>
          </div>
        </div>
      </section>

      <FullCalendar />
      <CommunityChannels />

      <section className="dynamo-community-contribute" aria-labelledby="community-contribute-title">
        <div>
          <p className="dynamo-community-eyebrow">Contribute</p>
          <h2 id="community-contribute-title">Help improve Dynamo</h2>
          <p>Contribute code or documentation, propose a larger change, or bring a topic to the community meeting.</p>
        </div>
        <ul>
          <li><a href="/dynamo/dev/contributing/contribution-guide">Contribution guide <Arrow /></a></li>
          <li><a href="https://github.com/ai-dynamo/enhancements" target="_blank" rel="noreferrer">Enhancement proposals <Arrow diagonal /></a></li>
          <li><a href={NOTES_URL} target="_blank" rel="noreferrer">Add a meeting topic <Arrow diagonal /></a></li>
        </ul>
      </section>
    </div>
  );
}

export default CommunityLanding;
