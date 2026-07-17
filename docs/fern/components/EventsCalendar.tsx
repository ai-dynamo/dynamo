/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EventsCalendar — Dynamo community events, split into upcoming (cards) and past
 * (a collapsed native <details> accordion).
 *
 * Fully static and self-contained: it reads the build-time data module
 * events.generated.ts (produced hourly by .github/scripts/generate-events.js from
 * the public Google Calendar) and renders it with inline styles. No hooks, no
 * browser APIs, no external CSS — safe to render on the server, and the accordion
 * is plain HTML so it needs no client JavaScript.
 *
 * Styling uses translucent neutral borders (rgba grays) and inherits the page's
 * text color, so it reads correctly in both light and dark themes. NVIDIA green
 * (#76b900) is the single accent and is legible on both.
 */

import { UPCOMING_EVENTS, PAST_EVENTS, type DynamoEvent } from "./events.generated";

const ACCENT = "#76b900";
const BORDER = "1px solid rgba(128,128,128,0.25)";
const MUTED = "rgba(128,128,128,1)";

function LocationLine({ event }: { event: DynamoEvent }) {
  if (!event.location) return null;
  return (
    <div style={{ fontSize: "12px", color: MUTED, display: "flex", alignItems: "center", gap: "5px" }}>
      <span aria-hidden>📍</span>
      {event.locationUrl ? (
        <a href={event.locationUrl} target="_blank" rel="noreferrer" style={{ color: ACCENT, textDecoration: "none" }}>
          {event.location}
        </a>
      ) : (
        <span>{event.location}</span>
      )}
    </div>
  );
}

function UpcomingCard({ event }: { event: DynamoEvent }) {
  return (
    <div style={{ border: BORDER, borderRadius: "12px", padding: "14px 16px", display: "flex", gap: "12px" }}>
      <div
        style={{
          flex: "none",
          width: "54px",
          textAlign: "center",
          borderRight: BORDER,
          paddingRight: "12px",
        }}
      >
        <div style={{ fontSize: "11px", fontWeight: 600, color: ACCENT, textTransform: "uppercase", letterSpacing: "0.04em" }}>
          {event.month}
        </div>
        <div style={{ fontSize: "20px", fontWeight: 600, lineHeight: 1.1 }}>{event.day}</div>
        <div style={{ fontSize: "11px", color: MUTED }}>{event.year}</div>
      </div>
      <div>
        <div style={{ fontSize: "14px", fontWeight: 600, marginBottom: "3px" }}>
          <a href={event.addUrl} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>
            {event.title}
          </a>
        </div>
        <LocationLine event={event} />
      </div>
    </div>
  );
}

function PastRow({ event }: { event: DynamoEvent }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", gap: "12px", padding: "8px 0", borderBottom: BORDER, fontSize: "13px" }}>
      <span style={{ flex: "none", width: "120px", color: MUTED, fontVariantNumeric: "tabular-nums" }}>{event.dateLabel}</span>
      <a href={event.addUrl} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>
        {event.title}
      </a>
      {event.location && <span style={{ marginLeft: "auto", color: MUTED, fontSize: "12px" }}>{event.location}</span>}
    </div>
  );
}

export function EventsCalendar() {
  const hasUpcoming = UPCOMING_EVENTS.length > 0;
  const hasPast = PAST_EVENTS.length > 0;

  return (
    <div style={{ margin: "0.5rem 0 1rem" }}>
      <div style={{ fontSize: "12px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.03em", color: MUTED, margin: "0 0 0.6rem" }}>
        Upcoming
      </div>
      {hasUpcoming ? (
        <div style={{ display: "grid", gap: "12px", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
          {UPCOMING_EVENTS.map((e) => (
            <UpcomingCard key={`${e.start}-${e.title}`} event={e} />
          ))}
        </div>
      ) : (
        <div style={{ fontSize: "13px", color: MUTED }}>No upcoming events right now — check back soon.</div>
      )}

      {hasPast && (
        <details style={{ border: BORDER, borderRadius: "12px", marginTop: "1rem", overflow: "hidden" }}>
          <summary style={{ cursor: "pointer", padding: "12px 16px", fontSize: "14px", fontWeight: 600, listStyle: "none" }}>
            Past events
          </summary>
          <div style={{ padding: "0 16px 10px" }}>
            {PAST_EVENTS.map((e) => (
              <PastRow key={`${e.start}-${e.title}`} event={e} />
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
