// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Checks the weekly meeting helpers in docs/fern/components/community-meeting.ts.
// Run: node --test docs/fern/scripts/test_community_meeting.mjs
import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

// The components import each other without a file extension, as Fern expects.
registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context);
    } catch (err) {
      if (!specifier.startsWith(".")) throw err;
      return next(`${specifier}.ts`, context);
    }
  },
});

const meeting = await import("../components/community-meeting.ts");
const { MONTH_INDEX } = await import("../components/calendar-today.ts");

const starts = (year, month) =>
  meeting.meetingsInMonth(year, month).map((event) => event.start);

test("meets at 9:00 AM Pacific across both DST changes", () => {
  assert.deepEqual(starts(2026, 8), []);
  assert.deepEqual(starts(2026, 9), [
    "2026-10-13T16:00:00.000Z",
    "2026-10-20T16:00:00.000Z",
    "2026-10-27T16:00:00.000Z",
  ]);
  assert.equal(starts(2026, 10)[0], "2026-11-03T17:00:00.000Z");
  assert.deepEqual(starts(2027, 2), [
    "2027-03-02T17:00:00.000Z",
    "2027-03-09T17:00:00.000Z",
    "2027-03-16T16:00:00.000Z",
    "2027-03-23T16:00:00.000Z",
    "2027-03-30T16:00:00.000Z",
  ]);
});

test("shows a meeting once when the calendar also lists it", () => {
  const [first] = meeting.meetingsInMonth(2026, 9);
  const copy = { ...first, addUrl: "calendar copy" };
  const oct13 = meeting
    .eventsInMonth([copy], 2026, 9, MONTH_INDEX)
    .filter((event) => event.day === "13");
  assert.equal(oct13.length, 1);
  const next = { ...meeting.nextMeeting(), addUrl: "calendar copy" };
  assert.equal(meeting.upcomingWithMeeting([next]).length, 1);
});
