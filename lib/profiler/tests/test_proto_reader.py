# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for proto_reader — roundtrip with Rust writer output."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamo_profiler import proto_reader


DEMO_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sysprofile-demo-output"
)


class TestProtoReaderUtils(unittest.TestCase):
    def test_merge_intervals_empty(self):
        self.assertEqual(proto_reader.merge_intervals([]), [])

    def test_merge_intervals_no_overlap(self):
        result = proto_reader.merge_intervals([(1, 3), (5, 7), (9, 11)])
        self.assertEqual(result, [(1, 3), (5, 7), (9, 11)])

    def test_merge_intervals_overlap(self):
        result = proto_reader.merge_intervals([(1, 5), (3, 8), (10, 12)])
        self.assertEqual(result, [(1, 8), (10, 12)])

    def test_merge_intervals_adjacent(self):
        result = proto_reader.merge_intervals([(1, 5), (5, 10)])
        self.assertEqual(result, [(1, 10)])

    def test_invert_intervals(self):
        busy = [(10, 20), (30, 40)]
        idle = proto_reader.invert_intervals(busy, 0, 50)
        self.assertEqual(idle, [(0, 10), (20, 30), (40, 50)])

    def test_invert_intervals_no_gaps(self):
        busy = [(0, 50)]
        idle = proto_reader.invert_intervals(busy, 0, 50)
        self.assertEqual(idle, [])


@unittest.skipUnless(
    os.path.isdir(DEMO_DIR), "sysprofile-demo-output not found"
)
class TestProtoReaderWithDemoData(unittest.TestCase):
    def test_iter_packets_yields_data(self):
        pf = os.path.join(DEMO_DIR, "frontend.pftrace.gz")
        packets = list(proto_reader.iter_packets(pf))
        self.assertGreater(len(packets), 0)

    def test_parse_packet_returns_tracks_and_events(self):
        pf = os.path.join(DEMO_DIR, "frontend.pftrace.gz")
        tracks_found = 0
        events_found = 0
        for raw in proto_reader.iter_packets(pf):
            event, track = proto_reader.parse_packet(raw)
            if track is not None:
                tracks_found += 1
                self.assertIsInstance(track.uuid, int)
                self.assertIsInstance(track.name, str)
            if event is not None:
                events_found += 1
                self.assertIn(event.event_type, (1, 2, 3, 4))
        self.assertGreater(tracks_found, 0)
        self.assertGreater(events_found, 0)

    def test_read_trace_returns_slices(self):
        pf = os.path.join(DEMO_DIR, "frontend.pftrace.gz")
        parsed = proto_reader.read_trace(pf)
        self.assertIsInstance(parsed, proto_reader.ParsedTrace)
        self.assertGreater(len(parsed.slices), 0)
        self.assertGreater(len(parsed.tracks), 0)
        for sl in parsed.slices:
            self.assertIsInstance(sl, proto_reader.Slice)
            self.assertGreater(sl.duration_ns, 0)
            self.assertGreater(sl.end_ns, sl.start_ns)


if __name__ == "__main__":
    unittest.main()
