#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.frontend.scripts.analysis.summarize_quic_qlog import summarize


class SummarizeQuicQlogTest(unittest.TestCase):
    def test_uses_packet_spans_when_stock_qlog_has_no_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qlog = root / "capture.sqlog"
            trace = root / "worker.jsonl"
            qlog.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "name": "transport:packet_sent",
                            "data": {"header": {"packet_number": packet_number}},
                        }
                    )
                    for packet_number in range(2)
                ),
                encoding="utf-8",
            )
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "target": "quinn_proto::connection::streams::state",
                            "message": "STREAM",
                            "span_name": "send",
                            "span_id": span_id,
                            "pn": packet_number,
                            "space": "Data",
                            "id": stream_id,
                        }
                    )
                    for span_id, packet_number, stream_id in [
                        ("packet-a", 0, "0"),
                        ("packet-a", 0, "4"),
                        ("packet-b", 1, "8"),
                    ]
                ),
                encoding="utf-8",
            )

            result = summarize([qlog], [trace])

        self.assertEqual(result["frame_detail_source"], "quinn_trace_log")
        self.assertEqual(result["response_carrying_packets"], 2)
        self.assertEqual(result["multi_response_stream_packets"], 1)
        self.assertEqual(result["multi_response_stream_packet_percentage"], 50.0)
        self.assertTrue(result["passes_50_percent_gate"])


if __name__ == "__main__":
    unittest.main()
