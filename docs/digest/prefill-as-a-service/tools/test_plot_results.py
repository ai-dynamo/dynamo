# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path

SCRIPT = Path(__file__).with_name("plot_results.py")
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT)
assert SPEC and SPEC.loader
plot_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_results)


class PlotResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = {
            "schema_version": 1,
            "spec_fingerprint": "1234567890abcdef",
            "locked_workload": {
                "repetitions": 3,
                "requests_per_repetition": 2,
                "configured_input_sequence_length": 4096,
                "configured_output_sequence_length": 256,
            },
            "repetitions": [self._repetition(index) for index in range(1, 4)],
            "requests": [
                self._request(1, "a", 1000, 2000),
                self._request(1, "b", 1200, 2200),
                self._request(2, "c", 1100, 2100),
                self._request(2, "d", 1300, 2400),
                self._request(3, "e", 1050, 2050),
                self._request(3, "f", 1250, 2300),
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _repetition(self, index: int) -> dict[str, object]:
        return {
            "repetition": index,
            "metrics": {
                "request_throughput": {"avg": 0.2 + index / 100},
                "output_token_throughput": {"avg": 50.0 + index},
            },
        }

    def _request(
        self,
        repetition: int,
        request_id: str,
        ttft_ms: float,
        request_latency_ms: float,
    ) -> dict[str, object]:
        return {
            "repetition": repetition,
            "request_id": request_id,
            "ttft_ms": ttft_ms,
            "request_latency_ms": request_latency_ms,
            "itl_ms": 4.0,
            "server_input_tokens": 4108,
            "server_output_tokens": 256,
        }

    def test_renders_three_deterministic_svg_figures(self) -> None:
        latency = self.root / "latency.svg"
        throughput = self.root / "throughput.svg"
        timeline = self.root / "timeline.svg"

        plot_results.render_all(self.data, latency, throughput, timeline)
        first = {
            path.name: path.read_bytes() for path in (latency, throughput, timeline)
        }
        plot_results.render_all(self.data, latency, throughput, timeline)
        second = {
            path.name: path.read_bytes() for path in (latency, throughput, timeline)
        }

        self.assertEqual(first, second)
        self.assertIn(b"Client-observed latency distributions", first["latency.svg"])
        self.assertIn(b"Inter-token latency", first["latency.svg"])
        self.assertIn(b"Per-repetition throughput", first["throughput.svg"])
        self.assertIn(b"not exact prefill", first["timeline.svg"])
        for rendered in first.values():
            self.assertIn(b"<svg", rendered)
            self.assertIn(b"1234567890ab", rendered)
            element_tree.fromstring(rendered)

    def test_rejects_timeline_with_first_token_after_completion(self) -> None:
        for request in self.data["requests"]:
            request["ttft_ms"] = 3000

        with self.assertRaisesRegex(ValueError, "ends before its first token"):
            plot_results.representative_request(self.data)

    def test_rejects_missing_normalized_data(self) -> None:
        path = self.root / "empty.json"
        path.write_text('{"schema_version": 1, "spec_fingerprint": "x"}\n')

        with self.assertRaisesRegex(ValueError, "request data is absent"):
            plot_results.load_normalized(path)

    def test_axis_maximum_preserves_fractional_scale(self) -> None:
        self.assertEqual(plot_results.nice_maximum(0.21), 0.25)
        self.assertEqual(plot_results.nice_maximum(2.2), 2.5)
        self.assertEqual(plot_results.nice_maximum(51), 75.0)
        self.assertEqual(plot_results.format_axis_tick(0.004, 0.004), "0.004")
        self.assertEqual(plot_results.format_axis_tick(0.625, 2.5), "0.6")


if __name__ == "__main__":
    unittest.main()
