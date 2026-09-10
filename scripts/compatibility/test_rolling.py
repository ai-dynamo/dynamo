# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rolling_client import main, payloads, request_once

from scripts.compatibility.rolling_contract import (
    check_budget,
    check_requests,
    converged,
    pod_record,
    request_count,
)
from scripts.compatibility.runner import ContractError


class RollingContracts(unittest.TestCase):
    def pod(self, uid="old", image="old", terminating=False, phase="Running"):
        return pod_record(
            {
                "metadata": {
                    "uid": uid,
                    "name": uid,
                    "labels": {"nvidia.com/dynamo-component": "decode"},
                    "deletionTimestamp": "now" if terminating else None,
                },
                "spec": {
                    "containers": [
                        {
                            "name": "main",
                            "image": image,
                            "resources": {"limits": {"nvidia.com/gpu": "1"}},
                        }
                    ]
                },
                "status": {
                    "phase": phase,
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containerStatuses": [
                        {"name": "main", "imageID": "sha256:actual", "restartCount": 0}
                    ],
                },
            }
        )

    def test_terminating_gpu_still_counts_until_terminal(self):
        with self.assertRaises(ContractError):
            check_budget(
                [self.pod(terminating=True), self.pod("new1"), self.pod("new2")]
            )
        check_budget([self.pod(phase="Succeeded"), self.pod("new1"), self.pod("new2")])

    def test_convergence_needs_new_uids_and_no_old_or_terminating_pods(self):
        new = [self.pod("n1", "candidate"), self.pod("n2", "candidate")]
        self.assertTrue(converged(new, "decode", "candidate", {"old"}))
        self.assertFalse(converged(new, "decode", "candidate", {"n1"}))
        self.assertFalse(
            converged(
                new + [self.pod(terminating=True)], "decode", "candidate", {"old"}
            )
        )
        new[0]["image_id"] = ""
        self.assertFalse(converged(new, "decode", "candidate", {"old"}))

    def test_final_report_cannot_hide_errors_or_missing_phase_traffic(self):
        good = {"failures": 0, "phases": {"before": {"a": 2, "b": 2, "c": 2}}}
        check_requests(good, ["before"])
        with self.assertRaises(ContractError):
            check_requests(good, ["before", "upgrade"])
        with self.assertRaises(ContractError):
            check_requests({**good, "failures": 1}, ["before"])

    def test_worker_counter_excludes_other_endpoints(self):
        metrics = "\n".join(
            [
                'dynamo_component_requests_total{dynamo_endpoint="generate"} 5',
                'dynamo_component_requests_total{dynamo_endpoint="clear_kv_blocks"} 100',
            ]
        )
        self.assertEqual(request_count(metrics), 5)
        with self.assertRaises(ContractError):
            request_count("# no usable request counters")

    def test_payloads_cover_float_batch_and_stream_without_cross_worker_stop(self):
        embedding = dict(payloads("embedding", "model"))
        self.assertNotIn("encoding_format", embedding["default"])
        self.assertEqual(embedding["float"]["encoding_format"], "float")
        self.assertEqual(len(embedding["batch"]["input"]), 2)
        chat = dict(payloads("chat", "model"))
        self.assertTrue(chat["stream"]["stream"])
        self.assertEqual(chat["limited"]["max_tokens"], 1)
        self.assertTrue(all("stop" not in body for body in chat.values()))

    def test_client_records_bad_embedding_without_retry(self):
        body = {
            "object": "list",
            "data": [{"index": 0, "object": "embedding", "embedding": "bad"}],
        }
        with patch("rolling_client.requests.post") as post:
            response = post.return_value.__enter__.return_value
            response.status_code = 200
            response.text = json.dumps(body)
            response.json.return_value = body
            result = request_once("http://unused", "embedding", {"input": "hi"}, 1024)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Expected float array", result["error"])
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result["response"], json.dumps(body))

    def test_client_records_partial_stream_on_error(self):
        with patch("rolling_client.requests.post") as post:
            response = post.return_value.__enter__.return_value
            response.status_code = 200
            response.iter_lines.return_value = [b'data: {"error":"worker gone"}']
            result = request_once("http://unused", "chat", {"stream": True}, 0)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["response"], ['data: {"error":"worker gone"}'])

    def test_stop_includes_the_last_inflight_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)

            def finish_request(*args):
                (state / "control.json").write_text(json.dumps({"phase": "stop"}))
                return {"status": "failed", "error": "stream interrupted"}

            with patch(
                "sys.argv",
                [
                    "client",
                    "--base",
                    "http://unused",
                    "--scenario",
                    "chat",
                    "--model",
                    "model",
                    "--state",
                    directory,
                ],
            ), patch("rolling_client.request_once", side_effect=finish_request), patch(
                "rolling_client.time.sleep"
            ), patch(
                "builtins.print"
            ) as printed:
                main()
            final = json.loads(printed.call_args.args[0])["summary"]
            self.assertEqual(final["requests"], 1)
            self.assertEqual(final["failures"], 1)
