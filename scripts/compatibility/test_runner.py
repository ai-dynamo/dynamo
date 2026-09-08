# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from runner import (
    DockerStack,
    main,
    matrix,
    probe,
    validate_chat,
    validate_embedding,
    validate_stream,
)


class CompatibilityTests(unittest.TestCase):
    def test_both_age_directions(self):
        releases = {
            "1.3": {"frontend": "f13", "worker": "w13"},
            "1.4": {"frontend": "f14", "worker": "w14"},
        }
        pairs = matrix(releases, "1.5", "fc", "wc")
        self.assertEqual(
            [(f, w) for _, f, w in pairs],
            [("fc", "wc"), ("f14", "wc"), ("fc", "w14"), ("f13", "wc"), ("fc", "w13")],
        )
        with self.assertRaises(KeyError):
            matrix(releases, "1.6", "fc", "wc")

    def test_default_window_tracks_candidate_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            with patch(
                "sys.argv",
                [
                    "runner",
                    "--plan",
                    "--frontend-image",
                    "fc",
                    "--worker-image",
                    "wc",
                    "--output",
                    str(output),
                ],
            ), patch("builtins.print") as printed:
                main()
            pairs = json.loads(printed.call_args.args[0])
            self.assertEqual(len(pairs), 5)
            self.assertTrue(any(":1.3.1" in ref for pair in pairs for ref in pair))
            self.assertTrue(any(":1.4.2" in ref for pair in pairs for ref in pair))
            self.assertFalse(any(":1.2." in ref for pair in pairs for ref in pair))
            self.assertFalse(output.exists())

    def test_float_contract_rejects_base64_and_nonfinite(self):
        body = {
            "object": "list",
            "data": [{"index": 0, "object": "embedding", "embedding": [1.0, 2.0]}],
        }
        validate_embedding(body, 1, 2)
        for value in ("AACAPwAAAEA=", [float("nan"), 2], [True, 2], [1.0]):
            body["data"][0]["embedding"] = value
            with self.assertRaises(AssertionError):
                validate_embedding(body, 1, 2)

    def test_chat_contract(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 2},
        }
        validate_chat(body, 32)
        with self.assertRaises(AssertionError):
            validate_chat(body, 1)
        body["choices"][0]["finish_reason"] = "error"
        with self.assertRaises(AssertionError):
            validate_chat(body, 32)

    def test_stop_contract(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1},
        }
        validate_chat(body, 32, "Hello")
        body["choices"][0]["message"]["content"] = "Hello world"
        with self.assertRaises(AssertionError):
            validate_chat(body, 32, "Hello")
        body["choices"][0]["message"]["content"] = ""
        body["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(AssertionError):
            validate_chat(body, 32, "Hello")

    def test_stream_must_finish_and_not_hide_errors(self):
        def chunk(delta, finish):
            return "data: " + json.dumps(
                {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            )

        valid = [chunk({"content": "Hello"}, None), chunk({}, "stop"), "data: [DONE]"]
        validate_stream(valid)
        for invalid in [
            valid[:-1],
            valid[:1] + ["data: [DONE]"],
            ['data: {"error":"worker failed"}'],
            ["event: error"],
            valid + [valid[0]],
        ]:
            with self.assertRaises(AssertionError):
                validate_stream(invalid)

    def test_probe_records_failures_and_runs_remaining_cases(self):
        class Response:
            status_code = 200
            text = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"index": 0, "object": "embedding", "embedding": "AACAPwAAAEA="}
                    ],
                }
            )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            def json(self):
                return json.loads(self.text)

        with tempfile.TemporaryDirectory() as directory, patch(
            "runner.requests.post", return_value=Response()
        ):
            results = probe(
                "http://unused",
                "embedding",
                {"id": "model", "dimensions": 2},
                Path(directory),
            )
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r["status"] == "failed" for r in results))
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 3)

    def test_all_pairs_run_and_failure_is_reported(self):
        config = {
            "releases": {
                line: {"frontend": "f" + line, "worker": "w" + line}
                for line in ("1.3", "1.4")
            },
            "infrastructure": {"etcd": "etcd", "nats": "nats"},
            "models": {
                s: {"id": s, "revision": "fixed"} for s in ("embedding", "chat")
            },
        }
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda: types.SimpleNamespace(
            model_info=lambda *a, **kw: types.SimpleNamespace(sha="fixed")
        )
        hub.snapshot_download = lambda *a, **kw: None
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = root / "config.json"
                manifest.write_text(json.dumps(config))
                output = root / "output"
                argv = [
                    "runner",
                    "--config",
                    str(manifest),
                    "--release-line",
                    "1.5",
                    "--frontend-image",
                    "fc",
                    "--worker-image",
                    "wc",
                    "--output",
                    str(output),
                ]
                effects = [[{"status": "passed"}]] * 10
                if failed:
                    effects[1] = RuntimeError("worker startup failed")
                    effects[2] = [{"status": "failed", "error": "expected float array"}]
                with patch("sys.argv", argv), patch.dict(
                    "sys.modules", {"huggingface_hub": hub}
                ), patch("runner.command"), patch(
                    "runner.resolve_image", side_effect=lambda r: {"id": r}
                ) as resolve, patch(
                    "runner.execute", side_effect=effects
                ) as execute:
                    if failed:
                        with self.assertRaises(SystemExit):
                            main()
                    else:
                        main()
                    self.assertEqual(execute.call_count, 10)
                    self.assertEqual(resolve.call_count, 8)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "failed" if failed else "passed")
                self.assertEqual(len(report["runs"]), 10)
                if failed:
                    self.assertIn("startup failed", report["runs"][1]["error"])
                    self.assertEqual(report["runs"][2]["status"], "failed")
                    self.assertEqual(report["runs"][-1]["status"], "passed")

    def test_cleanup_after_partial_startup(self):
        calls = []

        def run(*args, **kwargs):
            calls.append(args)
            if args[:2] == ("docker", "create"):
                raise OSError("startup failed")
            return "{}"

        with tempfile.TemporaryDirectory() as directory, patch(
            "runner.command", side_effect=run
        ):
            with self.assertRaises(OSError):
                with DockerStack(Path(directory)) as stack:
                    stack.start("worker", "image", ["python3"])
        self.assertTrue(any(c[:3] == ("docker", "rm", "-f") for c in calls))
        self.assertTrue(any(c[:3] == ("docker", "network", "rm") for c in calls))


if __name__ == "__main__":
    unittest.main()
