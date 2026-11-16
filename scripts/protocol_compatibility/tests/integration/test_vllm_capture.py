# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test preparation failure boundaries without installing or importing vLLM."""

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.protocol_compatibility.assessment.responses import prepare_response_source
from scripts.protocol_compatibility.frameworks import capture_vllm_openapi as capture
from scripts.protocol_compatibility.tests.integration.test_vllm_response import (
    PROFILE,
    recipe_inputs,
)


def capture_inputs():
    """Extend the shared synthetic event fixture with an explicit JSON error."""
    _, document, _ = recipe_inputs()
    document["components"]["schemas"]["Error"] = {"type": "object"}
    for path in document["paths"].values():
        event = path["post"]["responses"]["200"]["content"]["text/event-stream"][
            "schema"
        ]
        event["properties"]["data"]["anyOf"].append(
            {
                "contentMediaType": "application/json",
                "contentSchema": {"$ref": "#/components/schemas/Error"},
            }
        )
    return document, json.loads(PROFILE.read_text())


class VllmCaptureTests(unittest.TestCase):
    def test_capture_clears_inherited_vllm_switches(self):
        """Only the helper's VLLM_* settings reach build_app; other settings survive."""
        seen = {}

        def build_app(args, supported_tasks):
            """Record the environment seen by app construction without importing vLLM."""
            seen.update(capture.os.environ)
            return "app"

        response = SimpleNamespace(content=b"{}", raise_for_status=lambda: None)
        client = SimpleNamespace(get=lambda _: response, close=lambda: None)
        modules = {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(is_initialized=lambda: False)
            ),
            "fastapi.testclient": SimpleNamespace(TestClient=lambda app: client),
            "vllm.entrypoints.launchers.app": SimpleNamespace(build_app=build_app),
        }
        inherited = {
            "VLLM_SERVER_DEV_MODE": "1",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1",
            "VLLM_ENABLE_COHERE_API": "1",
            "VLLM_PLUGINS": "caller_plugin",
            "CAPTURE_TEST_UNRELATED": "preserved",
        }
        with patch.dict(capture.os.environ, inherited), patch.dict(
            capture.sys.modules, modules
        ):
            self.assertEqual(capture.capture(), b"{}")
        switches = {k: v for k, v in seen.items() if k.startswith("VLLM_")}
        self.assertEqual(switches, {"VLLM_PLUGINS": ""})
        self.assertEqual(seen["CAPTURE_TEST_UNRELATED"], "preserved")

    def test_loaded_sources_reject_edits_and_missing_records(self):
        """Imported source must match its recorded digest, including generated files."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "module.py"
            source.write_bytes(b"# original\n")
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            entry = SimpleNamespace(
                locate=lambda: source, hash=SimpleNamespace(mode="sha256", value=digest)
            )
            dist = SimpleNamespace(files=[entry])
            with patch.dict(
                capture.sys.modules, {"vllm": SimpleNamespace(__file__=str(source))}
            ):
                capture.verify_loaded_sources(dist)
                source.write_bytes(b"# changed\n")
                with self.assertRaises(ValueError):
                    capture.verify_loaded_sources(dist)
                dist.files = []
                with self.assertRaises(ValueError):
                    capture.verify_loaded_sources(dist)

    def test_invalid_capture_creates_no_outputs(self):
        """A failed declaration check cannot leave an apparently usable input pair."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs"
            argv = ["capture", "--profile", str(PROFILE), "--output-dir", str(output)]
            with patch.object(capture.sys, "argv", argv), patch.object(
                capture, "verify_installation"
            ), patch.object(capture, "verify_loaded_sources"), patch.object(
                capture, "capture", return_value=b"{}"
            ):
                with self.assertRaises(ValueError):
                    capture.main()
                self.assertFalse(output.exists())

    def test_adapter_uses_actual_bytes_and_reviewed_selectors(self):
        """Whitespace is hashed as captured and the generic loader accepts the pair."""
        document, profile = capture_inputs()
        raw = json.dumps(document, indent=3).encode() + b"\n"
        config = capture.make_adapter(raw, profile)["sources"]["framework"]
        self.assertEqual(config["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(config["streams"], profile["response_contract"]["streams"])
        self.assertEqual(config["revision"], profile["source"]["revision"])
        self.assertEqual(
            len(prepare_response_source(document, config, config["sha256"])[1]), 2
        )

    def test_rejects_drift_and_unresolved_references(self):
        """Each input defect fails instead of guessing or repairing the declaration."""
        for defect in (
            "dialect",
            "selector",
            "mode",
            "unary",
            "event",
            "ref",
            "external",
        ):
            with self.subTest(defect=defect):
                document, profile = capture_inputs()
                content = document["paths"]["/v1/completions"]["post"]["responses"][
                    "200"
                ]["content"]
                if defect == "dialect":
                    document["openapi"] = "3.2.0"
                elif defect == "selector":
                    profile["response_contract"]["streams"].pop()
                elif defect == "mode":
                    profile["response_contract"]["streams"][0][
                        "interpretation"
                    ] = "decoded-chunk"
                elif defect == "unary":
                    del content["application/json"]
                elif defect == "event":
                    del content["text/event-stream"]["schema"]["properties"]["data"]
                else:
                    document["components"]["schemas"]["Error"] = {
                        "$ref": "#/missing"
                        if defect == "ref"
                        else "https://example.invalid/schema"
                    }
                with self.assertRaises((ValueError, KeyError)):
                    capture.make_adapter(json.dumps(document).encode(), profile)

    def test_installation_requires_profile_identity(self):
        """Reject absent metadata, wrong version/URL, editable installs and overlays."""
        _, profile = capture_inputs()
        wheel = profile["wheel"]
        for defect in (None, "metadata", "version", "url", "editable", "overlay"):
            with self.subTest(defect=defect):
                origin = {"url": wheel["url"], "archive_info": {}}
                if defect == "url":
                    origin["url"] = "file:///different.whl"
                if defect == "editable":
                    origin.pop("archive_info")
                dist = SimpleNamespace(
                    version="wrong" if defect == "version" else wheel["version"],
                    read_text=lambda _: None
                    if defect == "metadata"
                    else json.dumps(origin),
                    locate_file=lambda _: Path("installed/vllm/__init__.py"),
                )
                spec = SimpleNamespace(
                    origin="overlay.py"
                    if defect == "overlay"
                    else "installed/vllm/__init__.py"
                )
                with patch.object(
                    capture.importlib.metadata, "distribution", return_value=dist
                ), patch.object(capture.importlib.util, "find_spec", return_value=spec):
                    if defect:
                        with self.assertRaises(ValueError):
                            capture.verify_installation(profile)
                    else:
                        self.assertIs(capture.verify_installation(profile), dist)

    def test_cli_writes_only_pair_and_never_overwrites(self):
        """Successful capture preserves raw bytes; a repeat fails before importing vLLM."""
        document, _ = capture_inputs()
        raw = json.dumps(document, indent=3).encode()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "inputs"
            argv = ["capture", "--profile", str(PROFILE), "--output-dir", str(output)]
            with patch.object(capture.sys, "argv", argv), patch.object(
                capture, "verify_installation"
            ), patch.object(capture, "verify_loaded_sources"), patch.object(
                capture, "capture", return_value=raw
            ) as mocked:
                capture.main()
                self.assertEqual(
                    {p.name for p in output.iterdir()},
                    {"vllm-openapi.json", "response-adapters.json"},
                )
                self.assertEqual((output / "vllm-openapi.json").read_bytes(), raw)
                with self.assertRaises(FileExistsError):
                    capture.main()
                mocked.assert_called_once()
