# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from scripts.protocol_compatibility.__main__ import main
from scripts.protocol_compatibility.acquisition.http import (
    NoRedirect,
    acquire,
    load_capture,
    read_limited,
    server_name,
    write_json,
)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            server="example-engine",
            url="https://example.com/pinned/openapi.yaml",
            file=None,
            metadata=None,
            output_dir=self.root / "capture",
        )

    def test_server_identity_is_generic_but_not_an_arbitrary_label_or_path(self):
        for name in ("dynamo", "example-engine", "another_engine2"):
            self.assertEqual(server_name(name), name)
        for name in ("", "../path", "Example Engine", "x" * 65, None):
            with self.assertRaises(ValueError):
                server_name(name)

    @patch("scripts.protocol_compatibility.acquisition.http.build_opener")
    def test_url_capture_preserves_bytes_without_docker_or_annotations(self, opener):
        response = MagicMock()
        response.url, response.status = self.args.url, 200
        response.read.return_value = (
            b"openapi: 3.1.0\ninfo: {title: Published, version: '1'}\n"
        )
        opener.return_value.open.return_value.__enter__.return_value = response
        with patch(
            "subprocess.run", side_effect=AssertionError("No deployment inspection")
        ):
            self.assertEqual(acquire(self.args), 0)
        loaded = load_capture(self.args.output_dir, self.args.server)
        self.assertEqual(loaded.raw, response.read.return_value)
        self.assertEqual(loaded.document["openapi"], "3.1.0")
        self.assertEqual(
            loaded.metadata["source"],
            {"kind": "url", "url": self.args.url, "http_status": 200},
        )
        self.assertEqual(
            loaded.metadata["sha256"], hashlib.sha256(loaded.raw).hexdigest()
        )
        self.assertIsNone(loaded.metadata["caller_metadata"])
        self.assertFalse(loaded.metadata["caller_metadata_verified"])
        self.assertNotIn("image", loaded.metadata)
        self.assertIn("captured_at", loaded.metadata)

    def test_file_cli_preserves_json_yaml_and_unverified_annotations(self):
        for suffix, raw in (
            ("json", b'{"openapi": "3.1.0", "x-number": 1e3}\n'),
            ("yaml", b"openapi: 3.1.0\n# retain this comment\n"),
        ):
            with self.subTest(suffix=suffix):
                source = self.root / f"spec.{suffix}"
                source.write_bytes(raw)
                metadata = self.root / "metadata.json"
                annotations = {
                    "image": "mutable:latest",
                    "source_revision": "caller claim",
                }
                write_json(metadata, annotations)
                output = self.root / suffix
                self.assertEqual(
                    main(
                        [
                            "acquire",
                            "--server",
                            "example-engine",
                            "--file",
                            str(source),
                            "--metadata",
                            str(metadata),
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
                loaded = load_capture(output, "example-engine")
                self.assertEqual(loaded.raw, raw)
                self.assertEqual(loaded.metadata["caller_metadata"], annotations)
                self.assertFalse(loaded.metadata["caller_metadata_verified"])
                self.assertEqual(
                    load_capture(source, "example-engine").document, loaded.document
                )
                if suffix == "json":
                    self.assertEqual(loaded.document["x-number"], 1000)
                source.write_bytes(b"changed after reading")
                saved = self.root / f"saved-{suffix}"
                loaded.save(saved)
                self.assertEqual((saved / "openapi.raw").read_bytes(), raw)

    def test_modified_capture_and_wrong_identity_fail(self):
        self.args.url = None
        self.args.file = self.root / "spec.json"
        self.args.file.write_text('{"openapi":"3.1.0"}')
        acquire(self.args)
        with self.assertRaisesRegex(ValueError, "Wrong acquisition"):
            load_capture(self.args.output_dir, "different-engine")
        (self.args.output_dir / "openapi.raw").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after acquisition"):
            load_capture(self.args.output_dir, self.args.server)

    @patch("scripts.protocol_compatibility.acquisition.http.build_opener")
    def test_invalid_urls_fail_before_network(self, opener):
        for url in (
            "file:///tmp/spec.json",
            "ftp://example.com/spec",
            "https:///spec",
            "http://user:secret@example.com/spec",
            "https://example.com/spec?token=secret",
            "https://example.com/spec#fragment",
        ):
            with self.subTest(url=url):
                self.args.url = url
                with self.assertRaises(ValueError):
                    acquire(self.args)
        opener.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    @patch("scripts.protocol_compatibility.acquisition.http.build_opener")
    def test_bad_responses_do_not_create_capture(self, opener):
        response = MagicMock()
        opener.return_value.open.return_value.__enter__.return_value = response
        for url, status, raw in (
            ("https://other.example/spec", 200, b'{"openapi":"3.1.0"}'),
            (self.args.url, 204, b'{"openapi":"3.1.0"}'),
            (self.args.url, 200, b"<html>not a spec</html>"),
            (self.args.url, 200, b'{"openapi":"3.0.0"}'),
            (self.args.url, 200, b"[]"),
        ):
            with self.subTest(status=status, raw=raw):
                response.url, response.status = url, status
                response.read.return_value = raw
                with self.assertRaises(ValueError):
                    acquire(self.args)
                self.assertFalse(self.args.output_dir.exists())
        opener.return_value.open.side_effect = HTTPError(
            self.args.url, 404, "missing", {}, None
        )
        with self.assertRaises(HTTPError):
            acquire(self.args)
        self.assertFalse(self.args.output_dir.exists())
        self.assertIsNone(
            NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://other.example"
            )
        )

    def test_size_limit_and_invalid_metadata(self):
        with patch("scripts.protocol_compatibility.acquisition.http.MAX_BYTES", 4):
            self.assertEqual(read_limited(io.BytesIO(b"1234")), b"1234")
            with self.assertRaisesRegex(ValueError, "capture limit"):
                read_limited(io.BytesIO(b"12345"))
        self.args.metadata = self.root / "invalid.json"
        self.args.metadata.write_text("[]")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            acquire(self.args)
        self.assertFalse(self.args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
