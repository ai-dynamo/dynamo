# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check supported local formats, exact-byte retention, and unverified provenance."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.protocol_compatibility.inputs import (
    load_input,
    parse_document,
    server_name,
    write_json,
)


class InputTests(unittest.TestCase):
    def test_openapi_versions_and_document_shape(self):
        """Preserve OpenAPI 3.2 keywords and reject unsupported or non-document inputs."""
        raw = b'{"openapi":"3.2.0","itemSchema":{"type":"object"}}'
        self.assertEqual(parse_document(raw)["itemSchema"], {"type": "object"})
        for raw in (b"<html>not a spec</html>", b'{"openapi":"3.0.0"}', b"[]"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_document(raw)

    def test_server_identity_is_generic_but_not_an_arbitrary_label_or_path(self):
        """Require lowercase identities without claiming to verify the named source."""
        for name in ("dynamo", "example-engine", "another_engine2"):
            self.assertEqual(server_name(name), name)
        for name in ("", "../path", "Example Engine", "x" * 65, None):
            with self.assertRaises(ValueError):
                server_name(name)

    def test_local_files_preserve_bytes_and_unverified_annotations(self):
        """Save JSON/YAML, labels and hashes without rereading a changed source."""
        for suffix, raw in (
            ("json", b'{"openapi": "3.1.0", "x-number": 1e3}\n'),
            ("yaml", b"openapi: 3.1.0\n# retain this comment\n"),
        ):
            with self.subTest(
                suffix=suffix
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, metadata = root / f"spec.{suffix}", root / "metadata.json"
                source.write_bytes(raw)
                annotations = {
                    "source_revision": "caller claim",
                    "scope": "full declared contract",
                    "configuration": {"dispatch_events": True},
                }
                write_json(metadata, annotations)
                loaded = load_input(source, "example-engine", metadata)
                self.assertEqual(loaded.raw, raw)
                self.assertEqual(
                    loaded.metadata["sha256"], hashlib.sha256(raw).hexdigest()
                )
                self.assertEqual(loaded.metadata["caller_metadata"], annotations)
                self.assertFalse(loaded.metadata["caller_metadata_verified"])
                self.assertEqual(
                    load_input(source, "example-engine").document, loaded.document
                )
                source.write_bytes(b"changed after reading")
                saved = root / "saved"
                loaded.save(saved)
                self.assertEqual((saved / "openapi.raw").read_bytes(), raw)
                self.assertEqual(
                    json.loads((saved / "acquisition.json").read_text()),
                    loaded.metadata,
                )

    def test_invalid_metadata(self):
        """Reject metadata that is not a JSON object."""
        with tempfile.TemporaryDirectory() as directory:
            source, metadata = (
                Path(directory) / "spec.json",
                Path(directory) / "metadata.json",
            )
            source.write_text('{"openapi":"3.1.0"}')
            write_json(metadata, [])
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_input(source, "example-engine", metadata)


if __name__ == "__main__":
    unittest.main()
