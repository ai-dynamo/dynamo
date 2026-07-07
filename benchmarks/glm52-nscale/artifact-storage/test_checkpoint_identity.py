# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("checkpoint_identity.py")
SPEC = importlib.util.spec_from_file_location("checkpoint_identity", MODULE_PATH)
assert SPEC and SPEC.loader
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


class CheckpointIdentityTest(unittest.TestCase):
    def test_is_stable_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested" / "a.json").write_text("a")
            (root / "b.json").write_text("bb")
            first = identity.tree_identity(root)
            self.assertEqual(first["files"], 2)
            self.assertEqual(first["directories"], 1)
            self.assertEqual(first["bytes"], 3)
            self.assertEqual(first, identity.tree_identity(root))
            (root / "b.json").write_text("bc")
            self.assertNotEqual(first["tree_sha256"], identity.tree_identity(root)["tree_sha256"])

    def test_rejects_file_directory_and_broken_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target").write_text("x")
            (root / "link").symlink_to("target")
            with self.assertRaisesRegex(ValueError, "symlink"):
                identity.tree_identity(root)
            (root / "link").unlink()
            (root / "directory").mkdir()
            (root / "link").symlink_to("directory", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                identity.tree_identity(root)
            (root / "link").unlink()
            (root / "broken").symlink_to("missing")
            with self.assertRaisesRegex(ValueError, "symlink"):
                identity.tree_identity(root)


if __name__ == "__main__":
    unittest.main()
