import inspect
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from _support import load_harness


class ColdStateTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()

    def test_eviction_uses_only_posix_fadvise_dontneed_for_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "checkpoint"
            root.mkdir()
            candidate = root / "pages.img"
            candidate.write_bytes(b"checkpoint")
            with mock.patch.object(self.harness.os, "posix_fadvise") as fadvise:
                self.harness.evict_candidate_files([candidate], allow_root=root)
            self.assertEqual(fadvise.call_count, 1)
            _, offset, length, advice = fadvise.call_args.args
            self.assertEqual((offset, length), (0, 0))
            self.assertEqual(advice, os.POSIX_FADV_DONTNEED)
            source = inspect.getsource(self.harness.evict_candidate_files)
            self.assertNotIn("drop_caches", source)
            self.assertNotIn("subprocess", source)

    def test_eviction_rejects_paths_outside_allow_root_or_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "checkpoint"
            root.mkdir()
            outside = pathlib.Path(directory) / "outside"
            outside.write_bytes(b"outside")
            linked = root / "linked"
            os.symlink(outside, linked)
            with self.assertRaises(ValueError):
                self.harness.evict_candidate_files([outside], allow_root=root)
            with self.assertRaises(ValueError):
                self.harness.evict_candidate_files([linked], allow_root=root)

    def test_eviction_prevalidates_every_candidate_before_any_fadvise_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "checkpoint"
            root.mkdir()
            valid = root / "valid.img"
            valid.write_bytes(b"valid")
            outside = pathlib.Path(directory) / "outside.img"
            outside.write_bytes(b"outside")
            with mock.patch.object(self.harness.os, "posix_fadvise") as fadvise:
                with self.assertRaises(ValueError):
                    self.harness.evict_candidate_files([valid, outside], allow_root=root)
            fadvise.assert_not_called()


if __name__ == "__main__":
    unittest.main()
