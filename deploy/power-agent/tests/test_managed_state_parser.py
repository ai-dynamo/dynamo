# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `_load_previously_managed_gpus` parsing robustness.

PR #9682 CodeRabbit review on power_agent.py:109 flagged the pre-fix
implementation as too narrow:

    try:
        with open(_MANAGED_STATE_PATH) as f:
            return set(json.load(f).get("managed_uuids", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

Failure modes the original missed and that this file pins:

  1. OSError siblings (PermissionError, IsADirectoryError, I/O errors).
     A host-volume permission glitch must not CrashLoopBackOff the
     agent — return empty + log a WARNING.
  2. Top-level JSON value is not an object. `json.load(f).get(...)`
     crashes with AttributeError on a list / int / string / null root.
  3. `managed_uuids` is present but not a list. The previous code
     would silently iterate the characters of a string or crash
     `set(...)` on an int.
  4. List entries that are not strings (bytes, ints, None, nested
     objects). Downstream `uuid in _previously_managed` checks would
     compare bytes/ints against `str` UUIDs and silently never match
     — orphan recovery would stop working without a clear error.

The contract is: every malformed-state branch returns `set()` and
logs at WARNING so operators can see corruption in pod logs without
losing the agent's cap-write availability.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import mock_open, patch

from power_agent import _load_previously_managed_gpus


class _LoadTestBase(unittest.TestCase):
    """Helper to drive `_load_previously_managed_gpus` against arbitrary
    file content + capture warnings emitted by the parser."""

    def _load(self, content: str):
        with patch("builtins.open", mock_open(read_data=content)):
            with self.assertLogs("power_agent", level="WARNING") as cm:
                result = _load_previously_managed_gpus()
        return result, cm.output

    def _load_no_warn(self, content: str):
        """Variant for happy-path cases where no WARNING should fire."""
        with patch("builtins.open", mock_open(read_data=content)):
            # The parser must not emit any WARNING on a well-formed file.
            # `assertNoLogs` doesn't exist on every Python we support, so
            # check via a custom handler.
            import logging

            captured: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = captured.append  # type: ignore[assignment]
            handler.setLevel(logging.WARNING)
            logger = logging.getLogger("power_agent")
            logger.addHandler(handler)
            try:
                result = _load_previously_managed_gpus()
            finally:
                logger.removeHandler(handler)
            self.assertEqual(
                [r.getMessage() for r in captured],
                [],
                "Well-formed input must not emit any WARNING.",
            )
        return result


# ---------------------------------------------------------------------------
# Happy path — well-formed state file
# ---------------------------------------------------------------------------


class TestHappyPath(_LoadTestBase):
    def test_well_formed_returns_uuid_set(self):
        content = json.dumps({"managed_uuids": ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]})
        result = self._load_no_warn(content)
        self.assertEqual(result, {"GPU-aaaa", "GPU-bbbb", "GPU-cccc"})

    def test_empty_managed_uuids_returns_empty_set(self):
        result = self._load_no_warn(json.dumps({"managed_uuids": []}))
        self.assertEqual(result, set())

    def test_missing_managed_uuids_key_returns_empty_set(self):
        # An object without `managed_uuids` is treated as "nothing
        # managed yet" — same as if the file didn't exist. Preserves
        # the pre-fix behaviour where `.get(..., [])` defaulted.
        result = self._load_no_warn(json.dumps({"other_key": "value"}))
        self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# Missing file — silent empty (pre-existing behaviour, must not regress)
# ---------------------------------------------------------------------------


class TestMissingFile(unittest.TestCase):
    def test_file_not_found_is_silent(self):
        """FileNotFoundError is the steady-state for the first startup
        on a fresh node. Must stay silent so we don't pollute logs
        every restart on every node in the cluster."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            import logging

            captured: list[logging.LogRecord] = []
            handler = logging.Handler()
            handler.emit = captured.append  # type: ignore[assignment]
            handler.setLevel(logging.WARNING)
            logger = logging.getLogger("power_agent")
            logger.addHandler(handler)
            try:
                result = _load_previously_managed_gpus()
            finally:
                logger.removeHandler(handler)

        self.assertEqual(result, set())
        self.assertEqual(captured, [], "Missing state file must not WARN")


# ---------------------------------------------------------------------------
# OSError siblings — must NOT propagate
# ---------------------------------------------------------------------------


class TestOsErrorSiblings(unittest.TestCase):
    """The pre-fix code only caught FileNotFoundError; the broader
    OSError family (PermissionError, IsADirectoryError,
    NotADirectoryError, plain IOError) propagated up the call stack
    and CrashLoopBackOff'd the agent. Pin each documented sibling.
    """

    def _assert_returns_empty_and_warns(self, exc):
        with patch("builtins.open", side_effect=exc):
            with self.assertLogs("power_agent", level="WARNING") as cm:
                result = _load_previously_managed_gpus()
        self.assertEqual(result, set())
        joined = "\n".join(cm.output)
        # The WARNING should name the exception type so an operator
        # can immediately spot whether it's a perm bit or a stale
        # mount issue.
        self.assertIn(type(exc).__name__, joined)

    def test_permission_error_is_caught(self):
        self._assert_returns_empty_and_warns(PermissionError("EACCES"))

    def test_is_a_directory_error_is_caught(self):
        self._assert_returns_empty_and_warns(IsADirectoryError("EISDIR"))

    def test_not_a_directory_error_is_caught(self):
        self._assert_returns_empty_and_warns(NotADirectoryError("ENOTDIR"))

    def test_generic_oserror_is_caught(self):
        # Synthesize an EIO — host volume hiccup, NFS stall, etc.
        self._assert_returns_empty_and_warns(OSError("EIO"))


# ---------------------------------------------------------------------------
# Malformed JSON — already covered by the original try/except, re-pin
# ---------------------------------------------------------------------------


class TestMalformedJson(_LoadTestBase):
    def test_truncated_json_returns_empty_and_warns(self):
        result, output = self._load('{"managed_uuids": ["GPU-aa"')
        self.assertEqual(result, set())
        self.assertTrue(any("JSONDecodeError" in line for line in output))

    def test_empty_file_returns_empty_and_warns(self):
        result, output = self._load("")
        self.assertEqual(result, set())
        self.assertTrue(any("JSONDecodeError" in line for line in output))


# ---------------------------------------------------------------------------
# Schema validation — root is not an object
# ---------------------------------------------------------------------------


class TestNonObjectRoot(_LoadTestBase):
    """`json.load(f).get(...)` raises AttributeError if the root isn't a
    dict. Pre-fix the AttributeError would have bubbled up. Each of these
    must return `set()` and WARN about the unexpected root type."""

    def test_top_level_list_returns_empty_and_warns(self):
        result, output = self._load(json.dumps(["GPU-aa", "GPU-bb"]))
        self.assertEqual(result, set())
        self.assertTrue(any("root type list" in line for line in output))

    def test_top_level_string_returns_empty_and_warns(self):
        result, output = self._load(json.dumps("GPU-aa"))
        self.assertEqual(result, set())
        self.assertTrue(any("root type str" in line for line in output))

    def test_top_level_int_returns_empty_and_warns(self):
        result, output = self._load(json.dumps(42))
        self.assertEqual(result, set())
        self.assertTrue(any("root type int" in line for line in output))

    def test_top_level_null_returns_empty_and_warns(self):
        result, output = self._load("null")
        self.assertEqual(result, set())
        self.assertTrue(any("root type NoneType" in line for line in output))


# ---------------------------------------------------------------------------
# Schema validation — `managed_uuids` is not a list
# ---------------------------------------------------------------------------


class TestManagedUuidsNotAList(_LoadTestBase):
    def test_string_value_returns_empty_and_warns(self):
        # Pre-fix `set(string)` would iterate characters silently —
        # operators would see {'G', 'P', 'U', '-', 'a'} sneak into
        # _previously_managed and never match anything.
        result, output = self._load(json.dumps({"managed_uuids": "GPU-aa"}))
        self.assertEqual(result, set())
        self.assertTrue(any("managed_uuids type str" in line for line in output))

    def test_int_value_returns_empty_and_warns(self):
        # Pre-fix `set(42)` would raise TypeError → not caught → crash.
        result, output = self._load(json.dumps({"managed_uuids": 42}))
        self.assertEqual(result, set())
        self.assertTrue(any("managed_uuids type int" in line for line in output))

    def test_dict_value_returns_empty_and_warns(self):
        result, output = self._load(json.dumps({"managed_uuids": {"k": "v"}}))
        self.assertEqual(result, set())
        self.assertTrue(any("managed_uuids type dict" in line for line in output))


# ---------------------------------------------------------------------------
# Schema validation — entries inside the list
# ---------------------------------------------------------------------------


class TestEntryTypeValidation(_LoadTestBase):
    def test_non_string_entries_are_dropped_and_warned(self):
        """Mixed list: keep strings, drop everything else, WARN once."""
        # JSON has no bytes literal, so use ints + null + a nested object
        # to exercise the type guard.
        content = json.dumps(
            {"managed_uuids": ["GPU-aa", 42, None, {"x": 1}, "GPU-bb"]}
        )
        result, output = self._load(content)
        self.assertEqual(result, {"GPU-aa", "GPU-bb"})
        self.assertTrue(
            any("3 non-string entries" in line for line in output),
            f"expected drop count in warning; got: {output}",
        )

    def test_all_entries_non_string_returns_empty(self):
        result, output = self._load(json.dumps({"managed_uuids": [1, 2, 3, None]}))
        self.assertEqual(result, set())
        self.assertTrue(any("4 non-string entries" in line for line in output))

    def test_well_formed_strings_dont_trigger_drop_warning(self):
        """All-string list → no "non-string entries" WARNING fires."""
        result = self._load_no_warn(json.dumps({"managed_uuids": ["GPU-a", "GPU-b"]}))
        self.assertEqual(result, {"GPU-a", "GPU-b"})


if __name__ == "__main__":
    unittest.main()
