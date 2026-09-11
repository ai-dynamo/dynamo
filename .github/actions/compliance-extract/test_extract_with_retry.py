# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real retry wrapper with a fake BuildKit command and sleep."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("extract-with-retry.sh")
TRANSIENT = "ERROR: failed to build: failed to solve: Unavailable: error reading from server: EOF"


class ExtractRetryTest(unittest.TestCase):
    def run_extract(self, errors):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, error in enumerate(errors, 1):
                (root / f"error-{index}").write_text(error)
            fake = root / "build"
            fake.write_text(
                """#!/usr/bin/env bash
set -eu
count=0
[ ! -f "$STATE/count" ] || count=$(cat "$STATE/count")
count=$((count + 1))
echo "$count" > "$STATE/count"
for arg in "$@"; do
    case "$arg" in type=local,dest=*) output=${arg#type=local,dest=} ;; esac
done
mkdir -p "$output"
echo "$count" > "$output/attempt-$count"
error="$STATE/error-$count"
if [ -s "$error" ]; then
    cat "$error"
    exit 7
fi
"""
            )
            fake.chmod(0o755)
            sleep = root / "sleep"
            sleep.write_text('#!/usr/bin/env bash\necho "$1" >> "$STATE/delays"\n')
            sleep.chmod(0o755)
            destination = root / "result"
            env = dict(
                os.environ,
                STATE=temp,
                RUNNER_TEMP=temp,
                PATH=temp + os.pathsep + os.environ["PATH"],
            )
            result = subprocess.run(
                ["bash", str(SCRIPT), str(destination), str(fake)],
                env=env,
                capture_output=True,
                text=True,
            )
            count = int((root / "count").read_text())
            delays = (
                (root / "delays").read_text().splitlines()
                if (root / "delays").exists()
                else []
            )
            files = sorted(p.name for p in destination.glob("*"))
            logs = list(root.glob("compliance-extract.*/attempt-*.log"))
            self.assertEqual(len(logs), count)
            return result.returncode, count, delays, files

    def test_success(self):
        self.assertEqual(self.run_extract([""]), (0, 1, [], ["attempt-1"]))

    def test_transient_recovers_without_partial_files(self):
        self.assertEqual(
            self.run_extract([TRANSIENT, ""]), (0, 2, ["10"], ["attempt-2"])
        )

    def test_third_attempt_recovers(self):
        self.assertEqual(
            self.run_extract([TRANSIENT, TRANSIENT, ""]),
            (0, 3, ["10", "30"], ["attempt-3"]),
        )

    def test_exhaustion_preserves_failure(self):
        self.assertEqual(self.run_extract([TRANSIENT] * 3), (7, 3, ["10", "30"], []))

    def test_deterministic_errors_do_not_retry(self):
        for error in [
            "ERROR: failed to solve: process did not complete: exit code: 1",
            "ERROR: failed to authorize: 401 Unauthorized",
            "ERROR: manifest unknown",
            "No solution found when resolving dependencies",
        ]:
            with self.subTest(error=error):
                self.assertEqual(self.run_extract([error]), (7, 1, [], []))

    def test_earlier_transient_does_not_mask_terminal_failure(self):
        self.assertEqual(
            self.run_extract([TRANSIENT + "\nERROR: manifest unknown"]), (7, 1, [], [])
        )

    def test_other_transport_errors_recover(self):
        for error in [
            "ERROR: failed to solve: connection reset by peer",
            "ERROR: failed to solve: rpc error: transport is closing",
            "ERROR: failed to solve: unexpected status from HEAD request: 503 Service Unavailable",
        ]:
            with self.subTest(error=error):
                self.assertEqual(
                    self.run_extract([error, ""]), (0, 2, ["10"], ["attempt-2"])
                )

    def test_build_command_containing_transport_text_does_not_retry(self):
        error = 'ERROR: process "echo connection reset by peer" did not complete successfully: exit code: 1'
        self.assertEqual(self.run_extract([error]), (7, 1, [], []))

    def test_permanent_error_after_retry_stops(self):
        self.assertEqual(
            self.run_extract([TRANSIENT, "ERROR: manifest unknown"]), (7, 2, ["10"], [])
        )


if __name__ == "__main__":
    unittest.main()
