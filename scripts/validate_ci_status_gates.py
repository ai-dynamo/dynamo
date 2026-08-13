#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the aggregate `*-status-check` gates in .github/workflows/.

Those jobs are what branch protection requires, and each one collapses the
results of every job it needs into a single jq expression:

    jq -e 'to_entries | map(.value.result)
           | all(. as $result | ["success", "skipped"] | any($result == .))'

so the literal list of accepted results decides what can pass the merge gate.
A job killed by its own `timeout-minutes` is reported as `cancelled`, not
`failure`; accepting `cancelled` therefore turns a matrix of tests that never
finished into a green required check. `failure` is just as wrong, and jq's
`-e` flag is what turns a false result into a non-zero exit.

This checks that every status-check job runs such a gate and that its
allowlist is exactly the results that really mean nothing broke: `success`
and `skipped` (jobs gated off by the changed-files filters). Both directions
matter. Accepting more lets a broken job through; accepting less fails a job
the filters legitimately skipped. The allowlist is read only from an active
`jq -e` call that the `needs` context is actually piped into, so a gate that
has been commented out, quoted inside an `echo`, split from its input or given
`-n` (which makes jq ignore that input) is reported as missing rather than
trusted.

Usage:
    python3 validate_ci_status_gates.py [repo_root]
    python3 validate_ci_status_gates.py --test
"""

import json
import re
import sys
from pathlib import Path

import yaml

WORKFLOWS_DIR = Path(".github") / "workflows"
STATUS_CHECK_SUFFIX = "-status-check"
ACCEPTED_RESULTS = frozenset({"success", "skipped"})
# The `[...]` literal the jq expression tests each job result against, e.g.
# '["success", "skipped"] | any($result == .)'.
ALLOWLIST = re.compile(r"(\[[^\[\]]*\])\s*\|\s*any\(\s*\$result\s*==\s*\.\s*\)")
COMMENT_LINE = re.compile(r"(?m)^[ \t]*#.*$")
# A `jq` call and its single-quoted program, plus the short flags in front of
# it. Only `-e` makes a false result exit non-zero, so a gate without it
# reports success no matter what the jobs did.
JQ_CALL = re.compile(r"\bjq\b(?P<flags>(?:\s+-[^\s']+)*)\s+'(?P<program>[^']*)'")
SHORT_FLAGS = re.compile(r"^-[A-Za-z]+$")
NEEDS_CONTEXT = re.compile(r"toJson\(\s*needs\s*\)")
# `jq -n` builds its input from the program instead of reading stdin, so a
# gate spelled that way never sees the job results piped into it.
NULL_INPUT = "--null-input"
# What can carry the piped context into jq: a pipeline, or an input redirect
# (`< file`, `<<<"$results"`). Anything else between the two is not a feed.
FEEDS_JQ = ("|", "<")


def _commands(text):
    """Split a `run:` block into shell commands, honouring quotes.

    Newlines, `;`, `&&` and `||` separate commands, but only outside quotes: a
    jq program is single-quoted and routinely contains all of them. Yields
    `(offset, command)` so callers can tell where each one started.
    """
    commands, start, quote, i = [], 0, None, 0
    while i < len(text):
        char = text[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if char == "\\":
                i += 1
            elif char == '"':
                quote = None
        elif char == "\\":
            i += 1
        elif char in "'\"":
            quote = char
        elif char in "\n;":
            commands.append((start, text[start:i]))
            start = i + 1
        elif text[i : i + 2] in ("&&", "||"):
            commands.append((start, text[start:i]))
            i += 1
            start = i + 1
        i += 1
    commands.append((start, text[start:]))
    return commands


def _quoted(text, index):
    """Whether `index` in `text` sits inside a shell quote."""
    quote, i = None, 0
    while i < index:
        char = text[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if char == "\\":
                i += 1
            elif char == '"':
                quote = None
        elif char == "\\":
            i += 1
        elif char in "'\"":
            quote = char
        i += 1
    return quote is not None


def gate_allowlists(run):
    """Allowlists belonging to real gates in one `run:` block.

    An allowlist counts only where it can actually decide the job's exit
    status: inside the program of a `jq -e` call that the `needs` context is
    piped into, in that same command. The same text sitting in a comment, or
    echoed to stdout, or in a `jq` invocation that never receives the results,
    gates nothing and must not be mistaken for a gate that is missing.
    """
    allowlists = []
    for _, command in _commands(COMMENT_LINE.sub("", run)):
        context = NEEDS_CONTEXT.search(command)
        if not context:
            continue
        for call in JQ_CALL.finditer(command):
            if _quoted(command, call.start()):
                continue  # printed, not run
            flags = call.group("flags").split()
            if not any(SHORT_FLAGS.match(f) and "e" in f[1:] for f in flags):
                continue
            if any(
                f == NULL_INPUT or (SHORT_FLAGS.match(f) and "n" in f[1:])
                for f in flags
            ):
                continue  # jq builds its own input and ignores the results
            if call.start() < context.end():
                continue  # jq runs before the results are produced
            feed = command[context.end() : call.start()]
            if not any(op in feed for op in FEEDS_JQ):
                continue  # the results go somewhere else, not into this jq
            allowlists.extend(ALLOWLIST.findall(call.group("program")))
    return allowlists


def check_workflow(name, text, errors):
    """Check one workflow's status-check gates; return how many were found."""
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        errors.append(f"{name}: invalid YAML: {exc}")
        return 0

    jobs = (workflow or {}).get("jobs")
    if not isinstance(jobs, dict):
        return 0

    found = 0
    for job_id, job in sorted(jobs.items()):
        if not job_id.endswith(STATUS_CHECK_SUFFIX) or not isinstance(job, dict):
            continue
        found += 1
        steps = job.get("steps") or []
        runs = [s.get("run", "") for s in steps if isinstance(s, dict)]
        allowlists = [m for run in runs for m in gate_allowlists(run)]
        if not allowlists:
            errors.append(
                f"{name}: job '{job_id}' has no parseable jq result allowlist; "
                f"a status-check job must aggregate needs with "
                f"\"jq -e '... | any($result == .)'\""
            )
            continue
        for raw in allowlists:
            try:
                accepted = set(json.loads(raw))
            except json.JSONDecodeError:
                errors.append(
                    f"{name}: job '{job_id}' has an unreadable result allowlist {raw}"
                )
                continue
            for result in sorted(accepted - ACCEPTED_RESULTS):
                errors.append(
                    f"{name}: job '{job_id}' accepts '{result}' as a passing "
                    f"result; only {', '.join(sorted(ACCEPTED_RESULTS))} mean "
                    f"the jobs it gates actually ran"
                )
            # The contract runs both ways: a gate that drops 'skipped' turns
            # every job the changed-files filters legitimately gated off into
            # a red required check.
            for result in sorted(ACCEPTED_RESULTS - accepted):
                errors.append(
                    f"{name}: job '{job_id}' does not accept '{result}' as a "
                    f"passing result; the allowlist must be exactly "
                    f"{', '.join(sorted(ACCEPTED_RESULTS))}"
                )
    return found


def run_tests():
    """Self-test the gate parser against hand-written workflow bodies."""
    good = """
jobs:
  deploy-status-check:
    if: always()
    steps:
      - name: Check all deploy test jobs
        run: |
          # A comment above the gate must not hide it.
          echo '${{ toJson(needs) }}' | jq -e 'to_entries | map(.value.result) | all(. as $result | ["success", "skipped"] | any($result == .))'
"""
    cases = [
        ("clean gate", good, []),
        (
            "cancelled accepted",
            good.replace('"skipped"', '"skipped", "cancelled"'),
            ["accepts 'cancelled'"],
        ),
        (
            "failure accepted",
            good.replace('"skipped"', '"skipped", "failure"'),
            ["accepts 'failure'"],
        ),
        (
            "gate replaced by a bare echo",
            """
jobs:
  deploy-status-check:
    steps:
      - run: echo ok
""",
            ["no parseable jq result allowlist"],
        ),
        (
            "non-gate job is not our business",
            """
jobs:
  deploy-cleanup:
    steps:
      - run: jq -e 'all(. as $result | ["success", "cancelled"] | any($result == .))'
""",
            [],
        ),
        (
            "skipped dropped from the allowlist",
            good.replace(', "skipped"', ""),
            ["does not accept 'skipped'"],
        ),
        (
            "gate commented out",
            """
jobs:
  deploy-status-check:
    steps:
      - run: |
          # echo '${{ toJson(needs) }}' | jq -e 'all(. as $result | ["success", "skipped"] | any($result == .))'
          echo 'gate temporarily disabled'
""",
            ["no parseable jq result allowlist"],
        ),
        (
            "gate printed but never run",
            """
jobs:
  deploy-status-check:
    steps:
      - run: echo 'would check ${{ toJson(needs) }} against ["success", "skipped"] | any($result == .)'
""",
            ["no parseable jq result allowlist"],
        ),
        (
            "jq without -e always exits 0",
            good.replace("jq -e", "jq"),
            ["no parseable jq result allowlist"],
        ),
        (
            "jq -n ignores the results piped at it",
            good.replace("jq -e", "jq -ne"),
            ["no parseable jq result allowlist"],
        ),
        (
            "the gate reads the results but jq is fed nothing",
            """
jobs:
  deploy-status-check:
    steps:
      - run: |
          echo '${{ toJson(needs) }}' >/dev/null
          jq -e 'to_entries | map(.value.result) | all(. as $result | ["success", "skipped"] | any($result == .))'
""",
            ["no parseable jq result allowlist"],
        ),
        (
            "a jq call quoted inside an echo runs nothing",
            """
jobs:
  deploy-status-check:
    steps:
      - run: |
          echo "${{ toJson(needs) }} would be checked by jq -e 'all(. as $result | [\\"success\\", \\"skipped\\"] | any($result == .))'"
""",
            ["no parseable jq result allowlist"],
        ),
        (
            "a passive mention does not shadow the real gate",
            good.replace(
                "          # A comment above the gate must not hide it.\n",
                '          echo \'cancelled is not in ["success", "cancelled"] '
                "| any($result == .)'\n",
            ),
            [],
        ),
    ]

    failures = []
    for label, text, expected in cases:
        errors = []
        check_workflow("test.yaml", text, errors)
        if len(errors) != len(expected) or not all(
            fragment in error for fragment, error in zip(expected, errors, strict=True)
        ):
            failures.append(f"{label}: expected {expected}, got {errors}")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print(f"all {len(cases)} status-gate parser cases passed")
    return 0


def main():
    args = sys.argv[1:]
    if args and args[0] == "--test":
        return run_tests()

    root = Path(args[0]) if args else Path(__file__).parents[1]
    workflows = root / WORKFLOWS_DIR
    if not workflows.is_dir():
        print(f"error: {workflows} is not a directory", file=sys.stderr)
        return 1

    errors = []
    gates = 0
    paths = sorted(p for p in workflows.iterdir() if p.suffix in (".yml", ".yaml"))
    for path in paths:
        gates += check_workflow(
            str(path.relative_to(root)), path.read_text(encoding="utf-8"), errors
        )

    if not gates:
        errors.append(
            f"{WORKFLOWS_DIR}: no '*{STATUS_CHECK_SUFFIX}' job found; the merge "
            f"gates were renamed and this check is no longer validating anything"
        )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"\n{len(errors)} status-gate error(s)", file=sys.stderr)
        return 1
    print(f"validated {gates} CI status-check gates: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
