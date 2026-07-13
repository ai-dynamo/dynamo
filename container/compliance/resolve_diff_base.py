#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the baseline commit to diff a build's OSRB CSV against.

The baseline depends on where the build runs:

1. PR targeting ``main``            -> latest commit on ``main``.
2. Post-merge push to ``main``      -> the previous commit on ``main`` (HEAD~1).
3. PR targeting / push to a ``release/*`` branch
                                    -> the release tag ``vX.Y.Z`` that is the
                                       highest one strictly older than the
                                       current version. Semver order first
                                       (``1.3.0`` beats ``1.2.5`` for a ``1.3.1``
                                       build, regardless of publish date); when
                                       several tags share the same ``X.Y.Z`` base
                                       (e.g. ``v1.2.3-nemo-3`` vs ``v1.2.3-minimax``)
                                       the one published later wins.
4. Nightly build                    -> the previous successful scheduled run of
                                       the same workflow, found via the Actions
                                       API; manual ``workflow_dispatch`` runs are
                                       excluded and the selected run's
                                       ``head_sha`` names the baseline artifact.

Prints four ``key=value`` lines to stdout (suitable for ``>> "$GITHUB_OUTPUT"``)::

    base_sha=<40-hex-sha or empty>
    base_label=<human description of the baseline>
    base_artifact_id=<GitHub Actions artifact id or empty>
    base_run_id=<generating workflow run id or empty>

An empty ``base_sha`` is not an error: it means there is no baseline to diff
against (first release of a line, unrecognized context, or a commit that could
not be resolved). Callers degrade gracefully.

Usage:
    resolve_diff_base.py --event-context {pr|push} --current-branch <ref_name> \\
        --base-branch <baseRefName|''> --current-sha <sha> \\
        --current-version <X.Y.Z|''> --repo <owner/repo>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

# Matches a leading X.Y.Z in a tag/version, tolerant of any suffix
# (-rc6, -nemo-3, -minimax, .dev1, ...). We deliberately do NOT rely on PEP 440
# parsing because release tags like ``v1.2.3-nemo-3`` are not PEP 440 valid.
_RELEASE_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


def parse_release_tuple(text: str) -> tuple[int, int, int] | None:
    """Extract the (major, minor, patch) tuple from a tag/version, or None."""
    if not text:
        return None
    m = _RELEASE_RE.match(text.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def pick_prior_release_tag(
    current_version: str,
    tag_dates: list[tuple[str, int]],
) -> str | None:
    """Pick the release tag to diff a release build against.

    ``tag_dates`` is a list of ``(tag_name, unix_publish_date)``. Returns the
    chosen tag name, or None when no suitable prior tag exists.

    Selection: among tags whose ``X.Y.Z`` base is strictly less than the current
    version's base, take the highest base (semver). If several tags share that
    highest base, take the one published latest.
    """
    cur = parse_release_tuple(current_version)
    if cur is None:
        return None

    candidates: list[tuple[tuple[int, int, int], int, str]] = []
    for tag, date in tag_dates:
        base = parse_release_tuple(tag)
        if base is None:  # unparseable tag -> skip, never crash
            continue
        if base < cur:  # strictly older by semver base
            candidates.append((base, date, tag))
    if not candidates:
        return None

    # Highest base wins (semver first); within the same base, latest publish
    # date wins (chronological tie-break).
    best = max(candidates, key=lambda c: (c[0], c[1]))
    return best[2]


def is_release_branch(ref: str) -> bool:
    """True for a release branch ref such as ``release/1.3.0``."""
    return bool(ref) and ref.startswith("release/")


def _git(*args: str) -> str | None:
    """Run a read-only git command; return stripped stdout or None on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"git {' '.join(args)} failed: {exc}", file=sys.stderr)
        return None
    sha = out.stdout.strip()
    return sha or None


def _tag_dates() -> list[tuple[str, int]]:
    """List (tag, unix_date) for every ``v*`` tag. Empty on failure."""
    out = _git(
        "for-each-ref",
        "--format=%(refname:short)%09%(creatordate:unix)",
        "refs/tags/v*",
    )
    if not out:
        return []
    pairs: list[tuple[str, int]] = []
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        tag, date = parts[0].strip(), parts[1].strip()
        try:
            pairs.append((tag, int(date)))
        except ValueError:
            continue
    return pairs


def _gh(*args: str) -> str | None:
    """Run a `gh api` call; return stripped stdout or None on failure.

    Reads GH_TOKEN from the environment (set by the CI resolve step).
    """
    try:
        out = subprocess.run(
            ["gh", "api", *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"gh api {' '.join(args)} failed: {exc}", file=sys.stderr)
        return None
    return out.stdout.strip() or None


def pick_previous_run_sha(current_run_id: int, runs: list[dict]) -> str | None:
    """Return the head SHA of the most recent successful scheduled prior run.

    ``runs`` is a list of ``{"id", "head_sha", "event", "conclusion"}``
    dicts. Run ids increase over time, so "before this run" is
    ``id < current_run_id`` and "most recent" is the max such id. Requiring
    ``event == "schedule"`` keeps successful manual nightlies from becoming the
    baseline for the next cron-triggered run.
    """
    prev = [
        r
        for r in runs
        if isinstance(r.get("id"), int)
        and r["id"] < current_run_id
        and r.get("event") == "schedule"
        and r.get("conclusion") == "success"
    ]
    if not prev:
        return None
    best = max(prev, key=lambda r: r["id"])
    return best.get("head_sha") or None


def _fetch_workflow_runs(repo: str, current_run_id: int) -> list[dict]:
    """List recent runs of the current run's workflow. Empty on any failure."""
    workflow_id = _gh(
        f"/repos/{repo}/actions/runs/{current_run_id}", "--jq", ".workflow_id"
    )
    if not workflow_id:
        return []
    # per_page goes in the query string (not `-f`) so `gh api` stays a GET; a
    # `-f` field would flip it to POST and the runs list would never resolve.
    raw = _gh(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs"
        "?event=schedule&status=success&per_page=50",
        "--jq",
        "[.workflow_runs[] | {id, head_sha, event, conclusion}]",
    )
    if not raw:
        return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def _short(sha: str) -> str:
    return sha[:8] if sha else ""


def find_artifact(
    repo: str, sha: str, prefix: str
) -> tuple[tuple[str, str] | None, bool]:
    """Return ((artifact_id, run_id), api_error) for a compliance artifact."""
    if not (repo and sha and prefix):
        return None, False
    out = _gh(
        f"/repos/{repo}/actions/artifacts?name=compliance-{sha}-{prefix}",
        "--jq",
        "[.artifacts[] | select(.expired == false)] "
        "| sort_by(.created_at) | last "
        '| if . == null then "absent" '
        r'else "\(.id)\t\(.workflow_run.id)" end',
    )
    if out is None:
        return None, True
    if out == "absent":
        return None, False
    ids = out.split("\t", 1)
    if len(ids) != 2 or not all(value.isdigit() for value in ids):
        return None, True
    return (ids[0], ids[1]), False


def _first_parent_shas(ref: str, limit: int) -> list[str]:
    """First-parent commit SHAs reachable from ref (newest first), capped at limit."""
    out = _git("rev-list", "--first-parent", "-n", str(limit), ref)
    return out.splitlines() if out else []


def pick_commit_with_artifact(
    shas, find_fn
) -> tuple[str | None, tuple[str, str] | None, bool]:
    """Return (first matching sha, artifact ids, saw_error).

    saw_error is True if any lookup reported an API failure, letting the
    caller fall back instead of mistaking a flaky API for "no baseline".
    """
    saw_error = False
    for sha in shas:
        artifact, api_error = find_fn(sha)
        saw_error = saw_error or api_error
        if artifact:
            return sha, artifact, saw_error
    return None, None, saw_error


def _resolve_main_baseline(
    start_ref: str, repo: str, artifact_prefix: str, limit: int = 25
) -> tuple[str, str, str, str]:
    """Baseline for the main cases: the most recent commit on main's first-parent
    history (from ``start_ref`` backward) that actually has a
    ``compliance-<sha>-<prefix>`` artifact.

    Post-merge builds every framework, but each container's artifact lands at a
    different time (and a leg can fail), so the tip of main frequently lacks a
    given container's artifact when a PR builds. Walking back finds the newest
    commit that does have it, per container.
    """
    tip = _git("rev-parse", start_ref)
    if not tip:
        return (
            "",
            f"main baseline ({start_ref} unresolved; shallow clone?)",
            "",
            "",
        )
    # Without a prefix (or token) we can't check availability -> use the tip,
    # preserving the original behavior.
    if not artifact_prefix:
        return tip, f"main {_short(tip)}", "", ""
    shas = _first_parent_shas(start_ref, limit)
    chosen, artifact, saw_error = pick_commit_with_artifact(
        shas, lambda s: find_artifact(repo, s, artifact_prefix)
    )
    if chosen and artifact:
        artifact_id, run_id = artifact
        if chosen == tip:
            return chosen, f"main {_short(chosen)}", artifact_id, run_id
        return (
            chosen,
            f"main {_short(chosen)} (nearest with {artifact_prefix} artifact)",
            artifact_id,
            run_id,
        )
    if saw_error:
        # Preserve the candidate SHA for diagnostics. Without an artifact id,
        # the download soft-degrades to a baseline_unavailable diff.
        return tip, f"main {_short(tip)} (artifact availability unchecked)", "", ""
    return (
        "",
        f"no main commit with a {artifact_prefix} artifact in last {limit}",
        "",
        "",
    )


def _with_artifact(
    sha: str, label: str, repo: str, artifact_prefix: str
) -> tuple[str, str, str, str]:
    """Attach the exact artifact and run ids to a resolved baseline."""
    if not (sha and artifact_prefix):
        return sha, label, "", ""
    artifact, _ = find_artifact(repo, sha, artifact_prefix)
    if not artifact:
        return sha, label, "", ""
    artifact_id, run_id = artifact
    return sha, label, artifact_id, run_id


def resolve(
    event_context: str,
    current_branch: str,
    base_branch: str,
    current_version: str,
    repo: str = "",
    current_run_id: str = "",
    artifact_prefix: str = "",
) -> tuple[str, str, str, str]:
    """Return (base_sha, base_label, artifact_id, run_id)."""
    # Rule 4: nightly build -> previous successful scheduled run of the same
    # workflow. Manual workflow_dispatch runs are deliberately excluded.
    if event_context == "nightly":
        try:
            rid = int(current_run_id)
        except (TypeError, ValueError):
            return "", "nightly baseline (no current run id)", "", ""
        sha = pick_previous_run_sha(rid, _fetch_workflow_runs(repo, rid))
        if sha:
            return _with_artifact(
                sha,
                f"previous scheduled nightly {_short(sha)}",
                repo,
                artifact_prefix,
            )
        return "", "no previous successful scheduled nightly run found", "", ""

    # Rule 1: PR targeting main -> most recent main commit that has this
    # container's artifact (walk back main; the tip often isn't built yet).
    if event_context == "pr" and base_branch == "main":
        ref = "origin/main" if _git("rev-parse", "origin/main") else "main"
        return _resolve_main_baseline(ref, repo, artifact_prefix)

    # Rule 2: post-merge push to main -> walk back from the previous main commit
    # (HEAD is the just-merged commit, still building, so start at HEAD~1).
    if event_context == "push" and current_branch == "main":
        return _resolve_main_baseline("HEAD~1", repo, artifact_prefix)

    # Rule 3: PR targeting / push to a release branch -> prior release tag.
    release_ref = base_branch if is_release_branch(base_branch) else current_branch
    if is_release_branch(release_ref):
        if not parse_release_tuple(current_version):
            return (
                "",
                f"release baseline (no parseable current version '{current_version}')",
                "",
                "",
            )
        tag = pick_prior_release_tag(current_version, _tag_dates())
        if tag is None:
            return (
                "",
                f"no prior release tag older than {current_version} (first of its line)",
                "",
                "",
            )
        sha = _git("rev-list", "-n", "1", tag)
        if sha:
            return _with_artifact(
                sha,
                f"release baseline {tag} ({_short(sha)})",
                repo,
                artifact_prefix,
            )
        return "", f"release baseline {tag} (could not resolve tag commit)", "", ""

    # Fallback: unrecognized context -> no baseline.
    return "", "no baseline (unrecognized build context)", "", ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve the baseline commit to diff a build's OSRB CSV against",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--event-context", required=True, choices=["pr", "push", "nightly"]
    )
    parser.add_argument(
        "--current-branch",
        default="",
        help="github.ref_name (main, release/X.Y.Z, ...)",
    )
    parser.add_argument(
        "--base-branch",
        default="",
        help="For PRs, the PR's target branch (gh pr view --json baseRefName); empty for pushes",
    )
    parser.add_argument("--current-sha", default="", help="github.sha (informational)")
    parser.add_argument(
        "--current-version",
        default="",
        help="Current X.Y.Z version (from pyproject.toml); required for release baselines",
    )
    parser.add_argument(
        "--repo", default="", help="owner/repo (required for the nightly context)"
    )
    parser.add_argument(
        "--current-run-id",
        default="",
        help="github.run_id of this build; used by the nightly context to find the previous run",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="",
        help="Container artifact prefix (target_tag_plain, e.g. vllm-runtime); for the "
        "main cases, the resolver walks main's history for a commit that has a "
        "compliance-<sha>-<prefix> artifact. Requires --repo and GH_TOKEN.",
    )
    args = parser.parse_args()

    base_sha, base_label, base_artifact_id, base_run_id = resolve(
        args.event_context,
        args.current_branch,
        args.base_branch,
        args.current_version,
        args.repo,
        args.current_run_id,
        args.artifact_prefix,
    )

    print(
        f"Resolved diff baseline: {base_label} (sha={base_sha or 'none'})",
        file=sys.stderr,
    )
    # stdout carries ONLY the GITHUB_OUTPUT key=value lines.
    print(f"base_sha={base_sha}")
    print(f"base_label={base_label}")
    print(f"base_artifact_id={base_artifact_id}")
    print(f"base_run_id={base_run_id}")


if __name__ == "__main__":
    main()
