#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve an exact, trusted workflow run and artifact for an OSRB comparison.

PRs use their stable merge-base and main pushes their first-parent predecessor.
Release branches retain the highest strictly older release-tag baseline.
Nightlies use the prior scheduled run with the matching artifact, regardless
of unrelated workflow failures. Missing or invalid evidence fails closed;
first-of-kind bootstrapping requires an explicit flag.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from urllib.parse import urlencode

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


class BaselineError(RuntimeError):
    """A comparison cannot be established safely."""


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _tag_dates() -> list[tuple[str, int]]:
    out = _git(
        "for-each-ref",
        "--format=%(refname:short)%09%(creatordate:unix)",
        "refs/tags/v*",
    )
    return [
        (tag, int(date))
        for tag, date in (line.split("\t") for line in out.splitlines())
    ]


def _gh_json(endpoint: str) -> dict:
    """API and decoding failures are errors, never evidence that a baseline is absent."""
    result = subprocess.run(
        ["gh", "api", endpoint], capture_output=True, text=True, check=True
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise BaselineError(f"Unexpected GitHub API response for {endpoint}")
    return payload


def _short(sha: str) -> str:
    return sha[:8]


def _run_matches(run: dict, *, sha: str, branch: str, workflow: str, repo: str) -> bool:
    return (
        run["head_sha"] == sha
        and run["head_branch"] == branch
        and run["path"].split("@")[0] == workflow
        and run["event"] == "push"
        and run["head_repository"]["full_name"] == repo
    )


def find_run_artifact(repo: str, run: dict, prefix: str) -> tuple[str | None, bool]:
    """Return an exact run's artifact id and whether an expired copy was found.

    Never search repository-wide by SHA: a manual or PR run of the same commit
    may have different image inputs. Pagination is necessary on large matrix runs.
    """
    name = f"compliance-{run['head_sha']}-{prefix}"
    page = 1
    matches = []
    expired = False
    while True:
        payload = _gh_json(
            f"/repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100&page={page}"
        )
        artifacts = payload["artifacts"]
        for artifact in artifacts:
            if artifact["name"] != name:
                continue
            provenance = artifact["workflow_run"]
            if (
                provenance["id"] != run["id"]
                or provenance["head_sha"] != run["head_sha"]
            ):
                raise BaselineError(
                    f"Artifact {artifact['id']} has inconsistent run provenance"
                )
            if artifact["expired"]:
                expired = True
            elif artifact["size_in_bytes"] <= 0:
                raise BaselineError(f"Baseline artifact {artifact['id']} is empty")
            else:
                matches.append(str(artifact["id"]))
        if page * 100 >= payload["total_count"]:
            break
        page += 1
    if len(matches) > 1:
        raise BaselineError(f"Ambiguous artifact {name} in run {run['id']}")
    return (matches[0] if matches else None), expired


def _fetch_workflow_runs(repo: str, current_run_id: int) -> list[dict]:
    current = _gh_json(f"/repos/{repo}/actions/runs/{current_run_id}")
    if current["head_repository"]["full_name"] != repo:
        raise BaselineError(
            "Nightly workflow does not belong to the expected repository"
        )
    query = urlencode(
        {"event": "schedule", "branch": current["head_branch"], "per_page": 100}
    )
    payload = _gh_json(
        f"/repos/{repo}/actions/workflows/{current['workflow_id']}/runs?{query}"
    )
    # Overall workflow success is deliberately irrelevant: unrelated test or
    # publication legs can fail after this container's compliance artifact lands.
    return sorted(
        (
            run
            for run in payload["workflow_runs"]
            if run["id"] < current_run_id
            and run["event"] == "schedule"
            and run["workflow_id"] == current["workflow_id"]
            and run["head_branch"] == current["head_branch"]
            and run["head_repository"]["full_name"] == repo
            and run["status"] == "completed"
        ),
        key=lambda run: run["id"],
        reverse=True,
    )


def _find_commit_artifact(
    repo: str, sha: str, prefix: str, branch: str
) -> tuple[tuple[str, str] | None, bool]:
    workflow = ".github/workflows/post-merge-ci.yml"
    query = urlencode(
        {"head_sha": sha, "branch": branch, "event": "push", "per_page": 100}
    )
    runs = _gh_json(f"/repos/{repo}/actions/workflows/post-merge-ci.yml/runs?{query}")[
        "workflow_runs"
    ]
    expired = False
    for run in sorted(runs, key=lambda run: run["id"], reverse=True):
        if not _run_matches(run, sha=sha, branch=branch, workflow=workflow, repo=repo):
            continue
        artifact_id, was_expired = find_run_artifact(repo, run, prefix)
        expired |= was_expired
        if artifact_id:
            return (artifact_id, str(run["id"])), expired
    return None, expired


def _missing(
    label: str, *, allow_bootstrap: bool, expired: bool = False
) -> tuple[str, str, str, str]:
    if allow_bootstrap and not expired:
        return "", f"explicit bootstrap: {label}", "", ""
    raise BaselineError(label + (" (prior artifact expired)" if expired else ""))


def _resolve_main_baseline(
    start_ref: str,
    repo: str,
    artifact_prefix: str,
    label: str = "main",
    limit: int = 25,
    allow_bootstrap: bool = False,
) -> tuple[str, str, str, str]:
    """Walk only first-parent history from the stable PR fork or previous main commit."""
    shas = _git("rev-list", "--first-parent", "-n", str(limit), start_ref).splitlines()
    expired = False
    for sha in shas:
        artifact, was_expired = _find_commit_artifact(
            repo, sha, artifact_prefix, "main"
        )
        expired |= was_expired
        if artifact:
            artifact_id, run_id = artifact
            return (
                sha,
                f"{label} {_short(sha)} (nearest with {artifact_prefix} artifact)",
                artifact_id,
                run_id,
            )
    return _missing(
        f"no {label} commit with a {artifact_prefix} artifact in last {limit}",
        allow_bootstrap=allow_bootstrap,
        expired=expired,
    )


def resolve(
    event_context: str,
    current_branch: str,
    base_branch: str,
    current_version: str,
    repo: str = "",
    current_run_id: str = "",
    artifact_prefix: str = "",
    merge_base_sha: str = "",
    allow_bootstrap: bool = False,
) -> tuple[str, str, str, str]:
    """Return a validated (base_sha, label, artifact_id, run_id), or raise."""
    if not repo or not artifact_prefix:
        raise BaselineError("Both repository and artifact prefix are required")
    if event_context == "nightly":
        if not current_run_id.isdigit():
            raise BaselineError("Nightly baseline requires the current run id")
        expired = False
        for run in _fetch_workflow_runs(repo, int(current_run_id)):
            artifact_id, was_expired = find_run_artifact(repo, run, artifact_prefix)
            expired |= was_expired
            if artifact_id:
                sha = run["head_sha"]
                return (
                    sha,
                    f"previous scheduled nightly {_short(sha)}",
                    artifact_id,
                    str(run["id"]),
                )
        return _missing(
            "no prior scheduled nightly with the matching compliance artifact in the last 100 runs",
            allow_bootstrap=allow_bootstrap,
            expired=expired,
        )
    if event_context == "pr" and base_branch == "main":
        if not merge_base_sha:
            raise BaselineError("PR merge-base unavailable")
        return _resolve_main_baseline(
            merge_base_sha,
            repo,
            artifact_prefix,
            label="PR merge-base",
            allow_bootstrap=allow_bootstrap,
        )
    if event_context == "push" and current_branch == "main":
        return _resolve_main_baseline(
            "HEAD~1", repo, artifact_prefix, allow_bootstrap=allow_bootstrap
        )
    release_ref = base_branch if is_release_branch(base_branch) else current_branch
    if is_release_branch(release_ref):
        if not parse_release_tuple(current_version):
            raise BaselineError(
                f"No parseable current release version: {current_version}"
            )
        tag = pick_prior_release_tag(current_version, _tag_dates())
        if tag is None:
            return _missing(
                f"no prior release tag older than {current_version}",
                allow_bootstrap=allow_bootstrap,
            )
        sha = _git("rev-list", "-n", "1", tag)
        # Keep vendor/dev suffixes: their artifacts come from the matching
        # release branch, not the unrelated plain X.Y.Z release line.
        branch = "release/" + tag.removeprefix("v")
        artifact, expired = _find_commit_artifact(repo, sha, artifact_prefix, branch)
        if artifact:
            return (
                sha,
                f"release baseline {tag} ({_short(sha)})",
                artifact[0],
                artifact[1],
            )
        return _missing(
            f"release baseline {tag} has no matching artifact",
            allow_bootstrap=allow_bootstrap,
            expired=expired,
        )
    raise BaselineError(
        "Unrecognized build context; a comparison cannot be silently disabled"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-context", required=True, choices=["pr", "push", "nightly"]
    )
    parser.add_argument("--current-branch", default="")
    parser.add_argument("--base-branch", default="")
    parser.add_argument("--merge-base-sha", default="")
    parser.add_argument("--current-sha", default="", help="Source SHA (informational)")
    parser.add_argument("--current-version", default="")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--current-run-id", default="")
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Explicit first-of-kind bootstrap; never bypass API errors or expired evidence",
    )
    args = parser.parse_args()
    try:
        result = resolve(
            args.event_context,
            args.current_branch,
            args.base_branch,
            args.current_version,
            args.repo,
            args.current_run_id,
            args.artifact_prefix,
            args.merge_base_sha,
            args.allow_bootstrap,
        )
    except (
        BaselineError,
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        parser.exit(1, f"Baseline resolution failed: {exc}\n")
    sha, label, artifact_id, run_id = result
    print(
        f"Resolved diff baseline: {label} (sha={sha or 'bootstrap'})", file=sys.stderr
    )
    for key, value in zip(
        ("base_sha", "base_label", "base_artifact_id", "base_run_id"), result
    ):
        print(f"{key}={value}")
    print(f"base_bootstrap={'true' if not sha else 'false'}")


if __name__ == "__main__":
    main()
