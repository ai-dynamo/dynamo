#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize JUnit XML artifacts from GitHub Actions history."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import io
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

UTC = dt.timezone.utc
API_ROOT = "https://api.github.com"
DEFAULT_REPO = os.environ.get("GITHUB_REPOSITORY", "ai-dynamo/dynamo")
DEFAULT_ARTIFACT_PREFIXES = ("junit-", "test-results-")
DEFAULT_PRESETS = {
    "nightly": {
        "workflow": "Nightly CI Pipeline",
        "event": "schedule",
        "branch": "main",
    },
    "post-merge": {
        "workflow": "Post-Merge CI Pipeline",
        "event": "push",
        "branch": "main",
    },
}


def default_cache_dir() -> pathlib.Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return pathlib.Path(xdg_cache_home).expanduser() / "dynamo-junit-history"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "dynamo-junit-history"
    return pathlib.Path.home() / ".cache" / "dynamo-junit-history"


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
        return None


@dataclasses.dataclass(frozen=True)
class WorkflowQuery:
    label: str
    workflow: str
    event: str | None
    branch: str | None


@dataclasses.dataclass(frozen=True)
class RunInfo:
    workflow: str
    run_id: int
    attempt: int
    created_at: dt.datetime
    event: str
    branch: str
    conclusion: str
    url: str


@dataclasses.dataclass(frozen=True)
class JUnitStats:
    workflow: str
    run_id: int
    run_attempt: int
    run_created_at: dt.datetime
    run_conclusion: str
    artifact_name: str
    xml_path: str
    tests: int
    failures: int
    errors: int
    skipped: int
    seconds: float
    failed_tests: tuple[str, ...]


class GithubClient:
    def __init__(self, repo: str, token: str | None) -> None:
        self.repo = repo
        self.token = token

    def request_json(
        self, path_or_url: str, params: dict[str, str | int] | None = None
    ) -> object:
        data, _headers = self.request(path_or_url, params=params)
        import json

        return json.loads(data.decode("utf-8"))

    def request_bytes(
        self, path_or_url: str, params: dict[str, str | int] | None = None
    ) -> bytes:
        data, _headers = self.request(path_or_url, params=params)
        return data

    def request_artifact_archive(self, archive_url: str) -> bytes:
        request = urllib.request.Request(
            archive_url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "dynamo-junit-history",
                **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
            },
        )
        opener = urllib.request.build_opener(NoRedirectHandler)
        try:
            with opener.open(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                message = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"GitHub artifact request failed: HTTP {exc.code} for {archive_url}\n{message}"
                ) from exc
            redirect_url = exc.headers.get("Location")
            if not redirect_url:
                raise RuntimeError(
                    f"GitHub artifact request returned HTTP {exc.code} without Location"
                ) from exc

        # The redirected object-store URL is already signed; GitHub auth headers break Azure blob downloads.
        redirected = urllib.request.Request(
            redirect_url, headers={"User-Agent": "dynamo-junit-history"}
        )
        with urllib.request.urlopen(redirected, timeout=60) as response:
            return response.read()

    def request(
        self,
        path_or_url: str,
        params: dict[str, str | int] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{API_ROOT}{path_or_url}"
        )
        if params:
            query = urllib.parse.urlencode(params)
            separator = "&" if urllib.parse.urlsplit(url).query else "?"
            url = f"{url}{separator}{query}"

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "dynamo-junit-history",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub API request failed: HTTP {exc.code} for {url}\n{message}"
            ) from exc

    def paginate(
        self, path: str, key: str, params: dict[str, str | int] | None = None
    ) -> list[dict[str, object]]:
        page = 1
        items: list[dict[str, object]] = []
        while True:
            page_params = dict(params or {})
            page_params["per_page"] = 100
            page_params["page"] = page
            payload = self.request_json(path, page_params)
            if not isinstance(payload, dict) or key not in payload:
                raise RuntimeError(
                    f"Unexpected GitHub API payload for {path}: missing {key}"
                )
            batch = payload[key]
            if not isinstance(batch, list):
                raise RuntimeError(
                    f"Unexpected GitHub API payload for {path}: {key} is not a list"
                )
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return items


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def token_from_environment_or_gh() -> str | None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    token = result.stdout.strip()
    return token or None


def resolve_workflow(client: GithubClient, workflow_spec: str) -> str:
    if workflow_spec.isdigit():
        return workflow_spec

    workflows = client.paginate(f"/repos/{client.repo}/actions/workflows", "workflows")
    matches: list[dict[str, object]] = []
    for workflow in workflows:
        name = str(workflow.get("name", ""))
        path = str(workflow.get("path", ""))
        basename = path.rsplit("/", 1)[-1]
        if workflow_spec in {name, path, basename}:
            matches.append(workflow)

    if len(matches) == 1:
        return str(matches[0]["id"])
    if not matches:
        known = ", ".join(
            sorted(str(w.get("name", "")) for w in workflows if w.get("name"))
        )
        raise RuntimeError(
            f"Could not resolve workflow {workflow_spec!r}. Known workflows: {known}"
        )
    raise RuntimeError(
        f"Workflow {workflow_spec!r} is ambiguous; use the workflow file or numeric ID."
    )


def list_runs(
    client: GithubClient,
    query: WorkflowQuery,
    cutoff: dt.datetime,
    include_reruns: bool,
    max_runs: int,
) -> list[RunInfo]:
    workflow_id = resolve_workflow(client, query.workflow)
    params: dict[str, str | int] = {
        "created": f">={cutoff.isoformat().replace('+00:00', 'Z')}",
    }
    if query.event:
        params["event"] = query.event
    if query.branch:
        params["branch"] = query.branch

    path = f"/repos/{client.repo}/actions/workflows/{urllib.parse.quote(workflow_id, safe='')}/runs"
    raw_runs = client.paginate(path, "workflow_runs", params)
    runs: list[RunInfo] = []
    for raw_run in raw_runs:
        created_at = parse_time(str(raw_run["created_at"]))
        if created_at < cutoff:
            continue
        attempt = int(raw_run.get("run_attempt") or 1)
        if attempt > 1 and not include_reruns:
            continue
        runs.append(
            RunInfo(
                workflow=query.label,
                run_id=int(raw_run["id"]),
                attempt=attempt,
                created_at=created_at,
                event=str(raw_run.get("event") or ""),
                branch=str(raw_run.get("head_branch") or ""),
                conclusion=str(
                    raw_run.get("conclusion") or raw_run.get("status") or ""
                ),
                url=str(raw_run.get("html_url") or ""),
            )
        )
        if max_runs > 0 and len(runs) >= max_runs:
            break
    return runs


def list_artifacts(client: GithubClient, run_id: int) -> list[dict[str, object]]:
    return client.paginate(
        f"/repos/{client.repo}/actions/runs/{run_id}/artifacts", "artifacts"
    )


def artifact_matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def download_artifact_zip(
    client: GithubClient,
    artifact: dict[str, object],
    cache_dir: pathlib.Path,
    refresh_cache: bool,
    no_cache: bool,
) -> bytes:
    artifact_id = int(artifact["id"])
    cache_path = cache_dir / f"artifact-{artifact_id}.zip"
    if not no_cache and cache_path.exists() and not refresh_cache:
        return cache_path.read_bytes()

    archive_url = str(artifact["archive_download_url"])
    payload = client.request_artifact_archive(archive_url)
    if not no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
    return payload


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def int_attr(element: ET.Element, name: str) -> int:
    try:
        return int(float(element.attrib.get(name, "0")))
    except ValueError:
        return 0


def float_attr(element: ET.Element, name: str) -> float:
    try:
        return float(element.attrib.get(name, "0") or 0)
    except ValueError:
        return 0.0


def testcase_name(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname", "").strip()
    name = testcase.attrib.get("name", "").strip()
    file_name = testcase.attrib.get("file", "").strip()
    if classname and name:
        return f"{classname}.{name}"
    if name:
        return name
    if file_name:
        return file_name
    return "<unnamed testcase>"


def parse_junit_xml(
    xml_bytes: bytes,
    run: RunInfo,
    artifact_name: str,
    xml_path: str,
) -> JUnitStats | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        print(
            f"warning: could not parse {artifact_name}/{xml_path}: {exc}",
            file=sys.stderr,
        )
        return None

    testcases = [
        element for element in root.iter() if local_name(element.tag) == "testcase"
    ]
    tests = len(testcases)
    failures = 0
    errors = 0
    skipped = 0
    seconds = 0.0
    failed_tests: list[str] = []

    for testcase in testcases:
        children = {local_name(child.tag) for child in list(testcase)}
        seconds += float_attr(testcase, "time")
        if "error" in children:
            errors += 1
            failed_tests.append(testcase_name(testcase))
        elif "failure" in children:
            failures += 1
            failed_tests.append(testcase_name(testcase))
        elif "skipped" in children:
            skipped += 1

    # Some JUnit producers only emit suite-level counters. Use them when no cases exist.
    if not testcases:
        suites = [
            element for element in root.iter() if local_name(element.tag) == "testsuite"
        ] or [root]
        tests = sum(int_attr(suite, "tests") for suite in suites)
        failures = sum(int_attr(suite, "failures") for suite in suites)
        errors = sum(int_attr(suite, "errors") for suite in suites)
        skipped = sum(int_attr(suite, "skipped") for suite in suites)
        seconds = sum(float_attr(suite, "time") for suite in suites)

    if tests == 0 and failures == 0 and errors == 0 and skipped == 0:
        return None

    return JUnitStats(
        workflow=run.workflow,
        run_id=run.run_id,
        run_attempt=run.attempt,
        run_created_at=run.created_at,
        run_conclusion=run.conclusion,
        artifact_name=artifact_name,
        xml_path=xml_path,
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        seconds=seconds,
        failed_tests=tuple(failed_tests),
    )


def parse_artifact(
    zip_bytes: bytes,
    run: RunInfo,
    artifact_name: str,
) -> list[JUnitStats]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        print(
            f"warning: could not read artifact {artifact_name}: {exc}", file=sys.stderr
        )
        return []

    stats: list[JUnitStats] = []
    for member in archive.namelist():
        if member.endswith("/") or not member.lower().endswith(".xml"):
            continue
        with archive.open(member) as file:
            parsed = parse_junit_xml(file.read(), run, artifact_name, member)
        if parsed:
            stats.append(parsed)
    return stats


def collect_stats(
    client: GithubClient,
    queries: list[WorkflowQuery],
    cutoff: dt.datetime,
    args: argparse.Namespace,
) -> tuple[list[RunInfo], list[JUnitStats]]:
    all_runs: list[RunInfo] = []
    all_stats: list[JUnitStats] = []

    for query in queries:
        runs = list_runs(client, query, cutoff, args.include_reruns, args.max_runs)
        all_runs.extend(runs)
        print(f"found {len(runs)} runs for {query.label}", file=sys.stderr)

        for run in runs:
            artifacts = list_artifacts(client, run.run_id)
            matched = [
                artifact
                for artifact in artifacts
                if not artifact.get("expired")
                and artifact_matches(
                    str(artifact.get("name", "")), args.artifact_prefix
                )
            ]
            print(
                f"run {run.run_id} {run.created_at.date()} {query.label}: "
                f"{len(matched)} matching artifacts",
                file=sys.stderr,
            )
            for artifact in matched:
                name = str(artifact["name"])
                try:
                    zip_bytes = download_artifact_zip(
                        client,
                        artifact,
                        args.cache_dir,
                        args.refresh_cache,
                        args.no_cache,
                    )
                except RuntimeError as exc:
                    print(
                        f"warning: could not download artifact {name}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                all_stats.extend(parse_artifact(zip_bytes, run, name))

    return all_runs, all_stats


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def build_buckets(
    runs: list[RunInfo], stats: list[JUnitStats]
) -> dict[tuple[str, str], dict[str, object]]:
    buckets: dict[tuple[str, str], dict[str, object]] = collections.defaultdict(
        lambda: {
            "runs": set(),
            "conclusions": collections.Counter(),
            "artifacts": set(),
            "xmls": 0,
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "seconds": 0.0,
        }
    )

    for run in runs:
        key = (run.created_at.date().isoformat(), run.workflow)
        bucket = buckets[key]
        bucket["runs"].add(run.run_id)
        bucket["conclusions"][run.conclusion or "unknown"] += 1

    for item in stats:
        key = (item.run_created_at.date().isoformat(), item.workflow)
        bucket = buckets[key]
        bucket["runs"].add(item.run_id)
        bucket["artifacts"].add((item.run_id, item.artifact_name))
        bucket["xmls"] += 1
        bucket["tests"] += item.tests
        bucket["failures"] += item.failures
        bucket["errors"] += item.errors
        bucket["skipped"] += item.skipped
        bucket["seconds"] += item.seconds

    return buckets


def conclusion_summary(counter: collections.Counter[str]) -> str:
    if not counter:
        return "-"
    parts = []
    for key in ("success", "failure", "cancelled", "timed_out", "skipped", "unknown"):
        count = counter.get(key, 0)
        if count:
            parts.append(f"{key}:{count}")
    for key, count in sorted(counter.items()):
        if key not in {
            "success",
            "failure",
            "cancelled",
            "timed_out",
            "skipped",
            "unknown",
        }:
            parts.append(f"{key}:{count}")
    return ",".join(parts)


def print_ascii_report(
    repo: str,
    cutoff: dt.datetime,
    runs: list[RunInfo],
    stats: list[JUnitStats],
    top: int,
) -> None:
    buckets = build_buckets(runs, stats)
    print(f"Repo: {repo}")
    print(f"Window: since {cutoff.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Runs: {len(runs)}")
    print(f"JUnit XML files: {len(stats)}")
    print()

    header = (
        f"{'Date':<10}  {'Workflow':<24} {'Runs':>4} {'Artifacts':>9} {'XMLs':>5} "
        f"{'Cases':>7} {'Pass':>7} {'Fail':>5} {'Err':>4} {'Skip':>6} {'Time':>8}  Conclusions"
    )
    print(header)
    print("-" * len(header))

    for (date, workflow), bucket in sorted(buckets.items()):
        tests = int(bucket["tests"])
        failures = int(bucket["failures"])
        errors = int(bucket["errors"])
        skipped = int(bucket["skipped"])
        passed = max(0, tests - failures - errors - skipped)
        print(
            f"{date:<10}  {shorten(workflow, 24):<24} "
            f"{len(bucket['runs']):>4} {len(bucket['artifacts']):>9} {int(bucket['xmls']):>5} "
            f"{tests:>7} {passed:>7} {failures:>5} {errors:>4} {skipped:>6} "
            f"{format_seconds(float(bucket['seconds'])):>8}  "
            f"{conclusion_summary(bucket['conclusions'])}"
        )

    failure_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    for item in stats:
        for failed_test in item.failed_tests:
            failure_counts[(item.workflow, failed_test)] += 1

    print()
    if not failure_counts:
        print("Top failures: none")
        return

    print(f"Top failures/errors by testcase over this window (top {top}):")
    print(f"{'Count':>5}  {'Workflow':<24} Testcase")
    print("-" * 96)
    for (workflow, test_name), count in failure_counts.most_common(top):
        print(f"{count:>5}  {shorten(workflow, 24):<24} {shorten(test_name, 62)}")


def aggregate_daily(stats: list[JUnitStats]) -> dict[str, dict[str, int]]:
    daily: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    )
    for item in stats:
        date = item.run_created_at.date().isoformat()
        daily[date]["tests"] += item.tests
        daily[date]["failures"] += item.failures
        daily[date]["errors"] += item.errors
        daily[date]["skipped"] += item.skipped
    return daily


def write_failure_svg(path: pathlib.Path, stats: list[JUnitStats]) -> None:
    daily = aggregate_daily(stats)
    dates = sorted(daily)
    width = 980
    height = 360
    margin_left = 58
    margin_right = 24
    margin_top = 28
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not dates:
        body = '<text x="40" y="80" font-size="16">No JUnit data matched.</text>'
        path.write_text(svg_document(width, height, body), encoding="utf-8")
        return

    bad_counts = [daily[date]["failures"] + daily[date]["errors"] for date in dates]
    max_bad = max(max(bad_counts), 1)
    bar_gap = 4
    bar_width = max(2, (plot_width - bar_gap * (len(dates) - 1)) / len(dates))
    elements = [
        f'<text x="{margin_left}" y="18" font-size="15" font-weight="600">'
        "JUnit failures/errors per day</text>",
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" stroke="#222"/>',
    ]

    for tick in range(0, 5):
        value = round(max_bad * tick / 4)
        y = margin_top + plot_height - (value / max_bad * plot_height)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="10" y="{y + 4:.1f}" font-size="11" fill="#555">{value}</text>'
        )

    for index, date in enumerate(dates):
        bad = bad_counts[index]
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = 0 if bad == 0 else max(2, bad / max_bad * plot_height)
        y = margin_top + plot_height - bar_height
        color = "#16a34a" if bad == 0 else "#dc2626"
        elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}"><title>{date}: {bad} failures/errors, '
            f'{daily[date]["tests"]} cases</title></rect>'
        )
        if len(dates) <= 16 or index % max(1, len(dates) // 12) == 0:
            label = date[5:]
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{height - 45}" font-size="10" '
                f'text-anchor="middle" transform="rotate(45 {x + bar_width / 2:.1f},{height - 45})">'
                f"{label}</text>"
            )
        if bad:
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 4:.1f}" font-size="10" '
                f'text-anchor="middle" fill="#7f1d1d">{bad}</text>'
            )

    path.write_text(svg_document(width, height, "\n".join(elements)), encoding="utf-8")


def svg_document(width: int, height: int, body: str) -> str:
    return textwrap.dedent(
        f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
          <rect width="100%" height="100%" fill="white"/>
          <style>
            text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
          </style>
          {body}
        </svg>
        """
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions JUnit XML artifacts and summarize recent CI history.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="GitHub repository as owner/name."
    )
    parser.add_argument("--days", type=int, default=30, help="History window in days.")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(DEFAULT_PRESETS),
        help="Reliable workflow preset. Defaults to nightly and post-merge when no workflow is supplied.",
    )
    parser.add_argument(
        "--workflow",
        action="append",
        help="Workflow name, path, basename, or numeric workflow ID. Overrides default presets when supplied.",
    )
    parser.add_argument(
        "--event", help="Event filter for --workflow, for example schedule or push."
    )
    parser.add_argument(
        "--branch", default="main", help="Branch filter for --workflow."
    )
    parser.add_argument(
        "--artifact-prefix",
        action="append",
        help="Artifact name prefix to parse. May be repeated.",
    )
    parser.add_argument(
        "--include-reruns",
        action="store_true",
        help="Include workflow runs with run_attempt > 1.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs per workflow query; 0 means unlimited.",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=default_cache_dir(),
        help="Artifact ZIP cache directory.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download artifacts even if cached.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the artifact cache.",
    )
    parser.add_argument(
        "--plot", type=pathlib.Path, help="Write a simple SVG failure history plot."
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Number of failing testcases to show."
    )
    return parser.parse_args()


def workflow_queries(args: argparse.Namespace) -> list[WorkflowQuery]:
    if args.workflow:
        return [
            WorkflowQuery(
                label=re.sub(r"\.ya?ml$", "", workflow.rsplit("/", 1)[-1]),
                workflow=workflow,
                event=args.event,
                branch=args.branch,
            )
            for workflow in args.workflow
        ]

    presets = args.preset or ["nightly", "post-merge"]
    queries = []
    for preset in presets:
        config = DEFAULT_PRESETS[preset]
        queries.append(
            WorkflowQuery(
                label=preset,
                workflow=config["workflow"],
                event=config["event"],
                branch=config["branch"],
            )
        )
    return queries


def main() -> int:
    args = parse_args()
    args.artifact_prefix = tuple(args.artifact_prefix or DEFAULT_ARTIFACT_PREFIXES)
    cutoff = dt.datetime.now(tz=UTC) - dt.timedelta(days=args.days)
    token = token_from_environment_or_gh()
    client = GithubClient(args.repo, token)

    queries = workflow_queries(args)
    runs, stats = collect_stats(client, queries, cutoff, args)
    print_ascii_report(args.repo, cutoff, runs, stats, args.top)

    if args.plot:
        write_failure_svg(args.plot, stats)
        print(f"\nWrote plot: {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
