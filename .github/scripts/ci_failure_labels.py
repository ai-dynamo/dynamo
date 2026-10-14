#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate CI failure verdicts and send them to Datadog Logs.

Each verdict must name a failed job of this repository on main; anything else
is rejected so the dashboard only ever contains real main-branch failures.
"""

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache

CLASSES = {"code", "infra", "unclear"}
CONFIDENCE = {"high", "medium", "low"}
MAX_VERDICTS = 100
BATCH = 100
SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def github(path):
    req = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


@lru_cache(maxsize=None)
def get_run(run_id):
    return github(f"actions/runs/{run_id}")


def validate(raw):
    verdicts = json.loads(raw)
    if not isinstance(verdicts, list) or not 0 < len(verdicts) <= MAX_VERDICTS:
        raise ValueError(f"verdicts must be a JSON array of 1..{MAX_VERDICTS}")
    seen = set()
    for v in verdicts:
        job_id = str(v.get("job_id", ""))
        if not job_id.isdigit() or job_id in seen:
            raise ValueError(f"bad or duplicate job_id: {job_id!r}")
        seen.add(job_id)
        if v.get("failure_class") not in CLASSES:
            raise ValueError(f"{job_id}: failure_class must be one of {CLASSES}")
        if not SLUG.match(str(v.get("cause", ""))):
            raise ValueError(f"{job_id}: cause must be a lowercase slug")
        if v.get("confidence") not in CONFIDENCE:
            raise ValueError(f"{job_id}: confidence must be one of {CONFIDENCE}")
        v["evidence"] = str(v.get("evidence", ""))[:500]
    return verdicts


def enrich(v, decided_by):
    job = github(f"actions/jobs/{v['job_id']}")
    if job.get("conclusion") != "failure" or job.get("head_branch") != "main":
        raise ValueError(f"{v['job_id']}: not a failed job on main")
    run = get_run(job["run_id"])
    return {
        "ddsource": "dynamo-ci-classifier",
        "service": "dynamo-ci-failure-labels",
        "ddtags": ",".join(
            [
                "team:dynamo",
                f"failure_class:{v['failure_class']}",
                f"cause:{v['cause']}",
                f"confidence:{v['confidence']}",
                f"decided_by:{decided_by}",
            ]
        ),
        "message": f"[{v['failure_class']}] {job['name']}: {v['cause']}",
        "failure_class": v["failure_class"],
        "cause": v["cause"],
        "confidence": v["confidence"],
        "decided_by": decided_by,
        "evidence": v["evidence"],
        "job_id": v["job_id"],
        "job_name": job["name"],
        "job_url": job["html_url"],
        "job_started_at": job.get("started_at"),
        "pipeline": run.get("name"),
        "run_id": job["run_id"],
        "commit": job.get("head_sha"),
        "classified_at": datetime.now(timezone.utc).isoformat(),
    }


def send(records):
    url = f"https://http-intake.logs.{os.environ.get('DD_SITE', 'datadoghq.com')}/api/v2/logs"
    for i in range(0, len(records), BATCH):
        req = urllib.request.Request(
            url,
            data=json.dumps(records[i : i + BATCH]).encode(),
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": os.environ["DD_API_KEY"],
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 202:
                raise RuntimeError(f"Datadog intake returned {resp.status}")


def main():
    decided_by = os.environ.get("DECIDED_BY", "devin")
    if not SLUG.match(decided_by):
        raise ValueError("decided_by must be a lowercase slug")
    records = [enrich(v, decided_by) for v in validate(os.environ["VERDICTS"])]
    send(records)
    counts = {c: sum(r["failure_class"] == c for r in records) for c in sorted(CLASSES)}
    print(f"Sent {len(records)} labels to Datadog: {counts}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        print(f"::error::{e}")
        sys.exit(1)
