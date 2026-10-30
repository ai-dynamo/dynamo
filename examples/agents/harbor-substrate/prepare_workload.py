# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Freeze selected tasks, prepared image references, templates, and a Harbor job."""

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def pinned_image(value: str) -> str:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value):
        raise ValueError("Use an image reference pinned by its sha256 digest")
    return value


def prepare(args) -> None:
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("The model URL must be an HTTP or HTTPS endpoint")
    if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError("Keep credentials and query parameters out of the model URL")
    if not 1 < args.max_tokens < args.context_window:
        raise ValueError("Require 1 < max-tokens < context-window")
    if args.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.job_name):
        raise ValueError("The job name must be a single directory name")
    if args.trace and args.endpoint_kind != "dynamo":
        raise ValueError("Native server tracing requires a Dynamo endpoint")
    images = json.loads(args.images.read_text())
    if not isinstance(images, dict) or not images:
        raise ValueError("The image map must select at least one task")
    pinned_image(args.guest_image)
    for task, image in images.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task):
            raise ValueError(
                "Task names must be directory names without path separators"
            )
        pinned_image(image)
        if not (args.dataset / task / "task.toml").is_file():
            raise ValueError(f"Missing task: {task}")
    args.output.mkdir(parents=True, exist_ok=False)
    templates = args.output / "templates"
    templates.mkdir()
    base = yaml.safe_load(Path(__file__).with_name("actor-template.yaml").read_text())
    records = []
    for task, image in images.items():
        source = args.dataset / task
        destination = args.output / "tasks" / task
        shutil.copytree(source, destination)
        template = copy.deepcopy(base)
        identity = json.dumps([task, image, args.guest_image, args.golden])
        name = "swe-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        template["metadata"]["name"] = name
        template["containers"][0]["image"] = image
        template["volumes"][0]["image"]["reference"] = args.guest_image
        if args.golden:
            template["snapshotConfig"]["goldenSnapshotConfig"][
                "mode"
            ] = "GOLDEN_SNAPSHOT_MODE_ENABLED"
        (templates / f"{name}.yaml").write_text(
            yaml.safe_dump(template, sort_keys=False)
        )
        commit = json.loads((source / "tests/config.json").read_text())["base_commit"]
        capture = (
            'set -eu; cd /testbed; out=/logs/artifacts/candidate; mkdir -p "$out"; '
            'git rev-parse HEAD > "$out/head.txt"; '
            'git status --porcelain=v1 > "$out/status.txt"; '
            f'git diff --binary --full-index {shlex.quote(commit)} -- > "$out/tracked.patch"; '
            'git ls-files --others --exclude-standard -z > "$out/untracked.list"; '
            'tar --null -T "$out/untracked.list" -czf "$out/untracked.tar.gz"; '
            'sha256sum "$out/tracked.patch" "$out/untracked.tar.gz" > "$out/sha256.txt"'
        )
        with (destination / "task.toml").open("a") as stream:
            stream.write(
                '\n[[verifier.collect]]\nservice = "main"\ntimeout_sec = 120\ncommand = '
                + json.dumps(capture)
                + "\n"
            )
        hashes = {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        records.append(
            {"task": task, "template": name, "image": image, "sha256": hashes}
        )
    (args.output / "task-lock.json").write_text(json.dumps(records, indent=2) + "\n")
    agent_env = {
        "DYNAMO_BASE_URL": args.base_url.rstrip("/"),
        "DYN_AGENT_SESSION_FINAL": "1" if args.endpoint_kind == "dynamo" else "0",
        "DYNAMO_CONTEXT_WINDOW": str(args.context_window),
        "DYNAMO_MAX_TOKENS": str(args.max_tokens),
    }
    environment = {
        "endpoint": "ate-env-api.ate-env.svc:7777",
        "kube_context": "in-cluster",
        "egress_policy": "/work/example/egress.yaml",
        "template_map": {r["task"]: r["template"] for r in records},
        "ready_timeout": 1200,
    }
    if args.trace:
        agent_env.update(
            {
                "DYN_REQUEST_TRACE": "1",
                "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT": "tcp://127.0.0.1:20390",
                "DYN_TOOL_TRACE_TLS_CONFIG": "/opt/tool-trace/stunnel-client.conf",
            }
        )
        environment["uploads"] = {"/work/tool-trace": "/opt/tool-trace"}
    job = {
        "job_name": args.job_name,
        "jobs_dir": "/work/jobs",
        "n_attempts": 1,
        "n_concurrent_trials": args.concurrency,
        "agents": [
            {
                "import_path": "tutorial_pi:TutorialPi",
                "model_name": f"dynamo/{args.model}",
                "kwargs": {
                    "version": "0.72.1",
                    "thinking": "off",
                    "preinstalled": True,
                },
                "env": agent_env,
            }
        ],
        "environment": {
            "import_path": "substrate_environment:SubstrateEnvironment",
            "delete": True,
            "kwargs": environment,
        },
        "tasks": [{"path": f"/work/tasks/{r['task']}"} for r in records],
    }
    (args.output / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    print(f"Prepared {len(records)} tasks in {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--images", type=Path, required=True, help="JSON task -> prepared image digest"
    )
    parser.add_argument("--guest-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-window", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--endpoint-kind", choices=["dynamo", "external"], default="dynamo"
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--job-name", default="swe-substrate")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--trace", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
