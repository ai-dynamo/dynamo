#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect a read-only Dynamo Kubernetes debug bundle without secrets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Tunables and conventional return codes (kept here to avoid magic numbers).
DEFAULT_KUBECTL_TIMEOUT_SEC = 30
DEFAULT_LOG_TAIL_LINES = 200
DGD_POD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
# POSIX-conventional return codes used when the wrapper itself fails before
# kubectl can produce a real one.
RETURNCODE_COMMAND_NOT_FOUND = 127  # `kubectl` not installed
RETURNCODE_TIMED_OUT = 124  # subprocess timeout

# `kubectl describe` and pod logs can echo secret env values (HF tokens,
# bearer tokens, passwords). Scrub them before anything is written to disk so
# common credentials are not persisted verbatim in the bundle.
_PLAIN_KV_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]*)(\s*[:=]\s*)(\S+)")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)")
_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{8,}\b")


def is_secret_key(key: str) -> bool:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    parts = tuple(
        part for part in re.split(r"[^A-Za-z0-9]+", camel_split.lower()) if part
    )
    if not parts:
        return False
    if parts == ("token",) or parts == ("secret",):
        return True
    if {"password", "passwd", "credential", "credentials", "authorization"} & set(
        parts
    ):
        return True
    if parts[-1] in {"secret", "auth"}:
        return True
    sensitive_key_pairs = {("api", "key"), ("access", "key"), ("private", "key")}
    if any(pair in sensitive_key_pairs for pair in zip(parts, parts[1:])):
        return True
    if parts[-1] == "token":
        telemetry_prefixes = {
            "prompt",
            "completion",
            "input",
            "output",
            "cached",
            "max",
            "num",
            "total",
        }
        return len(parts) == 1 or parts[-2] not in telemetry_prefixes
    return False


def json_string_end(text: str, opening_quote: int) -> int | None:
    """Return a JSON string's closing quote without regex backtracking."""
    cursor = opening_quote + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
        elif text[cursor] == '"':
            return cursor
        else:
            cursor += 1
    return None


def redact_json_string_values(text: str) -> str:
    """Redact sensitive JSON string values embedded in arbitrary text."""
    cursor = 0
    unchanged_start = 0
    chunks: list[str] = []
    while cursor < len(text):
        key_start = text.find('"', cursor)
        if key_start == -1:
            break
        key_end = json_string_end(text, key_start)
        if key_end is None:
            break

        separator = key_end + 1
        while separator < len(text) and text[separator].isspace():
            separator += 1
        if separator >= len(text) or text[separator] != ":":
            cursor = key_end + 1
            continue

        value_start = separator + 1
        while value_start < len(text) and text[value_start].isspace():
            value_start += 1
        if value_start >= len(text) or text[value_start] != '"':
            cursor = key_end + 1
            continue

        value_end = json_string_end(text, value_start)
        if value_end is None:
            break

        raw_key = text[key_start : key_end + 1]
        try:
            key = json.loads(raw_key)
        except json.JSONDecodeError:
            key = text[key_start + 1 : key_end]
        if isinstance(key, str) and is_secret_key(key):
            chunks.append(text[unchanged_start : value_start + 1])
            chunks.append("<redacted>")
            unchanged_start = value_end
        cursor = value_end + 1

    if not chunks:
        return text
    chunks.append(text[unchanged_start:])
    return "".join(chunks)


def redact(text: str) -> str:
    if not text:
        return text

    def redact_plain_value(match: re.Match[str]) -> str:
        if not is_secret_key(match.group(1)):
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}<redacted>"

    text = redact_json_string_values(text)
    text = _PLAIN_KV_RE.sub(redact_plain_value, text)
    text = _BEARER_RE.sub(lambda m: f"{m.group(1)}<redacted>", text)
    text = _HF_TOKEN_RE.sub("<redacted-hf-token>", text)
    return text


def run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout, check=False
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "returncode": RETURNCODE_COMMAND_NOT_FOUND,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": RETURNCODE_TIMED_OUT,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Timed out after {timeout}s",
        }


def write_result(outdir: Path, name: str, result: dict[str, Any]) -> None:
    safe = name.replace("/", "_").replace(" ", "_")
    (outdir / f"{safe}.txt").write_text(
        "$ "
        + " ".join(result["cmd"])
        + "\n\n"
        + "RETURN_CODE="
        + str(result["returncode"])
        + "\n\n"
        + "STDOUT\n"
        + redact(str(result["stdout"]))
        + "\n\n"
        + "STDERR\n"
        + redact(str(result["stderr"]))
        + "\n",
        encoding="utf-8",
    )


def write_pod_discovery_result(outdir: Path, name: str, result: dict[str, Any]) -> None:
    """Write pod-discovery status without persisting full pod specs."""
    safe_result = dict(result)
    safe_result["stdout"] = "<omitted: pod JSON used only for local discovery>"
    write_result(outdir, name, safe_result)


def kubectl_json(args: list[str], timeout: int) -> tuple[Any | None, dict[str, Any]]:
    result = run(["kubectl", *args, "-o", "json"], timeout)
    if result["returncode"] != 0:
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError:
        invalid_result = dict(result)
        invalid_result["returncode"] = 1
        invalid_result["stderr"] = (
            str(result["stderr"]) + "\nFailed to parse kubectl JSON output."
        ).lstrip()
        return None, invalid_result


def pod_names(
    namespace: str, selector: str | None, timeout: int
) -> tuple[list[str], dict[str, Any]]:
    args = ["get", "pods", "-n", namespace]
    if selector:
        args.extend(["-l", selector])
    body, result = kubectl_json(args, timeout)
    if not body:
        return [], result
    return (
        [
            item.get("metadata", {}).get("name")
            for item in body.get("items", [])
            if item.get("metadata", {}).get("name")
        ],
        result,
    )


def container_names(
    namespace: str, pod: str, timeout: int
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    body, result = kubectl_json(["get", "pod", pod, "-n", namespace], timeout)
    if not body:
        return [], result
    specs = body.get("spec", {})
    containers: list[tuple[str, str]] = []
    for kind, field in [
        ("init", "initContainers"),
        ("container", "containers"),
    ]:
        for item in specs.get(field, []):
            if item.get("name"):
                containers.append((kind, item["name"]))
    return containers, result


def deployment_pod_selector(
    deployment_name: str | None, selector: str | None
) -> str | None:
    """Build the pod selector, scoping to the named DGD when provided."""
    selectors = []
    if deployment_name:
        selectors.append(f"{DGD_POD_LABEL}={deployment_name}")
    if selector:
        selectors.append(selector)
    return ",".join(selectors) or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", "-n", required=True)
    parser.add_argument(
        "--deployment-name", help="DynamoGraphDeployment name, if known"
    )
    parser.add_argument(
        "--selector", help="Optional pod selector, for example app=my-app"
    )
    parser.add_argument(
        "--outdir",
        "--output-dir",
        dest="outdir",
        default=None,
        help="Output dir; defaults to a private mkdtemp dynamo-debug-* directory",
    )
    parser.add_argument("--tail", type=int, default=DEFAULT_LOG_TAIL_LINES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_KUBECTL_TIMEOUT_SEC)
    args = parser.parse_args()

    pod_selector = deployment_pod_selector(args.deployment_name, args.selector)

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        # mkdtemp gives an unpredictable name with 0700 perms, unlike a
        # guessable /tmp/dynamo-debug-<timestamp> path on a shared host.
        outdir = Path(tempfile.mkdtemp(prefix="dynamo-debug-")).resolve()

    pod_command = ["kubectl", "get", "pods", "-n", args.namespace, "-o", "wide"]
    if pod_selector:
        pod_command.extend(["-l", pod_selector])

    commands: list[tuple[str, list[str]]] = [
        ("context", ["kubectl", "config", "current-context"]),
        ("nodes", ["kubectl", "get", "nodes", "-o", "wide"]),
        ("storageclass", ["kubectl", "get", "storageclass"]),
        ("namespace", ["kubectl", "get", "namespace", args.namespace, "-o", "yaml"]),
        (
            "dgd",
            [
                "kubectl",
                "get",
                "dynamographdeployment",
                "-n",
                args.namespace,
                "-o",
                "wide",
            ],
        ),
        ("pods", pod_command),
        ("services", ["kubectl", "get", "svc", "-n", args.namespace, "-o", "wide"]),
        ("pvc", ["kubectl", "get", "pvc", "-n", args.namespace, "-o", "wide"]),
        ("jobs", ["kubectl", "get", "jobs", "-n", args.namespace, "-o", "wide"]),
        (
            "events",
            [
                "kubectl",
                "get",
                "events",
                "-n",
                args.namespace,
                "--sort-by=.lastTimestamp",
            ],
        ),
    ]
    if args.deployment_name:
        commands.append(
            (
                "describe_dgd",
                [
                    "kubectl",
                    "describe",
                    "dynamographdeployment",
                    args.deployment_name,
                    "-n",
                    args.namespace,
                ],
            )
        )

    summary: dict[str, Any] = {
        "outdir": str(outdir),
        "namespace": args.namespace,
        "pod_selector": pod_selector,
        "commands": [],
        "detail_commands": [],
    }
    for name, cmd in commands:
        result = run(cmd, args.timeout)
        write_result(outdir, name, result)
        summary["commands"].append(
            {"name": name, "cmd": cmd, "returncode": result["returncode"]}
        )

    pods, pod_inventory_result = pod_names(args.namespace, pod_selector, args.timeout)
    write_pod_discovery_result(outdir, "pods_json", pod_inventory_result)
    summary["detail_commands"].append(
        {
            "name": "pods_json",
            "cmd": pod_inventory_result["cmd"],
            "returncode": pod_inventory_result["returncode"],
            "required": True,
        }
    )
    summary["pods"] = pods
    for pod in pods:
        result = run(
            ["kubectl", "describe", "pod", pod, "-n", args.namespace], args.timeout
        )
        write_result(outdir, f"describe_pod_{pod}", result)
        summary["detail_commands"].append(
            {
                "name": f"describe_pod_{pod}",
                "cmd": result["cmd"],
                "returncode": result["returncode"],
                "required": True,
            }
        )
        containers, pod_json_result = container_names(args.namespace, pod, args.timeout)
        write_pod_discovery_result(outdir, f"pod_json_{pod}", pod_json_result)
        summary["detail_commands"].append(
            {
                "name": f"pod_json_{pod}",
                "cmd": pod_json_result["cmd"],
                "returncode": pod_json_result["returncode"],
                "required": True,
            }
        )
        for kind, container in containers:
            result = run(
                [
                    "kubectl",
                    "logs",
                    pod,
                    "-c",
                    container,
                    "-n",
                    args.namespace,
                    f"--tail={args.tail}",
                ],
                args.timeout,
            )
            log_name = f"logs_{kind}_{pod}_{container}"
            write_result(outdir, log_name, result)
            summary["detail_commands"].append(
                {
                    "name": log_name,
                    "cmd": result["cmd"],
                    "returncode": result["returncode"],
                    "required": True,
                }
            )
            previous_result = run(
                [
                    "kubectl",
                    "logs",
                    pod,
                    "-c",
                    container,
                    "-n",
                    args.namespace,
                    "--previous",
                    f"--tail={args.tail}",
                ],
                args.timeout,
            )
            previous_name = f"logs_previous_{kind}_{pod}_{container}"
            write_result(outdir, previous_name, previous_result)
            summary["detail_commands"].append(
                {
                    "name": previous_name,
                    "cmd": previous_result["cmd"],
                    "returncode": previous_result["returncode"],
                    "required": False,
                }
            )

    failed_commands = [
        item["name"]
        for item in [*summary["commands"], *summary["detail_commands"]]
        if item["returncode"] != 0 and item.get("required", True)
    ]
    summary["failed_commands"] = failed_commands
    summary["optional_failed_commands"] = [
        item["name"]
        for item in summary["detail_commands"]
        if item["returncode"] != 0 and not item["required"]
    ]
    summary["complete"] = not failed_commands

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failed_commands:
        print(
            "Debug bundle is incomplete; inspect failed_commands and the matching files "
            f"under {outdir}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
