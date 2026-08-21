#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic state helpers for the reviewers-slack skill."""

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

THREAD_PATH = re.compile(r"^/archives/(?P<channel>[CDG][A-Z0-9]+)/p(?P<digits>\d{16})$")
MARKER = re.compile(
    r"reviewers-slack:v1(?:\s+|:)repo=(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"(?:\s+|:)pr=(?P<pr>\d+)(?:\s+|:)scope=(?P<scope>[0-9a-f]{12})"
    r"(?:(?:\s+|:)state=(?P<state>[A-Za-z0-9_-]+))?"
)
SLACK_USER_ID = re.compile(r"^U[A-Z0-9]+$")


def read_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def parse_thread_url(value: str) -> dict[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc.endswith(".slack.com"):
        raise ValueError("expected an https://<workspace>.slack.com thread permalink")
    match = THREAD_PATH.fullmatch(parsed.path.rstrip("/"))
    if not match:
        raise ValueError("expected /archives/<channel-id>/p<16-digit-message-ts>")
    digits = match.group("digits")
    message_ts = f"{digits[:10]}.{digits[10:]}"
    thread_ts = parse_qs(parsed.query).get("thread_ts", [message_ts])[0]
    if not re.fullmatch(r"\d{10}\.\d{6}", thread_ts):
        raise ValueError("thread_ts query parameter is not a Slack timestamp")
    return {
        "workspace": parsed.netloc,
        "channel_id": match.group("channel"),
        "message_ts": thread_ts,
        "permalink": value,
    }


def normalized_scope(document: dict) -> dict:
    repo = document.get("repo")
    pr = document.get("pr")
    files = document.get("files")
    if not isinstance(repo, str) or "/" not in repo:
        raise ValueError("scope JSON requires repo as owner/name")
    if not isinstance(pr, int) or pr <= 0:
        raise ValueError("scope JSON requires a positive integer pr")
    if not isinstance(files, list):
        raise ValueError("scope JSON requires a files list")

    normalized_files = []
    seen_paths = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each files entry must be an object")
        path = entry.get("path")
        owners = entry.get("owners")
        if not isinstance(path, str) or not path or path in seen_paths:
            raise ValueError("each changed path must be non-empty and unique")
        if not isinstance(owners, list) or not all(
            isinstance(owner, str) for owner in owners
        ):
            raise ValueError(f"owners for {path!r} must be a string list")
        seen_paths.add(path)
        normalized_files.append({"path": path, "owners": sorted(set(owners))})

    sorted_files = sorted(normalized_files, key=lambda item: item["path"])
    return {"repo": repo, "pr": pr, "files": sorted_files}


def fingerprint(document: dict) -> str:
    payload = json.dumps(
        normalized_scope(document), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def normalized_reviewers(document: dict) -> dict:
    groups = document.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("reviewer JSON requires a non-empty groups object")

    normalized_groups = {}
    for group, reviewers in sorted(groups.items()):
        if not isinstance(group, str) or not group:
            raise ValueError("reviewer group names must be non-empty strings")
        if not isinstance(reviewers, list) or not reviewers:
            raise ValueError(f"reviewers for {group!r} must be a non-empty list")

        normalized_group = []
        seen_github = set()
        seen_slack = set()
        for reviewer in reviewers:
            if not isinstance(reviewer, dict):
                raise ValueError(f"each reviewer for {group!r} must be an object")
            github = reviewer.get("github")
            slack_id = reviewer.get("slack_id")
            if not isinstance(github, str) or not github:
                raise ValueError(f"reviewer for {group!r} requires a GitHub login")
            if not isinstance(slack_id, str) or not SLACK_USER_ID.fullmatch(slack_id):
                raise ValueError(f"reviewer {github!r} requires a Slack U... user ID")
            if github in seen_github or slack_id in seen_slack:
                raise ValueError(f"duplicate reviewer in group {group!r}")
            seen_github.add(github)
            seen_slack.add(slack_id)
            normalized_group.append({"github": github, "slack_id": slack_id})
        normalized_groups[group] = normalized_group

    return {"groups": normalized_groups}


def encode_reviewers(document: dict) -> str:
    payload = json.dumps(
        normalized_reviewers(document), sort_keys=True, separators=(",", ":")
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_reviewers(token: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError("encoded reviewer state is not base64url")
    padding = "=" * (-len(token) % 4)
    try:
        document = json.loads(base64.urlsafe_b64decode(token + padding))
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError("encoded reviewer state is invalid") from error
    return normalized_reviewers(document)


def parse_marker(text: str) -> dict[str, object]:
    matches = list(MARKER.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one reviewers-slack marker, found {len(matches)}"
        )
    values = matches[0].groupdict()
    parsed = {"repo": values["repo"], "pr": int(values["pr"]), "scope": values["scope"]}
    if values["state"]:
        parsed["reviewers"] = decode_reviewers(values["state"])
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_url_parser = subparsers.add_parser(
        "parse-url", help="parse a Slack thread permalink"
    )
    parse_url_parser.add_argument("url")

    fingerprint_parser = subparsers.add_parser(
        "fingerprint", help="hash normalized PR scope JSON"
    )
    fingerprint_parser.add_argument("json_file", help="JSON file path, or - for stdin")

    marker_parser = subparsers.add_parser(
        "parse-marker", help="parse one managed-message marker"
    )
    marker_parser.add_argument("text_file", help="text file path, or - for stdin")

    encode_parser = subparsers.add_parser(
        "encode-reviewers", help="encode GitHub-to-Slack reviewer state"
    )
    encode_parser.add_argument("json_file", help="JSON file path, or - for stdin")

    decode_parser = subparsers.add_parser(
        "decode-reviewers", help="decode GitHub-to-Slack reviewer state"
    )
    decode_parser.add_argument("token")

    args = parser.parse_args()
    try:
        if args.command == "parse-url":
            print(json.dumps(parse_thread_url(args.url), sort_keys=True))
        elif args.command == "fingerprint":
            print(fingerprint(read_json(args.json_file)))
        elif args.command == "parse-marker":
            print(json.dumps(parse_marker(read_text(args.text_file)), sort_keys=True))
        elif args.command == "encode-reviewers":
            print(encode_reviewers(read_json(args.json_file)))
        elif args.command == "decode-reviewers":
            print(json.dumps(decode_reviewers(args.token), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
