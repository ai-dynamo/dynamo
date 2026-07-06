#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durably prefill Docker Hub task images through the campaign pull-through cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RATE_LIMIT = re.compile(r"^(?P<remaining>[0-9]+);w=(?P<window>[0-9]+)$")
SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
REPO_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class QuotaUnavailable(RuntimeError):
    """Raised when Docker Hub does not publish a trustworthy quota header."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: Any, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(source.read_bytes())
        temporary.chmod(0o444)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_registry_reference(reference: str) -> str:
    for prefix in ("docker.io/", "index.docker.io/"):
        if reference.startswith(prefix):
            return reference[len(prefix) :]
    return reference


def repository_from_reference(reference: str) -> str:
    normalized = normalize_registry_reference(reference)
    if "/" not in normalized or "@" in normalized:
        raise ValueError(f"unsupported Docker Hub tag reference: {reference}")
    last_slash = normalized.rfind("/")
    last_colon = normalized.rfind(":")
    repository = normalized[:last_colon] if last_colon > last_slash else normalized
    if not repository.startswith("swebench/sweb.eval."):
        raise ValueError(f"not a SWE-bench task image: {reference}")
    return repository


def parse_rate_limit(value: str | None) -> tuple[int, int]:
    match = RATE_LIMIT.fullmatch((value or "").strip())
    if not match:
        raise ValueError(f"invalid Docker Hub RateLimit-Remaining header: {value!r}")
    return int(match.group("remaining")), int(match.group("window"))


def docker_hub_quota() -> tuple[int, int, str]:
    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            repository = "library/hello-world"
            query = urllib.parse.urlencode(
                {
                    "service": "registry.docker.io",
                    "scope": f"repository:{repository}:pull",
                }
            )
            with urllib.request.urlopen(
                f"https://auth.docker.io/token?{query}", timeout=30
            ) as response:
                token = json.load(response)["token"]
            request = urllib.request.Request(
                f"https://registry-1.docker.io/v2/{repository}/manifests/latest",
                method="HEAD",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.oci.image.index.v1+json",
                },
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                header = response.headers.get("RateLimit-Remaining")
            remaining, window = parse_rate_limit(header)
            return remaining, window, header or ""
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < 3:
                time.sleep(5)
    raise QuotaUnavailable("; ".join(errors))


def registry_catalog(registry: str) -> set[str]:
    url = f"{registry.rstrip('/')}/v2/_catalog?n=1000"
    repositories: set[str] = set()
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
            link = response.headers.get("Link")
        current = payload.get("repositories") if isinstance(payload, dict) else None
        if not isinstance(current, list) or not all(
            isinstance(item, str) for item in current
        ):
            raise ValueError("registry catalog response has no repositories list")
        repositories.update(current)
        url = ""
        if link:
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            if not match:
                raise ValueError(f"unrecognized registry catalog Link header: {link}")
            url = urllib.parse.urljoin(registry, match.group(1))
    return repositories


def image_identity(reference: str) -> dict[str, Any]:
    output = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values = json.loads(output)
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise ValueError(f"unexpected docker image inspect output for {reference}")
    image_id = values[0].get("Id")
    repo_digests = sorted(set(values[0].get("RepoDigests") or []))
    if not isinstance(image_id, str) or not SHA256_ID.fullmatch(image_id):
        raise ValueError(f"image {reference} has no immutable sha256 image ID")
    if not repo_digests or not all(
        isinstance(digest, str) and REPO_DIGEST.fullmatch(digest)
        for digest in repo_digests
    ):
        raise ValueError(f"image {reference} has no valid immutable RepoDigest")
    content = {"image_id": image_id, "repo_digests": repo_digests}
    return {
        "requested_ref": reference,
        **content,
        "content_identity_sha256": canonical_sha256(content),
    }


def load_manifest(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("task-image manifest must use schema version 1")
    suite = payload.get("suite")
    images = payload.get("images")
    if not isinstance(suite, str) or not isinstance(images, dict) or not images:
        raise ValueError("task-image manifest has no suite/images map")
    if suite != "verified":
        raise ValueError("this prefill implementation currently supports only verified")
    if len(images) != 500:
        raise ValueError(f"verified task-image manifest must contain 500 images, got {len(images)}")
    for instance_id, identity in images.items():
        if not isinstance(instance_id, str) or not SAFE_INSTANCE_ID.fullmatch(instance_id):
            raise ValueError(f"unsafe instance ID: {instance_id!r}")
        if not isinstance(identity, dict):
            raise ValueError(f"invalid identity for {instance_id}")
        repository_from_reference(identity.get("requested_ref", ""))
    return suite, images


def load_completed(
    entries_dir: Path, images: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not entries_dir.exists():
        return completed
    for path in sorted(entries_dir.glob("*.json")):
        record = json.loads(path.read_text())
        instance_id = record.get("instance_id") if isinstance(record, dict) else None
        if instance_id not in images or path.name != f"{instance_id}.json":
            raise ValueError(f"invalid prefill entry: {path}")
        if record.get("identity") != images[instance_id]:
            raise ValueError(f"prefill entry identity drifted: {instance_id}")
        completed[instance_id] = record
    return completed


def write_status(
    path: Path,
    *,
    state: str,
    suite: str,
    total: int,
    completed: int,
    initial_cached: int,
    cold_fill_required: int,
    **extra: Any,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "state": state,
            "suite": suite,
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "initial_cached": initial_cached,
            "cold_fill_required": cold_fill_required,
            **extra,
        },
        mode=0o644,
    )


def pull_and_validate(
    reference: str, expected: dict[str, Any], timeout_seconds: int
) -> None:
    subprocess.run(
        ["docker", "pull", "--quiet", "--platform", "linux/amd64", reference],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    actual = image_identity(reference)
    if actual != expected:
        raise ValueError(
            f"immutable image identity mismatch for {reference}: "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"actual={json.dumps(actual, sort_keys=True)}"
        )


def remove_local_image(reference: str) -> None:
    subprocess.run(
        ["docker", "image", "rm", "--force", reference],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    state_dir = args.state_dir.resolve()
    status_path = state_dir / "status.json"
    entries_dir = state_dir / "entries"
    suite, images = load_manifest(manifest)
    manifest_sha256 = file_sha256(manifest)
    script_sha256 = file_sha256(Path(__file__).resolve())
    total = len(images)
    state_dir.mkdir(parents=True, exist_ok=True)

    retained_manifest = state_dir / "source-task-images.json"
    if retained_manifest.exists():
        if file_sha256(retained_manifest) != manifest_sha256:
            raise ValueError("retained task-image manifest differs from requested source")
    else:
        atomic_copy(manifest, retained_manifest)

    metadata_path = state_dir / "metadata.json"
    expected_metadata = {
        "schema_version": 1,
        "suite": suite,
        "source_manifest_sha256": manifest_sha256,
        "cache_binding_sha256": args.cache_binding_sha256,
        "total": total,
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        legacy_without_binding = "cache_binding_sha256" not in metadata
        comparable_metadata = dict(metadata)
        if legacy_without_binding:
            comparable_metadata["cache_binding_sha256"] = args.cache_binding_sha256
        if {
            key: comparable_metadata.get(key) for key in expected_metadata
        } != expected_metadata:
            raise ValueError("prefill metadata differs from this invocation")
        script_sha256s = list(metadata.get("script_sha256s", []))
        if legacy_script_sha256 := metadata.pop("script_sha256", None):
            script_sha256s.append(legacy_script_sha256)
        if not isinstance(script_sha256s, list) or not all(
            isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
            for item in script_sha256s
        ):
            raise ValueError("prefill metadata has invalid script hashes")
        if script_sha256 not in script_sha256s:
            script_sha256s.append(script_sha256)
        normalized_metadata = {
            **expected_metadata,
            "started_at": metadata["started_at"],
            "script_sha256s": sorted(set(script_sha256s)),
        }
        if metadata != normalized_metadata:
            atomic_write_json(metadata_path, normalized_metadata)
        metadata = normalized_metadata
    else:
        metadata = {
            **expected_metadata,
            "started_at": utc_now(),
            "script_sha256s": [script_sha256],
        }
        atomic_write_json(metadata_path, metadata)

    expected_repositories = {
        repository_from_reference(identity["requested_ref"])
        for identity in images.values()
    }
    if len(expected_repositories) != total:
        raise ValueError("SWE-bench task-image repositories are not one-to-one with instances")

    initial_catalog_path = state_dir / "initial-catalog.json"
    if initial_catalog_path.exists():
        initial_catalog_payload = json.loads(initial_catalog_path.read_text())
        initial_catalog = set(initial_catalog_payload.get("repositories", []))
    else:
        initial_catalog = registry_catalog(args.registry)
        atomic_write_json(
            initial_catalog_path,
            {"schema_version": 1, "captured_at": utc_now(), "repositories": sorted(initial_catalog)},
        )
    initial_cached_repositories = expected_repositories & initial_catalog
    initial_cached = len(initial_cached_repositories)
    cold_fill_required = total - initial_cached
    completed = load_completed(entries_dir, images)
    entries_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        images.items(),
        key=lambda item: (
            repository_from_reference(item[1]["requested_ref"])
            not in initial_cached_repositories,
            item[0],
        ),
    )
    write_status(
        status_path,
        state="running",
        suite=suite,
        total=total,
        completed=len(completed),
        initial_cached=initial_cached,
        cold_fill_required=cold_fill_required,
        source_manifest_sha256=manifest_sha256,
        script_sha256=script_sha256,
        cache_binding_sha256=args.cache_binding_sha256,
        started_at=metadata["started_at"],
    )

    last_quota_remaining: int | None = None
    last_quota_window: int | None = None
    last_quota_header: str | None = None
    for instance_id, expected in ordered:
        revalidating = instance_id in completed
        reference = expected["requested_ref"]
        repository = repository_from_reference(reference)
        failures = 0
        while True:
            try:
                remaining, window, quota_header = docker_hub_quota()
            except QuotaUnavailable as error:
                write_status(
                    status_path,
                    state="waiting_for_quota_probe",
                    suite=suite,
                    total=total,
                    completed=len(completed),
                    initial_cached=initial_cached,
                    cold_fill_required=cold_fill_required,
                    source_manifest_sha256=manifest_sha256,
                    script_sha256=script_sha256,
                    cache_binding_sha256=args.cache_binding_sha256,
                    started_at=metadata["started_at"],
                    next_instance_id=instance_id,
                    quota_probe_error=str(error),
                    retry_seconds=args.retry_seconds,
                )
                print(
                    f"{utc_now()} quota probe unavailable; "
                    f"waiting {args.retry_seconds}s: {error}",
                    flush=True,
                )
                time.sleep(args.retry_seconds)
                continue
            last_quota_remaining = remaining
            last_quota_window = window
            last_quota_header = quota_header
            if remaining <= args.quota_reserve:
                write_status(
                    status_path,
                    state="waiting_for_quota",
                    suite=suite,
                    total=total,
                    completed=len(completed),
                    initial_cached=initial_cached,
                    cold_fill_required=cold_fill_required,
                    source_manifest_sha256=manifest_sha256,
                    script_sha256=script_sha256,
                    cache_binding_sha256=args.cache_binding_sha256,
                    started_at=metadata["started_at"],
                    next_instance_id=instance_id,
                    quota_remaining=quota_header,
                    poll_seconds=args.poll_seconds,
                )
                print(
                    f"{utc_now()} quota={quota_header} reserve={args.quota_reserve}; "
                    f"waiting {args.poll_seconds}s",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue
            write_status(
                status_path,
                state="revalidating" if revalidating else "pulling",
                suite=suite,
                total=total,
                completed=len(completed),
                initial_cached=initial_cached,
                cold_fill_required=cold_fill_required,
                source_manifest_sha256=manifest_sha256,
                script_sha256=script_sha256,
                cache_binding_sha256=args.cache_binding_sha256,
                started_at=metadata["started_at"],
                next_instance_id=instance_id,
                quota_remaining=quota_header,
            )
            try:
                pull_and_validate(reference, expected, args.pull_timeout_seconds)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                failures += 1
                stderr = (error.stderr or "")[-4000:] if isinstance(error.stderr, str) else ""
                if isinstance(error, subprocess.TimeoutExpired):
                    stderr = (
                        f"docker pull exceeded {args.pull_timeout_seconds}s timeout; "
                        f"stderr={stderr}"
                    )
                rate_limited = "toomanyrequests" in stderr.lower() or "rate limit" in stderr.lower()
                wait_seconds = max(window + 5, args.poll_seconds) if rate_limited else args.retry_seconds
                if failures > args.max_pull_failures and not rate_limited:
                    raise RuntimeError(
                        f"docker pull failed {failures} times for {reference}: {stderr}"
                    ) from error
                write_status(
                    status_path,
                    state="waiting_for_quota" if rate_limited else "retrying_pull",
                    suite=suite,
                    total=total,
                    completed=len(completed),
                    initial_cached=initial_cached,
                    cold_fill_required=cold_fill_required,
                    source_manifest_sha256=manifest_sha256,
                    script_sha256=script_sha256,
                    cache_binding_sha256=args.cache_binding_sha256,
                    started_at=metadata["started_at"],
                    next_instance_id=instance_id,
                    quota_remaining=quota_header,
                    pull_failures=failures,
                    last_pull_error=stderr,
                    retry_seconds=wait_seconds,
                )
                print(
                    f"{utc_now()} pull failure instance={instance_id} "
                    f"rate_limited={rate_limited}; waiting {wait_seconds}s",
                    flush=True,
                )
                remove_local_image(reference)
                time.sleep(wait_seconds)
                continue
            finally:
                remove_local_image(reference)
            break

        if revalidating:
            record = {
                **completed[instance_id],
                "cache_binding_sha256": args.cache_binding_sha256,
                "last_revalidated_at": utc_now(),
                "last_revalidated_script_sha256": script_sha256,
            }
        else:
            record = {
                "schema_version": 1,
                "instance_id": instance_id,
                "repository": repository,
                "was_cached_at_start": repository in initial_cached_repositories,
                "quota_before": quota_header,
                "script_sha256": script_sha256,
                "cache_binding_sha256": args.cache_binding_sha256,
                "identity": expected,
                "verified_at": utc_now(),
            }
        atomic_write_json(entries_dir / f"{instance_id}.json", record)
        completed[instance_id] = record
        print(
            f"{utc_now()} {'revalidated' if revalidating else 'verified'}="
            f"{len(completed)}/{total} instance={instance_id} "
            f"cached_at_start={record['was_cached_at_start']} quota_before={quota_header}",
            flush=True,
        )
        write_status(
            status_path,
            state="running",
            suite=suite,
            total=total,
            completed=len(completed),
            initial_cached=initial_cached,
            cold_fill_required=cold_fill_required,
            source_manifest_sha256=manifest_sha256,
            script_sha256=script_sha256,
            cache_binding_sha256=args.cache_binding_sha256,
            started_at=metadata["started_at"],
            last_instance_id=instance_id,
            quota_remaining=quota_header,
        )

    final_catalog = registry_catalog(args.registry)
    missing = sorted(expected_repositories - final_catalog)
    if missing:
        raise ValueError(f"completed task repositories absent from registry catalog: {missing[:5]}")
    if (
        last_quota_remaining is None
        or last_quota_window is None
        or last_quota_header is None
    ):
        raise RuntimeError("prefill completed without a successful quota observation")
    write_status(
        status_path,
        state="complete",
        suite=suite,
        total=total,
        completed=len(completed),
        initial_cached=initial_cached,
        cold_fill_required=cold_fill_required,
        source_manifest_sha256=manifest_sha256,
        script_sha256=script_sha256,
        cache_binding_sha256=args.cache_binding_sha256,
        started_at=metadata["started_at"],
        completed_at=utc_now(),
        final_catalog_verified=total,
        quota_last_observed_before_pull=last_quota_header,
        quota_window_seconds=last_quota_window,
        quota_last_observed_count=last_quota_remaining,
    )
    print(
        f"{utc_now()} prefill complete: {total}/{total} "
        f"last_quota_before_pull={last_quota_header}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--cache-binding-sha256", required=True)
    parser.add_argument("--registry", default="http://dockerhub-pull-cache:5000")
    parser.add_argument("--quota-reserve", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    parser.add_argument("--max-pull-failures", type=int, default=5)
    parser.add_argument("--pull-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (
        args.quota_reserve < 1
        or args.poll_seconds < 1
        or args.retry_seconds < 1
        or args.pull_timeout_seconds < 1
    ):
        parser.error("quota reserve and wait intervals must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", args.cache_binding_sha256):
        parser.error("cache binding must be a lowercase SHA-256 digest")
    try:
        run(args)
    except BaseException as error:
        try:
            suite, images = load_manifest(args.manifest)
            completed = load_completed(args.state_dir / "entries", images)
            initial_catalog_path = args.state_dir / "initial-catalog.json"
            initial_catalog = (
                set(json.loads(initial_catalog_path.read_text()).get("repositories", []))
                if initial_catalog_path.exists()
                else set()
            )
            expected_repositories = {
                repository_from_reference(identity["requested_ref"])
                for identity in images.values()
            }
            write_status(
                args.state_dir / "status.json",
                state="failed",
                suite=suite,
                total=len(images),
                completed=len(completed),
                initial_cached=len(expected_repositories & initial_catalog),
                cold_fill_required=len(expected_repositories - initial_catalog),
                cache_binding_sha256=args.cache_binding_sha256,
                error=f"{type(error).__name__}: {error}",
            )
        except Exception as status_error:
            print(f"could not persist failure status: {status_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
