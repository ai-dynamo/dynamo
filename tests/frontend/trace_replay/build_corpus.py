# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acquire pinned assets, capture a reference server, and publish a replay corpus."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import assets, capture, prepare
from .contract import ReplayError, dumps, file_digest, manifest_path


def resolve_profile(value: str) -> Path:
    """Bare aliases win; explicit relative/absolute paths select local profiles."""
    if value == "deepseek-v41":
        return Path(__file__).resolve().parent / "fixtures/deepseek_v41/manifest.json"
    return manifest_path(Path(value))


def _empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ReplayError("build output must be fresh or empty")


def _private_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def replay_command(output_dir: Path, model_dir: Path) -> str:
    """Print a command for the target installation, never the builder's uv Python."""
    return shlex.join(
        [
            "env",
            "PYTHONPATH=" + str(Path(__file__).resolve().parents[3]),
            "python",
            "-P",
            "-m",
            "tests.frontend.trace_replay",
            "run",
            "--corpus",
            str(output_dir.resolve()),
            "--model-dir",
            str(model_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve() / "replay-results"),
        ]
    )


def _reviewed_identity(
    path: Path | None, profile: dict[str, Any], reference: dict[str, Any]
) -> str | None:
    if path is None:
        return None
    _, evidence = prepare._reviewed(path, profile, reference)
    return str(evidence["manifest_sha256"])


def build_corpus(
    requests_path: Path,
    profile_path: Path,
    reference_url: str,
    output_dir: Path,
    *,
    cache_dir: Path,
    timeout: float = 600,
    concurrency: int = 2,
    api_key_env: str | None = None,
    reviewed_corpus: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Build a fresh corpus; every failure retains evidence without a root manifest.

    This CLI boundary intentionally maps unexpected vendor/network exceptions to
    exit 2 by class and stage, without including possibly sensitive exception text.
    """
    stage = "preflight"
    committed: dict[str, Any] | None = None
    try:
        output_dir = output_dir.resolve()
        cache_dir = cache_dir.resolve()
        _empty_output(output_dir)
        if cache_dir.is_relative_to(output_dir):
            raise ReplayError("asset cache must be outside the output directory")
        reference_url, _ = capture.validate_target(
            reference_url, timeout, concurrency, api_key_env
        )
        profile_data = manifest_path(profile_path).read_bytes()
        request_data = requests_path.read_bytes()
        with tempfile.TemporaryDirectory(prefix="frontend-replay-build-") as scratch:
            scratch_dir = Path(scratch)
            profile_snapshot = scratch_dir / "profile.json"
            request_snapshot = scratch_dir / "requests.jsonl"
            _private_bytes(profile_snapshot, profile_data)
            _private_bytes(request_snapshot, request_data)
            profile, reference = prepare.load_profile(profile_snapshot)
            seeds = prepare.read_seeds(request_data, profile)
            capture._seeds(request_data, profile["model_name"])
            prepare.validate_seeds(seeds, captured=True)
            reviewed_identity = _reviewed_identity(reviewed_corpus, profile, reference)
            prepare.validate_tokenizers_version(reference["tokenizers_version"])

            stage = "assets"
            bundle = assets.resolve_cached_assets(profile, reference, cache_dir)
            stage = "prompt_preflight"
            count = prepare.preflight_requests(
                profile_snapshot,
                bundle.model_dir,
                request_snapshot,
                encoder=bundle.encoder,
            )
            if (
                _reviewed_identity(reviewed_corpus, profile, reference)
                != reviewed_identity
            ):
                raise ReplayError("reviewed corpus changed after preflight")

            stage = "claim_output"
            _empty_output(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            marker = output_dir / ".build-in-progress"
            _private_bytes(marker, b"")
            inputs = output_dir / "evidence/inputs"
            frozen_profile = inputs / "profile.json"
            frozen_requests = inputs / "requests.jsonl"
            _private_bytes(frozen_profile, profile_data)
            _private_bytes(frozen_requests, request_data)
            capture_dir = output_dir / "evidence/capture"
            stage = "capture"
            captured = capture.capture_requests(
                frozen_requests,
                frozen_profile,
                reference_url,
                capture_dir,
                timeout=timeout,
                concurrency=concurrency,
                api_key_env=api_key_env,
            )
            if captured["complete"] is not True:
                return 1, {
                    "status": "incomplete_capture",
                    "stage": stage,
                    "count": captured["count"],
                    "successful_count": captured["successful_count"],
                }
            stage = "prepare"
            if (
                _reviewed_identity(reviewed_corpus, profile, reference)
                != reviewed_identity
            ):
                raise ReplayError("reviewed corpus changed during capture")
            staging = output_dir / ".prepare-staging"
            manifest = prepare.prepare_corpus(
                frozen_profile,
                bundle.model_dir,
                staging,
                encoder=bundle.encoder,
                captures=capture_dir,
                reviewed_corpus=reviewed_corpus,
            )
            if (
                reviewed_identity is not None
                and manifest["provenance"]["reviewed_corpus"]["manifest_sha256"]
                != reviewed_identity
            ):
                raise ReplayError("prepared response approvals differ from preflight")
            stage = "publish"
            result = {
                "status": "prepared",
                "count": count,
                "corpus": str(output_dir),
                "manifest_sha256": file_digest(staging / "manifest.json"),
                "model_dir": str(bundle.model_dir.resolve()),
                "response_expectations": len(
                    manifest.get("provenance", {})
                    .get("golden_review", {})
                    .get("ids", [])
                ),
                "replay_command": replay_command(output_dir, bundle.model_dir),
            }
            # Hard links atomically publish without overwriting any competing file.
            # The manifest is the commit point; a partial publication is unloadable.
            os.link(staging / "cases.jsonl", output_dir / "cases.jsonl")
            os.link(staging / "manifest.json", output_dir / "manifest.json")
            committed = result
            stage = "cleanup"
            shutil.rmtree(staging)
            marker.unlink()
            return 0, result
    except Exception as exc:
        # KeyboardInterrupt/SystemExit intentionally retain normal process semantics.
        if committed is not None:
            committed["warnings"] = [{"stage": stage, "error": type(exc).__name__}]
            return 0, committed
        result: dict[str, Any] = {
            "status": "error",
            "stage": stage,
            "error": type(exc).__name__,
        }
        if isinstance(exc, ReplayError):
            result["message"] = str(exc)
        return 2, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", default="deepseek-v41")
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache"))
        / "dynamo/frontend-replay",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--api-key-env")
    parser.add_argument("--reviewed-corpus", type=Path)
    args = parser.parse_args(argv)
    try:
        code, result = build_corpus(
            args.requests,
            resolve_profile(args.profile),
            args.reference_url,
            args.output,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
            concurrency=args.concurrency,
            api_key_env=args.api_key_env,
            reviewed_corpus=args.reviewed_corpus,
        )
    except Exception as exc:
        code, result = 2, {
            "status": "error",
            "stage": "preflight",
            "error": type(exc).__name__,
        }
    print(dumps(result))
    if code == 0:
        print("Run in the target installed frontend environment:")
        print(result["replay_command"])
    return code


if __name__ == "__main__":
    sys.exit(main())
