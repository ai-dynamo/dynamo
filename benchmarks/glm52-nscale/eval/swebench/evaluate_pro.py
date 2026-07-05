#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned public Pro evaluator with authoritative dataset image tags."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import shutil
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from pro_adapter import docker_image
from capture_task_images import install_docker_image_guard


_RUNTIME_STATE = threading.local()


def load_images(raw_sample_path: Path) -> dict[str, str]:
    images = {}
    with raw_sample_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            images[row["instance_id"]] = docker_image(row)
    return images


def write_status(
    status_dir: Path, instance_id: str, status: str, error: str | None = None
) -> None:
    digest = hashlib.sha256(instance_id.encode()).hexdigest()
    path = status_dir / f"{digest}.json"
    temporary = status_dir / f".{digest}.tmp"
    payload = {"instance_id": instance_id, "status": status}
    if error is not None:
        payload["error"] = error
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def track_evaluator(
    evaluate: Callable[..., Any], status_dir: Path
) -> Callable[..., Any]:
    @wraps(evaluate)
    def tracked(patch: str, sample: Any, *args: Any, **kwargs: Any) -> Any:
        instance_id = sample["instance_id"]
        _RUNTIME_STATE.error = None
        try:
            output = evaluate(patch, sample, *args, **kwargs)
        except Exception as error:
            runtime_error = getattr(_RUNTIME_STATE, "error", None)
            write_status(
                status_dir,
                instance_id,
                "error",
                runtime_error or f"{type(error).__name__}: {error}",
            )
            raise
        runtime_error = getattr(_RUNTIME_STATE, "error", None)
        tests = output.get("tests") if isinstance(output, dict) else None
        valid_tests = isinstance(tests, list) and all(
            isinstance(test, dict)
            and isinstance(test.get("name"), str)
            and isinstance(test.get("status"), str)
            for test in tests
        )
        try:
            fail_to_pass = set(ast.literal_eval(sample["fail_to_pass"]))
            pass_to_pass = set(ast.literal_eval(sample["pass_to_pass"]))
            passed_tests = {
                test["name"] for test in tests or [] if test["status"] == "PASSED"
            }
            valid_score_inputs = all(
                isinstance(test_name, str) for test_name in fail_to_pass | pass_to_pass
            )
            _ = (fail_to_pass | pass_to_pass) <= passed_tests
        except (KeyError, TypeError, ValueError, SyntaxError):
            valid_score_inputs = False
        if runtime_error or not valid_tests or not valid_score_inputs:
            write_status(
                status_dir,
                instance_id,
                "error",
                runtime_error or "missing valid test output",
            )
        else:
            write_status(status_dir, instance_id, "completed")
        return output

    return tracked


def wait_with_cleanup(
    container: Any,
    wait: Callable[..., Any],
    timeout: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Bound an upstream Docker wait and remove the container on any wait failure."""
    kwargs.setdefault("timeout", timeout)
    try:
        return wait(container, *args, **kwargs)
    except Exception as error:
        cleanup_errors = []
        for operation, call in (
            ("kill", container.kill),
            ("remove", lambda: container.remove(force=True)),
        ):
            try:
                call()
            except Exception as cleanup_error:
                cleanup_errors.append(
                    f"{operation}={type(cleanup_error).__name__}: {cleanup_error}"
                )
        message = (
            f"container wait failed or exceeded the {timeout}s hard timeout: "
            f"{type(error).__name__}: {error}"
        )
        if cleanup_errors:
            message += "; cleanup: " + ", ".join(cleanup_errors)
        _RUNTIME_STATE.error = message
        raise


def install_docker_wait_timeout(docker_module: Any, timeout: int) -> None:
    if timeout < 1:
        raise ValueError("evaluator timeout must be positive")
    container_class = docker_module.models.containers.Container
    original_wait = container_class.wait

    @wraps(original_wait)
    def bounded_wait(container: Any, *args: Any, **kwargs: Any) -> Any:
        return wait_with_cleanup(container, original_wait, timeout, *args, **kwargs)

    container_class.wait = bounded_wait


def remove_local_image_after(
    evaluate: Callable[..., Any], images: dict[str, str], docker_module: Any
) -> Callable[..., Any]:
    """Remove an evaluated Pro image when Docker confirms it is unused."""

    @wraps(evaluate)
    def cleaned(patch: str, sample: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return evaluate(patch, sample, *args, **kwargs)
        finally:
            image = images.get(sample["instance_id"])
            if image:
                client = None
                try:
                    client = docker_module.from_env()
                    client.images.remove(image, force=False, noprune=False)
                except Exception as error:
                    # A concurrent container may still reference the same image. Docker
                    # refuses that removal; the batch-boundary cleanup retries it later.
                    print(
                        f"Deferred cleanup for {image}: {type(error).__name__}: {error}"
                    )
                finally:
                    if client is not None:
                        client.close()

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pro-repo", required=True, type=Path)
    parser.add_argument("--raw-sample-path", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--evaluator-timeout", required=True, type=int)
    parser.add_argument("--task-images", required=True, type=Path)
    known, evaluator_args = parser.parse_known_args()

    pro_repo = known.pro_repo.resolve()
    raw_sample_path = known.raw_sample_path.resolve()
    status_dir = known.status_dir.resolve()
    images = load_images(raw_sample_path)

    shutil.rmtree(status_dir, ignore_errors=True)
    status_dir.mkdir(parents=True)

    sys.path.insert(0, str(pro_repo))
    evaluator = importlib.import_module("swe_bench_pro_eval")

    def exact_image(instance_id: str, _username: str, _repo: str = "") -> str:
        try:
            return images[instance_id]
        except KeyError as error:
            raise KeyError(
                f"no dockerhub_tag for Pro instance {instance_id}"
            ) from error

    # The pinned evaluator reconstructs image tags from repo and instance_id. Replace
    # only that bound function; all scoring and test execution remain upstream code.
    evaluator.get_dockerhub_image_uri = exact_image
    if known.evaluator_timeout < 1:
        raise ValueError("evaluator timeout must be positive")
    if evaluator.docker is not None:
        install_docker_image_guard(evaluator.docker, known.task_images)
        install_docker_wait_timeout(evaluator.docker, known.evaluator_timeout)
        evaluator.eval_with_docker = remove_local_image_after(
            evaluator.eval_with_docker, images, evaluator.docker
        )
    evaluator.eval_with_docker = track_evaluator(evaluator.eval_with_docker, status_dir)
    evaluator.eval_with_modal = track_evaluator(evaluator.eval_with_modal, status_dir)
    os.chdir(pro_repo)
    sys.argv = [
        "swe_bench_pro_eval.py",
        f"--raw_sample_path={raw_sample_path}",
        *evaluator_args,
    ]
    evaluator.main()


if __name__ == "__main__":
    main()
