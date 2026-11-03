# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove completed test executables named by this job's successful Cargo logs."""

import argparse
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

CARGO_REASONS = {
    "compiler-artifact",
    "compiler-message",
    "build-script-executed",
    "build-finished",
    "future-incompat-report",
}
LIBRARY_KINDS = {"lib", "rlib", "dylib", "cdylib", "staticlib"}
PRESERVED_TARGET = "request_trace_to_mooncake"


@dataclass(frozen=True)
class Executable:
    path: Path
    identity: tuple[int, ...]
    logical_bytes: int
    allocated_bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    workspace: Path
    executables: tuple[Executable, ...]
    candidate_count: int
    duplicate_records: int
    missing: tuple[str, ...]
    hardlinks: tuple[str, ...]
    retained_artifacts: int
    ignored_stdout_lines: int


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_blocks,
    )


def absolute_path(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected an absolute path without traversal: {path}")


def directory_chain(root: Path, path: Path) -> None:
    current = root
    for part in (None, *path.relative_to(root).parts):
        if part is not None:
            current = current / part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Non-directory or symlink in artifact path: {current}")


def cargo_candidates(messages: list[Path]) -> tuple[list[tuple[Path, str]], int, int]:
    candidates = []
    retained = 0
    noise = 0
    for message_file in messages:
        before = message_file.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise ValueError(f"Cargo log must be an owned regular file: {message_file}")
        finished = False
        with message_file.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    noise += 1
                    continue
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("reason"), str)
                    or record["reason"] not in CARGO_REASONS
                ):
                    noise += 1
                    continue
                if finished:
                    raise ValueError("Cargo messages follow build-finished")
                if record["reason"] == "build-finished":
                    if record.get("success") is not True:
                        raise ValueError("Cargo build did not finish successfully")
                    finished = True
                    continue
                if record["reason"] != "compiler-artifact":
                    continue
                target = record.get("target")
                profile = record.get("profile")
                if not isinstance(target, dict) or not isinstance(profile, dict):
                    raise ValueError("Malformed Cargo artifact target/profile")
                name, kinds = target.get("name"), target.get("kind")
                executable, filenames = record.get("executable"), record.get(
                    "filenames"
                )
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(kinds, list)
                    or not kinds
                    or not all(isinstance(kind, str) for kind in kinds)
                    or type(profile.get("test")) is not bool
                    or not isinstance(filenames, list)
                    or not all(isinstance(filename, str) for filename in filenames)
                    or (executable is not None and not isinstance(executable, str))
                ):
                    raise ValueError("Malformed Cargo compiler-artifact")
                kind_set = set(kinds)
                eligible_kind = kind_set <= LIBRARY_KINDS or kind_set in (
                    {"bin"},
                    {"test"},
                )
                if (
                    not profile["test"]
                    or not eligible_kind
                    or "proc-macro" in target.get("crate_types", [])
                    or name.replace("-", "_") == PRESERVED_TARGET
                    or executable is None
                ):
                    retained += 1
                    continue
                if executable not in filenames:
                    raise ValueError(
                        "Executable is absent from Cargo artifact filenames"
                    )
                candidates.append((Path(executable), name.replace("-", "_")))
        if not finished or file_identity(before) != file_identity(message_file.lstat()):
            raise ValueError("Cargo log is incomplete or changed during preflight")
    return candidates, retained, noise


def plan_cleanup(
    messages: list[Path], workspace: Path, target_dir: Path
) -> CleanupPlan:
    """Validate the complete selection without deleting any file."""
    absolute_path(workspace)
    absolute_path(target_dir)
    if target_dir == workspace or not target_dir.is_relative_to(workspace):
        raise ValueError("Cargo target directory must be inside this workspace")
    directory_chain(workspace, target_dir)
    if (
        workspace.stat().st_uid != os.getuid()
        or target_dir.stat().st_uid != os.getuid()
    ):
        raise ValueError(
            "Workspace and Cargo target directory must be owned by this user"
        )
    candidates, retained, noise = cargo_candidates(messages)
    selected = []
    missing, hardlinks = [], []
    seen = set()
    duplicates = 0
    for path, name in candidates:
        absolute_path(path)
        if (
            not path.is_relative_to(target_dir)
            or path.parent.name != "deps"
            or set(path.relative_to(target_dir).parts)
            & {"build", "incremental", ".fingerprint", "examples"}
            or not re.fullmatch(re.escape(name) + r"-[0-9a-f]{16}", path.name)
        ):
            raise ValueError(
                f"Not a hashed Cargo test executable under target/deps: {path}"
            )
        if path in seen:
            duplicates += 1
            continue
        seen.add(path)
        try:
            directory_chain(target_dir, path.parent)
            info = path.lstat()
        except FileNotFoundError:
            missing.append(str(path))
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or not info.st_mode & 0o111
        ):
            raise ValueError(
                f"Test executable is not an owned regular executable: {path}"
            )
        if info.st_nlink != 1:
            hardlinks.append(str(path))
            continue
        selected.append(
            Executable(path, file_identity(info), info.st_size, info.st_blocks * 512)
        )
    return CleanupPlan(
        workspace,
        tuple(selected),
        len(seen),
        duplicates,
        tuple(missing),
        tuple(hardlinks),
        retained,
        noise,
    )


def execute_cleanup(plan: CleanupPlan) -> dict:
    """Recheck every identity before starting, and again immediately before unlink."""
    for executable in plan.executables:
        directory_chain(plan.workspace, executable.path.parent)
        if file_identity(executable.path.lstat()) != executable.identity:
            raise ValueError(
                f"Test executable changed before cleanup: {executable.path}"
            )
    removed = []
    for executable in plan.executables:
        directory_chain(plan.workspace, executable.path.parent)
        if file_identity(executable.path.lstat()) != executable.identity:
            raise ValueError(
                f"Test executable changed before unlink: {executable.path}"
            )
        executable.path.unlink()
        removed.append(executable)
    return {
        "candidate_count": plan.candidate_count,
        "selected_count": len(plan.executables),
        "removed_count": len(removed),
        "selected_logical_bytes": sum(item.logical_bytes for item in plan.executables),
        "selected_allocated_bytes": sum(
            item.allocated_bytes for item in plan.executables
        ),
        "removed_logical_bytes": sum(item.logical_bytes for item in removed),
        "removed_allocated_bytes": sum(item.allocated_bytes for item in removed),
        "duplicate_records": plan.duplicate_records,
        "skipped_missing": len(plan.missing),
        "skipped_hardlinks": len(plan.hardlinks),
        "retained_artifacts": plan.retained_artifacts,
        "ignored_stdout_lines": plan.ignored_stdout_lines,
        "removed_files": [str(item.path) for item in removed],
    }


def filesystem_observations(target_dir: Path) -> dict:
    observations = {}
    for label, value in (
        ("target", str(target_dir)),
        ("HOME", os.environ.get("HOME")),
        ("RUNNER_TEMP", os.environ.get("RUNNER_TEMP")),
    ):
        if value is None:
            observations[label] = {"path": None, "available": False}
            continue
        path = Path(value)
        info, filesystem = path.stat(), os.statvfs(path)
        observations[label] = {
            "path": str(path),
            "device": info.st_dev,
            "available_bytes": filesystem.f_bavail * filesystem.f_frsize,
            "free_bytes": filesystem.f_bfree * filesystem.f_frsize,
            "total_bytes": filesystem.f_blocks * filesystem.f_frsize,
            "available_inodes": filesystem.f_favail,
            "free_inodes": filesystem.f_ffree,
            "total_inodes": filesystem.f_files,
        }
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-messages", type=Path, action="append", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = plan_cleanup(args.cargo_messages, args.workspace, args.target_dir)
    before = filesystem_observations(args.target_dir)
    result = execute_cleanup(plan)
    result.update(
        filesystem_before=before,
        filesystem_after=filesystem_observations(args.target_dir),
        allocated_bytes_are_file_accounting_not_a_free_space_claim=True,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
