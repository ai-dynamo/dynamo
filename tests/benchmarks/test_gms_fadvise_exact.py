# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact GMS checkpoint cache-eviction helper."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
    pytest.mark.unit,
]

HELPER_PATH = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "gms_cuda_init_ab"
    / "gms-fadvise-exact.py"
)
RUNNER_PATH = HELPER_PATH.with_name("run-variant.sh")
CACHE_HELPER_PATH = HELPER_PATH.with_name("cache-helper.yaml")
SPEC = importlib.util.spec_from_file_location("gms_fadvise_exact", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def test_reports_machine_readable_per_root_success(tmp_path, capsys):
    roots = [tmp_path / "checkpoint", tmp_path / "nvme"]
    sizes = [3, 5]
    for root, size in zip(roots, sizes):
        root.mkdir()
        (root / "block").write_bytes(b"x" * size)

    assert HELPER.main([str(root) for root in roots]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "roots": [
            {
                "root": str(root),
                "status": "ok",
                "files": 1,
                "bytes": size,
                "errors": 0,
            }
            for root, size in zip(roots, sizes)
        ],
        "total": {
            "roots": 2,
            "files": 2,
            "bytes": sum(sizes),
            "errors": 0,
        },
    }


def test_rejects_every_invalid_root(tmp_path, capsys):
    missing = tmp_path / "missing"
    not_directory = tmp_path / "checkpoint-file"
    not_directory.write_text("checkpoint", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    assert HELPER.main([str(missing), str(not_directory), str(empty)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert [(result["status"], result["files"]) for result in payload["roots"]] == [
        ("missing", 0),
        ("not_directory", 0),
        ("empty", 0),
    ]


def test_rejects_per_root_eviction_error(tmp_path, capsys, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "block").write_bytes(b"checkpoint")

    def fail_fadvise(*_args):
        raise OSError("injected fadvise failure")

    monkeypatch.setattr(HELPER.os, "posix_fadvise", fail_fadvise)

    assert HELPER.main([str(root)]) == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["roots"] == [
        {
            "root": str(root),
            "status": "error",
            "files": 0,
            "bytes": 0,
            "errors": 1,
        }
    ]
    assert "injected fadvise failure" in captured.err


def test_cache_helper_is_root_with_least_privilege_read_only_mounts():
    pod = yaml.safe_load(CACHE_HELPER_PATH.read_text(encoding="utf-8"))
    spec = pod["spec"]
    container = spec["containers"][0]
    security = container["securityContext"]

    assert pod["metadata"]["name"] == "gms-cuda-init-cache-helper-root-v2"
    assert spec["automountServiceAccountToken"] is False
    assert spec["hostNetwork"] is False
    assert spec["hostPID"] is False
    assert "resourceClaims" not in spec
    assert "resources" not in container
    assert security == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "privileged": False,
        "readOnlyRootFilesystem": True,
        "runAsGroup": 0,
        "runAsUser": 0,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert {mount["name"] for mount in container["volumeMounts"]} == {
        "checkpoint-storage",
        "nvme2",
        "nvme4",
        "nvme5",
        "nvme6",
        "nvme7",
        "nvme8",
        "nvme9",
    }
    assert all(mount["readOnly"] is True for mount in container["volumeMounts"])


def run_access_preflight_harness(tmp_path, uid="0", gid="0", blocked_root=""):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "id").write_text(
        """#!/bin/sh
case "$1" in
    -u) printf '%s\\n' "$FAKE_UID" ;;
    -g) printf '%s\\n' "$FAKE_GID" ;;
    *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    (bin_dir / "id").chmod(0o755)
    (bin_dir / "find").write_text(
        """#!/bin/sh
printf '%s\\n' "$1" >> "$FAKE_FIND_LOG"
test "$1" != "$FAKE_BLOCKED_ROOT"
""",
        encoding="utf-8",
    )
    (bin_dir / "find").chmod(0o755)

    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    functions = runner_source.partition("\nstamp PREFLIGHT_BEGIN")[0]
    harness = tmp_path / "access-preflight-harness.sh"
    harness.write_text(
        f"""{functions}
trap - EXIT INT TERM
k() {{
    while [[ "$1" != -- ]]; do
        shift
    done
    shift
    "$@"
}}
validate_cache_helper_access \\
    "$ART/preflight/cache-helper-access.txt" \\
    "$ART/preflight/cache-helper-access.err" \\
    "${{NFS_CACHE_ROOTS[@]}}"
""",
        encoding="utf-8",
    )
    harness.chmod(0o755)

    evidence = tmp_path / "evidence"
    find_log = tmp_path / "find.log"
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_UID": uid,
        "FAKE_GID": gid,
        "FAKE_BLOCKED_ROOT": blocked_root,
        "FAKE_FIND_LOG": str(find_log),
    }
    result = subprocess.run(
        [harness, "a", evidence],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result, evidence, find_log


def test_cache_helper_access_preflight_records_root_and_both_exact_roots(tmp_path):
    result, evidence, find_log = run_access_preflight_harness(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (evidence / "preflight" / "cache-helper-access.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "uid=0 gid=0",
        "traversable\t/checkpoints/57a124961e2a47a2cf9c2712e58a0a2b/versions/1",
        ("traversable\t/checkpoints/gms/" "g52-t8-gms-prof-r29604929787-r2/versions/1"),
    ]
    assert find_log.read_text(encoding="utf-8").splitlines() == [
        "/checkpoints/57a124961e2a47a2cf9c2712e58a0a2b/versions/1",
        "/checkpoints/gms/g52-t8-gms-prof-r29604929787-r2/versions/1",
    ]


@pytest.mark.parametrize(
    ("uid", "gid", "blocked_root"),
    [
        ("1000", "0", ""),
        ("0", "1000", ""),
        ("0", "0", "/checkpoints/57a124961e2a47a2cf9c2712e58a0a2b/versions/1"),
        (
            "0",
            "0",
            "/checkpoints/gms/g52-t8-gms-prof-r29604929787-r2/versions/1",
        ),
    ],
)
def test_cache_helper_access_preflight_fails_closed(tmp_path, uid, gid, blocked_root):
    result, _, _ = run_access_preflight_harness(
        tmp_path, uid=uid, gid=gid, blocked_root=blocked_root
    )

    assert result.returncode != 0


def test_runner_validates_root_identity_and_exact_nfs_traversal_before_fadvise():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    invocation = source.index(
        "validate_cache_helper_access \\\n", source.index("k apply")
    )
    fadvise = source.index("stamp FADVISE_NFS_BEGIN")

    assert "CACHE_HELPER=gms-cuda-init-cache-helper-root-v2" in source
    assert invocation < fadvise
    assert source.count('"${NFS_CACHE_ROOTS[@]}"') == 2
    preflight = source[source.index("validate_cache_helper_access()") : invocation]
    assert "uid=$(id -u)" in preflight
    assert "gid=$(id -g)" in preflight
    assert '[ "$uid" -eq 0 ]' in preflight
    assert '[ "$gid" -eq 0 ]' in preflight
    assert 'find "$root" -print >/dev/null' in preflight
    assert "printf 'uid=0 gid=0\\n'" in preflight
    assert "printf 'traversable\\t%s\\n' \"$@\"" in preflight


def test_runner_rejects_non_empty_evidence_directory(tmp_path):
    stale = tmp_path / "evidence" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [RUNNER_PATH, "a", stale.parent],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to reuse non-empty evidence directory" in result.stderr
    assert stale.read_text(encoding="utf-8") == "keep"
    assert sorted(stale.parent.iterdir()) == [stale]


def test_runner_rejects_symlink_to_non_empty_evidence_directory(tmp_path):
    stale = tmp_path / "stale-evidence" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("keep", encoding="utf-8")
    evidence_link = tmp_path / "evidence"
    evidence_link.symlink_to(stale.parent, target_is_directory=True)

    result = subprocess.run(
        [RUNNER_PATH, "a", evidence_link],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing evidence path symlink" in result.stderr
    assert stale.read_text(encoding="utf-8") == "keep"
    assert sorted(stale.parent.iterdir()) == [stale]
    assert evidence_link.is_symlink()


def write_fake_cleanup_commands(bin_dir):
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [[ "$1" != "--context" ||
      "$2" != "nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster" ||
      "$3" != "-n" || "$4" != "schwinns" ]]; then
    echo "kubectl call is not explicitly scoped: $*" >&2
    exit 90
fi
printf '%s\\n' "$*" >> "$FAKE_KUBECTL_LOG"
shift 4

if [[ "$1" == patch && "$2" == dynamographdeployment ]]; then
    attempts=$(cat "$FAKE_PATCH_ATTEMPTS")
    attempts=$((attempts + 1))
    printf '%s\\n' "$attempts" > "$FAKE_PATCH_ATTEMPTS"
    if [[ "$attempts" -le "$FAKE_PATCH_FAILURES" ]]; then
        echo "injected patch failure" >&2
        exit 7
    fi
    printf '0\\n' > "$FAKE_DGD_REPLICAS"
    printf 'patched\\n'
    exit
fi

if [[ "$1" != get ]]; then
    echo "unexpected kubectl call: $*" >&2
    exit 91
fi
case "$2" in
    dynamographdeployment)
        replicas=$(cat "$FAKE_DGD_REPLICAS")
        printf '{"spec":{"components":['
        printf '{"name":"Frontend","replicas":1},'
        printf '{"name":"VllmDecodeWorker","replicas":%s}' "$replicas"
        printf ']}}\\n'
        ;;
    dynamocomponentdeployments|deployments|resourceclaims)
        printf '{"items":[]}\\n'
        ;;
    pods)
        ;;
    *)
        echo "unexpected kubectl get: $*" >&2
        exit 92
        ;;
esac
""",
        encoding="utf-8",
    )
    kubectl.chmod(0o755)

    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "sleep").chmod(0o755)
    (bin_dir / "seq").write_text("#!/bin/sh\nprintf '1\\n2\\n3\\n'\n", encoding="utf-8")
    (bin_dir / "seq").chmod(0o755)


def run_cleanup_harness(tmp_path, patch_failures):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_fake_cleanup_commands(bin_dir)

    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    functions = runner_source.partition("\nstamp PREFLIGHT_BEGIN")[0]
    harness = tmp_path / "cleanup-harness.sh"
    harness.write_text(
        f"{functions}\nCLEANUP_OWNED=1\nZERO_CONFIRMED=0\nexit 0\n",
        encoding="utf-8",
    )
    harness.chmod(0o755)

    state = {
        "FAKE_DGD_REPLICAS": tmp_path / "dgd-replicas",
        "FAKE_PATCH_ATTEMPTS": tmp_path / "patch-attempts",
        "FAKE_KUBECTL_LOG": tmp_path / "kubectl.log",
    }
    state["FAKE_DGD_REPLICAS"].write_text("1\n", encoding="utf-8")
    state["FAKE_PATCH_ATTEMPTS"].write_text("0\n", encoding="utf-8")
    state["FAKE_KUBECTL_LOG"].touch()
    evidence = tmp_path / "evidence"
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_PATCH_FAILURES": str(patch_failures),
        **{name: str(path) for name, path in state.items()},
    }

    result = subprocess.run(
        [harness, "a", evidence],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, evidence, state


def test_cleanup_patch_failure_does_not_confirm_or_finalize(tmp_path):
    result, evidence, state = run_cleanup_harness(tmp_path, patch_failures=3)

    assert result.returncode != 0
    assert state["FAKE_DGD_REPLICAS"].read_text(encoding="utf-8").strip() == "1"
    assert state["FAKE_PATCH_ATTEMPTS"].read_text(encoding="utf-8").strip() == "3"
    timestamps = (evidence / "timestamps.tsv").read_text(encoding="utf-8")
    assert "SCALE_DOWN_CONFIRMED" not in timestamps
    assert "RUNNER_EXIT\tstatus=1" in timestamps
    assert not (evidence / "SHA256SUMS").exists()


def test_cleanup_retries_until_dgd_zero_and_checksums_all_evidence(tmp_path):
    result, evidence, state = run_cleanup_harness(tmp_path, patch_failures=1)

    assert result.returncode == 0, result.stderr
    assert state["FAKE_DGD_REPLICAS"].read_text(encoding="utf-8").strip() == "0"
    assert state["FAKE_PATCH_ATTEMPTS"].read_text(encoding="utf-8").strip() == "2"
    timestamps = (evidence / "timestamps.tsv").read_text(encoding="utf-8")
    assert "SCALE_DOWN_CONFIRMED\tattempt=2 patch_status=0" in timestamps
    assert "RUNNER_EXIT\tstatus=0" in timestamps

    manifest = evidence / "SHA256SUMS"
    verification = subprocess.run(
        ["sha256sum", "-c", manifest],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verification.returncode == 0, verification.stderr

    evidence_files = {
        str(path) for path in evidence.rglob("*") if path.is_file() and path != manifest
    }
    checksummed_files = {
        line.split("  ", maxsplit=1)[1]
        for line in manifest.read_text(encoding="utf-8").splitlines()
    }
    assert checksummed_files == evidence_files
    assert not (evidence / ".SHA256SUMS.tmp").exists()


def test_runner_owns_cleanup_before_scale_up_and_finalizes_last():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    scale_up = source.index("stamp SCALE_UP_SENT")

    assert source.index("trap cleanup EXIT INT TERM") < scale_up
    cleanup_owned = source.index("CLEANUP_OWNED=1")
    assert cleanup_owned < scale_up
    assert source.rfind("ZERO_CONFIRMED=0", 0, scale_up) > cleanup_owned
    assert source.index("stamp RUNNER_EXIT") < source.index(
        "finalize_evidence || status=1"
    )
    assert source.index("stamp RUN_COMPLETE") > source.index(
        "capture_objects post-teardown"
    )
    assert "SHA256SUMS" not in source[source.index("stamp RUN_COMPLETE") :]
