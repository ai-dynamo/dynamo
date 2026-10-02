# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Recovery-visible behavior under stale ownership and interrupted commits."""

import signal
import subprocess
import sys
from dataclasses import replace

import pytest
from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.backends.mmap import MmapKVRecoveryManager
from gpu_memory_service.kv_recovery.types import (
    KVBlockRecord,
    KVRecoveryManifest,
    KVTensorLayout,
    RecoveryResult,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def binding():
    allocations = (PersistentAllocation("kv", 4096, "server", "GPU-0"),)
    manifest = KVRecoveryManifest(
        "model",
        8,
        16,
        "sha256",
        (KVTensorLayout("layer", "kv", 0, (8, 16), (16, 1), "float16"),),
        "0.30.0",
    )
    return allocations, manifest


def _record(block_id=1):
    return KVBlockRecord(block_id, b"x" * 32 + b"\0" * 4, 16)


def test_takeover_recovers_only_published_blocks_and_rejects_old_writer(
    tmp_path, binding
):
    manager = MmapKVRecoveryManager(tmp_path / "metadata")
    first = manager.recover(*binding)
    try:
        assert first.result is RecoveryResult.FRESH
        first.publish_blocks([_record(1), _record(2)])
        first.invalidate_blocks([1])
        replacement = manager.recover(*binding)
        try:
            assert replacement.result is RecoveryResult.RECOVERED
            assert replacement.blocks() == (_record(2),)
            for mutation in (
                lambda: first.publish_blocks([_record()]),
                lambda: first.invalidate_blocks([2]),
                first.clear,
            ):
                with pytest.raises(RuntimeError, match="stale"):
                    mutation()
            replacement.clear()
        finally:
            replacement.close()
        final = manager.recover(*binding)
        try:
            assert final.blocks() == ()
        finally:
            final.close()
    finally:
        first.close()


@pytest.mark.parametrize("change", ["allocation", "layout", "corruption"])
def test_incompatible_or_corrupt_metadata_starts_fresh(tmp_path, binding, change):
    manager = MmapKVRecoveryManager(tmp_path / "metadata")
    first = manager.recover(*binding)
    first.publish_blocks([_record()])
    allocations, manifest = binding
    if change == "allocation":
        allocations = (replace(allocations[0], server_nonce="new-server"),)
    elif change == "layout":
        manifest = replace(manifest, hash_algorithm="xxhash")
    else:
        first._mapping[4096 + 128 + 8] ^= 1
    replacement = manager.recover(allocations, manifest)
    try:
        assert replacement.result is RecoveryResult.FRESH
        assert replacement.blocks() == ()
    finally:
        replacement.close()
        first.close()


_CRASH = """
import os, signal, sys
from gpu_memory_service.common.persistent_pool import PersistentAllocation
from gpu_memory_service.kv_recovery.backends.mmap import (
    MmapKVRecoveryManager, MmapRecoverySession, _EPOCH)
from gpu_memory_service.kv_recovery.types import (
    KVBlockRecord, KVRecoveryManifest, KVTensorLayout)
allocations = (PersistentAllocation('kv', 4096, 'server', 'GPU-0'),)
manifest = KVRecoveryManifest('model', 8, 16, 'sha256',
    (KVTensorLayout('layer', 'kv', 0, (8, 16), (16, 1), 'float16'),), '0.30.0')
original = MmapRecoverySession._store
def interrupted(self, offset, value):
    if sys.argv[2] == 'publish' and offset == 4096 + 128 and value == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    original(self, offset, value)
    if sys.argv[2] == 'owner' and offset == _EPOCH:
        os.kill(os.getpid(), signal.SIGKILL)
MmapRecoverySession._store = interrupted
session = MmapKVRecoveryManager(sys.argv[1]).recover(allocations, manifest)
session.publish_blocks([KVBlockRecord(1, b'y' * 32 + b'\\0' * 4, 16)])
"""


@pytest.mark.parametrize("phase", ["publish", "owner"])
def test_sigkill_during_commit_cannot_publish_torn_state(tmp_path, binding, phase):
    path = tmp_path / "metadata"
    manager = MmapKVRecoveryManager(path)
    first = manager.recover(*binding)
    first.publish_blocks([_record(1), _record(2)])
    crashed = subprocess.run(
        [sys.executable, "-c", _CRASH, str(path), phase],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    replacement = manager.recover(*binding)
    try:
        if phase == "publish":
            assert replacement.result is RecoveryResult.RECOVERED
            assert replacement.blocks() == (_record(2),)
        else:
            assert replacement.result is RecoveryResult.FRESH
            assert replacement.blocks() == ()
    finally:
        replacement.close()
        first.close()
