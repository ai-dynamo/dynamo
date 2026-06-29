# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failover at scale — push a lot of KV through, restart, verify every byte.

Same architecture as test_failover_replay but at volume:
  - 64 layers
  - 256 blocks per layer
  - 8 KiB per block
  - Total: 128 MiB of HBM spilled to the daemon

Each block gets a unique deterministic pattern (function of layer, block_id)
so we can verify byte-equality per-block after restart.

Engine A spills all 16384 (64x256) blocks, then closes its handle (simulating
crash). Engine B opens a fresh HBM allocation with the SAME engine_id and
restores every block to a remapped offset (block i -> block N-1-i). The test
asserts byte-equality across every layer and every block — zero recompute,
all bytes recovered from the daemon's storage tier.
"""

from __future__ import annotations

import os
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


from tests.test_demote_hbm_to_storage import (  # noqa: E402
    _FakeGpuDirectBackend,
    _spawn_daemon,
)

N_LAYERS = 64
N_BLOCKS = 256
BLOCK_BYTES = 8 * 1024  # 8 KiB
TOTAL_BYTES = N_LAYERS * N_BLOCKS * BLOCK_BYTES  # 128 MiB


def _pattern_bytes(
    layer: int, block: int, block_bytes: int, *, generation: int = 0
) -> bytes:
    """Deterministic, distinct-per-(generation, layer, block) byte pattern."""
    seed = (layer * 1009 + block * 31337 + generation * 2654435761) & 0xFFFFFFFF
    out = bytearray(block_bytes)
    x = seed
    for i in range(block_bytes):
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


def _pattern(layer: int, block: int) -> bytes:
    return _pattern_bytes(layer, block, BLOCK_BYTES)


def _make_layers_sized(
    dev: torch.Tensor, n_layers: int, n_blocks: int, block_bytes: int
):
    """Build LayerDesc-shaped dicts the daemon expects."""
    layer_stride = n_blocks * block_bytes
    base = int(dev.data_ptr())
    return [
        {
            "layer_idx": L,
            "va": base + L * layer_stride,
            "size": layer_stride,
            "stride": block_bytes,
        }
        for L in range(n_layers)
    ]


def _make_layers(dev: torch.Tensor):
    """Build LayerDesc-shaped dicts the daemon expects (matches the
    shape used in test_failover_replay._make_layers)."""
    return _make_layers_sized(dev, N_LAYERS, N_BLOCKS, BLOCK_BYTES)


def _open_handle(eid: str, sock: str, layers: list):
    from gms_kv_ring.engines.handle import GMSKvRing

    return GMSKvRing(
        engine_id=eid,
        daemon_socket=sock,
        layers=layers,
    )


@pytest.mark.timeout(120)
def test_failover_at_scale_128mib_all_bytes_survive(tmp_path):
    """End-to-end: 128 MiB KV spilled → engine restart → every byte
    restored byte-identically from the daemon."""
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "scale-failover.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    eid = "scale-failover"

    try:
        # === ENGINE A: allocate HBM, stamp pattern, spill all blocks ===
        dev_a = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device="cuda")
        layer_stride = N_BLOCKS * BLOCK_BYTES

        # Stamp every block with its unique pattern.
        t0 = time.monotonic()
        host_scratch = torch.zeros(layer_stride, dtype=torch.uint8)
        for L in range(N_LAYERS):
            buf = bytearray(layer_stride)
            for B in range(N_BLOCKS):
                buf[B * BLOCK_BYTES : (B + 1) * BLOCK_BYTES] = _pattern(L, B)
            # bytearray is writable, frombuffer needs a writable buffer
            host_scratch.copy_(
                torch.frombuffer(buf, dtype=torch.uint8),
            )
            start = L * layer_stride
            dev_a[start : start + layer_stride].copy_(host_scratch)
        torch.cuda.synchronize()
        t_stamp = time.monotonic() - t0
        print(
            f"\n[scale] stamped {TOTAL_BYTES // (1024 * 1024)} MiB "
            f"into HBM in {t_stamp:.2f}s",
        )

        layers_a = _make_layers(dev_a)
        h_a = _open_handle(eid, sock, layers_a)
        try:
            t0 = time.monotonic()
            # Spill every block of every layer.
            n_spilled = 0
            for L in range(N_LAYERS):
                for B in range(N_BLOCKS):
                    ok = h_a.demote_hbm_to_storage(
                        L,
                        B * BLOCK_BYTES,
                        BLOCK_BYTES,
                        generation=1,
                    )
                    assert ok, f"A's demote failed at layer={L} block={B}"
                    n_spilled += 1
            t_spill = time.monotonic() - t0
            print(
                f"[scale] spilled {n_spilled} blocks "
                f"({TOTAL_BYTES // (1024 * 1024)} MiB) "
                f"to daemon storage in {t_spill:.2f}s "
                f"({TOTAL_BYTES / (t_spill * 1024 * 1024):.0f} MiB/s)",
            )
        finally:
            h_a.close()

        # Free A's HBM — proves the bytes are gone from GPU memory.
        del dev_a
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # === ENGINE B: fresh HBM, same engine_id ===
        dev_b = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device="cuda")
        # B initializes to zero; if a block isn't restored, the
        # subsequent equality check will fail loudly.
        torch.cuda.synchronize()
        layers_b = _make_layers(dev_b)
        h_b = _open_handle(eid, sock, layers_b)
        try:
            t0 = time.monotonic()
            n_restored = 0
            # Remap: original block i -> new block (N-1-i). Proves the
            # restore writes to the destination we asked, not a stale
            # cached VA from A's handle.
            for L in range(N_LAYERS):
                for B in range(N_BLOCKS):
                    dst = N_BLOCKS - 1 - B
                    ok = h_b.promote_storage_to_hbm(
                        L,
                        B * BLOCK_BYTES,
                        BLOCK_BYTES,
                        dest_offset=dst * BLOCK_BYTES,
                        expected_generation=0,  # failover opt-out
                    )
                    assert (
                        ok
                    ), f"B's restore failed at layer={L} src_block={B} dst_block={dst}"
                    n_restored += 1
            t_restore = time.monotonic() - t0
            print(
                f"[scale] restored {n_restored} blocks in "
                f"{t_restore:.2f}s "
                f"({TOTAL_BYTES / (t_restore * 1024 * 1024):.0f} MiB/s)",
            )

            # === BYTE-FOR-BYTE VERIFICATION ===
            t0 = time.monotonic()
            host = dev_b.cpu()
            mismatches = []
            for L in range(N_LAYERS):
                for B in range(N_BLOCKS):
                    dst = N_BLOCKS - 1 - B
                    start = L * layer_stride + dst * BLOCK_BYTES
                    got = bytes(host[start : start + BLOCK_BYTES].numpy())
                    want = _pattern(L, B)  # A's original at (L, B)
                    if got != want:
                        mismatches.append((L, B, dst))
                        if len(mismatches) > 5:
                            break
                if len(mismatches) > 5:
                    break
            t_verify = time.monotonic() - t0
            print(
                f"[scale] verified {n_restored} blocks in {t_verify:.2f}s",
            )
            assert not mismatches, f"byte mismatches after failover: {mismatches[:5]}"
            print(
                f"[scale] PASS — all {TOTAL_BYTES // (1024 * 1024)} MiB of "
                f"KV survived engine restart byte-identically",
            )
        finally:
            h_b.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=5)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _stamp_generation(
    dev: torch.Tensor,
    *,
    n_layers: int,
    n_blocks: int,
    block_bytes: int,
    generation: int,
) -> None:
    layer_stride = n_blocks * block_bytes
    host_scratch = torch.zeros(layer_stride, dtype=torch.uint8)
    for layer in range(n_layers):
        buf = bytearray(layer_stride)
        for block in range(n_blocks):
            start = block * block_bytes
            buf[start : start + block_bytes] = _pattern_bytes(
                layer, block, block_bytes, generation=generation
            )
        host_scratch.copy_(torch.frombuffer(buf, dtype=torch.uint8))
        start = layer * layer_stride
        dev[start : start + layer_stride].copy_(host_scratch)
    torch.cuda.synchronize()


def _verify_generation_remap(
    dev: torch.Tensor,
    *,
    n_layers: int,
    n_blocks: int,
    block_bytes: int,
    generation: int,
    block_shift: int,
) -> None:
    layer_stride = n_blocks * block_bytes
    host = dev.cpu()
    mismatches = []
    for layer in range(n_layers):
        for block in range(n_blocks):
            dst = (block + block_shift) % n_blocks
            start = layer * layer_stride + dst * block_bytes
            got = bytes(host[start : start + block_bytes].numpy().tobytes())
            want = _pattern_bytes(layer, block, block_bytes, generation=generation)
            if got != want:
                mismatches.append((layer, block, dst, generation))
                if len(mismatches) >= 5:
                    break
        if len(mismatches) >= 5:
            break
    assert not mismatches, f"byte mismatches after pressure failover: {mismatches}"


@pytest.mark.timeout(180)
def test_failover_small_kv_pressure_repeated_cycles(tmp_path):
    """Repeated small-KV failover pressure.

    The large 128 MiB test proves bulk correctness. This test stresses the
    transition shape we care about for production: a small reusable KV namespace
    is overwritten many times, the primary handle detaches every cycle, and a
    replacement handle restores the latest generation into a remapped HBM layout.
    A wrong-generation restore must fail before any bytes are consumed.
    """
    long_stress = os.environ.get("GMS_KV_RING_LONG_STRESS") == "1"
    cycles = _env_int(
        "GMS_KV_RING_FAILOVER_PRESSURE_CYCLES", 256 if long_stress else 32
    )
    n_layers = _env_int("GMS_KV_RING_FAILOVER_PRESSURE_LAYERS", 4)
    n_blocks = _env_int("GMS_KV_RING_FAILOVER_PRESSURE_BLOCKS", 32)
    block_bytes = _env_int("GMS_KV_RING_FAILOVER_PRESSURE_BLOCK_BYTES", 4 * 1024)
    total_bytes = n_layers * n_blocks * block_bytes
    from gms_kv_ring.common import metrics

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "small-pressure-failover.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    eid = "small-pressure-failover"
    generation_mismatches_before = metrics.daemon_generation_mismatches.get(
        engine_id=eid
    )
    stale_demotes_before = metrics.daemon_stale_demotes.get(engine_id=eid)

    try:
        order = [
            (layer, block) for layer in range(n_layers) for block in range(n_blocks)
        ]
        for cycle in range(cycles):
            generation = cycle + 1

            # Primary: stamp a tiny KV pool, spill every block, then disappear.
            dev_a = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
            _stamp_generation(
                dev_a,
                n_layers=n_layers,
                n_blocks=n_blocks,
                block_bytes=block_bytes,
                generation=generation,
            )
            layers_a = _make_layers_sized(dev_a, n_layers, n_blocks, block_bytes)
            h_a = _open_handle(eid, sock, layers_a)
            try:
                shifted_order = (
                    order[cycle % len(order) :] + order[: cycle % len(order)]
                )
                for layer, block in shifted_order:
                    assert h_a.demote_hbm_to_storage(
                        layer,
                        block * block_bytes,
                        block_bytes,
                        generation=generation,
                    ), f"demote failed at cycle={cycle} layer={layer} block={block}"
            finally:
                h_a.close()

            del dev_a
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Shadow/replacement: fresh HBM allocation, same engine_id, same
            # storage namespace. Wrong-generation restore must fail; current
            # generation must restore byte-identically into a remapped layout.
            dev_b = torch.zeros(total_bytes, dtype=torch.uint8, device="cuda")
            layers_b = _make_layers_sized(dev_b, n_layers, n_blocks, block_bytes)
            h_b = _open_handle(eid, sock, layers_b)
            try:
                assert not h_b.promote_storage_to_hbm(
                    0,
                    0,
                    block_bytes,
                    expected_generation=generation + 1,
                )
                if generation > 1:
                    assert not h_b.demote_hbm_to_storage(
                        0,
                        0,
                        block_bytes,
                        generation=generation - 1,
                    )
                block_shift = cycle % n_blocks
                for layer, block in shifted_order:
                    dst = (block + block_shift) % n_blocks
                    assert h_b.promote_storage_to_hbm(
                        layer,
                        block * block_bytes,
                        block_bytes,
                        dest_offset=dst * block_bytes,
                        expected_generation=generation,
                    ), f"restore failed at cycle={cycle} layer={layer} block={block}"
                torch.cuda.synchronize()
                _verify_generation_remap(
                    dev_b,
                    n_layers=n_layers,
                    n_blocks=n_blocks,
                    block_bytes=block_bytes,
                    generation=generation,
                    block_shift=block_shift,
                )
            finally:
                h_b.close()

            del dev_b
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            if (cycle + 1) % max(1, cycles // 4) == 0:
                print(
                    f"[pressure] completed {cycle + 1}/{cycles} cycles "
                    f"with {total_bytes // 1024} KiB KV namespace "
                    f"({n_layers} layers x {n_blocks} blocks x {block_bytes} bytes)"
                )

        assert len(fake.demote_calls) == cycles * n_layers * n_blocks
        assert len(fake.promote_calls) == cycles * n_layers * n_blocks
        assert (
            metrics.daemon_generation_mismatches.get(engine_id=eid)
            - generation_mismatches_before
        ) == cycles
        assert (
            metrics.daemon_stale_demotes.get(engine_id=eid) - stale_demotes_before
        ) == max(0, cycles - 1)
        assert metrics.daemon_persisted_generations.get(engine_id=eid) == n_blocks
        print(
            f"[pressure] PASS — {cycles} small-KV failover cycles, "
            f"{cycles * n_layers * n_blocks} block restores, no stale-generation reuse"
        )
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=5)
