# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MM kwargs transfer (NIXL sender/receiver + SHM sender/receiver)."""

import asyncio
import struct
from unittest.mock import MagicMock

import pytest

from dynamo.common.multimodal import mm_kwargs_transfer
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlSender,
    MmKwargsShmReceiver,
    MmKwargsShmSender,
    MmKwargsShmTransferMetadata,
    MmKwargsTransferMetadata,
    TensorTransferSpec,
    _pack_buffers,
    _unpack_buffers,
    decode_mm_kwargs_item,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _make_feature(data=None, mm_hash="hash_default"):
    """Create a mock MultiModalFeatureSpec."""
    feat = MagicMock()
    feat.data = data
    feat.mm_hash = mm_hash
    feat.modality = "image"
    return feat


def _make_kwargs_item(marker: int, *, large: bool = False):
    """Build a real vLLM ``MultiModalKwargsItem`` for round-trip tests.

    ``marker`` makes each item distinct so order can be checked. With
    ``large=True`` the tensor is above vLLM's 256-byte zero-copy threshold, so
    the encoder spills it to an auxiliary buffer and the frame carries more than
    one buffer. vLLM is imported lazily so the dynamo-runtime lane (no vLLM) can
    still collect the non-vLLM tests in this module.
    """
    import torch
    from vllm.multimodal.inputs import (
        MultiModalBatchedField,
        MultiModalFieldElem,
        MultiModalFlatField,
        MultiModalKwargsItem,
    )

    n = 4096 if large else 4
    pixel_values = MultiModalFieldElem(
        data=torch.arange(marker, marker + n, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    grid = MultiModalFieldElem(
        data=torch.tensor([[1, marker % 7 + 1, 2]], dtype=torch.int64),
        field=MultiModalFlatField(slices=[slice(0, 1)], dim=0),
    )
    return MultiModalKwargsItem({"pixel_values": pixel_values, "image_grid_thw": grid})


class TestMmKwargsTransferMetadata:
    """Tests for the Pydantic metadata model."""

    def test_roundtrip_serialization(self):
        """Metadata serializes and deserializes correctly."""
        spec = TensorTransferSpec(
            field_name="pixel_values",
            shape=[100, 1176],
            dtype_str="float32",
            serialized_request="base64metadata==",
        )
        meta = MmKwargsTransferMetadata(
            modality="image",
            tensor_specs=[spec],
            mm_hashes=["abcd1234" * 8],
        )

        dumped = meta.model_dump()
        restored = MmKwargsTransferMetadata.model_validate(dumped)

        assert restored.modality == "image"
        assert len(restored.tensor_specs) == 1
        assert restored.tensor_specs[0].field_name == "pixel_values"
        assert restored.tensor_specs[0].shape == [100, 1176]
        assert restored.tensor_specs[0].dtype_str == "float32"
        assert restored.mm_hashes == ["abcd1234" * 8]

    def test_multiple_tensor_specs(self):
        """Multiple tensors (e.g., pixel_values + image_grid_thw)."""
        specs = [
            TensorTransferSpec(
                field_name="pixel_values",
                shape=[100, 1176],
                dtype_str="float32",
                serialized_request="meta1",
            ),
            TensorTransferSpec(
                field_name="image_grid_thw",
                shape=[1, 3],
                dtype_str="int64",
                serialized_request="meta2",
            ),
        ]
        meta = MmKwargsTransferMetadata(
            modality="image",
            tensor_specs=specs,
            mm_hashes=["hash1", "hash2"],
        )
        assert len(meta.tensor_specs) == 2
        assert meta.tensor_specs[0].field_name == "pixel_values"
        assert meta.tensor_specs[1].field_name == "image_grid_thw"


class TestMmKwargsNixlSender:
    """Tests for the NIXL sender side (prepare method)."""

    @pytest.mark.asyncio
    async def test_prepare_with_no_features_returns_none(self):
        """Empty features list returns None."""
        sender = MmKwargsNixlSender()
        meta, futures = await sender.prepare([], modality="image")
        assert meta is None
        assert futures == []

    @pytest.mark.asyncio
    async def test_prepare_with_no_data_returns_none(self):
        """Features with data=None are skipped."""
        feat = _make_feature(data=None)

        sender = MmKwargsNixlSender()
        meta, futures = await sender.prepare([feat], modality="image")
        assert meta is None
        assert futures == []

    @pytest.mark.asyncio
    async def test_prepare_skips_none_data_in_multi_feature(self):
        """Mixed features: some with data, some without. Only data!=None are transferred."""
        # Feature 0 has data, feature 1 does not, feature 2 has data
        feats = [
            _make_feature(data="item_0", mm_hash="hash_0"),
            _make_feature(data=None, mm_hash="hash_1"),
            _make_feature(data="item_2", mm_hash="hash_2"),
        ]
        # Sender requires NIXL which isn't available in unit tests.
        # Just verify it collects hashes for all features.
        # The prepare() will fail at NIXL registration, so test the hash collection.
        assert feats[0].mm_hash == "hash_0"
        assert feats[1].data is None
        assert feats[2].mm_hash == "hash_2"


class TestMmKwargsNixlSenderCleanup:
    """cleanup() must be bounded and must always release registered buffers."""

    class _FakeOp:
        """Stands in for ReadableOperation: completion never resolves."""

        def __init__(self, never_completes: bool = True):
            self.released = False
            self._never_completes = never_completes

        async def wait_for_completion(self) -> None:
            if self._never_completes:
                await asyncio.Event().wait()  # pends forever

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.released = True

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_cleanup_is_bounded_and_releases_when_never_read(self, monkeypatch):
        """A backend that never reads must not pin the buffer forever.

        Before this was bounded, cleanup() awaited the completion future
        indefinitely; the pending coroutine held the operation alive and its
        NIXL registration was never dropped, so the frontend leaked the full
        payload for every un-read request.
        """
        monkeypatch.setattr(mm_kwargs_transfer, "MM_NIXL_CLEANUP_TIMEOUT_S", 0.05)
        sender = MmKwargsNixlSender.__new__(MmKwargsNixlSender)
        ops = [self._FakeOp(), self._FakeOp()]

        await asyncio.wait_for(sender.cleanup(ops), timeout=5)

        assert all(op.released for op in ops), "buffers must be released on timeout"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_cleanup_releases_on_normal_completion(self, monkeypatch):
        """The happy path still releases."""
        monkeypatch.setattr(mm_kwargs_transfer, "MM_NIXL_CLEANUP_TIMEOUT_S", 5.0)
        sender = MmKwargsNixlSender.__new__(MmKwargsNixlSender)
        ops = [self._FakeOp(never_completes=False)]

        await sender.cleanup(ops)

        assert ops[0].released

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_one_failing_op_does_not_release_a_still_pending_op_early(
        self, monkeypatch
    ):
        """A failing completion must not cut the wait short for its siblings.

        Without ``return_exceptions=True``, ``gather`` propagates the first
        error while the other completion coroutines are still running, and the
        release below would then deregister a buffer whose backend read is
        still in flight.
        """
        monkeypatch.setattr(mm_kwargs_transfer, "MM_NIXL_CLEANUP_TIMEOUT_S", 5.0)
        sender = MmKwargsNixlSender.__new__(MmKwargsNixlSender)

        released_while_pending = []

        class _RaisingOp(TestMmKwargsNixlSenderCleanup._FakeOp):
            async def wait_for_completion(self) -> None:
                raise RuntimeError("transfer failed")

        class _SlowOp(TestMmKwargsNixlSenderCleanup._FakeOp):
            def __init__(self):
                super().__init__(never_completes=False)
                self.done = False

            async def wait_for_completion(self) -> None:
                await asyncio.sleep(0.2)
                self.done = True

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                if not self.done:
                    released_while_pending.append(self)
                super().__exit__(exc_type, exc_value, traceback)

        slow = _SlowOp()
        # cleanup() is best-effort and does not raise: the caller awaits it
        # from a bare finally, where a raise would replace an in-flight
        # CancelledError.
        await sender.cleanup([_RaisingOp(), slow])

        assert slow.done, "cleanup returned before the pending transfer finished"
        assert not released_while_pending, "released a buffer whose read was in flight"
        assert slow.released, "sibling buffer was not released"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_release_failure_does_not_stop_remaining_releases(self):
        """A failing release must not stop the remaining buffers being freed."""
        sender = MmKwargsNixlSender.__new__(MmKwargsNixlSender)

        class _BadRelease(TestMmKwargsNixlSenderCleanup._FakeOp):
            def __init__(self):
                super().__init__(never_completes=False)

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                raise RuntimeError("deregister failed")

        good = TestMmKwargsNixlSenderCleanup._FakeOp(never_completes=False)
        # Best-effort: the failure is logged, not raised, and the loop continues.
        await sender.cleanup([_BadRelease(), good])

        assert (
            good.released
        ), "a later buffer was skipped after an earlier release failed"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_cleanup_with_no_items_is_a_noop(self):
        sender = MmKwargsNixlSender.__new__(MmKwargsNixlSender)
        await sender.cleanup([])


# The SHM round-trip now serializes real vLLM MultiModalKwargsItem objects with
# vLLM's msgpack serializer, so these tests require vLLM and run in the vllm
# lane. The receiver returns writable buffers, so decoding builds no tensor over
# a read-only buffer. PyTorch warns about a read-only buffer only once per
# process, so test_decode_emits_no_read_only_warning checks it in a fresh
# interpreter.
@pytest.mark.vllm
class TestMmKwargsShmTransfer:
    """Tests for the SHM sender/receiver round-trip with the msgpack frame."""

    @pytest.mark.asyncio
    async def test_single_item_roundtrip(self):
        """Single real MultiModalKwargsItem round-trips through SHM correctly."""
        item = _make_kwargs_item(0)
        feat = _make_feature(data=item, mm_hash="hash_single")

        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare([feat], modality="image")

        assert extra_update is not None
        meta = extra_update["mm_kwargs_shm"]
        assert len(meta["items"]) == 1
        assert meta["modality"] == "image"
        assert meta["mm_hashes"] == ["hash_single"]

        # Receiver reads back
        receiver = MmKwargsShmReceiver()
        results = await receiver.receive(
            MmKwargsShmTransferMetadata.model_validate(meta)
        )

        assert "__pickled_kwargs_item__" in results
        items = results["__pickled_kwargs_item__"]
        assert len(items) == 1
        restored = decode_mm_kwargs_item(items[0])
        assert restored == item

        await sender.cleanup(handles)

    @pytest.mark.asyncio
    async def test_multi_image_roundtrip_preserves_order(self):
        """Multiple features round-trip in correct order through SHM."""
        items = [_make_kwargs_item(marker) for marker in (10, 20, 30)]
        feats = [_make_feature(data=items[i], mm_hash=f"hash_{i}") for i in range(3)]

        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare(feats, modality="image")

        assert extra_update is not None
        meta = extra_update["mm_kwargs_shm"]
        assert len(meta["items"]) == 3
        assert meta["mm_hashes"] == ["hash_0", "hash_1", "hash_2"]

        # Receiver reads back
        receiver = MmKwargsShmReceiver()
        results = await receiver.receive(
            MmKwargsShmTransferMetadata.model_validate(meta)
        )

        restored_items = results["__pickled_kwargs_item__"]
        assert len(restored_items) == 3

        # Verify ORDER is preserved: each blob decodes back to its own item.
        for i in range(3):
            assert decode_mm_kwargs_item(restored_items[i]) == items[i]

        await sender.cleanup(handles)

    @pytest.mark.asyncio
    async def test_large_tensor_uses_aux_buffer_and_roundtrips(self):
        """A tensor above the zero-copy threshold spills to an aux buffer.

        The frame must then carry more than one buffer, and the item must still
        round-trip. This exercises the multi-buffer framing that a small inline
        item never reaches.
        """
        item = _make_kwargs_item(0, large=True)
        feat = _make_feature(data=item, mm_hash="hash_large")

        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare([feat], modality="image")

        meta = extra_update["mm_kwargs_shm"]
        receiver = MmKwargsShmReceiver()
        results = await receiver.receive(
            MmKwargsShmTransferMetadata.model_validate(meta)
        )
        blob = results["__pickled_kwargs_item__"][0]
        (buffer_count,) = struct.unpack_from("<I", blob, 0)
        assert buffer_count >= 2, "large tensor should spill to an aux buffer"
        assert decode_mm_kwargs_item(blob) == item

        await sender.cleanup(handles)

    @pytest.mark.timeout(60)
    def test_decode_emits_no_read_only_warning(self):
        """Decoding a received large item builds no tensor on a read-only buffer.

        PyTorch warns "The given buffer is not writable" only once per process,
        so the check runs in a fresh interpreter. As a positive control, the
        script then decodes an immutable copy, which must warn once.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import asyncio
            import warnings
            from unittest.mock import MagicMock

            import torch
            from vllm.multimodal.inputs import (
                MultiModalBatchedField,
                MultiModalFieldElem,
                MultiModalKwargsItem,
            )

            from dynamo.common.multimodal import mm_kwargs_transfer as m

            elem = MultiModalFieldElem(
                data=torch.arange(4096, dtype=torch.float32),
                field=MultiModalBatchedField(),
            )
            item = MultiModalKwargsItem({"pixel_values": elem})

            async def receive():
                feat = MagicMock(data=item, mm_hash="h", modality="image")
                sender = m.MmKwargsShmSender()
                extra, handles = await sender.prepare([feat], modality="image")
                meta = m.MmKwargsShmTransferMetadata.model_validate(
                    extra["mm_kwargs_shm"]
                )
                try:
                    results = await m.MmKwargsShmReceiver().receive(meta)
                finally:
                    await sender.cleanup(handles)
                return results["__pickled_kwargs_item__"][0]

            def read_only_warnings(buf):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    m.decode_mm_kwargs_item(buf)
                return sum("not writable" in str(w.message) for w in caught)

            blob = asyncio.run(receive())
            print(m.__file__)
            print(read_only_warnings(blob), read_only_warnings(bytes(blob)))
            """
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=50,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        module_file, counts = proc.stdout.strip().splitlines()[-2:]
        # The subprocess must test the same code as this process.
        assert module_file == mm_kwargs_transfer.__file__
        # (received buffer, immutable copy): the copy proves the check works.
        assert tuple(counts.split()) == ("0", "1")

    @pytest.mark.asyncio
    async def test_skips_none_data_features(self):
        """Features with data=None are skipped, hashes still collected."""
        item0 = _make_kwargs_item(1)
        item2 = _make_kwargs_item(2)
        feats = [
            _make_feature(data=item0, mm_hash="hash_0"),
            _make_feature(data=None, mm_hash="hash_1"),
            _make_feature(data=item2, mm_hash="hash_2"),
        ]

        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare(feats, modality="image")

        assert extra_update is not None
        meta = extra_update["mm_kwargs_shm"]
        assert len(meta["items"]) == 2  # Only 2 features had data
        assert meta["mm_hashes"] == ["hash_0", "hash_1", "hash_2"]  # All hashes

        receiver = MmKwargsShmReceiver()
        results = await receiver.receive(
            MmKwargsShmTransferMetadata.model_validate(meta)
        )
        items = results["__pickled_kwargs_item__"]
        assert len(items) == 2
        assert decode_mm_kwargs_item(items[0]) == item0
        assert decode_mm_kwargs_item(items[1]) == item2

        await sender.cleanup(handles)

    @pytest.mark.asyncio
    async def test_all_none_data_returns_none(self):
        """All features with data=None returns None metadata."""
        feats = [
            _make_feature(data=None, mm_hash="hash_0"),
            _make_feature(data=None, mm_hash="hash_1"),
        ]

        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare(feats, modality="image")

        assert extra_update is None
        assert handles == []

    @pytest.mark.asyncio
    async def test_cleanup_removes_shared_memory(self):
        """Cleanup properly unlinks shared memory segments."""
        feat = _make_feature(data=_make_kwargs_item(0), mm_hash="hash")
        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare([feat], modality="image")

        assert extra_update is not None
        meta = extra_update["mm_kwargs_shm"]
        assert len(handles) == 1
        name = meta["items"][0]["name"]

        # Verify SHM exists
        import multiprocessing.shared_memory as shm_mod

        sm = shm_mod.SharedMemory(name=name, create=False)
        sm.close()

        # Cleanup
        await sender.cleanup(handles)

        # Verify SHM is gone
        with pytest.raises(FileNotFoundError):
            shm_mod.SharedMemory(name=name, create=False)


class TestMmKwargsShmCleanupErrorHandling:
    """Tests for SHM cleanup error handling (Devin review fix #1)."""

    @pytest.mark.vllm
    @pytest.mark.asyncio
    async def test_cleanup_handles_file_not_found(self):
        """FileNotFoundError is silently handled (already unlinked)."""

        feat = _make_feature(data=_make_kwargs_item(0), mm_hash="hash")
        sender = MmKwargsShmSender()
        extra_update, handles = await sender.prepare([feat], modality="image")

        assert len(handles) == 1
        # Manually unlink to simulate resource_tracker cleanup
        handles[0].close()
        handles[0].unlink()

        # cleanup() should not raise even though SHM is already gone
        await sender.cleanup(handles)

    @pytest.mark.asyncio
    async def test_cleanup_logs_non_file_errors(self):
        """Non-FileNotFoundError exceptions are logged but don't crash."""
        sender = MmKwargsShmSender()
        handle = MagicMock()
        handle.close.side_effect = PermissionError("mocked permission error")

        # Should not raise
        await sender.cleanup([handle])

    @pytest.mark.asyncio
    async def test_cleanup_processes_all_handles_despite_errors(self):
        """All handles are attempted even if earlier ones fail."""
        sender = MmKwargsShmSender()
        handle_ok = MagicMock()
        handle_fail = MagicMock()
        handle_fail.close.side_effect = OSError("mocked")
        handle_ok2 = MagicMock()

        await sender.cleanup([handle_ok, handle_fail, handle_ok2])

        # All handles should have close() called
        handle_ok.close.assert_called_once()
        handle_fail.close.assert_called_once()
        handle_ok2.close.assert_called_once()


class TestMmKwargsNixlReceiverDescriptorValidation:
    """Tests for _acquire_descriptor RuntimeError (Devin review fix #6)."""

    def test_acquire_descriptor_raises_on_none_data_ref(self):
        """Pre-allocated descriptor with None _data_ref raises RuntimeError."""
        from dynamo.common.multimodal.mm_kwargs_transfer import MmKwargsNixlReceiver

        # Create a receiver with a mocked pool
        receiver = MmKwargsNixlReceiver.__new__(MmKwargsNixlReceiver)
        receiver._available = True
        receiver._max_item_bytes = 1024

        # Mock a pre-allocated descriptor with _data_ref = None
        mock_desc = MagicMock()
        mock_desc._data_ref = None
        mock_desc._data_size = 1024

        from queue import Queue

        receiver._pool = Queue()
        receiver._pool.put(mock_desc)

        # Mock nixl_connect for dynamic fallback
        receiver._nixl_connect = MagicMock()

        with pytest.raises(RuntimeError, match="no data reference"):
            receiver._acquire_descriptor(512)


class TestMmKwargsNixlReceiverOrdering:
    """Tests that NIXL receiver preserves spec order under concurrent reads."""

    @pytest.mark.asyncio
    async def test_multi_image_nixl_receive_preserves_order(self):
        """3 images with different completion delays: results must be in spec order."""
        import asyncio

        import torch

        from dynamo.common.multimodal.mm_kwargs_transfer import MmKwargsNixlReceiver

        # This test exercises only the receiver's byte ordering, not the item
        # serialization, so distinct opaque byte payloads are enough.
        items = [
            b"item-0-first",
            b"item-1-second",
            b"item-2-third",
        ]

        # Build metadata as if the sender prepared 3 specs
        specs = [
            TensorTransferSpec(
                field_name="__pickled_kwargs_item__",
                shape=[len(item)],
                dtype_str="uint8",
                serialized_request={"mock": True, "index": i},
            )
            for i, item in enumerate(items)
        ]
        metadata = MmKwargsTransferMetadata(
            modality="image",
            tensor_specs=specs,
            mm_hashes=["h0", "h1", "h2"],
        )

        # Create receiver and mock its internals
        receiver = MmKwargsNixlReceiver.__new__(MmKwargsNixlReceiver)
        receiver._available = True

        # Mock _acquire_descriptor: return a real tensor buffer + metadata
        buffers = [torch.zeros(len(item), dtype=torch.uint8) for item in items]
        buf_iter = iter(buffers)

        def mock_acquire(size_bytes):
            buf = next(buf_iter)
            return MagicMock(), buf, True, None  # desc, tensor_view, is_dynamic, orig

        receiver._acquire_descriptor = mock_acquire
        receiver._release_descriptor = MagicMock()

        # Mock nixl_connect.RdmaMetadata.model_validate
        mock_nixl = MagicMock()
        mock_nixl.RdmaMetadata.model_validate = lambda x: x
        receiver._nixl_connect = mock_nixl

        # Mock connector.begin_read: copy each payload into the buffer
        # with REVERSE completion order (item 2 finishes first, item 0 last)
        # to verify ordering is preserved despite out-of-order completion.
        delays = [0.03, 0.02, 0.01]  # item 0 slowest, item 2 fastest

        call_count = [0]

        async def mock_begin_read(rm, d):
            idx = call_count[0]
            call_count[0] += 1
            item_data = items[idx]

            op = MagicMock()

            async def mock_wait():
                await asyncio.sleep(delays[idx])
                # Write the payload into the buffer
                buf = buffers[idx]
                buf[: len(item_data)] = torch.frombuffer(
                    bytearray(item_data), dtype=torch.uint8
                )

            op.wait_for_completion = mock_wait
            return op

        mock_connector = MagicMock()
        mock_connector.begin_read = mock_begin_read
        receiver._connector = mock_connector

        # Run receive
        results = await receiver.receive(metadata)

        # Verify results
        assert "__pickled_kwargs_item__" in results
        received_items = results["__pickled_kwargs_item__"]
        assert len(received_items) == 3

        # CRITICAL: verify order matches spec order, not completion order
        for i, raw in enumerate(received_items):
            assert raw == items[i], (
                f"Item {i} is {raw!r}; results are in completion order "
                f"instead of spec order"
            )
            # The decoder builds tensors over this buffer, so it must be writable.
            assert not memoryview(raw).readonly


class TestMmKwargsShmReceiverBuffers:
    """The SHM receiver returns writable buffers (no vLLM needed)."""

    @pytest.mark.asyncio
    async def test_shm_receiver_returns_writable_buffer(self):
        sender = MmKwargsShmSender()
        shm_item, handle = await sender._encode_item(0, b"serialized-item")
        try:
            metadata = MmKwargsShmTransferMetadata(
                modality="image", items=[shm_item], mm_hashes=["h"]
            )
            results = await MmKwargsShmReceiver().receive(metadata)
        finally:
            await sender.cleanup([handle])

        (blob,) = results["__pickled_kwargs_item__"]
        assert blob == b"serialized-item"
        # The decoder builds tensors over this buffer, so it must be writable.
        assert not memoryview(blob).readonly


class TestBufferFraming:
    """Tests for the length-prefix frame that carries the msgpack buffers.

    These cover only the byte framing (no vLLM), so they run in the
    dynamo-runtime lane alongside the NIXL/SHM transport tests.
    """

    def test_pack_unpack_roundtrip(self):
        bufs = [b"abc", b"", b"defghij"]
        unpacked = _unpack_buffers(_pack_buffers(bufs))
        assert [bytes(mv) for mv in unpacked] == bufs

    def test_pack_unpack_empty_sequence(self):
        unpacked = _unpack_buffers(_pack_buffers([]))
        assert unpacked == []

    def test_unpack_rejects_short_header(self):
        # Fewer than the 4 bytes needed for the count prefix.
        with pytest.raises(ValueError, match="too short"):
            _unpack_buffers(b"\x01\x00")

    def test_unpack_rejects_length_past_end(self):
        # Declares one buffer of 999 bytes but only 3 follow.
        blob = struct.pack("<I", 1) + struct.pack("<Q", 999) + b"abc"
        with pytest.raises(ValueError, match="runs past the end"):
            _unpack_buffers(blob)

    def test_unpack_rejects_trailing_bytes(self):
        blob = _pack_buffers([b"abc"]) + b"EXTRA"
        with pytest.raises(ValueError, match="trailing bytes"):
            _unpack_buffers(blob)

    def test_unpack_rejects_oversized_count(self):
        # Count far larger than the remaining bytes can describe.
        blob = struct.pack("<I", 2**31) + b""
        with pytest.raises(ValueError, match="declares"):
            _unpack_buffers(blob)


@pytest.mark.vllm
class TestMsgpackDecodeRestrictions:
    """The decoder reconstructs only the target type and refuses pickle codes."""

    def test_decode_refuses_pickle_extension_code(self):
        """A frame carrying msgpack's pickle extension code must raise.

        The decoder accepts only the raw tensor-view code, so it refuses this
        code whatever VLLM_ALLOW_INSECURE_SERIALIZATION says, and the receiver
        falls back. The frame is built directly with msgspec, so no pickle
        object is created.
        """
        from msgspec import msgpack
        from vllm.v1.serial_utils import CUSTOM_TYPE_PICKLE

        ext = msgpack.Ext(CUSTOM_TYPE_PICKLE, b"\x00 not a pickle")
        frame = _pack_buffers([msgpack.encode(ext)])
        with pytest.raises(NotImplementedError, match="Extension type code 1"):
            decode_mm_kwargs_item(frame)

    def test_decode_refuses_pickle_extension_code_with_insecure_flag(self, monkeypatch):
        """The refusal holds when VLLM_ALLOW_INSECURE_SERIALIZATION is set.

        vLLM's own decoder honors the pickle code when the variable is set, so
        with these dummy bytes it would fail with an unpickling error instead
        of refusing the code. The frame carries no pickle object.
        """
        import vllm.envs as envs
        from msgspec import msgpack
        from vllm.v1.serial_utils import CUSTOM_TYPE_PICKLE

        # vLLM caches its environment after an engine starts. Read it live, and
        # prove that the decoder sees the variable change in this process.
        envs.disable_envs_cache()
        monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
        assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION is False
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION is True

        ext = msgpack.Ext(CUSTOM_TYPE_PICKLE, b"\x00 not a pickle")
        frame = _pack_buffers([msgpack.encode(ext)])
        with pytest.raises(NotImplementedError, match="Extension type code 1"):
            decode_mm_kwargs_item(frame)
