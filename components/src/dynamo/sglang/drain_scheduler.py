# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confirm scheduler idle after a disaggregated worker closes admission."""

import asyncio
import logging
from collections.abc import Callable, Mapping
from functools import partial

import zmq
import zmq.asyncio

from dynamo.common.forward_pass_metrics import decode

logger = logging.getLogger(__name__)


def scheduler_idle_probe(
    handler, supports_idle_heartbeat: Callable[[], bool]
) -> "SchedulerIdleProbe | None":
    args = handler.config.server_args
    if not getattr(args, "enable_forward_pass_metrics", False):
        return None
    if (
        args.disaggregation_transfer_backend != "nixl"
        or args.nnodes != 1
        or args.tp_size != 1
        or args.dp_size != 1
        or args.pp_size != 1
        or getattr(args, "attn_cp_size", 1) != 1
        or args.tokenizer_worker_num != 1
        or not supports_idle_heartbeat()
    ):
        return None

    address = getattr(args, "forward_pass_metrics_ipc_name", None)
    if not address or not address.startswith("ipc://"):
        return None
    return SchedulerIdleProbe(
        handler.engine.tokenizer_manager,
        f"{address}.0",
        str(handler.generate_endpoint.connection_id()),
    )


def configure_prefill_shutdown(
    shutdown, handler, *, integrated: bool, supports_idle_heartbeat: Callable[[], bool]
) -> bool:
    """Install the terminal KV check and indicate that TCP writes must be tracked."""
    if (
        shutdown is None
        or not integrated
        or handler.config.dynamo_args.response_plane != "tcp"
    ):
        return False
    probe = scheduler_idle_probe(handler, supports_idle_heartbeat)
    if probe is None:
        return False
    shutdown.wait_for_kv_transfers = partial(
        probe.wait_for_terminal_idle, handler.generate_endpoint
    )
    return True


class SchedulerIdleProbe:
    def __init__(self, tokenizer_manager, address: str, worker_id: str):
        self._tokenizer_manager = tokenizer_manager
        self._address = address
        self._worker_id = worker_id
        self._scheduler_read: asyncio.Task | None = None

    async def __call__(self) -> bool:
        try:
            async with asyncio.timeout(8):
                # SGLang's RPC has no request ID. Keep a timed-out read alive
                # until its reply is consumed so a later probe gets its own ACK.
                if self._scheduler_read is not None and not self._scheduler_read.done():
                    await asyncio.shield(self._scheduler_read)
                read = asyncio.create_task(self._tokenizer_manager.get_internal_state())
                read.add_done_callback(self._observe_read)
                self._scheduler_read = read
                if len(await asyncio.shield(read)) != 1:
                    return False

                # A fresh subscriber has no backlog from before the scheduler ACK.
                with zmq.asyncio.Context.instance().socket(zmq.SUB) as socket:
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.subscribe(b"")
                    socket.connect(self._address)
                    last_idle = None
                    while True:
                        parts = await socket.recv_multipart()
                        if len(parts) != 3 or len(parts[1]) != 8:
                            last_idle = None
                            continue
                        _, seq, payload = parts
                        frame = decode(payload)
                        sequence = int.from_bytes(seq, "big")
                        if (
                            frame is None
                            or frame.worker_id != self._worker_id
                            or frame.dp_rank != 0
                            or frame.counter_id != sequence
                            or frame.wall_time != 0
                            or not frame.is_idle()
                        ):
                            last_idle = None
                            continue
                        # An already constructed heartbeat can race the ACK.
                        if last_idle is not None and sequence == last_idle + 1:
                            return True
                        last_idle = sequence
        except (TimeoutError, zmq.ZMQError):
            logger.debug(
                "SGLang scheduler idle confirmation unavailable", exc_info=True
            )
            return False

    async def wait_for_terminal_idle(self, endpoint) -> None:
        """Confirm local and scheduler idle after terminal admission has closed."""

        def local_requests_finished() -> bool:
            states = getattr(self._tokenizer_manager, "rid_to_state", None)
            return (
                isinstance(states, Mapping)
                and not states
                and endpoint.inflight_requests() == 0
            )

        try:
            while True:
                if (
                    local_requests_finished()
                    and await self()
                    and local_requests_finished()
                ):
                    return
                await asyncio.sleep(0.1)
        finally:
            # A timed-out read is normally retained until the next probe. After
            # terminal drain this probe will never reuse its scheduler channel.
            if self._scheduler_read is not None and not self._scheduler_read.done():
                self._scheduler_read.cancel()

    @staticmethod
    def _observe_read(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()
