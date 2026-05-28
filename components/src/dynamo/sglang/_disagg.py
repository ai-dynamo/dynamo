# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregation helpers shared by the legacy (`init_llm.py`,
`register.py`) and unified (`llm_engine.py`) entry points.

Two helpers:

* :func:`compute_bootstrap_address` — resolve the `(host, port)` triple a
  prefill worker advertises to decode peers from a live `sgl.Engine`.
  Returns `(None, None)` on any failure so legacy callers can keep their
  graceful-degradation behaviour; the unified path treats `None` as a
  fatal configuration error and raises.

* :func:`warmup_prefill_engine` — drive one request through the disagg
  path with SGLang's `FAKE_BOOTSTRAP_HOST` so the first real request
  doesn't pay the JIT/CUDA-graph compile cost.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import sglang as sgl

# Matches the prior in-tree value. Long enough for the slowest cold-start
# we've seen (TP8 70B with FlashInfer JIT).
_PREFILL_WARMUP_TIMEOUT_S = 1800.0


def compute_bootstrap_address(
    engine: sgl.Engine,
) -> tuple[Optional[str], Optional[int]]:
    """Return `(host, port)` for this prefill worker's bootstrap address.

    Returns `(None, None)` when the engine doesn't advertise a
    `disaggregation_bootstrap_port` or when address resolution raises.
    Caller decides whether `None` is fatal — legacy `register.py` treats
    it as soft-skip; the unified path treats it as a fatal misconfig.
    """
    # Deferred to function body so pre-commit test collection can import
    # `_disagg` without sglang installed.
    from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

    try:
        inner_tm = engine.tokenizer_manager
        bootstrap_port = getattr(
            inner_tm.server_args, "disaggregation_bootstrap_port", None
        )
        if bootstrap_port is None:
            return None, None

        if inner_tm.server_args.dist_init_addr:
            dist_init = NetworkAddress.parse(inner_tm.server_args.dist_init_addr)
            resolved = dist_init.resolved()
            bootstrap_host = (
                NetworkAddress(resolved.host, bootstrap_port)
                .to_host_port_str()
                .rsplit(":", 1)[0]
            )
            logging.info(
                "Resolved bootstrap host '%s' -> '%s' (%s)",
                dist_init.host,
                resolved.host,
                "IPv6" if resolved.is_ipv6 else "IPv4",
            )
        else:
            # SGLANG_HOST_IP overrides auto-detection (use bracketed [addr]
            # for IPv6); falls back to a v4-then-v6 probe.
            local_ip = get_local_ip_auto()
            local_addr = NetworkAddress(local_ip, bootstrap_port)
            bootstrap_host = local_addr.to_host_port_str().rsplit(":", 1)[0]
            logging.info(
                "Using auto-detected local IP: %s (%s)",
                local_ip,
                "IPv6" if local_addr.is_ipv6 else "IPv4",
            )
        return bootstrap_host, bootstrap_port
    except Exception as e:
        logging.warning("Failed to compute bootstrap address: %s", e)
        return None, None


async def warmup_prefill_engine(engine: sgl.Engine, bootstrap_port: int) -> None:
    """Drive one request through the prefill disagg path with SGLang's
    `FAKE_BOOTSTRAP_HOST` so JIT/CUDA-graph compile happens before the
    first real request. Raises on timeout/failure — an unwarmed prefill
    silently drops production requests."""
    from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 8,
        "ignore_eos": True,
    }

    async def _do_warmup() -> None:
        results = await engine.async_generate(
            input_ids=[0, 1, 2, 3],
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_port=bootstrap_port,
            bootstrap_room=999999,
        )
        async for _ in results:
            pass

    logging.info("SGLang prefill warmup starting...")
    await asyncio.wait_for(_do_warmup(), timeout=_PREFILL_WARMUP_TIMEOUT_S)
    logging.info("SGLang prefill warmup complete")
