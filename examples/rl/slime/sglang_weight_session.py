# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opens an SGLang weight-update session around each Slime NCCL weight update.

SGLang 0.5.21 and later reject ``update_weights_from_distributed`` with HTTP
400 unless the call runs between ``begin_weight_update`` and
``end_weight_update``. The pinned Slime revision does not open that session.
``launch-slime.sh`` passes ``install`` to Slime as
``--custom-megatron-init-path`` so every external engine gets one session per
policy update, around all weight buckets.
"""

from __future__ import annotations

import functools
import logging
from argparse import Namespace
from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

_WRAPPED_ATTR = "_dynamo_weight_session_wrapped"
# Seconds to wait for one engine to answer begin_weight_update or
# end_weight_update. Both run while the engines are paused.
SESSION_TIMEOUT_S = 120.0


def engine_base_url(addr: str) -> str:
    """Return ``http://host:port`` for a Slime ``host:port`` or URL address."""
    # Same rules as slime.backends.sglang_utils.external.normalize_external_engine_addr.
    if "://" not in addr:
        addr = f"http://{addr}"
    addr = addr.rstrip("/")
    parsed = urlsplit(addr)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Invalid external SGLang engine address {addr!r}. "
            "Use host:port or http://host:port (IPv6 must be bracketed)."
        )
    return addr


def end_weight_update(
    engine_urls: Sequence[str], timeout: float = SESSION_TIMEOUT_S
) -> None:
    """Close the session on every engine; raise after trying all of them."""
    errors = []
    for url in engine_urls:
        try:
            response = requests.post(
                f"{url}/end_weight_update", json={}, timeout=timeout
            )
        except requests.RequestException as error:
            errors.append(f"{url}: {error}")
            continue
        if not response.ok:
            errors.append(f"{url}: HTTP {response.status_code}: {response.text}")
    if errors:
        raise RuntimeError("end_weight_update failed: " + "; ".join(errors))


def begin_weight_update(
    engine_urls: Sequence[str], timeout: float = SESSION_TIMEOUT_S
) -> list[str]:
    """Open a session on every engine and return the engines that opened one.

    An engine that answers 404 runs an SGLang release without sessions and is
    left out of the result. On any other failure, close the sessions that
    already opened, so the next update can begin again, and raise.
    """
    opened: list[str] = []
    for url in engine_urls:
        try:
            response = requests.post(
                f"{url}/begin_weight_update", json={}, timeout=timeout
            )
            if response.status_code == 404:
                logger.info("%s has no weight-update session API; skipping", url)
                continue
            if not response.ok:
                raise RuntimeError(
                    f"begin_weight_update failed on {url}: "
                    f"HTTP {response.status_code}: {response.text}"
                )
        except Exception:
            try:
                end_weight_update(opened, timeout=timeout)
            except RuntimeError:
                logger.exception("Could not close weight-update sessions")
            raise
        opened.append(url)
    return opened


def wrap_send_weights(
    send_weights: Callable,
    engine_urls: Sequence[str],
    *,
    is_session_owner: Callable[[], bool],
    share_error: Callable[[str | None], str | None],
    timeout: float = SESSION_TIMEOUT_S,
) -> Callable:
    """Wrap ``UpdateWeightFromDistributed._send_weights`` in one session.

    The owner rank opens the session before any rank sends a bucket and closes
    it after its own send returns. ``_send_weights`` ends with a barrier, so the
    other ranks have finished sending by then.

    ``share_error`` is a collective that every rank calls with its own error
    text (``None`` on success) and that returns the owner's. All ranks call it
    after begin and again after end, so an owner failure makes every rank
    raise instead of leaving the others waiting in the next collective.
    """

    @functools.wraps(send_weights)
    def wrapped(self, *args, **kwargs):
        """Send all weight buckets and share session errors across ranks."""
        owner = is_session_owner()
        opened: list[str] = []
        error = None
        if owner:
            try:
                opened = begin_weight_update(engine_urls, timeout=timeout)
            except Exception as exc:
                error = str(exc)
        error = share_error(error)
        if error is not None:
            raise RuntimeError(f"Weight-update session did not open: {error}")

        try:
            result = send_weights(self, *args, **kwargs)
        except Exception:
            # Ranks are already out of step here, as in Slime when a send
            # fails, so close the session without another collective.
            if owner and opened:
                try:
                    end_weight_update(opened, timeout=timeout)
                except RuntimeError:
                    logger.exception("Could not close weight-update sessions")
            raise

        error = None
        if owner and opened:
            try:
                end_weight_update(opened, timeout=timeout)
            except Exception as exc:
                error = str(exc)
        error = share_error(error)
        if error is not None:
            raise RuntimeError(f"Weight-update session did not close: {error}")
        return result

    setattr(wrapped, _WRAPPED_ATTR, True)
    return wrapped


def broadcast_from_rank0(value: str | None, group) -> str | None:
    """Send global rank 0's value to every rank in ``group``."""
    # Torch is optional when importing this module for GPU-free tests.
    import torch.distributed as dist

    box = [value]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


def install(args: Namespace) -> None:
    """Slime ``--custom-megatron-init-path`` hook for external NCCL updates."""
    if not getattr(args, "rollout_external", False):
        return
    # Disk, delta, and colocated tensor updates do not call
    # update_weights_from_distributed, so they need no session.
    if getattr(args, "update_weight_transport", "nccl") != "nccl":
        return
    if getattr(args, "update_weight_mode", "full") != "full":
        return
    if getattr(args, "colocate", False):
        return

    # Load trainer-only dependencies here so GPU-free tests can import the module.
    import torch.distributed as dist
    from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
        UpdateWeightFromDistributed,
    )
    from slime.utils.distributed_utils import get_gloo_group

    if getattr(UpdateWeightFromDistributed._send_weights, _WRAPPED_ATTR, False):
        return

    engine_urls = [engine_base_url(a) for a in args.rollout_external_engine_addrs]
    # Global rank 0 is the rank that pauses and resumes the engines in
    # UpdateWeightFromDistributed.update_weights.
    UpdateWeightFromDistributed._send_weights = wrap_send_weights(
        UpdateWeightFromDistributed._send_weights,
        engine_urls,
        is_session_owner=lambda: dist.get_rank() == 0,
        share_error=lambda error: broadcast_from_rank0(error, get_gloo_group()),
    )
