# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort RLIMIT_NOFILE raising for long-lived Dynamo processes.

Processes that hold many sockets (a frontend accepting connections, a router
subscribing to every worker's KV event publisher, a mocker hosting many
workers) can exhaust the default soft limit (1024 on most distros) and fail
with EMFILE. The hard limit stays the operator's control: the soft limit is
never raised past it.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def raise_fd_limit(target: Optional[int] = None) -> None:
    """Raise the soft RLIMIT_NOFILE toward `target`, bounded by the hard limit.

    `target=None` raises the soft limit to the hard limit. A non-positive target
    disables the raise. No-op when the soft limit is already sufficient, when no
    target is given and the hard limit is unbounded, when the Unix-only
    `resource` module is unavailable (e.g. Windows), or when the raise is denied.
    """
    if target is not None and target <= 0:
        return
    try:
        import resource  # Unix-only; imported lazily so Windows stays a no-op.
    except ImportError:
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            return
        if target is None:
            if hard == resource.RLIM_INFINITY:
                return
            new_soft = hard
        else:
            new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if new_soft <= soft:
            return
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        # Best-effort hardening must never block startup (e.g. setrlimit denied
        # in a restricted environment, or an out-of-range target).
        logger.debug("Could not raise RLIMIT_NOFILE; continuing", exc_info=True)
        return
    logger.info(
        "Raised RLIMIT_NOFILE soft limit %s -> %s (hard=%s)", soft, new_soft, hard
    )
