# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter: EventEmitter protocol -> dynamo._core.SweeperEventPublisher PyO3
binding (DEP #15073). Isolates sweeper_event_plane.py's transport-agnostic
logic from the Rust binding's exact constructor/method names -- this is the
only file that needs to change if those names change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamo._core import Endpoint


class DynamoEventPlaneEmitter:
    """Publishes onto Dynamo's event plane via the (new) SweeperEventPublisher
    PyO3 class. One instance manages all three subjects for a single Sweeper
    run -- the Rust side is expected to lazily create one underlying
    EventPublisher per distinct subject it's asked to publish to, matching
    the DEP's "one subject per event type" design without the Python side
    needing to know that detail.
    """

    def __init__(self, endpoint: "Endpoint") -> None:
        from dynamo._core import SweeperEventPublisher as _RustPublisher

        # No run_uid here: subject strings already arrive fully built
        # (sweeper.<run_uid>.<event_name>) from sweeper_event_plane.py, so
        # the Rust side never needs the run UID itself.
        self._inner = _RustPublisher(endpoint)

    def publish(self, subject: str, payload: bytes) -> None:
        # Contract expected of the Rust binding: a blocking call (the
        # background thread in SweeperEventPublisher is already off the
        # search's own thread, so blocking here is fine and keeps the Rust
        # side simple -- no extra internal queue needed there).
        self._inner.publish_subject(subject, payload)

    def close(self) -> None:
        self._inner.close()
