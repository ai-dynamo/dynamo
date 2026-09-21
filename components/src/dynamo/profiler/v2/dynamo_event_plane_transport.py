# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter: EventEmitter protocol -> the new dynamo._core.SweeperEventPublisher
PyO3 binding proposed in DEP #15073.

This is the ONLY file that needs to change if the Rust binding's exact
constructor/method names end up differing from the draft in
`rust/sweeper_events.rs` in this same delivery -- everything in
sweeper_event_plane.py is written against the transport-agnostic
EventEmitter protocol and does not need to change.

Not yet runnable: dynamo._core.SweeperEventPublisher does not exist until
the Rust binding (see rust/sweeper_events.rs and RUST_BINDING_NOTES.md) is
added and built with `maturin develop` in a real checkout. The import
below is deliberately lazy (inside __init__) so importing this module
elsewhere does not fail before that binding exists.
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
        # Local import: see module docstring -- this binding does not exist
        # in dynamo._core yet.
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
