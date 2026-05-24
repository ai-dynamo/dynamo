// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`RegistrationHandle`] — owned handle wrapping an active server stream.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tonic::{Status, Streaming};

use crate::proto::{Accepted, Event};

/// Owned handle representing an active `Register` stream.
///
/// The handle exposes the `Accepted` metadata from the first stream event and
/// implements [`Stream`] to surface subsequent `Event`s (e.g. `Heartbeat`).
///
/// Dropping the handle closes the stream, which the server interprets as an
/// ungraceful detach and triggers cleanup on its side.
pub struct RegistrationHandle {
    accepted: Accepted,
    stream: Streaming<Event>,
}

impl RegistrationHandle {
    pub(crate) fn new(accepted: Accepted, stream: Streaming<Event>) -> Self {
        Self { accepted, stream }
    }

    /// Server-assigned opaque registration ID.
    pub fn registration_id(&self) -> &str {
        &self.accepted.registration_id
    }

    /// Number of GPU slots reserved by this registration (typically equal to
    /// the `tp_size` in the [`RegistrationKey`](crate::proto::RegistrationKey)).
    pub fn reserved_slots(&self) -> u32 {
        self.accepted.reserved_slots
    }

    /// Full `Accepted` payload from the server.
    pub fn accepted(&self) -> &Accepted {
        &self.accepted
    }
}

/// Expose subsequent stream events via [`Stream`] so callers can write
/// `while let Some(ev) = handle.next().await { ... }`.
impl Stream for RegistrationHandle {
    type Item = Result<Event, Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}
