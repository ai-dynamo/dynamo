// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral publication boundary for DC Relay state.

mod cbi1;
mod codec;
mod hub;
mod source;
mod stream;

pub use codec::{PublicationFrame, PublicationFrameKind};
pub use source::{PublicationError, PublicationErrorKind, RelayPublicationSource};
pub use stream::PoolPublicationStream;

pub(super) use cbi1::MAX_BUCKET_COUNT;
pub(super) use hub::{
    PublicationHub, PublicationHubConfig, PublicationHubError, PublicationHubSubscription,
    TerminalFailure, publication_lease,
};
pub(super) use source::{
    DEFAULT_ACTIVE_POOL_STREAMS, DEFAULT_SNAPSHOT_ENCODING_CONCURRENCY,
    DEFAULT_SNAPSHOT_PROGRESS_TIMEOUT, RegistryPublicationSource,
};
