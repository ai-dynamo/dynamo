// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::{PoolDirectory, PoolRecord};
use crate::global_view::PoolKey;

/// Decides whether an authenticated Relay may stream state for a pool. The
/// rule is the same in every mode; only the source of `relay_identity`
/// differs. A record without one admits any authenticated Relay.
///
/// Uses `PoolDirectory::get`, so it reads the watch cache and never lists.
/// The runtime keys the stream by its `PlaneLease`, not by the claimed key.
// Signature only; the body comes with the first mode's implementation.
#[allow(unused_variables)]
pub async fn admit(
    pools: &PoolDirectory,
    key: &PoolKey,
    caller: &VerifiedRelayIdentity,
) -> Result<PoolRecord, PoolDiscoveryError> {
    todo!()
}
