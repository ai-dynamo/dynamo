// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;

use super::PoolDiscoveryError;

/// Expected identity of a pool's Relay, for example a SPIFFE ID.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelayIdentity(pub String);

/// An identity proven by a [`RelayAuthenticator`]. Only authenticators in
/// this module tree can create one, so writes cannot use a claimed identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VerifiedRelayIdentity(pub(super) RelayIdentity);

/// Raw credentials the transport took from a Relay's connection.
/// No `Debug`: it can hold a secret.
#[derive(Clone)]
#[non_exhaustive]
pub enum PeerCredentials {
    /// From a client certificate that TLS already checked.
    Spiffe(String),
    BearerToken(String),
}

#[async_trait]
pub trait RelayAuthenticator: Send + Sync {
    async fn authenticate(
        &self,
        credentials: &PeerCredentials,
    ) -> Result<VerifiedRelayIdentity, PoolDiscoveryError>;
}
