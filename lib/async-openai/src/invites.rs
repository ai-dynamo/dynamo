// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use serde::Serialize;

use crate::{
    Client,
    config::Config,
    error::OpenAIError,
    types::{Invite, InviteDeleteResponse, InviteListResponse, InviteRequest},
};

/// Invite and manage invitations for an organization. Invited users are automatically added to the Default project.
pub struct Invites<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Invites<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Returns a list of invites in the organization.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn list<Q>(&self, query: &Q) -> Result<InviteListResponse, OpenAIError>
    where
        Q: Serialize + ?Sized,
    {
        self.client
            .get_with_query("/organization/invites", &query)
            .await
    }

    /// Retrieves an invite.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn retrieve(&self, invite_id: &str) -> Result<Invite, OpenAIError> {
        self.client
            .get(format!("/organization/invites/{invite_id}").as_str())
            .await
    }

    /// Create an invite for a user to the organization. The invite must be accepted by the user before they have access to the organization.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn create(&self, request: InviteRequest) -> Result<Invite, OpenAIError> {
        self.client.post("/organization/invites", request).await
    }

    /// Delete an invite. If the invite has already been accepted, it cannot be deleted.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn delete(&self, invite_id: &str) -> Result<InviteDeleteResponse, OpenAIError> {
        self.client
            .delete(format!("/organization/invites/{invite_id}").as_str())
            .await
    }
}
