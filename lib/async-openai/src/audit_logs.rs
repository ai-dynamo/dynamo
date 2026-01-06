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

use crate::{Client, config::Config, error::OpenAIError, types::ListAuditLogsResponse};

/// Logs of user actions and configuration changes within this organization.
/// To log events, you must activate logging in the [Organization Settings](https://platform.openai.com/settings/organization/general).
/// Once activated, for security reasons, logging cannot be deactivated.
pub struct AuditLogs<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> AuditLogs<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// List user actions and configuration changes within this organization.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn get<Q>(&self, query: &Q) -> Result<ListAuditLogsResponse, OpenAIError>
    where
        Q: Serialize + ?Sized,
    {
        self.client
            .get_with_query("/organization/audit_logs", &query)
            .await
    }
}
