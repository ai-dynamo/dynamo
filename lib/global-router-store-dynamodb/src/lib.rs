// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DynamoDB accessors for affinity. Regional and MREC global tables use exactly
//! the same implementation. Conditional writes are regional, not a global lock:
//! on an MREC table, pins are soft, and last writer wins between regions. Two
//! regions that bind one session before either write replicates both succeed,
//! and replication keeps the later write. The tracker rewrites a pin only when
//! it changes or needs a refresh, so the session then settles on one pool.

mod item;

use async_trait::async_trait;
use aws_sdk_dynamodb::{Client, error::ProvideErrorMetadata, types::AttributeValue};
use dynamo_global_router::affinity::{
    AffinityBinding, AffinityError, AffinityKey, AffinityStore, Expected, WriteOutcome,
};

/// Reuses one SDK client, whose region, endpoint, credentials, timeout and retry
/// policy are configured by the host. Table creation and TTL are infrastructure.
pub struct DynamoDbAffinityStore {
    client: Client,
    table: String,
    consistent_reads: bool,
}

impl DynamoDbAffinityStore {
    pub fn new(
        client: Client,
        table: String,
        consistent_reads: bool,
    ) -> Result<Self, AffinityError> {
        if table.trim().is_empty() {
            return Err(AffinityError::Configuration(
                "DynamoDB table name is empty".into(),
            ));
        }
        Ok(Self {
            client,
            table,
            consistent_reads,
        })
    }
}

#[async_trait]
impl AffinityStore for DynamoDbAffinityStore {
    async fn read(&self, key: &AffinityKey) -> Result<Option<AffinityBinding>, AffinityError> {
        let output = self
            .client
            .get_item()
            .table_name(&self.table)
            .set_key(Some(item::key(key)))
            .consistent_read(self.consistent_reads)
            .send()
            .await
            .map_err(|error| {
                let code = error.as_service_error().and_then(|error| error.code());
                AffinityError::Backend(format!(
                    "GetItem: {} ({error})",
                    code.unwrap_or("transport")
                ))
            })?;
        output.item.map(item::decode).transpose()
    }

    async fn compare_exchange(
        &self,
        key: &AffinityKey,
        expected: &Expected,
        binding: &AffinityBinding,
    ) -> Result<WriteOutcome, AffinityError> {
        let mut put = self
            .client
            .put_item()
            .table_name(&self.table)
            .set_item(Some(item::encode(key, binding)));
        put = match expected {
            Expected::Revision(revision) => put
                .condition_expression("#revision = :expected")
                .expression_attribute_names("#revision", "revision")
                .expression_attribute_values(":expected", AttributeValue::S(revision.clone())),
            Expected::Absent => put
                .condition_expression("attribute_not_exists(#pk)")
                .expression_attribute_names("#pk", "itemHashKey"),
            // Repairing a row no reader can decode: its revision cannot be
            // read, and a conditional write would fail forever, leaving the
            // row to outlive even DynamoDB's TTL sweep when `expiresAt` is
            // what is corrupt. An unconditional PutItem restores a row that
            // every version in the deployment can read.
            Expected::Unreadable => put,
        };
        match put.send().await {
            Ok(_) => Ok(WriteOutcome::Applied),
            Err(error)
                if error
                    .as_service_error()
                    .is_some_and(|error| error.is_conditional_check_failed_exception()) =>
            {
                Ok(WriteOutcome::Conflict)
            }
            Err(error) => {
                let code = error.as_service_error().and_then(|error| error.code());
                Err(AffinityError::Backend(format!(
                    "PutItem: {} ({error})",
                    code.unwrap_or("transport")
                )))
            }
        }
    }
}
