// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Queue abstraction for dispatching decode-created remote-prefill requests.

use anyhow::Result;
use futures::{FutureExt, future::BoxFuture};
use kvbm_disagg_protocol::RemotePrefillRequest;
use kvbm_hub::ConditionalDisaggClient;
use std::sync::Arc;

pub trait RemotePrefillQueue: Send + Sync {
    fn enqueue(&self, request: RemotePrefillRequest) -> BoxFuture<'static, Result<()>>;
}

/// Hub-backed remote-prefill queue for decode participants.
pub struct HubRemotePrefillQueue {
    client: Arc<ConditionalDisaggClient>,
}

impl HubRemotePrefillQueue {
    pub fn new(client: Arc<ConditionalDisaggClient>) -> Arc<Self> {
        Arc::new(Self { client })
    }

    pub fn client(&self) -> &Arc<ConditionalDisaggClient> {
        &self.client
    }
}

impl RemotePrefillQueue for HubRemotePrefillQueue {
    fn enqueue(&self, request: RemotePrefillRequest) -> BoxFuture<'static, Result<()>> {
        let client = self.client.clone();
        async move { client.push_prefill_request(&request).await }.boxed()
    }
}
