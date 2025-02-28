// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

pub use crate::kv_router::protocols::ForwardPassMetrics;

use anyhow::Result;
use triton_distributed_runtime::pipeline::network::{
    ingress::push_endpoint::PushEndpoint,
    PushWorkHandler,
};

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing as log;

pub struct KvMetricsAggregator {
    pub service_name: String,

    pub nats: nats::Client,
    pub service_handler: Arc<dyn PushWorkHandler>,
    pub metrics_rx: watch::Receiver<Arc<ForwardPassMetrics>>,
    pub cancellation_token: CancellationToken,
}

/// version of crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

impl KvRoutedIngress {
    pub fn builder() -> KvRoutedIngressBuilder {
        KvRoutedIngressBuilder::default()
    }

    pub async fn start(self) -> Result<()> {
        let worker_id = self.worker_id;

        log::trace!(
            worker_id,
            "Starting nats service: {}:{}",
            self.service_name,
            VERSION
        );

        let mut metrics_rx = self.metrics_rx;
        let worker_id_clone = worker_id.clone();

        let service = self
            .nats
            .client()
            .service_builder()
            .description("A handy min max service")
            .stats_handler(move |name, stats| {
                log::debug!(
                    worker_id = worker_id_clone.as_str(),
                    "[IN worker?] Stats for service {}: {:?}",
                    name,
                    stats
                );
                let metrics = metrics_rx.borrow_and_update().clone();
                serde_json::to_value(&*metrics).unwrap()
            })
            .start(self.service_name.as_str(), VERSION)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start service: {e}"))?;

        let group = service.group(self.service_name.as_str());

        log::trace!(worker_id, "Starting endpoint: {}", worker_id);

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(worker_id.clone())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        let push_endpoint = PushEndpoint::builder()
            .service_handler(self.service_handler)
            .cancellation_token(self.cancellation_token)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        push_endpoint.start(service_endpoint).await
    }
}
