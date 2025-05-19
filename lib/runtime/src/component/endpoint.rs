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

use derive_getters::Dissolve;

use super::*;

pub use async_nats::service::endpoint::Stats as EndpointStats;

#[derive(Educe, Builder, Dissolve)]
#[educe(Debug)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct EndpointConfig {
    #[builder(private)]
    endpoint: Endpoint,

    // todo: move lease to component/service
    /// Lease
    #[educe(Debug(ignore))]
    #[builder(default)]
    lease: Option<Lease>,

    /// Endpoint handler
    #[educe(Debug(ignore))]
    handler: Arc<dyn PushWorkHandler>,

    /// Stats handler
    #[educe(Debug(ignore))]
    #[builder(default, private)]
    _stats_handler: Option<EndpointStatsHandler>,
}

impl EndpointConfigBuilder {
    pub(crate) fn from_endpoint(endpoint: Endpoint) -> Self {
        Self::default().endpoint(endpoint)
    }

    pub fn stats_handler<F>(self, handler: F) -> Self
    where
        F: FnMut(EndpointStats) -> serde_json::Value + Send + Sync + 'static,
    {
        self._stats_handler(Some(Box::new(handler)))
    }

    pub async fn start(self) -> Result<()> {
        let (endpoint, lease, handler, stats_handler) = self.build_internal()?.dissolve();
        let lease = lease.or(endpoint.drt().primary_lease());
        let lease_id = lease.as_ref().map(|l| l.id()).unwrap_or(0);

        tracing::debug!("Starting endpoint: {}", endpoint.etcd_path(lease_id));

        let service_name = endpoint.component.service_name();

        // acquire the registry lock
        let registry = endpoint.drt().component_registry.inner.lock().await;

        // get the group
        let group = registry
            .services
            .get(&service_name)
            .map(|service| service.group(endpoint.component.service_name()))
            .ok_or(error!("Service not found"))?;

        // get the stats handler map
        let handler_map = registry
            .stats_handlers
            .get(&service_name)
            .cloned()
            .expect("no stats handler registry; this is unexpected");

        drop(registry);

        // insert the stats handler
        if let Some(stats_handler) = stats_handler {
            handler_map
                .lock()
                .unwrap()
                .insert(endpoint.subject_to(lease_id), stats_handler);
        }

        // creates an endpoint for the service
        let service_endpoint = group
            .endpoint(&endpoint.name_with_id(lease_id))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start endpoint: {e}"))?;

        let cancel_token = lease
            .map(|l| l.child_token())
            .unwrap_or_else(|| endpoint.drt().child_token());

        let push_endpoint = PushEndpoint::builder()
            .service_handler(handler)
            .cancellation_token(cancel_token.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build push endpoint: {e}"))?;

        // launch in primary runtime
        let task = tokio::spawn(push_endpoint.start(service_endpoint));

        // make the components service endpoint discovery in etcd

        // client.register_service()
        let info = Instance {
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
            namespace: endpoint.component.namespace.name.clone(),
            instance_id: lease_id,
            transport: TransportType::NatsTcp(endpoint.subject_to(lease_id)),
        };

        let info = serde_json::to_vec_pretty(&info)?;

        if let Some(etcd_client) = &endpoint.component.drt.etcd_client {
            if let Err(e) = etcd_client
                .kv_create(endpoint.etcd_path(lease_id), info, Some(lease_id))
                .await
            {
                tracing::error!("Failed to register discoverable service: {:?}", e);
                cancel_token.cancel();
                return Err(error!("Failed to register discoverable service"));
            }
        }
        task.await??;

        Ok(())
    }
}
