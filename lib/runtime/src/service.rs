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

// TODO - refactor this entire module
//
// we want to carry forward the concept of live vs ready for the components
// we will want to associate the components cancellation token with the
// component's "service state"

use crate::{error, transports::nats, utils::stream, Result};

use async_nats::Message;
use async_stream::try_stream;
use bytes::Bytes;
use derive_getters::Dissolve;
use futures::stream::{StreamExt, TryStreamExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;

pub struct ServiceClient {
    nats_client: nats::Client,
}

impl ServiceClient {
    pub fn new(nats_client: nats::Client) -> Self {
        ServiceClient { nats_client }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSet {
    services: Vec<ServiceInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub name: String,
    pub id: String,
    pub version: String,
    pub started: String,
    pub endpoints: Vec<EndpointInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct EndpointInfo {
    pub name: String,
    pub subject: String,

    #[serde(flatten)]
    pub data: Option<Metrics>,
}

impl EndpointInfo {
    pub fn id(&self) -> Result<i64> {
        let id = self
            .subject
            .split('-')
            .next_back()
            .ok_or_else(|| error!("No id found in subject"))?;

        i64::from_str_radix(id, 16).map_err(|e| error!("Invalid id format: {}", e))
    }
}

// TODO: This is _really_ close to the async_nats::service::Stats object,
// but it's missing a few fields like "name", so use a temporary struct
// for easy deserialization. Ideally, this type already exists or can
// be exposed in the library somewhere.
/// Stats structure returned from NATS service API
#[derive(Debug, Clone, Serialize, Deserialize, Dissolve)]
pub struct Metrics {
    // Standard NATS Service API fields
    pub average_processing_time: f64,
    pub last_error: String,
    pub num_errors: u64,
    pub num_requests: u64,
    pub processing_time: u64,
    pub queue_group: String,
    // Field containing custom stats handler data
    pub data: serde_json::Value,
}

impl Metrics {
    pub fn decode<T: for<'de> Deserialize<'de>>(self) -> Result<T> {
        serde_json::from_value(self.data).map_err(Into::into)
    }
}

impl ServiceClient {
    pub async fn unary(
        &self,
        subject: impl Into<String>,
        payload: impl Into<Bytes>,
    ) -> Result<Message> {
        let response = self
            .nats_client
            .client()
            .request(subject.into(), payload.into())
            .await?;
        Ok(response)
    }

    pub async fn collect_services(
        &self,
        service_name: &str,
        timeout: Duration,
    ) -> Result<ServiceSet> {
        let sub = self.nats_client.scrape_service(service_name).await?;
        if timeout.is_zero() {
            tracing::warn!("collect_services: timeout is zero");
        }
        if timeout > Duration::from_secs(10) {
            tracing::warn!("collect_services: timeout is greater than 10 seconds");
        }
        let deadline = tokio::time::Instant::now() + timeout;

        let mut services = vec![];
        let mut s = stream::until_deadline(sub, deadline);
        while let Some(message) = s.next().await {
            if message.payload.is_empty() {
                // Expected while we wait for KV metrics in worker to start
                tracing::trace!(service_name, "collect_services: empty payload from nats");
                continue;
            }
            let info = serde_json::from_slice::<ServiceInfo>(&message.payload);
            match info {
                Ok(info) => services.push(info),
                Err(err) => {
                    let payload = String::from_utf8_lossy(&message.payload);
                    tracing::debug!(%err, service_name, %payload, "error decoding service info");
                }
            }
        }

        Ok(ServiceSet { services })
    }
}

impl ServiceSet {
    pub fn into_endpoints(self) -> impl Iterator<Item = EndpointInfo> {
        self.services
            .into_iter()
            .flat_map(|s| s.endpoints.into_iter())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_service_set() {
        let services = vec![
            ServiceInfo {
                name: "service1".to_string(),
                id: "1".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Some(Metrics {
                            average_processing_time: 0.1,
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                    EndpointInfo {
                        name: "endpoint2-foo".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(Metrics {
                            average_processing_time: 0.1,
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                ],
            },
            ServiceInfo {
                name: "service1".to_string(),
                id: "2".to_string(),
                version: "1.0".to_string(),
                started: "2021-01-01".to_string(),
                endpoints: vec![
                    EndpointInfo {
                        name: "endpoint1".to_string(),
                        subject: "subject1".to_string(),
                        data: Some(Metrics {
                            average_processing_time: 0.1,
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value1"}),
                        }),
                    },
                    EndpointInfo {
                        name: "endpoint2-bar".to_string(),
                        subject: "subject2".to_string(),
                        data: Some(Metrics {
                            average_processing_time: 0.1,
                            last_error: "none".to_string(),
                            num_errors: 0,
                            num_requests: 10,
                            processing_time: 100,
                            queue_group: "group1".to_string(),
                            data: serde_json::json!({"key": "value2"}),
                        }),
                    },
                ],
            },
        ];

        let service_set = ServiceSet { services };

        let endpoints: Vec<_> = service_set
            .into_endpoints()
            .filter(|e| e.name.starts_with("endpoint2"))
            .collect();

        assert_eq!(endpoints.len(), 2);
    }
}
