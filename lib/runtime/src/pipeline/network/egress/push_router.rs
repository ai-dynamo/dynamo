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

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use crate::{
    component::{Client, Endpoint, EndpointSource},
    engine::{AsyncEngine, Data},
    pipeline::{AddressedPushRouter, AddressedRequest, Error, ManyOut, SingleIn},
    traits::DistributedRuntimeProvider,
};

#[derive(Clone)]
pub struct PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    // TODO: This shouldn't be pub, but lib/bindings/python/rust/lib.rs exposes it.
    /// The Client is how we gather remote endpoint information from etcd.
    pub client: Client,

    /// How we choose which endpoint to send traffic to.
    ///
    /// Setting this to KV means we never intend to call `generate` on this PushRouter. We are
    /// not using it as an AsyncEngine.
    /// Instead we will decide whether to call random/round_robin/direct ourselves and call them directly.
    /// dynamo-llm's KV Routing does this.
    router_mode: RouterMode,

    /// Number of round robin requests handled. Used to decide which server is next.
    round_robin_counter: Arc<AtomicU64>,

    /// The next step in the chain. PushRouter (this object) picks an endpoint,
    /// addresses it, then passes it to AddressedPushRouter which does the network traffic.
    addressed: Arc<AddressedPushRouter>,

    /// An internal Rust type. This says that PushRouter is generic over the T and U types,
    /// which are the input and output types of it's `generate` function. It allows the
    /// compiler to specialize us at compile time.
    _phantom: PhantomData<(T, U)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum RouterMode {
    #[default]
    RoundRobin,
    Random,
    Direct(i64),
    // Marker value, KV routing itself is in dynamo-llm
    KV,
}

impl RouterMode {
    pub fn is_kv_routing(&self) -> bool {
        *self == RouterMode::KV
    }
}

async fn addressed_router(endpoint: &Endpoint) -> anyhow::Result<Arc<AddressedPushRouter>> {
    AddressedPushRouter::new(
        endpoint.drt().nats_client.client().clone(),
        endpoint.drt().tcp_server().await?,
    )
}

impl<T, U> PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    pub async fn from_client(client: Client, router_mode: RouterMode) -> anyhow::Result<Self> {
        let addressed = addressed_router(&client.endpoint).await?;
        Ok(PushRouter {
            client,
            addressed,
            router_mode,
            round_robin_counter: Arc::new(AtomicU64::new(0)),
            _phantom: PhantomData,
        })
    }

    /// Issue a request to the next available endpoint in a round-robin fashion
    pub async fn round_robin(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);

        let endpoint_id = {
            let endpoints = self.client.endpoints();
            let count = endpoints.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no endpoints found for endpoint {:?}",
                    self.client.endpoint.etcd_path()
                ));
            }
            let offset = counter % count as u64;
            endpoints[offset as usize].id()
        };
        tracing::trace!("round robin router selected {endpoint_id}");

        let subject = self.client.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.addressed.generate(request).await
    }

    /// Issue a request to a random endpoint
    pub async fn random(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let endpoint_id = {
            let endpoints = self.client.endpoints();
            let count = endpoints.len();
            if count == 0 {
                return Err(anyhow::anyhow!(
                    "no endpoints found for endpoint {:?}",
                    self.client.endpoint.etcd_path()
                ));
            }
            let counter = rand::rng().random::<u64>();
            let offset = counter % count as u64;
            endpoints[offset as usize].id()
        };
        tracing::trace!("random router selected {endpoint_id}");

        let subject = self.client.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.addressed.generate(request).await
    }

    /// Issue a request to a specific endpoint
    pub async fn direct(
        &self,
        request: SingleIn<T>,
        endpoint_id: i64,
    ) -> anyhow::Result<ManyOut<U>> {
        let found = {
            let endpoints = self.client.endpoints();
            endpoints.iter().any(|ep| ep.id() == endpoint_id)
        };

        if !found {
            return Err(anyhow::anyhow!(
                "endpoint_id={} not found for endpoint {:?}",
                endpoint_id,
                self.client.endpoint.etcd_path()
            ));
        }

        let subject = self.client.endpoint.subject_to(endpoint_id);
        let request = request.map(|req| AddressedRequest::new(req, subject));

        self.addressed.generate(request).await
    }

    pub async fn r#static(&self, request: SingleIn<T>) -> anyhow::Result<ManyOut<U>> {
        let subject = self.client.endpoint.subject();
        tracing::debug!("static got subject: {subject}");
        let request = request.map(|req| AddressedRequest::new(req, subject));
        tracing::debug!("router generate");
        self.addressed.generate(request).await
    }
}

#[async_trait]
impl<T, U> AsyncEngine<SingleIn<T>, ManyOut<U>, Error> for PushRouter<T, U>
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de>,
{
    async fn generate(&self, request: SingleIn<T>) -> Result<ManyOut<U>, Error> {
        match &self.client.endpoints {
            EndpointSource::Static => self.r#static(request).await,
            EndpointSource::Dynamic(_) => match self.router_mode {
                RouterMode::Random => self.random(request).await,
                RouterMode::RoundRobin => self.round_robin(request).await,
                RouterMode::Direct(endpoint_id) => self.direct(request, endpoint_id).await,
                RouterMode::KV => {
                    anyhow::bail!("KV routing should not call generate on PushRouter");
                }
            },
        }
    }
}
