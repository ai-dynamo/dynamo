// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::{pin::Pin, sync::Arc};

use dynamo_llm::{
    backend::Backend,
    engines::StreamingEngineAdapter,
    http::service::discovery::{ModelEntry, ModelNetworkName},
    key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
    model_card::{self, ModelDeploymentCard},
    model_type::ModelType,
    preprocessor::{BackendInput, BackendOutput},
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        Annotated,
    },
};
use dynamo_runtime::pipeline::{
    network::Ingress, Context, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source,
};
use dynamo_runtime::{component::Endpoint, engine::AsyncEngineStream};
use dynamo_runtime::{protocols::Endpoint as EndpointId, DistributedRuntime};

use crate::EngineConfig;

pub async fn run(
    distributed_runtime: DistributedRuntime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let cancel_token = distributed_runtime.primary_token().clone();
    let endpoint_id: EndpointId = path.parse()?;

    let (rt_fut, mut card) = match engine_config {
        EngineConfig::StaticFull {
            service_name,
            engine,
            mut card,
        } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            card.requires_preprocessing = false;

            let ingress_chat = Ingress::<
                Context<NvCreateChatCompletionRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
            >::for_engine(engine)?;
            let endpoint_chat = register(
                distributed_runtime.clone(),
                &service_name,
                endpoint_id,
                *card.clone(),
                ModelType::Chat,
            )
            .await?;
            let fut_chat = endpoint_chat
                .endpoint_builder()
                .handler(ingress_chat)
                .start();

            (fut_chat, card)
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            mut card,
        } => {
            // Pre-processing is done ingress-side, so it should be already done.
            let frontend =
                SegmentSource::<SingleIn<BackendInput>, ManyOut<Annotated<BackendOutput>>>::new();
            let backend = Backend::from_mdc(*card.clone()).await?.into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);

            let pipeline = frontend
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(frontend)?;

            let ingress = Ingress::for_pipeline(pipeline)?;
            card.requires_preprocessing = true;
            let endpoint = register(
                distributed_runtime.clone(),
                &service_name,
                endpoint_id,
                *card.clone(),
                ModelType::Backend,
            )
            .await?;
            (endpoint.endpoint_builder().handler(ingress).start(), card)
        }
        EngineConfig::Dynamic(_) => {
            anyhow::bail!("Cannot use endpoint for both in and out");
        }
        EngineConfig::None => unreachable!(),
    };

    tokio::select! {
        _ = rt_fut => {
            tracing::debug!("Endpoint ingress ended");
        }
        _ = cancel_token.cancelled() => {
        }
    }

    // Cleanup on shutdown
    if let Err(err) = card
        .delete_from_nats(distributed_runtime.nats_client())
        .await
    {
        tracing::error!(%err, "delete_from_nats error on shutdown");
    }

    Ok(())
}

async fn register(
    distributed_runtime: DistributedRuntime,
    service_name: &str,
    endpoint_id: EndpointId,
    mut card: ModelDeploymentCard,
    model_type: ModelType,
) -> anyhow::Result<Endpoint> {
    let component = distributed_runtime
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?;
    let endpoint = component
        .service_builder()
        .create()
        .await?
        .endpoint(&endpoint_id.name);

    // A static component doesn't have an etcd_client because it doesn't need to register
    if let Some(etcd_client) = distributed_runtime.etcd_client() {
        // Store model config files in NATS object store
        let nats_client = distributed_runtime.nats_client();
        card.move_to_nats(nats_client.clone()).await?;

        // Publish the Model Deployment Card to etcd
        let kvstore: Box<dyn KeyValueStore> =
            Box::new(EtcdStorage::new(etcd_client.clone(), endpoint_id.clone()));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        let key = card.slug().to_string();
        card_store
            .publish(model_card::BUCKET_NAME, None, &key, &mut card)
            .await?;

        // Publish our ModelEntry to etcd. This allows ingress to find the model card.
        // (Why don't we put the model card directly under this key?)
        let network_name = ModelNetworkName::from_local(&endpoint, etcd_client.lease_id());
        tracing::debug!("Registering with etcd as {network_name}");
        let model_registration = ModelEntry {
            name: service_name.to_string(),
            endpoint: endpoint_id.clone(),
            model_type,
        };
        etcd_client
            .kv_create(
                network_name.to_string(),
                serde_json::to_vec_pretty(&model_registration)?,
                None, // use primary lease
            )
            .await?;
    }
    Ok(endpoint)
}
