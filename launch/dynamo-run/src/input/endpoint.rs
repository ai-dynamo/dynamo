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

use std::sync::Arc;

use dynamo_llm::{
    backend::Backend,
    engines::StreamingEngineAdapter,
    http::service::discovery::{ModelEntry, ModelNetworkName},
    key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
    model_card,
    model_type::ModelType,
    preprocessor::OpenAIPreprocessor,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        Annotated,
    },
};
use dynamo_runtime::pipeline::{
    network::Ingress, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source,
};
use dynamo_runtime::{protocols::Endpoint, DistributedRuntime};

use crate::EngineConfig;

pub async fn run(
    distributed_runtime: DistributedRuntime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    // This will attempt to connect to NATS and etcd

    let cancel_token = distributed_runtime.primary_token().clone();
    let endpoint_id: Endpoint = path.parse()?;

    let etcd_client = distributed_runtime.etcd_client();

    let (ingress, service_name, mut card, requires_preprocessing) = match engine_config {
        EngineConfig::StaticFull {
            service_name,
            engine,
            card,
        } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            (Ingress::for_engine(engine)?, service_name, card, false)
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let frontend = SegmentSource::<
                SingleIn<NvCreateChatCompletionRequest>,
                ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(*card.clone())
                .await?
                .into_operator();
            let backend = Backend::from_mdc(*card.clone()).await?.into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);

            let pipeline = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            // TODO: switch last 'false' to 'true' once we have ingress-side pre-processing
            (Ingress::for_pipeline(pipeline)?, service_name, card, false)
        }
        EngineConfig::Dynamic(_) => {
            anyhow::bail!("Cannot use endpoint for both in and out");
        }
        EngineConfig::None => unreachable!(),
    };

    let model_registration = ModelEntry {
        name: service_name.to_string(),
        endpoint: endpoint_id.clone(),
        model_type: ModelType::Chat,
    };

    let component = distributed_runtime
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?;
    let endpoint = component
        .service_builder()
        .create()
        .await?
        .endpoint(&endpoint_id.name);

    if let Some(etcd_client) = etcd_client {
        // Store model config files in NATS object store
        let nats_client = distributed_runtime.nats_client();
        card.move_to_nats(nats_client.clone()).await?;

        // Publish the Model Deployment Card to etcd
        let kvstore: Box<dyn KeyValueStore> =
            Box::new(EtcdStorage::new(etcd_client.clone(), endpoint_id));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        card.requires_preprocessing = requires_preprocessing; // Not used yet. Soon.
        let key = card.slug().to_string();
        card_store
            .publish(model_card::BUCKET_NAME, None, &key, &mut *card.clone())
            .await?;

        // Publish our ModelEntry to etcd. This allows ingress to find the model card.
        // (Why don't we put the model card directly under this key?)
        let network_name = ModelNetworkName::from_local(&endpoint, etcd_client.lease_id());
        tracing::debug!("Registering with etcd as {network_name}");
        etcd_client
            .kv_create(
                network_name.to_string(),
                serde_json::to_vec_pretty(&model_registration)?,
                None, // use primary lease
            )
            .await?;
    }

    let rt_fut = endpoint.endpoint_builder().handler(ingress).start();
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
