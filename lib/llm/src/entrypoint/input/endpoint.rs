// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, future::Future, pin::Pin, sync::Arc};

use anyhow::Context as _;
use futures::{StreamExt, future::pending};
use tokio::time::{Duration, Instant, sleep_until};

use crate::{
    backend::Backend,
    discovery::{KvWorkerMonitor, ModelManager, WORKER_TYPE_DECODE},
    engines::StreamingEngineAdapter,
    entrypoint::{EngineConfig, build_preprocessed_routing},
    http::service::metrics::Metrics,
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    namespace::NamespaceFilter,
    preprocessor::{BackendOutput, PreprocessedRequest},
    protocols::common::llm_backend::LLMEngineOutput,
    types::{
        Annotated,
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
    },
    worker_type::WorkerType,
};

use dynamo_runtime::engine::AsyncEngineStream;
use dynamo_runtime::{
    DistributedRuntime,
    config::HealthStatus,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, DiscoverySpec,
    },
    pipeline::{
        Context, ManyOut, Operator, RouterMode, SegmentSource, ServiceBackend, SingleIn, Source,
        network::Ingress,
    },
    protocols::EndpointId,
};

pub async fn run(
    distributed_runtime: DistributedRuntime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let cancel_token = distributed_runtime.primary_token().clone();
    let endpoint_id: EndpointId = path.parse()?;

    let component = distributed_runtime
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?;
    let endpoint = component.endpoint(&endpoint_id.name);

    let rt_fut: Pin<Box<dyn Future<Output = _> + Send + 'static>> = match engine_config {
        EngineConfig::InProcessText { engine, mut model } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let ingress_chat = Ingress::<
                Context<NvCreateChatCompletionRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
            >::for_engine(engine)?;
            // In-process engines are always Aggregated: they own the full
            // serving stack inline and don't depend on a peer worker.
            model
                .attach(
                    &endpoint,
                    ModelType::Chat,
                    ModelInput::Text,
                    None,
                    Some(WorkerType::Aggregated),
                    Vec::new(),
                )
                .await?;
            let fut_chat = endpoint.endpoint_builder().handler(ingress_chat).start();

            Box::pin(fut_chat)
        }
        EngineConfig::InProcessTokens {
            engine: inner_engine,
            mut model,
            is_prefill,
            is_decode,
        } => {
            // Pre-processing is done ingress-side, so it should be already done.
            let frontend = SegmentSource::<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
            >::new();
            let backend = Backend::from_mdc(model.card()).into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);
            let pipeline = frontend
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(frontend)?;
            let ingress = Ingress::for_pipeline(pipeline)?;

            // The disaggregation role is carried by `worker_type`, not
            // `model_type`. Prefill workers register with an empty
            // `model_type` (no OpenAI surface) and `WorkerType::Prefill`; a
            // decode worker keeps its OpenAI surface and registers as
            // `WorkerType::Decode` so the readiness gate can pair it with its
            // prefill peer (mirrors the real vLLM/SGLang/TRT-LLM backends:
            // decode `needs` Prefill and prefill `needs` Decode). A worker
            // that is neither is a standalone `Aggregated` worker with no
            // peer dependency.
            let (model_type, worker_type, needs) = if is_prefill {
                (
                    ModelType::empty(),
                    Some(WorkerType::Prefill),
                    vec![vec![WorkerType::Decode]],
                )
            } else if is_decode {
                (
                    ModelType::Chat | ModelType::Completions,
                    Some(WorkerType::Decode),
                    vec![vec![WorkerType::Prefill]],
                )
            } else {
                (
                    ModelType::Chat | ModelType::Completions,
                    Some(WorkerType::Aggregated),
                    Vec::new(),
                )
            };
            model
                .attach(
                    &endpoint,
                    model_type,
                    ModelInput::Tokens,
                    None,
                    worker_type,
                    needs,
                )
                .await?;

            let fut = endpoint.endpoint_builder().handler(ingress).start();
            Box::pin(fut)
        }
        EngineConfig::Dynamic {
            model: local_model,
            chat_engine_factory,
            prefill_load_estimator,
        } => {
            if chat_engine_factory.is_some() {
                tracing::warn!(
                    "Bulwark gateway endpoint mode receives preprocessed request-plane traffic; \
                     dyn-chat-processor factories are ignored in this mode"
                );
            }

            let namespace_filter = NamespaceFilter::from_namespace_and_prefix(
                local_model.namespace(),
                local_model.namespace_prefix(),
            );
            if namespace_filter.is_global() {
                anyhow::bail!(
                    "Bulwark gateway endpoint mode requires --namespace or --namespace-prefix \
                     to select private primary/shadow workers"
                );
            }

            let (private_endpoint_id, private_card) =
                wait_for_gateway_backend(&distributed_runtime, &endpoint_id, &namespace_filter)
                    .await?;

            let private_component = distributed_runtime
                .namespace(&private_endpoint_id.namespace)?
                .component(&private_endpoint_id.component)?;
            let private_endpoint = private_component.endpoint(&private_endpoint_id.name);
            let private_client = private_endpoint.client().await?;

            let model_manager = Arc::new(ModelManager::new());
            let metrics = Arc::new(Metrics::new());
            let router_config = private_card
                .router_config
                .as_ref()
                .unwrap_or(local_model.router_config());

            let kv_chooser = if router_config.router_mode == RouterMode::KV {
                Some(
                    model_manager
                        .kv_chooser_for(
                            &private_endpoint,
                            private_card.kv_cache_block_size,
                            Some(router_config.kv_router_config.clone()),
                            prefill_load_estimator.clone(),
                            WORKER_TYPE_DECODE,
                            Some(private_card.display_name.clone()),
                            private_card.runtime_config.enable_eagle,
                        )
                        .await?,
                )
            } else {
                None
            };

            let monitor_client = kv_chooser
                .as_ref()
                .map(|chooser| chooser.client().clone())
                .unwrap_or_else(|| private_client.clone());
            let worker_monitor = Some(KvWorkerMonitor::new(
                monitor_client,
                router_config.load_threshold_config.clone(),
            ));

            let routing = build_preprocessed_routing(
                &private_client,
                model_manager,
                router_config.router_mode,
                worker_monitor,
                kv_chooser,
                None,
                false,
                router_config.enforce_disagg,
                router_config.session_affinity_ttl_secs,
            )
            .await
            .context("build Bulwark gateway preprocessed routing")?;
            let pipeline = routing
                .build_preprocessed_pipeline(
                    &private_card,
                    local_model.migration_limit(),
                    local_model.migration_max_seq_len(),
                    metrics,
                )
                .context("build Bulwark gateway preprocessed pipeline")?;
            let ingress = Ingress::<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<LLMEngineOutput>>,
            >::for_engine(pipeline)?;

            let public_card = gateway_public_card(&private_card);
            register_gateway_model_card(&distributed_runtime, &endpoint_id, &public_card).await?;
            spawn_gateway_backend_readiness_monitor(
                distributed_runtime.clone(),
                endpoint_id.clone(),
                namespace_filter,
            );

            tracing::info!(
                public_endpoint = %endpoint_id,
                private_endpoint = %private_endpoint_id,
                model = %public_card.name(),
                "Bulwark gateway endpoint registered"
            );

            let fut = endpoint.endpoint_builder().handler(ingress).start();
            Box::pin(fut)
        }
    };

    // Capture the actual error from rt_fut when it completes
    // Note: We must return rt_result to propagate the actual error back to the user.
    // If we don't return the specific error, the programmer/user won't know what actually
    // caused the endpoint service to fail, making debugging much more difficult.
    tokio::select! {
        rt_result = rt_fut => {
            tracing::debug!("Endpoint service completed");
            match rt_result {
                Ok(_) => {
                    tracing::warn!("Endpoint service completed unexpectedly for endpoint: {}", path);
                    Err(anyhow::anyhow!("Endpoint service completed unexpectedly for endpoint: {}", path))
                }
                Err(e) => {
                    tracing::error!(%e, "Endpoint service failed for endpoint: {} - Error: {}", path, e);
                    Err(anyhow::anyhow!("Endpoint service failed for endpoint: {} - Error: {}", path, e))
                }
            }
        }
        _ = cancel_token.cancelled() => {
            tracing::debug!("Endpoint service cancelled");
            Ok(())
        }
    }
}

fn gateway_public_card(private_card: &ModelDeploymentCard) -> ModelDeploymentCard {
    let mut card = private_card.clone();
    let mut model_type = ModelType::empty();
    if private_card.model_type.supports_chat() {
        model_type |= ModelType::Chat;
    }
    if private_card.model_type.supports_completions() {
        model_type |= ModelType::Completions;
    }
    card.model_type = model_type;
    card.model_input = ModelInput::Tokens;
    card.worker_type = Some(WorkerType::Aggregated);
    card.needs.clear();
    card.router_config = None;
    card
}

fn is_gateway_backend_card(card: &ModelDeploymentCard) -> bool {
    card.model_input == ModelInput::Tokens
        && (card.model_type.supports_chat() || card.model_type.supports_completions())
        && !card.model_type.supports_prefill()
        && matches!(card.worker_type, None | Some(WorkerType::Aggregated))
        && card.needs.is_empty()
}

async fn wait_for_gateway_backend(
    distributed_runtime: &DistributedRuntime,
    public_endpoint_id: &EndpointId,
    namespace_filter: &NamespaceFilter,
) -> anyhow::Result<(EndpointId, ModelDeploymentCard)> {
    let discovery_stream = distributed_runtime
        .discovery()
        .list_and_watch(
            DiscoveryQuery::AllModels,
            Some(distributed_runtime.primary_token().clone()),
        )
        .await?;
    tokio::pin!(discovery_stream);

    tracing::info!(
        public_endpoint = %public_endpoint_id,
        namespace_filter = ?namespace_filter,
        "Waiting for private aggregated token worker for Bulwark gateway"
    );

    while let Some(event) = discovery_stream.next().await {
        let event = event?;
        let DiscoveryEvent::Added(instance) = event else {
            continue;
        };
        let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            ..
        } = &instance
        else {
            continue;
        };

        if !namespace_filter.matches(namespace) {
            continue;
        }

        let endpoint_id = EndpointId {
            namespace: namespace.clone(),
            component: component.clone(),
            name: endpoint.clone(),
        };
        if &endpoint_id == public_endpoint_id {
            tracing::debug!(%endpoint_id, "Skipping Bulwark gateway public endpoint");
            continue;
        }

        let card = match instance.deserialize_model::<ModelDeploymentCard>() {
            Ok(card) => card,
            Err(err) => {
                tracing::warn!(%endpoint_id, %err, "Skipping unreadable model card");
                continue;
            }
        };
        if !is_gateway_backend_card(&card) {
            tracing::debug!(
                %endpoint_id,
                model = %card.name(),
                model_input = %card.model_input.as_str(),
                model_type = %card.model_type.as_str(),
                worker_type = ?card.worker_type,
                needs = ?card.needs,
                "Skipping model card that is not an aggregated token backend for Bulwark gateway"
            );
            continue;
        }

        tracing::info!(
            private_endpoint = %endpoint_id,
            model = %card.name(),
            "Found private Bulwark backend for gateway"
        );
        return Ok((endpoint_id, card));
    }

    anyhow::bail!(
        "model discovery stream ended before a private aggregated token worker appeared for Bulwark gateway"
    )
}

const BULWARK_GATEWAY_NOTREADY_GRACE_MS_ENV: &str = "DYN_BULWARK_GATEWAY_NOTREADY_GRACE_MS";
const DEFAULT_BULWARK_GATEWAY_NOTREADY_GRACE_MS: u64 = 10_000;

fn gateway_notready_grace() -> Duration {
    match std::env::var(BULWARK_GATEWAY_NOTREADY_GRACE_MS_ENV) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(ms) => Duration::from_millis(ms),
            Err(err) => {
                tracing::warn!(
                    value = %raw,
                    %err,
                    default_ms = DEFAULT_BULWARK_GATEWAY_NOTREADY_GRACE_MS,
                    "Invalid DYN_BULWARK_GATEWAY_NOTREADY_GRACE_MS; using default gateway readiness grace"
                );
                Duration::from_millis(DEFAULT_BULWARK_GATEWAY_NOTREADY_GRACE_MS)
            }
        },
        Err(_) => Duration::from_millis(DEFAULT_BULWARK_GATEWAY_NOTREADY_GRACE_MS),
    }
}

fn spawn_gateway_backend_readiness_monitor(
    distributed_runtime: DistributedRuntime,
    public_endpoint_id: EndpointId,
    namespace_filter: NamespaceFilter,
) {
    let cancel_token = distributed_runtime.primary_token().clone();
    let health_runtime = distributed_runtime.clone();

    tokio::spawn(async move {
        let discovery_stream = match distributed_runtime
            .discovery()
            .list_and_watch(DiscoveryQuery::AllModels, Some(cancel_token.clone()))
            .await
        {
            Ok(stream) => stream,
            Err(err) => {
                tracing::error!(
                    %err,
                    public_endpoint = %public_endpoint_id,
                    "Bulwark gateway failed to start private backend readiness monitor"
                );
                set_gateway_health(&health_runtime, HealthStatus::NotReady, 0);
                return;
            }
        };
        tokio::pin!(discovery_stream);

        let notready_grace = gateway_notready_grace();
        tracing::info!(
            public_endpoint = %public_endpoint_id,
            notready_grace_ms = notready_grace.as_millis(),
            "Bulwark gateway readiness monitor configured"
        );

        let mut active_backends: HashSet<DiscoveryInstanceId> = HashSet::new();
        let mut last_ready = false;
        let mut pending_notready_deadline: Option<Instant> = None;
        set_gateway_health(&health_runtime, HealthStatus::NotReady, 0);

        loop {
            let pending_notready = async {
                match pending_notready_deadline {
                    Some(deadline) => sleep_until(deadline).await,
                    None => pending::<()>().await,
                }
            };

            tokio::select! {
                event = discovery_stream.next() => {
                    let Some(event) = event else {
                        tracing::warn!(
                            public_endpoint = %public_endpoint_id,
                            "Bulwark gateway private backend readiness stream ended"
                        );
                        set_gateway_health(&health_runtime, HealthStatus::NotReady, 0);
                        break;
                    };

                    match event {
                        Ok(DiscoveryEvent::Added(instance)) => {
                            let id = instance.id();
                            if is_gateway_backend_instance(
                                &instance,
                                &public_endpoint_id,
                                &namespace_filter,
                            ) {
                                active_backends.insert(id);
                            } else {
                                active_backends.remove(&id);
                            }
                        }
                        Ok(DiscoveryEvent::Removed(id)) => {
                            active_backends.remove(&id);
                        }
                        Err(err) => {
                            tracing::warn!(
                                %err,
                                public_endpoint = %public_endpoint_id,
                                "Bulwark gateway private backend readiness watch event failed"
                            );
                            continue;
                        }
                    }

                    let ready = !active_backends.is_empty();
                    if ready {
                        pending_notready_deadline = None;
                        if !last_ready {
                            set_gateway_health(&health_runtime, HealthStatus::Ready, active_backends.len());
                            tracing::info!(
                                public_endpoint = %public_endpoint_id,
                                active_private_backends = active_backends.len(),
                                ready = true,
                                "Bulwark gateway readiness changed"
                            );
                            last_ready = true;
                        }
                    } else if last_ready {
                        if notready_grace.is_zero() {
                            set_gateway_health(&health_runtime, HealthStatus::NotReady, 0);
                            tracing::info!(
                                public_endpoint = %public_endpoint_id,
                                active_private_backends = 0,
                                ready = false,
                                "Bulwark gateway readiness changed"
                            );
                            last_ready = false;
                        } else if pending_notready_deadline.is_none() {
                            pending_notready_deadline = Some(Instant::now() + notready_grace);
                            tracing::info!(
                                public_endpoint = %public_endpoint_id,
                                notready_grace_ms = notready_grace.as_millis(),
                                "Bulwark gateway private backends empty; delaying NotReady"
                            );
                        }
                    }
                }
                _ = pending_notready => {
                    pending_notready_deadline = None;
                    if active_backends.is_empty() && last_ready {
                        set_gateway_health(&health_runtime, HealthStatus::NotReady, 0);
                        tracing::info!(
                            public_endpoint = %public_endpoint_id,
                            active_private_backends = 0,
                            ready = false,
                            "Bulwark gateway readiness changed after grace"
                        );
                        last_ready = false;
                    }
                }
            }
        }
    });
}

fn set_gateway_health(
    distributed_runtime: &DistributedRuntime,
    status: HealthStatus,
    active_private_backends: usize,
) {
    distributed_runtime
        .system_health()
        .lock()
        .set_health_status(status.clone());
    tracing::debug!(
        ?status,
        active_private_backends,
        "Bulwark gateway system health updated"
    );
}

fn is_gateway_backend_instance(
    instance: &DiscoveryInstance,
    public_endpoint_id: &EndpointId,
    namespace_filter: &NamespaceFilter,
) -> bool {
    let DiscoveryInstance::Model {
        namespace,
        component,
        endpoint,
        ..
    } = instance
    else {
        return false;
    };

    if !namespace_filter.matches(namespace) {
        return false;
    }

    let endpoint_id = EndpointId {
        namespace: namespace.clone(),
        component: component.clone(),
        name: endpoint.clone(),
    };
    if &endpoint_id == public_endpoint_id {
        return false;
    }

    instance
        .deserialize_model::<ModelDeploymentCard>()
        .is_ok_and(|card| is_gateway_backend_card(&card))
}

async fn register_gateway_model_card(
    distributed_runtime: &DistributedRuntime,
    endpoint_id: &EndpointId,
    card: &ModelDeploymentCard,
) -> anyhow::Result<()> {
    let spec = DiscoverySpec::from_model(
        endpoint_id.namespace.clone(),
        endpoint_id.component.clone(),
        endpoint_id.name.clone(),
        card,
    )?;
    distributed_runtime.discovery().register(spec).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entrypoint::RouterConfig;

    fn token_chat_card() -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::default();
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::Chat;
        card.worker_type = Some(WorkerType::Aggregated);
        card
    }

    #[test]
    fn gateway_backend_card_accepts_aggregated_token_chat() {
        let card = token_chat_card();
        assert!(is_gateway_backend_card(&card));
    }

    #[test]
    fn gateway_backend_card_accepts_legacy_missing_worker_type() {
        let mut card = token_chat_card();
        card.worker_type = None;
        assert!(is_gateway_backend_card(&card));
    }

    #[test]
    fn gateway_backend_card_rejects_disaggregated_decode() {
        let mut card = token_chat_card();
        card.worker_type = Some(WorkerType::Decode);
        card.needs = vec![vec![WorkerType::Prefill]];
        assert!(!is_gateway_backend_card(&card));
    }

    #[test]
    fn gateway_backend_card_rejects_text_or_prefill_cards() {
        let mut text_card = token_chat_card();
        text_card.model_input = ModelInput::Text;
        assert!(!is_gateway_backend_card(&text_card));

        let mut prefill_card = token_chat_card();
        prefill_card.model_type = ModelType::Prefill;
        prefill_card.worker_type = Some(WorkerType::Prefill);
        prefill_card.needs = vec![vec![WorkerType::Decode]];
        assert!(!is_gateway_backend_card(&prefill_card));
    }

    #[test]
    fn gateway_public_card_hides_private_topology() {
        let mut private = token_chat_card();
        private.model_type = ModelType::Chat | ModelType::Completions | ModelType::Images;
        private.worker_type = Some(WorkerType::Decode);
        private.needs = vec![vec![WorkerType::Prefill]];
        private.router_config = Some(RouterConfig::default());

        let public = gateway_public_card(&private);

        assert!(public.model_type.supports_chat());
        assert!(public.model_type.supports_completions());
        assert!(!public.model_type.supports_images());
        assert!(!public.model_type.supports_prefill());
        assert_eq!(public.model_input, ModelInput::Tokens);
        assert_eq!(public.worker_type, Some(WorkerType::Aggregated));
        assert!(public.needs.is_empty());
        assert!(public.router_config.is_none());
    }
}

#[cfg(test)]
#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;
    use dynamo_runtime::protocols::EndpointId;

    async fn create_test_environment() -> anyhow::Result<(DistributedRuntime, EngineConfig)> {
        // Create a minimal distributed runtime and engine config for testing
        let runtime = dynamo_runtime::Runtime::from_settings()
            .map_err(|e| anyhow::anyhow!("Failed to create runtime: {}", e))?;

        let distributed_runtime = dynamo_runtime::DistributedRuntime::from_settings(runtime)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create distributed runtime: {}", e))?;

        let engine_config = EngineConfig::InProcessText {
            engine: crate::engines::make_echo_engine(),
            model: Box::new(
                crate::local_model::LocalModelBuilder::default()
                    .model_name(Some("test-model".to_string()))
                    .build()
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to build LocalModel: {}", e))?,
            ),
        };

        Ok((distributed_runtime, engine_config))
    }

    #[tokio::test]
    #[ignore = "Failing in CI"]
    async fn test_run_function_valid_endpoint() {
        // Test that run() works correctly with valid endpoints

        let (runtime, engine_config) = match create_test_environment().await {
            Ok(env) => env,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Test with valid endpoint - start the service and then connect to it
        let valid_path = "dyn://valid-endpoint.mocker.generate";
        let valid_endpoint: EndpointId = valid_path.parse().expect("Valid endpoint should parse");

        let runtime_clone = runtime.clone();
        let engine_config_clone = engine_config.clone();
        let valid_path_clone = valid_path.to_string();

        let service_handle =
            tokio::spawn(
                async move { run(runtime_clone, valid_path_clone, engine_config_clone).await },
            );

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let client_result = async {
            let namespace = runtime.namespace(&valid_endpoint.namespace)?;
            let component = namespace.component(&valid_endpoint.component)?;
            let client = component.endpoint(&valid_endpoint.name).client().await?;
            client.wait_for_instances().await?;
            Ok::<_, anyhow::Error>(client)
        }
        .await;

        match client_result {
            Ok(_client) => {
                println!("Valid endpoint: Successfully connected to service");
                service_handle.abort(); // Abort the service since we've verified it works
            }
            Err(e) => {
                println!("Valid endpoint: Failed to connect to service: {}", e);
                service_handle.abort(); // Abort the service since the test failed
                panic!(
                    "Valid endpoint should allow client connections, but failed: {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "DistributedRuntime drop issue persists - test logic validates error propagation correctly"]
    async fn test_run_function_invalid_endpoint() {
        // Test that invalid endpoints fail validation during run()
        let invalid_path = "dyn://@@@123.mocker.generate";

        // Create test environment
        let (runtime, engine_config) = create_test_environment()
            .await
            .expect("Failed to create test environment");

        // Call run() directly - it should fail quickly for invalid endpoints
        let result = run(runtime, invalid_path.to_string(), engine_config).await;

        // Should return an error for invalid endpoints
        assert!(
            result.is_err(),
            "run() should fail for invalid endpoint: {:?}",
            result
        );

        // Check that the error message contains validation-related keywords
        let error_msg = result.unwrap_err().to_string().to_lowercase();
        assert!(
            error_msg.contains("invalid")
                || error_msg.contains("namespace")
                || error_msg.contains("validation")
                || error_msg.contains("failed"),
            "Error message should contain validation keywords, got: {}",
            error_msg
        );
    }
}
