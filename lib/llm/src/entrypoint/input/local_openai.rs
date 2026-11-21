// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loopback OpenAI endpoint for worker validation checks.
//!
//! A gated worker is unreachable by design: its card is held, so no frontend
//! routes to it. Its checks still need to send it OpenAI requests that go
//! through the frontend's preprocessing, tool-call and reasoning parsers. This
//! module serves the worker's model on `127.0.0.1` with the same pipeline the
//! in-process token HTTP input builds, once the worker's engine is registered.
//!
//! With `DYN_PUBLISH_GATE`, the endpoint is on a port chosen at startup, the
//! gates start once it serves, and it stops once the worker is released. With
//! `DYN_VALIDATION_ENDPOINT_PORT`, it serves on that port for the worker's
//! lifetime, for a sidecar that runs its own checks and reports through a
//! readiness probe. Every response carries the worker's instance id in
//! [`INSTANCE_ID_HEADER`], so such a sidecar can tell a restarted worker from
//! the one it checked.
//!
//! Nothing here runs unless one of those variables is set.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::local_endpoint_registry::{LocalAsyncEngine, LocalEndpointRegistry};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::validation::{GateEnv, WorkerValidation};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::backend::ExecutionContext;
use crate::http::service::service_v2::HttpService;
use crate::model_card::ModelDeploymentCard;
use crate::model_type::ModelInput;
use crate::protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest};

/// Response header carrying the worker's instance id, in hex.
pub const INSTANCE_ID_HEADER: &str = "x-dynamo-instance-id";

/// Serve the first card a worker registers on the loopback endpoint, if the
/// worker has a publish gate or a fixed endpoint port, and start its gates. A
/// card the endpoint can't serve (prefill, encode, embeddings, text input,
/// LoRA) starts the gates with no `base_url`.
pub async fn serve_if_gated(endpoint: &Endpoint, card: &ModelDeploymentCard) -> anyhow::Result<()> {
    let drt = endpoint.drt();
    if !drt.serves_validation_endpoint() || !drt.validation_endpoint().claim() {
        return Ok(());
    }
    let fixed_port = drt.validation_endpoint().fixed_port();
    let mut env = GateEnv::default();
    let cancel = drt.primary_token().child_token();
    let mut serving = None;
    if servable(card) {
        let listener = tokio::net::TcpListener::bind(SocketAddr::from((
            Ipv4Addr::LOCALHOST,
            fixed_port.unwrap_or(0),
        )))
        .await
        .context("binding the validation endpoint")?;
        let addr = listener.local_addr()?;
        let instance_id = axum::http::HeaderValue::from_str(&format!("{:x}", drt.connection_id()))?;
        let http_service = HttpService::builder()
            .host(Ipv4Addr::LOCALHOST.to_string())
            .port(addr.port())
            .cancel_token(Some(cancel.clone()))
            .build()?
            .map_router(|router| {
                router.layer(axum::middleware::map_response(
                    move |mut response: axum::response::Response| {
                        let instance_id = instance_id.clone();
                        async move {
                            response
                                .headers_mut()
                                .insert(INSTANCE_ID_HEADER, instance_id);
                            response
                        }
                    },
                ))
            });
        let engine: ExecutionContext = Arc::new(RegistryEngine {
            registry: drt.local_endpoint_registry().clone(),
            endpoint_name: endpoint.name().to_string(),
        });
        super::http::serve_in_process_tokens(&http_service, card, engine).await?;
        env.openai_base_url = Some(format!("http://{addr}/v1"));
        serving = Some((http_service, listener));
    } else {
        tracing::warn!(
            model = %card.display_name,
            "Card has no token chat or completions surface; checks get no base_url"
        );
    }
    tokio::spawn(serve_when_registered(
        drt.worker_validation().cloned(),
        drt.local_endpoint_registry().clone(),
        endpoint.name().to_string(),
        serving,
        env,
        drt.child_token(),
        cancel,
        fixed_port.is_none(),
    ));
    Ok(())
}

/// A base model that takes tokens and has a chat or completions surface.
fn servable(card: &ModelDeploymentCard) -> bool {
    card.model_input == ModelInput::Tokens
        && (card.model_type.supports_chat() || card.model_type.supports_completions())
        && card.lora.is_none()
}

/// Once the worker's engine is registered, serve the endpoint and start the
/// gates, so neither a check nor a sidecar's first request races it. Only a
/// servable card registers a local engine, so without an endpoint to serve
/// the gates start at once. With `stop_on_release`, stop the endpoint once
/// the worker is released; a failed gate leaves it up for debugging the held
/// worker.
#[allow(clippy::too_many_arguments)]
async fn serve_when_registered(
    validation: Option<Arc<WorkerValidation>>,
    registry: LocalEndpointRegistry,
    endpoint_name: String,
    serving: Option<(HttpService, tokio::net::TcpListener)>,
    env: GateEnv,
    gates_cancel: CancellationToken,
    endpoint_cancel: CancellationToken,
    stop_on_release: bool,
) {
    if let Some((http_service, listener)) = serving {
        let mut registrations = registry.subscribe();
        tokio::select! {
            Ok(_) = registrations.wait_for(|_| registry.get(&endpoint_name).is_some()) => {}
            _ = endpoint_cancel.cancelled() => return,
        }
        let addr = listener.local_addr().ok();
        let serving = http_service
            .spawn_with_listener(endpoint_cancel.clone(), listener)
            .await;
        tokio::spawn(async move {
            if let Ok(Err(error)) = serving.await {
                tracing::error!(%error, "Validation endpoint stopped");
            }
        });
        tracing::info!(?addr, "Serving the validation endpoint");
    }
    let Some(validation) = validation else {
        return;
    };
    validation.start(env, gates_cancel);
    if stop_on_release {
        tokio::select! {
            _ = validation.wait_released() => endpoint_cancel.cancel(),
            _ = endpoint_cancel.cancelled() => {}
        }
    }
}

/// Token engine that forwards to the worker's local engine, looked up by
/// endpoint name on every request. Requests and responses go through serde,
/// exactly as over the request plane, so the parsers see what a frontend sees.
struct RegistryEngine {
    registry: LocalEndpointRegistry,
    endpoint_name: String,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RegistryEngine
{
    async fn generate(
        &self,
        input: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let Some(engine) = self.registry.get(&self.endpoint_name) else {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Unavailable)
                .message(format!(
                    "local engine for endpoint '{}' is not registered yet",
                    self.endpoint_name
                ))
                .build()
                .into());
        };
        let (request, context) = input.into_parts();
        let request = serde_json::to_value(&request).context("serializing request")?;
        let stream = engine.generate(context.map(|_| request)).await?;
        let context = stream.context();
        let stream = stream.map(|annotated| {
            annotated.map_data(|value| {
                serde_json::from_value::<LLMEngineOutput>(value)
                    .map_err(|error| format!("invalid engine output: {error}"))
            })
        });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

/// JSON-shaped wrapper around a token engine, for registering it in the
/// `LocalEndpointRegistry` so [`RegistryEngine`] can reach it.
pub(crate) struct JsonEngine {
    inner: ExecutionContext,
}

impl JsonEngine {
    pub(crate) fn local(inner: ExecutionContext) -> LocalAsyncEngine {
        Arc::new(Self { inner })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<serde_json::Value>, ManyOut<Annotated<serde_json::Value>>, Error>
    for JsonEngine
{
    async fn generate(
        &self,
        input: SingleIn<serde_json::Value>,
    ) -> Result<ManyOut<Annotated<serde_json::Value>>, Error> {
        let (request, context) = input.into_parts();
        let request: PreprocessedRequest =
            serde_json::from_value(request).context("deserializing request")?;
        let stream = self.inner.generate(context.map(|_| request)).await?;
        let context = stream.context();
        let stream = stream.map(|annotated| {
            annotated.map_data(|output| serde_json::to_value(output).map_err(|e| e.to_string()))
        });
        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Echoes the request's tokens back as one output.
    struct EchoTokens;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for EchoTokens
    {
        async fn generate(
            &self,
            input: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let (request, context) = input.into_parts();
            let output = LLMEngineOutput {
                token_ids: request.token_ids.to_vec(),
                ..LLMEngineOutput::stop()
            };
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter([Annotated::from_data(output)])),
                context.context(),
            ))
        }
    }

    #[tokio::test]
    async fn registry_engine_round_trips_through_the_local_json_engine() {
        let registry = LocalEndpointRegistry::new();
        let engine = RegistryEngine {
            registry: registry.clone(),
            endpoint_name: "generate".into(),
        };
        let request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap();

        assert!(
            engine
                .generate(SingleIn::new(request.clone()))
                .await
                .is_err()
        );
        registry.register("generate".into(), JsonEngine::local(Arc::new(EchoTokens)));
        let outputs: Vec<_> = engine
            .generate(SingleIn::new(request))
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(outputs[0].data.as_ref().unwrap().token_ids, vec![1, 2, 3]);
    }

    #[cfg(feature = "integration")]
    #[tokio::test]
    async fn an_unservable_card_still_starts_the_gates_with_no_base_url() {
        use crate::model_type::ModelType;
        use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};

        let checks = dynamo_runtime::validation::testing::CheckServer::start().await;
        let gate = checks.url("pass");
        let drt = temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_PUBLISH_GATE", Some(gate.as_str())),
            ],
            async {
                DistributedRuntime::new(
                    Runtime::from_current().unwrap(),
                    DistributedConfig::process_local(),
                )
                .await
                .unwrap()
            },
        )
        .await;
        let endpoint = drt
            .namespace("ns")
            .unwrap()
            .component("c")
            .unwrap()
            .endpoint("generate");
        // A prefill worker's card: tokens in, no chat or completions surface.
        let mut card = ModelDeploymentCard::with_name_only("prefill");
        card.model_input = ModelInput::Tokens;
        card.model_type = ModelType::empty();

        // No local engine is ever registered for a card the endpoint can't
        // serve, so the gates must not wait for one.
        serve_if_gated(&endpoint, &card).await.unwrap();
        let validation = drt.worker_validation().unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            validation.wait_released(),
        )
        .await
        .unwrap_or_else(|_| panic!("{}", validation.status_json()));
        let (_, body) = &checks.requests()[0];
        assert_eq!(*body, serde_json::json!({"base_url": null}));
    }
}
