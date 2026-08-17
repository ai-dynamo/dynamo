// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{num::NonZeroU32, sync::Arc};

use async_trait::async_trait;
use dynamo_runtime::component::{Endpoint, StartedEndpoint};
use dynamo_runtime::engine_routes::EngineRouteRegistry;
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use futures::stream;
use serde_json::{Value, json};

const DEFAULT_RL_ENDPOINT: &str = "rl";

pub(crate) struct RlServeEndpoint {
    started: StartedEndpoint,
}

pub(crate) struct RlEndpointConfig {
    endpoint_name: String,
    system_url: String,
    metadata: Option<RlWorkerMetadata>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RlWorkerMetadata {
    world_size: NonZeroU32,
    weight_transfer_backend: Option<String>,
    admin_base_url: Option<String>,
}

impl RlWorkerMetadata {
    pub fn new(
        world_size: u32,
        weight_transfer_backend: Option<String>,
        admin_base_url: Option<String>,
    ) -> anyhow::Result<Self> {
        let world_size = NonZeroU32::new(world_size)
            .ok_or_else(|| anyhow::anyhow!("RL worker world size must be positive"))?;
        let weight_transfer_backend = normalized_optional(
            weight_transfer_backend,
            "RL weight-transfer backend must not be blank",
        )?;
        let admin_base_url =
            normalized_optional(admin_base_url, "RL admin base URL must not be blank")?;
        Ok(Self {
            world_size,
            weight_transfer_backend,
            admin_base_url,
        })
    }
}

fn normalized_optional(value: Option<String>, error: &str) -> anyhow::Result<Option<String>> {
    value
        .map(|value| {
            let value = value.trim();
            if value.is_empty() {
                anyhow::bail!(error.to_string());
            }
            Ok(value.to_string())
        })
        .transpose()
}

impl RlServeEndpoint {
    pub(crate) async fn shutdown(self) -> anyhow::Result<()> {
        self.started.shutdown().await
    }
}

pub(crate) fn prepare_endpoint(
    primary: &Endpoint,
    metadata: Option<RlWorkerMetadata>,
) -> anyhow::Result<RlEndpointConfig> {
    let endpoint_name = resolve_endpoint_name(&primary.id().name)?;
    let system_url = self_host_base_url(primary.drt()).ok_or_else(|| {
        anyhow::anyhow!(
            "RL discovery requires the Dynamo system server; set DYN_SYSTEM_PORT to 0 or a positive port"
        )
    })?;
    Ok(RlEndpointConfig {
        endpoint_name,
        system_url,
        metadata,
    })
}

pub(crate) async fn serve_endpoint(
    primary: &Endpoint,
    config: RlEndpointConfig,
) -> anyhow::Result<RlServeEndpoint> {
    let endpoint = primary.component().endpoint(config.endpoint_name);
    let handler = Arc::new(RlRouteHandler {
        routes: primary.drt().engine_routes().clone(),
        system_url: config.system_url,
        metadata: config.metadata,
    });
    let ingress = Ingress::for_engine(handler)?;
    let started = endpoint
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start_with_registration()
        .await?;
    Ok(RlServeEndpoint { started })
}

fn self_host_base_url(drt: &dynamo_runtime::DistributedRuntime) -> Option<String> {
    let info = drt.system_status_server_info()?;
    let socket_addr = info.socket_addr;
    if socket_addr.ip().is_unspecified() {
        let host = dynamo_runtime::utils::local_ip_for_advertise();
        Some(format!("http://{host}:{}", socket_addr.port()))
    } else {
        Some(format!("http://{socket_addr}"))
    }
}

fn resolve_endpoint_name(primary_name: &str) -> anyhow::Result<String> {
    let endpoint_name =
        std::env::var("DYN_RL_ENDPOINT").unwrap_or_else(|_| DEFAULT_RL_ENDPOINT.into());
    validate_endpoint_name(endpoint_name.trim(), primary_name)
}

fn validate_endpoint_name(endpoint_name: &str, primary_name: &str) -> anyhow::Result<String> {
    if endpoint_name.is_empty() {
        anyhow::bail!("DYN_RL_ENDPOINT must not be empty");
    }
    if endpoint_name == primary_name {
        anyhow::bail!("DYN_RL_ENDPOINT `{endpoint_name}` collides with the serving endpoint");
    }
    if !endpoint_name
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        anyhow::bail!("DYN_RL_ENDPOINT must contain only letters, digits, '-' or '_'");
    }
    Ok(endpoint_name.to_string())
}

struct RlRouteHandler {
    routes: EngineRouteRegistry,
    system_url: String,
    metadata: Option<RlWorkerMetadata>,
}

impl RlRouteHandler {
    fn dispatch(&self, request: &Value) -> Value {
        let Some(method) = request
            .as_object()
            .and_then(|request| request.get("method"))
            .and_then(Value::as_str)
        else {
            return json!({"status": "error", "message": "rl_dispatch: missing 'method' (str)"});
        };
        if method != "routes" {
            return json!({
                "status": "error",
                "method": method,
                "message": "rl request-plane endpoint only supports method='routes'",
            });
        }

        let mut routes = self.routes.routes().into_iter().collect::<Vec<_>>();
        routes.sort();
        routes.dedup();
        let mut response = json!({
            "status": "ok",
            "routes": routes,
            "system_url": self.system_url,
        });
        if let Some(metadata) = &self.metadata {
            response["world_size"] = json!(metadata.world_size.get());
            if let Some(backend) = &metadata.weight_transfer_backend {
                response["weight_transfer_backend"] = json!(backend);
            }
            if let Some(url) = &metadata.admin_base_url {
                response["admin_base_url"] = json!(url);
            }
        }
        response
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<Value>, ManyOut<Annotated<Value>>, anyhow::Error> for RlRouteHandler {
    async fn generate(&self, input: SingleIn<Value>) -> anyhow::Result<ManyOut<Annotated<Value>>> {
        let (request, context) = input.into_parts();
        let response = self.dispatch(&request);
        Ok(ResponseStream::new(
            Box::pin(stream::once(async move { Annotated::from_data(response) })),
            context.context(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rl_dispatch_only_describes_the_worker_engine_surface() {
        let routes = EngineRouteRegistry::new();
        routes.register(
            "control/pause_generation",
            Arc::new(|_| Box::pin(async { Ok(json!({"status": "ok"})) })),
        );
        let handler = RlRouteHandler {
            routes,
            system_url: "http://worker:8080".to_string(),
            metadata: Some(
                RlWorkerMetadata::new(
                    4,
                    Some(" nccl ".to_string()),
                    Some(" http://worker:8120 ".to_string()),
                )
                .expect("valid metadata"),
            ),
        };

        assert_eq!(
            handler.dispatch(&json!({"method": "routes"})),
            json!({
                "status": "ok",
                "routes": ["control/pause_generation"],
                "system_url": "http://worker:8080",
                "admin_base_url": "http://worker:8120",
                "world_size": 4,
                "weight_transfer_backend": "nccl",
            })
        );
        assert_eq!(
            handler.dispatch(&json!({"method": "control/pause_generation"}))["status"],
            "error"
        );
    }

    #[test]
    fn rl_worker_metadata_rejects_invalid_values() {
        assert!(RlWorkerMetadata::new(0, None, None).is_err());
        assert!(RlWorkerMetadata::new(1, Some("   ".to_string()), None).is_err());
        assert!(RlWorkerMetadata::new(1, None, Some("   ".to_string())).is_err());
    }
}
