use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_runtime::component::Endpoint;
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
const TRUE_ENV_VALUES: [&str; 4] = ["1", "true", "yes", "on"];
const RESERVED_ENDPOINTS: [&str; 3] = ["load_lora", "unload_lora", "list_loras"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RlWorkerMetadata {
    pub admin_base_url: String,
    pub world_size: u32,
}

pub(crate) fn enabled() -> bool {
    std::env::var("DYN_ENABLE_RL")
        .ok()
        .is_some_and(|value| TRUE_ENV_VALUES.contains(&value.trim().to_ascii_lowercase().as_str()))
}

pub(crate) struct RlServeEndpoint {
    pub endpoint: Endpoint,
    pub future: Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>,
}

pub(crate) fn serve_endpoint(
    primary: &Endpoint,
    metadata: RlWorkerMetadata,
) -> anyhow::Result<RlServeEndpoint> {
    if metadata.world_size == 0 {
        anyhow::bail!("RL worker world_size must be positive");
    }
    let endpoint_name = resolve_endpoint_name(&primary.id().name)?;
    let endpoint = primary.component().endpoint(endpoint_name);
    let system_url = self_host_base_url(primary.drt())?;
    let handler = Arc::new(RlRouteHandler {
        routes: primary.drt().engine_routes().clone(),
        metadata,
        system_url,
    });
    let ingress = Ingress::for_engine(handler.clone())?;
    let future = endpoint
        .endpoint_builder()
        .handler(ingress)
        .register_local_engine(handler)?
        .graceful_shutdown(true)
        .start();
    Ok(RlServeEndpoint {
        endpoint,
        future: Box::pin(future),
    })
}

fn self_host_base_url(drt: &dynamo_runtime::DistributedRuntime) -> anyhow::Result<Option<String>> {
    let Some(info) = drt.system_status_server_info() else {
        return Ok(None);
    };
    let configured = dynamo_runtime::RuntimeConfig::from_settings()
        .unwrap_or_default()
        .system_host;
    let host = match configured.as_str() {
        "0.0.0.0" | "::" | "[::]" => dynamo_runtime::utils::local_ip_for_advertise(),
        _ => configured,
    };
    Ok(Some(format!("http://{host}:{}", info.port())))
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
    if endpoint_name == primary_name || RESERVED_ENDPOINTS.contains(&endpoint_name) {
        anyhow::bail!(
            "DYN_RL_ENDPOINT `{endpoint_name}` collides with an existing worker endpoint"
        );
    }
    Ok(endpoint_name.to_string())
}

struct RlRouteHandler {
    routes: EngineRouteRegistry,
    metadata: RlWorkerMetadata,
    system_url: Option<String>,
}

impl RlRouteHandler {
    fn dispatch(&self, request: &Value) -> Value {
        let Some(request) = request.as_object() else {
            return json!({"status": "error", "message": "rl_dispatch: request required"});
        };
        let Some(method) = request
            .get("method")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
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
        routes.retain(|route| {
            !route.contains('/')
                || matches!(
                    route.as_str(),
                    "update/load_lora" | "update/unload_lora" | "update/list_loras"
                )
        });
        routes.sort();
        routes.dedup();
        json!({
            "status": "ok",
            "routes": routes,
            "system_url": self.system_url,
            "admin_base_url": self.metadata.admin_base_url,
            "world_size": self.metadata.world_size,
        })
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

    fn handler() -> RlRouteHandler {
        let routes = EngineRouteRegistry::new();
        routes.register(
            "update/load_lora",
            Arc::new(|_| Box::pin(async { Ok(json!({"status": "ok"})) })),
        );
        RlRouteHandler {
            routes,
            metadata: RlWorkerMetadata {
                admin_base_url: "http://worker:8120".to_string(),
                world_size: 2,
            },
            system_url: Some("http://worker:8181".to_string()),
        }
    }

    #[test]
    fn routes_request_describes_direct_vllm_admin_surface() {
        assert_eq!(
            handler().dispatch(&json!({"method": "routes"})),
            json!({
                "status": "ok",
                "routes": ["update/load_lora"],
                "system_url": "http://worker:8181",
                "admin_base_url": "http://worker:8120",
                "world_size": 2,
            })
        );
    }

    #[test]
    fn endpoint_name_rejects_primary_and_control_collisions() {
        assert!(validate_endpoint_name("", "generate").is_err());
        assert!(validate_endpoint_name("generate", "generate").is_err());
        assert!(validate_endpoint_name("load_lora", "generate").is_err());
        assert_eq!(validate_endpoint_name("rl", "generate").unwrap(), "rl");
    }
}
