use super::{RouteDoc, service_v2};
use anyhow::anyhow;
use axum::{Json, Router, http::Method, http::StatusCode, response::IntoResponse, routing::get};
use dynamo_runtime::{
    component::Instance,
    instances::list_all_instances,
    pipeline::{AsyncEngine, Context, PushRouter, RouterMode},
    protocols::maybe_error::MaybeError,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConfigInstance {
    instance: Instance,
    config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DumpConfigRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DumpConfigResponse(serde_json::Value);

impl MaybeError for DumpConfigResponse {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self(json!({
            "error": format!("{:?}", err)
        }))
    }

    fn err(&self) -> Option<anyhow::Error> {
        // Only return an error if the response contains an "error" field or "status": "error"
        if let Some(error_msg) = self.0.get("error") {
            return Some(anyhow!("Config dump error: {}", error_msg));
        }
        if let Some(status) = self.0.get("status") {
            if status == "error" {
                let message = self
                    .0
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error");
                return Some(anyhow!("Config dump failed: {}", message));
            }
        }
        None
    }
}

pub fn dump_config_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let config_path = path.unwrap_or_else(|| "/dump_config".to_string());
    let docs: Vec<RouteDoc> = vec![RouteDoc::new(Method::GET, &config_path)];
    let router = Router::new()
        .route(&config_path, get(get_config_handler))
        .with_state(state);
    (docs, router)
}

async fn get_config_handler_inner(
    state: Arc<service_v2::State>,
) -> Result<Json<serde_json::Value>, String> {
    let etcd_client = state.etcd_client().ok_or("No etcd client found")?;

    let instances = list_all_instances(etcd_client)
        .await
        .map_err(|e| e.to_string())?;

    if instances.is_empty() {
        return Ok(Json(json!({
            "message": "No active instances found"
        })));
    }

    let drt = state.drt().ok_or("No distributed runtime available")?;
    let mut configs = Vec::new();

    for instance in instances {
        // Skip non-dump_config endpoints
        if instance.endpoint != "dump_config" {
            continue;
        }

        tracing::debug!(
            "Fetching config from instance: namespace={}, component={}, endpoint={}, id={}",
            instance.namespace,
            instance.component,
            instance.endpoint,
            instance.instance_id
        );

        match fetch_instance_config(drt, &instance).await {
            Ok(config) => {
                configs.push(ConfigInstance {
                    instance: instance.clone(),
                    config,
                });
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to fetch config from instance {}: {}",
                    instance.instance_id,
                    e
                );
                // Continue with other instances even if one fails
                configs.push(ConfigInstance {
                    instance: instance.clone(),
                    config: json!({
                        "error": format!("Failed to fetch config: {}", e)
                    }),
                });
            }
        }
    }

    Ok(Json(json!(configs)))
}

async fn fetch_instance_config(
    drt: &dynamo_runtime::DistributedRuntime,
    instance: &Instance,
) -> Result<Value, String> {
    // Create an endpoint for this specific instance's dump_config endpoint
    let endpoint = drt
        .namespace(&instance.namespace)
        .map_err(|e| format!("Failed to create namespace: {}", e))?
        .component(&instance.component)
        .map_err(|e| format!("Failed to create component: {}", e))?
        .endpoint(&instance.endpoint);

    // Create a client for this endpoint
    let client = endpoint
        .client()
        .await
        .map_err(|e| format!("Failed to create client: {}", e))?;

    // TODO: this is very hacky and needs to be improved, should I be tracking all the endpoints as they come up?
    // Wait for the client to discover instances from etcd
    client
        .wait_for_instances()
        .await
        .map_err(|e| format!("Failed to wait for instances: {}", e))?;

    // Additional wait: Give the background monitor_instance_source task time to populate instance_avail
    // The Client spawns a background task that updates instance_avail from instance_source,
    // but it runs asynchronously. We need to ensure it has run at least once.
    let max_retries = 50; // 50 * 10ms = 500ms max wait
    for _ in 0..max_retries {
        let avail_ids = client.instance_ids_avail();
        if avail_ids.contains(&instance.instance_id) {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Final check: ensure the instance is available
    let avail_ids = client.instance_ids_avail();
    if !avail_ids.contains(&instance.instance_id) {
        return Err(format!(
            "Instance {} not found in available instances after waiting. Available: {:?}",
            instance.instance_id,
            avail_ids.as_ref()
        ));
    }

    // Create a router that targets this specific instance
    let router: PushRouter<DumpConfigRequest, DumpConfigResponse> =
        PushRouter::from_client(client, RouterMode::Direct(instance.instance_id))
            .await
            .map_err(|e| format!("Failed to create router: {}", e))?;

    // Create the request
    let request = Context::new(DumpConfigRequest {});

    // Call the endpoint
    let mut stream = router
        .generate(request)
        .await
        .map_err(|e| format!("Failed to generate request: {}", e))?;

    // Collect the response (dump_config should return a single response)
    let mut responses = Vec::new();
    while let Some(response) = stream.next().await {
        responses.push(response.0);
    }

    // Get the first response or error if empty
    let mut response = responses
        .into_iter()
        .next()
        .ok_or_else(|| "No response received".to_string())?;
    // Should be of the format {"data": {"message": "json_string"}}
    // I'm not sure why I can't nest more than one level, but when
    // I do, it passes through weird json
    let message = response
        .get_mut("data")
        .map(|v| v.take())
        .ok_or_else(|| format!("No data field in response {:?}", response))?
        .get_mut("message")
        .map(|v| v.take())
        .ok_or_else(|| format!("No message field in response {:?}", response))?;
    if let Some(message_str) = message.as_str() {
        // The message is itself a json string
        tracing::warn!("message: {}", message_str);
        let message_json = serde_json::from_str(message_str)
            .map_err(|e| format!("Failed to parse message as json: {}", e))?;
        Ok(message_json)
    } else {
        Err(format!("message field is not a string {:?}", response))
    }
}

async fn get_config_handler(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
) -> impl IntoResponse {
    match get_config_handler_inner(state).await {
        Ok(response) => (StatusCode::OK, response),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "message": error
            })),
        ),
    }
}
