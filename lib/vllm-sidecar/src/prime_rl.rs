// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prime RL worker-admin routes forwarded to the engine's typed gRPC extension.

use std::sync::Arc;

use dynamo_runtime::engine_routes::EngineRouteCallback;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::client::PrimeRlClient;
use crate::proto::prime_rl as pb;

#[derive(Clone, Copy)]
pub(crate) enum Route {
    LivenessProbe,
    PauseGeneration,
    ResumeGeneration,
    FlushCache,
    AbortRequest,
    InitWeightsUpdateGroup,
    DestroyWeightsUpdateGroup,
    UpdateWeightsFromDisk,
    UpdateWeightsFromDistributed,
    GetWeightVersion,
}

pub(crate) const ROUTES: [Route; 10] = [
    Route::LivenessProbe,
    Route::PauseGeneration,
    Route::ResumeGeneration,
    Route::FlushCache,
    Route::AbortRequest,
    Route::InitWeightsUpdateGroup,
    Route::DestroyWeightsUpdateGroup,
    Route::UpdateWeightsFromDisk,
    Route::UpdateWeightsFromDistributed,
    Route::GetWeightVersion,
];

impl Route {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::LivenessProbe => "liveness_probe",
            Self::PauseGeneration => "pause_generation",
            Self::ResumeGeneration => "resume_generation",
            Self::FlushCache => "flush_cache",
            Self::AbortRequest => "abort_request",
            Self::InitWeightsUpdateGroup => "init_weights_update_group",
            Self::DestroyWeightsUpdateGroup => "destroy_weights_update_group",
            Self::UpdateWeightsFromDisk => "update_weights_from_disk",
            Self::UpdateWeightsFromDistributed => "update_weights_from_distributed",
            Self::GetWeightVersion => "get_weight_version",
        }
    }
}

pub(crate) fn callback(
    route: Route,
    client: PrimeRlClient,
    cancel: CancellationToken,
) -> EngineRouteCallback {
    Arc::new(move |body| {
        let mut client = client.clone();
        let cancel = cancel.clone();
        Box::pin(async move {
            if cancel.is_cancelled() {
                return Ok(json!({
                    "status": "error",
                    "message": "vllm sidecar is shutting down",
                }));
            }
            tokio::select! {
                _ = cancel.cancelled() => Ok(json!({
                    "status": "error",
                    "message": "vllm sidecar is shutting down",
                })),
                result = dispatch(&mut client, route, body) => Ok(match result {
                    Ok(response) => response,
                    Err(message) => json!({"status": "error", "message": message}),
                }),
            }
        })
    })
}

async fn dispatch(client: &mut PrimeRlClient, route: Route, body: Value) -> Result<Value, String> {
    let body = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    match route {
        Route::LivenessProbe => client
            .liveness_probe(pb::LivenessProbeRequest {})
            .await
            .map(|response| admin_json(response.into_inner(), Some(("alive", json!(true)))))
            .map_err(status_message),
        Route::PauseGeneration => client
            .pause_generation(pb::PauseGenerationRequest {
                mode: string(body, "mode", "wait")?,
                clear_cache: boolean(body, "clear_cache", false)?,
            })
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::ResumeGeneration => client
            .resume_generation(pb::ResumeGenerationRequest {})
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::FlushCache => client
            .flush_cache(pb::FlushCacheRequest {
                reset_running_requests: boolean(body, "reset_running_requests", false)?,
                reset_connector: boolean(body, "reset_connector", true)?,
            })
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::AbortRequest => client
            .abort_request(pb::AbortRequestRequest {
                request_id: required_string(body, "request_id")?,
            })
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::InitWeightsUpdateGroup => client
            .init_weights_update_group(pb::InitWeightsUpdateGroupRequest {
                host: required_string(body, "host")?,
                port: unsigned_u32(body, "port", None)?,
                rank_offset: unsigned_u32(body, "rank_offset", Some(0))?,
                inference_world_size: unsigned_u32(body, "inference_world_size", None)?,
                timeout: unsigned_u64(body, "timeout", Some(0))?,
                quantize_in_weight_transfer: boolean(body, "quantize_in_weight_transfer", false)?,
                engine_rpc: string(body, "engine_rpc", "init_broadcaster")?,
            })
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::DestroyWeightsUpdateGroup => client
            .destroy_weights_update_group(pb::DestroyWeightsUpdateGroupRequest {
                engine_rpc: string(body, "engine_rpc", "destroy_broadcaster")?,
            })
            .await
            .map(|response| admin_json(response.into_inner(), None))
            .map_err(status_message),
        Route::UpdateWeightsFromDisk => client
            .update_weights_from_disk(pb::UpdateWeightsFromDiskRequest {
                model_path: required_string(body, "model_path")?,
                weight_version: required_string(body, "weight_version")?,
                engine_rpc: string(body, "engine_rpc", "reload_weights")?,
            })
            .await
            .map(|response| weight_update_json(response.into_inner()))
            .map_err(status_message),
        Route::UpdateWeightsFromDistributed => client
            .update_weights_from_distributed(pb::UpdateWeightsFromDistributedRequest {
                weight_dir: string(body, "weight_dir", "")?,
                weight_version: required_string(body, "weight_version")?,
                engine_rpc: string(body, "engine_rpc", "update_weights_from_path")?,
                allow_unpaused: boolean(body, "allow_unpaused", false)?,
                reset_prefix_cache: boolean(body, "reset_prefix_cache", true)?,
            })
            .await
            .map(|response| weight_update_json(response.into_inner()))
            .map_err(status_message),
        Route::GetWeightVersion => client
            .get_weight_version(pb::GetWeightVersionRequest {})
            .await
            .map(|response| {
                let response = response.into_inner();
                json!({"status": response.status, "version": response.weight_version})
            })
            .map_err(status_message),
    }
}

fn admin_json(response: pb::AdminResponse, extra: Option<(&str, Value)>) -> Value {
    let mut value = Map::from_iter([
        ("status".to_string(), Value::String(response.status)),
        ("message".to_string(), Value::String(response.message)),
    ]);
    if let Some((key, extra)) = extra {
        value.insert(key.to_string(), extra);
    }
    Value::Object(value)
}

fn weight_update_json(response: pb::WeightUpdateResponse) -> Value {
    json!({
        "status": response.status,
        "message": response.message,
        "version": response.weight_version,
    })
}

fn status_message(status: tonic::Status) -> String {
    format!("{} ({:?})", status.message(), status.code())
}

fn required_string(body: &Map<String, Value>, key: &str) -> Result<String, String> {
    let value = string(body, key, "")?;
    if value.trim().is_empty() {
        Err(format!("Missing '{key}' in body"))
    } else {
        Ok(value)
    }
}

fn string(body: &Map<String, Value>, key: &str, default: &str) -> Result<String, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default.to_string()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(format!("'{key}' must be a string")),
    }
}

fn boolean(body: &Map<String, Value>, key: &str, default: bool) -> Result<bool, String> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("'{key}' must be a boolean")),
    }
}

fn unsigned_u32(body: &Map<String, Value>, key: &str, default: Option<u32>) -> Result<u32, String> {
    let value = unsigned_u64(body, key, default.map(u64::from))?;
    u32::try_from(value).map_err(|_| format!("'{key}' exceeds u32::MAX"))
}

fn unsigned_u64(body: &Map<String, Value>, key: &str, default: Option<u64>) -> Result<u64, String> {
    match body.get(key) {
        None | Some(Value::Null) => default.ok_or_else(|| format!("Missing '{key}' in body")),
        Some(Value::Number(value)) => value
            .as_u64()
            .ok_or_else(|| format!("'{key}' must be a non-negative integer")),
        Some(_) => Err(format!("'{key}' must be a non-negative integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_names_match_prime_rl_worker_contract() {
        let names = ROUTES.map(Route::name);
        assert!(names.contains(&"init_weights_update_group"));
        assert!(names.contains(&"pause_generation"));
        assert!(names.contains(&"resume_generation"));
        assert!(names.contains(&"update_weights_from_disk"));
        assert!(names.contains(&"update_weights_from_distributed"));
    }

    #[test]
    fn typed_body_helpers_reject_wrong_types() {
        let body = json!({"allow_unpaused": "false", "port": -1});
        let body = body.as_object().unwrap();
        assert!(boolean(body, "allow_unpaused", false).is_err());
        assert!(unsigned_u32(body, "port", None).is_err());
    }
}
