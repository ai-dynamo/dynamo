// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-control handlers for the native vLLM backend.

use dynamo_backend_common::DynamoError;
use serde::Deserialize;
use vllm_engine_core_client::protocol::utility::PauseMode;
use vllm_engine_core_client::EngineCoreClient;

const SUPPORTED_CONTROLS: [&str; 3] = ["sleep", "wake_up", "reset_prefix_cache"];

pub(crate) fn supported_controls() -> Vec<String> {
    SUPPORTED_CONTROLS
        .iter()
        .map(|control| control.to_string())
        .collect()
}

pub(crate) fn is_supported(control: &str) -> bool {
    SUPPORTED_CONTROLS.contains(&control)
}

pub(crate) async fn engine_control(
    client: &EngineCoreClient,
    control: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, DynamoError> {
    let body = if body.is_null() {
        serde_json::json!({})
    } else {
        body
    };

    match control.as_str() {
        "sleep" => sleep(client, body).await,
        "wake_up" => wake_up(client, body).await,
        "reset_prefix_cache" => reset_prefix_cache(client, body).await,
        _ => Ok(unsupported_response(control)),
    }
}

async fn sleep(
    client: &EngineCoreClient,
    body: serde_json::Value,
) -> Result<serde_json::Value, DynamoError> {
    #[derive(Debug, Deserialize)]
    struct SleepBody {
        level: Option<u32>,
        mode: Option<String>,
    }

    let body = match serde_json::from_value::<SleepBody>(body) {
        Ok(body) => body,
        Err(error) => {
            return Ok(error_response(format!(
                "invalid sleep control body: {error}"
            )));
        }
    };
    let level = body.level.unwrap_or(1);
    let mode = body
        .mode
        .as_deref()
        .unwrap_or("abort")
        .parse::<PauseMode>()
        .unwrap_or(PauseMode::Abort);

    match client.sleep(level, mode).await {
        Ok(()) => Ok(serde_json::json!({
            "status": "ok",
            "message": format!("Engine slept (level={level})"),
        })),
        Err(error) => Ok(error_response(format!("failed to sleep engine: {error}"))),
    }
}

async fn wake_up(
    client: &EngineCoreClient,
    body: serde_json::Value,
) -> Result<serde_json::Value, DynamoError> {
    #[derive(Debug, Deserialize)]
    struct WakeUpBody {
        tags: Option<Vec<String>>,
    }

    let body = match serde_json::from_value::<WakeUpBody>(body) {
        Ok(body) => body,
        Err(error) => {
            return Ok(error_response(format!(
                "invalid wake_up control body: {error}"
            )));
        }
    };

    match client.wake_up(body.tags).await {
        Ok(()) => Ok(serde_json::json!({
            "status": "ok",
            "message": "Engine woke",
        })),
        Err(error) => Ok(error_response(format!("failed to wake up engine: {error}"))),
    }
}

async fn reset_prefix_cache(
    client: &EngineCoreClient,
    body: serde_json::Value,
) -> Result<serde_json::Value, DynamoError> {
    #[derive(Debug, Deserialize)]
    struct ResetPrefixCacheBody {
        #[serde(default)]
        reset_running_requests: bool,
        #[serde(default, alias = "reset_connector")]
        reset_external: bool,
    }

    let body = match serde_json::from_value::<ResetPrefixCacheBody>(body) {
        Ok(body) => body,
        Err(error) => {
            return Ok(error_response(format!(
                "invalid reset_prefix_cache control body: {error}"
            )));
        }
    };

    match client
        .reset_prefix_cache(body.reset_running_requests, body.reset_external)
        .await
    {
        Ok(true) => Ok(serde_json::json!({
            "status": "ok",
            "message": "Prefix cache reset",
        })),
        Ok(false) => Ok(error_response(
            "failed to reset prefix cache on at least one engine",
        )),
        Err(error) => Ok(error_response(format!(
            "failed to reset prefix cache: {error}"
        ))),
    }
}

pub(crate) fn unsupported_response(control: impl std::fmt::Display) -> serde_json::Value {
    error_response(format!("unsupported engine control: {control}"))
}

pub(crate) fn error_response(message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({"status": "error", "message": message.into()})
}

#[cfg(test)]
mod tests {
    use crate::backend::VllmBackend;

    #[tokio::test]
    async fn supported_controls_advertise_typed_engine_core_controls() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
        ]))
        .unwrap();

        let controls = dynamo_backend_common::LLMEngine::supported_controls(&engine)
            .await
            .unwrap();

        assert_eq!(controls, vec!["sleep", "wake_up", "reset_prefix_cache"]);
    }

    #[tokio::test]
    async fn unsupported_engine_control_returns_json_error() {
        let (engine, _config) = VllmBackend::from_args(Some(vec![
            "dynamo-vllm-rs-backend".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
        ]))
        .unwrap();

        let response = dynamo_backend_common::LLMEngine::engine_control(
            &engine,
            "start_profile".to_string(),
            serde_json::json!({}),
        )
        .await
        .unwrap();

        assert_eq!(response["status"], "error");
        assert_eq!(
            response["message"],
            "unsupported engine control: start_profile"
        );
    }
}
