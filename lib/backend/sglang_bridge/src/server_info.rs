// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse the subset of SGLang's `GetServerInfo` / `GetModelInfo` JSON fields
//! the bridge needs to populate Dynamo's MDC.

#[derive(Debug, Default)]
pub struct SglangServerInfo {
    pub model_path: Option<String>,
    pub served_model_name: Option<String>,
    pub page_size: Option<u32>,
    pub max_running_requests: Option<u64>,
    pub max_prefill_tokens: Option<u64>,
    pub dp_size: Option<u32>,
    pub max_total_num_tokens: Option<u64>,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_host: Option<String>,
}

/// ServerArgs is nested under `server_args` on newer SGLang and at the top
/// level on older versions; we accept both.
pub fn parse_server_info(json_info: &str) -> SglangServerInfo {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json_info) else {
        tracing::warn!("GetServerInfo.json_info is not valid JSON");
        return SglangServerInfo::default();
    };
    let args = v.get("server_args").unwrap_or(&v);
    let nonempty_str = |key: &str, src: &serde_json::Value| {
        src.get(key)
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    SglangServerInfo {
        model_path: nonempty_str("model_path", args),
        served_model_name: nonempty_str("served_model_name", args),
        page_size: args.get("page_size").and_then(|n| n.as_u64()).map(|n| n as u32),
        max_running_requests: args.get("max_running_requests").and_then(|n| n.as_u64()),
        max_prefill_tokens: args.get("max_prefill_tokens").and_then(|n| n.as_u64()),
        dp_size: args.get("dp_size").and_then(|n| n.as_u64()).map(|n| n as u32),
        // scheduler_info lives at the top level alongside server_args.
        max_total_num_tokens: v.get("max_total_num_tokens").and_then(|n| n.as_u64()),
        bootstrap_port: args
            .get("disaggregation_bootstrap_port")
            .and_then(|n| n.as_u64())
            .map(|n| n as u16),
        bootstrap_host: nonempty_str("dist_init_addr", args).map(|addr| {
            addr.rsplit_once(':')
                .map(|(h, _)| h.to_string())
                .unwrap_or(addr)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_info_fields_and_layouts() {
        let info = parse_server_info(
            r#"{
              "server_args": {
                "model_path": "Qwen/Qwen3-0.6B",
                "served_model_name": "my-model",
                "page_size": 16,
                "max_running_requests": 256,
                "max_prefill_tokens": 16384,
                "dp_size": 2,
                "disaggregation_bootstrap_port": 8998,
                "dist_init_addr": "10.0.0.1:5555"
              },
              "max_total_num_tokens": 1048576
            }"#,
        );
        assert_eq!(info.model_path.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(info.served_model_name.as_deref(), Some("my-model"));
        assert_eq!(info.page_size, Some(16));
        assert_eq!(info.max_total_num_tokens, Some(1048576));
        assert_eq!(info.bootstrap_port, Some(8998));
        assert_eq!(info.bootstrap_host.as_deref(), Some("10.0.0.1"));

        // Top-level layout (older SGLang) + empty string → None.
        let info = parse_server_info(r#"{"model_path": "X", "served_model_name": ""}"#);
        assert_eq!(info.model_path.as_deref(), Some("X"));
        assert!(info.served_model_name.is_none());

        // Bad JSON → all-None.
        assert!(parse_server_info("not json").model_path.is_none());
    }
}
