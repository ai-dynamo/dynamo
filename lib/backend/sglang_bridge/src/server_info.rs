// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse the subset of SGLang's `GetServerInfo` / `GetModelInfo` JSON fields
//! the bridge needs to populate Dynamo's MDC. Targets SGLang 0.5.12+: the
//! response is a flat JSON of `dataclasses.asdict(ServerArgs) | scheduler_info`.

use dynamo_backend_common::DisaggregationMode;

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
    /// SGLang's own `--disaggregation-mode`. `"null"` (the SGLang default)
    /// and missing both map to `Aggregated`.
    pub disaggregation_mode: Option<DisaggregationMode>,
    /// Parsed `--kv-events-config`; `None` when SGLang was launched
    /// without zmq KV events.
    pub kv_events: Option<KvEventsConfig>,
}

/// Subset of SGLang's `KVEventsConfig` the bridge cares about. DP rank N
/// connects at `base_port + N` (see [`offset_endpoint_port`]).
#[derive(Debug, Clone)]
pub struct KvEventsConfig {
    pub endpoint: String,
    pub topic: String,
}

pub fn parse_server_info(json_info: &str) -> SglangServerInfo {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json_info) else {
        tracing::warn!("GetServerInfo.json_info is not valid JSON");
        return SglangServerInfo::default();
    };
    let nonempty_str = |key: &str| {
        v.get(key)
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let u32_field = |key: &str| {
        v.get(key)
            .and_then(|n| n.as_u64())
            .and_then(|n| u32::try_from(n).ok())
    };
    let u64_field = |key: &str| v.get(key).and_then(|n| n.as_u64());
    SglangServerInfo {
        model_path: nonempty_str("model_path"),
        served_model_name: nonempty_str("served_model_name"),
        page_size: u32_field("page_size"),
        max_running_requests: u64_field("max_running_requests"),
        max_prefill_tokens: u64_field("max_prefill_tokens"),
        dp_size: u32_field("dp_size"),
        max_total_num_tokens: u64_field("max_total_num_tokens"),
        bootstrap_port: v
            .get("disaggregation_bootstrap_port")
            .and_then(|n| n.as_u64())
            .and_then(|n| u16::try_from(n).ok()),
        bootstrap_host: nonempty_str("dist_init_addr").map(|addr| {
            addr.rsplit_once(':')
                .map(|(h, _)| h.to_string())
                .unwrap_or(addr)
        }),
        disaggregation_mode: v.get("disaggregation_mode").and_then(|v| match v.as_str()? {
            "prefill" => Some(DisaggregationMode::Prefill),
            "decode" => Some(DisaggregationMode::Decode),
            "null" | "" => Some(DisaggregationMode::Aggregated),
            _ => None,
        }),
        kv_events: v
            .get("kv_events_config")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .and_then(parse_kv_events_config),
    }
}

/// Returns `None` for the SGLang default (`publisher: "null"`) or any
/// malformed input — treat "no usable config" the same as "feature off"
/// rather than failing start.
fn parse_kv_events_config(raw: &str) -> Option<KvEventsConfig> {
    let v: serde_json::Value = serde_json::from_str(raw)
        .inspect_err(|e| tracing::warn!(error = %e, raw, "kv_events_config invalid JSON"))
        .ok()?;
    let publisher = v.get("publisher").and_then(|p| p.as_str())?;
    if publisher != "zmq" {
        tracing::info!(publisher, "kv_events publisher != zmq; KV routing disabled");
        return None;
    }
    Some(KvEventsConfig {
        endpoint: v.get("endpoint").and_then(|e| e.as_str())?.to_string(),
        topic: v
            .get("topic")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string(),
    })
}

/// Mirror of SGLang's `ZmqEventPublisher.offset_endpoint_port`. DP rank N
/// connects at `base_port + N`. `tcp://*:5557` + rank 1 → `tcp://*:5558`.
/// The bridge always shares a pod with SGLang, so we only need `tcp://`.
pub(crate) fn offset_endpoint_port(endpoint: &str, dp_rank: u32) -> Option<String> {
    if dp_rank == 0 {
        return Some(endpoint.to_string());
    }
    if !endpoint.starts_with("tcp://") {
        return None;
    }
    let (base, port) = endpoint.rsplit_once(':')?;
    let port: u16 = port.parse().ok()?;
    Some(format!("{base}:{}", port as u32 + dp_rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_server_info_fields() {
        let info = parse_server_info(
            r#"{
              "model_path": "Qwen/Qwen3-0.6B",
              "served_model_name": "my-model",
              "page_size": 16,
              "max_running_requests": 256,
              "max_prefill_tokens": 16384,
              "dp_size": 2,
              "disaggregation_bootstrap_port": 8998,
              "dist_init_addr": "10.0.0.1:5555",
              "disaggregation_mode": "prefill",
              "max_total_num_tokens": 1048576
            }"#,
        );
        assert_eq!(info.model_path.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(info.served_model_name.as_deref(), Some("my-model"));
        assert_eq!(info.page_size, Some(16));
        assert_eq!(info.max_total_num_tokens, Some(1048576));
        assert_eq!(info.bootstrap_port, Some(8998));
        assert_eq!(info.bootstrap_host.as_deref(), Some("10.0.0.1"));
        assert!(matches!(info.disaggregation_mode, Some(DisaggregationMode::Prefill)));

        // Empty string served name → None.
        let info = parse_server_info(r#"{"model_path": "X", "served_model_name": ""}"#);
        assert_eq!(info.model_path.as_deref(), Some("X"));
        assert!(info.served_model_name.is_none());

        // "null" mode (SGLang default) → Aggregated.
        let info = parse_server_info(r#"{"disaggregation_mode": "null"}"#);
        assert!(matches!(info.disaggregation_mode, Some(DisaggregationMode::Aggregated)));

        // Out-of-range u32 silently drops (rather than wrapping).
        let info = parse_server_info(r#"{"page_size": 4294967296}"#);
        assert!(info.page_size.is_none());

        // Bad JSON → all-None.
        assert!(parse_server_info("not json").model_path.is_none());
    }

    #[test]
    fn parses_kv_events_config_when_publisher_is_zmq() {
        let info = parse_server_info(
            r#"{"kv_events_config": "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:5557\"}"}"#,
        );
        let kv = info.kv_events.expect("kv_events parsed");
        assert_eq!(kv.endpoint, "tcp://*:5557");
        assert_eq!(kv.topic, "kv-events");
    }

    #[test]
    fn kv_events_disabled_for_null_publisher_or_missing() {
        // publisher: null is SGLang's default → feature off
        let info = parse_server_info(r#"{"kv_events_config": "{\"publisher\":\"null\"}"}"#);
        assert!(info.kv_events.is_none());

        // missing field → off
        let info = parse_server_info(r#"{"model_path": "X"}"#);
        assert!(info.kv_events.is_none());

        // empty string → off
        let info = parse_server_info(r#"{"kv_events_config": ""}"#);
        assert!(info.kv_events.is_none());

        // malformed JSON-inside-JSON → off, not panic
        let info = parse_server_info(r#"{"kv_events_config": "not json"}"#);
        assert!(info.kv_events.is_none());
    }

    #[test]
    fn offset_endpoint_port_matches_sglang() {
        assert_eq!(offset_endpoint_port("tcp://*:5557", 0).as_deref(), Some("tcp://*:5557"));
        assert_eq!(offset_endpoint_port("tcp://*:5557", 1).as_deref(), Some("tcp://*:5558"));
        assert_eq!(offset_endpoint_port("tcp://*:5557", 4).as_deref(), Some("tcp://*:5561"));
        assert!(offset_endpoint_port("udp://x:5557", 1).is_none());
        assert!(offset_endpoint_port("tcp://*:notaport", 1).is_none());
    }
}
