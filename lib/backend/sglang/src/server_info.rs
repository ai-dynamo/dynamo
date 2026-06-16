// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse the subset of SMG's `GetServerInfo` fields the backend needs to
//! populate Dynamo's MDC.

use dynamo_backend_common::DisaggregationMode;
use prost_types::{Struct, Value, value::Kind};

use crate::proto::scheduler::GetServerInfoResponse;

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

/// Subset of SGLang's `KVEventsConfig` the backend cares about. DP rank N
/// connects at `base_port + N` (see [`offset_endpoint_port`]).
#[derive(Debug, Clone)]
pub struct KvEventsConfig {
    pub endpoint: String,
    pub topic: String,
}

pub fn parse_server_info(resp: &GetServerInfoResponse) -> SglangServerInfo {
    let server_args = resp.server_args.as_ref();
    let scheduler_info = resp.scheduler_info.as_ref();

    let nonempty_str =
        |key: &str| struct_str(server_args, key).or_else(|| struct_str(scheduler_info, key));
    let u32_field = |key: &str| {
        struct_u64(server_args, key)
            .or_else(|| struct_u64(scheduler_info, key))
            .and_then(|n| u32::try_from(n).ok())
    };
    let u64_field =
        |key: &str| struct_u64(server_args, key).or_else(|| struct_u64(scheduler_info, key));

    let max_total_num_tokens = if resp.max_total_num_tokens > 0 {
        Some(resp.max_total_num_tokens as u64)
    } else {
        u64_field("max_total_num_tokens")
    };

    SglangServerInfo {
        model_path: nonempty_str("model_path"),
        served_model_name: nonempty_str("served_model_name"),
        page_size: u32_field("page_size"),
        max_running_requests: u64_field("max_running_requests"),
        max_prefill_tokens: u64_field("max_prefill_tokens"),
        dp_size: u32_field("dp_size"),
        max_total_num_tokens,
        bootstrap_port: u64_field("disaggregation_bootstrap_port")
            .and_then(|n| u16::try_from(n).ok()),
        bootstrap_host: nonempty_str("dist_init_addr").map(|addr| {
            addr.rsplit_once(':')
                .map(|(h, _)| h.to_string())
                .unwrap_or(addr)
        }),
        disaggregation_mode: struct_str(server_args, "disaggregation_mode").and_then(|mode| {
            match mode.as_str() {
                "prefill" => Some(DisaggregationMode::Prefill),
                "decode" => Some(DisaggregationMode::Decode),
                "null" | "" => Some(DisaggregationMode::Aggregated),
                _ => None,
            }
        }),
        kv_events: struct_value(server_args, "kv_events_config").and_then(parse_kv_events_value),
    }
}

fn struct_value<'a>(s: Option<&'a Struct>, key: &str) -> Option<&'a Value> {
    s?.fields.get(key)
}

fn struct_str(s: Option<&Struct>, key: &str) -> Option<String> {
    let value = struct_value(s, key)?;
    match value.kind.as_ref()? {
        Kind::StringValue(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

fn struct_u64(s: Option<&Struct>, key: &str) -> Option<u64> {
    let value = struct_value(s, key)?;
    match value.kind.as_ref()? {
        Kind::NumberValue(n) if n.is_finite() && *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
        Kind::StringValue(s) => s.parse().ok(),
        _ => None,
    }
}

fn parse_kv_events_value(value: &Value) -> Option<KvEventsConfig> {
    match value.kind.as_ref()? {
        Kind::StringValue(raw) if !raw.is_empty() => parse_kv_events_config(raw),
        Kind::StructValue(s) => parse_kv_events_struct(s),
        _ => None,
    }
}

fn parse_kv_events_struct(s: &Struct) -> Option<KvEventsConfig> {
    let publisher = struct_str(Some(s), "publisher")?;
    if publisher != "zmq" {
        tracing::info!(publisher, "kv_events publisher != zmq; KV routing disabled");
        return None;
    }
    let endpoint = struct_str(Some(s), "endpoint")?;
    let topic = struct_str(Some(s), "topic").unwrap_or_default();
    Some(KvEventsConfig { endpoint, topic })
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
    let endpoint = v
        .get("endpoint")
        .and_then(|e| e.as_str())
        .filter(|s| !s.is_empty())?
        .to_string();
    let topic = v
        .get("topic")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string();
    Some(KvEventsConfig { endpoint, topic })
}

/// Mirror of SGLang's `ZmqEventPublisher.offset_endpoint_port`. DP rank N
/// connects at `base_port + N`. `tcp://*:5557` + rank 1 → `tcp://*:5558`.
/// This backend always shares a pod with SGLang, so we only need `tcp://`.
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

    fn server_info(
        server_args: impl IntoIterator<Item = (&'static str, Value)>,
        scheduler_info: impl IntoIterator<Item = (&'static str, Value)>,
    ) -> GetServerInfoResponse {
        GetServerInfoResponse {
            server_args: Some(Struct {
                fields: server_args
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            }),
            scheduler_info: Some(Struct {
                fields: scheduler_info
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            }),
            ..Default::default()
        }
    }

    fn str_value(s: &str) -> Value {
        Value {
            kind: Some(Kind::StringValue(s.to_string())),
        }
    }

    fn num_value(n: u64) -> Value {
        Value {
            kind: Some(Kind::NumberValue(n as f64)),
        }
    }

    fn proto_struct_value(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
        Value {
            kind: Some(Kind::StructValue(Struct {
                fields: fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            })),
        }
    }

    #[test]
    fn parses_server_info_fields() {
        let mut resp = server_info(
            [
                ("model_path", str_value("Qwen/Qwen3-0.6B")),
                ("served_model_name", str_value("my-model")),
                ("page_size", num_value(16)),
                ("max_running_requests", num_value(256)),
                ("max_prefill_tokens", num_value(16384)),
                ("dp_size", num_value(2)),
                ("disaggregation_bootstrap_port", num_value(8998)),
                ("dist_init_addr", str_value("10.0.0.1:5555")),
                ("disaggregation_mode", str_value("prefill")),
            ],
            [("max_total_num_tokens", num_value(1048576))],
        );
        let info = parse_server_info(&resp);
        assert_eq!(info.model_path.as_deref(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(info.served_model_name.as_deref(), Some("my-model"));
        assert_eq!(info.page_size, Some(16));
        assert_eq!(info.max_total_num_tokens, Some(1048576));
        assert_eq!(info.bootstrap_port, Some(8998));
        assert_eq!(info.bootstrap_host.as_deref(), Some("10.0.0.1"));
        assert!(matches!(
            info.disaggregation_mode,
            Some(DisaggregationMode::Prefill)
        ));

        // Empty string served name → None.
        let info = parse_server_info(&server_info(
            [
                ("model_path", str_value("X")),
                ("served_model_name", str_value("")),
            ],
            [],
        ));
        assert_eq!(info.model_path.as_deref(), Some("X"));
        assert!(info.served_model_name.is_none());

        // "null" mode (SGLang default) → Aggregated.
        let info = parse_server_info(&server_info(
            [("disaggregation_mode", str_value("null"))],
            [],
        ));
        assert!(matches!(
            info.disaggregation_mode,
            Some(DisaggregationMode::Aggregated)
        ));

        // Out-of-range u32 silently drops (rather than wrapping).
        let info = parse_server_info(&server_info([("page_size", num_value(4294967296))], []));
        assert!(info.page_size.is_none());

        // Top-level max_total_num_tokens wins when populated.
        resp.max_total_num_tokens = 42;
        assert_eq!(parse_server_info(&resp).max_total_num_tokens, Some(42));
    }

    #[test]
    fn parses_kv_events_config_when_publisher_is_zmq() {
        let info = parse_server_info(&server_info(
            [(
                "kv_events_config",
                str_value(r#"{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}"#),
            )],
            [],
        ));
        let kv = info.kv_events.expect("kv_events parsed");
        assert_eq!(kv.endpoint, "tcp://*:5557");
        assert_eq!(kv.topic, "kv-events");

        let info = parse_server_info(&server_info(
            [(
                "kv_events_config",
                proto_struct_value([
                    ("publisher", str_value("zmq")),
                    ("topic", str_value("kv-events")),
                    ("endpoint", str_value("tcp://*:5557")),
                ]),
            )],
            [],
        ));
        assert_eq!(info.kv_events.expect("kv_events parsed").topic, "kv-events");
    }

    #[test]
    fn kv_events_disabled_for_null_publisher_or_missing() {
        // publisher: null is SGLang's default → feature off
        let info = parse_server_info(&server_info(
            [("kv_events_config", str_value(r#"{"publisher":"null"}"#))],
            [],
        ));
        assert!(info.kv_events.is_none());

        // missing field → off
        let info = parse_server_info(&server_info([("model_path", str_value("X"))], []));
        assert!(info.kv_events.is_none());

        // empty string → off
        let info = parse_server_info(&server_info([("kv_events_config", str_value(""))], []));
        assert!(info.kv_events.is_none());

        // malformed JSON-inside-JSON → off, not panic
        let info = parse_server_info(&server_info(
            [("kv_events_config", str_value("not json"))],
            [],
        ));
        assert!(info.kv_events.is_none());

        // empty endpoint → off (would otherwise reach ZMQ subscribe).
        let info = parse_server_info(&server_info(
            [(
                "kv_events_config",
                str_value(r#"{"publisher":"zmq","endpoint":""}"#),
            )],
            [],
        ));
        assert!(info.kv_events.is_none());
    }

    #[test]
    fn offset_endpoint_port_matches_sglang() {
        assert_eq!(
            offset_endpoint_port("tcp://*:5557", 0).as_deref(),
            Some("tcp://*:5557")
        );
        assert_eq!(
            offset_endpoint_port("tcp://*:5557", 1).as_deref(),
            Some("tcp://*:5558")
        );
        assert_eq!(
            offset_endpoint_port("tcp://*:5557", 4).as_deref(),
            Some("tcp://*:5561")
        );
        assert!(offset_endpoint_port("udp://x:5557", 1).is_none());
        assert!(offset_endpoint_port("tcp://*:notaport", 1).is_none());
    }
}
