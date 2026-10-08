// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVCR discovery from SGLang's engine-reported configuration, not sidecar-local overrides.

use std::collections::HashMap;
use std::net::IpAddr;

use dynamo_backend_common::{DisaggregationMode, DynamoError, KV_HINT_TRANSFER_CAPABILITY_KEY};
use serde_json::{Map, Value, json};

use crate::client;

pub(crate) fn discovery_runtime_data(
    server_info: &Value,
    mode: DisaggregationMode,
) -> Result<HashMap<String, Value>, DynamoError> {
    let extra_key = if server_info["enable_unified_cache_external_linker"] == true {
        if server_info["unified_cache_external_linker_backend"] != "kvcr" {
            return Ok(HashMap::new());
        }
        "unified_cache_external_linker_extra_config"
    } else if server_info["enable_hierarchical_cache"] == true
        && server_info["hicache_storage_backend"] == "kvcr"
    {
        "hicache_storage_backend_extra_config"
    } else {
        return Ok(HashMap::new());
    };

    // GetServerInfo can contain either a decoded mapping or the JSON CLI argument.
    // Never read an @file in the sidecar: its filesystem need not be the engine's.
    let extra = match server_info.get(extra_key) {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(extra)) => extra.clone(),
        Some(Value::String(raw)) if raw.is_empty() => Map::new(),
        Some(Value::String(raw)) if raw.starts_with('@') => {
            tracing::warn!(
                extra_config_field = extra_key,
                "SGLang reports an unresolved @file config; KVCR hint discovery is disabled until the engine reports resolved JSON"
            );
            return Ok(HashMap::new());
        }
        Some(Value::String(raw)) => {
            serde_json::from_str::<Map<String, Value>>(raw).map_err(|_| {
                client::invalid_arg(format!(
                    "SGLang {extra_key} must report a resolved object or JSON object"
                ))
            })?
        }
        _ => {
            return Err(client::invalid_arg(format!(
                "SGLang {extra_key} must be a JSON object"
            )));
        }
    };
    match extra.get("enable_remote_hint") {
        None | Some(Value::Bool(false)) => return Ok(HashMap::new()),
        Some(Value::Bool(true)) => {}
        _ => {
            return Err(client::invalid_arg(
                "KVCR enable_remote_hint must be a boolean",
            ));
        }
    }

    // The rank-zero gRPC frontend represents all engine ranks. One advertised
    // host cannot describe peers on other nodes; guessing would misdirect reads.
    if positive_u32(server_info, "nnodes", 1)? != 1 {
        return Err(client::invalid_arg(
            "native SGLang KVCR hint discovery requires nnodes=1 until the engine reports per-node control endpoints",
        ));
    }
    if positive_u32(server_info, "pp_size", 1)? != 1 {
        return Err(client::invalid_arg(
            "native SGLang KVCR hint discovery does not support pipeline parallelism",
        ));
    }
    let tp_size = positive_u32(server_info, "tp_size", 1)?;
    let dp_size = positive_u32(server_info, "dp_size", 1)?;
    if dp_size > 1 && server_info["enable_dp_attention"] != true {
        return Err(client::invalid_arg(
            "native SGLang KVCR hint discovery requires enable_dp_attention for dp_size > 1",
        ));
    }
    if tp_size % dp_size != 0 {
        return Err(client::invalid_arg(
            "KVCR hint discovery requires tp_size divisible by dp_size",
        ));
    }
    let port = extra
        .get("control_port")
        .and_then(Value::as_u64)
        .filter(|port| *port > 0 && *port <= u64::from(u16::MAX))
        .ok_or_else(|| {
            client::invalid_arg("KVCR remote hints require an explicit control_port in 1..=65535")
        })?;
    if port + u64::from(tp_size) - 1 > u64::from(u16::MAX) {
        return Err(client::invalid_arg(
            "KVCR control_port leaves no room for all tensor-parallel ranks",
        ));
    }
    let host = extra
        .get("control_advertise_host")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_arg("KVCR remote hints require control_advertise_host"))?;
    let host = control_host(host, port as u16)?;

    // KVCR binds base_port + global TP rank. Attention-DP rank d starts at
    // d * (tp_size / dp_size); the engine derives the remaining TP/CP peers.
    let stride = tp_size / dp_size;
    let endpoints: Map<String, Value> = (0..dp_size)
        .map(|rank| {
            (
                rank.to_string(),
                json!(format!("tcp://{host}:{}", port + u64::from(rank * stride))),
            )
        })
        .collect();
    let worker_type = match mode {
        DisaggregationMode::Aggregated => "aggregated",
        DisaggregationMode::Prefill => "prefill",
        DisaggregationMode::Decode => "decode",
        DisaggregationMode::Encode => {
            return Err(client::invalid_arg(
                "SGLang KVCR hints do not support encoder workers",
            ));
        }
    };
    Ok(HashMap::from([
        (KV_HINT_TRANSFER_CAPABILITY_KEY.to_string(), json!(true)),
        ("router_hint_worker_type".to_string(), json!(worker_type)),
        (
            "router_hint_source_control_endpoints".to_string(),
            Value::Object(endpoints),
        ),
    ]))
}

fn positive_u32(info: &Value, field: &str, default: u32) -> Result<u32, DynamoError> {
    match info.get(field) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                client::invalid_arg(format!("KVCR hint discovery requires a positive {field}"))
            }),
    }
}

fn control_host(host: &str, port: u16) -> Result<String, DynamoError> {
    let invalid = || {
        client::invalid_arg(
            "KVCR control_advertise_host must be a non-wildcard TCP host without a scheme, port, or URL components",
        )
    };
    if host.is_empty() || host.contains(|c: char| c.is_whitespace() || "/?#@%*\\".contains(c)) {
        return Err(invalid());
    }
    let bare = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    let host = match bare.parse::<IpAddr>() {
        Ok(address) if address.is_unspecified() => return Err(invalid()),
        Ok(IpAddr::V6(address)) => format!("[{address}]"),
        Ok(IpAddr::V4(address)) => address.to_string(),
        Err(_) => host.to_string(),
    };
    let endpoint = url::Url::parse(&format!("tcp://{host}:{port}")).map_err(|_| invalid())?;
    if endpoint.host_str().is_none()
        || endpoint.port() != Some(port)
        || !endpoint.username().is_empty()
        || endpoint.password().is_some()
        || !endpoint.path().is_empty()
        || endpoint.query().is_some()
        || endpoint.fragment().is_some()
    {
        return Err(invalid());
    }
    Ok(host)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> Value {
        json!({
            "enable_unified_cache_external_linker": true,
            "unified_cache_external_linker_backend": "kvcr",
            "unified_cache_external_linker_extra_config": {
                "enable_remote_hint": true, "control_advertise_host": "peer.example", "control_port": 12000,
            },
            "tp_size": 8, "dp_size": 2, "enable_dp_attention": true, "nnodes": 1, "pp_size": 1,
        })
    }

    #[test]
    fn attention_dp_endpoints_follow_global_tp_rank_offsets() {
        let data = discovery_runtime_data(&info(), DisaggregationMode::Aggregated).unwrap();
        assert_eq!(data[KV_HINT_TRANSFER_CAPABILITY_KEY], true);
        assert_eq!(data["router_hint_worker_type"], "aggregated");
        assert_eq!(
            data["router_hint_source_control_endpoints"],
            json!({
                "0": "tcp://peer.example:12000", "1": "tcp://peer.example:12004",
            })
        );
    }

    #[test]
    fn tp_only_and_worker_roles_are_reported() {
        let mut info = info();
        info["dp_size"] = json!(1);
        for (mode, role) in [
            (DisaggregationMode::Aggregated, "aggregated"),
            (DisaggregationMode::Prefill, "prefill"),
            (DisaggregationMode::Decode, "decode"),
        ] {
            let data = discovery_runtime_data(&info, mode).unwrap();
            assert_eq!(data["router_hint_worker_type"], role);
            assert_eq!(
                data["router_hint_source_control_endpoints"],
                json!({"0": "tcp://peer.example:12000"})
            );
        }
    }

    #[test]
    fn storage_and_json_encoded_config_use_the_same_contract() {
        let mut storage = info();
        storage["enable_unified_cache_external_linker"] = json!(false);
        storage["enable_hierarchical_cache"] = json!(true);
        storage["hicache_storage_backend"] = json!("kvcr");
        storage["hicache_storage_backend_extra_config"] =
            storage["unified_cache_external_linker_extra_config"].clone();
        let expected = discovery_runtime_data(&storage, DisaggregationMode::Aggregated).unwrap();
        storage["hicache_storage_backend_extra_config"] =
            json!(storage["hicache_storage_backend_extra_config"].to_string());
        assert_eq!(
            discovery_runtime_data(&storage, DisaggregationMode::Aggregated).unwrap(),
            expected
        );
        let mut linker = info();
        linker["unified_cache_external_linker_extra_config"] =
            json!(linker["unified_cache_external_linker_extra_config"].to_string());
        assert_eq!(
            discovery_runtime_data(&linker, DisaggregationMode::Aggregated).unwrap(),
            expected
        );
    }

    #[test]
    fn disabled_and_non_kvcr_backends_do_not_advertise_hints() {
        let mut cases = vec![json!({}), json!({"hicache_storage_backend": "kvcr"})];
        for (field, value) in [
            ("enable_unified_cache_external_linker", json!(false)),
            ("unified_cache_external_linker_backend", json!("mooncake")),
        ] {
            let mut info = info();
            info[field] = value;
            cases.push(info);
        }
        for extra in [Value::Null, json!({}), json!({"enable_remote_hint": false})] {
            let mut info = info();
            info["unified_cache_external_linker_extra_config"] = extra;
            cases.push(info);
        }
        for info in cases {
            assert!(
                discovery_runtime_data(&info, DisaggregationMode::Aggregated)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn invalid_config_is_rejected() {
        for extra in [
            json!([]),
            json!("not json"),
            json!("[]"),
            json!({"enable_remote_hint": "true"}),
        ] {
            let mut info = info();
            info["unified_cache_external_linker_extra_config"] = extra;
            assert!(discovery_runtime_data(&info, DisaggregationMode::Aggregated).is_err());
        }
    }

    #[test]
    fn unresolved_engine_file_disables_only_hint_discovery() {
        for extra_key in [
            "unified_cache_external_linker_extra_config",
            "hicache_storage_backend_extra_config",
        ] {
            let mut info = info();
            if extra_key == "hicache_storage_backend_extra_config" {
                info["enable_unified_cache_external_linker"] = json!(false);
                info["enable_hierarchical_cache"] = json!(true);
                info["hicache_storage_backend"] = json!("kvcr");
            }
            info[extra_key] = json!("@/engine/config.json");
            // Do not read the engine's file or reject ordinary multinode serving.
            info["nnodes"] = json!(2);
            assert!(
                discovery_runtime_data(&info, DisaggregationMode::Aggregated)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn control_ports_are_checked_for_the_entire_rank_range() {
        for port in [
            json!(0),
            json!(-1),
            json!(65536),
            json!(65529),
            json!(true),
            json!(12000.5),
            Value::Null,
        ] {
            let mut info = info();
            info["unified_cache_external_linker_extra_config"]["control_port"] = port;
            assert!(discovery_runtime_data(&info, DisaggregationMode::Aggregated).is_err());
        }
        let mut info = info();
        info["unified_cache_external_linker_extra_config"]["control_port"] = json!(65528);
        assert!(discovery_runtime_data(&info, DisaggregationMode::Aggregated).is_ok());
    }

    #[test]
    fn unsupported_or_invalid_topologies_are_rejected() {
        for (field, value) in [
            ("nnodes", json!(2)),
            ("pp_size", json!(2)),
            ("enable_dp_attention", json!(false)),
            ("dp_size", json!(3)),
            ("dp_size", json!(0)),
            ("tp_size", json!(0)),
            ("tp_size", json!(u64::MAX)),
            ("nnodes", json!("1")),
        ] {
            let mut info = info();
            info[field] = value;
            assert!(discovery_runtime_data(&info, DisaggregationMode::Aggregated).is_err());
        }
        // The topology restrictions do not change ordinary, non-remote engines.
        let mut info = info();
        info["nnodes"] = json!(2);
        info["unified_cache_external_linker_extra_config"]["enable_remote_hint"] = json!(false);
        assert!(
            discovery_runtime_data(&info, DisaggregationMode::Aggregated)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn ipv6_hosts_are_bracketed_without_changing_the_port() {
        for host in ["2001:db8::1", "[2001:db8::1]"] {
            let mut info = info();
            info["unified_cache_external_linker_extra_config"]["control_advertise_host"] =
                json!(host);
            let data = discovery_runtime_data(&info, DisaggregationMode::Aggregated).unwrap();
            assert_eq!(
                data["router_hint_source_control_endpoints"]["1"],
                "tcp://[2001:db8::1]:12004"
            );
        }
    }

    #[test]
    fn wildcard_and_malformed_advertised_hosts_are_rejected() {
        for host in [
            json!(""),
            json!("0.0.0.0"),
            json!("::"),
            json!("[::]"),
            json!("*"),
            json!("*.example"),
            json!("peer\\host"),
            json!("tcp://peer"),
            json!("peer:123"),
            json!("peer/path"),
            json!("user@peer"),
            json!("peer?query"),
            json!("peer#fragment"),
            json!(" peer "),
            json!("[bad-ip]"),
            Value::Null,
        ] {
            let mut info = info();
            info["unified_cache_external_linker_extra_config"]["control_advertise_host"] = host;
            assert!(discovery_runtime_data(&info, DisaggregationMode::Aggregated).is_err());
        }
    }
}
