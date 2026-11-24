// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP exporter settings shared by the trace, log and metrics exporters.

use crate::config::environment_names::logging::otlp as env_otlp;

/// Default OTLP endpoint
pub(crate) const DEFAULT_OTLP_ENDPOINT: &str = "http://localhost:4317";

/// Default OTLP HTTP endpoint
const DEFAULT_OTLP_HTTP_ENDPOINT: &str = "http://localhost:4318";

/// Default service name
const DEFAULT_OTEL_SERVICE_NAME: &str = "dynamo";

/// Get the service name from environment or use default
pub(crate) fn get_service_name() -> String {
    std::env::var(env_otlp::OTEL_SERVICE_NAME)
        .unwrap_or_else(|_| DEFAULT_OTEL_SERVICE_NAME.to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OtlpProtocol {
    Grpc,
    HttpProtobuf,
}

impl OtlpProtocol {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Grpc => "grpc",
            Self::HttpProtobuf => "http/protobuf",
        }
    }
}

fn parse_otlp_protocol_for_env(value: Option<&str>, env_name: &str) -> OtlpProtocol {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => OtlpProtocol::Grpc,
        Some(value) if value.eq_ignore_ascii_case("grpc") => OtlpProtocol::Grpc,
        Some(value) if value.eq_ignore_ascii_case("http/protobuf") => OtlpProtocol::HttpProtobuf,
        Some(value) => {
            eprintln!(
                "WARNING: unsupported {} '{}'; falling back to grpc",
                env_name, value
            );
            OtlpProtocol::Grpc
        }
    }
}

fn parse_otlp_protocol(value: Option<&str>) -> OtlpProtocol {
    parse_otlp_protocol_for_env(value, env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL)
}

pub(crate) fn otlp_protocol_from_env() -> OtlpProtocol {
    parse_otlp_protocol(
        std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL)
            .ok()
            .as_deref(),
    )
}

pub(crate) fn resolve_signal_otlp_protocol(
    generic_protocol: OtlpProtocol,
    signal_protocol: Option<&str>,
    signal_protocol_env: &str,
) -> OtlpProtocol {
    match signal_protocol
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => parse_otlp_protocol_for_env(Some(value), signal_protocol_env),
        None => generic_protocol,
    }
}

fn append_otlp_http_path(endpoint: &str, path: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    format!("{endpoint}{path}")
}

pub(crate) fn resolve_otlp_endpoint(
    protocol: OtlpProtocol,
    signal_endpoint: Option<String>,
    generic_endpoint: Option<String>,
    http_path: &str,
) -> String {
    if let Some(endpoint) = signal_endpoint.filter(|value| !value.trim().is_empty()) {
        return endpoint;
    }

    match protocol {
        OtlpProtocol::Grpc => generic_endpoint
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_OTLP_ENDPOINT.to_string()),
        OtlpProtocol::HttpProtobuf => append_otlp_http_path(
            generic_endpoint
                .filter(|value| !value.trim().is_empty())
                .as_deref()
                .unwrap_or(DEFAULT_OTLP_HTTP_ENDPOINT),
            http_path,
        ),
    }
}

/// Parse the `key=value,key=value` form used for headers and resource
/// attributes.
///
/// Splits on the *first* `=` so base64 padding in a bearer token survives.
/// Percent-decoding, which the spec allows, is deliberately not applied.
pub(crate) fn parse_key_value_list(raw: &str) -> Vec<(String, String)> {
    raw.split(',')
        .filter_map(|pair| pair.split_once('='))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otlp_protocol_defaults_to_grpc() {
        assert_eq!(parse_otlp_protocol(None), OtlpProtocol::Grpc);
        assert_eq!(parse_otlp_protocol(Some("")), OtlpProtocol::Grpc);
        assert_eq!(parse_otlp_protocol(Some("grpc")), OtlpProtocol::Grpc);
        assert_eq!(
            parse_otlp_protocol(Some("http/protobuf")),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(
            parse_otlp_protocol(Some("HTTP/PROTOBUF")),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(parse_otlp_protocol(Some("bad")), OtlpProtocol::Grpc);
    }

    #[test]
    fn otlp_signal_protocol_overrides_generic_protocol() {
        let generic_protocol = OtlpProtocol::Grpc;
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                Some("http/protobuf"),
                env_otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::HttpProtobuf
        );
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                Some(""),
                env_otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::Grpc
        );
        assert_eq!(
            resolve_signal_otlp_protocol(
                generic_protocol,
                None,
                env_otlp::OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
            ),
            OtlpProtocol::Grpc
        );
    }

    #[test]
    fn otlp_http_endpoint_appends_signal_paths_from_generic_endpoint() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com".to_string()),
                "/v1/traces",
            ),
            "https://llm-observe.weizhipin.com/v1/traces"
        );
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com/".to_string()),
                "/v1/logs",
            ),
            "https://llm-observe.weizhipin.com/v1/logs"
        );
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                None,
                Some("https://llm-observe.weizhipin.com/v1/traces".to_string()),
                "/v1/traces",
            ),
            "https://llm-observe.weizhipin.com/v1/traces/v1/traces"
        );
    }

    #[test]
    fn otlp_signal_endpoint_is_used_verbatim() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::HttpProtobuf,
                Some("https://collector.example/custom/traces".to_string()),
                Some("https://collector.example".to_string()),
                "/v1/traces",
            ),
            "https://collector.example/custom/traces"
        );
    }

    #[test]
    fn otlp_grpc_endpoint_keeps_generic_endpoint_verbatim() {
        assert_eq!(
            resolve_otlp_endpoint(
                OtlpProtocol::Grpc,
                None,
                Some("http://otel-collector:4317".to_string()),
                "/v1/traces",
            ),
            "http://otel-collector:4317"
        );
    }
}
