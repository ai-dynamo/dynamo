// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::str::FromStr;

use axum::http::{HeaderMap, uri::Authority};

pub const PREFILLER_HOST_PORT: &str = "x-prefiller-host-port";

const EPP_ONLY_HEADERS: &[&str] = &[
    PREFILLER_HOST_PORT,
    "x-gateway-destination-endpoint",
    "x-gateway-destination-endpoint-served",
    "x-gateway-destination-endpoint-subset",
    "x-gateway-inference-fairness-id",
    "x-gateway-inference-objective",
    "x-gateway-model-name-rewrite",
    "x-dynamo-worker-instance-id",
    "x-dynamo-prefill-instance-id",
    "x-dynamo-dp-rank",
    "x-dynamo-prefill-dp-rank",
    "x-dynamo-routing-mode",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillEndpoint(Authority);

impl PrefillEndpoint {
    pub fn parse_headers(headers: &HeaderMap) -> Result<Option<Self>, InvalidEppMetadata> {
        let mut values = headers.get_all(PREFILLER_HOST_PORT).iter();
        let Some(value) = values.next() else {
            return Ok(None);
        };
        if values.next().is_some() {
            return Err(InvalidEppMetadata);
        }

        let value = value.to_str().map_err(|_| InvalidEppMetadata)?;
        if value.is_empty() || value.contains(',') || value.contains('@') || value.trim() != value {
            return Err(InvalidEppMetadata);
        }
        let authority = Authority::from_str(value).map_err(|_| InvalidEppMetadata)?;
        if authority.host().is_empty() || authority.port_u16().is_none_or(|port| port == 0) {
            return Err(InvalidEppMetadata);
        }
        Ok(Some(Self(authority)))
    }

    pub fn authority(&self) -> &Authority {
        &self.0
    }
}

impl fmt::Display for PrefillEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("invalid EPP routing metadata")]
pub struct InvalidEppMetadata;

pub fn strip_epp_headers(headers: &mut HeaderMap) {
    for header in EPP_ONLY_HEADERS {
        headers.remove(*header);
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};

    use super::*;

    #[test]
    fn accepts_host_ipv4_and_bracketed_ipv6() {
        for endpoint in [
            "prefill.default.svc:8001",
            "10.0.0.2:8001",
            "[2001:db8::1]:8001",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(PREFILLER_HOST_PORT, HeaderValue::from_static(endpoint));
            assert_eq!(
                PrefillEndpoint::parse_headers(&headers)
                    .unwrap()
                    .unwrap()
                    .to_string(),
                endpoint
            );
        }
    }

    #[test]
    fn rejects_invalid_or_ambiguous_values() {
        for endpoint in [
            "",
            "prefill",
            "prefill:0",
            "prefill:abc",
            "2001:db8::1:8001",
            "prefill:8001,other:8001",
            "user:pass@prefill:8001",
            " prefill:8001",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                PREFILLER_HOST_PORT,
                HeaderValue::from_str(endpoint).unwrap(),
            );
            assert!(
                PrefillEndpoint::parse_headers(&headers).is_err(),
                "{endpoint}"
            );
        }
    }

    #[test]
    fn rejects_repeated_values() {
        let mut headers = HeaderMap::new();
        headers.append(PREFILLER_HOST_PORT, HeaderValue::from_static("one:8001"));
        headers.append(PREFILLER_HOST_PORT, HeaderValue::from_static("two:8001"));
        assert!(PrefillEndpoint::parse_headers(&headers).is_err());
    }

    #[test]
    fn strips_all_epp_owned_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            PREFILLER_HOST_PORT,
            HeaderValue::from_static("prefill:8001"),
        );
        headers.insert(
            "x-gateway-destination-endpoint",
            HeaderValue::from_static("decode:8000"),
        );
        headers.insert("authorization", HeaderValue::from_static("Bearer token"));

        strip_epp_headers(&mut headers);

        assert!(!headers.contains_key(PREFILLER_HOST_PORT));
        assert!(!headers.contains_key("x-gateway-destination-endpoint"));
        assert_eq!(headers["authorization"], "Bearer token");
    }
}
