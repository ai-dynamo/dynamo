// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::{router::MODEL_HEADER, server::Error};

pub const MAX_BODY: usize = 2 * 1024 * 1024;

pub fn headers(input: &[(String, String)]) -> anyhow::Result<(http::HeaderMap, Vec<String>)> {
    let mut map = http::HeaderMap::new();
    let mut method = None;
    let mut path = None;
    let mut remove = [
        MODEL_HEADER,
        "x-gateway-destination-endpoint",
        "x-worker-instance-id",
        "x-prefill-instance-id",
        "x-prefiller-host-port",
        "x-dp-rank",
        "x-data-parallel-rank",
        "x-prefill-dp-rank",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for (key, value) in input {
        let key = key.to_ascii_lowercase();
        if key == ":method" {
            anyhow::ensure!(
                method.replace(value.to_owned()).is_none(),
                "duplicate method"
            );
            continue;
        }
        if key == ":path" {
            anyhow::ensure!(path.replace(value.to_owned()).is_none(), "duplicate path");
            continue;
        }
        if key.starts_with(':') {
            continue;
        }
        if key.starts_with("x-dynamo-") || key.starts_with("x-gateway-") {
            remove.push(key.clone());
        }
        let name = http::HeaderName::from_bytes(key.as_bytes())?;
        anyhow::ensure!(!map.contains_key(&name), "duplicate request header");
        map.insert(name, http::HeaderValue::from_str(value)?);
    }
    anyhow::ensure!(method.as_deref() == Some("POST"), "only POST is supported");
    anyhow::ensure!(
        path.as_deref().and_then(|p| p.split('?').next()) == Some("/v1/chat/completions"),
        "only /v1/chat/completions is supported"
    );
    anyhow::ensure!(
        map.get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))),
        "content-type must be application/json"
    );
    if map.contains_key("content-encoding") {
        return Err(Error::new(415, "compressed request bodies are unsupported").into());
    }
    if let Some(length) = map.get("content-length")
        && length.to_str()?.parse::<usize>()? > MAX_BODY
    {
        return Err(Error::new(413, "request body exceeds 2 MiB").into());
    }
    remove.sort();
    remove.dedup();
    Ok((map, remove))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Vec<(String, String)> {
        [
            (":method", "POST"),
            (":path", "/v1/chat/completions"),
            ("content-type", "application/json"),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect()
    }

    #[test]
    fn strips_forged_routing_headers_and_rejects_duplicate_session() {
        let mut input = input();
        input.push(("x-dynamo-worker-id".into(), "forged".into()));
        let (_, remove) = headers(&input).unwrap();
        for key in [
            MODEL_HEADER,
            "x-gateway-destination-endpoint",
            "x-dynamo-worker-id",
            "x-worker-instance-id",
            "x-prefill-instance-id",
            "x-prefiller-host-port",
            "x-dp-rank",
            "x-data-parallel-rank",
            "x-prefill-dp-rank",
        ] {
            assert!(remove.iter().any(|value| value == key));
        }
        input.extend([
            ("x-switchyard-session-id".into(), "a".into()),
            ("x-switchyard-session-id".into(), "b".into()),
        ]);
        assert!(headers(&input).is_err());
    }
}
