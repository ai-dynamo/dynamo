// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};

#[derive(Clone)]
pub(super) struct EngineControlAuth {
    expected: Option<[u8; 32]>,
}

impl EngineControlAuth {
    pub(super) fn from_env() -> anyhow::Result<Self> {
        use crate::config::environment_names::runtime::system::DYN_SYSTEM_ENGINE_AUTH_TOKEN;
        match std::env::var(DYN_SYSTEM_ENGINE_AUTH_TOKEN) {
            Ok(secret) => Self::from_secret(Some(&secret)),
            Err(std::env::VarError::NotPresent) => Self::from_secret(None),
            Err(_) => anyhow::bail!("Invalid engine control authentication configuration"),
        }
    }

    pub(super) fn from_secret(secret: Option<&str>) -> anyhow::Result<Self> {
        let expected = secret
            .map(|secret| {
                anyhow::ensure!(
                    secret.len() >= 32
                        && secret.bytes().all(|byte| byte.is_ascii_graphic()),
                    "Engine control authentication requires at least 32 printable non-whitespace ASCII bytes"
                );
                Ok::<[u8; 32], anyhow::Error>(Sha256::digest(secret.as_bytes()).into())
            })
            .transpose()?;
        Ok(Self { expected })
    }
}

pub(super) async fn authorize(
    State(auth): State<EngineControlAuth>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = auth.expected else {
        return next.run(request).await;
    };
    let mut headers = request.headers().get_all(header::AUTHORIZATION).iter();
    let supplied = headers
        .next()
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split_once(' '))
        .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("Bearer"))
        .map(|(_, token)| token);
    if headers.next().is_none()
        && let Some(token) = supplied
    {
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        if constant_time_eq::constant_time_eq_32(&digest, &expected) {
            return next.run(request).await;
        }
    }
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer")],
        Json(serde_json::json!({"error": "Engine control authentication required"})),
    )
        .into_response()
}
