// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TTS uses a single HTTP request with a streaming response. Model-specific
//! prompt construction, codec encoding, and speed adjustment stay upstream.

use bytes::Bytes;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;

use crate::{Client, Error};

impl Client {
    /// Send an upstream speech request. Dynamo delivery fields and media policy
    /// are handled by the embedding worker before calling this transport.
    pub async fn speech(
        &self,
        mut request: Value,
    ) -> Result<BoxStream<'static, Result<Bytes, Error>>, Error> {
        request["model"] = self.inner.model.clone().into();
        let response = tokio::time::timeout(
            self.inner.limits.response_timeout,
            self.inner
                .http
                .post(self.url("v1/audio/speech"))
                .json(&request)
                .send(),
        )
        .await
        .map_err(|_| Error::Timeout)??;
        let response = response.error_for_status()?;
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        if !content_type.starts_with("audio/") && content_type != "application/octet-stream" {
            return Err(Error::Protocol(
                "speech endpoint returned a non-audio response".into(),
            ));
        }
        let timeout = self.inner.limits.response_timeout;
        let shutdown = self.inner.shutdown.clone();
        let mut bytes = response.bytes_stream();
        Ok(Box::pin(async_stream::try_stream! {
            loop {
                let chunk = tokio::select! {
                    _ = shutdown.cancelled() => Err(Error::Closed),
                    result = tokio::time::timeout(timeout, bytes.next()) => result.map_err(|_| Error::Timeout),
                }?;
                match chunk {
                    Some(chunk) => yield chunk?,
                    None => break,
                }
            }
        }))
    }
}
