// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Periodic 1-token canary. SGLang's HealthCheck is liveness-only and won't
//! catch a hung scheduler, so we send a real Generate and self-evict via
//! SIGTERM on persistent failure.

use std::time::Duration;

use tokio_stream::StreamExt;
use tonic::transport::Channel;

use crate::proto::v1::{
    GenerateRequest, SamplingParams, sglang_service_client::SglangServiceClient,
};

const INTERVAL: Duration = Duration::from_secs(30);
const TIMEOUT: Duration = Duration::from_secs(10);
const FAILURE_THRESHOLD: u32 = 3;
const CANARY_INPUT_TOKEN: i32 = 1;

pub fn build_canary_request() -> GenerateRequest {
    let nonce: u64 = rand::random();
    GenerateRequest {
        input_ids: vec![CANARY_INPUT_TOKEN],
        sampling_params: Some(SamplingParams {
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: Some(-1),
            max_new_tokens: Some(1),
            ignore_eos: Some(false),
            ..Default::default()
        }),
        stream: Some(true),
        rid: Some(format!("dyn-canary-{nonce:016x}")),
        ..Default::default()
    }
}

async fn run_probe(client: &mut SglangServiceClient<Channel>) -> Result<(), String> {
    let work = async {
        let mut stream = client
            .generate(build_canary_request())
            .await
            .map_err(|e| format!("generate RPC: {e}"))?
            .into_inner();
        loop {
            match stream.next().await {
                Some(Ok(resp)) if resp.finished => return Ok(()),
                Some(Ok(_)) => continue,
                Some(Err(e)) => return Err(format!("stream error: {e}")),
                None => return Err("stream closed without terminal chunk".to_string()),
            }
        }
    };
    tokio::time::timeout(TIMEOUT, work)
        .await
        .map_err(|_| format!("canary timed out after {TIMEOUT:?}"))?
}

pub async fn run_loop(mut client: SglangServiceClient<Channel>) {
    let mut consecutive_failures: u32 = 0;
    loop {
        tokio::time::sleep(INTERVAL).await;
        match run_probe(&mut client).await {
            Ok(()) => {
                if consecutive_failures > 0 {
                    tracing::info!(recovered_after = consecutive_failures, "canary recovered");
                }
                consecutive_failures = 0;
            }
            Err(reason) => {
                consecutive_failures += 1;
                tracing::warn!(consecutive_failures, %reason, "canary failed");
                if consecutive_failures >= FAILURE_THRESHOLD {
                    tracing::error!(
                        consecutive_failures,
                        "canary past threshold; raising SIGTERM"
                    );
                    // Process-local signal -> tokio's signal handler triggers graceful shutdown.
                    unsafe { libc::raise(libc::SIGTERM) };
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_request_shape_matches_python_payload() {
        let r = build_canary_request();
        let s = r.sampling_params.expect("sampling_params");
        assert_eq!(r.input_ids, vec![1]);
        assert_eq!(s.max_new_tokens, Some(1));
        assert_eq!(s.temperature, Some(0.0));
    }
}
