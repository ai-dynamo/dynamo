// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SubSocket};

const INITIAL_SETUP_BACKOFF_MS: u64 = 10;
const MAX_SETUP_BACKOFF_MS: u64 = 5000;
const MAX_SETUP_BACKOFF_EXPONENT: u32 = 8;

fn calculate_setup_backoff_ms(consecutive_errors: u32) -> u64 {
    std::cmp::min(
        INITIAL_SETUP_BACKOFF_MS * 2_u64.pow(consecutive_errors.min(MAX_SETUP_BACKOFF_EXPONENT)),
        MAX_SETUP_BACKOFF_MS,
    )
}

pub(crate) async fn connect_sub_socket_with_retry(
    zmq_endpoint: &str,
    zmq_topic: Option<&str>,
    cancellation_token: &CancellationToken,
    log_prefix: &str,
) -> Option<SubSocket> {
    let mut consecutive_errors = 0u32;
    let topic = zmq_topic.unwrap_or("");

    loop {
        if cancellation_token.is_cancelled() {
            tracing::debug!("{log_prefix}: cancelled before connecting to {zmq_endpoint}");
            return None;
        }

        let mut socket = SubSocket::new();

        match socket.subscribe(topic).await {
            Ok(()) => {}
            Err(e) => {
                consecutive_errors += 1;
                let backoff_ms = calculate_setup_backoff_ms(consecutive_errors);
                tracing::warn!(
                    error=%e,
                    consecutive_errors=%consecutive_errors,
                    backoff_ms=%backoff_ms,
                    "{log_prefix}: failed to subscribe on ZMQ socket during setup, retrying"
                );
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => return None,
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
                continue;
            }
        }

        match socket.connect(zmq_endpoint).await {
            Ok(()) => return Some(socket),
            Err(e) => {
                consecutive_errors += 1;
                let backoff_ms = calculate_setup_backoff_ms(consecutive_errors);
                tracing::warn!(
                    error=%e,
                    consecutive_errors=%consecutive_errors,
                    backoff_ms=%backoff_ms,
                    "{log_prefix}: failed to connect ZMQ SUB during setup, retrying"
                );
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => return None,
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
            }
        }
    }
}
