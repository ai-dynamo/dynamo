// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::DynamoError;
use futures::future::try_join_all;
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::{GrpcEndpoint, GrpcTransportConfig, cannot_connect, invalid_argument};

pub const DEFAULT_MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const RETRY_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Connected channels distributed in round-robin order.
pub struct GrpcChannelPool {
    channels: Vec<Channel>,
    next: AtomicUsize,
}

impl GrpcChannelPool {
    pub async fn connect(
        peer: &str,
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
    ) -> Result<Self, DynamoError> {
        let endpoint_label = endpoint.to_string();
        let tonic_endpoint = Endpoint::from_shared(endpoint_label.clone()).map_err(|error| {
            invalid_argument(format!("invalid {peer} endpoint after validation: {error}"))
        })?;
        let deadline = checked_instant_add(
            Instant::now(),
            transport.startup_deadline,
            "gRPC startup deadline",
        )?;
        let first = connect_until_ready(
            peer,
            tonic_endpoint.clone(),
            endpoint_label.clone(),
            1,
            transport,
            deadline,
        )
        .await?;
        let mut channels = vec![first];
        let remaining = try_join_all((1..transport.connections.get()).map(|index| {
            let endpoint = tonic_endpoint.clone();
            let endpoint_label = endpoint_label.clone();
            async move {
                connect_until_ready(
                    peer,
                    endpoint,
                    endpoint_label,
                    index + 1,
                    transport,
                    deadline,
                )
                .await
            }
        }))
        .await?;
        channels.extend(remaining);
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
        })
    }

    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub fn next_channel(&self) -> Channel {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.channels.len();
        self.channels[index].clone()
    }
}

async fn connect_until_ready(
    peer: &str,
    endpoint: Endpoint,
    endpoint_label: String,
    pool_slot: usize,
    transport: GrpcTransportConfig,
    deadline: Instant,
) -> Result<Channel, DynamoError> {
    let started = Instant::now();
    let mut attempt = 0_u64;
    let mut last_error = None;
    let mut last_logged_at = None;
    let mut last_logged_error = None;
    let mut suppressed_attempts = 0_u64;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(startup_timeout(
                peer,
                &endpoint_label,
                pool_slot,
                attempt,
                started.elapsed(),
                transport,
                last_error.as_deref(),
            ));
        }

        attempt += 1;
        let attempt_endpoint = endpoint
            .clone()
            .connect_timeout(transport.connect_attempt_timeout.min(remaining));
        match timeout_at(deadline, attempt_endpoint.connect()).await {
            Ok(Ok(channel)) => return Ok(channel),
            Ok(Err(error)) => {
                let detailed_error = format_error_chain(&error);
                let now = Instant::now();
                let error_changed = last_logged_error.as_deref() != Some(detailed_error.as_str());
                let log_interval_elapsed = last_logged_at
                    .is_none_or(|last| now.duration_since(last) >= RETRY_LOG_INTERVAL);
                if error_changed || log_interval_elapsed {
                    tracing::debug!(
                        peer,
                        endpoint = %endpoint_label,
                        pool_slot,
                        attempt,
                        elapsed = ?started.elapsed(),
                        remaining = ?deadline.saturating_duration_since(now),
                        retry_interval = ?transport.retry_interval,
                        suppressed_attempts,
                        error = ?error,
                        "sidecar gRPC connection attempt failed"
                    );
                    last_logged_at = Some(now);
                    last_logged_error = Some(detailed_error.clone());
                    suppressed_attempts = 0;
                } else {
                    suppressed_attempts += 1;
                }
                last_error = Some(detailed_error);
            }
            Err(_) => {
                return Err(startup_timeout(
                    peer,
                    &endpoint_label,
                    pool_slot,
                    attempt,
                    started.elapsed(),
                    transport,
                    last_error.as_deref(),
                ));
            }
        }

        if Instant::now() >= deadline {
            return Err(startup_timeout(
                peer,
                &endpoint_label,
                pool_slot,
                attempt,
                started.elapsed(),
                transport,
                last_error.as_deref(),
            ));
        }
        let retry_at = checked_instant_add(
            Instant::now(),
            transport.retry_interval,
            "gRPC retry interval",
        )?;
        sleep_until(retry_at.min(deadline)).await;
    }
}

fn checked_instant_add(
    instant: Instant,
    duration: Duration,
    setting: &str,
) -> Result<Instant, DynamoError> {
    instant.checked_add(duration).ok_or_else(|| {
        invalid_argument(format!(
            "{setting} {duration:?} exceeds the supported monotonic clock range"
        ))
    })
}

fn startup_timeout(
    peer: &str,
    endpoint: &str,
    pool_slot: usize,
    attempts: u64,
    elapsed: Duration,
    transport: GrpcTransportConfig,
    last_error: Option<&str>,
) -> DynamoError {
    let cause = last_error.unwrap_or("the connection attempt exceeded the startup deadline");
    cannot_connect(format!(
        "failed to establish the {peer} gRPC connection pool to {endpoint} for pool slot {pool_slot} within {:?} after {attempts} attempts over {elapsed:?}: {cause}",
        transport.startup_deadline,
    ))
}

fn format_error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let _ = write!(message, ": {cause}");
        source = cause.source();
    }
    message
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::time::Duration;

    use tokio::net::TcpListener;

    use super::GrpcChannelPool;
    use crate::{GrpcEndpoint, GrpcTransportConfig};

    #[tokio::test]
    async fn startup_deadline_caps_connection_retries() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let address = listener.local_addr().expect("address");
        drop(listener);

        let transport = GrpcTransportConfig {
            connections: NonZeroUsize::new(2).unwrap(),
            connect_attempt_timeout: Duration::from_millis(50),
            retry_interval: Duration::from_millis(10),
            startup_deadline: Duration::from_millis(100),
        };
        let endpoint = GrpcEndpoint::parse(&address.to_string(), "--test-endpoint").unwrap();
        let result = tokio::time::timeout(
            Duration::from_millis(300),
            GrpcChannelPool::connect("test", &endpoint, transport),
        )
        .await
        .expect("connection retries must respect the startup deadline");

        let error = match result {
            Ok(_) => panic!("the endpoint is closed"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("within 100ms"),
            "unexpected error: {error}"
        );
    }
}
