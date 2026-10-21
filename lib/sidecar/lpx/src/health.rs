// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sidecar readiness that follows the engine's `grpc.health.v1` status.
//!
//! The sidecar's system status server reports only its runtime dependencies.
//! This watches the engine's health for the server as a whole (service `""`)
//! and keeps a [`RuntimeReadinessGate`] closed while the engine is not serving
//! or its health cannot be read, so the sidecar's readiness probe follows the
//! engine too. A lost watch is re-established with backoff; the sidecar does
//! not exit over it. LPX engines serve `grpc.health.v1`, so an engine that
//! answers `UNIMPLEMENTED` is one whose health cannot be read.
//!
//! An LPX engine reports `SERVING` for `""` exactly while it admits requests,
//! and `NOT_SERVING` while it starts, drains or stops. It reports
//! `NOT_SERVING` before it accepts its first connection, so the sidecar never
//! sees the `SERVING` a fresh tonic-health reporter starts with.
//!
//! The sidecar holds a watch open only while the engine reports `SERVING`.
//! On any other status it ends the watch and checks again after the retry
//! interval. A `Watch` stream lasts as long as the engine keeps a status for
//! the service (tonic-health's ends only once the engine clears it), and a
//! gRPC server that shuts down gracefully waits for every open stream, so a
//! watch held on a draining engine would keep it from exiting until it is
//! killed.

use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::system_health::RuntimeReadinessGate;
use dynamo_sidecar_common::format_error_chain;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_client::HealthClient;

/// The service name `grpc.health.v1` gives the server as a whole.
const SERVER: &str = "";

/// Longest wait between attempts to re-establish a lost watch, unless the
/// configured retry interval is longer still.
const MAX_RETRY_INTERVAL: Duration = Duration::from_secs(10);

/// What one health message means for readiness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EngineHealth {
    Serving,
    NotServing,
}

impl EngineHealth {
    fn from_status(status: i32) -> Self {
        match ServingStatus::try_from(status) {
            Ok(ServingStatus::Serving) => Self::Serving,
            // NOT_SERVING, UNKNOWN, SERVICE_UNKNOWN, and values newer than
            // this client.
            _ => Self::NotServing,
        }
    }
}

/// How one `Watch` call ended.
enum WatchEnd {
    /// The engine reported a status other than `SERVING`, and the sidecar
    /// ended the watch so that it holds no stream open on the engine.
    Released,
    /// Any other ending, a clean one or `UNIMPLEMENTED` included; it is
    /// retried with backoff.
    Lost { reason: String, delivered: bool },
}

/// Follows the health of `engine` at `endpoint` until `cancel`, keeping `gate`
/// closed whenever the engine is not serving or its health cannot be read.
/// Returning drops `gate`, which opens it.
pub(crate) async fn follow(
    channel: Channel,
    mut gate: RuntimeReadinessGate,
    engine: Arc<str>,
    endpoint: String,
    retry_interval: Duration,
    cancel: CancellationToken,
) {
    let mut client = HealthClient::new(channel);
    let max_retry_interval = retry_interval.max(MAX_RETRY_INTERVAL);
    let mut backoff = retry_interval;
    let mut last_reason: Option<String> = None;
    // The last health logged. It outlives a released watch, so a status that
    // repeats across checks is logged once.
    let mut reported: Option<EngineHealth> = None;
    loop {
        let end = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            end = watch(&mut client, &mut gate, &mut reported, &engine, &endpoint) => end,
        };
        let wait = match end {
            // Not a failure: check again soon, and start any later backoff
            // afresh.
            WatchEnd::Released => {
                backoff = retry_interval;
                last_reason = None;
                retry_interval
            }
            WatchEnd::Lost { reason, delivered } => {
                if delivered {
                    backoff = retry_interval;
                    last_reason = None;
                }
                reported = None;
                let was_ready = !gate.is_closed();
                gate.close();
                // Escaped: the reason embeds the engine's status text.
                if was_ready || last_reason.as_deref() != Some(reason.as_str()) {
                    tracing::warn!(
                        engine = &*engine,
                        endpoint,
                        reason = %reason.escape_debug(),
                        retry_in = ?backoff,
                        "lost the engine health watch; sidecar readiness is withdrawn until it \
                         is re-established"
                    );
                } else {
                    tracing::debug!(
                        engine = &*engine,
                        endpoint,
                        reason = %reason.escape_debug(),
                        retry_in = ?backoff,
                        "the engine health watch is still unavailable"
                    );
                }
                last_reason = Some(reason);
                let wait = backoff;
                backoff = backoff.saturating_mul(2).min(max_retry_interval);
                wait
            }
        };
        tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(wait) => {}
        }
    }
}

/// One `Watch` call, applying each status the engine reports to `gate`. It
/// lasts only while the engine reports `SERVING`.
async fn watch(
    client: &mut HealthClient<Channel>,
    gate: &mut RuntimeReadinessGate,
    reported: &mut Option<EngineHealth>,
    engine: &str,
    endpoint: &str,
) -> WatchEnd {
    let request = HealthCheckRequest {
        service: SERVER.to_string(),
    };
    let mut stream = match client.watch(request).await {
        Ok(response) => response.into_inner(),
        Err(status) => return ended(status, false),
    };
    let mut delivered = false;
    loop {
        let response = match stream.message().await {
            Ok(Some(response)) => response,
            Ok(None) => {
                return WatchEnd::Lost {
                    reason: "the engine ended the health watch".to_string(),
                    delivered,
                };
            }
            Err(status) => return ended(status, delivered),
        };
        delivered = true;
        let health = EngineHealth::from_status(response.status);
        let was_ready = !gate.is_closed();
        match health {
            EngineHealth::Serving => gate.open(),
            EngineHealth::NotServing => gate.close(),
        }
        if *reported != Some(health) {
            *reported = Some(health);
            log_health(health, response.status, was_ready, engine, endpoint);
        }
        if health != EngineHealth::Serving {
            // Returning drops `stream`, which cancels the call on the engine.
            return WatchEnd::Released;
        }
    }
}

fn log_health(health: EngineHealth, status: i32, was_ready: bool, engine: &str, endpoint: &str) {
    let status =
        ServingStatus::try_from(status).map_or("unrecognized", |status| status.as_str_name());
    match health {
        EngineHealth::Serving => {
            tracing::info!(engine, endpoint, status, "the engine is serving")
        }
        EngineHealth::NotServing if was_ready => tracing::info!(
            engine,
            endpoint,
            status,
            "the engine is not serving; sidecar readiness is withdrawn"
        ),
        EngineHealth::NotServing => tracing::info!(
            engine,
            endpoint,
            status,
            "the engine is not serving yet; sidecar readiness waits for it"
        ),
    }
}

/// A watch that ended with `status`. The reason names the failure by its
/// code, message and source only: the status's metadata carries per-response
/// headers such as `date`, which would make every retry read as a new failure.
fn ended(status: tonic::Status, delivered: bool) -> WatchEnd {
    let reason = if status.code() == tonic::Code::Unimplemented {
        "the engine does not serve grpc.health.v1".to_string()
    } else {
        let mut reason = format!("{:?}: {}", status.code(), status.message());
        if let Some(source) = std::error::Error::source(&status) {
            reason = format!("{reason}: {}", format_error_chain(source));
        }
        reason
    };
    WatchEnd::Lost { reason, delivered }
}

#[cfg(test)]
mod tests {
    use super::{EngineHealth, ServingStatus};

    #[test]
    fn unknown_statuses_leave_readiness_closed() {
        for (status, expected) in [
            (
                ServingStatus::ServiceUnknown as i32,
                EngineHealth::NotServing,
            ),
            (ServingStatus::Unknown as i32, EngineHealth::NotServing),
            (42, EngineHealth::NotServing),
        ] {
            assert_eq!(EngineHealth::from_status(status), expected, "{status}");
        }
    }
}
