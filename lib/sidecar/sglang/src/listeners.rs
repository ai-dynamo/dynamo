// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request dispatch over the SGLang Rust listeners of one node. A
//! listener runs every request on its own DP rank, so the router's rank
//! picks the listener.

use std::time::Duration;

use dynamo_backend_common::{DisaggregationMode, DynamoError, PreprocessedRequest};
use dynamo_sidecar_common::GrpcEndpoint;

use crate::client::{AbortRpc, Pool};
use crate::native_http::NativeHttp;
use crate::protocol::validate_dp_rank;

/// Where one request runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    /// The listener, by its index in rank order.
    pub index: usize,
    /// The rank a prefill request's rendezvous carries, in its bootstrap
    /// port and room. A single-rank deployment carries none.
    pub prefill_rank: Option<u32>,
}

/// One Rust listener of the node.
pub(crate) struct Listener {
    pub dp_rank: u32,
    pub endpoint: GrpcEndpoint,
    pub pool: Pool,
    /// Native `/generate`, when every listener of the node serves it.
    pub native_http: Option<NativeHttp>,
    /// Its prefill bootstrap registry's port (its HTTP port).
    pub bootstrap_port: Option<u16>,
}

/// The node's listeners, by DP rank `[start_rank, start_rank + len)`.
pub(crate) struct NodeListeners {
    pub start_rank: u32,
    /// Every DP rank of the deployment.
    pub global_dp_size: u32,
    pub listeners: Vec<Listener>,
}

impl NodeListeners {
    pub(crate) fn new(start_rank: u32, global_dp_size: u32, listeners: Vec<Listener>) -> Self {
        assert!(!listeners.is_empty(), "a node has at least one listener");
        Self {
            start_rank,
            global_dp_size,
            listeners,
        }
    }

    fn ranks(&self) -> std::ops::Range<u32> {
        self.start_rank..self.start_rank + self.listeners.len() as u32
    }

    /// Choose the listener for `request` by the rank the router named. A
    /// rank this node does not serve is refused before any stream opens,
    /// which keeps the worker routable.
    pub(crate) fn select(
        &self,
        request: &PreprocessedRequest,
        mode: DisaggregationMode,
    ) -> Result<Dispatch, DynamoError> {
        let index = match validate_dp_rank(request, mode, self.ranks())? {
            Some(rank) => (rank - self.start_rank) as usize,
            None => 0,
        };
        let carries_rank = mode.is_prefill() && self.global_dp_size > 1;
        Ok(Dispatch {
            index,
            prefill_rank: carries_rank.then_some(self.listeners[index].dp_rank),
        })
    }

    /// Abort `request_id` on every listener: it runs on exactly one, and the
    /// others answer that nothing matched. Each RPC is bounded by `timeout`.
    pub(crate) async fn abort(&self, request_id: &str, timeout: Duration) {
        let aborts = self.listeners.iter().map(|listener| async move {
            AbortRpc::new(listener.pool.control_client())
                .abort(request_id, timeout)
                .await
        });
        futures::future::join_all(aborts).await;
    }
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::engine::RoutingHints;
    use dynamo_backend_common::{
        BackendError, ErrorType, OutputOptions, SamplingOptions, StopConditions,
    };

    use super::*;

    fn endpoint(port: u16) -> GrpcEndpoint {
        GrpcEndpoint::parse(&format!("127.0.0.1:{port}"), "test").unwrap()
    }

    /// A node with ranks `[start, start + count)` of a `global`-rank
    /// deployment, whose listeners never connect.
    fn node(start: u32, count: u32, global: u32) -> NodeListeners {
        let listeners = (0..count)
            .map(|offset| {
                let endpoint = endpoint(1 + offset as u16);
                Listener {
                    dp_rank: start + offset,
                    endpoint: endpoint.clone(),
                    pool: Pool::lazy(&endpoint),
                    native_http: None,
                    bootstrap_port: None,
                }
            })
            .collect();
        NodeListeners::new(start, global, listeners)
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .build()
            .unwrap()
    }

    fn routed(dp_rank: Option<u32>, prefill_dp_rank: Option<u32>) -> PreprocessedRequest {
        let mut request = request();
        request.routing = Some(RoutingHints {
            dp_rank,
            prefill_dp_rank,
            ..Default::default()
        });
        request
    }

    fn assert_invalid(error: DynamoError, message: &str) {
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }

    #[tokio::test]
    async fn routing_rank_selects_listener() {
        use DisaggregationMode::{Aggregated, Decode, Prefill};
        // Node 1 of two: ranks [2, 4) of four.
        let listeners = node(2, 2, 4);
        for (mode, request, rank, prefill_rank) in [
            (Aggregated, routed(Some(3), None), 3, None),
            (Decode, routed(Some(2), Some(3)), 2, None),
            // Prefill takes its own rank first.
            (Prefill, routed(Some(2), Some(3)), 3, Some(3)),
            (Aggregated, request(), 2, None),
            (Prefill, request(), 2, Some(2)),
        ] {
            let index = (rank - 2) as usize;
            assert_eq!(
                listeners.select(&request, mode).unwrap(),
                Dispatch {
                    index,
                    prefill_rank,
                },
                "{mode:?}"
            );
        }
        for rank in [1, 4] {
            assert_invalid(
                listeners
                    .select(&routed(Some(rank), None), Aggregated)
                    .unwrap_err(),
                "serves DP ranks [2, 4)",
            );
        }
        assert_invalid(
            listeners
                .select(&routed(None, Some(4)), Prefill)
                .unwrap_err(),
            "serves DP ranks [2, 4)",
        );
        // A single-rank deployment carries no rank.
        assert_eq!(
            node(0, 1, 1)
                .select(&request(), Prefill)
                .unwrap()
                .prefill_rank,
            None
        );
    }
}
