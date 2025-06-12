// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! MockSchedulerEngine - AsyncEngine wrapper around the Scheduler
//!
//! This module provides an AsyncEngine implementation that wraps the Scheduler
//! to provide streaming token generation with realistic timing simulation.

use crate::kv_router::publisher::WorkerMetricsPublisher;
use crate::mocker::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::mocker::scheduler::Scheduler;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Component,
    engine::AsyncEngineContextProvider,
    pipeline::{async_trait, AsyncEngine, Error, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
    Result,
};

use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::time::{interval, Duration};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

/// Generate a random printable character
fn generate_random_char() -> String {
    let mut rng = rand::rng();
    let selection = match rng.random_range(0..4) {
        0 => ('a'..='z').nth(rng.random_range(0..26)).unwrap(), // lowercase
        1 => ('A'..='Z').nth(rng.random_range(0..26)).unwrap(), // uppercase
        2 => ('0'..='9').nth(rng.random_range(0..10)).unwrap(), // digits
        _ => [' ', '.', ',', '!', '?'][rng.random_range(0..5)], // punctuation/space
    };
    selection.to_string()
}

/// AsyncEngine wrapper around the Scheduler that generates random character tokens
pub struct MockVllmEngine {
    schedulers: Vec<Scheduler>,
    active_requests: Arc<Mutex<HashMap<Uuid, mpsc::Sender<OutputSignal>>>>,
    dp_size: u32,
    cancel_token: CancellationToken,
}

impl MockVllmEngine {
    /// Create a new MockVllmEngine with the given parameters
    pub async fn new(
        args: MockEngineArgs,
        component: Option<Component>,
        cancel_token: Option<CancellationToken>,
    ) -> Result<Self> {
        let active_requests = Arc::new(Mutex::new(
            HashMap::<Uuid, mpsc::Sender<OutputSignal>>::new(),
        ));

        let cancel_token = cancel_token.unwrap_or_default();

        // Create schedulers and start their background tasks
        let schedulers =
            Self::start_schedulers(args.clone(), active_requests.clone(), cancel_token.clone());

        // Start metrics publishing tasks
        Self::start_metrics_publishing(&schedulers, component, cancel_token.clone()).await?;

        let engine = Self {
            schedulers,
            active_requests,
            dp_size: args.dp_size,
            cancel_token,
        };

        Ok(engine)
    }

    /// Create schedulers and spawn their background tasks for distributing token notifications
    fn start_schedulers(
        args: MockEngineArgs,
        active_requests: Arc<Mutex<HashMap<Uuid, mpsc::Sender<OutputSignal>>>>,
        cancel_token: CancellationToken,
    ) -> Vec<Scheduler> {
        let mut schedulers = Vec::new();

        // Create multiple schedulers and their background tasks
        for dp_rank in 0..args.dp_size {
            // Create a shared output channel that this scheduler will use
            let (output_tx, output_rx) = mpsc::channel::<OutputSignal>(1024);

            let scheduler = Scheduler::new(
                args.clone(),
                Some(dp_rank),
                Some(output_tx),
                Some(cancel_token.clone()),
            );

            schedulers.push(scheduler);

            // Spawn a background task for this scheduler to distribute token notifications to active requests
            let output_rx = Arc::new(Mutex::new(output_rx));
            let active_requests_clone = active_requests.clone();
            let cancel_token_cloned = cancel_token.clone();

            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        signal_result = async {
                            let mut rx = output_rx.lock().await;
                            rx.recv().await
                        } => {
                            let Some(signal) = signal_result else {
                                break; // Channel closed
                            };

                            // Notify the specific request that a token was generated
                            let active = active_requests_clone.lock().await;
                            if let Some(request_tx) = active.get(&signal.uuid) {
                                let _ = request_tx.send(signal).await;
                            }
                        }
                        _ = cancel_token_cloned.cancelled() => {
                            break;
                        }
                    }
                }
            });
        }

        schedulers
    }

    /// Start background tasks to poll and publish metrics every second
    async fn start_metrics_publishing(
        schedulers: &[Scheduler],
        component: Option<Component>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let metrics_publisher = Arc::new(WorkerMetricsPublisher::new()?);

        if let Some(comp) = component {
            metrics_publisher.create_endpoint(comp).await?;
        }

        for (dp_rank, scheduler) in schedulers.iter().enumerate() {
            let scheduler = scheduler.clone();
            let publisher = metrics_publisher.clone();
            let dp_rank = dp_rank as u32;
            let cancel_token = cancel_token.clone();

            tokio::spawn(async move {
                let mut interval = interval(Duration::from_secs(1));

                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            // Get metrics from scheduler
                            let metrics = scheduler.get_forward_pass_metrics().await;

                            // Publish metrics
                            if let Err(e) = publisher.publish(Arc::new(metrics)) {
                                tracing::warn!("Failed to publish metrics for DP rank {}: {}", dp_rank, e);
                            } else {
                                tracing::trace!("Published metrics for DP rank {}", dp_rank);
                            }
                        }
                        _ = cancel_token.cancelled() => {
                            tracing::info!("Metrics publishing cancelled for DP rank {}", dp_rank);
                            break;
                        }
                    }
                }
            });
        }

        Ok(())
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<DirectRequest>, ManyOut<Annotated<String>>, Error> for MockVllmEngine {
    async fn generate(
        &self,
        input: SingleIn<DirectRequest>,
    ) -> Result<ManyOut<Annotated<String>>, Error> {
        let (mut request, ctx) = input.into_parts();

        let dp_rank = request.dp_rank.unwrap_or(0);

        // Validate dp_rank
        if dp_rank >= self.dp_size {
            return Err(Error::msg(format!(
                "dp_rank {} is out of bounds for dp_size {}",
                dp_rank, self.dp_size
            )));
        }

        let request_uuid = ctx.id().parse().unwrap_or(Uuid::new_v4());
        request.uuid = Some(request_uuid);

        let (request_tx, mut request_rx) = mpsc::channel::<OutputSignal>(64);
        {
            let mut active = self.active_requests.lock().await;
            active.insert(request_uuid, request_tx);
        }

        // Send the request to the appropriate scheduler based on dp_rank
        self.schedulers[dp_rank as usize]
            .receive(request.clone())
            .await;

        // Create a simple channel for the stream
        let (stream_tx, stream_rx) = mpsc::channel::<Annotated<String>>(64);

        let active_requests = self.active_requests.clone();
        let async_context = ctx.context();
        let cancel_token = self.cancel_token.clone();

        // Spawn a task to handle the complex async logic
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some(signal) = request_rx.recv() => {
                        if signal.completed {
                            break;
                        }
                        let output = generate_random_char();
                        if stream_tx.send(Annotated::from_data(output)).await.is_err() {
                            break;
                        }
                    }

                    _ = async_context.stopped() => {
                        break;
                    }

                    _ = cancel_token.cancelled() => {
                        break;
                    }
                }
            }

            // Clean up: remove this request from active requests
            let mut active = active_requests.lock().await;
            active.remove(&request_uuid);
        });

        // Create a simple ReceiverStream which is naturally Send + Sync
        let stream = ReceiverStream::new(stream_rx);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::pipeline::Context;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_multiple_workers_with_token_limit() {
        const DP_SIZE: u32 = 2;
        const TOKENS_PER_REQUEST: usize = 20;

        // Create the MockVllmEngine using builder pattern
        let args = MockEngineArgs::builder()
            .speedup_ratio(10.0)
            .dp_size(DP_SIZE)
            .build()
            .unwrap();

        let engine = MockVllmEngine::new(args, None, None).await.unwrap();

        // Create 4 DirectRequests: 2 for worker 0, 2 for worker 1
        let requests = vec![
            DirectRequest {
                tokens: vec![1, 2, 3, 4],
                max_output_tokens: TOKENS_PER_REQUEST,
                uuid: None,
                dp_rank: Some(0),
            },
            DirectRequest {
                tokens: vec![5, 6, 7, 8],
                max_output_tokens: TOKENS_PER_REQUEST,
                uuid: None,
                dp_rank: Some(0),
            },
            DirectRequest {
                tokens: vec![9, 10, 11, 12],
                max_output_tokens: TOKENS_PER_REQUEST,
                uuid: None,
                dp_rank: Some(1),
            },
            DirectRequest {
                tokens: vec![13, 14, 15, 16],
                max_output_tokens: TOKENS_PER_REQUEST,
                uuid: None,
                dp_rank: Some(1),
            },
        ];

        // Generate streams and collect all tokens from each
        for request in requests {
            let ctx = Context::new(request);
            let stream = engine.generate(ctx).await.unwrap();

            let tokens: Vec<_> = stream.collect().await;

            // Verify each stream produces exactly the expected number of tokens
            assert_eq!(tokens.len(), TOKENS_PER_REQUEST);

            // Verify all tokens contain valid data
            for token in tokens {
                assert!(token.data.is_some());
            }
        }

        // Give a small delay to ensure cleanup tasks complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Verify that active_requests is empty (all requests cleaned up)
        let active_requests = engine.active_requests.lock().await;
        assert!(
            active_requests.is_empty(),
            "Active requests should be empty after streams complete"
        );
    }
}
