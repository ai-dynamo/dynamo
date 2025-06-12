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

use crate::mocker::protocols::{DirectRequest, MockEngineArgs, OutputSignal};
use crate::mocker::scheduler::Scheduler;

use dynamo_runtime::{
    engine::AsyncEngineContextProvider,
    pipeline::{async_trait, AsyncEngine, Error, ManyOut, ResponseStream, SingleIn},
    protocols::annotated::Annotated,
};

use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
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
}

impl MockVllmEngine {
    /// Create a new MockVllmEngine with the given parameters
    pub fn new(args: MockEngineArgs) -> Self {
        let mut schedulers = Vec::new();
        let active_requests = Arc::new(Mutex::new(
            HashMap::<Uuid, mpsc::Sender<OutputSignal>>::new(),
        ));

        // Create multiple schedulers and their background tasks
        for dp_rank in 0..args.dp_size {
            // Create a shared output channel that this scheduler will use
            let (output_tx, output_rx) = mpsc::channel::<OutputSignal>(1024);

            let scheduler = Scheduler::new(
                args.clone(),
                Some(dp_rank),
                Some(output_tx),
                None, // No global cancellation token
            );

            schedulers.push(scheduler);

            // Spawn a background task for this scheduler to distribute token notifications to active requests
            let output_rx = Arc::new(Mutex::new(output_rx));
            let active_requests_clone = active_requests.clone();

            tokio::spawn(async move {
                loop {
                    let signal = {
                        let mut rx = output_rx.lock().await;
                        match rx.recv().await {
                            Some(signal) => signal,
                            None => break, // Channel closed
                        }
                    };

                    // Notify the specific request that a token was generated
                    let active = active_requests_clone.lock().await;
                    if let Some(request_tx) = active.get(&signal.uuid) {
                        let _ = request_tx.send(signal).await;
                    }
                }
            });
        }

        Self {
            schedulers,
            active_requests,
            dp_size: args.dp_size,
        }
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

        let engine = MockVllmEngine::new(args);

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
