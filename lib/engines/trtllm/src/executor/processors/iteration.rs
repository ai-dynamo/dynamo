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

use dynamo_llm::kv_router::protocols::ForwardPassMetrics;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    thread,
};
use tokio::sync::broadcast;

use super::*;

const CHANNEL_CAPACITY: usize = 256;
type ChannelType = broadcast::Sender<Arc<ForwardPassMetrics>>;
pub type SubscriptionChannel = broadcast::Receiver<Arc<ForwardPassMetrics>>;

pub struct IterationProcessor {
    handle: thread::JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
    channel: Weak<ChannelType>,
}

impl IterationProcessor {
    /// Creates a new KV Event Processor
    pub fn new(state: ProcessorState) -> Self {
        // Shutdown Token
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Event Channel
        let channel = Arc::new(broadcast::channel(CHANNEL_CAPACITY).0);
        let channel_clone = channel.clone();

        let handle = std::thread::spawn(move || {
            process_events(state, shutdown_clone, channel_clone);
        });

        IterationProcessor {
            handle,
            shutdown,
            channel: Arc::downgrade(&channel),
        }
    }

    /// Subscribes to the KV Events broadcast channel
    /// Multiple subscribers can be created to monitor the KV Events
    pub fn subscribe(&self) -> Option<SubscriptionChannel> {
        self.channel.upgrade().map(|channel| channel.subscribe())
    }

    /// Joins the thread and waits for it to finish
    pub fn join(self) -> thread::Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        self.handle.join()
    }
}

fn process_events(state: ProcessorState, shutdown: Arc<AtomicBool>, channel: Arc<ChannelType>) {
    loop {
        // this blocks the thread until the response is ready or the server is shutdown
        let iters = state
            .executor
            .await_iter_stats()
            .expect("Failed to await responses");

        let should_shutdown = shutdown.load(Ordering::Relaxed);

        for iter in iters.stats {
            tracing::debug!("Received iteration stats: {:?}", iter);
            let iter = Arc::new(iter);

            if let Err(e) = channel.send(iter) {
                tracing::debug!("Failed to send message to channel: {:?}", e);
                break;
            }
        }

        if should_shutdown {
            tracing::debug!("Shutting down KV Event Processor");
            break;
        }
    }
}
