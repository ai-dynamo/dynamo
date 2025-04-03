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

use dynamo_llm::kv_router::protocols::KvCacheEvents;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    thread,
};
use tokio::sync::broadcast;

use super::*;

const KV_EVENT_CHANNEL_CAPACITY: usize = 65536;
type EventChannelType = broadcast::Sender<KvCacheEvents>;
pub type KvEventSubscriptionChannel = broadcast::Receiver<KvCacheEvents>;

pub struct KvEventProcessor {
    handle: thread::JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
    channel: Weak<EventChannelType>,
}

impl KvEventProcessor {
    /// Creates a new KV Event Processor
    pub fn new(state: ProcessorState) -> Self {
        // Shutdown Token
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        // Event Channel
        let channel = Arc::new(broadcast::channel(KV_EVENT_CHANNEL_CAPACITY).0);
        let channel_clone = channel.clone();

        let handle = std::thread::spawn(move || {
            process_events(state, shutdown_clone, channel_clone);
        });

        KvEventProcessor {
            handle,
            shutdown,
            channel: Arc::downgrade(&channel),
        }
    }

    /// Subscribes to the KV Events broadcast channel
    /// Multiple subscribers can be created to monitor the KV Events
    pub fn subscribe(&self) -> Option<broadcast::Receiver<KvCacheEvents>> {
        self.channel.upgrade().map(|channel| channel.subscribe())
    }

    /// Joins the thread and waits for it to finish
    pub fn join(self) -> thread::Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        self.handle.join()
    }
}

fn process_events(
    state: ProcessorState,
    shutdown: Arc<AtomicBool>,
    channel: Arc<EventChannelType>,
) {
    loop {
        // this blocks the thread until the response is ready or the server is shutdown
        let mut message = state
            .executor
            .await_kv_events()
            .expect("Failed to await responses");

        let should_shutdown = message.shutdown || shutdown.load(Ordering::Relaxed);

        message.shutdown = should_shutdown;

        if let Err(e) = channel.send(message) {
            tracing::debug!("Failed to send message to channel: {:?}", e);
        }

        if should_shutdown {
            tracing::debug!("Shutting down KV Event Processor");
            break;
        }
    }
}
