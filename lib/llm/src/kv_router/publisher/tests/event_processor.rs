// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// Mock publisher that collects published events
#[derive(Debug, Clone)]
struct MockPublisher {
    events: Arc<Mutex<Vec<RouterEvent>>>,
}

impl MockPublisher {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn get_events(&self) -> Vec<RouterEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl RouterEventSink for MockPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.events.lock().unwrap().push(event.clone());
        async { Ok(()) }
    }
}

fn local_gpu_event(event: KvCacheEvent) -> PlacementEvent {
    PlacementEvent::local_gpu(1, event)
}

fn local_host_event(event: KvCacheEvent) -> PlacementEvent {
    PlacementEvent::new(
        Placement::local_worker(1, event.dp_rank, StorageTier::HostPinned),
        event,
    )
}

mod chain;
mod metadata;
mod removed;
mod shutdown;
mod slow_input;
mod stored;
mod tier_switch;
mod valkey;
