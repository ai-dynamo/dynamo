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

use std::sync::Arc;
use tokio::sync::broadcast;

use crate::block_manager::events::{
    EventManager, EventPublisher, EventReleaseManager, EventType, RegistrationHandle,
};
use crate::tokens::SequenceHash;

/// An event manager that propagates cache hits to the offload manager.
pub struct OffloadEventManager {
    tx: broadcast::Sender<Vec<SequenceHash>>,
}

impl Default for OffloadEventManager {
    fn default() -> Self {
        Self::new()
    }
}

impl OffloadEventManager {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(1024);
        Self { tx }
    }

    pub fn receiver(&self) -> broadcast::Receiver<Vec<SequenceHash>> {
        self.tx.subscribe()
    }
}

impl EventPublisher for OffloadEventManager {
    fn publish(&self, handles: Vec<Arc<RegistrationHandle>>, event_type: EventType) {
        if event_type != EventType::CacheHit {
            return;
        }

        let _ = self.tx.send(
            handles
                .iter()
                .map(|handle| handle.sequence_hash())
                .collect(),
        );
    }
}

impl EventReleaseManager for OffloadEventManager {
    fn block_release(&self, _registration_handle: &RegistrationHandle) {}
}

impl EventManager for OffloadEventManager {}
