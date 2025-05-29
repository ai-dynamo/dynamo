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

use super::request::OffloadRequest;
use crate::block_manager::{BlockMetadata, Storage};
use crate::tokens::SequenceHash;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use tokio::sync::broadcast;

/// A simple priority queue for offloading requests.
pub struct OffloadQueue<Source: Storage, Metadata: BlockMetadata> {
    queue: BTreeSet<Arc<OffloadRequest<Source, Metadata>>>,
    lookup: HashMap<SequenceHash, Arc<OffloadRequest<Source, Metadata>>>,
    cache_hit_receiver: broadcast::Receiver<Vec<SequenceHash>>,
}

impl<Source: Storage, Metadata: BlockMetadata> OffloadQueue<Source, Metadata> {
    pub fn new(cache_hit_receiver: broadcast::Receiver<Vec<SequenceHash>>) -> Self {
        Self {
            queue: BTreeSet::new(),
            lookup: HashMap::new(),
            cache_hit_receiver,
        }
    }

    pub fn push(&mut self, request: OffloadRequest<Source, Metadata>) {
        let request = Arc::new(request);
        self.lookup.insert(request.sequence_hash, request.clone());
        self.queue.insert(request);
    }

    pub fn pop(&mut self) -> Option<Arc<OffloadRequest<Source, Metadata>>> {
        if let Some(request) = self.queue.pop_first() {
            self.lookup.remove(&request.sequence_hash);
            Some(request)
        } else {
            None
        }
    }

    pub async fn cache_hit_worker(&mut self) {
        while let Ok(cache_hits) = self.cache_hit_receiver.recv().await {
            for sequence_hash in cache_hits {
                if let Some(mut request) = self.lookup.remove(&sequence_hash) {
                    self.queue.remove(&request);
                    if request.block.upgrade().is_some() {
                        // We'd probably want something more fancy than this.
                        // But this works for now. Just increment priority by 1.

                        Arc::get_mut(&mut request).unwrap().key.priority += 1;
                        self.queue.insert(request.clone());
                        self.lookup.insert(sequence_hash, request);
                    }
                }
            }
        }
    }
}
