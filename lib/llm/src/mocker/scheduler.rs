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

use crate::mocker::kv_manager::KvManager;
use crate::mocker::protocols::DirectRequest;
use crate::mocker::sequence::ActiveSequence;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{interval, sleep, Duration};
use tokio::sync::CancellationToken;

// Change SchedulerState to not include KvManager
struct SchedulerState {
    waiting_requests: VecDeque<DirectRequest>,
    running_requests: VecDeque<ActiveSequence>,
}

/// Manages scheduling of requests using KvManager resources
pub struct Scheduler {
    state: Arc<Mutex<SchedulerState>>,
    kv_manager: Arc<KvManager>,  // Store KvManager directly in Scheduler
    watermark: f64,
    block_size: usize,
    chunk_size: usize,
    background_handle: Option<JoinHandle<()>>,
    request_tx: mpsc::Sender<DirectRequest>,
    cancellation_token: CancellationToken,
}

impl Scheduler {
    /// Create a new Scheduler with the given KvManager and watermark threshold
    pub fn new(kv_manager: KvManager, watermark: f64, block_size: Option<usize>, chunk_size: Option<usize>) -> Self {
        let state = Arc::new(Mutex::new(SchedulerState {
            waiting_requests: VecDeque::new(),
            running_requests: VecDeque::new(),
        }));
        
        let kv_manager = Arc::new(kv_manager);
        let block_size = block_size.unwrap_or(64);
        let chunk_size = chunk_size.unwrap_or(256);
        
        // Create channel for request handling
        let (request_tx, mut request_rx) = mpsc::channel::<DirectRequest>(1024);
        
        // Create cancellation token
        let cancellation_token = CancellationToken::new();
        let token_clone = cancellation_token.clone();
        
        // Create a clone for the background task
        let state_clone = state.clone();
        let kv_manager_clone = kv_manager.clone();
        let watermark_clone = watermark;
        let block_size_clone = block_size;
        let chunk_size_clone = chunk_size;
        
        // Spawn background task with cancellation token
        let background_handle = tokio::spawn(async move {
            let mut schedule_interval = interval(Duration::from_millis(10));
            let mut process_interval = interval(Duration::from_millis(100));
            
            loop {
                tokio::select! {
                    Some(request) = request_rx.recv() => {
                        let mut state = state_clone.lock().await;
                        state.waiting_requests.push_back(request);
                    }
                    
                    _ = schedule_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        
                        // Check if capacity is available for new requests
                        let current_capacity = kv_manager_clone.current_capacity().await;
                        let max_capacity = kv_manager_clone.max_capacity;
                        
                        // Only schedule if we're below the watermark threshold
                        if current_capacity < ((1.0 - watermark_clone) * max_capacity as f64) as usize {
                            // Try to schedule a waiting request
                            if let Some(request) = state_guard.waiting_requests.pop_front() {
                                let event_tx = kv_manager_clone.get_event_sender();
                                
                                // Create an ActiveSequence for the request
                                let sequence = ActiveSequence::new(
                                    request.hashes,
                                    Some(block_size_clone),
                                    Some(chunk_size_clone),
                                    request.max_output_tokens as usize,
                                    Some(event_tx),
                                );
                                
                                // Add the sequence to running requests
                                state_guard.running_requests.push_back(sequence);
                            }
                        }
                    }
                    
                    _ = process_interval.tick() => {
                        let mut state_guard = state_clone.lock().await;
                        let mut completed_indices = Vec::new();
                        
                        // Process each running request
                        for (idx, sequence) in state_guard.running_requests.iter_mut().enumerate() {
                            // Check if we've generated all tokens for this sequence
                            if sequence.generated_tokens >= sequence.max_output_tokens {
                                completed_indices.push(idx);
                                continue;
                            }
                            
                            // Generate one token
                            sequence.generate();
                            
                            // Check if we're done after generating
                            if sequence.generated_tokens >= sequence.max_output_tokens {
                                completed_indices.push(idx);
                            }
                        }
                        
                        // Remove completed sequences in reverse order to preserve indices
                        for idx in completed_indices.into_iter().rev() {
                            state_guard.running_requests.remove(idx);
                        }
                    }
                    
                    // Check for cancellation
                    _ = token_clone.cancelled() => {
                        break;
                    }
                }
            }
        });
        
        Self {
            state,
            kv_manager,
            watermark,
            block_size,
            chunk_size,
            background_handle: Some(background_handle),
            request_tx,
            cancellation_token,
        }
    }

    /// Add a new request to the waiting queue
    pub async fn receive_request(&self, request: DirectRequest) {
        let _ = self.request_tx.send(request).await;
    }

    /// Get the count of waiting requests
    pub async fn waiting_count(&self) -> usize {
        let state = self.state.lock().await;
        state.waiting_requests.len()
    }
    
    /// Get the count of running requests
    pub async fn running_count(&self) -> usize {
        let state = self.state.lock().await;
        state.running_requests.len()
    }
}

// Implement Clone for Scheduler to support sharing between tasks
impl Clone for Scheduler {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            kv_manager: self.kv_manager.clone(),
            watermark: self.watermark,
            block_size: self.block_size,
            chunk_size: self.chunk_size,
            background_handle: None,
            request_tx: self.request_tx.clone(),
            cancellation_token: self.cancellation_token.clone(),
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        // Only the original instance should cancel
        if self.background_handle.is_some() {
            self.cancellation_token.cancel();
        }
        
        // Abort the background task if this is the original instance
        if let Some(handle) = self.background_handle.take() {
            handle.abort();
        }
    }
}
