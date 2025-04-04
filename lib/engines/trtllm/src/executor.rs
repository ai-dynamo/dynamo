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

mod cpp;
mod engine;
mod processors;

// pub mod protos {
//     include!(concat!(env!("OUT_DIR"), "/nvidia.nvllm.trt.proto.rs"));
// }
pub mod protocols;

pub mod config;

use anyhow::Result;
use std::{
    collections::HashMap,
    ffi::CString,
    sync::{atomic::AtomicU64, Arc, Mutex, OnceLock, Weak},
};
use tokio::sync::mpsc;

use processors::{
    IterationProcessor, IterationStatsSubscriptionChannel, KvEventProcessor,
    KvEventSubscriptionChannel, ProcessorState, ResponseProcessor,
};

pub struct Executor {
    executor: Arc<cpp::Executor>,
    next_id: AtomicU64,
    response_queues: ResponseQueues,
    response_processor: OnceLock<ResponseProcessor>,
    kv_event_processor: OnceLock<KvEventProcessor>,
    iteration_processor: OnceLock<IterationProcessor>,
}

type ResponseQueues = Arc<Mutex<HashMap<u64, mpsc::Sender<Result<protocols::Output>>>>>;

impl Executor {
    pub fn from_model_path<P: ToString>(model_path: P) -> Result<Self> {
        let config = config::ExecutorConfig::new(model_path.to_string());
        Self::new(config)
    }

    pub fn new(config: config::ExecutorConfig) -> Result<Self> {
        Ok(Self {
            executor: Arc::new(cpp::Executor::new(config)?),
            next_id: AtomicU64::new(0),
            response_queues: Arc::new(Mutex::new(HashMap::new())),
            response_processor: OnceLock::new(),
            kv_event_processor: OnceLock::new(),
            iteration_processor: OnceLock::new(),
        })
    }

    pub fn has_started(&self) -> bool {
        self.executor.has_started()
    }

    pub fn has_completed(&self) -> bool {
        self.executor.has_completed()
    }

    pub fn enqueue_request(&self, request: protocols::Request) -> Result<ExecutionContext> {
        let client_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let (tx, rx) = mpsc::channel(128);

        self.response_queues
            .lock()
            .expect("response_queues lock poisoned")
            .insert(client_id, tx);

        let json = serde_json::to_string(&request)?;
        let str = CString::new(json)?;

        let request_id = self
            .executor
            .enqueue_request(client_id, str)
            .inspect_err(|_| {
                self.response_queues
                    .lock()
                    .expect("response_queues lock poisoned")
                    .remove(&client_id);
            })?;

        println!("request_id: {}", request_id);

        Ok(ExecutionContext {
            request_id,
            response_rx: Some(rx),
            executor: Arc::downgrade(&self.executor),
        })
    }

    pub fn cancel_request(&self, client_id: u64) {
        self.executor.cancel_request(client_id)
    }

    /// Start a background task to process responses from the TensorRT LLM AsyncEngine
    pub fn start_response_processor(&self) {
        self.response_processor.get_or_init(|| {
            ResponseProcessor::new(self.create_processor(), self.response_queues.clone())
        });
    }

    /// Starts a background task to process kv events
    /// TODO - check the TensorRT LLM config and only start this if the server is configured to send kv events
    pub fn start_kv_event_processor(&self) {
        self.kv_event_processor
            .get_or_init(|| KvEventProcessor::new(self.create_processor()));
    }

    /// Starts a background task to process forward pass / iteration statistics
    pub fn start_iteration_metrics_processor(&self) {
        self.iteration_processor
            .get_or_init(|| IterationProcessor::new(self.create_processor()));
    }

    /// Subscribes to the KV Events broadcast channel
    pub fn subscribe_to_kv_events(&self) -> Result<KvEventSubscriptionChannel> {
        self.kv_event_processor
            .get_or_init(|| KvEventProcessor::new(self.create_processor()))
            .subscribe()
            .ok_or(anyhow::anyhow!("Failed to subscribe to KV events"))
    }

    pub fn subscribe_to_iteration_stats(&self) -> Result<IterationStatsSubscriptionChannel> {
        self.iteration_processor
            .get_or_init(|| IterationProcessor::new(self.create_processor()))
            .subscribe()
            .ok_or(anyhow::anyhow!("Failed to subscribe to iteration stats"))
    }

    /// Issues a shutdown request to the TensorRT LLM AsyncEngine
    /// This is a blocking call. After the async engine has shutdown each background processor/thread/task
    /// will be joined and the resources will be released.
    pub fn shutdown(&mut self) {
        self.executor.shutdown();
        self.response_processor.take().map(|p| p.join());
        self.kv_event_processor.take().map(|p| p.join());
        self.iteration_processor.take().map(|p| p.join());
    }

    /// Constructs a new ProcessorState instance which packages up any bits from the Executor for the processor task
    fn create_processor(&self) -> ProcessorState {
        ProcessorState::new(self.executor.clone())
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct ExecutionContext {
    /// Internal TensorRT LLM request_id; used to cancel the request
    /// This value is present in the response but because we do not know it before hand, it is only used for cancellation
    request_id: u64,

    /// Hold a weak pointer to the executor for cancellation
    executor: Weak<cpp::Executor>,

    /// Response stream associated with this request
    response_rx: Option<mpsc::Receiver<Result<protocols::Output>>>,
}

impl ExecutionContext {
    pub fn cancel(&self) {
        if let Some(executor) = self.executor.upgrade() {
            executor.cancel_request(self.request_id);
        }
    }

    pub fn take_response_rx(&mut self) -> Option<mpsc::Receiver<Result<protocols::Output>>> {
        self.response_rx.take()
    }
}
