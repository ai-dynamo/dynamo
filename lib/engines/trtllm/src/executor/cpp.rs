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

use anyhow::{Context, Error, Result};
use bindings::nvllm_trt_engine_destroy;
use std::ffi::CStr;
use std::ffi::CString;
use std::ptr::NonNull;

use super::protocols;
use dynamo_llm::kv_router::protocols::{ForwardPassMetrics, KvCacheEvents};

mod bindings {
    #![allow(warnings, missing_docs)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use bindings::{
    nvllm_trt_engine, nvllm_trt_engine_await_iter_stats, nvllm_trt_engine_await_kv_events,
    nvllm_trt_engine_await_responses, nvllm_trt_engine_cancel_request, nvllm_trt_engine_create,
    nvllm_trt_engine_enqueue_request, nvllm_trt_engine_free_responses,
    nvllm_trt_engine_has_completed, nvllm_trt_engine_is_ready, nvllm_trt_engine_shutdown,
};

use super::config;

#[derive(Debug, Clone)]
pub struct Executor {
    engine: NonNull<nvllm_trt_engine>,
}

// nvllm_trt_engine is thread safe
// rust does not know that it is thread safe, so we have to tell it
unsafe impl Send for Executor {}
unsafe impl Sync for Executor {}

// The following implementation of ThreaadSafeEngine are the convenience methods used for call
// the C/C++ TensorRT API from Rust.
impl Executor {
    /// Creates a new instance of the TensorRT LLM engine and takes ownership of the pointer to
    /// the C/C++ TensorRT LLM engine object.
    ///
    /// Executor implements the Drop trait, so this object is an RAII object and will
    /// free the C/C++ TensorRT LLM engine object when it goes out of scope.
    pub fn new(config: config::ExecutorConfig) -> Result<Self> {
        let json = serde_json::to_string(&config)?;
        let c_config = CString::new(json)?;
        let engine = unsafe { nvllm_trt_engine_create(c_config.as_ptr()) };
        let engine = NonNull::new(engine)
            .ok_or_else(|| Error::msg("Failed to create nvllm_trt_engine".to_string()))?;
        Ok(Self { engine })
    }

    /// Checks if the engine has started asking for new work
    pub fn has_started(&self) -> bool {
        let result = unsafe { nvllm_trt_engine_is_ready(self.engine.as_ptr()) };
        if result != 0 {
            return true;
        }
        false
    }

    /// Checks if the engine has completed all work and shutdown
    pub fn has_completed(&self) -> bool {
        let result = unsafe { nvllm_trt_engine_has_completed(self.engine.as_ptr()) };
        if result != 0 {
            return true;
        }
        false
    }

    /// Enqueues a request to the engine
    /// The request it sent to the engine as a json encoded string; however, we reserve the right to change
    /// the encoding in the future.
    pub fn enqueue_request(&self, client_id: u64, request: CString) -> Result<u64> {
        tracing::trace!("enqueuing request to trtllm engine");
        let id = unsafe {
            nvllm_trt_engine_enqueue_request(self.engine.as_ptr(), client_id, request.as_ptr())
        };
        if id == 0 {
            return Err(Error::msg("Failed to enqueue request".to_string()));
        }
        Ok(id)
    }

    /// Block on [`nvllm_trt_engine_await_responses`] until a set response is received
    /// If the server shutdown, the list of Responses will be empty
    pub fn await_responses(&self) -> Result<protocols::Responses> {
        let responses;
        unsafe {
            let ptr = nvllm_trt_engine_await_responses(self.engine.as_ptr());
            let c_str = CStr::from_ptr(ptr);
            let bytes = c_str.to_bytes();
            responses = serde_json::from_slice(bytes).context("Failed to parse responses")?;
            nvllm_trt_engine_free_responses(ptr);
        }
        Ok(responses)
    }

    pub fn await_kv_events(&self) -> Result<KvCacheEvents> {
        let events: KvCacheEvents;
        unsafe {
            let ptr = nvllm_trt_engine_await_kv_events(self.engine.as_ptr());
            if ptr.is_null() {
                return Err(Error::msg(
                    "No KvEvents will be emitted for this model".to_string(),
                ));
            }
            let c_str = CStr::from_ptr(ptr);
            let bytes = c_str.to_bytes();
            events = serde_json::from_slice(bytes)
                .context(format!("Failed to parse kv cache events: {:?}", c_str))?;
            nvllm_trt_engine_free_responses(ptr);
        }
        Ok(events)
    }

    #[allow(dead_code)]
    pub fn await_iter_stats(&self) -> Result<protocols::stats::IterStats> {
        let stats: Vec<ForwardPassMetrics>;
        unsafe {
            let ptr = nvllm_trt_engine_await_iter_stats(self.engine.as_ptr());
            if ptr.is_null() {
                return Err(Error::msg(
                    "No iter stats will be emitted for this model".to_string(),
                ));
            }
            let c_str = CStr::from_ptr(ptr);
            let bytes = c_str.to_bytes();
            stats = serde_json::from_slice(bytes)
                .context(format!("Failed to parse iter stats: {:?}", c_str))?;
            nvllm_trt_engine_free_responses(ptr);
        }
        let stats = protocols::stats::IterStats { stats };
        Ok(stats)
    }

    /// Cancels a request by its request_id
    pub fn cancel_request(&self, request_id: u64) {
        unsafe { nvllm_trt_engine_cancel_request(self.engine.as_ptr(), request_id) };
    }

    /// Shuts down the engine
    pub fn shutdown(&self) {
        unsafe { nvllm_trt_engine_shutdown(self.engine.as_ptr()) };
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        unsafe {
            nvllm_trt_engine_shutdown(self.engine.as_ptr());
            nvllm_trt_engine_destroy(self.engine.as_ptr());
        }
    }
}
