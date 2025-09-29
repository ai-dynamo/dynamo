// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::sync::Arc;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

#[derive(Serialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32,
    pub request_id: String,
    pub requested_streaming: bool,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>,
}

pub struct AuditHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    req_full: Option<Arc<NvCreateChatCompletionRequest>>,
    resp_full: Option<Arc<NvCreateChatCompletionResponse>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn set_request(&mut self, req: Arc<NvCreateChatCompletionRequest>) {
        self.req_full = Some(req);
    }
    pub fn set_response(&mut self, resp: Arc<NvCreateChatCompletionResponse>) {
        self.resp_full = Some(resp);
    }

    /// Emit exactly once (publishes to the bus; sinks do I/O).
    pub fn emit(self) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            model: self.model,
            request: self.req_full,
            response: self.resp_full,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    if !config::policy().enabled || !req.inner.store.unwrap_or(false) {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();

    Some(AuditHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        req_full: None,
        resp_full: None,
    })
}
