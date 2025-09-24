// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::sync::Arc;

use super::{bus, config};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

// Use the actual CompletionUsage type
pub type CompletionUsage = dynamo_async_openai::types::CompletionUsage; // Fixed import

#[derive(Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuditMode {
    UsageOnly,
    Full,
}

#[derive(Serialize, Clone)]
pub struct AuditRecord {
    pub schema_version: u32, // 1
    pub request_id: String,
    pub requested_streaming: bool, // original inbound flag
    pub mode: AuditMode,
    pub model: String,
    pub usage: Option<CompletionUsage>, // always when enabled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<Arc<NvCreateChatCompletionRequest>>, // only when Full
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<Arc<NvCreateChatCompletionResponse>>, // only when Full
}

pub struct AuditHandle {
    mode: AuditMode,
    requested_streaming: bool,
    request_id: String,
    model: String,
    usage: Option<CompletionUsage>,
    req_full: Option<Arc<NvCreateChatCompletionRequest>>,
    resp_full: Option<Arc<NvCreateChatCompletionResponse>>,
}

impl AuditHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }
    pub fn mode(&self) -> AuditMode {
        self.mode
    }

    pub fn add_usage(&mut self, usage: CompletionUsage) {
        self.usage = Some(usage);
    }

    pub fn set_request(&mut self, req: Arc<NvCreateChatCompletionRequest>) {
        if self.mode == AuditMode::Full {
            self.req_full = Some(req);
        }
    }
    pub fn set_response(&mut self, resp: Arc<NvCreateChatCompletionResponse>) {
        if self.mode == AuditMode::Full {
            self.resp_full = Some(resp);
        }
    }

    /// Emit exactly once (publishes to the bus; sinks do I/O).
    pub fn emit(self) {
        let rec = AuditRecord {
            schema_version: 1,
            request_id: self.request_id,
            requested_streaming: self.requested_streaming,
            mode: self.mode,
            model: self.model,
            usage: self.usage,
            request: self.req_full,
            response: self.resp_full,
        };
        bus::publish(rec);
    }
}

pub fn create_handle(req: &NvCreateChatCompletionRequest, request_id: &str) -> Option<AuditHandle> {
    if !config::policy().enabled {
        return None;
    }
    let store = req.inner.store.unwrap_or(false);
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let mode = if store {
        AuditMode::Full
    } else {
        AuditMode::UsageOnly
    };
    let model = req.inner.model.clone();

    Some(AuditHandle {
        mode,
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        usage: None,
        req_full: None,
        resp_full: None,
    })
}
