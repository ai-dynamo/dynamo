// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    let policy = config::policy();
    if !policy.enabled {
        return None;
    }
    // If force_logging is enabled, ignore the store flag
    if !policy.force_logging && !req.inner.store.unwrap_or(false) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use temp_env::with_vars;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    /// Test that DYN_AUDIT_FORCE_LOGGING=true bypasses store=false
    /// When force logging is enabled, audit handle should be created even when store=false
    #[test]
    fn test_force_logging_bypasses_store() {
        with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("stderr")),
                ("DYN_AUDIT_FORCE_LOGGING", Some("true")),
            ],
            || {
                // Create request with store=false
                let request = create_test_request("test-model", false);
                let handle = create_handle(&request, "test-id");

                assert!(
                    handle.is_some(),
                    "When DYN_AUDIT_FORCE_LOGGING=true, handle should be created even with store=false"
                );
            },
        );
    }
}
