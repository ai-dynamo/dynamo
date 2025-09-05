// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use std::sync::OnceLock;

fn audit_enabled() -> bool {
    static AUDIT_ON: OnceLock<bool> = OnceLock::new();
    *AUDIT_ON.get_or_init(|| {
        std::env::var("DYN_AUDIT_STDOUT")
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// Returns true iff auditing is enabled, store=true, and not streaming
pub fn should_audit_flags(store: bool, streaming: bool) -> bool {
    !streaming && audit_enabled() && store
}

// No env/store checks here; call only if should_audit() was true
pub fn log_stored_completion(
    request_id: &str,
    req: &NvCreateChatCompletionRequest,
    response_json: Value,
) {
    let ts_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let line = json!({
        "log_type": "audit",
        "ts_ms": ts_ms,
        "request_id": request_id,              // added
        "store_id": format!("store_{}", Uuid::new_v4().simple()),
        "model": req.inner.model,
        "store": true,
        "request": req.inner,
        "response": response_json,
        "streaming": false,
    })
    .to_string();

    println!("{line}");
}
