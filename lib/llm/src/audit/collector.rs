// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast::Receiver;
use tokio::{select, time};

use super::event::AuditEvent;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

#[derive(Default)]
struct Pending {
    req: Option<Arc<NvCreateChatCompletionRequest>>,
    resp: Option<Arc<NvCreateChatCompletionResponse>>,
    first_seen_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn normalize_id(id: &str) -> String {
    id.strip_prefix("chatcmpl-").unwrap_or(id).to_string()
}

pub async fn spawn_stderr(mut rx: Receiver<Arc<AuditEvent>>) {
    let mut pending: HashMap<String, Pending> = HashMap::new();
    let mut sweep = time::interval(Duration::from_secs(60));
    const TTL_MS: u64 = 5 * 60 * 1000; // 5 minutes; tune as needed

    loop {
        select! {
            msg = rx.recv() => {
                match msg {
                    Ok(evt) => match &*evt {
                        AuditEvent::Request { id, req } => {
                            let normalized_id = get_or_create_entry(id, &mut pending);
                            let e = pending.get_mut(&normalized_id).unwrap();
                            e.req = Some(req.clone());

                            try_emit_and_cleanup(&normalized_id, &mut pending);
                        }
                        AuditEvent::Response { id, resp } => {
                            let normalized_id = get_or_create_entry(id, &mut pending);
                            let e = pending.get_mut(&normalized_id).unwrap();
                            e.resp = Some(resp.clone());

                            try_emit_and_cleanup(&normalized_id, &mut pending);
                        }
                    },
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(dropped = n, "audit bus lagged; dropped events");
                    }
                }
            }
            _ = sweep.tick() => {
                let cutoff = now_ms().saturating_sub(TTL_MS);
                pending.retain(|_id, e| {
                    e.first_seen_ms > cutoff
                });
            }
        }
    }
}

fn get_or_create_entry(id: &str, pending: &mut HashMap<String, Pending>) -> String {
    let normalized_id = normalize_id(id);
    let e = pending.entry(normalized_id.clone()).or_default();
    if e.first_seen_ms == 0 {
        e.first_seen_ms = now_ms();
    }
    normalized_id
}

fn try_emit_and_cleanup(id: &str, pending: &mut HashMap<String, Pending>) {
    if let Some(entry) = pending.get(id)
        && let (Some(req), Some(resp)) = (&entry.req, &entry.resp)
    {
        emit_joined(id, req, resp);
        pending.remove(id);
    }
}

fn emit_joined(
    id: &str,
    req: &Arc<NvCreateChatCompletionRequest>,
    resp: &Arc<NvCreateChatCompletionResponse>,
) {
    let req_json = serde_json::to_string(&**req).unwrap_or_else(|_| "{}".into());
    let resp_json = serde_json::to_string(&**resp).unwrap_or_else(|_| "{}".into());

    tracing::info!(
        target = "dynamo_llm::audit",
        log_type = "audit",
        schema_version = 1.0,
        request_id = %id,
        request = req_json,
        response = %resp_json,
        "Audit log (joined req+resp)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_normalize_id_consistency() {
        // Critical for joining - request and response IDs must normalize to same value
        let req_id = "407508e2-1ee3-423e-bda0-29a18626cbdb";
        let resp_id = "chatcmpl-407508e2-1ee3-423e-bda0-29a18626cbdb";
        assert_eq!(normalize_id(req_id), normalize_id(resp_id));
    }
    #[test]
    fn test_get_or_create_entry() {
        // Test the entry creation logic
        let mut pending = HashMap::new();

        let id1 = get_or_create_entry("test-123", &mut pending);
        assert_eq!(id1, "test-123");
        assert!(pending.contains_key("test-123"));

        // Second call should return same normalized ID
        let id2 = get_or_create_entry("test-123", &mut pending);
        assert_eq!(id1, id2);
        assert_eq!(pending.len(), 1);
    }
}
