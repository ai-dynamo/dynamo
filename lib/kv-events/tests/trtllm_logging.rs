// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A separate test binary keeps the once-per-process warning counter isolated.

use std::io::Write;
use std::sync::{Arc, Mutex};

use dynamo_kv_events::zmq_wire::{RawKvEvent, decode_event_batch};
use rmp_serde::{from_slice, to_vec_named};
use serde_json::json;

#[derive(Clone, Default)]
struct LogBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for LogBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Missing image metadata must stay visible without hiding later removals or text stores.
#[test]
fn trtllm_digest_first_store_without_image_routing_is_dropped_and_logged() {
    let logs = LogBuffer::default();
    let writer = logs.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        let digest = "c".repeat(64);
        let mut token_ids: Vec<serde_json::Value> = (1..=31).map(|id| json!(id)).collect();
        token_ids.insert(27, json!(digest));
        let store = json!({
            "type": "BlockStored", "block_hashes": [100, 101], "parent_block_hash": null,
            "token_ids": token_ids, "block_size": 16,
            "mm_keys": [[], [{"hash": digest, "start_offset": 0}]],
        });
        let batch = json!([0.0, [store.clone(), store,
            {"type": "BlockRemoved", "block_hashes": [100]}
        ], 0]);
        let decoded = decode_event_batch(&to_vec_named(&batch).unwrap()).unwrap();
        assert!(matches!(decoded.events[0], RawKvEvent::Ignored));
        assert!(matches!(decoded.events[1], RawKvEvent::Ignored));
        assert!(matches!(decoded.events[2], RawKvEvent::BlockRemoved { .. }));

        let text = json!({
            "type": "BlockStored", "block_hashes": [100], "token_ids": [1, 2], "block_size": 2,
        });
        let raw: RawKvEvent = from_slice(&to_vec_named(&text).unwrap()).unwrap();
        assert!(matches!(raw, RawKvEvent::BlockStored { .. }));
    });

    let output = String::from_utf8(logs.0.lock().unwrap().clone()).unwrap();
    let lines: Vec<_> = output.lines().collect();
    assert_eq!(lines.len(), 2, "{output}");
    assert!(lines[0].trim_start().starts_with("WARN "), "{output}");
    assert!(
        lines[0].contains("trtllm_image_routing metadata"),
        "{output}"
    );
    assert!(lines[0].contains("NVIDIA/TensorRT-LLM#21125"), "{output}");
    assert!(lines[0].contains("dropped=1 blocks=2"), "{output}");
    assert!(lines[1].starts_with("DEBUG "), "{output}");
    assert!(
        lines[1].contains("without trtllm_image_routing metadata"),
        "{output}"
    );
    assert!(lines[1].contains("dropped=2 blocks=2"), "{output}");
}
