// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use tokio::sync::broadcast;

use super::types::AgentTraceRecord;

static BUS: OnceLock<broadcast::Sender<AgentTraceRecord>> = OnceLock::new();

pub fn init(capacity: usize) {
    let (tx, _rx) = broadcast::channel::<AgentTraceRecord>(capacity.max(1));
    if BUS.set(tx).is_err() {
        tracing::debug!(
            capacity,
            "agent trace bus already initialized; keeping existing sender"
        );
    }
}

/// Panics if the trace bus has not been initialized.
pub fn subscribe() -> broadcast::Receiver<AgentTraceRecord> {
    BUS.get()
        .expect("agent trace bus not initialized")
        .subscribe()
}

pub fn publish(rec: AgentTraceRecord) {
    if let Some(tx) = BUS.get() {
        let _ = tx.send(rec);
    }
}
