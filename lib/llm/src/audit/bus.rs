// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::event::AuditEvent;
use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::broadcast;

static BUS: OnceLock<broadcast::Sender<Arc<AuditEvent>>> = OnceLock::new();

pub fn init(capacity: usize) -> broadcast::Receiver<Arc<AuditEvent>> {
    let (tx, rx) = broadcast::channel(capacity);
    let _ = BUS.set(tx);
    rx
}

pub fn publish(evt: AuditEvent) {
    if let Some(tx) = BUS.get() {
        let _ = tx.send(Arc::new(evt));
    }
}
