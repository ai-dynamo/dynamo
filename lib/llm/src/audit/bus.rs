// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::handle::AuditRecord;
use std::sync::{Arc, OnceLock};
use tokio::sync::broadcast;

static BUS: OnceLock<broadcast::Sender<Arc<AuditRecord>>> = OnceLock::new();

pub fn init(capacity: usize) {
    let (tx, _rx) = broadcast::channel::<Arc<AuditRecord>>(capacity);
    let _ = BUS.set(tx);
}

pub fn subscribe() -> broadcast::Receiver<Arc<AuditRecord>> {
    BUS.get().expect("audit bus not initialized").subscribe()
}

pub fn publish(rec: AuditRecord) {
    if let Some(tx) = BUS.get() {
        let _ = tx.send(Arc::new(rec));
    }
}
