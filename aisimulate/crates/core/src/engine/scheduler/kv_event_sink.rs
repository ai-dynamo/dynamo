// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::engine::KvEvent;
use crate::engine::common::protocols::KvCacheEventSink;

/// Captures engine-owned, router-neutral KV events.
#[derive(Clone, Default)]
pub(crate) struct CapturedKvEventBuffer {
    events: Arc<Mutex<Vec<KvEvent>>>,
}

impl CapturedKvEventBuffer {
    pub(crate) fn drain(&self) -> Vec<KvEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

#[derive(Clone)]
struct KvEventCaptureSink {
    buffer: CapturedKvEventBuffer,
}

impl KvCacheEventSink for KvEventCaptureSink {
    fn publish(&self, event: KvEvent) -> Result<()> {
        self.buffer.events.lock().unwrap().push(event);
        Ok(())
    }
}

pub(crate) fn capture_kv_event_sink() -> (CapturedKvEventBuffer, Arc<dyn KvCacheEventSink>) {
    let buffer = CapturedKvEventBuffer::default();
    let sink = Arc::new(KvEventCaptureSink {
        buffer: buffer.clone(),
    });
    (buffer, sink)
}
