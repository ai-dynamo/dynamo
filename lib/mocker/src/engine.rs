// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine factory — creates the appropriate scheduler based on [`EngineType`].

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{EngineType, KvCacheEventSink, MockEngineArgs, OutputSignal};
use crate::scheduler::{Scheduler, SchedulerHandle, SglangScheduler};

/// Create a scheduler for the configured engine type.
///
/// Returns a boxed [`SchedulerHandle`] that the engine wrapper can use
/// without knowing which backend is running underneath.
pub fn create_engine(
    args: MockEngineArgs,
    dp_rank: u32,
    output_tx: Option<mpsc::UnboundedSender<OutputSignal>>,
    kv_event_sink: Option<Arc<dyn KvCacheEventSink>>,
    cancellation_token: Option<CancellationToken>,
) -> Box<dyn SchedulerHandle> {
    match args.engine_type {
        EngineType::Vllm => Box::new(Scheduler::new(
            args,
            dp_rank,
            output_tx,
            kv_event_sink,
            cancellation_token,
        )),
        EngineType::Sglang => Box::new(SglangScheduler::new(
            args,
            dp_rank,
            output_tx,
            kv_event_sink,
            cancellation_token,
        )),
    }
}
