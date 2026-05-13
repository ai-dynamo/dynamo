// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine factory — creates the appropriate scheduler based on [`EngineType`].

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    EngineType, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::scheduler::{Scheduler, SchedulerHandle, SglangScheduler};

/// Create a scheduler for the configured engine type.
///
/// Returns a boxed [`SchedulerHandle`] that the engine wrapper can use
/// without knowing which backend is running underneath.
pub fn create_engine(
    args: MockEngineArgs,
    dp_rank: u32,
    output_tx: Option<mpsc::UnboundedSender<Vec<OutputSignal>>>,
    kv_event_publishers: KvEventPublishers,
    cancellation_token: Option<CancellationToken>,
    fpm_publisher: FpmPublisher,
) -> Box<dyn SchedulerHandle> {
    match args.engine_type {
        EngineType::Vllm => Box::new(Scheduler::new(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            fpm_publisher,
        )),
        EngineType::Sglang => Box::new(SglangScheduler::new(
            args,
            dp_rank,
            output_tx,
            kv_event_publishers,
            cancellation_token,
            fpm_publisher,
        )),
    }
}
