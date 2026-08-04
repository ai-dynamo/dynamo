// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine factory — creates the appropriate scheduler based on [`EngineType`].

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{
    EngineType, FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal,
};
use crate::scheduler::{
    Scheduler, SchedulerEventSender, SchedulerHandle, SchedulerOutputSender, SglangScheduler,
};

pub(crate) struct LiveEngineScheduler {
    pub(crate) handle: Box<dyn SchedulerHandle>,
    pub(crate) actor: JoinHandle<anyhow::Result<()>>,
}

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
    create_engine_with_output_sender(
        args,
        dp_rank,
        output_tx.map(SchedulerOutputSender::from),
        kv_event_publishers,
        cancellation_token,
        fpm_publisher,
    )
}

pub(crate) fn create_engine_with_output_sender(
    args: MockEngineArgs,
    dp_rank: u32,
    output_tx: Option<SchedulerOutputSender>,
    kv_event_publishers: KvEventPublishers,
    cancellation_token: Option<CancellationToken>,
    fpm_publisher: FpmPublisher,
) -> Box<dyn SchedulerHandle> {
    let LiveEngineScheduler { handle, actor } = create_engine_with_event_sender(
        args,
        dp_rank,
        output_tx.map(SchedulerEventSender::from),
        kv_event_publishers,
        cancellation_token,
        fpm_publisher,
    );
    drop(actor);
    handle
}

pub(crate) fn create_engine_with_event_sender(
    args: MockEngineArgs,
    dp_rank: u32,
    event_tx: Option<SchedulerEventSender>,
    kv_event_publishers: KvEventPublishers,
    cancellation_token: Option<CancellationToken>,
    fpm_publisher: FpmPublisher,
) -> LiveEngineScheduler {
    let (handle, actor): (Box<dyn SchedulerHandle>, _) = match args.engine_type {
        // TRT-LLM reuses the vLLM scheduler core; the GUARANTEED_NO_EVICT
        // policy is carried in `args` and read by the core per pass.
        EngineType::Vllm | EngineType::Trtllm => {
            let (scheduler, actor) = Scheduler::spawn_with_event_sender(
                args,
                dp_rank,
                event_tx,
                kv_event_publishers,
                cancellation_token,
                fpm_publisher,
            );
            (Box::new(scheduler), actor)
        }
        EngineType::Sglang => {
            let (scheduler, actor) = SglangScheduler::spawn_with_event_sender(
                args,
                dp_rank,
                event_tx,
                kv_event_publishers,
                cancellation_token,
                fpm_publisher,
            );
            (Box::new(scheduler), actor)
        }
    };
    LiveEngineScheduler { handle, actor }
}
