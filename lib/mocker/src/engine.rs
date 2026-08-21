// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-rank compatibility entry points for the grouped generalized engine.

use anyhow::{Context, ensure};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::common::protocols::{FpmPublisher, KvEventPublishers, MockEngineArgs, OutputSignal};
use crate::grouped_scheduler::{
    GroupedSchedulerRankEventSinks, GroupedSchedulers,
    create_single_rank_scheduler_with_event_sender,
};
use crate::scheduler::{SchedulerEventSender, SchedulerHandle};

pub(crate) struct LiveEngineScheduler {
    pub(crate) handle: Box<dyn SchedulerHandle>,
    pub(crate) actor: JoinHandle<anyhow::Result<()>>,
    pub(crate) completion_drain: crate::grouped_scheduler::CompletionBoundaryDrain,
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
) -> anyhow::Result<Box<dyn SchedulerHandle>> {
    let LiveEngineScheduler {
        handle,
        actor,
        completion_drain: _,
    } = create_engine_with_event_sender(
        args,
        dp_rank,
        output_tx.map(SchedulerEventSender::from),
        kv_event_publishers,
        cancellation_token,
        fpm_publisher,
    )?;
    // Dropping a Tokio JoinHandle detaches the actor. The SchedulerHandle's
    // cancellation guard remains the compatibility API's shutdown owner.
    drop(actor);
    Ok(handle)
}

pub(crate) fn create_engine_with_event_sender(
    args: MockEngineArgs,
    dp_rank: u32,
    event_tx: Option<SchedulerEventSender>,
    kv_event_publishers: KvEventPublishers,
    cancellation_token: Option<CancellationToken>,
    fpm_publisher: FpmPublisher,
) -> anyhow::Result<LiveEngineScheduler> {
    // This compatibility API cannot safely construct attention-DP ranks one at
    // a time: each call would own a different generalized engine and bypass the
    // group barrier. Production Live Mocker constructs the complete group once
    // through `create_grouped_scheduler`.
    ensure!(
        args.dp_size == 1,
        "single-rank create_engine does not support attention DP; use create_grouped_scheduler"
    );
    ensure!(
        dp_rank == 0,
        "single-rank create_engine requires dp_rank=0; use create_grouped_scheduler"
    );
    let GroupedSchedulers {
        mut schedulers,
        actor,
        completion_drain,
    } = create_single_rank_scheduler_with_event_sender(
        args,
        dp_rank,
        GroupedSchedulerRankEventSinks {
            event_tx,
            kv_event_publishers,
            fpm_publisher,
        },
        cancellation_token,
    )?;
    let handle = schedulers
        .pop()
        .context("single-rank generalized Mocker engine returned no scheduler handle")?;
    ensure!(
        schedulers.is_empty(),
        "single-rank generalized Mocker engine returned extra scheduler handles"
    );
    Ok(LiveEngineScheduler {
        handle,
        actor,
        completion_drain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn compatibility_entrypoint_rejects_attention_dp() {
        let args = MockEngineArgs {
            dp_size: 4,
            ..MockEngineArgs::default()
        };
        let cancel = CancellationToken::new();

        let result = create_engine_with_event_sender(
            args,
            3,
            None,
            KvEventPublishers::default(),
            Some(cancel.clone()),
            FpmPublisher::default(),
        );
        let error = match result {
            Ok(_) => panic!("attention DP must be rejected by the single-rank entrypoint"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not support attention DP"));
    }
}
