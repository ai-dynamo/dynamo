// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_nova::{Nova, am::NovaHandler};
use std::sync::Arc;

use crate::distributed::worker::LeaderLayoutConfig;

use super::*;

pub fn init(nova: &Arc<Nova>, state: Arc<WorkerState>) {
    register_initialize_handler(nova, &state);
    register_completion_handlers(nova, &state);
    register_get_layout_config_handler(nova, &state);
}

/// Register the configure_layouts handler for leader-driven initialization.
///
/// This handler is called by the leader after collecting handshake metadata.
/// It completes NIXL registration and creates G1/G2/G3 layouts.
fn register_initialize_handler(nova: &Arc<Nova>, state: &Arc<WorkerState>) {
    let state = Arc::clone(state);
    let handler = NovaHandler::typed_unary_async(INITIALIZE_HANDLER, move |ctx| {
        let state = Arc::clone(&state);
        async move {
            let config: LeaderLayoutConfig = ctx.input;
            state
                .initialize(config)
                .inspect_err(|e| tracing::error!("Error initializing worker: {}", e))
        }
    })
    .build();

    if let Err(e) = nova.register_handler(handler) {
        tracing::error!("Failed to register configure_layouts handler: {}", e);
    }
}

/// Register completion handlers for leader notifications.
fn register_completion_handlers(nova: &Arc<Nova>, state: &Arc<WorkerState>) {
    let state = Arc::clone(state);
    let onboard_state = Arc::clone(&state);
    let onboard_handler = NovaHandler::typed_unary_async(ONBOARD_COMPLETE_HANDLER, move |ctx| {
        let state = Arc::clone(&onboard_state);
        async move {
            let msg: OnboardCompleteMessage = ctx.input;
            tracing::debug!(request_id = %msg.request_id, "Worker received onboard complete");
            state.mark_onboarding_complete(msg.request_id);
            Ok(())
        }
    })
    .build();

    if let Err(e) = nova.register_handler(onboard_handler) {
        tracing::error!("Failed to register onboard_complete handler: {}", e);
    }

    // Handler: "kvbm.connector.worker.offload_complete"
    let offload_state = Arc::clone(&state);
    let offload_handler = NovaHandler::typed_unary_async(OFFLOAD_COMPLETE_HANDLER, move |ctx| {
        let state = Arc::clone(&offload_state);
        async move {
            let msg: OffloadCompleteMessage = ctx.input;
            tracing::debug!(request_id = %msg.request_id, "Worker received offload complete");
            state.mark_offloading_complete(msg.request_id);
            Ok(())
        }
    })
    .build();

    if let Err(e) = nova.register_handler(offload_handler) {
        tracing::error!("Failed to register offload_complete handler: {}", e);
    }

    // Handler: "kvbm.connector.worker.failed_onboard"
    let failed_state = state;
    let failed_handler = NovaHandler::typed_unary_async(FAILED_ONBOARD_HANDLER, move |ctx| {
        let state = Arc::clone(&failed_state);
        async move {
            let msg: FailedOnboardMessage = ctx.input;
            tracing::debug!(
                request_id = %msg.request_id,
                block_ids = ?msg.block_ids,
                "Worker received failed onboard notification"
            );
            state.mark_failed_onboarding(msg.block_ids);
            Ok(())
        }
    })
    .build();

    if let Err(e) = nova.register_handler(failed_handler) {
        tracing::error!("Failed to register failed_onboard handler: {}", e);
    }
}

fn register_get_layout_config_handler(nova: &Arc<Nova>, state: &Arc<WorkerState>) {
    let state = Arc::clone(state);
    let handler = NovaHandler::unary_handler_async(GET_LAYOUT_CONFIG_HANDLER, move |_ctx| {
        let state = Arc::clone(&state);
        async move {
            Ok(Some(Bytes::from(serde_json::to_vec(
                &state.layout_config()?,
            )?)))
        }
    })
    .build();

    if let Err(e) = nova.register_handler(handler) {
        tracing::error!("Failed to register get_layout_config handler: {}", e);
    }
}
