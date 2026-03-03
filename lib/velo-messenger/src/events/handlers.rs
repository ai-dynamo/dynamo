// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System handler registration for distributed event messages.

use std::sync::Arc;

use crate::handlers::{Handler, HandlerManager};

use super::VeloEvents;

/// Register the three event system handlers with the handler manager.
pub(crate) fn register_event_handlers(
    handlers: &HandlerManager,
    events: Arc<VeloEvents>,
) -> anyhow::Result<()> {
    // _event_subscribe: Remote node subscribes to a local event
    let events_clone = events.clone();
    handlers.register_internal_handler(
        Handler::am_handler("_event_subscribe", move |ctx| {
            if let Err(e) = events_clone.handle_subscribe(ctx.payload) {
                tracing::warn!("_event_subscribe handler error: {}", e);
            }
            Ok(())
        })
        .spawn()
        .build(),
    )?;

    // _event_trigger: Completion notification from remote owner to subscriber
    let events_clone = events.clone();
    handlers.register_internal_handler(
        Handler::am_handler("_event_trigger", move |ctx| {
            if let Err(e) = events_clone.handle_trigger(ctx.payload) {
                tracing::warn!("_event_trigger handler error: {}", e);
            }
            Ok(())
        })
        .spawn()
        .build(),
    )?;

    // _event_trigger_request: Remote trigger/poison request + ACK/NACK
    let events_clone = events.clone();
    handlers.register_internal_handler(
        Handler::am_handler("_event_trigger_request", move |ctx| {
            if let Err(e) = events_clone.handle_trigger_request(ctx.payload) {
                tracing::warn!("_event_trigger_request handler error: {}", e);
            }
            Ok(())
        })
        .spawn()
        .build(),
    )?;

    Ok(())
}
