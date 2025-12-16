// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Console hook definitions for connector lifecycle instrumentation.

use std::sync::Arc;

use uuid::Uuid;

use crate::v2::integrations::connector::slot::{OperationInfo, SlotState};

/// Hook interface used to surface connector lifecycle events to an optional console.
///
/// Implementations can record, log, or pause execution around critical events. The default
/// `NoopConsoleHook` provides a zero-cost stub for production builds.
pub trait ConnectorConsoleHook: Send + Sync {
    fn on_slot_created(&self, _request_id: &str, _state: SlotState) {}

    fn on_state_transition(&self, _request_id: &str, _prev: SlotState, _next: SlotState) {}

    fn on_operation_registered(
        &self,
        _request_id: &str,
        _operation_id: Uuid,
        _info: &OperationInfo,
    ) {
    }

    fn on_operation_completed(&self, _request_id: &str, _operation_id: Uuid) {}

    fn on_request_finish_started(&self, _request_id: &str, _outstanding: usize) {}

    fn on_slot_finished(&self, _request_id: &str) {}
}

/// Type alias for shared hook handles.
pub type ConsoleHookRef = Arc<dyn ConnectorConsoleHook>;

#[derive(Debug, Default)]
pub struct NoopConsoleHook;

impl ConnectorConsoleHook for NoopConsoleHook {}

impl NoopConsoleHook {
    pub fn shared() -> ConsoleHookRef {
        Arc::new(Self::default())
    }
}
