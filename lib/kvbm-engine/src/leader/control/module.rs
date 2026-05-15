// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`ControlModule`] extension contract.

use std::sync::Arc;

use anyhow::Result;
use velo::Messenger;

pub use kvbm_protocols::control::ModuleId;

/// A togglable control-plane module (plugin).
///
/// A module is the service-impl half of a `{protocol + client + service impl}`
/// bundle — the protocol/client halves live in
/// `kvbm_protocols::control::modules`. Attaching a module to a
/// [`ControlPlaneBuilder`](super::ControlPlaneBuilder) is what "enables" it;
/// the set of attached modules is reported by the `list_modules` query.
pub trait ControlModule: Send + Sync {
    /// Stable identity, shared with the protocol crate and reported by
    /// `list_modules`.
    fn id(&self) -> ModuleId;

    /// Register this module's velo handlers against the leader messenger.
    ///
    /// Handlers should capture their own `Arc` clones of whatever state they
    /// need; the boxed module itself may be dropped once this returns.
    fn register(&self, messenger: &Arc<Messenger>) -> Result<()>;
}
