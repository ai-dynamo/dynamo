// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public KVBM leader control plane (service-impl half).
//!
//! This is the engine-side counterpart to `kvbm_protocols::control`: a velo
//! service built from togglable [`ControlModule`]s. It is distinct from
//! [`crate::leader::velo::VeloLeaderService`], which carries engine-internal
//! leader↔leader RPC and is *not* public surface.
//!
//! Wiring: an [`InstanceLeader`](crate::leader::InstanceLeader) builds a
//! [`ControlPlane`] via [`ControlPlane::builder`], attaches the modules it
//! wants (`core` always, `transfer` always, `dev`/`metrics` opt-in), and calls
//! [`ControlPlaneBuilder::register`].

pub mod core;
pub mod dev;
pub mod module;
pub mod modules;

use std::sync::Arc;

use anyhow::{Context, Result};
use velo::{Handler, InstanceId, Messenger};

use kvbm_protocols::control::{
    ControlReply, LIST_MODULES_HANDLER, ListModulesRequest, ListModulesResponse, ModuleId,
};

pub use core::CoreModule;
pub use dev::DevModule;
pub use module::ControlModule;
pub use modules::metrics::MetricsModule;
pub use modules::transfer::TransferModule;

/// A registered leader control plane.
///
/// Holds the set of enabled module ids for introspection; the handlers
/// themselves are owned by the velo messenger after registration.
pub struct ControlPlane {
    instance_id: InstanceId,
    enabled: Vec<ModuleId>,
}

impl ControlPlane {
    /// Start building a control plane for `instance_id` over `messenger`.
    pub fn builder(messenger: Arc<Messenger>, instance_id: InstanceId) -> ControlPlaneBuilder {
        ControlPlaneBuilder {
            messenger,
            instance_id,
            modules: Vec::new(),
        }
    }

    /// The instance this control plane serves.
    pub fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    /// The modules enabled on this instance (as reported by `list_modules`).
    pub fn enabled_modules(&self) -> &[ModuleId] {
        &self.enabled
    }
}

/// Builder for [`ControlPlane`]: attach modules, then [`register`](Self::register).
pub struct ControlPlaneBuilder {
    messenger: Arc<Messenger>,
    instance_id: InstanceId,
    modules: Vec<Box<dyn ControlModule>>,
}

impl ControlPlaneBuilder {
    /// Attach (enable) a control module.
    pub fn with_module<M: ControlModule + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Register the always-on `list_modules` handler plus every attached
    /// module's handlers against the messenger.
    pub fn register(self) -> Result<Arc<ControlPlane>> {
        let enabled: Vec<ModuleId> = self.modules.iter().map(|m| m.id()).collect();

        register_list_modules(&self.messenger, enabled.clone())
            .context("registering control-plane list_modules handler")?;

        for module in &self.modules {
            let id = module.id();
            module
                .register(&self.messenger)
                .with_context(|| format!("registering control module {id:?}"))?;
            tracing::info!(module = ?id, "registered control module");
        }

        Ok(Arc::new(ControlPlane {
            instance_id: self.instance_id,
            enabled,
        }))
    }
}

/// Register the `list_modules` introspection handler — the "which plugins are
/// enabled" query. Naturally per-instance: the velo call addresses one leader.
fn register_list_modules(messenger: &Arc<Messenger>, enabled: Vec<ModuleId>) -> Result<()> {
    let handler = Handler::typed_unary_async(LIST_MODULES_HANDLER, move |ctx| {
        let enabled = enabled.clone();
        async move {
            let _req: ListModulesRequest = ctx.input;
            let reply: ControlReply<ListModulesResponse> =
                ControlReply::Ok(ListModulesResponse { modules: enabled });
            Ok::<ControlReply<ListModulesResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({LIST_MODULES_HANDLER}): {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::create_messenger_pair_tcp;
    use kvbm_protocols::control::client::LeaderControlClient;

    /// A trivial module that registers no handlers — exercises the
    /// builder/registration path and `list_modules` reporting.
    struct NoopModule(ModuleId);
    impl ControlModule for NoopModule {
        fn id(&self) -> ModuleId {
            self.0
        }
        fn register(&self, _messenger: &Arc<Messenger>) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn list_modules_reports_enabled_set() {
        let pair = create_messenger_pair_tcp().await.expect("messenger pair");
        let server = pair.messenger_b;
        let client_messenger = pair.messenger_a;

        let _plane = ControlPlane::builder(server.clone(), server.instance_id())
            .with_module(NoopModule(ModuleId::Core))
            .with_module(NoopModule(ModuleId::Transfer))
            .register()
            .expect("register control plane");

        let client = LeaderControlClient::new(client_messenger, server.instance_id());
        let modules = client.list_modules().await.expect("list_modules");

        assert_eq!(modules, vec![ModuleId::Core, ModuleId::Transfer]);
    }
}
