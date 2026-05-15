// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `test` control module: test-only G2 block helpers.
//!
//! Opt-in via `control.test = true`. Usable in production but the engine
//! logs a warning when it is enabled — the handlers here are not meant for
//! production use. A leader without it enabled simply will not list
//! [`ModuleId::Test`] in its `list_modules` response.

use std::sync::Arc;

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_logical::manager::BlockManager;
use kvbm_protocols::control::modules::test::{
    REGISTER_TEST_BLOCKS_HANDLER, RegisterTestBlocksRequest, RegisterTestBlocksResponse,
};
use kvbm_protocols::control::{ControlError, ControlReply, ModuleId};

use crate::G2;
use crate::leader::control::ControlModule;

/// The `test` control module.
pub struct TestModule {
    g2_manager: Arc<BlockManager<G2>>,
}

impl TestModule {
    pub fn new(g2_manager: Arc<BlockManager<G2>>) -> Self {
        Self { g2_manager }
    }
}

impl ControlModule for TestModule {
    fn id(&self) -> ModuleId {
        ModuleId::Test
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        let g2 = self.g2_manager.clone();
        let handler = Handler::typed_unary_async(REGISTER_TEST_BLOCKS_HANDLER, move |ctx| {
            let g2 = g2.clone();
            async move {
                let req: RegisterTestBlocksRequest = ctx.input;
                let reply: ControlReply<RegisterTestBlocksResponse> =
                    register_test_blocks(&g2, req).into();
                Ok::<ControlReply<RegisterTestBlocksResponse>, anyhow::Error>(reply)
            }
        })
        .build();
        messenger.register_handler(handler).map_err(|e| {
            anyhow::anyhow!("velo register_handler({REGISTER_TEST_BLOCKS_HANDLER}): {e}")
        })?;
        Ok(())
    }
}

/// Allocate one G2 block per requested hash, stamp the hash onto it, register
/// it, then release it.
///
/// G2 allocation is all-or-nothing: if the pool cannot satisfy the full
/// request, `allocated` is `0`. The registered blocks are dropped at the end
/// of this function — that returns them to the inactive pool but leaves them
/// in the registry, so a subsequent `transfer` search can still find them.
fn register_test_blocks(
    g2: &BlockManager<G2>,
    req: RegisterTestBlocksRequest,
) -> Result<RegisterTestBlocksResponse, ControlError> {
    let n = req.sequence_hashes.len();

    let Some(mutables) = g2.allocate_blocks(n) else {
        tracing::warn!(
            requested = n,
            "register_test_blocks: G2 pool cannot satisfy request"
        );
        return Ok(RegisterTestBlocksResponse { allocated: 0 });
    };

    let block_size = g2.block_size();
    let mut completes = Vec::with_capacity(n);
    for (mutable, hash) in mutables.into_iter().zip(req.sequence_hashes) {
        let complete = mutable
            .stage(hash, block_size)
            .map_err(|e| ControlError::Internal(format!("stage test block: {e}")))?;
        completes.push(complete);
    }

    // Register then release: the `ImmutableBlock`s drop here, returning the
    // blocks to the inactive pool while keeping their registry entries.
    let _immutables = g2.register_blocks(completes);

    tracing::info!(allocated = n, "register_test_blocks complete");
    Ok(RegisterTestBlocksResponse { allocated: n })
}
