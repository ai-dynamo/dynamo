// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `transfer` control module: G2 search → disagg-session creation.
//!
//! Both handlers search the leader's G2 block manager for a set of sequence
//! hashes. If nothing matches, no session is created and the caller gets
//! `NoBlocksFound`. If blocks are found, a disagg session is opened, populated
//! with the matched blocks (committed + made available), parked in the
//! [`SessionManager`], and its id returned. The caller resolves the session's
//! `SessionEndpoint` out-of-band (e.g. the hub peer registry) to attach.

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use velo::{Handler, Messenger};

use kvbm_logical::blocks::ImmutableBlock;
use kvbm_logical::manager::BlockManager;
use kvbm_protocols::control::modules::transfer::{
    SEARCH_PREFIX_HANDLER, SEARCH_SCATTER_HANDLER, SearchRequest, SearchResponse,
};
use kvbm_protocols::control::{ControlError, ControlReply, ModuleId};

use crate::disagg::session::{SessionFactory, SessionManager};
use crate::leader::control::ControlModule;
use crate::{G2, SequenceHash};

/// Handle to the lazily-populated disagg `SessionFactory` cell.
type SessionFactoryCell = Arc<OnceLock<Arc<dyn SessionFactory>>>;

/// The `transfer` control module — always enabled.
pub struct TransferModule {
    g2_manager: Arc<BlockManager<G2>>,
    session_factory: SessionFactoryCell,
    session_manager: Arc<SessionManager>,
}

impl TransferModule {
    pub fn new(
        g2_manager: Arc<BlockManager<G2>>,
        session_factory: SessionFactoryCell,
        session_manager: Arc<SessionManager>,
    ) -> Self {
        Self {
            g2_manager,
            session_factory,
            session_manager,
        }
    }
}

impl ControlModule for TransferModule {
    fn id(&self) -> ModuleId {
        ModuleId::Transfer
    }

    fn register(&self, messenger: &Arc<Messenger>) -> Result<()> {
        register_search(
            messenger,
            SEARCH_PREFIX_HANDLER,
            SearchMode::Prefix,
            self.g2_manager.clone(),
            self.session_factory.clone(),
            self.session_manager.clone(),
        )?;
        register_search(
            messenger,
            SEARCH_SCATTER_HANDLER,
            SearchMode::Scatter,
            self.g2_manager.clone(),
            self.session_factory.clone(),
            self.session_manager.clone(),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum SearchMode {
    /// Contiguous-prefix match — stops at the first missing hash.
    Prefix,
    /// Scatter (gather-all) — every hash present, regardless of gaps.
    Scatter,
}

fn register_search(
    messenger: &Arc<Messenger>,
    handler_name: &'static str,
    mode: SearchMode,
    g2: Arc<BlockManager<G2>>,
    session_factory: SessionFactoryCell,
    session_manager: Arc<SessionManager>,
) -> Result<()> {
    let handler = Handler::typed_unary_async(handler_name, move |ctx| {
        let g2 = g2.clone();
        let session_factory = session_factory.clone();
        let session_manager = session_manager.clone();
        async move {
            let req: SearchRequest = ctx.input;
            let reply: ControlReply<SearchResponse> =
                search(mode, &g2, &session_factory, &session_manager, req).into();
            Ok::<ControlReply<SearchResponse>, anyhow::Error>(reply)
        }
    })
    .build();
    messenger
        .register_handler(handler)
        .map_err(|e| anyhow::anyhow!("velo register_handler({handler_name}): {e}"))?;
    Ok(())
}

/// Search G2, and on a non-empty match open + populate a disagg session.
fn search(
    mode: SearchMode,
    g2: &BlockManager<G2>,
    session_factory: &SessionFactoryCell,
    session_manager: &Arc<SessionManager>,
    req: SearchRequest,
) -> Result<SearchResponse, ControlError> {
    let blocks: Vec<ImmutableBlock<G2>> = match mode {
        SearchMode::Prefix => g2.match_blocks(&req.sequence_hashes),
        // `touch: false` — an RPC search should not perturb G2 LRU.
        SearchMode::Scatter => g2
            .scan_matches(&req.sequence_hashes, false)
            .into_values()
            .collect(),
    };

    if blocks.is_empty() {
        return Ok(SearchResponse::NoBlocksFound);
    }

    let factory = session_factory
        .get()
        .ok_or(ControlError::NotInitialized)?
        .clone();

    let session_id = open_and_populate(&factory, session_manager, blocks)?;
    Ok(SearchResponse::Session { session_id })
}

/// Open a holder-side disagg session, commit + publish the matched blocks,
/// park it in the [`SessionManager`], and return its id.
fn open_and_populate(
    factory: &Arc<dyn SessionFactory>,
    session_manager: &Arc<SessionManager>,
    blocks: Vec<ImmutableBlock<G2>>,
) -> Result<uuid::Uuid, ControlError> {
    let session_id = uuid::Uuid::new_v4();
    let hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.sequence_hash()).collect();

    let session = factory
        .open(session_id)
        .map_err(|e| ControlError::Internal(format!("open session: {e:#}")))?;
    session
        .commit(hashes)
        .map_err(|e| ControlError::Internal(format!("commit: {e:#}")))?;
    session
        .make_available(blocks)
        .map_err(|e| ControlError::Internal(format!("make_available: {e:#}")))?;
    session
        .finish_commits()
        .map_err(|e| ControlError::Internal(format!("finish_commits: {e:#}")))?;
    session
        .finish_availability()
        .map_err(|e| ControlError::Internal(format!("finish_availability: {e:#}")))?;

    session_manager.register(session);
    Ok(session_id)
}
