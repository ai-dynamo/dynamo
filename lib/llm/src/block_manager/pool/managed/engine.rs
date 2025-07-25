// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod progress;

pub use client::Client;
pub use progress::ProgressEngine;

use super::*;

// Specific request type aliases for our use cases
type AllocateBlocksReq<S, L, M> = RequestResponse<usize, BlockPoolResult<MutableBlocks<S, L, M>>>;
type RegisterBlocksReq<S, L, M> =
    RequestResponse<MutableBlocks<S, L, M>, BlockPoolResult<ImmutableBlocks<S, L, M>>>;
type MatchHashesReq<S, L, M> =
    RequestResponse<Vec<SequenceHash>, BlockPoolResult<ImmutableBlocks<S, L, M>>>;
type TouchBlocksReq = RequestResponse<Vec<SequenceHash>, BlockPoolResult<()>>;
type AddBlocksReq<S, L, M> = RequestResponse<Vec<Block<S, L, M>>, BlockPoolResult<()>>;
type ResetReq = RequestResponse<(), BlockPoolResult<()>>;
type ReturnBlockReq<S, L, M> = RequestResponse<Vec<Block<S, L, M>>, BlockPoolResult<()>>;
type StatusReq = RequestResponse<(), BlockPoolResult<BlockPoolStatus>>;
type ResetBlocksReq = RequestResponse<Vec<SequenceHash>, BlockPoolResult<ResetBlocksResponse>>;

// Update the request enums to use the cleaner types
pub enum PriorityRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    AllocateBlocks(AllocateBlocksReq<S, L, M>),
    RegisterBlocks(RegisterBlocksReq<S, L, M>),
    MatchSequenceHashes(MatchHashesReq<S, L, M>),
    TouchBlocks(TouchBlocksReq),
    Reset(ResetReq),
    ReturnBlock(ReturnBlockReq<S, L, M>),
}

pub enum ControlRequest<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    AddBlocks(AddBlocksReq<S, L, M>),
    Status(StatusReq),
    ResetBlocks(ResetBlocksReq),
}

pub fn create<S: Storage, L: LocalityProvider, M: BlockMetadata>(
    state: Arc<Mutex<State<S, L, M>>>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    cancel_token: CancellationToken,
) -> (Client<S, L, M>, ProgressEngine<S, L, M>) {
    let (priority_tx, priority_rx) = tokio::sync::mpsc::unbounded_channel();
    let (ctrl_tx, ctrl_rx) = tokio::sync::mpsc::unbounded_channel();

    let client = Client::new(priority_tx, ctrl_tx);

    let progress_engine = ProgressEngine::<S, L, M>::new(
        state.clone(),
        cancel_token,
        return_rx,
        priority_rx,
        ctrl_rx,
    );

    (client, progress_engine)
}
