// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub struct ProgressEngine<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    state: Arc<Mutex<State<S, L, M>>>,
    cancel_token: CancellationToken,
    priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, L, M>>,
    ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, L, M>>,
    return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
    metrics: Arc<PoolMetrics>,
}

impl<S: Storage, L: LocalityProvider + 'static, M: BlockMetadata> ProgressEngine<S, L, M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: Arc<Mutex<State<S, L, M>>>,
        cancel_token: CancellationToken,
        return_rx: tokio::sync::mpsc::UnboundedReceiver<Block<S, L, M>>,
        priority_rx: tokio::sync::mpsc::UnboundedReceiver<PriorityRequest<S, L, M>>,
        ctrl_rx: tokio::sync::mpsc::UnboundedReceiver<ControlRequest<S, L, M>>,
    ) -> Self {
        let metrics = state.lock().unwrap().metrics.clone();
        Self {
            state,
            cancel_token,
            return_rx,
            priority_rx,
            ctrl_rx,
            metrics,
        }
    }

    /// Steps the progress engine forward one event
    /// The facade also holds the state, which means there can be contention for the mutex
    /// The facade only has high priority operations, thus blocking the background operations
    /// is acceptable.
    pub async fn step(&mut self) -> bool {
        tokio::select! {
            biased;

            Some(block) = self.return_rx.recv() => {
                self.metrics.gauge("return_block_queue_size").set(self.return_rx.len() as i64);
                self.state.lock().unwrap().return_block(block)
            }

            Some(priority_req) = self.priority_rx.recv(), if !self.priority_rx.is_closed() => {
                self.metrics.gauge("priority_request_queue_size").set(self.priority_rx.len() as i64);
                self.handle_priority_request(priority_req).await;
            }

            Some(req) = self.ctrl_rx.recv(), if !self.ctrl_rx.is_closed() => {
                self.metrics.gauge("control_request_queue_size").set(self.ctrl_rx.len() as i64);
                self.handle_control_request(req);
            }

            _ = self.cancel_token.cancelled() => {
                return false;
            }
        }

        true
    }

    #[allow(clippy::await_holding_lock)]
    pub async fn handle_priority_request(&mut self, req: PriorityRequest<S, L, M>) {
        let mut guard = self.state.lock().unwrap();
        match req {
            PriorityRequest::AllocateBlocks(req) => {
                let (count, resp_tx) = req.dissolve();
                let blocks = guard.allocate_blocks(count);
                if resp_tx.send(blocks).is_err() {
                    tracing::error!("failed to send response to allocate blocks");
                }
            }
            PriorityRequest::RegisterBlocks(req) => {
                let (blocks, resp_tx) = req.dissolve();
                let immutable_blocks = guard.register_blocks(blocks, &mut self.return_rx).await;
                if resp_tx.send(immutable_blocks).is_err() {
                    tracing::error!("failed to send response to register blocks");
                }
            }
            PriorityRequest::MatchSequenceHashes(req) => {
                let (sequence_hashes, resp_tx) = req.dissolve();
                let immutable_blocks = guard
                    .match_sequence_hashes(sequence_hashes, &mut self.return_rx)
                    .await;
                if resp_tx.send(Ok(immutable_blocks)).is_err() {
                    tracing::error!("failed to send response to match sequence hashes");
                }
            }
            PriorityRequest::TouchBlocks(req) => {
                let (sequence_hashes, resp_tx) = req.dissolve();
                guard
                    .touch_blocks(&sequence_hashes, &mut self.return_rx)
                    .await;
                if resp_tx.send(Ok(())).is_err() {
                    tracing::error!("failed to send response to touch blocks");
                }
            }
            PriorityRequest::Reset(req) => {
                let (_req, resp_tx) = req.dissolve();
                let result = guard.inactive.reset();
                if resp_tx.send(result).is_err() {
                    tracing::error!("failed to send response to reset");
                }
            }
            PriorityRequest::ReturnBlock(req) => {
                let (returnable_blocks, resp_tx) = req.dissolve();
                for block in returnable_blocks {
                    guard.return_block(block);
                }
                if resp_tx.send(Ok(())).is_err() {
                    tracing::error!("failed to send response to return block");
                }
            }
        }
    }

    pub fn handle_control_request(&mut self, req: ControlRequest<S, L, M>) {
        let mut guard = self.state.lock().unwrap();
        match req {
            ControlRequest::AddBlocks(blocks) => {
                let (blocks, resp_rx) = blocks.dissolve();
                let result = guard.inactive.add_blocks(blocks);
                if resp_rx.send(result).is_err() {
                    tracing::error!("failed to send response to add blocks");
                }
            }
            ControlRequest::Status(req) => {
                let (_, resp_rx) = req.dissolve();
                if resp_rx.send(Ok(guard.status())).is_err() {
                    tracing::error!("failed to send response to status");
                }
            }
            ControlRequest::ResetBlocks(req) => {
                let (sequence_hashes, resp_rx) = req.dissolve();
                if resp_rx
                    .send(Ok(guard.try_reset_blocks(&sequence_hashes)))
                    .is_err()
                {
                    tracing::error!("failed to send response to reset blocks");
                }
            }
        }
    }


}
