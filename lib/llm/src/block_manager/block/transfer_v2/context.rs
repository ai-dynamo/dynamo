// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use cudarc::driver::{sys::CUevent_flags, CudaEvent, CudaStream};
use nixl_sys::Agent as NixlAgent;

use std::sync::Arc;
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot},
    task,
};
use tokio_util::sync::CancellationToken;

/// Transfer context containing resources and configuration for transfers
pub struct TransferContext {
    pub async_rt_handle: Handle,
    pub cancel_token: CancellationToken,
    pub cuda_ctx: Option<Arc<CudaTransferContext>>,
    pub nixl_ctx: Option<Arc<NixlTransferContext>>,
}

/// Context for CUDA transfers, holding the stream and event worker details.
pub struct CudaTransferContext {
    pub stream: Arc<CudaStream>,
    pub cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>)>,
    pub cuda_event_worker: Option<task::JoinHandle<()>>,
}

/// Context for NIXL transfers, encapsulating the NIXL agent.
#[derive(Clone)]
pub struct NixlTransferContext {
    agent: Arc<NixlAgent>,
}

impl TransferContext {
    /// Creates a new basic `TransferContext` suitable for simple memory transfers.
    pub fn new(async_rt_handle: Handle) -> Self {
        Self {
            async_rt_handle,
            cancel_token: CancellationToken::new(),
            cuda_ctx: None,
            nixl_ctx: None,
        }
    }

    /// Attaches a CUDA context to the `TransferContext`.
    pub fn with_cuda(mut self, cuda_ctx: CudaTransferContext) -> Self {
        self.cuda_ctx = Some(Arc::new(cuda_ctx));
        self
    }

    /// Attaches a NIXL context to the `TransferContext`.
    pub fn with_nixl(mut self, nixl_ctx: NixlTransferContext) -> Self {
        self.nixl_ctx = Some(Arc::new(nixl_ctx));
        self
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        // CudaTransferContext now manages its own worker thread.
    }
}

impl CudaTransferContext {
    pub fn new(stream: Arc<CudaStream>) -> Self {
        let (cuda_event_tx, mut cuda_event_rx): (
            mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>)>,
            mpsc::UnboundedReceiver<(CudaEvent, oneshot::Sender<()>)>,
        ) = mpsc::unbounded_channel();
        let cancel_token = CancellationToken::new();
        let inner_cancel_token = cancel_token.clone();

        let cuda_event_worker = task::spawn(async move {
            loop {
                tokio::select! {
                    Some((event, tx)) = cuda_event_rx.recv() => {
                        if let Err(e) = event.synchronize() {
                            tracing::error!("Failed to synchronize CUDA event: {:?}", e);
                        }
                        let _ = tx.send(());
                    }
                    _ = inner_cancel_token.cancelled() => {
                        break;
                    }
                }
            }
        });

        Self {
            stream,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
        }
    }

    pub fn record_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        let event = self
            .stream
            .record_event(None)
            .map_err(|e| TransferError::ExecutionError(e.to_string()))?;

        self.cuda_event_tx
            .send((event, tx))
            .map_err(|_| TransferError::ExecutionError("CUDA event worker exited.".into()))?;
        Ok(())
    }
}

impl Drop for CudaTransferContext {
    fn drop(&mut self) {
        if let Some(handle) = self.cuda_event_worker.take() {
            handle.abort();
        }
    }
}

impl NixlTransferContext {
    pub fn new(agent: Arc<NixlAgent>) -> Self {
        Self { agent }
    }

    pub fn agent(&self) -> &NixlAgent {
        &self.agent
    }
}
