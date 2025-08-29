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

use cudarc::driver::{CudaEvent, CudaStream, sys::CUevent_flags};
use nixl_sys::Agent as NixlAgent;

use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub struct TransferContext {
    nixl_agent: Arc<Option<NixlAgent>>,
    stream: Arc<CudaStream>,
    async_rt_handle: Handle,

    cuda_event_tx: mpsc::UnboundedSender<(CudaEvent, oneshot::Sender<()>)>,
    cuda_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferContext {
    pub fn new(
        nixl_agent: Arc<Option<NixlAgent>>,
        stream: Arc<CudaStream>,
        async_rt_handle: Handle,
    ) -> Self {
        let (cuda_event_tx, mut cuda_event_rx) =
            mpsc::unbounded_channel::<(CudaEvent, oneshot::Sender<()>)>();

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let cuda_event_worker = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime for CUDA event worker.");

            runtime.block_on(async move {
                loop {
                    tokio::select! {
                        Some((event, tx)) = cuda_event_rx.recv() => {
                            if let Err(e) = event.synchronize() {
                                tracing::error!("Error synchronizing CUDA event: {}", e);
                            }
                            let _ = tx.send(());
                        }
                        _ = cancel_token_clone.cancelled() => {
                            break;
                        }
                    }
                }
            });
        });

        Self {
            nixl_agent,
            stream,
            async_rt_handle,
            cuda_event_tx,
            cuda_event_worker: Some(cuda_event_worker),
            cancel_token,
        }
    }

    pub fn nixl_agent(&self) -> Arc<Option<NixlAgent>> {
        self.nixl_agent.clone()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn async_rt_handle(&self) -> &Handle {
        &self.async_rt_handle
    }

    pub fn cuda_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        let event = self
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(|e| TransferError::ExecutionError(e.to_string()))?;

        self.cuda_event_tx
            .send((event, tx))
            .map_err(|_| TransferError::ExecutionError("CUDA event worker exited.".into()))?;
        Ok(())
    }
}

impl Drop for TransferContext {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        if let Some(handle) = self.cuda_event_worker.take()
            && let Err(e) = handle.join()
        {
            tracing::error!("Error joining CUDA event worker: {:?}", e);
        }
    }
}
