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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;

use dynamo_llm::engines::MultiNodeConfig;
use dynamo_llm::kv_router::publisher::KvMetricsPublisher;
use dynamo_llm::protocols::common::llm_backend::{BackendInput, LLMEngineOutput};
use dynamo_runtime::engine::{AsyncEngine, AsyncEngineContextProvider, ResponseStream};
use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::runtime::CancellationToken;

use crate::worker;

pub struct VllmEngine {
    cancel_token: CancellationToken,
    worker: worker::VllmWorker,
}

impl VllmEngine {
    pub async fn new(
        cancel_token: CancellationToken,
        sock_code: &str,
        model_path: &Path,
        node_conf: MultiNodeConfig,
        tensor_parallel_size: u32,
        extra_engine_args: Option<PathBuf>,
        kv_metrics_publisher: Option<Arc<KvMetricsPublisher>>,
    ) -> anyhow::Result<Self> {
        let w = worker::start(
            cancel_token.clone(),
            sock_code,
            model_path,
            node_conf,
            tensor_parallel_size,
            extra_engine_args,
            kv_metrics_publisher,
        )
        .await?;
        let engine = VllmEngine {
            cancel_token,
            worker: w,
        };

        Ok(engine)
    }

    pub fn take_vllm_worker_handle(&mut self) -> tokio::task::JoinHandle<()> {
        self.worker.take_vllm_handle()
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<BackendInput>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for VllmEngine
{
    async fn generate(
        &self,
        request: SingleIn<BackendInput>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = request.into_parts();
        let ctx = context.context();
        let request_id = ctx.id().to_string();

        let (resp_tx, mut resp_rx) = tokio::sync::mpsc::channel(128);
        let work_req = worker::WorkRequest {
            request_id: context.id().to_string(),
            request,
            response_channel: resp_tx,
        };
        self.worker.enqueue_request(work_req).await?;

        let cancel_token = self.cancel_token.clone();
        let output = stream! {
            loop {
                let maybe_resp = tokio::select!{
                    _ = cancel_token.cancelled() => {
                        break;
                    }
                    maybe_resp = resp_rx.recv() => {
                        maybe_resp
                    }
                };
                match maybe_resp {
                    Some(out) => {
                        yield out;
                    },
                    None => {
                        tracing::trace!(request_id, "generate: response channel closed");
                        break;
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}
