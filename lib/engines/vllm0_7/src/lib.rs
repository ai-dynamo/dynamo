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

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use pyo3::prelude::*;

use dynamo_runtime::pipeline::error as pipeline_error;
use dynamo_runtime::CancellationToken;

use dynamo_llm::backend::ExecutionContext;
use dynamo_llm::engines::MultiNodeConfig;
use dynamo_llm::kv_router::publisher::KvMetricsPublisher;

mod engine;
use engine::VllmEngine;

mod ray;
use ray::Ray;

mod subprocess;
pub use subprocess::run_subprocess;

mod worker;

pub async fn make_leader_engine(
    cancel_token: CancellationToken,
    // Full path to the model, either a GGUF file or an HF repo dir
    model_path: &Path,
    // Unique string to name zmq sockets
    sock_code: &str,
    // Multi node settings
    node_conf: MultiNodeConfig,
    // How many GPUs to use
    tensor_parallel_size: u32,
    // Path to extra engine args file
    extra_engine_args: Option<PathBuf>,
    // When using our vllm fork, this is how we publish it's KV metrics for the KV router
    kv_metrics_publisher: Option<Arc<KvMetricsPublisher>>,
) -> pipeline_error::Result<(ExecutionContext, impl Future<Output = ()>)> {
    let ray_obj = if node_conf.num_nodes > 1 {
        let r = ray::start_leader(node_conf.leader_addr.parse()?)?;
        tracing::info!("Leader waiting for {} total nodes", node_conf.num_nodes);
        r.wait_for(cancel_token.clone(), node_conf.num_nodes)
            .await?;
        tracing::info!("All nodes registered");
        Some(r)
    } else {
        None
    };

    let mut engine = VllmEngine::new(
        cancel_token,
        sock_code,
        model_path,
        node_conf,
        tensor_parallel_size,
        extra_engine_args,
        kv_metrics_publisher,
    )
    .await?;
    let vllm_process = engine.take_vllm_worker_handle();
    let vllm_future = async move {
        if let Err(err) = vllm_process.await {
            tracing::error!("Failed stopping vllm process: {err:#}");
        }
        if let Some(r) = ray_obj {
            if let Err(err) = r.stop().await {
                tracing::error!("Failed stopping ray: {err:#}");
            }
        }
    };
    let engine: ExecutionContext = Arc::new(engine);
    Ok((engine, vllm_future))
}

pub async fn start_follower(
    cancel_token: CancellationToken,
    node_conf: MultiNodeConfig,
) -> pipeline_error::Result<StopFuture> {
    let r = ray::start_follower(node_conf.leader_addr.parse()?)?;
    tracing::info!("Follower waiting for {} total nodes", node_conf.num_nodes);
    r.wait_for(cancel_token, node_conf.num_nodes).await?;
    tracing::info!("All nodes registered");

    Ok(StopFuture {
        state: Some(StopFutureState::New(r)),
    })
}

pub struct StopFuture {
    state: Option<StopFutureState>,
}

enum StopFutureState {
    New(Ray),
    Running(Pin<Box<dyn Future<Output = ()> + Send>>),
}

impl Future for StopFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let state = match self.state.take() {
            None => return Poll::Ready(()),
            Some(state) => state,
        };
        match state {
            StopFutureState::New(obj) => {
                // Convert object to a stop future
                let future = Box::pin(async move {
                    if let Err(err) = obj.stop().await {
                        tracing::error!("Failed calling 'ray stop': {err:#}");
                    }
                });
                self.state = Some(StopFutureState::Running(future));
                // Recurse to poll the new future immediately
                self.poll(cx)
            }
            StopFutureState::Running(mut future) => {
                // Poll the stop future
                match future.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        // Done, leave state as None
                        Poll::Ready(())
                    }
                    Poll::Pending => {
                        // Not ready yet, preserve the future
                        self.state = Some(StopFutureState::Running(future));
                        Poll::Pending
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn fix_venv(venv: String, py: Python<'_>) -> anyhow::Result<()> {
    let version_info = py.version_info();
    let sys: PyObject = py.import("sys")?.into();
    let sys_path = sys.getattr(py, "path")?;
    let venv_path = format!(
        "{venv}/lib/python{}.{}/site-packages",
        version_info.major, version_info.minor
    );
    // TODO: This should go _before_ the site-packages
    sys_path.call_method1(py, "append", (venv_path,))?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn fix_venv(_venv: String, _py: Python<'_>) -> anyhow::Result<()> {
    Ok(())
}
