// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python hosts this Rust OpenEngine server; engine calls cross PyO3 directly.

mod bridge;
mod service;

use bridge::{Bridge, Ready, python_error};
use dynamo_trtllm_sidecar::proto as pb;
use pyo3::prelude::*;
use std::{
    net::{SocketAddr, TcpListener},
    thread::JoinHandle,
};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;

#[pyclass]
struct ServerHandle {
    shutdown: CancellationToken,
    thread: Option<JoinHandle<anyhow::Result<()>>>,
}

#[pymethods]
impl ServerHandle {
    fn is_alive(&self) -> bool {
        self.thread
            .as_ref()
            .is_some_and(|thread| !thread.is_finished())
    }

    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        self.shutdown.cancel();
        if let Some(thread) = self.thread.take() {
            // The server may be acquiring the GIL to cancel requests while we join.
            py.allow_threads(move || thread.join())
                .map_err(|_| python_error("OpenEngine server thread panicked"))?
                .map_err(python_error)?;
        }
        Ok(())
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[pyfunction]
#[pyo3(signature = (runtime_handle, model, listen="127.0.0.1:30001", context_length=8192, max_requests=32))]
fn start_server(
    py: Python<'_>,
    runtime_handle: Py<PyAny>,
    model: String,
    listen: &str,
    context_length: u32,
    max_requests: usize,
) -> PyResult<ServerHandle> {
    let address: SocketAddr = listen.parse().map_err(python_error)?;
    if !address.ip().is_loopback() || context_length < 2 || max_requests == 0 {
        return Err(python_error(
            "requires a loopback address, context_length >= 2, max_requests > 0",
        ));
    }
    let ready = Ready {
        vocab_size: runtime_handle.getattr(py, "vocab_size")?.extract(py)?,
        eos_token_id: runtime_handle.getattr(py, "eos_token_id")?.extract(py)?,
    };
    if ready.vocab_size == 0 || ready.eos_token_id >= ready.vocab_size {
        return Err(python_error("invalid tokenizer metadata"));
    }
    let listener = TcpListener::bind(address).map_err(python_error)?;
    listener.set_nonblocking(true).map_err(python_error)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("deepinfer-grpc")
        .enable_all()
        .build()
        .map_err(python_error)?;
    let bridge = Bridge::new(py, runtime_handle, ready, max_requests)?;
    let shutdown = CancellationToken::new();
    let stop = shutdown.clone();
    let thread = std::thread::Builder::new().name("deepinfer-openengine".into()).spawn(move || {
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::from_std(listener)?;
            let service = service::Service::new(bridge.clone(), model, context_length);
            let (reporter, health) = tonic_health::server::health_reporter();
            reporter.set_serving::<pb::inference_server::InferenceServer<service::Service>>().await;
            reporter.set_serving::<pb::control_server::ControlServer<service::Service>>().await;
            let cancellation = bridge.clone();
            let result = tonic::transport::Server::builder()
                .add_service(health)
                .add_service(pb::inference_server::InferenceServer::new(service.clone())
                    .max_decoding_message_size(64 * 1024 * 1024))
                .add_service(pb::control_server::ControlServer::new(service))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    tokio::select! { _ = stop.cancelled() => {}, _ = cancellation.failed.cancelled() => {} }
                    cancellation.abort(None);
                }).await;
            result?;
            anyhow::ensure!(!bridge.failed.is_cancelled(), "DeepInfer engine failed");
            Ok(())
        })
    }).map_err(python_error)?;
    Ok(ServerHandle {
        shutdown,
        thread: Some(thread),
    })
}

#[pymodule]
fn _deepinfer_openengine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    module.add_function(wrap_pyfunction!(start_server, module)?)?;
    module.add_class::<ServerHandle>()?;
    Ok(())
}
