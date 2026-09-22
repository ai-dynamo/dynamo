// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Context;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

pub struct TestServer {
    address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<anyhow::Result<()>>>,
}

impl TestServer {
    pub async fn start<F, Fut>(serve: F) -> anyhow::Result<Self>
    where
        F: FnOnce(TcpListener, oneshot::Receiver<()>) -> Fut,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (shutdown, receiver) = oneshot::channel();
        let task = tokio::spawn(serve(listener, receiver));
        Ok(Self {
            address,
            shutdown: Some(shutdown),
            task: Some(task),
        })
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}", self.address)
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.as_mut() {
            let result = tokio::time::timeout(Duration::from_secs(10), task)
                .await
                .context("test server did not shut down")?;
            self.task.take();
            result.context("test server task failed")??;
        }
        Ok(())
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
