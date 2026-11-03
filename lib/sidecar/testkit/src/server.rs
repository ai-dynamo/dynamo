// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::net::SocketAddr;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Context;
use tokio::net::TcpListener;
use tokio::sync::{oneshot, watch};

pub struct TestServer {
    address: SocketAddr,
    shutdown: watch::Sender<bool>,
    completed: Option<oneshot::Receiver<anyhow::Result<()>>>,
    thread: Option<JoinHandle<()>>,
}

impl TestServer {
    pub async fn start<F, Fut>(serve: F) -> anyhow::Result<Self>
    where
        F: FnOnce(TcpListener, oneshot::Receiver<()>) -> Fut + Send + 'static,
        Fut: Future<Output = anyhow::Result<()>> + Send + 'static,
    {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let (shutdown, mut stopping) = watch::channel(false);
        let (completed_tx, completed) = oneshot::channel();
        let thread = std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<()> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                // The dedicated runtime owns tonic's connection tasks and RPC handlers too.
                runtime.block_on(async move {
                    let listener = TcpListener::from_std(listener)?;
                    let (_graceful, receiver) = oneshot::channel();
                    tokio::select! {
                        biased;
                        _ = stopping.wait_for(|stopped| *stopped) => Ok(()),
                        result = serve(listener, receiver) => result,
                    }
                })
            })();
            let _ = completed_tx.send(result);
        });
        Ok(Self {
            address,
            shutdown,
            completed: Some(completed),
            thread: Some(thread),
        })
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}", self.address)
    }

    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.shutdown.send_replace(true);
        if let Some(completed) = self.completed.as_mut() {
            tokio::time::timeout(Duration::from_secs(10), completed)
                .await
                .context("test server did not shut down")?
                .context("test server thread failed")??;
            self.completed.take();
            self.thread
                .take()
                .unwrap()
                .join()
                .map_err(|_| anyhow::anyhow!("test server panicked"))?;
        }
        Ok(())
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.shutdown.send_replace(true);
    }
}
