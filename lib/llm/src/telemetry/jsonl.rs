// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Context as _;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::broadcast::{
    Receiver,
    error::{RecvError, TryRecvError},
};
use tokio_util::sync::CancellationToken;

use crate::recorder::{Recorder, RecorderOptions};

#[derive(Clone, Copy, Debug)]
pub struct JsonlSinkOptions {
    pub buffer_bytes: usize,
    pub flush_interval: Duration,
}

impl Default for JsonlSinkOptions {
    fn default() -> Self {
        Self {
            buffer_bytes: 32768,
            flush_interval: Duration::from_millis(1000),
        }
    }
}

/// Spawn an async JSONL sink for records received from a typed telemetry bus.
pub async fn spawn_jsonl_worker_with_shutdown<T>(
    mut rx: Receiver<T>,
    path: String,
    options: JsonlSinkOptions,
    shutdown: CancellationToken,
) -> anyhow::Result<()>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    let recorder_shutdown = CancellationToken::new();
    let recorder: Recorder<T> = Recorder::new_with_options(
        recorder_shutdown.clone(),
        &path,
        RecorderOptions {
            buffer_bytes: options.buffer_bytes.max(1),
            flush_interval: Some(options.flush_interval.max(Duration::from_millis(1))),
            append: true,
            ..Default::default()
        },
    )
    .await
    .with_context(|| format!("opening jsonl telemetry sink at {path}"))?;

    let tx = recorder.event_sender();
    tokio::spawn(async move {
        let _recorder = recorder;
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    loop {
                        match rx.try_recv() {
                            Ok(rec) => {
                                if tx.send(rec).await.is_err() {
                                    break;
                                }
                            }
                            Err(TryRecvError::Lagged(n)) => {
                                tracing::warn!(dropped = n, "telemetry bus lagged during shutdown; dropped records");
                            }
                            Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                        }
                    }
                    recorder_shutdown.cancel();
                    return;
                }
                msg = rx.recv() => {
                    match msg {
                        Ok(rec) => {
                            if tx.send(rec).await.is_err() {
                                break;
                            }
                        }
                        Err(RecvError::Lagged(n)) => {
                            tracing::warn!(dropped = n, "telemetry bus lagged; dropped records")
                        }
                        Err(RecvError::Closed) => break,
                    }
                }
            }
        }
        recorder_shutdown.cancel();
    });

    tracing::info!(
        path,
        buffer_bytes = options.buffer_bytes,
        flush_interval_ms = options.flush_interval.as_millis(),
        "async JSONL telemetry sink ready"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use crate::telemetry::bus::TelemetryBus;

    use super::{JsonlSinkOptions, spawn_jsonl_worker_with_shutdown};

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestRecord {
        id: u64,
        name: String,
    }

    #[tokio::test]
    async fn writes_jsonl_from_bus() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("telemetry.jsonl");
        let path_string = path.to_string_lossy().to_string();
        let shutdown = tokio_util::sync::CancellationToken::new();
        let bus = TelemetryBus::<TestRecord>::new();
        bus.init(4);

        spawn_jsonl_worker_with_shutdown(
            bus.subscribe(),
            path_string,
            JsonlSinkOptions {
                buffer_bytes: 64,
                flush_interval: Duration::from_millis(5),
            },
            shutdown.clone(),
        )
        .await
        .unwrap();

        bus.publish(TestRecord {
            id: 1,
            name: "record".to_string(),
        });

        let mut content = String::new();
        for _ in 0..50 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"name\":\"record\"") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        shutdown.cancel();

        let line = content.lines().next().expect("jsonl line");
        let wrapper: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(wrapper.get("timestamp").is_some());
        assert_eq!(
            serde_json::from_value::<TestRecord>(wrapper["event"].clone()).unwrap(),
            TestRecord {
                id: 1,
                name: "record".to_string()
            }
        );
    }
}
