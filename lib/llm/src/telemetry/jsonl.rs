// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Context as _;
use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::mpsc;
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

/// Channel-backed buffered JSONL sink.
///
/// Drop may abandon queued records; use [`Self::shutdown`] to drain them.
pub struct JsonlWriter<T> {
    tx: Option<mpsc::Sender<T>>,
    recorder: Option<Recorder<T>>,
}

impl<T> JsonlWriter<T>
where
    T: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    pub async fn new(path: String, options: JsonlSinkOptions) -> anyhow::Result<Self> {
        let recorder_shutdown = CancellationToken::new();
        let recorder: Recorder<T> = Recorder::new_with_options(
            recorder_shutdown,
            &path,
            RecorderOptions {
                buffer_bytes: options.buffer_bytes.max(1),
                flush_interval: Some(options.flush_interval.max(Duration::from_millis(1))),
                append: true,
                ..Default::default()
            },
        )
        .await
        .with_context(|| format!("opening jsonl sink at {path}"))?;
        let tx = recorder.event_sender();
        Ok(Self {
            tx: Some(tx),
            recorder: Some(recorder),
        })
    }

    pub async fn send(&self, rec: T) -> Result<(), mpsc::error::SendError<T>> {
        match &self.tx {
            Some(tx) => tx.send(rec).await,
            None => Err(mpsc::error::SendError(rec)),
        }
    }

    /// Stops accepting records, drains the queue, and flushes. Calls are idempotent.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.tx.take();
        if let Some(recorder) = self.recorder.take() {
            recorder.close().await?;
        }
        Ok(())
    }

    /// Drains and consumes the writer.
    pub async fn close(mut self) -> anyhow::Result<()> {
        self.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Serialize, Serializer};
    use tempfile::tempdir;

    use super::{JsonlSinkOptions, JsonlWriter};

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestRecord {
        id: u64,
        name: String,
    }

    /// Parks the recorder inside `Serialize` without timing assumptions.
    struct BarrierGate {
        parked_tx: tokio::sync::mpsc::UnboundedSender<()>,
        release_rx: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    /// Record that can block during serialization.
    #[derive(Clone, Deserialize)]
    struct BarrierRecord {
        id: u64,
        #[serde(skip)]
        gate: Option<Arc<BarrierGate>>,
    }

    impl Serialize for BarrierRecord {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            if let Some(gate) = &self.gate {
                gate.parked_tx.send(()).expect("test still listening");
                // Ignore the result: a dropped release sender also releases us.
                let _ = gate.release_rx.lock().unwrap().recv();
            }
            let mut state = serializer.serialize_struct("BarrierRecord", 1)?;
            state.serialize_field("id", &self.id)?;
            state.end()
        }
    }

    #[tokio::test]
    async fn writes_record_to_jsonl_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("telemetry.jsonl");

        let writer: JsonlWriter<TestRecord> = JsonlWriter::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 64,
                flush_interval: Duration::from_millis(5),
            },
        )
        .await
        .unwrap();

        writer
            .send(TestRecord {
                id: 1,
                name: "record".to_string(),
            })
            .await
            .unwrap();

        let mut content = String::new();
        for _ in 0..50 {
            content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
            if content.contains("\"name\":\"record\"") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_drains_records_queued_behind_a_busy_writer() {
        const RECORDS: u64 = 32;

        let dir = tempdir().unwrap();
        let path = dir.path().join("barrier.jsonl");

        let writer: JsonlWriter<BarrierRecord> = JsonlWriter::new(
            path.display().to_string(),
            JsonlSinkOptions {
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
            },
        )
        .await
        .unwrap();

        let (parked_tx, mut parked_rx) = tokio::sync::mpsc::unbounded_channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(BarrierGate {
            parked_tx,
            release_rx: Mutex::new(release_rx),
        });

        writer
            .send(BarrierRecord {
                id: 1,
                gate: Some(gate),
            })
            .await
            .unwrap();
        parked_rx.recv().await.expect("writer task parked");

        for id in 2..=RECORDS {
            writer
                .send(BarrierRecord { id, gate: None })
                .await
                .expect("send accepted while writer is busy");
        }

        let shutdown = tokio::spawn(async move { writer.close().await });
        release_tx.send(()).expect("writer task waiting on release");
        shutdown.await.unwrap().expect("shutdown");

        let content = std::fs::read_to_string(&path).unwrap();
        let ids: Vec<u64> = content
            .lines()
            .map(|line| {
                let wrapper: serde_json::Value = serde_json::from_str(line).unwrap();
                wrapper["event"]["id"].as_u64().unwrap()
            })
            .collect();
        assert_eq!(
            ids,
            (1..=RECORDS).collect::<Vec<_>>(),
            "every accepted record must be written exactly once, in order"
        );
    }
}
