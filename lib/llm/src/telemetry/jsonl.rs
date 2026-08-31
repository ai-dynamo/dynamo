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

/// Channel-backed handle for a buffered JSONL sink. Wraps a `Recorder<T>`,
/// which appends records to disk on its own background task. Drop cancels
/// the recorder.
pub struct JsonlWriter<T> {
    tx: mpsc::Sender<T>,
    // Holding the recorder keeps its background task alive; its Drop cancels.
    _recorder: Recorder<T>,
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
            tx,
            _recorder: recorder,
        })
    }

    pub async fn send(&self, rec: T) -> Result<(), mpsc::error::SendError<T>> {
        self.tx.send(rec).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use super::{JsonlSinkOptions, JsonlWriter};

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestRecord {
        id: u64,
        name: String,
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
}
