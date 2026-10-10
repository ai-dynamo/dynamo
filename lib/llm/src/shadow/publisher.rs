// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The background half of a tap: encode and publish, off the request path.

use std::time::Duration;

use anyhow::Result;
use dynamo_runtime::pipeline::async_trait;
use dynamo_runtime::transports::event_plane::EventPublisher;
use prometheus::IntCounter;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::envelope::ShadowEnvelope;

/// How long a stopping frontend keeps publishing what is already queued. The
/// tail of a run is worth a short wait; a stalled shadow is not worth a hang.
const DRAIN_DEADLINE: Duration = Duration::from_secs(2);

#[async_trait]
pub trait EnvelopeSink: Send + Sync + 'static {
    async fn publish(&self, envelope: &ShadowEnvelope) -> Result<()>;
}

#[async_trait]
impl EnvelopeSink for EventPublisher {
    async fn publish(&self, envelope: &ShadowEnvelope) -> Result<()> {
        EventPublisher::publish(self, envelope).await
    }
}

pub async fn run<S: EnvelopeSink>(
    tap: String,
    mut queue: mpsc::Receiver<ShadowEnvelope>,
    sink: S,
    publish_errors: IntCounter,
    shutdown: CancellationToken,
) {
    let publish = async |envelope: ShadowEnvelope| {
        if let Err(error) = sink.publish(&envelope).await {
            publish_errors.inc();
            tracing::warn!(tap, seq = envelope.seq, %error, "shadow tap publish failed");
        }
    };

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            envelope = queue.recv() => match envelope {
                Some(envelope) => publish(envelope).await,
                None => return,
            },
        }
    }

    queue.close();
    let drain = async {
        while let Some(envelope) = queue.recv().await {
            publish(envelope).await;
        }
    };
    if tokio::time::timeout(DRAIN_DEADLINE, drain).await.is_err() {
        tracing::warn!(
            tap,
            "shadow tap drain deadline passed; queued records were lost"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::shadow::envelope::{ENVELOPE_SCHEMA_VERSION, ShadowOrigin};

    #[derive(Clone, Default)]
    struct Collect(Arc<Mutex<Vec<u64>>>);

    #[async_trait]
    impl EnvelopeSink for Collect {
        async fn publish(&self, envelope: &ShadowEnvelope) -> Result<()> {
            self.0.lock().unwrap().push(envelope.seq);
            if envelope.seq == 1 {
                anyhow::bail!("transport refused");
            }
            Ok(())
        }
    }

    fn envelope(seq: u64) -> ShadowEnvelope {
        ShadowEnvelope {
            schema_version: ENVELOPE_SCHEMA_VERSION,
            tap: "t".into(),
            seq,
            request_id: String::new(),
            origin: ShadowOrigin::Chat,
            filters: Vec::new().into(),
            arrival_unix_ns: 0,
            request: Arc::new(
                crate::protocols::common::preprocessor::PreprocessedRequest::builder()
                    .model(String::new())
                    .token_ids(Vec::new())
                    .stop_conditions(Default::default())
                    .sampling_options(Default::default())
                    .output_options(Default::default())
                    .build()
                    .unwrap(),
            ),
            response: None,
        }
    }

    #[tokio::test]
    async fn shutdown_drains_the_queue_and_a_failed_publish_does_not_stop_the_task() {
        let (tx, rx) = mpsc::channel(8);
        for seq in 0..3 {
            tx.try_send(envelope(seq)).unwrap();
        }
        let sink = Collect::default();
        let errors = IntCounter::new("errors", "errors").unwrap();
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        tokio::time::timeout(
            Duration::from_secs(5),
            run("t".to_string(), rx, sink.clone(), errors.clone(), shutdown),
        )
        .await
        .expect("publisher task must exit after shutdown");

        assert_eq!(*sink.0.lock().unwrap(), vec![0, 1, 2]);
        assert_eq!(errors.get(), 1);
        assert!(
            tx.try_send(envelope(3)).is_err(),
            "queue is closed after shutdown"
        );
    }
}
