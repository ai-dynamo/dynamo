// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tracing::Instrument;

use dynamo_runtime::{pipeline::SingleIn, protocols::maybe_error::MaybeError};

use crate::protocols::common::{
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::{ENCODER_RESULT_RELEASE_SUPPORTED, RELEASE_ENCODER_RESULT},
};

use super::EncodePushRouter;

/// Owns the producer allocation until downstream consumption finishes. The
/// payload remains engine-opaque; cleanup is an idempotent producer operation.
pub(super) struct EncoderResultCleanup {
    pub(super) router: Arc<EncodePushRouter>,
    pub(super) worker_id: u64,
    pub(super) request: Option<PreprocessedRequest>,
}

impl EncoderResultCleanup {
    pub(super) fn observe(&mut self, model: &str, output: &LLMEngineOutput) -> Result<()> {
        if output
            .extra_args
            .as_ref()
            .and_then(|args| args.get(ENCODER_RESULT_RELEASE_SUPPORTED))
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Ok(());
        }
        let Some(result) = output
            .encoder_result
            .as_ref()
            .filter(|value| value.is_object())
        else {
            return Ok(());
        };
        anyhow::ensure!(
            self.request.is_none(),
            "Encode worker returned multiple releasable results"
        );
        self.request = Some(
            PreprocessedRequest::builder()
                .model(model.to_string())
                .token_ids(Vec::new())
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .encoder_result(Some(result.clone()))
                .extra_args(Some(serde_json::json!({RELEASE_ENCODER_RESULT: true})))
                .build()?,
        );
        Ok(())
    }

    async fn release_request(
        router: Arc<EncodePushRouter>,
        worker_id: u64,
        request: PreprocessedRequest,
    ) {
        let release = async {
            // A fresh controller lets cleanup run after caller cancellation.
            // Resource cleanup must reach its producer despite inference overload.
            let mut stream = router
                .direct_cleanup(SingleIn::new(request), worker_id)
                .await?;
            let mut terminal = false;
            while let Some(item) = stream.next().await {
                if let Some(error) = item.err() {
                    anyhow::bail!("encoder result release failed: {error}");
                }
                terminal |= item
                    .data
                    .is_some_and(|output| output.finish_reason.is_some());
            }
            anyhow::ensure!(
                terminal,
                "encoder result release ended without a terminal chunk"
            );
            Ok::<_, anyhow::Error>(())
        };
        match tokio::time::timeout(Duration::from_secs(2), release).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(%error, worker_id, "Unable to release encoder result"),
            Err(_) => tracing::warn!(worker_id, "Encoder result release timed out"),
        }
    }
}

impl Drop for EncoderResultCleanup {
    fn drop(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        let router = self.router.clone();
        let worker_id = self.worker_id;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(Self::release_request(router, worker_id, request).in_current_span());
        } else {
            tracing::warn!(worker_id, "Runtime stopped before encoder result release");
        }
    }
}
