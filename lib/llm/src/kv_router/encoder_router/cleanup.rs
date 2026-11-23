// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use tracing::Instrument;

use dynamo_runtime::{
    error::{ErrorClass, match_error_chain},
    pipeline::{CleanupRoute, SingleIn},
    protocols::maybe_error::MaybeError,
};

use crate::protocols::common::{
    FinishReason,
    llm_backend::{LLMEngineOutput, PreprocessedRequest},
    preprocessor::{ENCODER_RESULT_RELEASE_SUPPORTED, RELEASE_ENCODER_RESULT},
};

use crate::types::Annotated;

/// Owns the producer allocation until downstream consumption finishes. The
/// payload remains engine-opaque; cleanup is an idempotent producer operation.
pub(super) struct EncoderResultCleanup {
    pub(super) route: CleanupRoute<PreprocessedRequest, Annotated<LLMEngineOutput>>,
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
            .as_deref()
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

    pub(super) async fn release(&mut self) {
        if let Some(request) = &self.request {
            // Retain ownership across the await so cancellation still schedules cleanup.
            Self::release_request(self.route.clone(), request.clone()).await;
            self.request = None;
        }
    }

    async fn release_request(
        route: CleanupRoute<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        request: PreprocessedRequest,
    ) {
        let worker_id = route.worker_id();
        let release = async {
            let mut retried = false;
            loop {
                match Self::release_once(&route, &request).await {
                    Ok(()) => return Ok(()),
                    Err(error)
                        if !retried
                            && match_error_chain(
                                error.as_ref(),
                                &[
                                    ErrorClass::Unavailable,
                                    ErrorClass::CannotConnect,
                                    ErrorClass::Disconnected,
                                    ErrorClass::ConnectionTimeout,
                                    ErrorClass::ResponseTimeout,
                                ],
                                &[],
                            ) =>
                    {
                        retried = true;
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    Err(error) => return Err(error),
                }
            }
        };
        match tokio::time::timeout(Duration::from_secs(2), release).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(
                %error,
                worker_id,
                "Encoder result release unconfirmed; producer reclamation required"
            ),
            Err(_) => tracing::warn!(
                worker_id,
                "Encoder result release timed out; producer reclamation required"
            ),
        }
    }

    pub(super) async fn release_once(
        route: &CleanupRoute<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        request: &PreprocessedRequest,
    ) -> Result<()> {
        // A fresh controller keeps cleanup independent of caller cancellation.
        let mut stream = route.dispatch(SingleIn::new(request.clone())).await?;
        let mut acknowledged = false;
        while let Some(item) = stream.next().await {
            if let Some(error) = item.err() {
                return Err(anyhow::Error::new(error).context("encoder result release failed"));
            }
            if let Some(reason) = item.data.and_then(|output| output.finish_reason) {
                anyhow::ensure!(
                    reason == FinishReason::Stop,
                    "encoder result release rejected: {reason}"
                );
                acknowledged = true;
            }
        }
        anyhow::ensure!(
            acknowledged,
            "encoder result release ended without a successful terminal chunk"
        );
        Ok(())
    }
}

impl Drop for EncoderResultCleanup {
    fn drop(&mut self) {
        let Some(request) = self.request.take() else {
            return;
        };
        let route = self.route.clone();
        let worker_id = route.worker_id();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(Self::release_request(route, request).in_current_span());
        } else {
            tracing::warn!(worker_id, "Runtime stopped before encoder result release");
        }
    }
}
