// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::error::Error as StdError;
use std::sync::Arc;

use anyhow::{Error, Result};
use futures::{stream, stream::StreamExt};

use crate::{
    http::service::metrics::Metrics,
    model_card::ModelDeploymentCard,
    protocols::{
        TokenIdType,
        common::llm_backend::{BackendOutput, LLMEngineOutput, PreprocessedRequest},
    },
};

use dynamo_runtime::engine::Data;
use dynamo_runtime::error::{self, BackendError, DynamoError, ErrorType};
use dynamo_runtime::pipeline::{
    AsyncEngineContext, AsyncEngineContextProvider, Context, ManyOut, Operator, PipelineOperator,
    ResponseStream, ServerStreamingEngine, SingleIn, async_trait,
};
use dynamo_runtime::protocols::annotated::Annotated;

/// Access to the generated token IDs of a response, so migration can
/// append already-delivered tokens onto the replayed request.
pub(crate) trait HasTokenIds {
    fn token_ids(&self) -> &[TokenIdType];
}

impl HasTokenIds for BackendOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
}

impl HasTokenIds for LLMEngineOutput {
    fn token_ids(&self) -> &[TokenIdType] {
        &self.token_ids
    }
}

/// Check if an error chain indicates the request should be migrated.
fn is_migratable(err: &(dyn StdError + 'static)) -> bool {
    const MIGRATABLE: &[ErrorType] = &[
        ErrorType::CannotConnect,
        ErrorType::Disconnected,
        ErrorType::ConnectionTimeout,
        ErrorType::Backend(BackendError::EngineShutdown),
    ];
    const NON_MIGRATABLE: &[ErrorType] = &[ErrorType::Cancelled, ErrorType::ResourceExhausted];
    error::match_error_chain(err, MIGRATABLE, NON_MIGRATABLE)
}

pub struct Migration {
    migration_limit: u32,
    max_seq_len: Option<u32>,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
}

impl Migration {
    pub fn new(
        migration_limit: u32,
        max_seq_len: Option<u32>,
        model_name: String,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        tracing::debug!(
            "model {} migration limit {} max_seq_len {:?}",
            model_name,
            migration_limit,
            max_seq_len
        );
        Arc::new(Self {
            migration_limit,
            max_seq_len,
            model_name: Arc::new(model_name),
            metrics,
        })
    }

    pub fn from_mdc(
        mdc: &ModelDeploymentCard,
        migration_limit: u32,
        max_seq_len: Option<u32>,
        metrics: Arc<Metrics>,
    ) -> Arc<Self> {
        Self::new(
            migration_limit,
            max_seq_len,
            mdc.display_name.clone(),
            metrics,
        )
    }

    /// Wrap as a `PipelineOperator` over the given response type to
    /// disambiguate between the `Operator` impls on `Migration` since
    /// the response type doesn't appear in the struct.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_operator_for<Resp>(
        self: &Arc<Self>,
    ) -> Arc<
        PipelineOperator<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<Resp>>,
        >,
    >
    where
        Resp: Data + HasTokenIds,
    {
        Operator::into_operator(self)
    }
}

#[async_trait]
impl<Resp>
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<Resp>>,
    > for Migration
where
    Resp: Data + HasTokenIds,
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    ) -> Result<ManyOut<Annotated<Resp>>> {
        let (preprocessed_request, context) = request.transfer(());
        let engine_ctx = context.context();
        let engine_ctx_ = engine_ctx.clone();
        let retry_manager = RetryManager::build(
            engine_ctx,
            preprocessed_request,
            next,
            self.migration_limit,
            self.max_seq_len,
            self.model_name.clone(),
            self.metrics.clone(),
        )
        .await?;
        let response_stream = stream::unfold(retry_manager, move |mut retry_manager| async move {
            retry_manager
                .next()
                .await
                .map(|response| (response, retry_manager))
        })
        .fuse();
        Ok(ResponseStream::new(Box::pin(response_stream), engine_ctx_))
    }
}

struct RetryManager<Resp>
where
    Resp: Data + HasTokenIds,
{
    context: Arc<dyn AsyncEngineContext>,
    request: PreprocessedRequest,
    next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
    next_stream: Option<ManyOut<Annotated<Resp>>>,
    retries_left: u32,
    max_seq_len: Option<u32>,
    model_name: Arc<String>,
    metrics: Arc<Metrics>,
}

impl<Resp> RetryManager<Resp>
where
    Resp: Data + HasTokenIds,
{
    pub async fn build(
        context: Arc<dyn AsyncEngineContext>,
        preprocessed_request: PreprocessedRequest,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<Resp>>,
        mut retries_left: u32,
        max_seq_len: Option<u32>,
        model_name: Arc<String>,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        // Disable migration for structured-output (guided-decoding) requests.
        // Inference backends initialize the guided-decoding FSM (finite state machine) fresh
        // for every new request and only advance it on newly-generated tokens, not on
        // context/prompt tokens. Migrating a partial structured-output response would replay
        // already-generated tokens as context, causing the FSM to restart from the schema
        // root and producing duplicated or nested JSON. This applies to all backends
        // (vLLM, SGLang, TRT-LLM) equally. Propagate the error cleanly instead.
        if preprocessed_request
            .sampling_options
            .guided_decoding
            .is_some()
        {
            if retries_left > 0 {
                tracing::warn!(
                    "Guided-decoding request: migration disabled — FSM state is not transferable (applies to all backends)"
                );
            }
            retries_left = 0;
        }

        if preprocessed_request.sampling_options.n.unwrap_or(1) > 1 {
            if retries_left > 0 {
                tracing::warn!(
                    "n>1 request: migration disabled - per-choice generation state is not transferable"
                );
            }
            retries_left = 0;
        }
        let mut slf = Self {
            context,
            request: preprocessed_request,
            next_generate: next,
            next_stream: None,
            retries_left: retries_left + 1, // +1 to account for the initial attempt
            max_seq_len,
            model_name,
            metrics,
        };
        slf.new_stream().await?;
        slf.exceed_max_seq_len(0); // disable migration if prompt len > max_seq_len
        Ok(slf)
    }

    pub async fn next(&mut self) -> Option<Annotated<Resp>> {
        loop {
            let response_stream = match self.next_stream.as_mut() {
                Some(stream) => stream,
                None => {
                    tracing::error!("next() called with next_stream is None - should not happen");
                    return Some(Annotated::from_error("next_stream is None"));
                }
            };
            if let Some(response) = response_stream.next().await {
                // Check if this is a migratable error that should trigger stream recreation.
                if let Some(err) = response.error.as_ref()
                    && is_migratable(err)
                {
                    tracing::warn!(error = %err, "Stream disconnected, recreating stream");
                    self.metrics.inc_migration_ongoing_request(&self.model_name);
                    if let Err(err) = self.new_stream().await {
                        tracing::warn!(error = ?err, "Cannot recreate stream");
                    } else {
                        continue;
                    }
                }
                self.track_response(&response);
                return Some(response);
            }
            return None;
        }
    }

    async fn new_stream(&mut self) -> Result<()> {
        let mut response_stream: Option<Result<ManyOut<Annotated<Resp>>>> = None;
        while self.retries_left > 0 {
            self.retries_left -= 1;
            let request = Context::with_id(self.request.clone(), self.context.id().to_string());
            self.context.link_child(request.context());
            if self.context.is_stopped() || self.context.is_killed() {
                tracing::debug!("Abort creating new stream after context is stopped or killed");
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Cancelled)
                    .message(format!(
                        "Context id {} is stopped or killed",
                        self.context.id()
                    ))
                    .build()
                    .into());
            }
            response_stream = Some(self.next_generate.generate(request).await);
            if let Some(err) = response_stream.as_ref().unwrap().as_ref().err()
                && is_migratable(err.as_ref())
            {
                tracing::warn!(error = %err, "Creating new stream, retrying");
                self.metrics.inc_migration_new_request(&self.model_name);
                continue;
            }
            break;
        }
        match response_stream {
            Some(Ok(next_stream)) => {
                self.next_stream = Some(next_stream);
                Ok(())
            }
            Some(Err(err)) => Err(err), // should propagate original error if any
            None => Err(Error::msg(
                "Migration limit exhausted", // should propagate original error if any
            )),
        }
    }

    fn track_response(&mut self, response: &Annotated<Resp>) {
        if self.retries_left == 0 {
            return;
        }
        let llm_engine_output = match response.data.as_ref() {
            Some(output) => output,
            None => return,
        };
        let token_ids = llm_engine_output.token_ids();
        if self.exceed_max_seq_len(token_ids.len() as u32) {
            return;
        }
        if let Some(max_tokens) = self.request.stop_conditions.max_tokens {
            self.request.stop_conditions.max_tokens =
                Some(max_tokens.saturating_sub(token_ids.len() as u32));
        }
        for token_id in token_ids.iter() {
            self.request.token_ids.push(*token_id);
        }
    }

    /// Returns `true` if the tracked request token length plus `new_output_len`
    /// exceeds the configured max_seq_len, in which case migration is disabled.
    fn exceed_max_seq_len(&mut self, new_output_len: u32) -> bool {
        if let Some(max_seq_len) = self.max_seq_len {
            let total_len = self.request.token_ids.len() as u32 + new_output_len;
            if total_len > max_seq_len {
                tracing::warn!(
                    "Sequence length {} exceeds migration max_seq_len {}, \
                     disabling migration",
                    total_len,
                    max_seq_len
                );
                self.metrics
                    .inc_migration_max_seq_len_exceeded(&self.model_name);
                self.retries_left = 0; // disable migration
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::metrics::Metrics;
    use crate::protocols::common::{
        GuidedDecodingOptions, OutputOptions, SamplingOptions, StopConditions,
    };
    use dynamo_runtime::error::{DynamoError, ErrorType};
    use dynamo_runtime::pipeline::AsyncEngine;
    use dynamo_runtime::pipeline::context::Controller;
    use dynamo_runtime::protocols::maybe_error::MaybeError;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::mpsc;

    const TEST_MODEL: &str = "test-model";

    // Helper to create a mock preprocessed request
    fn create_mock_request(max_tokens: u32) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("mock".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions {
                max_tokens: Some(max_tokens),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    // Helper to create mock LLM engine output
    fn create_mock_output(token_id: u32) -> Annotated<BackendOutput> {
        Annotated::from_data(BackendOutput {
            token_ids: vec![token_id],
            tokens: vec![],
            text: Some(format!("token_{token_id}")),
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            stop_reason: None,
            index: None,
            disaggregated_params: None,
            completion_usage: None,
            engine_data: None,
        })
    }

    #[derive(Debug, Clone)]
    enum MockBehavior {
        /// Always succeeds with all responses
        Success,
        /// Fails on first call with NoResponders error, then succeeds on subsequent calls
        FailThenSuccess,
        /// Succeeds initially, fails mid-stream with specific error, then succeeds on retry
        MidStreamFail { fail_after: usize },
        /// Succeeds initially, fails mid-stream with specific error, then always fails on retry attempts
        MidStreamFailAlways { fail_after: usize },
        /// Succeeds initially, fails mid-stream, then always fails with stream error on retry attempts
        MidStreamFailAlwaysStreamError { fail_after: usize },
        /// Always fails with NoResponders error (same as FailThenSuccess first call)
        AlwaysFail,
    }

    // Unified mock server streaming engine that can simulate different scenarios
    struct MockEngine {
        behavior: MockBehavior,
        num_responses: usize,
        token_offset: u32,
        call_count: Arc<AtomicU32>,
        context_id: String,
    }

    impl MockEngine {
        fn new(
            behavior: MockBehavior,
            num_responses: usize,
            token_offset: u32,
            context_id: String,
        ) -> Self {
            Self {
                behavior,
                num_responses,
                token_offset,
                call_count: Arc::new(AtomicU32::new(0)),
                context_id,
            }
        }
    }

    #[async_trait]
    impl
        AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<BackendOutput>>, anyhow::Error>
        for MockEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let call_num = self.call_count.fetch_add(1, Ordering::SeqCst);
            let (preprocessed_request, context) = request.transfer(());

            // Assert that the context_id matches the expected one
            assert_eq!(
                context.id().to_string(),
                self.context_id,
                "Context ID mismatch"
            );

            // Calculate how many responses we've already generated based on request token_ids
            // Initial request has [1, 2, 3], so anything beyond that are generated responses
            let initial_tokens = 3; // [1, 2, 3]
            let responses_already_generated = preprocessed_request
                .token_ids
                .len()
                .saturating_sub(initial_tokens);

            // Assert that max_tokens reflects the expected remaining tokens
            let expected_max_tokens =
                self.num_responses
                    .saturating_sub(responses_already_generated) as u32;
            assert_eq!(
                preprocessed_request.stop_conditions.max_tokens,
                Some(expected_max_tokens),
                "max_tokens should be {} but got {:?}",
                expected_max_tokens,
                preprocessed_request.stop_conditions.max_tokens
            );

            match &self.behavior {
                MockBehavior::Success => {
                    // Always succeed with remaining responses
                    self.send_responses(responses_already_generated, self.num_responses)
                        .await
                }
                MockBehavior::FailThenSuccess => {
                    if call_num == 0 {
                        // First call - return "No responders available" error to trigger retry
                        return Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ));
                    } else {
                        // Subsequent calls - succeed with remaining responses
                        self.send_responses(responses_already_generated, self.num_responses)
                            .await
                    }
                }
                MockBehavior::MidStreamFail { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });
                    } else {
                        // Second call - send remaining responses from where we left off
                        tokio::spawn(async move {
                            for i in responses_already_generated..num_responses {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                        });
                    }

                    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                    let ctx = Arc::new(Controller::new(self.context_id.clone()));
                    Ok(dynamo_runtime::pipeline::ResponseStream::new(
                        Box::pin(stream),
                        ctx,
                    ))
                }
                MockBehavior::MidStreamFailAlways { fail_after } => {
                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        let (tx, rx) = mpsc::channel(1);
                        let token_offset = self.token_offset;
                        let fail_after = *fail_after;
                        let num_responses = self.num_responses;

                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - always fail with NoResponders error (same as AlwaysFail)
                        Err(anyhow::anyhow!(
                            DynamoError::builder()
                                .error_type(ErrorType::CannotConnect)
                                .message("no responders")
                                .build()
                        ))
                    }
                }
                MockBehavior::MidStreamFailAlwaysStreamError { fail_after } => {
                    let (tx, rx) = mpsc::channel(1);
                    let token_offset = self.token_offset;
                    let fail_after = *fail_after;
                    let num_responses = self.num_responses;

                    if call_num == 0 {
                        // First call - send some responses then an error to simulate disconnection
                        tokio::spawn(async move {
                            // Send responses from current position to fail_after
                            for i in responses_already_generated..fail_after.min(num_responses) {
                                let response = create_mock_output(token_offset + 1 + i as u32);
                                if tx.send(response).await.is_err() {
                                    break;
                                }
                            }
                            // Send the specific error that triggers retry logic
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    } else {
                        // Subsequent calls - immediately send stream error (no successful responses)
                        tokio::spawn(async move {
                            // Send the stream error immediately
                            let error_response = Annotated::from_err(
                                DynamoError::builder()
                                    .error_type(ErrorType::Disconnected)
                                    .message("Stream ended before generation completed")
                                    .build(),
                            );
                            let _ = tx.send(error_response).await;
                        });

                        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
                        let ctx = Arc::new(Controller::new(self.context_id.clone()));
                        Ok(dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(stream),
                            ctx,
                        ))
                    }
                }
                MockBehavior::AlwaysFail => {
                    // Always fail with NoResponders error (same as FailThenSuccess first call)
                    Err(anyhow::anyhow!(
                        DynamoError::builder()
                            .error_type(ErrorType::CannotConnect)
                            .message("no responders")
                            .build()
                    ))
                }
            }
        }
    }

    impl MockEngine {
        async fn send_responses(
            &self,
            start: usize,
            end: usize,
        ) -> Result<ManyOut<Annotated<BackendOutput>>> {
            let (tx, rx) = mpsc::channel(1);
            let token_offset = self.token_offset;

            tokio::spawn(async move {
                for i in start..end {
                    let response = create_mock_output(token_offset + 1 + i as u32);
                    if tx.send(response).await.is_err() {
                        break;
                    }
                }
            });

            let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
            let ctx = Arc::new(Controller::new(self.context_id.clone()));
            Ok(dynamo_runtime::pipeline::ResponseStream::new(
                Box::pin(stream),
                ctx,
            ))
        }
    }

    /// Test case 1: No migration needed
    /// Tests the normal case where the RetryManager successfully processes all responses
    /// from a single stream without any failures or need for retries/migration.
    /// Expected behavior: All 10 responses should be received successfully.
    #[tokio::test]
    async fn test_retry_manager_no_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            0,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 2: New request migration
    /// Tests the scenario where a worker becomes unreachable for new requests initially,
    /// triggering the RetryManager to retry the request. The MockEngine with FailThenSuccess
    /// fails on the first call with a "No responders available" error, then succeeds
    /// on subsequent calls, simulating a worker becoming available after initial failure.
    /// Expected behavior: All 10 responses should be received successfully after retry.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::FailThenSuccess,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        assert_eq!(responses.len(), 10);
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 1);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 3: Ongoing request migration
    /// Tests the scenario where a worker fails mid-stream during an ongoing request.
    /// This simulates a connection being lost after partial response delivery, requiring
    /// the RetryManager to detect the failure (via "Stream ended before generation completed" error),
    /// create a new stream, and continue from where it left off.
    /// Expected behavior: 5 responses from first stream + 5 responses from retry stream = 10 total.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 5 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received all 10 responses (5 from first stream + 5 from second stream)
        assert_eq!(responses.len(), 10);

        // Check that we received responses from both streams
        for (i, response) in responses.iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103, ..., 110
            }
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
    }

    /// Test case 4: New request migration - indefinite failure
    /// Tests the scenario where a worker becomes unreachable for new requests indefinitely.
    /// The RetryManager should exhaust all retries and return the original error from the first attempt.
    /// Expected behavior: Should receive an error after all retries are exhausted, with the original error.
    #[tokio::test]
    async fn test_retry_manager_new_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(0);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::AlwaysFail,
            0,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        // Should fail to build due to initial stream creation failure after exhausting all 3 retries
        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(error.to_string().contains("no responders"));
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 5: Ongoing request migration - indefinite failure
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests.
    /// The RetryManager should exhaust all retries and return the original stream disconnection error.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlways { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 3); // 2 retries + 1 final failure
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1); // initial ongoing failure retry
    }

    /// Test case 6: Ongoing request migration - indefinite failure with stream errors
    /// Tests the scenario where a worker fails mid-stream indefinitely during ongoing requests,
    /// and all retry attempts also fail with stream errors instead of NATS errors.
    /// Expected behavior: Should receive some responses from first stream, then error after retries exhausted.
    #[tokio::test]
    async fn test_retry_manager_ongoing_request_migration_indefinite_failure_stream_error() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFailAlwaysStreamError { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        ) // 3 retries
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();

        // Collect all responses (both successful and error responses)
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Should have received 4 total responses: 3 successful + 1 error
        assert_eq!(responses.len(), 4);

        // First 3 responses should be successful with tokens 101, 102, 103
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none());
            if let Some(output) = &response.data {
                assert_eq!(output.token_ids, vec![101 + i as u32]); // 101, 102, 103
            }
        }

        // 4th response should be a Disconnected error after retries are exhausted
        let error_response = &responses[3];
        let err = error_response.err().expect("expected error response");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 4); // 3 retries + 1 final failure
    }

    /// Test case 7: Request cancelled when creating new stream
    /// Tests the scenario where context.stop_generating() is called when creating a new stream.
    /// The RetryManager should detect that the context is stopped and abort creating new streams.
    /// Expected behavior: Should fail to build RetryManager with "Context is stopped or killed" error.
    #[tokio::test]
    async fn test_retry_manager_context_stopped_before_stream() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::Success,
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));

        // Stop the context before building RetryManager
        ctx.stop_generating();

        // Should fail to build due to stopped context
        let metrics = Arc::new(Metrics::new());
        let retry_manager_result = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await;

        assert!(retry_manager_result.is_err());
        if let Err(error) = retry_manager_result {
            assert!(
                error
                    .to_string()
                    .contains(&format!("Context id {} is stopped or killed", context_id))
            );
            // Verify the error is a typed DynamoError with Cancelled type
            let dynamo_err = error
                .downcast_ref::<DynamoError>()
                .expect("Error should be a DynamoError");
            assert_eq!(
                dynamo_err.error_type(),
                ErrorType::Cancelled,
                "Stopped/killed context should produce a Cancelled error"
            );
        }

        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 0);
    }

    /// Test case 8: No migration for guided-decoding (structured-output) requests
    ///
    /// Bug (#7634): When a worker crashes mid-stream during a structured-output
    /// (json_schema) request, migration appends already-generated token IDs back onto
    /// token_ids and replays the request to a new worker. However, backends initialize
    /// the guided-decoding FSM fresh for every new request and only advance it on newly-
    /// generated tokens — not on context/prompt tokens. This causes the FSM to restart
    /// from the schema root while treating already-generated tokens as context, producing
    /// duplicated or nested JSON in the final response.
    ///
    /// Fix: Disable migration for structured-output requests by zeroing retries_left in
    /// RetryManager::build() when guided_decoding is set, propagating the error cleanly.
    ///
    /// Expected behavior BEFORE fix: All 10 responses received (migration happened — wrong)
    /// Expected behavior AFTER fix: 3 successful + 1 error (migration blocked — correct)
    #[tokio::test]
    async fn test_retry_manager_no_migration_for_guided_decoding() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        let mut request = create_mock_request(10);
        // Set guided decoding (json_schema structured output) on the request
        request.sampling_options.guided_decoding = Some(GuidedDecodingOptions::new(
            Some(serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}})),
            None,
            None,
            None,
            None,
            None,
            None,
        ));

        // MidStreamFail after 3 tokens: without the fix, migration would succeed and
        // deliver all 10 responses; with the fix, migration is blocked and an error
        // is returned after the 3 partial responses.
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3, // migration_limit=3 — should be ignored for guided-decoding requests
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Must receive 3 successful tokens + 1 Disconnected error, NOT all 10.
        // Before the fix this assertion fails because migration proceeds and returns 10.
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error response (migration must be blocked for \
             guided-decoding), but got {} responses",
            responses.len()
        );

        // First 3 responses should be successful
        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(
                response.err().is_none(),
                "Response {} should be successful",
                i
            );
        }

        // Last response must be the stream-disconnection error
        let last = responses.last().unwrap();
        let err = last
            .err()
            .expect("Last response should be a Disconnected error");
        assert_eq!(
            err.error_type(),
            ErrorType::Disconnected,
            "Error type should be Disconnected"
        );
    }

    /// Test case 9: max_seq_len exceeded limit + 1 disables migration
    ///
    /// Boundary test: prompt has 3 tokens, max_seq_len = 5. After 2 generated tokens the
    /// total is 5 (== max_seq_len) — still migratable. The 3rd generated token would push
    /// the total to 6 (> max_seq_len), which disables migration and stops caching.
    /// The failure is placed right at that point (fail_after: 3) so we see the error
    /// propagated instead of retried.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_exceeded() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 5.
        // Token 101 → total 4 ≤ 5: tracked.
        // Token 102 → total 5 ≤ 5: tracked.
        // Token 103 → would-be 6 > 5: NOT tracked, migration disabled.
        // Error follows immediately (fail_after: 3) → not retried.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Some(5), // prompt(3) + 3 generated = 6 > 5 → disables migration
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 3 successful tokens + 1 Disconnected error (migration disabled at token 103).
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error (migration disabled by max_seq_len)"
        );

        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none(), "Response {} should be OK", i);
        }

        let err = responses[3]
            .err()
            .expect("Last response should be Disconnected error");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        // Migration was attempted but blocked because max_seq_len set retries_left to 0.
        // The ongoing metric is still incremented (it counts attempts, not successes).
        assert_eq!(metrics.get_migration_new_request_count(TEST_MODEL), 0);
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
        // max_seq_len limit was triggered once (at token 103).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
    }

    /// Test case 10: Migration succeeds when sequence length is at max_seq_len
    ///
    /// Boundary test: prompt has 3 tokens, max_seq_len = 5. After 2 generated tokens
    /// the total is exactly 5 (== max_seq_len). The failure occurs at that point
    /// (fail_after: 2). Because we use strict inequality (> not >=), the request is
    /// still migratable and the retry succeeds.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_at_limit() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 5.
        // Token 101 → total 4 ≤ 5: tracked.
        // Token 102 → total 5 == 5: tracked (still migratable — strict >).
        // Error (fail_after: 2) → migration succeeds, retry delivers remaining tokens.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 2 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Some(5), // prompt(3) + 2 generated = 5 == max_seq_len → still migratable
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // Migration succeeds — all 10 responses delivered
        assert_eq!(responses.len(), 10);
        for response in &responses {
            assert!(response.err().is_none());
        }

        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);

        // Tracked token_ids must equal exactly max_seq_len (5).
        // The 2 tokens from the first stream were tracked (prompt 3 + gen 2 = 5).
        // After migration the retry stream delivers remaining tokens, but the first
        // new token would push to 6 > 5, so tracking stops and no more are appended.
        assert_eq!(
            retry_manager.request.token_ids.len(),
            5,
            "tracked token_ids should be exactly max_seq_len"
        );

        // The limit was triggered once (first token of the retry stream exceeded 5).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
    }

    /// Test case 11: Prompt length alone exceeds max_seq_len
    ///
    /// When the prompt tokens already exceed max_seq_len, migration is disabled
    /// in RetryManager::build before any tokens are generated. A mid-stream
    /// failure should propagate the error without attempting migration.
    #[tokio::test]
    async fn test_retry_manager_max_seq_len_exceeded_by_prompt() {
        dynamo_runtime::logging::init();

        let context_id = uuid::Uuid::new_v4().to_string();
        // Prompt = [1, 2, 3] (len 3). max_seq_len = 2, so prompt alone exceeds the limit.
        let request = create_mock_request(10);
        let mock_engine = Arc::new(MockEngine::new(
            MockBehavior::MidStreamFail { fail_after: 3 },
            10,
            100,
            context_id.clone(),
        ));
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<BackendOutput>> =
            mock_engine;

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            3,
            Some(2), // prompt(3) > max_seq_len(2) → migration disabled at build time
            Arc::new(TEST_MODEL.to_string()),
            metrics.clone(),
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(response) = retry_manager.next().await {
            responses.push(response);
        }

        // 3 successful tokens + 1 Disconnected error (migration disabled from the start).
        assert_eq!(
            responses.len(),
            4,
            "Expected 3 successful + 1 error (migration disabled by prompt exceeding max_seq_len)"
        );

        for (i, response) in responses[0..3].iter().enumerate() {
            assert!(response.err().is_none(), "Response {} should be OK", i);
        }

        let err = responses[3]
            .err()
            .expect("Last response should be Disconnected error");
        assert_eq!(err.error_type(), ErrorType::Disconnected);

        // max_seq_len was exceeded at build time (prompt len 3 > 2).
        assert_eq!(
            metrics.get_migration_max_seq_len_exceeded_count(TEST_MODEL),
            1
        );
        // Migration was attempted but blocked (retries_left was already 0).
        assert_eq!(metrics.get_migration_ongoing_request_count(TEST_MODEL), 1);
    }

    /// Smoke test for the byo-preprocessor response shape.
    #[tokio::test]
    async fn test_retry_manager_generic_over_llm_engine_output() {
        dynamo_runtime::logging::init();
        let context_id = uuid::Uuid::new_v4().to_string();

        struct LlmEngineMock(String);
        #[async_trait]
        impl
            AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<LLMEngineOutput>>,
                anyhow::Error,
            > for LlmEngineMock
        {
            async fn generate(
                &self,
                _request: SingleIn<PreprocessedRequest>,
            ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
                let responses = stream::iter((0..3u32).map(|i| {
                    Annotated::from_data(LLMEngineOutput {
                        token_ids: vec![200 + i],
                        ..Default::default()
                    })
                }));
                let ctx = Arc::new(Controller::new(self.0.clone()));
                Ok(ResponseStream::new(Box::pin(responses), ctx))
            }
        }

        let request = create_mock_request(3);
        let next_generate: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(LlmEngineMock(context_id.clone()));

        let ctx = Arc::new(Controller::new(context_id.clone()));
        let metrics = Arc::new(Metrics::new());
        let mut retry_manager = RetryManager::build(
            ctx,
            request,
            next_generate,
            1,
            None,
            Arc::new(TEST_MODEL.to_string()),
            metrics,
        )
        .await
        .expect("Failed to build RetryManager");

        let mut responses = Vec::new();
        while let Some(r) = retry_manager.next().await {
            responses.push(r);
        }
        assert_eq!(responses.len(), 3);
        assert_eq!(
            retry_manager.request.token_ids,
            vec![1, 2, 3, 200, 201, 202]
        );
    }
}
