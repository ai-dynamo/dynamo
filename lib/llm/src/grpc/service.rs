// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod kserve;
pub mod openai;
pub mod tensor;

use tonic::Status;

use crate::http::service::error::SanitizedError;
use crate::http::service::metrics::request_was_unavailable;

/// Map a dispatch error that is not a client-visible rejection onto its gRPC status, logging it
/// at the level its classification deserves.
///
/// Worker-scoped and pool-scoped unavailability both mean the request found no
/// servable worker, so both answer `UNAVAILABLE` (14) and let the client retry.
/// Anything else is a server fault and keeps `INTERNAL` (13). The unavailable
/// message is sanitized to match the HTTP frontends; only the internal arm keeps
/// the caller's context string.
pub(crate) fn dispatch_error_status(
    error: &(dyn std::error::Error + 'static),
    internal_context: &str,
) -> Status {
    if request_was_unavailable(error) {
        // Retryable by the client, so degraded rather than actionable.
        tracing::warn!(error = %error, "{internal_context}: no worker available");
        return Status::unavailable(SanitizedError::Unavailable.to_string());
    }
    tracing::error!(error = %error, "{internal_context}");
    Status::internal(format!("{internal_context}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::tensor::NvCreateTensorRequest;
    use crate::types::Annotated;
    use dynamo_runtime::engine::AsyncEngine;
    use dynamo_runtime::error::{DynamoError, ErrorType};
    use dynamo_runtime::pipeline::{Error, ManyOut, SingleIn};
    use std::sync::Arc;
    use tonic::metadata::MetadataMap;

    fn error(error_type: ErrorType) -> anyhow::Error {
        DynamoError::builder()
            .error_type(error_type)
            .message("boom")
            .build()
            .into()
    }

    /// Fails dispatch the way an addressed worker that no longer serves the instance does.
    struct WorkerUnavailableEngine;

    macro_rules! impl_worker_unavailable_engine {
        ($request:ty, $response:ty) => {
            #[async_trait::async_trait]
            impl AsyncEngine<SingleIn<$request>, ManyOut<Annotated<$response>>, Error>
                for WorkerUnavailableEngine
            {
                async fn generate(
                    &self,
                    _request: SingleIn<$request>,
                ) -> Result<ManyOut<Annotated<$response>>, Error> {
                    Err(error(ErrorType::WorkerUnavailable))
                }
            }
        };
    }

    impl_worker_unavailable_engine!(
        NvCreateTensorRequest,
        crate::protocols::tensor::NvCreateTensorResponse
    );
    impl_worker_unavailable_engine!(
        crate::protocols::openai::completions::NvCreateCompletionRequest,
        crate::protocols::openai::completions::NvCreateCompletionResponse
    );

    fn kserve_state() -> Arc<kserve::State> {
        kserve::KserveService::builder()
            .build()
            .expect("kserve service should build")
            .state_clone()
    }

    /// The tensor dispatch path must answer UNAVAILABLE, not INTERNAL. Pins the call site in
    /// `tensor.rs`, which the helper's own unit test cannot reach.
    #[tokio::test]
    async fn tensor_dispatch_maps_worker_unavailable_to_grpc_unavailable() {
        let state = kserve_state();
        state
            .manager()
            .add_tensor_model("test-model", "checksum", Arc::new(WorkerUnavailableEngine))
            .expect("tensor model should register");

        let status = crate::grpc::service::tensor::tensor_response_stream(
            state,
            NvCreateTensorRequest {
                id: None,
                model: "test-model".to_string(),
                tensors: vec![],
                parameters: Default::default(),
                nvext: None,
            },
            false,
            &MetadataMap::new(),
        )
        .await
        .err()
        .expect("dispatch must fail");

        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), "Service temporarily unavailable");
    }

    /// The completions dispatch path must answer UNAVAILABLE, not INTERNAL. Pins the call site
    /// in `openai.rs`, which the helper's own unit test cannot reach.
    #[tokio::test]
    async fn completions_dispatch_maps_worker_unavailable_to_grpc_unavailable() {
        use crate::protocols::openai::completions::NvCreateCompletionRequest;

        let state = kserve_state();
        state
            .manager()
            .add_completions_model("test-model", "checksum", Arc::new(WorkerUnavailableEngine))
            .expect("completions model should register");

        let request = NvCreateCompletionRequest {
            inner: dynamo_protocols::types::CreateCompletionRequest {
                model: "test-model".to_string(),
                prompt: "Hello".into(),
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            metadata: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        };

        let status = crate::grpc::service::openai::completion_response_stream(
            state,
            request,
            &MetadataMap::new(),
        )
        .await
        .err()
        .expect("dispatch must fail");

        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), "Service temporarily unavailable");
    }

    #[test]
    fn unavailable_dispatch_errors_map_to_grpc_unavailable() {
        for error_type in [ErrorType::WorkerUnavailable, ErrorType::Unavailable] {
            let status = dispatch_error_status(error(error_type).as_ref(), "ctx");
            assert_eq!(status.code(), tonic::Code::Unavailable, "{error_type}");
            assert_eq!(status.message(), "Service temporarily unavailable");
        }
    }

    #[test]
    fn other_dispatch_errors_stay_internal_with_context() {
        let status = dispatch_error_status(error(ErrorType::Unknown).as_ref(), "ctx");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(status.message().starts_with("ctx: "));
    }
}
