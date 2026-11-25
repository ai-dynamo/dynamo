// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

/// Dynamo's request wrapper for the OpenAI Batch API.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
pub struct NvCreateBatchRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::BatchRequest,
}

/// Dynamo's response wrapper for an OpenAI Batch object.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
pub struct NvBatch {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::Batch,
}

/// Dynamo's response wrapper for an OpenAI File object.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone, PartialEq)]
pub struct NvFile {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::OpenAIFile,
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_protocols::types::{BatchEndpoint, BatchStatus, OpenAIFilePurpose};

    #[test]
    fn create_batch_request_uses_shared_openai_shape() {
        let request: NvCreateBatchRequest = serde_json::from_value(serde_json::json!({
            "input_file_id": "file-123",
            "endpoint": "/v1/completions",
            "completion_window": "24h",
            "metadata": {
                "campaign": "synthetic-data"
            }
        }))
        .unwrap();

        assert_eq!(request.inner.input_file_id, "file-123");
        assert_eq!(request.inner.endpoint, BatchEndpoint::V1Completions);
        assert_eq!(
            request.inner.metadata.unwrap().get("campaign").unwrap(),
            &serde_json::json!("synthetic-data")
        );
    }

    #[test]
    fn batch_response_uses_shared_openai_shape() {
        let batch: NvBatch = serde_json::from_value(serde_json::json!({
            "id": "batch-123",
            "object": "batch",
            "endpoint": "/v1/completions",
            "input_file_id": "file-123",
            "completion_window": "24h",
            "status": "in_progress",
            "created_at": 1_700_000_000,
            "request_counts": {
                "total": 10,
                "completed": 4,
                "failed": 1
            }
        }))
        .unwrap();

        assert_eq!(batch.inner.status, BatchStatus::InProgress);
        assert_eq!(batch.inner.request_counts.unwrap().total, 10);
    }

    #[test]
    fn file_response_uses_shared_openai_shape() {
        let file: NvFile = serde_json::from_value(serde_json::json!({
            "id": "file-123",
            "object": "file",
            "bytes": 42,
            "created_at": 1_700_000_000,
            "filename": "batch.jsonl",
            "purpose": "batch"
        }))
        .unwrap();

        assert_eq!(file.inner.id, "file-123");
        assert_eq!(file.inner.purpose, OpenAIFilePurpose::Batch);
    }
}
