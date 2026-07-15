// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NvBatchStatus {
    Validating,
    Failed,
    InProgress,
    Finalizing,
    Completed,
    Expired,
    Cancelling,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvBatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvBatchError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvBatchErrors {
    pub object: String,
    pub data: Vec<NvBatchError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvCreateBatchRequest {
    pub input_file_id: String,
    pub endpoint: String,
    pub completion_window: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvBatch {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: NvBatchStatus,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_progress_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finalizing_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expired_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelling_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub errors: Option<NvBatchErrors>,
    pub request_counts: NvBatchRequestCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NvFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_status_serializes_as_openai_style_snake_case() {
        assert_eq!(
            serde_json::to_value(NvBatchStatus::InProgress).unwrap(),
            serde_json::json!("in_progress")
        );
    }

    #[test]
    fn create_batch_request_accepts_first_slice_shape() {
        let request: NvCreateBatchRequest = serde_json::from_value(serde_json::json!({
            "input_file_id": "file-123",
            "endpoint": "/v1/completions",
            "completion_window": "24h",
            "metadata": {
                "campaign": "synthetic-data"
            }
        }))
        .unwrap();

        assert_eq!(request.input_file_id, "file-123");
        assert_eq!(request.endpoint, "/v1/completions");
        assert_eq!(request.completion_window, "24h");
        assert_eq!(
            request.metadata.unwrap().get("campaign").unwrap(),
            "synthetic-data"
        );
    }

    #[test]
    fn batch_errors_serialize_as_openai_list() {
        let errors = NvBatchErrors {
            object: "list".to_string(),
            data: vec![NvBatchError {
                code: "invalid_request".to_string(),
                message: "The request is invalid.".to_string(),
                param: Some("body".to_string()),
                line: Some(3),
            }],
        };

        assert_eq!(
            serde_json::to_value(errors).unwrap(),
            serde_json::json!({
                "object": "list",
                "data": [{
                    "code": "invalid_request",
                    "message": "The request is invalid.",
                    "param": "body",
                    "line": 3
                }]
            })
        );
    }
}
