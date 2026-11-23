// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol types for SGLang-compatible cross-encoder reranking at
//! `POST /v1/rerank`.

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;

pub use super::embeddings::{NvExt, NvExtProvider};

/// Request for the `/v1/rerank` endpoint.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateRerankRequest {
    /// The model to use for reranking.
    pub model: String,

    /// Query to compare with each document.
    pub query: String,

    /// Candidate documents to rank.
    pub documents: Vec<String>,

    /// Maximum number of ranked documents to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,

    /// Include the source document in each result.
    #[serde(default = "default_return_documents")]
    pub return_documents: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

fn default_return_documents() -> bool {
    true
}

impl NvCreateRerankRequest {
    /// Validate the cross-encoder request before dispatching it to a worker.
    pub fn validate_semantics(&self) -> Result<(), &'static str> {
        if self.query.trim().is_empty() {
            return Err("Query cannot be empty or whitespace only");
        }
        if self.documents.is_empty() {
            return Err("Documents cannot be empty");
        }
        if self
            .documents
            .iter()
            .any(|document| document.trim().is_empty())
        {
            return Err("Each document cannot be empty or whitespace only");
        }
        if self.top_n == Some(0) {
            return Err("Parameter 'top_n' must be larger than 0");
        }
        Ok(())
    }
}

/// One result in the SGLang-compatible rerank response array.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RerankResult {
    pub score: f64,
    pub index: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta_info: Option<serde_json::Value>,
}

/// SGLang returns reranking results as a bare JSON array.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(transparent)]
pub struct NvCreateRerankResponse(pub Vec<RerankResult>);

impl NvExtProvider for NvCreateRerankRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

impl AnnotationsProvider for NvCreateRerankRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .is_some_and(|annotations| annotations.iter().any(|item| item == annotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_defaults_match_sglang() {
        let request: NvCreateRerankRequest = serde_json::from_value(json!({
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "What is Dynamo?",
            "documents": ["A", "B"]
        }))
        .unwrap();

        assert_eq!(request.top_n, None);
        assert!(request.return_documents);
        assert!(request.validate_semantics().is_ok());
    }

    #[test]
    fn invalid_requests_are_rejected() {
        for (query, documents, top_n) in [
            ("", vec!["doc"], None),
            ("query", vec![], None),
            ("query", vec![" "], None),
            ("query", vec!["doc"], Some(0)),
        ] {
            let request = NvCreateRerankRequest {
                model: "model".to_string(),
                query: query.to_string(),
                documents: documents.into_iter().map(str::to_string).collect(),
                top_n,
                return_documents: true,
                nvext: None,
            };
            assert!(request.validate_semantics().is_err());
        }
    }

    #[test]
    fn response_serializes_as_bare_array_and_omits_documents() {
        let response = NvCreateRerankResponse(vec![RerankResult {
            score: 0.9,
            index: 1,
            document: None,
            meta_info: None,
        }]);
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!([{"score": 0.9, "index": 1}])
        );
    }
}
