// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::error::OpenAIError;

use super::StaticChunkingStrategy;

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "CreateVectorStoreRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateVectorStoreRequest {
    /// A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that the vector store should use. Useful for tools like `file_search` that can access files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
    /// The name of the vector store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The expiration policy for a vector store.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<VectorStoreExpirationAfter>,

    /// The chunking strategy used to chunk the file(s). If not set, will use the `auto` strategy. Only applicable if `file_ids` is non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_strategy: Option<VectorStoreChunkingStrategy>,

    /// Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maximum of 512 characters long.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum VectorStoreChunkingStrategy {
    /// The default strategy. This strategy currently uses a `max_chunk_size_tokens` of `800` and `chunk_overlap_tokens` of `400`.
    #[default]
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "static")]
    Static {
        #[serde(rename = "static")]
        config: StaticChunkingStrategy,
    },
}

/// Vector store expiration policy
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct VectorStoreExpirationAfter {
    /// Anchor timestamp after which the expiration policy applies. Supported anchors: `last_active_at`.
    pub anchor: String,
    /// The number of days after the anchor time that the vector store will expire.
    pub days: u16, // min: 1, max: 365
}

/// A vector store is a collection of processed files can be used by the `file_search` tool.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreObject {
    /// The identifier, which can be referenced in API endpoints.
    pub id: String,
    /// The object type, which is always `vector_store`.
    pub object: String,
    /// The Unix timestamp (in seconds) for when the vector store was created.
    pub created_at: u32,
    /// The name of the vector store.
    pub name: Option<String>,
    /// The total number of bytes used by the files in the vector store.
    pub usage_bytes: u64,
    pub file_counts: VectorStoreFileCounts,
    /// The status of the vector store, which can be either `expired`, `in_progress`, or `completed`. A status of `completed` indicates that the vector store is ready for use.
    pub status: VectorStoreStatus,
    pub expires_after: Option<VectorStoreExpirationAfter>,
    /// The Unix timestamp (in seconds) for when the vector store will expire.
    pub expires_at: Option<u32>,
    /// The Unix timestamp (in seconds) for when the vector store was last active.
    pub last_active_at: Option<u32>,

    /// Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maximum of 512 characters long.
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStoreStatus {
    Expired,
    InProgress,
    Completed,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileCounts {
    /// The number of files that are currently being processed.
    pub in_progress: u32,
    /// The number of files that have been successfully processed.
    pub completed: u32,
    /// The number of files that have failed to process.
    pub failed: u32,
    /// The number of files that were cancelled.
    pub cancelled: u32,
    /// The total number of files.
    pub total: u32,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ListVectorStoresResponse {
    pub object: String,
    pub data: Vec<VectorStoreObject>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct DeleteVectorStoreResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "UpdateVectorStoreRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct UpdateVectorStoreRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<VectorStoreExpirationAfter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ListVectorStoreFilesResponse {
    pub object: String,
    pub data: Vec<VectorStoreFileObject>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileObject {
    /// The identifier, which can be referenced in API endpoints.
    pub id: String,
    /// The object type, which is always `vector_store.file`.
    pub object: String,
    /// The total vector store usage in bytes. Note that this may be different from the original file size.
    pub usage_bytes: u64,
    /// The Unix timestamp (in seconds) for when the vector store file was created.
    pub created_at: u32,
    /// The ID of the [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object) that the [File](https://platform.openai.com/docs/api-reference/files) is attached to.
    pub vector_store_id: String,
    /// The status of the vector store file, which can be either `in_progress`, `completed`, `cancelled`, or `failed`. The status `completed` indicates that the vector store file is ready for use.
    pub status: VectorStoreFileStatus,
    /// The last error associated with this vector store file. Will be `null` if there are no errors.
    pub last_error: Option<VectorStoreFileError>,
    /// The strategy used to chunk the file.
    pub chunking_strategy: Option<VectorStoreFileObjectChunkingStrategy>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStoreFileStatus {
    InProgress,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileError {
    pub code: VectorStoreFileErrorCode,
    /// A human-readable description of the error.
    pub message: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStoreFileErrorCode {
    ServerError,
    UnsupportedFile,
    InvalidFile,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum VectorStoreFileObjectChunkingStrategy {
    /// This is returned when the chunking strategy is unknown. Typically, this is because the file was indexed before the `chunking_strategy` concept was introduced in the API.
    Other,
    Static {
        r#static: StaticChunkingStrategy,
    },
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "CreateVectorStoreFileRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateVectorStoreFileRequest {
    /// A [File](https://platform.openai.com/docs/api-reference/files) ID that the vector store should use. Useful for tools like `file_search` that can access files.
    pub file_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_strategy: Option<VectorStoreChunkingStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<HashMap<String, AttributeValue>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct DeleteVectorStoreFileResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Debug, Serialize, Default, Clone, Builder, PartialEq, Deserialize)]
#[builder(name = "CreateVectorStoreFileBatchRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct CreateVectorStoreFileBatchRequest {
    /// A list of [File](https://platform.openai.com/docs/api-reference/files) IDs that the vector store should use. Useful for tools like `file_search` that can access files.
    pub file_ids: Vec<String>, // minItems: 1, maxItems: 500
    pub chunking_strategy: Option<VectorStoreChunkingStrategy>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorStoreFileBatchStatus {
    InProgress,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileBatchCounts {
    /// The number of files that are currently being processed.
    pub in_progress: u32,
    /// The number of files that have been successfully processed.
    pub completed: u32,
    /// The number of files that have failed to process.
    pub failed: u32,
    /// The number of files that were cancelled.
    pub cancelled: u32,
    /// The total number of files.
    pub total: u32,
}

///  A batch of files attached to a vector store.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileBatchObject {
    /// The identifier, which can be referenced in API endpoints.
    pub id: String,
    /// The object type, which is always `vector_store.file_batch`.
    pub object: String,
    /// The Unix timestamp (in seconds) for when the vector store files batch was created.
    pub created_at: u32,
    /// The ID of the [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object) that the [File](https://platform.openai.com/docs/api-reference/files) is attached to.
    pub vector_store_id: String,
    /// The status of the vector store files batch, which can be either `in_progress`, `completed`, `cancelled` or `failed`.
    pub status: VectorStoreFileBatchStatus,
    pub file_counts: VectorStoreFileBatchCounts,
}

/// Represents the parsed content of a vector store file.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileContentResponse {
    /// The object type, which is always `vector_store.file_content.page`
    pub object: String,

    /// Parsed content of the file.
    pub data: Vec<VectorStoreFileContentObject>,

    /// Indicates if there are more content pages to fetch.
    pub has_more: bool,

    /// The token for the next page, if any.
    pub next_page: Option<String>,
}

/// Represents the parsed content of a vector store file.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreFileContentObject {
    /// The content type (currently only `"text"`)
    pub r#type: String,

    /// The text content
    pub text: String,
}

#[derive(Debug, Serialize, Default, Clone, Builder, PartialEq, Deserialize)]
#[builder(name = "VectorStoreSearchRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "OpenAIError"))]
pub struct VectorStoreSearchRequest {
    /// A query string for a search.
    pub query: VectorStoreSearchQuery,

    /// Whether to rewrite the natural language query for vector search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewrite_query: Option<bool>,

    /// The maximum number of results to return. This number should be between 1 and 50 inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<u8>,

    /// A filter to apply based on file attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<VectorStoreSearchFilter>,

    /// Ranking options for search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<RankingOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum VectorStoreSearchQuery {
    /// A single query to search for.
    Text(String),
    /// A list of queries to search for.
    Array(Vec<String>),
}

impl Default for VectorStoreSearchQuery {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for VectorStoreSearchQuery {
    fn from(query: String) -> Self {
        Self::Text(query)
    }
}

impl From<&str> for VectorStoreSearchQuery {
    fn from(query: &str) -> Self {
        Self::Text(query.to_string())
    }
}

impl From<Vec<String>> for VectorStoreSearchQuery {
    fn from(query: Vec<String>) -> Self {
        Self::Array(query)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum VectorStoreSearchFilter {
    Comparison(ComparisonFilter),
    Compound(CompoundFilter),
}

impl From<ComparisonFilter> for VectorStoreSearchFilter {
    fn from(filter: ComparisonFilter) -> Self {
        Self::Comparison(filter)
    }
}

impl From<CompoundFilter> for VectorStoreSearchFilter {
    fn from(filter: CompoundFilter) -> Self {
        Self::Compound(filter)
    }
}

/// A filter used to compare a specified attribute key to a given value using a defined comparison operation.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ComparisonFilter {
    /// Specifies the comparison operator: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`.
    pub r#type: ComparisonType,

    /// The key to compare against the value.
    pub key: String,

    /// The value to compare against the attribute key; supports string, number, or boolean types.
    pub value: AttributeValue,
}

/// Specifies the comparison operator: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ComparisonType {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

/// The value to compare against the attribute key; supports string, number, or boolean types.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Number(i64),
    Boolean(bool),
}

impl From<String> for AttributeValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<i64> for AttributeValue {
    fn from(value: i64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for AttributeValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<&str> for AttributeValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

/// Ranking options for search.
#[derive(Debug, Serialize, Default, Deserialize, Clone, PartialEq)]
pub struct RankingOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<Ranker>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Ranker {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default-2024-11-15")]
    Default20241115,
}

/// Combine multiple filters using `and` or `or`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CompoundFilter {
    /// Type of operation: `and` or `or`.
    pub r#type: CompoundFilterType,

    /// Array of filters to combine. Items can be `ComparisonFilter` or `CompoundFilter`
    pub filters: Vec<VectorStoreSearchFilter>,
}

/// Type of operation: `and` or `or`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CompoundFilterType {
    And,
    Or,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreSearchResultsPage {
    /// The object type, which is always `vector_store.search_results.page`.
    pub object: String,

    /// The query used for this search.
    pub search_query: Vec<String>,

    /// The list of search result items.
    pub data: Vec<VectorStoreSearchResultItem>,

    /// Indicates if there are more results to fetch.
    pub has_more: bool,

    /// The token for the next page, if any.
    pub next_page: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreSearchResultItem {
    /// The ID of the vector store file.
    pub file_id: String,

    /// The name of the vector store file.
    pub filename: String,

    /// The similarity score for the result.
    pub score: f32, // minimum: 0, maximum: 1

    /// Attributes of the vector store file.
    pub attributes: HashMap<String, AttributeValue>,

    /// Content chunks from the file.
    pub content: Vec<VectorStoreSearchResultContentObject>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct VectorStoreSearchResultContentObject {
    /// The type of content
    pub r#type: String,

    /// The text content returned from search.
    pub text: String,
}
