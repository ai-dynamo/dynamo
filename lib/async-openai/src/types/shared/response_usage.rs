use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputTokenDetails {
    /// The number of tokens that were retrieved from the cache.
    pub cached_tokens: u32,
}

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct OutputTokenDetails {
    /// The number of reasoning tokens.
    pub reasoning_tokens: u32,
}

/// Usage statistics for a response.
#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ResponseUsage {
    /// The number of input tokens.
    pub input_tokens: u32,
    /// A detailed breakdown of the input tokens.
    pub input_tokens_details: InputTokenDetails,
    /// The number of output tokens.
    pub output_tokens: u32,
    /// A detailed breakdown of the output tokens.
    pub output_tokens_details: OutputTokenDetails,
    /// The total number of tokens used.
    pub total_tokens: u32,
}
