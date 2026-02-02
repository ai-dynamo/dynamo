// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod api;
mod conversation;
mod impls;
mod response;
mod sdk;
mod stream;

pub use api::*;
pub use conversation::*;
pub use response::*;
pub use stream::*;

// Re-export shared types used by responses
pub use crate::types::shared::ComparisonFilter;
pub use crate::types::shared::ComparisonType;
pub use crate::types::shared::CompoundFilter;
pub use crate::types::shared::CompoundType;
pub use crate::types::shared::CustomGrammarFormatParam;
pub use crate::types::shared::Filter;
pub use crate::types::shared::GrammarSyntax;
pub use crate::types::shared::InputTokenDetails;
pub use crate::types::shared::OutputTokenDetails;
pub use crate::types::shared::ResponseUsage;

// Re-export types from parent module that response.rs imports via `crate::types::responses::`
pub use crate::types::ImageDetail;
pub use crate::types::ReasoningEffort;
pub use crate::types::ResponseFormatJsonSchema;

/// Stream of response events
pub type ResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<ResponseStreamEvent, crate::error::OpenAIError>> + Send>,
>;

// Backward-compatible type aliases for Dynamo consumer code migration.
// These map old Dynamo type names to the upstream names.
// TODO: Remove these once all consumer code is fully migrated.
pub type Input = InputParam;
pub type PromptConfig = Prompt;
pub type TextConfig = ResponseTextParam;
pub type TextResponseFormat = TextResponseFormatConfiguration;
