// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag builder for guided decoding with tool calls.
//!
//! Parsers with a [`StructuralTagBuilder`] can generate xgrammar structural
//! tags that constrain guided decoding to a model-specific tool-call format.
//!
//! # Module layout
//!
//! - [`format`] — typed representations of xgrammar format nodes
//! - [`builder`] — public builder API used by the preprocessor
//! - [`triggered_tags`] — `triggered_tags` format implementation
//! - [`dsml`] — DeepSeek V3.2+ DSML tool-call format implementation

pub mod builder;
pub mod dsml;
pub mod format;
pub mod triggered_tags;

/// Placeholder in structural tag templates that is replaced with the tool
/// function name at request time.
pub const TOOL_NAME_PLACEHOLDER: &str = "{name}";

pub use builder::{StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext};
pub use dsml::DsmlToolCallsConfig;
pub use triggered_tags::TriggeredTagsConfig;

#[cfg(test)]
mod tests;
