// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

pub mod backend;
pub mod common;
pub mod disagg_router;
pub mod discovery;
pub mod engines;
pub mod gguf;
pub mod http;
pub mod hub;
pub mod key_value_store;
pub mod kv_router;
pub mod local_model;
pub mod mocker;
pub mod model_card;
pub mod model_type;
pub mod preprocessor;
pub mod protocols;
pub mod recorder;
pub mod request_template;
pub mod tokenizers;
pub mod tokens;
pub mod types;

#[cfg(feature = "block-manager")]
pub mod block_manager;
