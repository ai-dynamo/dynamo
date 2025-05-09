// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Dynamo LLM
//!
//! The `dynamo.llm` crate is a Rust library that provides a set of traits and types for building
//! distributed LLM inference solutions.

pub mod backend;
pub mod common;
pub mod disagg_router;
pub mod engines;
pub mod gguf;
pub mod http;
pub mod hub;
pub mod key_value_store;
pub mod kv_router;
pub mod model_card;
pub mod model_type;
pub mod preprocessor;
pub mod protocols;
pub mod recorder;
pub mod request_template;
pub mod tokenizers;
pub mod tokens;
pub mod types;

mod local_model;
pub use local_model::LocalModel;

#[cfg(feature = "block-manager")]
pub mod block_manager;
