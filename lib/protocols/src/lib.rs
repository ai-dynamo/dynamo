// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

//! Protocol type definitions for OpenAI-compatible inference APIs.
//!
//! This crate provides types for multiple inference API protocols:
//! - **OpenAI Chat Completions & Completions** (via upstream `async-openai` re-exports + extensions)
//! - **OpenAI Responses API** (via upstream `async-openai` re-exports)
//! - **Anthropic Messages API** (fully custom)
//!
//! Inference-serving extensions (reasoning content, stop reasons, multimodal)
//! are locally defined and documented.
#![allow(deprecated)]
#![allow(warnings)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;
pub mod types;
