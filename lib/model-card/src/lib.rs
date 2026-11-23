// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Dynamo Model Card
//!
//! The model vocabulary every Dynamo component shares: the model deployment card, the
//! runtime configuration a worker publishes, the local model a process registers, and
//! the configuration those carry. Request handling lives in `dynamo-llm`, which depends
//! on this crate.

pub mod checked_file;
pub mod endpoint_type;
pub mod frontend_config;
pub mod hub;
pub mod json;
pub mod local_model;
pub mod media_config;
pub mod model_card;
pub mod model_type;
pub mod parser_family;
pub mod reasoning_field;
pub mod request_template;
pub mod router_config;
pub mod tensor_model;
