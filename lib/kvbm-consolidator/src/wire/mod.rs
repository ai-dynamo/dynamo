// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire formats.
//!
//! - [`vllm_in`]: re-exports of the vLLM / TRT-LLM input wire types from `dynamo-kv-router`.
//! - [`router_out`]: egress format emitted to downstream kv-router subscribers.

pub mod router_out;
pub mod vllm_in;
