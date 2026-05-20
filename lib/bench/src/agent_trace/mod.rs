// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent-trace conversion utilities used by the `agent_trace_to_mooncake` binary.
//!
//! - [`load`] reads Dynamo agent-trace JSONL/gz into an in-memory
//!   [`load::LoadedAgentTrace`] of request and tool entries.
//! - [`mooncake`] lowers requests to the flat Mooncake row format.
//! - [`agentic`] lowers requests to the agentic Mooncake row format, infers
//!   the workflow DAG, attributes harness tool spans to the LLM row that
//!   consumed them, and produces the convert-time tool summary.

pub mod agentic;
pub mod load;
pub mod mooncake;
