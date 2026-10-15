// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native protocol-contract tests, discovered by Cargo as the `protocols` target.
//!
//! Run: `cargo test -p dynamo-llm --no-default-features --test protocols`.

mod openapi_request_fidelity;
mod openapi_request_schema;
mod openapi_response_schema;
