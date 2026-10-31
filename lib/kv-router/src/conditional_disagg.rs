// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conditional-disagg bypass policy.
//!
//! Decides whether a request should skip remote prefill and run prefill
//! locally on the chosen decode worker. The trait operates over summary
//! signals so future policies can plug in without changing the router call site.
//!
//! Lives in the worker-selection stage; re-exported here for every existing path.

pub use dynamo_worker_selection::conditional_disagg::*;
