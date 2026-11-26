// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The static approach, shaped like `lib/router-plugins/catalog`: a custom image swaps this
//! crate for its own and the host calls `register` once at startup.

use dynamo_parsers_v2::UnifiedFamily;
use dynamo_parsers_v2::unified::qwen3::Qwen3;

pub fn register() {
    dynamo_dummy_parser::Dummy::register();
    Qwen3::register();
}
