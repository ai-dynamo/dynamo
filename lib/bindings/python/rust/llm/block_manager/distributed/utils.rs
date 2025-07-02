// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn get_barrier_id() -> String {
    std::env::var("DYNAMO_KVBM_BARRIER_ID").unwrap_or("kvbm".to_string())
}
