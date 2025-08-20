// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn get_barrier_id_prefix() -> String {
    std::env::var("DYN_KVBM_BARRIER_ID_PREFIX").unwrap_or("kvbm".to_string())
}
