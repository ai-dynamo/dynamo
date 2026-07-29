// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache data structures for KV block management.

pub mod radix_cache;
pub(crate) mod vllm_block_pool;

pub use radix_cache::RadixCache;
