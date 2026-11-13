// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

dynamo_allocator::jemalloc_global_allocator!();

fn main() -> anyhow::Result<()> {
    dynamo_ext_proc::build_runtime()?.block_on(dynamo_ext_proc::run(None))
}
