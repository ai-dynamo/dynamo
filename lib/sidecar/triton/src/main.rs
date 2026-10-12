// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> anyhow::Result<()> {
    dynamo_sidecar_common::run_task(dynamo_triton_sidecar::from_cli()?)
}
