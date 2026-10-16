// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The empty catalog used when a custom image does not replace Dynamo's
//! `/generate` format catalog slot.

use dynamo_llm::generate_format::Registry;

/// Register the formats linked into this image.
///
/// Custom catalogs replace this crate under the same dependency name in the
/// frontend extension and in workers that read native responses. Dynamo's own
/// formats are registered first, so a catalog cannot shadow one.
pub fn register(_registry: &mut Registry) -> anyhow::Result<()> {
    Ok(())
}
