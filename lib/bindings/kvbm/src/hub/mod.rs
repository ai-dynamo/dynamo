// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-side pyclasses for the KVBM prefill router.

mod prefill_router;

pub use prefill_router::{CompletionEvent, PrefillRouterHandler};

use pyo3::prelude::*;

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompletionEvent>()?;
    m.add_class::<PrefillRouterHandler>()?;
    Ok(())
}
