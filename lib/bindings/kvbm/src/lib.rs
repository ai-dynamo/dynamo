// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[allow(unused_imports)]
use pyo3::exceptions::PyException;
#[allow(unused_imports)]
use pyo3::prelude::*;
use std::fmt::Display;

#[cfg(feature = "dynamo")]
mod dynamo;

// TODO: kernels module needs adaptation - operational_copy API doesn't exist in decomposed kvbm-kernels
// #[cfg(feature = "kernels")]
// mod kernels;

#[cfg(feature = "v1")]
mod v1;
#[cfg(feature = "v1")]
pub use v1::*;

#[cfg(feature = "v2")]
mod v2;

// Re-export runtime helpers so v1 code using `crate::get_current_*` still works
#[cfg(feature = "dynamo")]
pub use dynamo::{
    extract_distributed_runtime_from_obj, get_current_cancel_token, get_current_tokio_handle,
};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    #[cfg(feature = "dynamo")]
    dynamo::add_to_module(m)?;

    #[cfg(feature = "v1")]
    v1::add_to_module(m)?;

    #[cfg(feature = "v2")]
    {
        let v2 = PyModule::new(m.py(), "v2")?;
        v2::add_to_module(&v2)?;
        m.add_submodule(&v2)?;
    }

    // TODO: kernels bindings disabled pending operational_copy API adaptation
    // #[cfg(feature = "kernels")]
    // {
    //     let kernels = PyModule::new(m.py(), "kernels")?;
    //     kernels::add_to_module(&kernels)?;
    //     m.add_submodule(&kernels)?;
    // }

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
