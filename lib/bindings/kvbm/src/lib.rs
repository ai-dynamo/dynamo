// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use pyo3::{exceptions::PyException, prelude::*};
use std::{fmt::Display, sync::Arc};

#[cfg(feature = "dynamo")]
mod dynamo;

#[cfg(feature = "kernels")]
mod kernels;

#[cfg(feature = "v1")]
mod block_manager;

#[cfg(feature = "v2")]
mod v2;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    #[cfg(feature = "dynamo")]
    dynamo::add_to_module(m)?;

    #[cfg(feature = "v1")]
    block_manager::add_to_module(m)?;

    #[cfg(feature = "v2")]
    {
        let v2 = PyModule::new(m.py(), "v2")?;
        v2::add_to_module(&v2)?;
        m.add_submodule(&v2)?;
    }

    #[cfg(feature = "kernels")]
    {
        let kernels = PyModule::new(m.py(), "kernels")?;
        kernels::add_to_module(&kernels)?;
        m.add_submodule(&kernels)?;
    }

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}
