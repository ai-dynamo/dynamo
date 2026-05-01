// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quick smoke test for the PyO3 image-token resolver.
//!
//! Run with:
//!   cargo run --release --features image-token-pyo3 \
//!     --example image_token_check -- <model_name_or_path>
//!
//! Defaults to "Qwen/Qwen3-VL-2B-Instruct" if no arg is passed.

#[cfg(not(feature = "image-token-pyo3"))]
fn main() {
    eprintln!("This example requires --features image-token-pyo3");
    std::process::exit(2);
}

#[cfg(feature = "image-token-pyo3")]
fn main() -> anyhow::Result<()> {
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Qwen/Qwen3-VL-2B-Instruct".to_string());

    println!("Resolving image_token for: {}", model);

    let result: PyResult<Option<String>> = Python::with_gil(|py| {
        let transformers = py.import("transformers")?;
        let auto_processor = transformers.getattr("AutoProcessor")?;

        let trust_remote_code = std::env::var("DYN_PYO3_TRUST_REMOTE_CODE")
            .map(|v| v != "0")
            .unwrap_or(true);

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("trust_remote_code", trust_remote_code)?;
        let processor = auto_processor.call_method("from_pretrained", (&model,), Some(&kwargs))?;
        match processor.getattr("image_token") {
            Ok(attr) if !attr.is_none() => {
                let s: String = attr.extract()?;
                Ok(Some(s))
            }
            _ => Ok(None),
        }
    });

    match result {
        Ok(Some(tok)) => {
            println!("✅ image_token = {:?}", tok);
            Ok(())
        }
        Ok(None) => {
            println!("❌ Processor loaded but has no image_token attribute");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ pyo3 error: {}", e);
            std::process::exit(1);
        }
    }
}
