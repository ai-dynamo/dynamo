// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python tokenizer wrapper for using Python-based tokenizers (SGLang, vLLM, or custom)
//! in Dynamo's Rust preprocessor.

use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyList;

use super::{
    Encoding, Error, Result, TokenIdType,
    traits::{Decoder, Encoder, Tokenizer},
};

/// A tokenizer that delegates to a Python tokenizer implementation.
///
/// This allows using Python-based tokenizers like SGLang's or vLLM's tokenizer
/// within Dynamo's Rust preprocessor, enabling support for models that require
/// specialized tokenizers (e.g., mistral-common for Ministral models).
pub struct PythonTokenizer {
    /// Python module path (e.g., "dynamo.common.tokenizers.sglang")
    module_path: String,
    /// Class name within the module (e.g., "SGLangTokenizer")
    class_name: String,
    /// Model path/name passed to tokenizer constructor
    model_path: String,
    /// Lazily initialized Python tokenizer object
    py_tokenizer: Mutex<Option<Py<PyAny>>>,
}

impl PythonTokenizer {
    /// Create a new PythonTokenizer.
    ///
    /// # Arguments
    /// * `module_path` - Full Python module path (e.g., "dynamo.common.tokenizers.sglang")
    /// * `class_name` - Class name to instantiate (e.g., "SGLangTokenizer")
    /// * `model_path` - Model path/name to pass to the tokenizer constructor
    pub fn new(module_path: String, class_name: String, model_path: String) -> Self {
        Self {
            module_path,
            class_name,
            model_path,
            py_tokenizer: Mutex::new(None),
        }
    }

    /// Create a PythonTokenizer configured for SGLang's tokenizer.
    pub fn sglang(model_path: String) -> Self {
        Self::new(
            "dynamo.common.tokenizers.sglang".to_string(),
            "SGLangTokenizer".to_string(),
            model_path,
        )
    }

    /// Create a PythonTokenizer configured for vLLM's tokenizer.
    pub fn vllm(model_path: String) -> Self {
        Self::new(
            "dynamo.common.tokenizers.vllm".to_string(),
            "VLLMTokenizer".to_string(),
            model_path,
        )
    }

    /// Ensure the tokenizer is initialized and execute a function with it.
    fn with_tokenizer<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Bound<'_, PyAny>) -> Result<T>,
    {
        let mut guard = self.py_tokenizer.lock().map_err(|e| {
            Error::msg(format!("Failed to lock tokenizer mutex: {}", e))
        })?;

        Python::with_gil(|py| {
            // Initialize if needed
            if guard.is_none() {
                let module = py.import(self.module_path.as_str()).map_err(|e| {
                    Error::msg(format!(
                        "Failed to import Python tokenizer module '{}': {}",
                        self.module_path, e
                    ))
                })?;

                let tokenizer_class = module.getattr(self.class_name.as_str()).map_err(|e| {
                    Error::msg(format!(
                        "Failed to get tokenizer class '{}' from module '{}': {}",
                        self.class_name, self.module_path, e
                    ))
                })?;

                let tokenizer = tokenizer_class
                    .call1((self.model_path.as_str(),))
                    .map_err(|e| {
                        Error::msg(format!(
                            "Failed to instantiate tokenizer '{}' with model '{}': {}",
                            self.class_name, self.model_path, e
                        ))
                    })?;

                *guard = Some(tokenizer.unbind());
            }

            // Now use the tokenizer
            let tokenizer_ref = guard.as_ref().unwrap();
            let tokenizer = tokenizer_ref.bind(py);
            f(tokenizer)
        })
    }
}

impl Encoder for PythonTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        self.with_tokenizer(|tokenizer| {
            let result = tokenizer.call_method1("encode", (input,)).map_err(|e| {
                Error::msg(format!("Python tokenizer encode failed: {}", e))
            })?;

            let token_ids: Vec<TokenIdType> = result.extract().map_err(|e| {
                Error::msg(format!(
                    "Failed to extract token IDs from Python tokenizer: {}",
                    e
                ))
            })?;

            Ok(Encoding::Py(token_ids))
        })
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        self.with_tokenizer(|tokenizer| {
            // Convert inputs to a Python list
            let py = tokenizer.py();
            let py_inputs = PyList::new(py, inputs).map_err(|e| {
                Error::msg(format!("Failed to create Python list from inputs: {}", e))
            })?;

            let result = tokenizer
                .call_method1("encode_batch", (py_inputs,))
                .map_err(|e| {
                    Error::msg(format!("Python tokenizer encode_batch failed: {}", e))
                })?;

            let batch_token_ids: Vec<Vec<TokenIdType>> = result.extract().map_err(|e| {
                Error::msg(format!(
                    "Failed to extract batch token IDs from Python tokenizer: {}",
                    e
                ))
            })?;

            Ok(batch_token_ids
                .into_iter()
                .map(Encoding::Py)
                .collect())
        })
    }
}

impl Decoder for PythonTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.with_tokenizer(|tokenizer| {
            let py = tokenizer.py();
            let py_token_ids = PyList::new(py, token_ids).map_err(|e| {
                Error::msg(format!(
                    "Failed to create Python list from token IDs: {}",
                    e
                ))
            })?;

            let result = tokenizer
                .call_method1("decode", (py_token_ids, skip_special_tokens))
                .map_err(|e| Error::msg(format!("Python tokenizer decode failed: {}", e)))?;

            let text: String = result.extract().map_err(|e| {
                Error::msg(format!(
                    "Failed to extract decoded text from Python tokenizer: {}",
                    e
                ))
            })?;

            Ok(text)
        })
    }
}

impl Tokenizer for PythonTokenizer {}

// PythonTokenizer is Send + Sync because:
// 1. All fields are Send + Sync (String, Mutex<Option<Py<PyAny>>>)
// 2. Mutex provides interior mutability with Send + Sync
// 3. We always acquire the GIL before accessing the Python object

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_tokenizer_creation() {
        let tokenizer = PythonTokenizer::new(
            "dynamo.common.tokenizers.sglang".to_string(),
            "SGLangTokenizer".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
        );
        assert_eq!(tokenizer.module_path, "dynamo.common.tokenizers.sglang");
        assert_eq!(tokenizer.class_name, "SGLangTokenizer");
        assert_eq!(tokenizer.model_path, "Qwen/Qwen3-0.6B");
    }

    #[test]
    fn test_sglang_convenience_constructor() {
        let tokenizer = PythonTokenizer::sglang("Qwen/Qwen3-0.6B".to_string());
        assert_eq!(tokenizer.module_path, "dynamo.common.tokenizers.sglang");
        assert_eq!(tokenizer.class_name, "SGLangTokenizer");
    }

    #[test]
    fn test_vllm_convenience_constructor() {
        let tokenizer = PythonTokenizer::vllm("Qwen/Qwen3-0.6B".to_string());
        assert_eq!(tokenizer.module_path, "dynamo.common.tokenizers.vllm");
        assert_eq!(tokenizer.class_name, "VLLMTokenizer");
    }
}
