// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-based custom preprocessor implementations
//!
//! This module implements the CustomPreprocessorFactory trait using Python functions
//! for formatting and tokenization. It bridges the gap between the Python-free core
//! library and Python-based custom implementations.

use super::*;
use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use pyo3::{PyObject, PyResult, Python, pyfunction};
use pyo3::types::PyDict;
use parking_lot::RwLock;

use llm_rs::preprocessor::custom::CustomPreprocessorFactory;
use llm_rs::preprocessor::prompt::{OAIPromptFormatter, OAIChatLikeRequest};
use llm_rs::tokenizers::{Encoding, traits::Tokenizer};

/// Global registry for Python preprocessor functions
static PYTHON_PREPROCESSOR_REGISTRY: once_cell::sync::Lazy<Arc<RwLock<PythonPreprocessorRegistry>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(PythonPreprocessorRegistry::new())));

/// Registry for managing Python preprocessor functions
#[derive(Debug, Default)]
pub struct PythonPreprocessorRegistry {
    formatters: HashMap<String, PyObject>,
    tokenizers: HashMap<String, PyObject>,
}

impl PythonPreprocessorRegistry {
    pub fn new() -> Self {
        Self {
            formatters: HashMap::new(),
            tokenizers: HashMap::new(),
        }
    }

    pub fn register_formatter(&mut self, name: String, formatter: PyObject) {
        self.formatters.insert(name, formatter);
    }

    pub fn register_tokenizer(&mut self, name: String, tokenizer: PyObject) {
        self.tokenizers.insert(name, tokenizer);
    }

    pub fn get_formatter(&self, name: &str) -> Option<PyObject> {
        self.formatters.get(name).cloned()
    }

    pub fn get_tokenizer(&self, name: &str) -> Option<PyObject> {
        self.tokenizers.get(name).cloned()
    }

    pub fn has_formatter(&self, name: &str) -> bool {
        self.formatters.contains_key(name)
    }

    pub fn has_tokenizer(&self, name: &str) -> bool {
        self.tokenizers.contains_key(name)
    }
}

/// Python-based prompt formatter implementation
pub struct PythonPromptFormatter {
    formatter_fn: PyObject,
    name: String,
}

impl PythonPromptFormatter {
    pub fn new(name: String, formatter_fn: PyObject) -> Self {
        Self { formatter_fn, name }
    }
}

impl OAIPromptFormatter for PythonPromptFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        // For now, assume Python formatters support this
        // Could be made configurable in the future
        true
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        Python::with_gil(|py| {
            // Convert the request to a Python dict
            let request_dict = PyDict::new(py);
            request_dict.set_item("model", req.model())?;

            // Convert minijinja::Value to string for Python consumption
            let messages_str = req.messages().to_string();
            request_dict.set_item("messages", messages_str)?;

            if let Some(tools) = req.tools() {
                request_dict.set_item("tools", tools.to_string())?;
            }

            if let Some(tool_choice) = req.tool_choice() {
                request_dict.set_item("tool_choice", tool_choice.to_string())?;
            }

            request_dict.set_item("add_generation_prompt", req.should_add_generation_prompt())?;

            if let Some(chat_template_args) = req.chat_template_args() {
                // Convert HashMap<String, serde_json::Value> to Python dict
                let py_dict = PyDict::new(py);
                for (k, v) in chat_template_args.iter() {
                    py_dict.set_item(k, v.to_string())?;
                }
                request_dict.set_item("chat_template_args", py_dict)?;
            }

            // For now, assume all Python functions are synchronous to keep it simple
            // TODO: Add async support in the future if needed
            let result = self.formatter_fn.call1(py, (request_dict,))?;

            let formatted: String = result.extract(py)?;
            Ok(formatted)
        }).map_err(|e: PyErr| anyhow::anyhow!("Python formatter '{}' failed: {}", self.name, e))
    }
}

/// Python-based tokenizer implementation
pub struct PythonTokenizer {
    tokenizer_obj: PyObject,
    name: String,
}

impl PythonTokenizer {
    pub fn new(name: String, tokenizer_obj: PyObject) -> Self {
        Self { tokenizer_obj, name }
    }
}

impl llm_rs::tokenizers::traits::Encoder for PythonTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        Python::with_gil(|py| {
            // Call the encode method - assume synchronous for simplicity
            let result = self.tokenizer_obj.call_method1(py, "encode", (input,))?;
            let token_ids: Vec<u32> = result.extract(py)?;
            // Return as Sp (SentencePiece) encoding since we don't have HF tokenizer object
            Ok(Encoding::Sp(token_ids))
        }).map_err(|e: PyErr| anyhow::anyhow!("Python tokenizer '{}' encode failed: {}", self.name, e))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // Try batch encode first, fall back to individual encodes
        let batch_result = Python::with_gil(|py| -> Result<Vec<Vec<u32>>, PyErr> {
            if let Ok(result) = self.tokenizer_obj.call_method1(py, "encode_batch", (inputs,)) {
                let batch_token_ids: Vec<Vec<u32>> = result.extract(py)?;
                Ok(batch_token_ids)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "encode_batch method not available"
                ))
            }
        });

        match batch_result {
            Ok(batch_token_ids) => {
                Ok(batch_token_ids.into_iter().map(Encoding::Sp).collect())
            }
            Err(_) => {
                // Fall back to individual encoding
                let mut results = Vec::new();
                for input in inputs {
                    results.push(self.encode(input)?);
                }
                Ok(results)
            }
        }
    }
}

impl llm_rs::tokenizers::traits::Decoder for PythonTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Python::with_gil(|py| {
            // Call the decode method - assume synchronous for simplicity
            let result = self.tokenizer_obj.call_method1(
                py,
                "decode",
                (token_ids.to_vec(), skip_special_tokens),
            )?;
            let decoded: String = result.extract(py)?;
            Ok(decoded)
        }).map_err(|e: PyErr| anyhow::anyhow!("Python tokenizer '{}' decode failed: {}", self.name, e))
    }
}

impl llm_rs::tokenizers::traits::Tokenizer for PythonTokenizer {}

/// Factory implementation for creating Python-based preprocessors
pub struct PythonPreprocessorFactory;

impl CustomPreprocessorFactory for PythonPreprocessorFactory {
    fn create_formatter(&self, name: &str) -> Result<Option<Arc<dyn OAIPromptFormatter>>> {
        let registry = PYTHON_PREPROCESSOR_REGISTRY.read();
        if let Some(formatter_fn) = registry.get_formatter(name) {
            Ok(Some(Arc::new(PythonPromptFormatter::new(
                name.to_string(),
                formatter_fn,
            ))))
        } else {
            Ok(None)
        }
    }

    fn create_tokenizer(&self, name: &str) -> Result<Option<Arc<dyn Tokenizer>>> {
        let registry = PYTHON_PREPROCESSOR_REGISTRY.read();
        if let Some(tokenizer_obj) = registry.get_tokenizer(name) {
            Ok(Some(Arc::new(PythonTokenizer::new(
                name.to_string(),
                tokenizer_obj,
            ))))
        } else {
            Ok(None)
        }
    }

    fn has_formatter(&self, name: &str) -> bool {
        let registry = PYTHON_PREPROCESSOR_REGISTRY.read();
        registry.has_formatter(name)
    }

    fn has_tokenizer(&self, name: &str) -> bool {
        let registry = PYTHON_PREPROCESSOR_REGISTRY.read();
        registry.has_tokenizer(name)
    }
}

/// Register a Python formatter function
#[pyfunction]
pub fn register_custom_formatter(name: String, formatter: PyObject) -> PyResult<()> {
    let mut registry = PYTHON_PREPROCESSOR_REGISTRY.write();
    registry.register_formatter(name, formatter);
    Ok(())
}

/// Register a Python tokenizer object
#[pyfunction]
pub fn register_custom_tokenizer(name: String, tokenizer: PyObject) -> PyResult<()> {
    let mut registry = PYTHON_PREPROCESSOR_REGISTRY.write();
    registry.register_tokenizer(name, tokenizer);
    Ok(())
}

/// Initialize the Python preprocessor factory (called once at startup)
pub fn init_python_preprocessor_factory() {
    llm_rs::preprocessor::custom::register_factory(Arc::new(PythonPreprocessorFactory));
}