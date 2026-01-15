// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C FFI Bindings for Request Tokenization
//!
//! This module provides C FFI bindings for tokenization and prompt template application
//! using the existing `OpenAIPreprocessor` from dynamo-llm.

use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

use libc::c_char;

use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;

/// Opaque handle for the preprocessor (uses OpenAIPreprocessor internally)
pub type PreprocessorHandle = *mut Arc<OpenAIPreprocessor>;

/// Result of tokenization (C-compatible)
#[repr(C)]
pub struct CTokenizedResult {
    /// Pointer to token IDs array (caller must free with preprocessor_free_result)
    pub token_ids: *mut u32,
    /// Number of tokens
    pub token_count: usize,
    /// Formatted prompt string (caller must free with preprocessor_free_result)
    pub formatted_prompt: *mut c_char,
}

impl Default for CTokenizedResult {
    fn default() -> Self {
        Self {
            token_ids: ptr::null_mut(),
            token_count: 0,
            formatted_prompt: ptr::null_mut(),
        }
    }
}

/// Result codes for C FFI
#[repr(u32)]
pub enum PreprocessorResult {
    Ok = 0,
    ErrInvalidHandle = 1,
    ErrInvalidParam = 2,
    ErrInitFailed = 3,
    ErrTokenizeFailed = 4,
    ErrModelLoadFailed = 5,
    ErrTemplateFailed = 6,
}

/// Create a new preprocessor from a model path
///
/// Uses the existing OpenAIPreprocessor which handles both template application
/// and tokenization.
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string pointing to a model directory
/// - The returned handle must be freed with `preprocessor_destroy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn preprocessor_create(
    model_path: *const c_char,
    out_handle: *mut PreprocessorHandle,
) -> PreprocessorResult {
    if model_path.is_null() || out_handle.is_null() {
        return PreprocessorResult::ErrInvalidParam;
    }

    let model_path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return PreprocessorResult::ErrInvalidParam,
    };

    // Load model card from path
    let card = match ModelDeploymentCard::load_from_disk(model_path, None) {
        Ok(card) => card,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to load model card from {}", model_path);
            return PreprocessorResult::ErrModelLoadFailed;
        }
    };

    // Create preprocessor from card (uses existing OpenAIPreprocessor::new)
    match OpenAIPreprocessor::new(card) {
        Ok(preprocessor) => {
            unsafe { *out_handle = Box::into_raw(Box::new(preprocessor)) };
            PreprocessorResult::Ok
        }
        Err(e) => {
            tracing::error!(error = ?e, "Failed to create preprocessor");
            PreprocessorResult::ErrInitFailed
        }
    }
}

/// Apply chat template and tokenize a request
///
/// Takes a chat completion request in JSON format, applies the model's chat template,
/// and tokenizes the result.
///
/// # Safety
/// - `handle` must be a valid preprocessor handle
/// - `request_json` must be a valid null-terminated C string containing JSON
/// - `out_result` must be a valid pointer
/// - Caller must free the result using `preprocessor_free_result`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn preprocessor_tokenize_chat(
    handle: PreprocessorHandle,
    request_json: *const c_char,
    out_result: *mut CTokenizedResult,
) -> PreprocessorResult {
    if handle.is_null() || request_json.is_null() || out_result.is_null() {
        return PreprocessorResult::ErrInvalidParam;
    }

    let preprocessor = unsafe { &**handle };
    let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return PreprocessorResult::ErrInvalidParam,
    };

    // Parse JSON to NvCreateChatCompletionRequest
    let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
        match serde_json::from_str(json_str) {
            Ok(req) => req,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to parse request JSON");
                return PreprocessorResult::ErrInvalidParam;
            }
        };

    // Apply chat template using existing OpenAIPreprocessor::apply_template
    let formatted_prompt = match preprocessor.apply_template(&request) {
        Ok(Some(prompt)) => prompt,
        Ok(None) => String::new(),
        Err(e) => {
            tracing::error!(error = ?e, "Failed to apply chat template");
            return PreprocessorResult::ErrTemplateFailed;
        }
    };

    // Tokenize using existing OpenAIPreprocessor::tokenize
    let encoding = match preprocessor.tokenize(&formatted_prompt) {
        Ok(enc) => enc,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to tokenize");
            return PreprocessorResult::ErrTokenizeFailed;
        }
    };

    // Allocate and copy token IDs
    let token_ids = encoding.token_ids().to_vec();
    let mut tokens = token_ids.into_boxed_slice();
    let token_ptr = tokens.as_mut_ptr();
    let token_count = tokens.len();
    std::mem::forget(tokens);

    // Allocate and copy formatted prompt
    let prompt_cstring = match CString::new(formatted_prompt) {
        Ok(s) => s,
        Err(_) => {
            drop(unsafe {
                Box::from_raw(std::slice::from_raw_parts_mut(token_ptr, token_count))
            });
            return PreprocessorResult::ErrTokenizeFailed;
        }
    };

    unsafe {
        *out_result = CTokenizedResult {
            token_ids: token_ptr,
            token_count,
            formatted_prompt: prompt_cstring.into_raw(),
        };
    }
    PreprocessorResult::Ok
}

/// Tokenize a raw prompt string (for completions API)
///
/// Directly tokenizes the prompt without template application.
///
/// # Safety
/// - `handle` must be a valid preprocessor handle
/// - `prompt` must be a valid null-terminated C string
/// - `out_result` must be a valid pointer
/// - Caller must free the result using `preprocessor_free_result`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn preprocessor_tokenize_raw(
    handle: PreprocessorHandle,
    prompt: *const c_char,
    out_result: *mut CTokenizedResult,
) -> PreprocessorResult {
    if handle.is_null() || prompt.is_null() || out_result.is_null() {
        return PreprocessorResult::ErrInvalidParam;
    }

    let preprocessor = unsafe { &**handle };
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(_) => return PreprocessorResult::ErrInvalidParam,
    };

    // Tokenize using existing OpenAIPreprocessor::tokenize
    let encoding = match preprocessor.tokenize(prompt_str) {
        Ok(enc) => enc,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to tokenize");
            return PreprocessorResult::ErrTokenizeFailed;
        }
    };

    // Allocate and copy token IDs
    let token_ids = encoding.token_ids().to_vec();
    let mut tokens = token_ids.into_boxed_slice();
    let token_ptr = tokens.as_mut_ptr();
    let token_count = tokens.len();
    std::mem::forget(tokens);

    // Allocate and copy formatted prompt
    let prompt_cstring = match CString::new(prompt_str) {
        Ok(s) => s,
        Err(_) => {
            drop(unsafe {
                Box::from_raw(std::slice::from_raw_parts_mut(token_ptr, token_count))
            });
            return PreprocessorResult::ErrTokenizeFailed;
        }
    };

    unsafe {
        *out_result = CTokenizedResult {
            token_ids: token_ptr,
            token_count,
            formatted_prompt: prompt_cstring.into_raw(),
        };
    }
    PreprocessorResult::Ok
}

/// Free a tokenized result
///
/// # Safety
/// - `result` must be a valid pointer to a CTokenizedResult previously returned by tokenize functions
#[unsafe(no_mangle)]
pub unsafe extern "C" fn preprocessor_free_result(result: *mut CTokenizedResult) {
    if result.is_null() {
        return;
    }

    let res = unsafe { &mut *result };

    // Free token IDs
    if !res.token_ids.is_null() && res.token_count > 0 {
        drop(unsafe {
            Box::from_raw(std::slice::from_raw_parts_mut(res.token_ids, res.token_count))
        });
        res.token_ids = ptr::null_mut();
        res.token_count = 0;
    }

    // Free formatted prompt
    if !res.formatted_prompt.is_null() {
        drop(unsafe { CString::from_raw(res.formatted_prompt) });
        res.formatted_prompt = ptr::null_mut();
    }
}

/// Destroy a preprocessor handle
///
/// # Safety
/// - `handle` must be a valid preprocessor handle or null
/// - After this call, `handle` must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn preprocessor_destroy(handle: PreprocessorHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
