// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C FFI Bindings for RequestTokenizer
//!
//! This module provides C FFI bindings for tokenization and prompt template application.
//! The actual `RequestTokenizer` implementation is in `dynamo_llm::kv_router::request_tokenizer`.

use std::ffi::{CStr, CString};
use std::ptr;

use libc::c_char;

use dynamo_llm::kv_router::RequestTokenizer;
use dynamo_llm::model_card::ModelDeploymentCard;

/// Opaque handle for RequestTokenizer
pub type RequestTokenizerHandle = *mut RequestTokenizer;

/// Result of tokenization (C-compatible)
#[repr(C)]
pub struct CTokenizedResult {
    /// Pointer to token IDs array (caller must free with request_tokenizer_free_tokens)
    pub token_ids: *mut u32,
    /// Number of tokens
    pub token_count: usize,
    /// Formatted prompt string (caller must free with request_tokenizer_free_string)
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
pub enum TokenizerResult {
    Ok = 0,
    ErrInvalidHandle = 1,
    ErrInvalidParam = 2,
    ErrInitFailed = 3,
    ErrTokenizeFailed = 4,
    ErrModelLoadFailed = 5,
}

/// Create a new RequestTokenizer from a model path
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string pointing to a model directory
/// - The returned handle must be freed with `request_tokenizer_destroy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn request_tokenizer_create(
    model_path: *const c_char,
    out_handle: *mut RequestTokenizerHandle,
) -> TokenizerResult {
    if model_path.is_null() || out_handle.is_null() {
        return TokenizerResult::ErrInvalidParam;
    }

    let model_path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return TokenizerResult::ErrInvalidParam,
    };

    // Load model card from path
    let card = match ModelDeploymentCard::load_from_disk(model_path, None) {
        Ok(card) => card,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to load model card from {}", model_path);
            return TokenizerResult::ErrModelLoadFailed;
        }
    };

    // Create tokenizer from card
    match RequestTokenizer::from_card(&card) {
        Ok(tokenizer) => {
            unsafe { *out_handle = Box::into_raw(Box::new(tokenizer)) };
            TokenizerResult::Ok
        }
        Err(e) => {
            tracing::error!(error = ?e, "Failed to create tokenizer");
            TokenizerResult::ErrInitFailed
        }
    }
}

/// Tokenize a chat completion request (JSON format)
///
/// # Safety
/// - `handle` must be a valid RequestTokenizer handle
/// - `request_json` must be a valid null-terminated C string containing JSON
/// - `out_result` must be a valid pointer
/// - Caller must free the result using `request_tokenizer_free_result`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn request_tokenizer_tokenize_chat(
    handle: RequestTokenizerHandle,
    request_json: *const c_char,
    out_result: *mut CTokenizedResult,
) -> TokenizerResult {
    if handle.is_null() || request_json.is_null() || out_result.is_null() {
        return TokenizerResult::ErrInvalidParam;
    }

    let tokenizer = unsafe { &*handle };
    let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return TokenizerResult::ErrInvalidParam,
    };

    // Parse JSON to NvCreateChatCompletionRequest
    let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
        match serde_json::from_str(json_str) {
            Ok(req) => req,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to parse request JSON");
                return TokenizerResult::ErrInvalidParam;
            }
        };

    // Tokenize
    match tokenizer.tokenize_chat_request(&request) {
        Ok(result) => {
            // Allocate and copy token IDs
            let mut tokens = result.token_ids.into_boxed_slice();
            let token_ptr = tokens.as_mut_ptr();
            let token_count = tokens.len();
            std::mem::forget(tokens); // Prevent deallocation - caller will free

            // Allocate and copy formatted prompt
            let prompt_cstring = match CString::new(result.formatted_prompt) {
                Ok(s) => s,
                Err(_) => {
                    // Free tokens if prompt conversion fails
                    drop(unsafe { Box::from_raw(std::slice::from_raw_parts_mut(token_ptr, token_count)) });
                    return TokenizerResult::ErrTokenizeFailed;
                }
            };

            unsafe {
                *out_result = CTokenizedResult {
                    token_ids: token_ptr,
                    token_count,
                    formatted_prompt: prompt_cstring.into_raw(),
                };
            }
            TokenizerResult::Ok
        }
        Err(e) => {
            tracing::error!(error = ?e, "Failed to tokenize request");
            TokenizerResult::ErrTokenizeFailed
        }
    }
}

/// Tokenize a raw prompt string (for completions API)
///
/// # Safety
/// - `handle` must be a valid RequestTokenizer handle
/// - `prompt` must be a valid null-terminated C string
/// - `out_result` must be a valid pointer
/// - Caller must free the result using `request_tokenizer_free_result`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn request_tokenizer_tokenize_raw(
    handle: RequestTokenizerHandle,
    prompt: *const c_char,
    out_result: *mut CTokenizedResult,
) -> TokenizerResult {
    if handle.is_null() || prompt.is_null() || out_result.is_null() {
        return TokenizerResult::ErrInvalidParam;
    }

    let tokenizer = unsafe { &*handle };
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(_) => return TokenizerResult::ErrInvalidParam,
    };

    match tokenizer.tokenize_raw(prompt_str) {
        Ok(result) => {
            // Allocate and copy token IDs
            let mut tokens = result.token_ids.into_boxed_slice();
            let token_ptr = tokens.as_mut_ptr();
            let token_count = tokens.len();
            std::mem::forget(tokens);

            // Allocate and copy formatted prompt
            let prompt_cstring = match CString::new(result.formatted_prompt) {
                Ok(s) => s,
                Err(_) => {
                    drop(unsafe { Box::from_raw(std::slice::from_raw_parts_mut(token_ptr, token_count)) });
                    return TokenizerResult::ErrTokenizeFailed;
                }
            };

            unsafe {
                *out_result = CTokenizedResult {
                    token_ids: token_ptr,
                    token_count,
                    formatted_prompt: prompt_cstring.into_raw(),
                };
            }
            TokenizerResult::Ok
        }
        Err(e) => {
            tracing::error!(error = ?e, "Failed to tokenize prompt");
            TokenizerResult::ErrTokenizeFailed
        }
    }
}

/// Free a tokenized result
///
/// # Safety
/// - `result` must be a valid pointer to a CTokenizedResult previously returned by tokenize functions
#[unsafe(no_mangle)]
pub unsafe extern "C" fn request_tokenizer_free_result(result: *mut CTokenizedResult) {
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

/// Destroy a RequestTokenizer handle
///
/// # Safety
/// - `handle` must be a valid RequestTokenizer handle or null
/// - After this call, `handle` must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn request_tokenizer_destroy(handle: RequestTokenizerHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
