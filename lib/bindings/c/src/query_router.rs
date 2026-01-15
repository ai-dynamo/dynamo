// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C FFI Bindings for QueryRouter
//!
//! This module provides C FFI bindings for query-only routing operations.
//! The actual `QueryRouter` implementation is in `dynamo_llm::kv_router::query_router`.

use std::ffi::CStr;

use libc::c_char;

use dynamo_llm::kv_router::{QueryRouter, RouteQueryResult};
use dynamo_runtime::{DistributedRuntime, Runtime};

/// C-compatible result of a routing query
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct CRouteQueryResult {
    /// Worker ID for prefill phase (only valid if is_disaggregated is true)
    pub prefill_worker_id: u64,
    /// Worker ID for decode phase (always valid)
    pub decode_worker_id: u64,
    /// True if disaggregated mode is active (prefill_worker_id is valid)
    pub is_disaggregated: bool,
}

impl From<RouteQueryResult> for CRouteQueryResult {
    fn from(result: RouteQueryResult) -> Self {
        Self {
            prefill_worker_id: result.prefill_worker_id,
            decode_worker_id: result.decode_worker_id,
            is_disaggregated: result.is_disaggregated,
        }
    }
}

/// Opaque handle for QueryRouter
pub type QueryRouterHandle = *mut QueryRouter;

/// Result codes for C FFI
#[repr(u32)]
pub enum QueryRouterResult {
    Ok = 0,
    ErrInvalidHandle = 1,
    ErrInvalidParam = 2,
    ErrInitFailed = 3,
    ErrQueryFailed = 4,
    ErrDisaggEnforced = 5,
}

/// Create a new QueryRouter
///
/// # Safety
/// - All string parameters must be valid null-terminated C strings
/// - The returned handle must be freed with `query_router_destroy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_create(
    namespace: *const c_char,
    component: *const c_char,
    model_name: *const c_char,
    block_size: u32,
    enforce_disagg: bool,
    out_handle: *mut QueryRouterHandle,
) -> QueryRouterResult {
    if namespace.is_null() || model_name.is_null() || out_handle.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let namespace = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let component = if component.is_null() {
        "backend"
    } else {
        match unsafe { CStr::from_ptr(component) }.to_str() {
            Ok(s) if !s.is_empty() => s,
            _ => "backend",
        }
    };

    let model_name = match unsafe { CStr::from_ptr(model_name) }.to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    // Get or create the runtime
    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to create runtime");
            return QueryRouterResult::ErrInitFailed;
        }
    };

    let result = runtime.secondary().block_on(async {
        let drt = match DistributedRuntime::from_settings(runtime.clone()).await {
            Ok(drt) => drt,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to create distributed runtime");
                return Err(QueryRouterResult::ErrInitFailed);
            }
        };

        match QueryRouter::new(
            &drt,
            namespace,
            component,
            model_name,
            block_size,
            None,
            enforce_disagg,
        )
        .await
        {
            Ok(router) => Ok(Box::into_raw(Box::new(router))),
            Err(e) => {
                tracing::error!(error = ?e, "Failed to create query router");
                Err(QueryRouterResult::ErrInitFailed)
            }
        }
    });

    match result {
        Ok(handle) => {
            unsafe { *out_handle = handle };
            QueryRouterResult::Ok
        }
        Err(code) => code,
    }
}

/// Query optimal worker(s) for a request
///
/// # Safety
/// - `handle` must be a valid QueryRouter handle
/// - `token_ids` must point to `token_count` valid u32 values
/// - `out_result` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_query(
    handle: QueryRouterHandle,
    request_id: *const c_char,
    token_ids: *const u32,
    token_count: usize,
    update_states: bool,
    out_result: *mut CRouteQueryResult,
) -> QueryRouterResult {
    if handle.is_null() || token_ids.is_null() || out_result.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let router = unsafe { &*handle };
    let tokens = unsafe { std::slice::from_raw_parts(token_ids, token_count) };

    let request_id = if request_id.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(request_id) }.to_str().ok()
    };

    // Get runtime to execute async query
    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime
        .secondary()
        .block_on(async { router.query_route(request_id, tokens, update_states).await });

    match result {
        Ok(route_result) => {
            unsafe { *out_result = route_result.into() };
            QueryRouterResult::Ok
        }
        Err(e) => {
            tracing::error!(error = ?e, "Query failed");
            if e.to_string().contains("enforced") {
                QueryRouterResult::ErrDisaggEnforced
            } else {
                QueryRouterResult::ErrQueryFailed
            }
        }
    }
}

/// Check if disaggregated mode is active
///
/// # Safety
/// - `handle` must be a valid QueryRouter handle
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_is_disaggregated(handle: QueryRouterHandle) -> bool {
    if handle.is_null() {
        return false;
    }

    let router = unsafe { &*handle };
    router.is_disaggregated()
}

/// Mark prefill as completed for a request
///
/// # Safety
/// - `handle` must be a valid QueryRouter handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_mark_prefill_complete(
    handle: QueryRouterHandle,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let router = unsafe { &*handle };
    let request_id = match unsafe { CStr::from_ptr(request_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime
        .secondary()
        .block_on(async { router.decode_router().mark_prefill_completed(request_id).await });

    match result {
        Ok(_) => QueryRouterResult::Ok,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to mark prefill complete");
            QueryRouterResult::ErrQueryFailed
        }
    }
}

/// Free a request (release resources)
///
/// # Safety
/// - `handle` must be a valid QueryRouter handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_free_request(
    handle: QueryRouterHandle,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let router = unsafe { &*handle };
    let request_id = match unsafe { CStr::from_ptr(request_id) }.to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime
        .secondary()
        .block_on(async { router.decode_router().free(request_id).await });

    match result {
        Ok(_) => QueryRouterResult::Ok,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to free request");
            QueryRouterResult::ErrQueryFailed
        }
    }
}

/// Destroy a QueryRouter handle
///
/// # Safety
/// - `handle` must be a valid QueryRouter handle or null
/// - After this call, `handle` must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn query_router_destroy(handle: QueryRouterHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
