// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C FFI Bindings for Query Routing
//!
//! This module provides C FFI bindings for query-only routing operations.
//! It directly manages PrefillRouter and KvRouter handles, delegating routing
//! logic to `PrefillRouter.query_route()`.

use std::ffi::CStr;
use std::sync::Arc;

use libc::c_char;

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::kv_router::{KvRouter, KvRouterConfig, PrefillRouter, RouteQueryResult};
use dynamo_runtime::pipeline::RouterMode;
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

/// Container holding both routers needed for query routing
pub struct RouterHandles {
    prefill_router: Arc<PrefillRouter>,
    decode_router: Arc<KvRouter>,
}

/// Opaque handle for the router pair
pub type RouterHandlesPtr = *mut RouterHandles;

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

/// Create router handles for query-only routing
///
/// # Safety
/// - All string parameters must be valid null-terminated C strings
/// - The returned handle must be freed with `router_handles_destroy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_create(
    namespace: *const c_char,
    component: *const c_char,
    model_name: *const c_char,
    block_size: u32,
    enforce_disagg: bool,
    out_handle: *mut RouterHandlesPtr,
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

    let model_name_str = match unsafe { CStr::from_ptr(model_name) }.to_str() {
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

        let kv_router_config = KvRouterConfig::default();

        // Get component and endpoint
        let component_handle = match drt.namespace(namespace) {
            Ok(ns) => match ns.component(component) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(error = ?e, "Failed to get component");
                    return Err(QueryRouterResult::ErrInitFailed);
                }
            },
            Err(e) => {
                tracing::error!(error = ?e, "Failed to get namespace");
                return Err(QueryRouterResult::ErrInitFailed);
            }
        };
        let endpoint = component_handle.endpoint("generate");

        let model_manager = Arc::new(ModelManager::new());

        // Create decode router
        let decode_router = match model_manager
            .kv_chooser_for(&endpoint, block_size, Some(kv_router_config.clone()))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to create decode router");
                return Err(QueryRouterResult::ErrInitFailed);
            }
        };

        // Create PrefillRouter (auto-activates when prefill workers discovered)
        let prefill_router = model_manager
            .register_prefill_router(model_name_str.to_string())
            .map(|rx| {
                let mut prefill_config = kv_router_config.clone();
                prefill_config.router_track_active_blocks = false;

                PrefillRouter::new(
                    rx,
                    model_manager.clone(),
                    RouterMode::KV,
                    block_size,
                    Some(prefill_config),
                    enforce_disagg,
                )
            })
            .unwrap_or_else(|| {
                PrefillRouter::disabled(model_manager.clone(), RouterMode::KV, enforce_disagg)
            });

        let handles = RouterHandles {
            prefill_router,
            decode_router,
        };

        Ok(Box::into_raw(Box::new(handles)))
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
/// - `handle` must be a valid RouterHandles handle
/// - `token_ids` must point to `token_count` valid u32 values
/// - `out_result` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_query(
    handle: RouterHandlesPtr,
    token_ids: *const u32,
    token_count: usize,
    update_states: bool,
    out_result: *mut CRouteQueryResult,
) -> QueryRouterResult {
    if handle.is_null() || token_ids.is_null() || out_result.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
    let tokens = unsafe { std::slice::from_raw_parts(token_ids, token_count) };

    // Get runtime to execute async query
    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime.secondary().block_on(async {
        handles
            .prefill_router
            .query_route(&handles.decode_router, tokens, update_states)
            .await
    });

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
/// - `handle` must be a valid RouterHandles handle
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_is_disaggregated(handle: RouterHandlesPtr) -> bool {
    if handle.is_null() {
        return false;
    }

    let handles = unsafe { &*handle };
    handles.prefill_router.is_activated()
}

/// Mark prefill as completed for a request
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_mark_prefill_complete(
    handle: RouterHandlesPtr,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
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
        .block_on(async { handles.decode_router.mark_prefill_completed(request_id).await });

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
/// - `handle` must be a valid RouterHandles handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_free_request(
    handle: RouterHandlesPtr,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
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
        .block_on(async { handles.decode_router.free(request_id).await });

    match result {
        Ok(_) => QueryRouterResult::Ok,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to free request");
            QueryRouterResult::ErrQueryFailed
        }
    }
}

/// Destroy router handles
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle or null
/// - After this call, `handle` must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn router_handles_destroy(handle: RouterHandlesPtr) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}
