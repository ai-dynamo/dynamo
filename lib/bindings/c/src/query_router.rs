// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query Router C Bindings
//!
//! This module provides C FFI bindings for query-only routing operations.
//! It is designed for use with the Inference Gateway EPP (External Pre-Processor)
//! to query optimal worker IDs without sending the actual request.
//!
//! The QueryRouter supports both aggregated and disaggregated modes:
//! - Aggregated: Returns a single decode_worker_id
//! - Disaggregated: Returns prefill_worker_id + decode_worker_id (when prefill workers available)

use std::ffi::CStr;
use std::sync::Arc;

use anyhow::{Context, Result};
use libc::c_char;
use tokio::sync::RwLock;

use dynamo_llm::{
    discovery::ModelManager,
    kv_router::{KvRouter, KvRouterConfig},
};
use dynamo_runtime::{DistributedRuntime, Runtime};

/// Result of a routing query
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct RouteQueryResult {
    /// Worker ID for prefill phase (only valid if is_disaggregated is true)
    pub prefill_worker_id: u64,
    /// Worker ID for decode phase (always valid)
    pub decode_worker_id: u64,
    /// Data parallel rank for the decode worker
    pub dp_rank: u32,
    /// Number of KV cache blocks that overlap with the request
    pub overlap_blocks: u32,
    /// True if disaggregated mode is active (prefill_worker_id is valid)
    pub is_disaggregated: bool,
}

/// Query Router handle for C FFI
///
/// This router only queries for optimal worker IDs without sending requests.
/// It supports both aggregated and disaggregated modes based on prefill worker availability.
pub struct QueryRouter {
    /// KV-aware router for decode workers (also handles aggregated mode)
    decode_router: Arc<KvRouter>,

    /// Optional KV-aware router for prefill workers (None = aggregated mode)
    /// This is lazily activated when prefill workers are discovered
    prefill_router: Arc<RwLock<Option<Arc<KvRouter>>>>,

    /// Whether to enforce disaggregated mode (fail if prefill unavailable)
    enforce_disagg: bool,
}

impl QueryRouter {
    /// Create a new QueryRouter
    ///
    /// The router starts in aggregated mode and automatically switches to
    /// disaggregated mode when prefill workers become available.
    pub async fn new(
        distributed_runtime: &DistributedRuntime,
        namespace: &str,
        component_name: &str,
        model_name: &str,
        block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        enforce_disagg: bool,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();

        let component = distributed_runtime
            .namespace(namespace)?
            .component(component_name)?;
        let endpoint = component.endpoint("generate");

        let model_manager = Arc::new(ModelManager::new());

        // Create KV router for decode workers
        let decode_router = model_manager
            .kv_chooser_for(&endpoint, block_size, Some(kv_router_config.clone()))
            .await
            .context("Failed to create decode router")?;

        // Register for prefill worker discovery
        let prefill_rx = model_manager.register_prefill_router(model_name.to_string());

        let prefill_router: Arc<RwLock<Option<Arc<KvRouter>>>> = Arc::new(RwLock::new(None));

        // Spawn background task to activate prefill router when workers discovered
        if let Some(rx) = prefill_rx {
            let prefill_router_clone = prefill_router.clone();
            let model_manager_clone = model_manager.clone();
            let kv_config = kv_router_config.clone();

            tokio::spawn(async move {
                match rx.await {
                    Ok(endpoint) => {
                        tracing::info!("Prefill workers discovered, activating disaggregated mode");

                        // Create prefill-specific config
                        let mut prefill_config = kv_config;
                        prefill_config.router_track_active_blocks = false;

                        // Create KV router for prefill workers
                        match model_manager_clone
                            .kv_chooser_for(&endpoint, block_size, Some(prefill_config))
                            .await
                        {
                            Ok(new_prefill_router) => {
                                let mut guard = prefill_router_clone.write().await;
                                *guard = Some(new_prefill_router);
                                tracing::info!("Disaggregated mode activated");
                            }
                            Err(e) => {
                                tracing::error!(error = %e, "Failed to create prefill router");
                            }
                        }
                    }
                    Err(_) => {
                        tracing::debug!("Prefill router activation channel closed");
                    }
                }
            });
        }

        Ok(Self {
            decode_router,
            prefill_router,
            enforce_disagg,
        })
    }

    /// Query the optimal worker(s) for a request
    ///
    /// This method does NOT send the request, it only queries for worker IDs.
    /// State is optionally updated based on `update_states` parameter.
    ///
    /// Returns:
    /// - In aggregated mode: decode_worker_id only
    /// - In disaggregated mode: both prefill_worker_id and decode_worker_id
    pub async fn query_route(
        &self,
        request_id: Option<&str>,
        token_ids: &[u32],
        update_states: bool,
    ) -> Result<RouteQueryResult> {
        // Check if disaggregated mode is available
        let prefill_router_guard = self.prefill_router.read().await;
        let is_disaggregated = prefill_router_guard.is_some();

        if self.enforce_disagg && !is_disaggregated {
            anyhow::bail!("Disaggregated mode enforced but no prefill workers available");
        }

        if is_disaggregated {
            let prefill_router = prefill_router_guard.as_ref().unwrap().clone();
            drop(prefill_router_guard); // Release lock before async operations

            // Disaggregated mode: query both prefill and decode routers
            // Query prefill worker
            let (prefill_worker, prefill_overlap) = prefill_router
                .find_best_match(request_id, token_ids, None, update_states)
                .await
                .context("Failed to query prefill worker")?;

            // Query decode worker with overlap_score_weight = 0
            // (we want to minimize load, not maximize overlap for decode)
            let mut decode_override = dynamo_llm::kv_router::RouterConfigOverride::default();
            decode_override.overlap_score_weight = Some(0.0);

            let (decode_worker, decode_overlap) = self
                .decode_router
                .find_best_match(request_id, token_ids, Some(&decode_override), update_states)
                .await
                .context("Failed to query decode worker")?;

            Ok(RouteQueryResult {
                prefill_worker_id: prefill_worker.worker_id,
                decode_worker_id: decode_worker.worker_id,
                dp_rank: decode_worker.dp_rank,
                overlap_blocks: prefill_overlap.max(decode_overlap),
                is_disaggregated: true,
            })
        } else {
            drop(prefill_router_guard); // Release lock

            // Aggregated mode: query decode router only
            let (worker, overlap) = self
                .decode_router
                .find_best_match(request_id, token_ids, None, update_states)
                .await
                .context("Failed to query worker")?;

            Ok(RouteQueryResult {
                prefill_worker_id: 0,
                decode_worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                overlap_blocks: overlap,
                is_disaggregated: false,
            })
        }
    }

    /// Check if disaggregated mode is currently active
    pub async fn is_disaggregated(&self) -> bool {
        self.prefill_router.read().await.is_some()
    }

    /// Mark prefill as completed for a request (for state tracking)
    pub async fn mark_prefill_complete(&self, request_id: &str) -> Result<()> {
        self.decode_router
            .mark_prefill_completed(request_id)
            .await
            .context("Failed to mark prefill complete")
    }

    /// Free a request (release resources)
    pub async fn free(&self, request_id: &str) -> Result<()> {
        self.decode_router
            .free(request_id)
            .await
            .context("Failed to free request")
    }
}

// =============================================================================
// C FFI Interface
// =============================================================================

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

    let namespace = match CStr::from_ptr(namespace).to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let component = if component.is_null() {
        "backend"
    } else {
        match CStr::from_ptr(component).to_str() {
            Ok(s) if !s.is_empty() => s,
            _ => "backend",
        }
    };

    let model_name = match CStr::from_ptr(model_name).to_str() {
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
            *out_handle = handle;
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
    out_result: *mut RouteQueryResult,
) -> QueryRouterResult {
    if handle.is_null() || token_ids.is_null() || out_result.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let router = &*handle;
    let tokens = std::slice::from_raw_parts(token_ids, token_count);

    let request_id = if request_id.is_null() {
        None
    } else {
        CStr::from_ptr(request_id).to_str().ok()
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
            *out_result = route_result;
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

    let router = &*handle;

    // Get runtime to execute async check
    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return false,
    };

    runtime.secondary().block_on(async { router.is_disaggregated().await })
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

    let router = &*handle;
    let request_id = match CStr::from_ptr(request_id).to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime
        .secondary()
        .block_on(async { router.mark_prefill_complete(request_id).await });

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

    let router = &*handle;
    let request_id = match CStr::from_ptr(request_id).to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(_) => return QueryRouterResult::ErrQueryFailed,
    };

    let result = runtime
        .secondary()
        .block_on(async { router.free(request_id).await });

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
        drop(Box::from_raw(handle));
    }
}
