// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::borrow::Cow;
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use dynamo_llm::kv_router::{protocols::*, publisher::KvEventPublisher};
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_runtime::discovery::DiscoveryQuery;
use dynamo_runtime::{DistributedRuntime, Worker};

use dynamo_runtime::Runtime;

use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::KvRouterConfig;
use dynamo_llm::kv_router::protocols::WorkerWithDpRank;
use dynamo_llm::kv_router::{KvRouter, PrefillRouter, RouterConfigOverride};
use dynamo_runtime::pipeline::RouterMode;

static WK: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
// [FIXME] shouldn't the publisher be instance passing between API calls?
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();

/// Convert a C string pointer to a Rust string, falling back to a default when:
/// - the pointer is NULL,
/// - the bytes are not valid UTF-8,
/// - or the resulting string is empty/whitespace.
#[inline]
unsafe fn cstr_or_default<'a>(ptr: *const c_char, default_val: &'a str) -> Cow<'a, str> {
    if ptr.is_null() {
        return Cow::from(default_val);
    }
    match unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(|s| s.trim())
    {
        Some(s) if !s.is_empty() => Cow::from(s.to_owned()),
        _ => Cow::from(default_val),
    }
}

fn initialize_tracing() {
    // Sets up RUST_LOG environment variable for logging while KV Publishing
    // Example: os.environ["RUST_LOG"] = "debug"
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    tracing::debug!("Tracing initialized");
}

#[repr(u32)]
pub enum DynamoLlmResult {
    OK = 0,
    ERR = 1,
}

// Wait for the discovery daemon to sync indefinitely and return at least one instance.
// This is because the Model info is registered by workers and it may take up to 30 min for the model weights to load and for the worker to register itself.
// The waiting timeout is implemented in the Kubernetes StartupProbe. The EPP waiting loops runs indefinitely, the Probe is a single source of truth with when to kill the EPP if discovery fails.
// If workers are not found within the probe's failureThreshold × periodSeconds, the pod will be killed and restarted.
// Users can adjust the StartupProbe waiting timed in the DGD for large models.
async fn wait_for_discovery_sync(drt: &DistributedRuntime) -> usize {
    tracing::info!(
        "Waiting for discovery to sync (no timeout - controlled by K8s StartupProbe)..."
    );
    let discovery = drt.discovery();

    loop {
        match discovery.list(DiscoveryQuery::AllModels).await {
            Ok(instances) if !instances.is_empty() => {
                return instances.len();
            }
            Ok(_) => {
                tracing::debug!("No instances yet, waiting...");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
            Err(e) => {
                // Log and continue - transient errors shouldn't stop the wait
                tracing::warn!("Discovery list error: {}, retrying...", e);
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    }
}

/// # Safety
/// the namespace_c_str and component_c_str are passed as pointers to C strings
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_llm_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    kv_block_size: u32,
) -> DynamoLlmResult {
    initialize_tracing();
    let wk = match WK.get_or_try_init(Worker::from_settings) {
        Ok(wk) => wk.clone(),
        Err(e) => {
            tracing::error!(error = ?e, "Failed to initialize runtime (Worker::from_settings)");
            return DynamoLlmResult::ERR;
        }
    };
    let rt = wk.runtime();
    let secondary = rt.secondary().clone();
    let result = secondary.block_on(async {
        // Initialize the distributed runtime
        match DRT
            .get_or_try_init(async { DistributedRuntime::from_settings(rt.clone()).await })
            .await
        {
            Ok(drt) => {
                // Wait for discovery to sync before returning.
                // This is needed because dynamo_create_worker_selection_pipeline() is called
                // immediately after, and it needs discovery.list() to return data.
                // The discovery daemon takes time to query K8s and returns async, so we need to wait.
                // Note: This waits indefinitely - the K8s StartupProbe is the timeout mechanism.
                wait_for_discovery_sync(drt).await;
                Ok(())
            }
            Err(e) => {
                tracing::error!(error = ?e, "Failed to initialize distributed runtime");
                Err(DynamoLlmResult::ERR)
            }
        }
    });
    let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            tracing::error!(error = ?e, "Failed to convert C string to Rust string (namespace)");
            return DynamoLlmResult::ERR;
        }
    };

    let component_cow = unsafe { cstr_or_default(component_c_str, "backend") };
    if let Cow::Borrowed("backend") = &component_cow {
        tracing::info!("defaulting to \"backend\" for component");
    }
    let component: String = component_cow.into_owned();

    match result {
        Ok(_) => match KV_PUB.get_or_try_init(move || {
            dynamo_create_kv_publisher(namespace, component, kv_block_size)
        }) {
            Ok(_) => DynamoLlmResult::OK,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to initialize distributed runtime");
                DynamoLlmResult::ERR
            }
        },
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_llm_shutdown() -> DynamoLlmResult {
    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            tracing::error!("Runtime not initialized");
            return DynamoLlmResult::ERR;
        }
    };

    wk.runtime().shutdown();

    DynamoLlmResult::OK
}

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_llm_load_publisher_create() -> DynamoLlmResult {
    DynamoLlmResult::OK
}

// instantiate a kv publisher
// this will bring up the task to publish and the channels to await publishing events
// the [`dynamo_kv_publish_store_event`] call will use a handle to the publisher to send events
// store and the [`dynamo_kv_event_create_removed`] will create remove events
// these call mus be driving by external c++ threads that are consuming the kv events from the
// c++ executor api

fn dynamo_create_kv_publisher(
    namespace: String,
    component: String,
    kv_block_size: u32,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!("Creating KV Publisher for model: {}", component);
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(backend, kv_block_size, None)
        }
        Err(e) => Err(e),
    }
}

fn kv_event_create_stored_block_from_parts(
    block_hash: u64,
    token_ids: *const u32,
    num_tokens: usize,
    kv_block_size: u32,
    _lora_id: u64,
) -> KvCacheStoredBlockData {
    let tokens_hash = compute_block_hash_for_seq(
        unsafe { std::slice::from_raw_parts(token_ids, num_tokens) },
        kv_block_size,
        None,
    )[0];
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(block_hash),
        tokens_hash,
        mm_extra_info: None,
    }
}
static WARN_COUNT: AtomicU32 = AtomicU32::new(0);

fn kv_event_create_stored_from_parts(
    kv_params: DynamoKvStoredEventParams,
    kv_block_size: u32,
) -> KvCacheEvent {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for block_idx in 0..kv_params.num_blocks {
        let block_hash = unsafe { *kv_params.block_ids.offset(block_idx.try_into().unwrap()) };
        let tokens = unsafe { kv_params.token_ids.offset(token_offset.try_into().unwrap()) };
        let num_toks = unsafe {
            *kv_params
                .num_block_tokens
                .offset(block_idx.try_into().unwrap())
        };

        if num_toks != (kv_block_size as usize) {
            if WARN_COUNT
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
                    if c < 3 { Some(c + 1) } else { None }
                })
                .is_ok()
            {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    num_toks
                );
            }
            break;
        }
        token_offset += num_toks;
        blocks.push(kv_event_create_stored_block_from_parts(
            block_hash,
            tokens,
            num_toks,
            kv_block_size,
            kv_params.lora_id,
        ));
    }

    KvCacheEvent {
        data: KvCacheEventData::Stored(KvCacheStoreData {
            blocks,
            parent_hash: kv_params.parent_hash.map(ExternalSequenceBlockHash),
        }),
        event_id: kv_params.event_id,
        dp_rank: 0,
    }
}

fn kv_event_create_removed_from_parts(
    event_id: u64,
    block_ids: *const u64,
    num_blocks: usize,
) -> KvCacheEvent {
    let block_hashes: Vec<ExternalSequenceBlockHash> =
        unsafe { std::slice::from_raw_parts(block_ids, num_blocks) }
            .to_vec()
            .iter()
            .map(|&v| ExternalSequenceBlockHash(v))
            .collect();
    KvCacheEvent {
        event_id,
        data: KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
        dp_rank: 0,
    }
}

pub struct DynamoKvStoredEventParams {
    pub event_id: u64,
    pub token_ids: *const u32,
    pub num_block_tokens: *const usize,
    pub block_ids: *const u64,
    pub num_blocks: usize,
    pub parent_hash: Option<u64>,
    pub lora_id: u64,
}

/// # Safety
/// parent_hash is passed as pointer to indicate whether the blocks
/// has a parent hash or not. nullptr is used to represent no parent hash
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_kv_event_publish_stored(
    event_id: u64,
    token_ids: *const u32,
    num_block_tokens: *const usize,
    block_ids: *const u64,
    num_blocks: usize,
    parent_hash: *const u64,
    lora_id: u64,
) -> DynamoLlmResult {
    let parent_hash = {
        if parent_hash.is_null() {
            None
        } else {
            Some(unsafe { *parent_hash })
        }
    };
    let kv_params = DynamoKvStoredEventParams {
        event_id,
        token_ids,
        num_block_tokens,
        block_ids,
        num_blocks,
        parent_hash,
        lora_id,
    };
    let publisher = KV_PUB.get().unwrap();
    let event = kv_event_create_stored_from_parts(kv_params, publisher.kv_block_size());
    match publisher.publish(event) {
        Ok(_) => DynamoLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing stored kv event {:?}", e);
            DynamoLlmResult::ERR
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_event_publish_removed(
    event_id: u64,
    block_ids: *const u64,
    num_blocks: usize,
) -> DynamoLlmResult {
    let publisher = KV_PUB.get().unwrap();
    let event = kv_event_create_removed_from_parts(event_id, block_ids, num_blocks);
    match publisher.publish(event) {
        Ok(_) => DynamoLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing removed kv event {:?}", e);
            DynamoLlmResult::ERR
        }
    }
}

/* ------------------------------------------------------------------------
 *  Router Bindings for GAIE EPP
 * ------------------------------------------------------------------------ */

// Default timeout for bookkeeping operations
const BOOKKEEPING_TIMEOUT_SEC: u64 = 5;
/// Complete routing result for a chat completion request (C-compatible)
#[repr(C)]
pub struct CRoutingResult {
    /// Whether disaggregated mode is active
    pub is_disaggregated: bool,
    /// Prefill worker ID (only valid if is_disaggregated is true)
    pub prefill_worker_id: u64,
    /// Decode worker ID
    pub decode_worker_id: u64,
    /// Token IDs (needed for add_request callback)
    pub token_ids: *mut u32,
    /// Number of tokens in the request
    pub token_count: usize,
}

impl Default for CRoutingResult {
    fn default() -> Self {
        Self {
            is_disaggregated: false,
            prefill_worker_id: 0,
            decode_worker_id: 0,
            token_ids: ptr::null_mut(),
            token_count: 0,
        }
    }
}

/// Container holding routers and preprocessor for query routing
pub struct RouterHandles {
    prefill_router: Arc<PrefillRouter>,
    decode_router: Arc<KvRouter>,
    #[allow(dead_code)]
    model_manager: Arc<ModelManager>,
    #[allow(dead_code)]
    namespace: String,
    /// Cached runtime for executing async operations (avoids creating new runtime per call)
    runtime: Runtime,
    /// Preprocessor for tokenization and template application (fetched via discovery)
    preprocessor: Option<Arc<OpenAIPreprocessor>>,
}

impl RouterHandles {
    /// Query optimal prefill worker for a request.
    /// Returns worker_id on success.
    async fn query_prefill_worker(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<dynamo_llm::kv_router::protocols::BlockExtraInfo>]>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
    ) -> Result<u64, QueryRouterResult> {
        self.prefill_router
            .query_prefill_worker(
                tokens,
                block_mm_infos,
                update_states,
                lora_name,
                priority_jump,
            )
            .await
            .map(|(worker_id, _dp_rank)| worker_id)
            .map_err(|e| {
                tracing::error!(error = ?e, "Prefill query failed");
                QueryRouterResult::ErrQueryFailed
            })
    }

    /// Query optimal decode worker for a request.
    /// For disaggregated mode, set `is_disaggregated` to true to use overlap_score_weight=0
    /// (since KV cache is being transferred from prefill, not reused).
    ///
    /// Note: The C bindings are query-only and must not mutate router state during worker
    /// selection. State updates require a `context_id` (request id) and are managed via the
    /// explicit bookkeeping APIs (`add_request`, `mark_prefill_complete`, `free_request`).
    /// Returns (worker, overlap_blocks) on success.
    async fn query_decode_worker(
        &self,
        tokens: &[u32],
        is_disaggregated: bool,
    ) -> Result<(WorkerWithDpRank, u32), QueryRouterResult> {
        // For decode phase in disaggregated mode, use overlap_score_weight=0
        // This matches prefill_router.rs
        let config_override = if is_disaggregated {
            Some(RouterConfigOverride {
                overlap_score_weight: Some(0.0),
                ..Default::default()
            })
        } else {
            None
        };

        self.decode_router
            .find_best_match(
                None,
                tokens,
                None,
                config_override.as_ref(),
                false,
                None,
                0.0,
            )
            .await
            .map_err(|e| {
                tracing::error!(error = ?e, "Decode query failed");
                QueryRouterResult::ErrQueryFailed
            })
    }
}

/// Opaque handle for the router pair
pub type RouterHandlesPtr = *mut RouterHandles;

/// Result codes for query router C FFI
#[repr(u32)]
pub enum QueryRouterResult {
    Ok = 0,
    ErrInvalidHandle = 1,
    ErrInvalidParam = 2,
    ErrInitFailed = 3,
    ErrQueryFailed = 4,
    ErrDisaggEnforced = 5,
    ErrTimeout = 6,
}

/// Build a `KvRouterConfig` from defaults, overridden by optional `DYN_*` environment variables.
///
/// Supported env vars (all optional — unset or empty values are ignored):
/// - `DYN_OVERLAP_SCORE_WEIGHT` — Weight for overlap score in worker selection (default: 1.0)
/// - `DYN_ROUTER_TEMPERATURE` — Temperature for worker sampling via softmax (default: 0.0)
/// - `DYN_USE_KV_EVENTS` — Use KV events for cache tracking (default: true)
/// - `DYN_ROUTER_REPLICA_SYNC` — Enable replica synchronization (default: false)
/// - `DYN_ROUTER_TRACK_ACTIVE_BLOCKS` — Track active blocks (default: true)
/// - `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` — Track output blocks during generation (default: false)
fn kv_router_config_from_env() -> KvRouterConfig {
    let mut cfg = KvRouterConfig::default();

    fn env_f64(key: &str) -> Option<f64> {
        std::env::var(key).ok().and_then(|v| v.parse().ok())
    }
    fn env_bool(key: &str) -> Option<bool> {
        std::env::var(key)
            .ok()
            .and_then(|v| match v.to_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => Some(true),
                "false" | "0" | "no" | "off" => Some(false),
                _ => None,
            })
    }

    if let Some(v) = env_f64("DYN_OVERLAP_SCORE_WEIGHT") {
        cfg.overlap_score_weight = v;
    }
    if let Some(v) = env_f64("DYN_ROUTER_TEMPERATURE") {
        cfg.router_temperature = v;
    }
    if let Some(v) = env_bool("DYN_USE_KV_EVENTS") {
        cfg.use_kv_events = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_REPLICA_SYNC") {
        cfg.router_replica_sync = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_TRACK_ACTIVE_BLOCKS") {
        cfg.router_track_active_blocks = v;
    }
    if let Some(v) = env_bool("DYN_ROUTER_TRACK_OUTPUT_BLOCKS") {
        cfg.router_track_output_blocks = v;
    }

    tracing::info!(
        overlap_score_weight = cfg.overlap_score_weight,
        router_temperature = cfg.router_temperature,
        use_kv_events = cfg.use_kv_events,
        router_replica_sync = cfg.router_replica_sync,
        router_track_active_blocks = cfg.router_track_active_blocks,
        router_track_output_blocks = cfg.router_track_output_blocks,
        "KvRouterConfig initialized (DYN_* env overrides applied)"
    );

    cfg
}

/// Create router handles for query-only routing
///
/// This function waits for at least one decode worker to be discovered before returning.
/// It auto-detects disaggregated mode by checking if prefill workers are present.
/// The KV cache block size is automatically fetched from the model card via discovery.
///
/// # Arguments
/// - `namespace`: Namespace for the model
/// - `component`: Component name (defaults to "backend" if NULL or empty)
/// - `enforce_disagg`: If true, disaggregated mode is required (fails if no prefill workers found)
/// - `out_handle`: Output handle
///
/// # Safety
/// - All string parameters must be valid null-terminated C strings
/// - The returned handle must be freed with `destroy`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn create_routers(
    namespace: *const c_char,
    component: *const c_char,
    enforce_disagg: bool,
    out_handle: *mut RouterHandlesPtr,
) -> QueryRouterResult {
    if namespace.is_null() || out_handle.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let namespace_str = match unsafe { CStr::from_ptr(namespace) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let component_str = if component.is_null() {
        "backend".to_string()
    } else {
        match unsafe { CStr::from_ptr(component) }.to_str() {
            Ok(s) if !s.is_empty() => s.to_owned(),
            _ => "backend".to_string(),
        }
    };

    // Create the runtime once - it will be stored in RouterHandles and reused
    let runtime = match Runtime::from_settings() {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to create runtime");
            return QueryRouterResult::ErrInitFailed;
        }
    };

    // Clone for use inside the async block (the original will be moved into handles)
    let runtime_for_async = runtime.clone();

    let result = runtime_for_async.secondary().block_on(async {
        let drt = match DistributedRuntime::from_settings(runtime_for_async.clone()).await {
            Ok(drt) => drt,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to create distributed runtime");
                return Err(QueryRouterResult::ErrInitFailed);
            }
        };

        // Wait for at least one worker to be discovered before proceeding
        // This ensures the decode router can be created successfully
        let instance_count = wait_for_discovery_sync(&drt).await;
        if instance_count == 0 {
            tracing::error!(
                "Discovery sync failed: no worker instances found. Is the backend running?"
            );
            return Err(QueryRouterResult::ErrInitFailed);
        }
        tracing::info!(
            "Discovery sync complete, {} worker(s) found",
            instance_count
        );

        let kv_router_config = kv_router_config_from_env();

        // Get component and endpoint
        let component_handle = match drt.namespace(&namespace_str) {
            Ok(ns) => match ns.component(&component_str) {
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

        // Fetch model card via discovery and create preprocessor + get block_size
        let (preprocessor, block_size, model_name) =
            match fetch_preprocessor_from_discovery(&drt, &namespace_str).await {
                Ok((prep, bs, name)) => {
                    tracing::info!(
                        kv_cache_block_size = bs,
                        "Preprocessor created from discovery"
                    );
                    (Some(prep), bs, name)
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Failed to fetch model card from discovery - cannot determine block_size"
                    );
                    return Err(QueryRouterResult::ErrInitFailed);
                }
            };

        // Create decode router
        let decode_router = match model_manager
            .kv_chooser_for(
                &endpoint,
                block_size,
                Some(kv_router_config),
                WORKER_TYPE_DECODE,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to create decode router");
                return Err(QueryRouterResult::ErrInitFailed);
            }
        };

        // Create PrefillRouter based on one-time discovery of prefill workers
        // Auto-detects disaggregated mode by checking if prefill workers are present
        // The prefill workers have to be created before the epp is created.
        // Given that we wait first for the decode worker to show up it is reasonable to assume the prefill will be up as well.
        let prefill_router = match find_prefill_endpoint(&drt, &namespace_str).await {
            Some(prefill_endpoint) => {
                tracing::info!("Prefill worker found, running in disaggregated mode");
                let mut prefill_config = kv_router_config;
                prefill_config.router_track_active_blocks = false;

                // Create immediately-resolved channel to activate router
                let (tx, rx) = tokio::sync::oneshot::channel();
                let _ = tx.send(prefill_endpoint);

                PrefillRouter::new(
                    rx,
                    model_manager.clone(),
                    RouterMode::KV,
                    block_size,
                    Some(prefill_config),
                    enforce_disagg,
                    model_name.clone(),
                    namespace_str.clone(),
                )
            }
            None if enforce_disagg => {
                tracing::error!("Prefill workers required (enforce_disagg=true) but none found");
                return Err(QueryRouterResult::ErrDisaggEnforced);
            }
            None => {
                tracing::info!("No prefill workers found, running in aggregated mode");
                PrefillRouter::disabled(model_manager.clone(), RouterMode::KV, enforce_disagg)
            }
        };

        Ok((
            prefill_router,
            decode_router,
            model_manager,
            namespace_str,
            preprocessor,
        ))
    });

    match result {
        Ok((prefill_router, decode_router, model_manager, namespace_str, preprocessor)) => {
            let handles = RouterHandles {
                prefill_router,
                decode_router,
                model_manager,
                namespace: namespace_str,
                runtime, // Store the runtime for reuse
                preprocessor,
            };
            unsafe { *out_handle = Box::into_raw(Box::new(handles)) };
            QueryRouterResult::Ok
        }
        Err(code) => code,
    }
}

/// Add a request to the router's bookkeeping after worker selection.
///
/// Register the request with the KvRouter's scheduler for tracking active blocks
/// and managing prefill/decode lifecycle. Call this after `query_decode` returns
/// worker IDs and before sending the request to the worker.
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle
/// - `request_id` must be a valid null-terminated C string
/// - `token_ids` must point to at least `token_count` valid u32 values
#[unsafe(no_mangle)]
pub unsafe extern "C" fn add_request(
    handle: RouterHandlesPtr,
    request_id: *const c_char,
    token_ids: *const u32,
    token_count: usize,
    worker_id: u64,
    dp_rank: u32,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
    let request_id_str = match unsafe { CStr::from_ptr(request_id) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let tokens: Vec<u32> = if token_count > 0 && !token_ids.is_null() {
        unsafe { std::slice::from_raw_parts(token_ids, token_count) }.to_vec()
    } else {
        Vec::new()
    };

    let decode_router = handles.decode_router.clone();

    let result = handles.runtime.secondary().block_on(async {
        let timeout_duration = Duration::from_secs(BOOKKEEPING_TIMEOUT_SEC);

        tokio::time::timeout(timeout_duration, async {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);

            // Compute overlap_blocks using the public method
            let overlap_blocks = match decode_router.get_overlap_blocks(&tokens, worker).await {
                Ok(overlap) => overlap,
                Err(e) => {
                    tracing::warn!(error = ?e, "Failed to compute overlap, using 0");
                    0
                }
            };

            decode_router
                .add_request(
                    request_id_str.clone(),
                    &tokens,
                    overlap_blocks,
                    None,
                    worker,
                    None, // lora_name
                    None, // router_config_override
                )
                .await;

            tracing::debug!(
                request_id = %request_id_str,
                worker_id = worker_id,
                dp_rank = dp_rank,
                overlap_blocks = overlap_blocks,
                token_count = tokens.len(),
                "add_request completed"
            );
        })
        .await
    });

    match result {
        Ok(()) => QueryRouterResult::Ok,
        Err(_elapsed) => {
            tracing::warn!(
                request_id = %request_id_str,
                timeout_secs = BOOKKEEPING_TIMEOUT_SEC,
                "add_request timed out"
            );
            QueryRouterResult::ErrTimeout
        }
    }
}

/// Mark prefill as completed for a request.
///
/// Call when the first token is generated to release prefill tokens from decode worker's load
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mark_prefill_complete(
    handle: RouterHandlesPtr,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
    let request_id_str = match unsafe { CStr::from_ptr(request_id) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let decode_router = handles.decode_router.clone();

    let result = handles.runtime.secondary().block_on(async {
        let timeout_duration = Duration::from_secs(BOOKKEEPING_TIMEOUT_SEC);

        tokio::time::timeout(timeout_duration, async {
            if let Err(e) = decode_router.mark_prefill_completed(&request_id_str).await {
                tracing::warn!(
                    request_id = %request_id_str,
                    error = %e,
                    "Failed to mark prefill complete"
                );
            } else {
                tracing::debug!(
                    request_id = %request_id_str,
                    "mark_prefill_complete completed"
                );
            }
        })
        .await
    });

    match result {
        Ok(()) => QueryRouterResult::Ok,
        Err(_elapsed) => {
            tracing::warn!(
                request_id = %request_id_str,
                timeout_secs = BOOKKEEPING_TIMEOUT_SEC,
                "mark_prefill_complete timed out"
            );
            QueryRouterResult::ErrTimeout
        }
    }
}

/// Free a request from the router's bookkeeping.
///
/// Call this when the stream is closed (completed or cancelled) to release all resources.
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle
/// - `request_id` must be a valid null-terminated C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_request(
    handle: RouterHandlesPtr,
    request_id: *const c_char,
) -> QueryRouterResult {
    if handle.is_null() || request_id.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };
    let request_id_str = match unsafe { CStr::from_ptr(request_id) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    let decode_router = handles.decode_router.clone();

    let result = handles.runtime.secondary().block_on(async {
        let timeout_duration = Duration::from_secs(BOOKKEEPING_TIMEOUT_SEC);

        tokio::time::timeout(timeout_duration, async {
            if let Err(e) = decode_router.free(&request_id_str).await {
                tracing::warn!(
                    request_id = %request_id_str,
                    error = %e,
                    "Failed to free request"
                );
            } else {
                tracing::debug!(
                    request_id = %request_id_str,
                    "free_request completed"
                );
            }
        })
        .await
    });

    match result {
        Ok(()) => QueryRouterResult::Ok,
        Err(_elapsed) => {
            tracing::warn!(
                request_id = %request_id_str,
                timeout_secs = BOOKKEEPING_TIMEOUT_SEC,
                "free_request timed out"
            );
            QueryRouterResult::ErrTimeout
        }
    }
}

/// Destroy router handles
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle or null
/// - After this call, `handle` must not be used
#[unsafe(no_mangle)]
pub unsafe extern "C" fn destroy(handle: RouterHandlesPtr) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Route a chat completion request in a single call.
///
/// This is the main function for EPP to route a `/v1/chat/completions` request.
/// It combines tokenization and worker selection in one call:
/// 1. Applies the chat template to the request JSON
/// 2. Tokenizes the formatted prompt
/// 3. Queries the prefill router (if disaggregated mode)
/// 4. Queries the decode router
/// 5. Returns worker IDs and token_ids
///
/// After this call, EPP should:
/// - Call `add_request()` to register the request for bookkeeping
/// - Set worker ID headers and forward to backend
/// - Call `mark_prefill_complete()` on first token
/// - Call `free_request()` when the stream ends
/// - Call `free_routing_result()` to free the result
///
/// # Safety
/// - `handle` must be a valid RouterHandles handle
/// - `request_json` must be a valid null-terminated C string containing JSON
/// - `out_result` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn route_request(
    handle: RouterHandlesPtr,
    request_json: *const c_char,
    out_result: *mut CRoutingResult,
) -> QueryRouterResult {
    if handle.is_null() || request_json.is_null() || out_result.is_null() {
        return QueryRouterResult::ErrInvalidParam;
    }

    let handles = unsafe { &*handle };

    // Get preprocessor
    let preprocessor = match &handles.preprocessor {
        Some(p) => p,
        None => {
            tracing::error!("Preprocessor not available");
            return QueryRouterResult::ErrInitFailed;
        }
    };

    let json_str = match unsafe { CStr::from_ptr(request_json) }.to_str() {
        Ok(s) => s,
        Err(_) => return QueryRouterResult::ErrInvalidParam,
    };

    // Parse JSON
    let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
        match serde_json::from_str(json_str) {
            Ok(req) => req,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to parse request JSON");
                return QueryRouterResult::ErrInvalidParam;
            }
        };

    // Apply chat template
    let formatted_prompt = match preprocessor.apply_template(&request) {
        Ok(Some(prompt)) => prompt,
        Ok(None) => String::new(),
        Err(e) => {
            tracing::error!(error = ?e, "Failed to apply chat template");
            return QueryRouterResult::ErrQueryFailed;
        }
    };

    // Tokenize
    let encoding = match preprocessor.tokenize(&formatted_prompt) {
        Ok(enc) => enc,
        Err(e) => {
            tracing::error!(error = ?e, "Failed to tokenize");
            return QueryRouterResult::ErrQueryFailed;
        }
    };

    let tokens = encoding.token_ids();
    let token_count = tokens.len();
    let is_disaggregated = handles.prefill_router.is_activated();

    // Query workers
    let result = handles.runtime.secondary().block_on(async {
        let prefill_worker_id = if is_disaggregated {
            handles
                .query_prefill_worker(tokens, None, false, None, 0.0)
                .await?
        } else {
            0
        };

        let (decode_worker, _overlap_blocks) = handles
            .query_decode_worker(tokens, is_disaggregated)
            .await?;

        tracing::info!(
            is_disaggregated = is_disaggregated,
            prefill_worker_id = prefill_worker_id,
            decode_worker_id = decode_worker.worker_id,
            decode_dp_rank = decode_worker.dp_rank,
            token_count = token_count,
            "Routed chat request"
        );

        Ok((prefill_worker_id, decode_worker))
    });

    match result {
        Ok((prefill_worker_id, decode_worker)) => {
            // Allocate and copy token IDs for caller (needed for add_request bookkeeping)
            let token_vec: Vec<u32> = tokens.to_vec();
            let mut tokens_boxed = token_vec.into_boxed_slice();
            let token_ptr = tokens_boxed.as_mut_ptr();
            std::mem::forget(tokens_boxed);

            unsafe {
                *out_result = CRoutingResult {
                    is_disaggregated,
                    prefill_worker_id,
                    decode_worker_id: decode_worker.worker_id,
                    token_ids: token_ptr,
                    token_count,
                };
            }
            QueryRouterResult::Ok
        }
        Err(code) => code,
    }
}

/// Free a routing result.
///
/// # Safety
/// - `result` must be a valid pointer to a CRoutingResult previously returned by route functions
#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_routing_result(result: *mut CRoutingResult) {
    if result.is_null() {
        return;
    }

    let res = unsafe { &mut *result };

    // Free token IDs
    if !res.token_ids.is_null() && res.token_count > 0 {
        drop(unsafe {
            Box::from_raw(std::slice::from_raw_parts_mut(
                res.token_ids,
                res.token_count,
            ))
        });
        res.token_ids = ptr::null_mut();
        res.token_count = 0;
    }
}

/// Fetch model card via discovery and create preprocessor.
///
/// This function:
/// 1. Lists all models via discovery
/// 2. Finds the first model in the target namespace (decode workers only)
/// 3. Downloads the model config (tokenizer files) if needed
/// 4. Creates an OpenAIPreprocessor from the model card
/// 5. Returns the preprocessor, the kv_cache_block_size, and model_name from the model card
async fn fetch_preprocessor_from_discovery(
    drt: &DistributedRuntime,
    target_namespace: &str,
) -> anyhow::Result<(Arc<OpenAIPreprocessor>, u32, String)> {
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_runtime::discovery::DiscoveryInstance;

    let discovery = drt.discovery();

    // List all models
    let instances = discovery.list(DiscoveryQuery::AllModels).await?;

    // Find first model card in the target namespace (decode workers only)
    let mut model_card: Option<ModelDeploymentCard> = None;

    for instance in instances {
        if let DiscoveryInstance::Model { namespace, .. } = &instance {
            // Filter by namespace
            if namespace != target_namespace {
                continue;
            }

            match instance.deserialize_model::<ModelDeploymentCard>() {
                Ok(card) => {
                    // Skip prefill-only workers, we want decode workers for routing
                    if card.model_type.supports_prefill()
                        && !card.model_type.supports_chat()
                        && !card.model_type.supports_completions()
                    {
                        continue;
                    }
                    model_card = Some(card);
                    break;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "Failed to deserialize model card, skipping");
                    continue;
                }
            }
        }
    }

    let mut card = model_card.ok_or_else(|| {
        anyhow::anyhow!(
            "No model found in namespace '{}' via discovery",
            target_namespace
        )
    })?;

    let kv_cache_block_size = card.kv_cache_block_size;
    let model_name = card.name().to_string();
    tracing::info!(
        model_name = model_name,
        kv_cache_block_size = kv_cache_block_size,
        "Found model card via discovery"
    );

    // Download config (tokenizer files) if not local
    card.download_config().await?;

    // Create preprocessor
    let preprocessor = OpenAIPreprocessor::new(card)?;
    Ok((preprocessor, kv_cache_block_size, model_name))
}

/// Find a prefill endpoint from already-discovered instances (one-time filter).
/// Returns the endpoint if a prefill worker is found in the target namespace.
async fn find_prefill_endpoint(
    drt: &DistributedRuntime,
    target_namespace: &str,
) -> Option<dynamo_runtime::component::Endpoint> {
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_runtime::discovery::DiscoveryInstance;

    let discovery = drt.discovery();
    let instances = match discovery.list(DiscoveryQuery::AllModels).await {
        Ok(instances) => instances,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to list instances for prefill discovery");
            return None;
        }
    };

    for instance in instances {
        if let DiscoveryInstance::Model {
            namespace,
            component,
            endpoint,
            ..
        } = &instance
        {
            // Filter by namespace
            if namespace != target_namespace {
                continue;
            }

            let card = match instance.deserialize_model::<ModelDeploymentCard>() {
                Ok(card) => card,
                Err(_) => continue,
            };

            // Only handle prefill models
            if !card.model_type.supports_prefill() {
                continue;
            }

            tracing::info!(
                model_name = card.name(),
                "Prefill worker found in discovered instances"
            );

            // Build and return the endpoint
            if let Ok(ns) = drt.namespace(namespace)
                && let Ok(comp) = ns.component(component)
            {
                return Some(comp.endpoint(endpoint));
            }
        }
    }

    None
}
