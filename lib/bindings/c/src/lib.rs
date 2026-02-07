// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_llm::{
    discovery::{KvWorkerMonitor, ModelWatcher},
    kv_router::{protocols::*, publisher::KvEventPublisher},
};
use dynamo_runtime::discovery::DiscoveryQuery;
use dynamo_runtime::{DistributedRuntime, Worker};
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

// Need to setup etcd and nats to run these tests
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::ffi::CString;

//     #[test]
//     fn test_dynamo_llm_init() {
//         // Create C-compatible strings
//         let namespace = CString::new("test_namespace").unwrap();
//         let component = CString::new("test_component").unwrap();

//         // Call the init function
//         let result = unsafe {
//             dynamo_llm_init(
//                 namespace.as_ptr(),
//                 component.as_ptr(),
//                 1,  // worker_id
//                 32, // kv_block_size
//             )
//         };

//         assert_eq!(result as u32, DynamoLlmResult::OK as u32);

//         assert!(WK.get().is_some());

//         let shutdown_result = dynamo_llm_shutdown();
//         assert_eq!(shutdown_result as u32, DynamoLlmResult::OK as u32);
//     }
// }
/* ------------------------------------------------------------------------
 * Worker selection pipeline
 * ------------------------------------------------------------------------ */
use std::pin::Pin;

const GENERATE_ENDPOINT: &str = "generate";

use anyhow::Context;
use dynamo_runtime::{Runtime, traits::DistributedRuntimeProvider};

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::build_routed_pipeline;
use dynamo_llm::http::service::metrics::Metrics;
use dynamo_llm::kv_router::KvRouterConfig;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::protocols::openai::nvext::NvExt;
use dynamo_llm::types::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::{
    engine::AsyncEngineStream,
    pipeline::{ManyOut, RouterMode, ServiceEngine, SingleIn},
};
/// Opaque handle exposed to C — it owns its own Worker/runtime and engine.
pub struct WorkerSelectionPipeline {
    wk: Worker,
    engine: ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
    /// KV router for bookkeeping operations (only present when router_mode is KV)
    kv_router: Option<Arc<dynamo_llm::kv_router::KvRouter>>,
}

/// Create a worker-selection pipeline ("generate" endpoint).
///
/// # Safety
/// - `namespace_c_str`, `component_c_str`, and `model_name_c_str` must be **non-null** pointers to
///   **NUL-terminated** C strings that contain **valid UTF-8**. They must remain valid for the
///   duration of this call.
/// - `pipeline_out` must be **non-null** and point to writable memory for a `*mut WorkerSelectionPipeline`.
///   On success this function writes exactly once to `*pipeline_out`. The caller becomes the owner of
///   that pointer and **must** later free it by calling `dynamo_destroy_worker_selection_pipeline`.
/// - Must be called **after** a successful `dynamo_llm_init()`; otherwise behavior is undefined.
/// - This function is not signal-safe and must not be called from a signal handler.
/// - This function may block internally; do not call it from contexts that forbid blocking.
///
/// # Errors
/// Returns `DynamoLlmResult::ERR` on failure and does not write to `pipeline_out`.
/// # Safety
/// See detailed safety docs above. Additional parameter:
/// - `enforce_disagg`: If true, requests fail when disaggregated serving is unavailable.
///   If false, falls back to aggregated serving.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_create_worker_selection_pipeline(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    model_name_c_str: *const c_char,
    use_kv_routing: bool,
    busy_threshold: f64,
    overlap_score_weight: f64,
    router_temperature: f64,
    use_kv_events: bool,
    router_replica_sync: bool,
    enforce_disagg: bool,
    pipeline_out: *mut *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    if pipeline_out.is_null() {
        tracing::error!("pipeline_out pointer is null");
        return DynamoLlmResult::ERR;
    }

    let wk = match WK.get() {
        Some(w) => w.clone(),
        None => {
            tracing::error!("Worker not initialized. Call dynamo_llm_init first.");
            return DynamoLlmResult::ERR;
        }
    };

    let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            tracing::error!(error = ?e, "bad namespace");
            return DynamoLlmResult::ERR;
        }
    };

    let component_cow = unsafe { cstr_or_default(component_c_str, "backend") };
    if let Cow::Borrowed("backend") = &component_cow {
        tracing::info!("defaulting to \"backend\" for component");
    }
    let component: String = component_cow.into_owned();

    let model = match unsafe { CStr::from_ptr(model_name_c_str) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            tracing::error!(error = ?e, "bad model");
            return DynamoLlmResult::ERR;
        }
    };

    let make_engine = || async {
        let router_mode = if use_kv_routing {
            RouterMode::KV
        } else {
            RouterMode::RoundRobin
        };

        let kv_router_config = if use_kv_routing {
            Some(KvRouterConfig {
                overlap_score_weight,
                router_temperature,
                use_kv_events,
                router_replica_sync,
                ..KvRouterConfig::default()
            })
        } else {
            None
        };

        create_worker_selection_pipeline_chat(
            &namespace,
            &component,
            &model,
            router_mode,
            (busy_threshold >= 0.0).then_some(busy_threshold),
            kv_router_config,
            enforce_disagg,
        )
        .await
    };

    let (engine, kv_router) = match wk.runtime().secondary().block_on(make_engine()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = ?e, "create_worker_selection_pipeline_chat failed");
            return DynamoLlmResult::ERR;
        }
    };

    let handle = Box::new(WorkerSelectionPipeline {
        wk,
        engine,
        kv_router,
    });
    unsafe {
        *pipeline_out = Box::into_raw(handle);
    }
    DynamoLlmResult::OK
}

/// Query worker selection on an existing pipeline and return:
/// - `decode_worker_id_out` (`i64`): The decode worker ID (primary worker)
/// - `prefill_worker_id_out` (`i64`): The prefill worker ID (-1 if not in disaggregated mode)
/// - `token_ids_out` (heap-allocated `*mut u32`; caller must free via
///   `dynamo_free_worker_selection_result`)
/// - `token_count_out` (`usize`)
/// - `annotated_request_json_out` (`*mut c_char` to a NUL-terminated C string;
///   caller frees via the same free function)
///
/// # Safety
/// - `pipeline`
///   - Must be a **non-null** pointer previously returned by
///     `dynamo_create_worker_selection_pipeline` and not yet passed to
///     `dynamo_destroy_worker_selection_pipeline`.
///   - Must remain valid for the entire duration of this call.
///   - **Do not** call this function concurrently on the same `pipeline` pointer
///     from multiple threads unless the surrounding code guarantees synchronization.
/// - `request_json_c_str`
///   - Must be a **non-null**, **NUL-terminated** C string containing **valid UTF-8**.
///   - The JSON must represent a valid `NvCreateChatCompletionRequest`; otherwise this
///     function returns `DynamoLlmResult::ERR`.
///   - Must remain valid for the duration of this call.
/// - Output pointers:
///   - `decode_worker_id_out`, `prefill_worker_id_out`, `token_ids_out`, `token_count_out`,
///     and `annotated_request_json_out` must each be **non-null** and point to
///     writable memory for their respective types. On success, this function
///     writes to all five outputs exactly once.
///   - On **error**, outputs are left unmodified.
/// - Ownership & deallocation:
///   - On success, if there are zero tokens, `*token_ids_out` may be set to `NULL`
///     and `*token_count_out` set to `0`.
///   - If non-null, the buffer written to `*token_ids_out` is allocated with the
///     Rust global allocator and **must** be freed by calling
///     `dynamo_free_worker_selection_result` with the same `token_count_out` value.
///   - The pointer written to `*annotated_request_json_out` is a `CString` allocated
///     by Rust and **must** be freed by calling `dynamo_free_worker_selection_result`.
///   - **Do not** free these with `free(3)` or any other allocator; doing so is
///     undefined behavior.
/// - Blocking & context:
///   - This function may **block** internally while it performs async work; do not
///     call it from contexts that forbid blocking (e.g., signal handlers).
/// - Process/ABI assumptions:
///   - The caller and callee must run in the same process and use the same Rust
///     global allocator for the paired allocation/free described above.
///   - This function is not signal-safe.
///
/// # Errors
/// Returns `DynamoLlmResult::ERR` if any precondition fails (null/invalid pointers,
/// malformed UTF-8/JSON, pipeline errors, allocation failures, etc.). On error, no
/// output pointer is written.
///
/// # Output values
/// - `decode_worker_id_out`: The decode worker ID (primary worker in aggregated mode)
/// - `prefill_worker_id_out`: The prefill worker ID (only set in disaggregated mode, -1 if not present)
/// - `token_ids_out`, `token_count_out`: Token IDs and count
/// - `annotated_request_json_out`: The annotated request JSON
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_query_worker_selection_and_annotate(
    pipeline: *mut WorkerSelectionPipeline,
    request_json_c_str: *const c_char,
    decode_worker_id_out: *mut i64,
    prefill_worker_id_out: *mut i64,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    annotated_request_json_out: *mut *mut c_char,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        tracing::error!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }
    if decode_worker_id_out.is_null()
        || prefill_worker_id_out.is_null()
        || token_ids_out.is_null()
        || token_count_out.is_null()
        || annotated_request_json_out.is_null()
    {
        tracing::error!("One or more output pointers are null");
        return DynamoLlmResult::ERR;
    }

    let req_str = match unsafe { CStr::from_ptr(request_json_c_str) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = ?e, "bad request json");
            return DynamoLlmResult::ERR;
        }
    };
    let request: NvCreateChatCompletionRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(error = ?e, "parse request failed");
            return DynamoLlmResult::ERR;
        }
    };

    let pl = unsafe { &*pipeline };
    let fut = async { query_worker_selection_and_annotate(&pl.engine, request).await };
    let (result, annotated_req) = match pl.wk.runtime().secondary().block_on(fut) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = ?e, "query_worker_selection_and_annotate failed");
            return DynamoLlmResult::ERR;
        }
    };

    let tokens_ptr = if result.tokens.is_empty() {
        std::ptr::null_mut()
    } else {
        let len = result.tokens.len();
        let layout = std::alloc::Layout::array::<u32>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut u32 };
        if ptr.is_null() {
            tracing::error!("alloc tokens failed");
            return DynamoLlmResult::ERR;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(result.tokens.as_ptr(), ptr, len);
        }
        ptr
    };

    let annotated_json = match serde_json::to_string(&annotated_req) {
        Ok(s) => s,
        Err(e) => {
            if !tokens_ptr.is_null() {
                let layout = std::alloc::Layout::array::<u32>(result.tokens.len()).unwrap();
                unsafe {
                    std::alloc::dealloc(tokens_ptr as *mut u8, layout);
                }
                tracing::error!(error = ?e, "serialize annotated request failed");
            }
            return DynamoLlmResult::ERR;
        }
    };
    let cjson = match std::ffi::CString::new(annotated_json) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = ?e, "CString::new for annotated JSON failed");
            if !tokens_ptr.is_null() {
                let layout = std::alloc::Layout::array::<u32>(result.tokens.len()).unwrap();
                unsafe {
                    std::alloc::dealloc(tokens_ptr as *mut u8, layout);
                }
            }
            return DynamoLlmResult::ERR;
        }
    };
    unsafe {
        *decode_worker_id_out = result.decode_worker_id.unwrap_or(0);
        *prefill_worker_id_out = result.prefill_worker_id.unwrap_or(-1);
        *token_ids_out = tokens_ptr;
        *token_count_out = result.tokens.len();
        *annotated_request_json_out = cjson.into_raw();
    }
    DynamoLlmResult::OK
}

/// Destroy a previously created pipeline.
///
/// # Safety
/// - `pipeline`
///   - **Must** be a non-null pointer that was **originally returned by**
///     `dynamo_create_worker_selection_pipeline` (i.e., obtained via
///     `Box::into_raw` on a `WorkerSelectionPipeline`).
///   - **Must not** have been passed to this function (or otherwise freed)
///     before. Passing the same pointer twice is a **double free** and is
///     undefined behavior.
///   - **Must not** be used by any other thread while this function runs.
///     Ensure no concurrent calls are in flight that read or write through
///     this handle (e.g., `dynamo_query_worker_selection_and_annotate`).
///   - After a successful call, the pointer is **invalid** and must not be
///     dereferenced or used again in any way.
/// - Allocator/ABI
///   - The caller and callee must be in the same process and share the same
///     allocator; this function reclaims the allocation that was created by
///     Rust for the handle.
/// - Lifetime/FFI
///   - Do not call from contexts that forbid blocking or running destructors
///     (e.g., signal handlers).
///
/// # Errors
/// - Returns `DynamoLlmResult::ERR` if `pipeline` is null.
/// - On `OK`, ownership of `pipeline` is taken and the underlying resources
///   are dropped; using the pointer after return is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_destroy_worker_selection_pipeline(
    pipeline: *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        tracing::error!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }
    let _boxed: Box<WorkerSelectionPipeline> = unsafe { Box::from_raw(pipeline) };
    DynamoLlmResult::OK
}

/// Free buffers allocated by `dynamo_query_worker_selection_and_annotate`.
///
/// # Safety
/// - `token_ids` and `annotated_request_json` **must come from this library**:
///   - `token_ids` must be the exact pointer previously returned by
///     `dynamo_query_worker_selection_and_annotate` for the tokens buffer,
///     allocated with Rust’s global allocator in this process.
///   - `annotated_request_json` must be the exact pointer previously returned by
///     `CString::into_raw` inside `dynamo_query_worker_selection_and_annotate`.
/// - **Call at most once** per pointer. Passing the same pointer again is a
///   double-free and is undefined behavior.
/// - Pointer/length invariants:
///   - If `token_ids` is non-null, `token_count` **must** be the exact length
///     originally returned. Mismatched lengths cause invalid deallocation.
///   - If `token_ids` is null, `token_count` should be `0`.
///   - Passing a non-null `token_ids` with `token_count == 0` will leak in this
///     implementation (we only dealloc when `token_count > 0`).
/// - After return, the pointers are **invalid** and must not be used again.
/// - The caller and callee must be in the same process and share the same
///   allocator/ABI (these deallocations use Rust’s global allocator).
/// - Ensure no other threads are concurrently reading/writing these buffers when
///   freeing them.
/// - Do not call from contexts that forbid running destructors (e.g., signal handlers).
///
/// Returns `DynamoLlmResult::OK` on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_free_worker_selection_result(
    token_ids: *mut u32,
    token_count: usize,
    annotated_request_json: *mut c_char,
) -> DynamoLlmResult {
    if token_count > 0 {
        match std::alloc::Layout::array::<u32>(token_count) {
            Ok(layout) if !token_ids.is_null() => unsafe {
                std::alloc::dealloc(token_ids as *mut u8, layout);
            },
            _ => {}
        }
    }
    if !annotated_request_json.is_null() {
        unsafe {
            drop(std::ffi::CString::from_raw(annotated_request_json));
        }
    }
    DynamoLlmResult::OK
}

/// Default timeout for GAIE bookkeeping operations (30 seconds)
const GAIE_BOOKKEEPING_TIMEOUT_SECS: u64 = 30;

/// Helper to validate pipeline pointer and extract request_id from C string.
/// Returns `Err(DynamoLlmResult::ERR)` on validation failure, `Ok((pipeline_ref, request_id))` on success.
unsafe fn validate_pipeline_and_request_id(
    pipeline: *mut WorkerSelectionPipeline,
    request_id_c_str: *const c_char,
    operation: &str,
) -> Result<(&'static WorkerSelectionPipeline, String), DynamoLlmResult> {
    if pipeline.is_null() {
        tracing::error!("[GAIE] {} failed: pipeline pointer is null", operation);
        return Err(DynamoLlmResult::ERR);
    }

    let request_id = match unsafe { CStr::from_ptr(request_id_c_str) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            tracing::error!(error = ?e, "[GAIE] {} failed: bad request_id", operation);
            return Err(DynamoLlmResult::ERR);
        }
    };

    // SAFETY: Caller guarantees pipeline is valid for the duration of the call
    let pl: &'static WorkerSelectionPipeline = unsafe { &*pipeline };
    Ok((pl, request_id))
}

/// Helper to run an async bookkeeping operation with timeout.
/// Returns `OK` on success or timeout, `ERR` only on validation failures (handled by caller).
fn run_bookkeeping_with_timeout<F, Fut>(
    pl: &WorkerSelectionPipeline,
    operation: &'static str,
    request_id: &str,
    f: F,
) -> DynamoLlmResult
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    use std::time::Duration;

    let timeout_duration = Duration::from_secs(GAIE_BOOKKEEPING_TIMEOUT_SECS);
    let fut = f();

    let result = pl
        .wk
        .runtime()
        .secondary()
        .block_on(async { tokio::time::timeout(timeout_duration, fut).await });

    match result {
        Ok(()) => DynamoLlmResult::OK,
        Err(_elapsed) => {
            tracing::warn!(
                request_id = %request_id,
                timeout_secs = GAIE_BOOKKEEPING_TIMEOUT_SECS,
                "[GAIE] {} timed out",
                operation
            );
            // Return OK to avoid blocking the caller - the operation may still complete
            DynamoLlmResult::OK
        }
    }
}

/// Router bookkeeping functions for GAIE integration
/// Add a request to the router's bookkeeping after worker selection.
/// Call this from GAIE Stage 1 after `dynamo_query_worker_selection_and_annotate`.
///
/// This function computes the overlap_blocks internally by querying the indexer,
/// so the caller doesn't need to provide it.
///
/// # Safety
/// - `pipeline` must be a valid, non-null pointer from `dynamo_create_worker_selection_pipeline`
/// - `request_id_c_str` must be a valid NUL-terminated UTF-8 C string
/// - `token_ids` must point to at least `token_count` valid u32 values
/// - Must not be called concurrently on the same pipeline without synchronization
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_router_add_request(
    pipeline: *mut WorkerSelectionPipeline,
    request_id_c_str: *const c_char,
    token_ids: *const u32,
    token_count: usize,
    worker_id: u64,
    dp_rank: u32,
) -> DynamoLlmResult {
    let (pl, request_id) = match unsafe {
        validate_pipeline_and_request_id(pipeline, request_id_c_str, "add_request")
    } {
        Ok(v) => v,
        Err(e) => return e,
    };

    let Some(ref kv_router) = pl.kv_router else {
        tracing::debug!(
            "[GAIE] KV router not available (router_mode is not KV), skipping add_request (no-op)"
        );
        return DynamoLlmResult::OK;
    };

    // Log after kv_router check to reduce noise
    tracing::debug!(
        request_id = %request_id,
        worker_id = worker_id,
        dp_rank = dp_rank,
        token_count = token_count,
        "[GAIE] dynamo_router_add_request processing"
    );

    let tokens: Vec<u32> = if token_count > 0 && !token_ids.is_null() {
        unsafe { std::slice::from_raw_parts(token_ids, token_count) }.to_vec()
    } else {
        Vec::new()
    };

    let kv_router = kv_router.clone();
    let request_id_clone = request_id.clone();

    run_bookkeeping_with_timeout(pl, "add_request", &request_id, || async move {
        let worker = dynamo_llm::kv_router::protocols::WorkerWithDpRank::new(worker_id, dp_rank);

        // Compute overlap_blocks using the public method
        let overlap_blocks = match kv_router.get_overlap_blocks(&tokens, worker).await {
            Ok(overlap) => overlap,
            Err(e) => {
                tracing::warn!(error = ?e, "Failed to compute overlap, using 0");
                0
            }
        };

        kv_router
            .add_request(
                request_id_clone.clone(),
                &tokens,
                overlap_blocks,
                None,
                worker,
                None, // lora_name not exposed in C API yet
            )
            .await;

        tracing::debug!(
            request_id = %request_id_clone,
            worker_id = worker_id,
            dp_rank = dp_rank,
            overlap_blocks = overlap_blocks,
            token_count = tokens.len(),
            "[GAIE] dynamo_router_add_request completed - request registered in router bookkeeping"
        );
    })
}

/// Mark prefill as completed for a request.
/// Call this from the EPP extension point when the first token is generated.
///
/// # Safety
/// - `pipeline` must be a valid, non-null pointer from `dynamo_create_worker_selection_pipeline`
/// - `request_id_c_str` must be a valid NUL-terminated UTF-8 C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_router_mark_prefill_complete(
    pipeline: *mut WorkerSelectionPipeline,
    request_id_c_str: *const c_char,
) -> DynamoLlmResult {
    let (pl, request_id) = match unsafe {
        validate_pipeline_and_request_id(pipeline, request_id_c_str, "mark_prefill_complete")
    } {
        Ok(v) => v,
        Err(e) => return e,
    };

    let Some(ref kv_router) = pl.kv_router else {
        tracing::debug!(
            "[GAIE] KV router not available (router_mode is not KV), skipping mark_prefill_complete (no-op)"
        );
        return DynamoLlmResult::OK;
    };

    // Log after kv_router check to reduce noise
    tracing::debug!(
        request_id = %request_id,
        "[GAIE] dynamo_router_mark_prefill_complete processing"
    );

    let kv_router = kv_router.clone();
    let request_id_clone = request_id.clone();

    run_bookkeeping_with_timeout(pl, "mark_prefill_complete", &request_id, || async move {
        if let Err(e) = kv_router.mark_prefill_completed(&request_id_clone).await {
            tracing::warn!(
                "Failed to mark prefill completed for {}: {}",
                request_id_clone,
                e
            );
        } else {
            tracing::debug!(
                request_id = %request_id_clone,
                "[GAIE] dynamo_router_mark_prefill_complete completed - prefill tokens released"
            );
        }
    })
}

/// Free a request from the router's bookkeeping.
/// Call this from GAIE hook when the stream is closed (completed or cancelled).
///
/// # Safety
/// - `pipeline` must be a valid, non-null pointer from `dynamo_create_worker_selection_pipeline`
/// - `request_id_c_str` must be a valid NUL-terminated UTF-8 C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_router_free_request(
    pipeline: *mut WorkerSelectionPipeline,
    request_id_c_str: *const c_char,
) -> DynamoLlmResult {
    let (pl, request_id) = match unsafe {
        validate_pipeline_and_request_id(pipeline, request_id_c_str, "free_request")
    } {
        Ok(v) => v,
        Err(e) => return e,
    };

    let Some(ref kv_router) = pl.kv_router else {
        tracing::debug!(
            "[GAIE] KV router not available (router_mode is not KV), skipping free_request (no-op)"
        );
        return DynamoLlmResult::OK;
    };

    // Log after kv_router check to reduce noise
    tracing::debug!(
        request_id = %request_id,
        "[GAIE] dynamo_router_free_request processing"
    );

    let kv_router = kv_router.clone();
    let request_id_clone = request_id.clone();

    run_bookkeeping_with_timeout(pl, "free_request", &request_id, || async move {
        if let Err(e) = kv_router.free(&request_id_clone).await {
            tracing::warn!("Failed to free request {}: {}", request_id_clone, e);
        } else {
            tracing::debug!(
                request_id = %request_id_clone,
                "[GAIE] dynamo_router_free_request completed - request removed from bookkeeping"
            );
        }
    })
}

/// Result of worker selection extraction
#[derive(Debug, Clone, Default)]
pub struct WorkerSelectionResult {
    /// Decode worker ID (primary worker for aggregated, decode-only for disaggregated)
    pub decode_worker_id: Option<i64>,
    /// Prefill worker ID (only present in disaggregated mode)
    pub prefill_worker_id: Option<i64>,
    /// Token IDs from tokenization
    pub tokens: Vec<u32>,
}

/// Helper function to extract worker selection information from the annotation stream
///
/// The response format (from disaggregated_params in nvext):
/// - worker_id: {"prefill_worker_id": 123, "decode_worker_id": 456}
/// - token_ids: [1, 2, 3, ...]
pub async fn extract_worker_selection_from_stream(
    mut stream: Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
) -> anyhow::Result<WorkerSelectionResult> {
    use dynamo_llm::protocols::openai::nvext::WorkerIdInfo;
    use futures::StreamExt;

    let mut result = WorkerSelectionResult::default();

    while let Some(response) = stream.next().await {
        // Check for data in nvext (worker_id and token_ids are direct fields)
        // nvext is a serde_json::Value, so we access it as a JSON object
        if let Some(data) = &response.data
            && let Some(nvext) = &data.nvext
        {
            // Extract worker_id
            if let Some(worker_id_value) = nvext.get("worker_id")
                && let Ok(worker_info) =
                    serde_json::from_value::<WorkerIdInfo>(worker_id_value.clone())
            {
                result.decode_worker_id = worker_info.decode_worker_id.map(|id| id as i64);
                result.prefill_worker_id = worker_info.prefill_worker_id.map(|id| id as i64);
                tracing::debug!(
                    decode_worker_id = ?result.decode_worker_id,
                    prefill_worker_id = ?result.prefill_worker_id,
                    "Parsed worker_id from nvext"
                );
            }

            // Extract token_ids
            if let Some(token_ids_value) = nvext.get("token_ids")
                && let Ok(parsed_tokens) =
                    serde_json::from_value::<Vec<u32>>(token_ids_value.clone())
            {
                result.tokens = parsed_tokens;
                tracing::debug!(
                    "Successfully parsed {} tokens from nvext",
                    result.tokens.len()
                );
            }
        }
    }

    tracing::info!(
        decode_worker_id = ?result.decode_worker_id,
        prefill_worker_id = ?result.prefill_worker_id,
        token_count = result.tokens.len(),
        "Worker selection extraction complete"
    );
    Ok(result)
}

/// Utility function to add the "query_instance_id" annotation to an OpenAI request
///
/// This function modifies the request to include the annotation that signals the KV router
/// to return worker selection information (worker_fid and token_data) instead of
/// performing actual inference.
///
/// # Parameters
/// - `request`: Mutable reference to the OpenAI chat completion request
///
/// # Returns
/// The same request with the "query_instance_id" annotation added
pub fn add_query_instance_id(
    request: &mut NvCreateChatCompletionRequest,
) -> &mut NvCreateChatCompletionRequest {
    // Send empty value - router treats empty as aggregated / aggregated worker selection
    set_kv_annotation(request, "query_instance_id".to_string(), "")
}

// Note: set_worker_ids_for_stage2 and set_token_data_for_stage2 have been removed.
// The EPP now handles routing configuration via HTTP headers:
// - `x-worker-instance-id`: decode worker ID
// - `x-prefill-instance-id`: prefill worker ID (disaggregated mode only)
// - `x-enable-local-updates`: set to "false" to disable router bookkeeping
//
// Body modifications are NOT sent to the inference engine (only headers are forwarded),
// so these functions were ineffective.

/// Ensure `nvext` exists and return a mutable slice of annotations.
fn ensure_annotations(request: &mut NvCreateChatCompletionRequest) -> &mut Vec<String> {
    let nvext = request.nvext.get_or_insert_with(|| {
        NvExt::builder()
            .build()
            .expect("NvExt builder should not fail")
    });
    nvext.annotations.get_or_insert_with(Vec::new)
}

/// Set a `key:value` annotation.
fn set_kv_annotation(
    request: &mut NvCreateChatCompletionRequest,
    key: String, // <- owned, only one borrowed param remains
    value: impl Into<String>,
) -> &mut NvCreateChatCompletionRequest {
    let prefix = format!("{}:", key);
    let kv = format!("{}{}", prefix, value.into());
    let annotations = ensure_annotations(request);
    annotations.retain(|a| !a.starts_with(&prefix));
    annotations.push(kv);
    request
}

/// Wrapper function that queries worker selection for GAIE Stage 1
///
/// This function performs the complete GAIE Stage 1 flow:
/// 1. Clones the original request and adds "query_instance_id:" (empty) annotation
/// 2. Calls engine.generate() with the modified request
/// 3. Extracts worker_id info and tokens from the response stream
/// 4. Returns WorkerSelectionResult and the original request
///
/// Note: The EPP (caller) is responsible for setting HTTP headers for Stage 2:
/// - `x-worker-instance-id`: decode worker ID
/// - `x-prefill-instance-id`: prefill worker ID (disaggregated mode only)
/// - `x-enable-local-updates`: "false" to disable router bookkeeping
///
/// Body modifications are NOT forwarded to the inference engine, so this function
/// does not modify the request body.
///
/// # Parameters
/// - `engine`: The worker selection pipeline engine
/// - `original_request`: The original OpenAI request to process
///
/// # Returns
/// A tuple containing (WorkerSelectionResult, original_request)
pub async fn query_worker_selection_and_annotate(
    engine: &ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
    original_request: NvCreateChatCompletionRequest,
) -> anyhow::Result<(WorkerSelectionResult, NvCreateChatCompletionRequest)> {
    // GAIE Stage 1: Query for worker selection
    let mut query_request = original_request.clone();
    add_query_instance_id(&mut query_request);
    let single_in = SingleIn::new(query_request);
    let response_stream = engine.generate(single_in).await?;
    let result = extract_worker_selection_from_stream(response_stream).await?;

    // Return the original request unchanged.
    // The EPP sets routing headers (worker IDs, enable_local_updates) which the
    // Dynamo frontend reads via apply_header_routing_overrides().
    Ok((result, original_request))
}

/// Spawn a background task to watch for prefill models and activate prefill routers.
/// This is a lightweight watcher that only handles prefill model discovery.
fn spawn_prefill_watcher(
    drt: DistributedRuntime,
    model_manager: Arc<ModelManager>,
    target_namespace: String,
) {
    use dynamo_llm::model_card::ModelDeploymentCard;
    use dynamo_runtime::discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryQuery};
    use dynamo_runtime::protocols::EndpointId;
    use futures::StreamExt;

    tokio::spawn(async move {
        let discovery = drt.discovery();
        let mut stream = match discovery
            .list_and_watch(DiscoveryQuery::AllModels, None)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "Failed to start prefill discovery stream");
                return;
            }
        };

        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(e) => e,
                Err(e) => {
                    tracing::error!(error = %e, "Error in prefill discovery stream");
                    continue;
                }
            };

            match event {
                DiscoveryEvent::Added(instance) => {
                    let (endpoint_id, card) = match &instance {
                        DiscoveryInstance::Model {
                            namespace,
                            component,
                            endpoint,
                            ..
                        } => {
                            // Filter by namespace
                            if namespace != &target_namespace {
                                continue;
                            }

                            let eid = EndpointId {
                                namespace: namespace.clone(),
                                component: component.clone(),
                                name: endpoint.clone(),
                            };

                            match instance.deserialize_model::<ModelDeploymentCard>() {
                                Ok(card) => (eid, card),
                                Err(_) => continue,
                            }
                        }
                        _ => continue,
                    };

                    // Only handle prefill models
                    if !card.model_type.supports_prefill() {
                        continue;
                    }

                    tracing::info!(
                        model_name = card.name(),
                        "Prefill model discovered, activating prefill router"
                    );

                    // Get the endpoint and activate the prefill router
                    if let Ok(ns) = drt.namespace(&endpoint_id.namespace)
                        && let Ok(comp) = ns.component(&endpoint_id.component)
                    {
                        let endpoint = comp.endpoint(&endpoint_id.name);
                        if let Err(e) = model_manager.activate_prefill_router(card.name(), endpoint)
                        {
                            tracing::warn!(
                                model_name = card.name(),
                                error = %e,
                                "Failed to activate prefill router"
                            );
                        } else {
                            tracing::info!(
                                model_name = card.name(),
                                "Prefill router activated successfully"
                            );
                        }
                    }
                }
                DiscoveryEvent::Removed(id) => {
                    // Log removal for observability
                    // Note: The PrefillRouter remains active - worker availability
                    // is handled dynamically by the underlying Client's instance tracking
                    tracing::debug!(
                        instance_id = id.instance_id(),
                        "Prefill worker instance removed from discovery"
                    );
                }
            }
        }
    });
}

/// Create a worker selection pipeline for OpenAI Chat Completion requests
///
/// This is a concrete implementation that works specifically with NvCreateChatCompletionRequest
/// and is designed for use with C bindings. Uses the "generate" endpoint by default.
///
/// # Parameters
/// - `namespace`: namespace name
/// - `component_name`: component name
/// - `model_name`: Name/slug of the model to load
/// - `router_mode`: How to route requests (KV, RoundRobin, etc.)
/// - `busy_threshold`: Optional threshold for busy worker detection
/// - `kv_router_config`: Optional KV router configuration (only used when router_mode is KV)
/// - `enforce_disagg`: If true, fail requests when disaggregated serving is unavailable
///
/// # Returns
/// A tuple of (engine, kv_router) where kv_router is Some when router_mode is KV
pub async fn create_worker_selection_pipeline_chat(
    namespace: &str,
    component_name: &str,
    model_name: &str,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    kv_router_config: Option<KvRouterConfig>,
    enforce_disagg: bool,
) -> anyhow::Result<(
    ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
    Option<Arc<dynamo_llm::kv_router::KvRouter>>,
)> {
    use dynamo_llm::discovery::WORKER_TYPE_DECODE;
    use dynamo_llm::kv_router::PrefillRouter;

    // Use the global DRT singleton - initialize if not already done
    // Check if already initialized (by dynamo_llm_init) to avoid redundant sync wait
    let needs_sync = DRT.get().is_none();

    let distributed_runtime = DRT
        .get_or_try_init(async {
            tracing::debug!("Initializing DistributedRuntime singleton (standalone mode)");
            DistributedRuntime::from_settings(Runtime::from_settings()?).await
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize DistributedRuntime: {}", e))?;

    // Only wait for discovery sync if we just initialized the DRT
    // (dynamo_llm_init already does this when it initializes)
    // Note: This waits indefinitely - the K8s StartupProbe is the timeout mechanism.
    if needs_sync {
        wait_for_discovery_sync(distributed_runtime).await;
    }

    let component = distributed_runtime
        .namespace(namespace)?
        .component(component_name)?;
    let endpoint = component.endpoint(GENERATE_ENDPOINT);
    let client = endpoint.client().await?;

    // Discover the model card by searching all instances with this model name
    tracing::debug!("Looking for model: {}", model_name);
    tracing::debug!("Namespace: {}", namespace);

    let model_manager = Arc::new(ModelManager::new());
    let router_config = dynamo_llm::entrypoint::RouterConfig {
        router_mode,
        kv_router_config: kv_router_config.unwrap_or_default(),
        load_threshold_config: dynamo_llm::discovery::LoadThresholdConfig {
            active_decode_blocks_threshold: busy_threshold,
            active_prefill_tokens_threshold: None,
            active_prefill_tokens_threshold_frac: None,
        },
        enforce_disagg,
    };
    // Create metrics for migration tracking (not exposed via /metrics in C bindings)
    let metrics = Arc::new(Metrics::new());
    let watcher = ModelWatcher::new(
        component.drt().clone(),
        model_manager.clone(),
        router_config,
        0, // migration_limit - default to 0 for C bindings
        None,
        metrics.clone(),
    );
    let cards = watcher
        .cards_for_model(model_name, Some(namespace), false)
        .await
        .with_context(|| format!("Failed to discover model: {}", model_name))?;

    tracing::debug!("Found {} cards for model {}", cards.len(), model_name);

    let card = cards.into_iter().next().ok_or_else(|| {
        tracing::error!("No ModelDeploymentCard found for model: {}", model_name);
        anyhow::anyhow!("ModelDeploymentCard not found for model: {}", model_name)
    })?;

    let chooser = if router_mode == RouterMode::KV {
        Some(
            model_manager
                .kv_chooser_for(
                    &endpoint,
                    card.kv_cache_block_size,
                    kv_router_config,
                    WORKER_TYPE_DECODE,
                )
                .await?,
        )
    } else {
        None
    };

    // Create prefill chooser for dynamic disaggregation support
    // This registers the model and returns a receiver that will be activated
    // when a prefill worker is discovered
    let prefill_chooser = model_manager
        .register_prefill_router(model_name.to_string())
        .map(|rx| {
            // Create prefill-specific config with track_active_blocks disabled
            let mut prefill_config = kv_router_config.unwrap_or_default();
            prefill_config.router_track_active_blocks = false;

            PrefillRouter::new(
                rx,
                model_manager.clone(),
                router_mode,
                card.kv_cache_block_size,
                Some(prefill_config),
                enforce_disagg,
                model_name.to_string(),
            )
        });

    // Start background watcher for prefill model discovery
    // This will activate the prefill router when prefill workers join
    spawn_prefill_watcher(
        component.drt().clone(),
        model_manager.clone(),
        namespace.to_string(),
    );

    // Download model config files from HuggingFace for EPP
    // The backend's card has NATS URLs which aren't accessible from EPP
    tracing::debug!(
        "Downloading model config files for EPP: {}",
        card.display_name
    );

    let local_path = dynamo_llm::hub::from_hf(&card.display_name, true)
        .await
        .with_context(|| {
            format!(
                "Failed to download model config files for: {}",
                card.display_name
            )
        })?;

    // Load a fresh card from local files, then copy runtime config from original card
    tracing::debug!("Loading ModelDeploymentCard from local path...");
    let mut card_with_local_files = ModelDeploymentCard::load_from_disk(&local_path, None)
        .with_context(|| format!("Failed to load card from disk: {:?}", local_path))?;

    // Copy runtime settings from the backend's card
    tracing::debug!("Copying runtime config from backend card...");
    card_with_local_files.runtime_config = card.runtime_config.clone();
    card_with_local_files.kv_cache_block_size = card.kv_cache_block_size;
    card_with_local_files.context_length = card.context_length;

    // Load the tokenizer from the downloaded files
    tracing::debug!("Loading tokenizer from local files...");
    let hf_tokenizer = card_with_local_files
        .tokenizer_hf()
        .with_context(|| format!("Failed to load tokenizer for: {}", card.display_name))?;

    // Create worker monitor if busy_threshold is set
    // Note: C bindings don't register with ModelManager, so HTTP endpoint won't see this
    let worker_monitor = busy_threshold.map(|t| {
        KvWorkerMonitor::new(
            client.clone(),
            dynamo_llm::discovery::LoadThresholdConfig {
                active_decode_blocks_threshold: Some(t),
                active_prefill_tokens_threshold: None,
                active_prefill_tokens_threshold_frac: None,
            },
        )
    });

    // Clone chooser before passing to build_routed_pipeline (which takes ownership)
    let kv_router = chooser.clone();

    let engine = build_routed_pipeline::<
        NvCreateChatCompletionRequest,
        NvCreateChatCompletionStreamResponse,
    >(
        &card_with_local_files,
        &client,
        model_manager.clone(),
        router_mode,
        worker_monitor,
        chooser,
        hf_tokenizer,
        prefill_chooser,
        enforce_disagg,
        0, // migration_limit - default to 0 for C bindings
        metrics,
    )
    .await?;

    Ok((engine, kv_router))
}
