// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_llm::kv_router::{
    indexer::compute_block_hash_for_seq, protocols::*, publisher::KvEventPublisher,
};
use dynamo_runtime::{DistributedRuntime, Worker, storage::key_value_store::Key};
static WK: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
// [FIXME] shouldn't the publisher be instance passing between API calls?
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();

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

/// # Safety
/// the namespace_c_str and component_c_str are passed as pointers to C strings
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_llm_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    worker_id: i64,
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
            Ok(_) => Ok(()),
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

    let component = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            tracing::error!(error = ?e, "Failed to convert C string to Rust string (component)");
            return DynamoLlmResult::ERR;
        }
    };

    match result {
        Ok(_) => match KV_PUB.get_or_try_init(move || {
            dynamo_create_kv_publisher(namespace, component, worker_id, kv_block_size)
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
    worker_id: i64,
    kv_block_size: u32,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!("Creating KV Publisher for model: {}", component);
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(backend, worker_id, kv_block_size, None)
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
    )[0];
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(block_hash),
        tokens_hash,
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
use std::{pin::Pin, sync::Arc};

const GENERATE_ENDPOINT: &str = "generate";

use anyhow::Context;
use dynamo_runtime::{
    Runtime, distributed::DistributedConfig, slug::Slug, traits::DistributedRuntimeProvider,
};

use dynamo_llm::discovery::ModelManager;
use dynamo_llm::entrypoint::build_routed_pipeline;
use dynamo_llm::kv_router::KvRouter;
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
    component::Client,
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
    let component = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            tracing::error!(error = ?e, "bad component");
            return DynamoLlmResult::ERR;
        }
    };
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
            Some(KvRouterConfig::new(
                (overlap_score_weight >= 0.0).then_some(overlap_score_weight),
                (router_temperature >= 0.0).then_some(router_temperature),
                Some(use_kv_events),
                Some(router_replica_sync),
                None,
                None,
                None,
            ))
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
        )
        .await
    };

    let engine = match wk.runtime().secondary().block_on(make_engine()) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(error = ?e, "create_worker_selection_pipeline_chat failed");
            return DynamoLlmResult::ERR;
        }
    };

    let handle = Box::new(WorkerSelectionPipeline { wk, engine });
    unsafe {
        *pipeline_out = Box::into_raw(handle);
    }
    DynamoLlmResult::OK
}

/// Query worker selection on an existing pipeline and return:
/// - `worker_instance_id_out` (`i64`)
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
///   - `worker_instance_id_out`, `token_ids_out`, `token_count_out`,
///     and `annotated_request_json_out` must each be **non-null** and point to
///     writable memory for their respective types. On success, this function
///     writes to all four outputs exactly once.
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_query_worker_selection_and_annotate(
    pipeline: *mut WorkerSelectionPipeline,
    request_json_c_str: *const c_char,
    worker_instance_id_out: *mut i64,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    annotated_request_json_out: *mut *mut c_char,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        tracing::error!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }
    if worker_instance_id_out.is_null()
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
    let (worker_id, tokens, annotated_req) = match pl.wk.runtime().secondary().block_on(fut) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = ?e, "query_worker_selection_and_annotate failed");
            return DynamoLlmResult::ERR;
        }
    };

    let tokens_ptr = if tokens.is_empty() {
        std::ptr::null_mut()
    } else {
        let len = tokens.len();
        let layout = std::alloc::Layout::array::<u32>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut u32 };
        if ptr.is_null() {
            tracing::error!("alloc tokens failed");
            return DynamoLlmResult::ERR;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, len);
        }
        ptr
    };

    let annotated_json = match serde_json::to_string(&annotated_req) {
        Ok(s) => s,
        Err(e) => {
            let layout = std::alloc::Layout::array::<u32>(tokens.len()).unwrap();
            unsafe {
                std::alloc::dealloc(tokens_ptr as *mut u8, layout);
            }
            if !tokens_ptr.is_null() {
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
                let layout = std::alloc::Layout::array::<u32>(tokens.len()).unwrap();
                unsafe {
                    std::alloc::dealloc(tokens_ptr as *mut u8, layout);
                }
            }
            return DynamoLlmResult::ERR;
        }
    };
    unsafe {
        *worker_instance_id_out = worker_id;
        *token_ids_out = tokens_ptr;
        *token_count_out = tokens.len();
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

/// Helper function to extract worker selection information from the annotation stream
pub async fn extract_worker_selection_from_stream(
    mut stream: Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
) -> anyhow::Result<(i64, Vec<u32>)> {
    use futures::StreamExt;

    let mut worker_id: i64 = 0;
    let mut tokens: Vec<u32> = Vec::new();

    while let Some(response) = stream.next().await {
        let Some(event) = &response.event else {
            tracing::error!("Response has no event field");
            continue;
        };

        match event.as_str() {
            "worker_instance_id" => {
                tracing::debug!("Found worker_instance_id event");

                let Some(first_comment) = response.comment.as_ref().and_then(|v| v.first()) else {
                    tracing::debug!("worker_instance_id event without comments");
                    continue;
                };

                // Try JSON string first (e.g. `"1732646935200805498"`), then plain integer.
                if let Ok(id_string) = serde_json::from_str::<String>(first_comment) {
                    match id_string.parse::<i64>() {
                        Ok(parsed_id) => {
                            worker_id = parsed_id;
                            tracing::debug!("parsed worker_id from JSON string: {}", worker_id);
                        }
                        Err(_) => {
                            tracing::error!(
                                "failed to parse number from JSON string: '{}'",
                                id_string
                            );
                        }
                    }
                    continue;
                }

                match first_comment.parse::<i64>() {
                    Ok(parsed_id) => {
                        worker_id = parsed_id;
                        tracing::debug!("parsed worker_id directly: {}", worker_id);
                    }
                    Err(_) => {
                        tracing::error!("failed to parse worker_id from: '{}'", first_comment);
                    }
                }
            }

            "token_data" => {
                tracing::debug!("Found token_data event");

                let Some(first_comment) = response.comment.as_ref().and_then(|v| v.first()) else {
                    tracing::debug!("token_data event without comments");
                    continue;
                };

                tracing::debug!("Token comment: '{}'", first_comment);
                match serde_json::from_str::<Vec<u32>>(first_comment) {
                    Ok(parsed_tokens) => {
                        tokens = parsed_tokens;
                        tracing::debug!("Successfully parsed {} tokens", tokens.len());
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse tokens from '{}': {}", first_comment, e);
                    }
                }
            }

            other => {
                tracing::debug!("Unknown event type: '{}'", other);
            }
        }
    }

    tracing::info!(
        "Final worker_id={}, tokens.len()={}",
        worker_id,
        tokens.len()
    );
    Ok((worker_id, tokens))
}

/// Utility function to add the "query_instance_id" annotation to an OpenAI request
///
/// This function modifies the request to include the annotation that signals the KV router
/// to return worker selection information (worker_instance_id and token_data) instead of
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
    add_annotation_unique(request, "query_instance_id")
}

/// Utility function to add worker_instance_id annotation to an OpenAI request
pub fn add_worker_instance_id_annotation(
    request: &mut NvCreateChatCompletionRequest,
    worker_id: i64,
) -> &mut NvCreateChatCompletionRequest {
    set_kv_annotation(
        request,
        "worker_instance_id".to_string(),
        worker_id.to_string(),
    )
}

/// Utility function to add token_data annotation to an OpenAI request
pub fn add_token_data_annotation<'a>(
    request: &'a mut NvCreateChatCompletionRequest,
    tokens: &[u32],
) -> &'a mut NvCreateChatCompletionRequest {
    let tokens_json = serde_json::to_string(tokens).unwrap_or_default();
    set_kv_annotation(request, "token_data".to_string(), tokens_json)
}

/// Ensure `nvext` exists and return a mutable slice of annotations.
fn ensure_annotations(request: &mut NvCreateChatCompletionRequest) -> &mut Vec<String> {
    let nvext = request.nvext.get_or_insert_with(|| {
        NvExt::builder()
            .build()
            .expect("NvExt builder should not fail")
    });
    nvext.annotations.get_or_insert_with(Vec::new)
}

/// Add a plain annotation once.
fn add_annotation_unique(
    request: &mut NvCreateChatCompletionRequest,
    annotation: impl Into<String>,
) -> &mut NvCreateChatCompletionRequest {
    let ann = annotation.into();
    let annotations = ensure_annotations(request);
    if !annotations.iter().any(|a| a == &ann) {
        annotations.push(ann);
    }
    request
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

/// Wrapper function that queries worker selection and annotates the original request
///
/// This function performs the complete flow:
/// 1. Clones the original request and adds "query_instance_id" annotation
/// 2. Calls engine.generate() with the modified request
/// 3. Extracts worker_instance_id and tokens from the response stream
/// 4. Adds worker_instance_id and token_data annotations to the original request
/// 5. Returns (worker_id, tokens, annotated_original_request)
///
/// # Parameters
/// - `engine`: The worker selection pipeline engine
/// - `original_request`: The original OpenAI request to process
///
/// # Returns
/// A tuple containing (worker_instance_id, tokens, modified_original_request)
/// where the modified_original_request has worker_instance_id and token_data annotations added
pub async fn query_worker_selection_and_annotate(
    engine: &ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
    mut original_request: NvCreateChatCompletionRequest,
) -> anyhow::Result<(i64, Vec<u32>, NvCreateChatCompletionRequest)> {
    let mut query_request = original_request.clone();
    add_query_instance_id(&mut query_request);
    let single_in = SingleIn::new(query_request);
    let response_stream = engine.generate(single_in).await?;
    let (worker_id, tokens) = extract_worker_selection_from_stream(response_stream).await?;
    add_worker_instance_id_annotation(&mut original_request, worker_id);
    add_token_data_annotation(&mut original_request, &tokens);

    Ok((worker_id, tokens, original_request))
}

/// Build a worker selection pipeline
/// The router handles query_instance_id annotations and returns worker_instance_id and token_data annotations.
pub async fn build_worker_selection_pipeline_chat(
    card: &ModelDeploymentCard,
    client: &Client,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    chooser: Option<Arc<KvRouter>>,
) -> anyhow::Result<
    ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
> {
    let hf_tokenizer = card
        .tokenizer_hf()
        .with_context(|| "Failed to load HF tokenizer")?;
    let engine = build_routed_pipeline::<
        NvCreateChatCompletionRequest,
        NvCreateChatCompletionStreamResponse,
    >(
        card,
        client,
        router_mode,
        busy_threshold,
        chooser,
        hf_tokenizer,
        None,
    )
    .await?;

    Ok(engine)
}

/// Helper function to create worker selection pipeline for OpenAI Chat Completion requests
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
///
/// # Returns
/// A configured worker selection pipeline ready to use
pub async fn create_worker_selection_pipeline_chat(
    namespace: &str,
    component_name: &str,
    model_name: &str,
    router_mode: RouterMode,
    busy_threshold: Option<f64>,
    kv_router_config: Option<KvRouterConfig>,
) -> anyhow::Result<
    ServiceEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
    >,
> {
    let runtime = Runtime::from_settings()?;
    let dst_config = DistributedConfig::from_settings(false);
    let drt_owned = DistributedRuntime::new(runtime, dst_config).await?;
    let distributed_runtime: &'static DistributedRuntime = Box::leak(Box::new(drt_owned));

    let component = distributed_runtime
        .namespace(namespace)?
        .component(component_name)?;
    let client = component.endpoint(GENERATE_ENDPOINT).client().await?;

    let model_slug = Slug::from_string(model_name);
    let card = match ModelDeploymentCard::load_from_store(
        &Key::from_raw(model_slug.to_string()),
        component.drt(),
    )
    .await
    {
        Ok(Some(card)) => card,
        Ok(None) => anyhow::bail!("ModelDeploymentCard not found for model: {}", model_name),
        Err(err) => anyhow::bail!(
            "Error fetching ModelDeploymentCard from storage under key {model_slug}. {err}"
        ),
    };

    let chooser = if router_mode == RouterMode::KV {
        let model_manager = std::sync::Arc::new(ModelManager::new());
        Some(
            model_manager
                .kv_chooser_for(&component, card.kv_cache_block_size, kv_router_config)
                .await?,
        )
    } else {
        None
    };

    build_worker_selection_pipeline_chat(&card, &client, router_mode, busy_threshold, chooser).await
}
