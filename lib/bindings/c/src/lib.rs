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
use dynamo_runtime::{DistributedRuntime, Worker};
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
            eprintln!("Failed to initialize runtime: {:?}", e);
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
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
                Err(DynamoLlmResult::ERR)
            }
        }
    });
    let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let component = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    // TODO rm
    if let Some(drt) = DRT.get() {
        match drt.namespace(namespace.clone()) {
            Ok(ns) => {
                match ns.component(component.clone()) {
                    Ok(comp) => {
                        eprintln!(
                            "!!! drt: component resolved: {}/{}",
                            comp.namespace().name(),
                            comp.name()
                        );
                        // NOTE: we can't list endpoints here; use a preflight call (below) to prove wiring.
                    }
                    Err(e) => {
                        eprintln!(
                            "!!! drt: missing component {}/{}: {:?}",
                            namespace, component, e
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("!!! drt: missing namespace {}: {:?}", namespace, e);
            }
        }
    }
    //end of TODO

    match result {
        Ok(_) => match KV_PUB.get_or_try_init(move || {
            dynamo_create_kv_publisher(namespace, component, worker_id, kv_block_size)
        }) {
            Ok(_) => DynamoLlmResult::OK,
            Err(e) => {
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
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
            eprintln!("Runtime not initialized");
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

use dynamo_llm::entrypoint::input::worker_selection_pipeline::{
    create_worker_selection_pipeline_chat, query_worker_selection_and_annotate,
};
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::types::{Annotated, openai::chat_completions::NvCreateChatCompletionRequest};
use dynamo_runtime::pipeline::{ManyOut, RouterMode, ServiceEngine, SingleIn};

// Opaque handle exposed to C — contains the engine directly.
// You will pass around a *mut WorkerSelectionPipeline allocated via Box.
pub struct WorkerSelectionPipeline {
    engine:
        ServiceEngine<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
}

#[unsafe(no_mangle)]
/// Create a worker-selection pipeline ("generate" endpoint) and return an opaque Box handle.
///
/// # Safety
/// - All `*_c_str` must be valid, nul-terminated C strings.
/// - `pipeline_out` must be a valid non-null pointer to receive the handle.
pub unsafe extern "C" fn dynamo_create_worker_selection_pipeline(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    model_name_c_str: *const c_char,
    use_kv_routing: bool,
    busy_threshold: f64,       // negative => None
    overlap_score_weight: f64, // negative => default
    router_temperature: f64,   // negative => default
    use_kv_events: bool,
    router_replica_sync: bool,
    pipeline_out: *mut *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    if pipeline_out.is_null() {
        eprintln!("pipeline_out pointer is null");
        return DynamoLlmResult::ERR;
    }

    // Ensure WK/DRT were already initialized by dynamo_llm_init
    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Worker not initialized. Call dynamo_llm_init first.");
            return DynamoLlmResult::ERR;
        }
    };
    if DRT.get().is_none() {
        eprintln!("DistributedRuntime not initialized. Call dynamo_llm_init first.");
        return DynamoLlmResult::ERR;
    }

    // Parse inputs
    let namespace = match CStr::from_ptr(namespace_c_str).to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            eprintln!("bad namespace: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };
    let component = match CStr::from_ptr(component_c_str).to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            eprintln!("bad component: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };
    let model = match CStr::from_ptr(model_name_c_str).to_str() {
        Ok(s) => s.to_owned(),
        Err(e) => {
            eprintln!("bad model: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };

    // Build on the same worker runtime (secondary handle)
    let make_pipeline = || async {
        // Optional env override: DYNAMO_FORCE_RR=1
        let force_rr = std::env::var("DYNAMO_FORCE_RR").ok().as_deref() == Some("1");
        let router_mode = if force_rr {
            RouterMode::RoundRobin
        } else if use_kv_routing {
            RouterMode::KV
        } else {
            RouterMode::RoundRobin
        };
        eprintln!("!!! router_mode={:?} (force_rr={})", router_mode, force_rr);

        let kv_router_config = if use_kv_routing {
            use dynamo_llm::kv_router::KvRouterConfig;
            Some(KvRouterConfig::new(
                (overlap_score_weight >= 0.0).then_some(overlap_score_weight),
                (router_temperature >= 0.0).then_some(router_temperature),
                Some(use_kv_events),
                Some(router_replica_sync),
                None, // max_num_batched_tokens (default)
                None, // router_snapshot_threshold (default)
                None, // router_reset_states (default)
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

    let engine = match wk.runtime().secondary().block_on(make_pipeline()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("create_worker_selection_pipeline_chat failed: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };

    // Box the engine inside our opaque handle and return it
    let handle = Box::new(WorkerSelectionPipeline { engine });
    eprintln!(
        "!!! pipeline created; engine at {:p}",
        &handle.engine as *const _
    );

    unsafe { *pipeline_out = Box::into_raw(handle) };
    DynamoLlmResult::OK
}

#[unsafe(no_mangle)]
/// Destroy a previously created pipeline (drops the Box).
///
/// # Safety
/// - `pipeline` must be a pointer previously returned by `dynamo_create_worker_selection_pipeline`.
pub unsafe extern "C" fn dynamo_destroy_worker_selection_pipeline(
    pipeline: *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        eprintln!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }

    // Re-box to drop
    let boxed: Box<WorkerSelectionPipeline> = unsafe { Box::from_raw(pipeline) };
    eprintln!(
        "!!! pipeline destroy; engine at {:p}",
        &boxed.engine as *const _
    );
    drop(boxed);
    DynamoLlmResult::OK
}

#[unsafe(no_mangle)]
/// Query worker selection on an existing pipeline and return:
/// - worker_instance_id_out (i64)
/// - token_ids_out (malloc'ed array, caller must free via `dynamo_free_worker_selection_result`)
/// - token_count_out
/// - annotated_request_json_out (CString*, caller frees via same free fn)
///
/// # Safety
/// - `pipeline` must be a pointer previously returned by `dynamo_create_worker_selection_pipeline`.
/// - `request_json_c_str` must be a valid, nul-terminated C string containing valid JSON.
pub unsafe extern "C" fn dynamo_query_worker_selection_and_annotate(
    pipeline: *mut WorkerSelectionPipeline,
    request_json_c_str: *const c_char,
    worker_instance_id_out: *mut i64,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    annotated_request_json_out: *mut *mut c_char,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        eprintln!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }
    if worker_instance_id_out.is_null()
        || token_ids_out.is_null()
        || token_count_out.is_null()
        || annotated_request_json_out.is_null()
    {
        eprintln!("One or more output pointers are null");
        return DynamoLlmResult::ERR;
    }

    // Ensure WK/DRT initialized by your init
    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Worker not initialized. Call dynamo_llm_init first.");
            return DynamoLlmResult::ERR;
        }
    };
    if DRT.get().is_none() {
        eprintln!("DistributedRuntime not initialized. Call dynamo_llm_init first.");
        return DynamoLlmResult::ERR;
    }

    // Parse request JSON
    let req_str = match CStr::from_ptr(request_json_c_str).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("bad request json: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };
    let request: NvCreateChatCompletionRequest = match serde_json::from_str(req_str) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("parse request failed: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };

    // Borrow the engine from the boxed handle
    let handle = unsafe { &*pipeline };
    let engine_ref = &handle.engine;
    eprintln!("!!! pipeline query; engine at {:p}", engine_ref as *const _);

    // Execute on the same worker runtime (secondary)
    let fut = async { query_worker_selection_and_annotate(engine_ref, request).await };

    let (worker_id, tokens, annotated_req) = match wk.runtime().secondary().block_on(fut) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("query_worker_selection_and_annotate failed: {e:?}");
            return DynamoLlmResult::ERR;
        }
    };

    // Marshal tokens to heap memory (C-side will free)
    let tokens_ptr = if tokens.is_empty() {
        std::ptr::null_mut()
    } else {
        let len = tokens.len();
        let layout = std::alloc::Layout::array::<u32>(len).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut u32 };
        if ptr.is_null() {
            eprintln!("alloc tokens failed");
            return DynamoLlmResult::ERR;
        }
        unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, len) };
        ptr
    };

    // Serialize annotated request JSON
    let annotated_json = match serde_json::to_string(&annotated_req) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("serialize annotated req failed: {e:?}");
            if !tokens_ptr.is_null() {
                let layout = std::alloc::Layout::array::<u32>(tokens.len()).unwrap();
                unsafe { std::alloc::dealloc(tokens_ptr as *mut u8, layout) };
            }
            return DynamoLlmResult::ERR;
        }
    };

    let cjson = match std::ffi::CString::new(annotated_json) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cstr annotated failed: {e:?}");
            if !tokens_ptr.is_null() {
                let layout = std::alloc::Layout::array::<u32>(tokens.len()).unwrap();
                unsafe { std::alloc::dealloc(tokens_ptr as *mut u8, layout) };
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

#[unsafe(no_mangle)]
/// Free buffers allocated by `dynamo_query_worker_selection_and_annotate`.
///
/// # Safety
/// - `token_ids`/`annotated_request_json` must be pointers returned by the query function.
pub unsafe extern "C" fn dynamo_free_worker_selection_result(
    token_ids: *mut u32,
    token_count: usize,
    annotated_request_json: *mut c_char,
) -> DynamoLlmResult {
    if !token_ids.is_null() && token_count > 0 {
        if let Ok(layout) = std::alloc::Layout::array::<u32>(token_count) {
            unsafe { std::alloc::dealloc(token_ids as *mut u8, layout) };
        }
    }
    if !annotated_request_json.is_null() {
        let _ = unsafe { std::ffi::CString::from_raw(annotated_request_json) };
    }
    DynamoLlmResult::OK
}
