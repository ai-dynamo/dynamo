// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_llm::entrypoint::input::worker_selection_pipeline::{
    create_worker_selection_pipeline_chat, query_worker_selection_and_annotate,
};
use dynamo_llm::kv_router::{
    indexer::compute_block_hash_for_seq, protocols::*, publisher::KvEventPublisher,
};
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use dynamo_llm::types::{Annotated, openai::chat_completions::NvCreateChatCompletionRequest};
use dynamo_runtime::{
    DistributedRuntime, Worker,
    pipeline::{ManyOut, RouterMode, ServiceEngine, SingleIn},
};
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

//
// Apis related to the best worker selection.
// Worker selection pipeline handle containing the actual pipeline
pub struct WorkerSelectionPipeline {
    pipeline:
        ServiceEngine<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<LLMEngineOutput>>>,
}

/// C FFI wrapper for creating a worker selection pipeline
///
/// Returns a pipeline handle that can be used repeatedly for queries.
/// Call dynamo_destroy_worker_selection_pipeline when done.
/// Uses the "generate" endpoint by default.
///
/// # Safety
/// The namespace_c_str, component_c_str, and model_name_c_str
/// are passed as pointers to C strings
///
/// # KV Router Configuration Parameters
/// - overlap_score_weight: Weight for KV cache overlap in worker selection (use negative value for default)
/// - router_temperature: Randomness in worker selection, 0.0 = deterministic (use negative value for default)
/// - use_kv_events: Whether to use KV cache events for tracking
/// - router_replica_sync: Whether to synchronize router state across replicas
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_create_worker_selection_pipeline(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    model_name_c_str: *const c_char,
    use_kv_routing: bool,
    busy_threshold: f64,       // Use negative value to indicate None
    overlap_score_weight: f64, // Use negative value for default
    router_temperature: f64,   // Use negative value for default
    use_kv_events: bool,
    router_replica_sync: bool,
    pipeline_out: *mut *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let rt = wk.runtime();
    let secondary = rt.secondary().clone();

    let result = secondary.block_on(async {
        // Convert C strings to Rust strings
        let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to convert namespace C string: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let component_name = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to convert component C string: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let model_name = match unsafe { CStr::from_ptr(model_name_c_str) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to convert model_name C string: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        // Determine router mode and busy threshold
        let router_mode = if use_kv_routing {
            RouterMode::KV
        } else {
            RouterMode::RoundRobin
        };

        let busy_threshold_opt = if busy_threshold < 0.0 {
            None
        } else {
            Some(busy_threshold)
        };

        // Create KV router config if using KV routing
        let kv_router_config = if use_kv_routing {
            use dynamo_llm::kv_router::KvRouterConfig;

            Some(KvRouterConfig::new(
                if overlap_score_weight < 0.0 {
                    None
                } else {
                    Some(overlap_score_weight)
                },
                if router_temperature < 0.0 {
                    None
                } else {
                    Some(router_temperature)
                },
                Some(use_kv_events),
                Some(router_replica_sync),
                None, // max_num_batched_tokens - use default
                None, // router_snapshot_threshold - use default
                None, // router_reset_states - use default
            ))
        } else {
            None
        };

        // Create the worker selection pipeline
        let pipeline = match create_worker_selection_pipeline_chat(
            namespace,
            component_name,
            model_name,
            router_mode,
            busy_threshold_opt,
            kv_router_config,
        )
        .await
        {
            Ok(pipeline) => pipeline,
            Err(e) => {
                eprintln!("Failed to create worker selection pipeline: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        // Wrap pipeline in handle struct and box it
        let pipeline_handle = WorkerSelectionPipeline { pipeline };
        let boxed_pipeline = Box::new(pipeline_handle);
        let pipeline_ptr = Box::into_raw(boxed_pipeline);

        unsafe {
            *pipeline_out = pipeline_ptr;
        }

        DynamoLlmResult::OK
    });

    result
}

/// Destroy a worker selection pipeline and free its memory
///
/// # Safety
/// The pipeline pointer must be valid and should not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_destroy_worker_selection_pipeline(
    pipeline: *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    if pipeline.is_null() {
        eprintln!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }

    // Convert back to Box with correct type and let it drop to free memory
    let _boxed_pipeline: Box<WorkerSelectionPipeline> = unsafe { Box::from_raw(pipeline) };

    DynamoLlmResult::OK
}

/// Query worker selection and return annotated request
///
/// This function takes an original OpenAI request (as JSON), adds query_instance_id annotation,
/// runs it through the pipeline to get worker selection, then returns the worker_id, tokens,
/// and the original request annotated with worker_instance_id and token_data.
///
/// # Safety
/// All pointer parameters must be valid. The request_json_c_str must be a valid C string
/// containing valid JSON. The caller is responsible for freeing the allocated memory
/// for token_ids_out and annotated_request_json_out.
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
        eprintln!("Pipeline pointer is null");
        return DynamoLlmResult::ERR;
    }

    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let rt = wk.runtime();
    let secondary = rt.secondary().clone();

    let result = secondary.block_on(async {
        // Convert C string to Rust string
        let request_json = match unsafe { CStr::from_ptr(request_json_c_str) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to convert request JSON C string: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        // Parse JSON into NvCreateChatCompletionRequest
        let original_request: NvCreateChatCompletionRequest =
            match serde_json::from_str(request_json) {
                Ok(req) => req,
                Err(e) => {
                    eprintln!("Failed to parse request JSON: {:?}", e);
                    return DynamoLlmResult::ERR;
                }
            };

        // Get pipeline reference
        let pipeline_ref = unsafe { &*pipeline };

        // Call the wrapper function
        let (worker_id, tokens, annotated_request) =
            match query_worker_selection_and_annotate(&pipeline_ref.pipeline, original_request)
                .await
            {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("Failed to query worker selection: {:?}", e);
                    return DynamoLlmResult::ERR;
                }
            };

        // Convert annotated request back to JSON
        let annotated_json = match serde_json::to_string(&annotated_request) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("Failed to serialize annotated request: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        // Allocate memory for tokens array
        let tokens_ptr = if tokens.is_empty() {
            std::ptr::null_mut()
        } else {
            let tokens_len = tokens.len();
            let layout = std::alloc::Layout::array::<u32>(tokens_len).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) as *mut u32 };
            if ptr.is_null() {
                eprintln!("Failed to allocate memory for tokens");
                return DynamoLlmResult::ERR;
            }
            // Copy tokens to allocated memory
            unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens_len) };
            ptr
        };

        // Allocate memory for annotated request JSON string
        let json_cstring = match std::ffi::CString::new(annotated_json) {
            Ok(cstr) => cstr,
            Err(e) => {
                eprintln!("Failed to create C string for annotated JSON: {:?}", e);
                if !tokens_ptr.is_null() {
                    let layout = std::alloc::Layout::array::<u32>(tokens.len()).unwrap();
                    unsafe { std::alloc::dealloc(tokens_ptr as *mut u8, layout) };
                }
                return DynamoLlmResult::ERR;
            }
        };

        // Set output parameters
        unsafe {
            *worker_instance_id_out = worker_id;
            *token_ids_out = tokens_ptr;
            *token_count_out = tokens.len();
            *annotated_request_json_out = json_cstring.into_raw();
        }

        DynamoLlmResult::OK
    });

    result
}

/// Free memory allocated by dynamo_query_worker_selection_and_annotate
///
/// # Safety
/// The pointers must have been allocated by dynamo_query_worker_selection_and_annotate
/// and should not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_free_worker_selection_result(
    token_ids: *mut u32,
    token_count: usize,
    annotated_request_json: *mut c_char,
) -> DynamoLlmResult {
    // Free tokens array if not null
    if !token_ids.is_null() && token_count > 0 {
        let layout = match std::alloc::Layout::array::<u32>(token_count) {
            Ok(layout) => layout,
            Err(_) => {
                eprintln!("Invalid layout for tokens array");
                return DynamoLlmResult::ERR;
            }
        };
        unsafe { std::alloc::dealloc(token_ids as *mut u8, layout) };
    }

    // Free JSON string if not null
    if !annotated_request_json.is_null() {
        let _cstring = unsafe { std::ffi::CString::from_raw(annotated_request_json) };
        // CString will be automatically freed when it goes out of scope
    }

    DynamoLlmResult::OK
}
