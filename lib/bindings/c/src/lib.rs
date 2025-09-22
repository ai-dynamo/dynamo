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

use std::panic::{AssertUnwindSafe, catch_unwind};

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

/* ---------- small helpers: panic guard + local runtime ---------- */

#[inline]
fn ffi_guard<F: FnOnce() -> DynamoLlmResult>(f: F) -> DynamoLlmResult {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Rust panic crossed FFI; returning ERR");
            DynamoLlmResult::ERR
        }
    }
}

/// Run async work on a fresh single-threaded Tokio runtime on a plain OS thread.
/// Uses `shutdown_background()` to avoid the "Cannot drop a runtime where blocking
/// is not allowed" panic during runtime drop.
fn run_on_local_runtime<T, Fut>(op: impl FnOnce() -> Fut + Send + 'static) -> Result<T, String>
where
    Fut: std::future::Future<Output = Result<T, String>> + Send + 'static,
    T: Send + 'static,
{
    std::thread::spawn(move || -> Result<T, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("build local runtime: {e:?}"))?;

        // run the future
        let out = rt.block_on(op());

        // IMPORTANT: avoid blocking shutdown on this thread
        rt.shutdown_background();

        out
    })
    .join()
    .map_err(|_| "panic in local runtime thread".to_string())?
}

/* ------------------------------ init ------------------------------ */

/// # Safety
/// the namespace_c_str and component_c_str are passed as pointers to C strings
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_llm_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    worker_id: i64,
    kv_block_size: u32,
) -> DynamoLlmResult {
    ffi_guard(|| {
        initialize_tracing();

        let wk = match WK.get_or_try_init(Worker::from_settings) {
            Ok(wk) => wk.clone(),
            Err(e) => {
                eprintln!("Failed to initialize runtime: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        // Convert C strings to owned Rust Strings before we jump threads.
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

        // Initialize DistributedRuntime on an isolated runtime/thread.
        let rt = wk.runtime().clone();
        let drt_init_res = run_on_local_runtime(move || async move {
            DRT.get_or_try_init(async { DistributedRuntime::from_settings(rt.clone()).await })
                .await
                .map(|_| ())
                .map_err(|e| format!("Failed to initialize distributed runtime: {e:?}"))
        });

        if let Err(msg) = drt_init_res {
            eprintln!("{msg}");
            return DynamoLlmResult::ERR;
        }

        // Initialize the KV publisher once.
        match KV_PUB.get_or_try_init(move || {
            dynamo_create_kv_publisher(namespace, component, worker_id, kv_block_size)
        }) {
            Ok(_) => DynamoLlmResult::OK,
            Err(e) => {
                eprintln!("Failed to create KV publisher: {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    })
}

/* ---------------------------- shutdown ---------------------------- */

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_llm_shutdown() -> DynamoLlmResult {
    ffi_guard(|| {
        let Some(wk) = WK.get().cloned() else {
            eprintln!("Runtime not initialized");
            return DynamoLlmResult::ERR;
        };
        let res = catch_unwind(AssertUnwindSafe(|| {
            // bounce to a plain thread to avoid Tokio’s guard
            let _ = std::thread::spawn(move || wk.runtime().shutdown()).join();
        }));
        if res.is_err() {
            eprintln!("Runtime shutdown panicked; ignoring");
            return DynamoLlmResult::ERR;
        }
        DynamoLlmResult::OK
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_llm_load_publisher_create() -> DynamoLlmResult {
    DynamoLlmResult::OK
}

/* ---------------------- KV publisher helpers ---------------------- */

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

/* --------------------------- KV FFI --------------------------- */

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
    ffi_guard(|| {
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
        let publisher = match KV_PUB.get() {
            Some(p) => p,
            None => {
                eprintln!("KV publisher not initialized");
                return DynamoLlmResult::ERR;
            }
        };
        let event = kv_event_create_stored_from_parts(kv_params, publisher.kv_block_size());
        match publisher.publish(event) {
            Ok(_) => DynamoLlmResult::OK,
            Err(e) => {
                eprintln!("Error publishing stored kv event {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_event_publish_removed(
    event_id: u64,
    block_ids: *const u64,
    num_blocks: usize,
) -> DynamoLlmResult {
    ffi_guard(|| {
        let publisher = match KV_PUB.get() {
            Some(p) => p,
            None => {
                eprintln!("KV publisher not initialized");
                return DynamoLlmResult::ERR;
            }
        };
        let event = kv_event_create_removed_from_parts(event_id, block_ids, num_blocks);
        match publisher.publish(event) {
            Ok(_) => DynamoLlmResult::OK,
            Err(e) => {
                eprintln!("Error publishing removed kv event {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    })
}

/* ------------------------------------------------------------- */
/* ----------------- worker selection pipeline ----------------- */
/* ------------------------------------------------------------- */
use tokio::sync::{mpsc, oneshot};

enum Cmd {
    CreatePipeline {
        namespace: String,
        component: String,
        model: String,
        use_kv_routing: bool,
        busy_threshold: Option<f64>,
        overlap_score_weight: Option<f64>,
        router_temperature: Option<f64>,
        use_kv_events: bool,
        router_replica_sync: bool,
        resp: oneshot::Sender<Result<u64, String>>, // returns a pipeline id
    },
    Query {
        pipeline_id: u64,
        request: NvCreateChatCompletionRequest,
        resp: oneshot::Sender<Result<(i64, Vec<u32>, NvCreateChatCompletionRequest), String>>,
    },
    DestroyPipeline {
        pipeline_id: u64,
        resp: oneshot::Sender<()>,
    },
}

struct Host {
    tx: mpsc::Sender<Cmd>,
}

static HOST: OnceCell<Host> = OnceCell::new();

fn ensure_host_started() -> Result<&'static Host, String> {
    HOST.get_or_try_init(|| -> Result<Host, String> {
        let (tx, mut rx) = mpsc::channel::<Cmd>(128);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("host runtime build failed");

            rt.block_on(async move {
                use std::collections::HashMap;
                use std::sync::atomic::{AtomicU64, Ordering};
                static NEXT_ID: AtomicU64 = AtomicU64::new(1);

                // Pipeline state lives inside the runtime
                struct HeldPipeline {
                    pipeline: ServiceEngine<
                        SingleIn<NvCreateChatCompletionRequest>,
                        ManyOut<Annotated<LLMEngineOutput>>,
                    >,
                }
                let mut pipelines: HashMap<u64, HeldPipeline> = HashMap::new();

                while let Some(cmd) = rx.recv().await {
                    match cmd {
                        Cmd::CreatePipeline {
                            namespace,
                            component,
                            model,
                            use_kv_routing,
                            busy_threshold,
                            overlap_score_weight,
                            router_temperature,
                            use_kv_events,
                            router_replica_sync,
                            resp,
                        } => {
                            tracing::info!(
                                target: "capi",
                                "CreatePipeline ns={:?} component={:?} model={:?} use_kv_routing={:?} busy_threshold={:?} overlap_score_weight={:?} router_temperature={:?} use_kv_events={:?} router_replica_sync={:?}",
                                namespace, component, model, use_kv_routing, busy_threshold, overlap_score_weight, router_temperature, use_kv_events, router_replica_sync
                            );
                            let fut = async {
                                let router_mode = if use_kv_routing {
                                    RouterMode::KV
                                } else {
                                    RouterMode::RoundRobin
                                };

                                let kv_router_config = if use_kv_routing {
                                    use dynamo_llm::kv_router::KvRouterConfig;
                                    Some(KvRouterConfig::new(
                                        overlap_score_weight,
                                        router_temperature,
                                        Some(use_kv_events),
                                        Some(router_replica_sync),
                                        None, // max_num_batched_tokens
                                        None, // router_snapshot_threshold
                                        None, // router_reset_states
                                    ))
                                } else {
                                    None
                                };

                                let pipeline = create_worker_selection_pipeline_chat(
                                    &namespace,
                                    &component,
                                    &model,
                                    router_mode,
                                    busy_threshold,
                                    kv_router_config,
                                )
                                .await
                                .map_err(|e| format!("{e:?}"))?;

                                let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                                pipelines.insert(id, HeldPipeline { pipeline });
                                Ok::<u64, String>(id)
                            };

                            let _ = resp.send(fut.await);
                        }

                        Cmd::Query {
                            pipeline_id,
                            request,
                            resp,
                        } => {
                            let fut = async {
                                let hp = pipelines
                                    .get(&pipeline_id)
                                    .ok_or_else(|| "invalid pipeline id".to_string())?;

                                query_worker_selection_and_annotate(&hp.pipeline, request)
                                    .await
                                    .map_err(|e| format!("{e:?}"))
                            };

                            let _ = resp.send(fut.await);
                        }

                        Cmd::DestroyPipeline { pipeline_id, resp } => {
                            pipelines.remove(&pipeline_id);
                            let _ = resp.send(());
                        }
                    }
                }
            });

            // runtime drops here when loop ends
        });

        Ok(Host { tx })
    })
}

// Worker selection pipeline handle containing the actual pipeline
pub struct WorkerSelectionPipeline {
    id: u64, // id known by the host thread
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
    busy_threshold: f64,       // negative => None
    overlap_score_weight: f64, // negative => default
    router_temperature: f64,   // negative => default
    use_kv_events: bool,
    router_replica_sync: bool,
    pipeline_out: *mut *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    ffi_guard(|| {
        if pipeline_out.is_null() {
            eprintln!("pipeline_out pointer is null");
            return DynamoLlmResult::ERR;
        }

        // start host once
        let host = match ensure_host_started() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{e}");
                return DynamoLlmResult::ERR;
            }
        };

        let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(e) => {
                eprintln!("bad namespace: {e:?}");
                return DynamoLlmResult::ERR;
            }
        };
        let component = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(e) => {
                eprintln!("bad component: {e:?}");
                return DynamoLlmResult::ERR;
            }
        };
        let model = match unsafe { CStr::from_ptr(model_name_c_str) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(e) => {
                eprintln!("bad model: {e:?}");
                return DynamoLlmResult::ERR;
            }
        };

        let (tx, rx) = oneshot::channel();
        let cmd = Cmd::CreatePipeline {
            namespace,
            component,
            model,
            use_kv_routing,
            busy_threshold: if busy_threshold < 0.0 {
                None
            } else {
                Some(busy_threshold)
            },
            overlap_score_weight: if overlap_score_weight < 0.0 {
                None
            } else {
                Some(overlap_score_weight)
            },
            router_temperature: if router_temperature < 0.0 {
                None
            } else {
                Some(router_temperature)
            },
            use_kv_events,
            router_replica_sync,
            resp: tx,
        };
        if let Err(e) = host.tx.blocking_send(cmd) {
            eprintln!("host channel closed: {e}");
            return DynamoLlmResult::ERR;
        }

        let id = match rx.blocking_recv() {
            Ok(Ok(id)) => id,
            Ok(Err(msg)) => {
                eprintln!("{msg}");
                return DynamoLlmResult::ERR;
            }
            Err(e) => {
                eprintln!("host dropped response: {e}");
                return DynamoLlmResult::ERR;
            }
        };

        let handle = Box::new(WorkerSelectionPipeline { id });
        unsafe {
            *pipeline_out = Box::into_raw(handle);
        }
        DynamoLlmResult::OK
    })
}

/// Destroy a worker selection pipeline and free its memory
///
/// # Safety
/// The pipeline pointer must be valid and should not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_destroy_worker_selection_pipeline(
    pipeline: *mut WorkerSelectionPipeline,
) -> DynamoLlmResult {
    ffi_guard(|| {
        if pipeline.is_null() {
            eprintln!("Pipeline pointer is null");
            return DynamoLlmResult::ERR;
        }
        let id = unsafe { &*pipeline }.id;
        let _boxed: Box<WorkerSelectionPipeline> = unsafe { Box::from_raw(pipeline) };

        if let Ok(host) = ensure_host_started() {
            let (tx, rx) = oneshot::channel();
            let _ = host.tx.blocking_send(Cmd::DestroyPipeline {
                pipeline_id: id,
                resp: tx,
            });
            let _ = rx.blocking_recv(); // best-effort
        }
        DynamoLlmResult::OK
    })
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
    ffi_guard(|| {
        if pipeline.is_null() {
            eprintln!("Pipeline pointer is null");
            return DynamoLlmResult::ERR;
        }
        let host = match ensure_host_started() {
            Ok(h) => h,
            Err(e) => {
                eprintln!("{e}");
                return DynamoLlmResult::ERR;
            }
        };

        let req_str = match unsafe { CStr::from_ptr(request_json_c_str) }.to_str() {
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

        let id = unsafe { &*pipeline }.id;

        let (tx, rx) = oneshot::channel();
        if let Err(e) = host.tx.blocking_send(Cmd::Query {
            pipeline_id: id,
            request,
            resp: tx,
        }) {
            eprintln!("host channel closed: {e}");
            return DynamoLlmResult::ERR;
        }

        let (worker_id, tokens, annotated_req) = match rx.blocking_recv() {
            Ok(Ok(t)) => t,
            Ok(Err(msg)) => {
                eprintln!("Failed to query worker selection: {msg}");
                return DynamoLlmResult::ERR;
            }
            Err(e) => {
                eprintln!("host dropped response: {e}");
                return DynamoLlmResult::ERR;
            }
        };

        // marshal outputs (same as your current code)
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

        let annotated_json = match serde_json::to_string(&annotated_req) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("serialize annotated req failed: {e:?}");
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
    })
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
    ffi_guard(|| {
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
    })
}
