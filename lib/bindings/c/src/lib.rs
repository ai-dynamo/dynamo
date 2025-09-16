// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_llm::kv_router::{
    KvRouter, KvRouterConfig, RouterConfigOverride, indexer::compute_block_hash_for_seq,
    protocols::*, publisher::KvEventPublisher,
};
use dynamo_llm::{
    discovery::{MODEL_ROOT_PATH, ModelEntry},
    preprocessor::OpenAIPreprocessor,
};
use dynamo_runtime::{DistributedRuntime, Worker};
use std::sync::Arc;
static WORKER: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
// [FIXME] shouldn't the publisher be instance passing between API calls?
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();
static KV_ROUTER: AsyncOnceCell<KvRouter> = AsyncOnceCell::new();
// Component-based tokenization (no manual etcd operations)
static COMPONENT: AsyncOnceCell<dynamo_runtime::component::Component> = AsyncOnceCell::new();
static PREPROCESSOR: AsyncOnceCell<Arc<OpenAIPreprocessor>> = AsyncOnceCell::new();

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
#[derive(Debug)]
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
    let wk = match WORKER.get_or_try_init(Worker::from_settings) {
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
    let wk = match WORKER.get() {
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

// KV Router configuration structure for C API
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DynamoKvRouterConfig {
    pub overlap_score_weight: f64,
    pub router_temperature: f64,
    pub use_kv_events: bool,
    pub router_replica_sync: bool,
    pub max_num_batched_tokens: u32,
}

impl Default for DynamoKvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.0,
            use_kv_events: true,
            router_replica_sync: false,
            max_num_batched_tokens: 8192,
        }
    }
}

// Helper function to convert DynamoKvRouterConfig to KvRouterConfig
fn convert_to_kv_router_config(config: DynamoKvRouterConfig) -> KvRouterConfig {
    KvRouterConfig::new(
        Some(config.overlap_score_weight),
        Some(config.router_temperature),
        Some(config.use_kv_events),
        Some(config.router_replica_sync),
        Some(config.max_num_batched_tokens),
        Some(Some(10000)), // router_snapshot_threshold - use default
        Some(false),       // router_reset_states - use default
    )
}

// KV Router per-request override structure for C API
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DynamoRouterConfigOverride {
    pub overlap_score_weight: f64,
    pub router_temperature: f64,
    pub has_overlap_score_weight: bool,
    pub has_router_temperature: bool,
}

// Helper function to convert DynamoRouterConfigOverride to RouterConfigOverride
fn convert_to_router_config_override(config: DynamoRouterConfigOverride) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_weight: if config.has_overlap_score_weight {
            Some(config.overlap_score_weight)
        } else {
            None
        },
        router_temperature: if config.has_router_temperature {
            Some(config.router_temperature)
        } else {
            None
        },
    }
}

// KV Router functions
/// Initialize the KV router using the standard Component abstraction
/// This leverages Dynamo's built-in discovery system.
/// Block size is auto-discovered, consumer_uuid is auto-generated
/// This must be called after dynamo_llm_init() and before any router operations
#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_router_init_with_config(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    config: *const DynamoKvRouterConfig,
) -> DynamoLlmResult {
    let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert namespace C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let component_name = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert component C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let wk = match WORKER.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let result = wk.runtime().secondary().block_on(async {
        eprintln!(
            "Starting standard KV router initialization for namespace='{}' component='{}'...",
            namespace, component_name
        );

        let drt = match DRT.get() {
            Some(drt) => drt,
            None => {
                eprintln!("Distributed runtime not initialized");
                return DynamoLlmResult::ERR;
            }
        };

        // Create Component object.
        let component = match drt
            .namespace(namespace.clone())
            .and_then(|ns| ns.component(component_name.clone()))
        {
            Ok(comp) => comp,
            Err(e) => {
                eprintln!(
                    "Failed to get component {}/{}: {:?}",
                    namespace, component_name, e
                );
                return DynamoLlmResult::ERR;
            }
        };

        // Store component for later use in tokenization (no manual etcd needed)
        if let Err(_) = COMPONENT
            .get_or_try_init(async { Ok::<_, anyhow::Error>(component.clone()) })
            .await
        {
            eprintln!("Warning: Component already stored");
        }

        let kv_router_config = if config.is_null() {
            None
        } else {
            let config = unsafe { *config };
            Some(convert_to_kv_router_config(config))
        };

        if KV_ROUTER.get().is_some() {
            eprintln!("KV Router already initialized");
            return DynamoLlmResult::ERR;
        }

        // Generate consumer UUID (matches standard router behavior)
        let consumer_uuid = uuid::Uuid::new_v4().to_string();
        eprintln!("Generated consumer UUID: {}", consumer_uuid);

        // Use standard KvRouter::new() - this handles:
        // - Worker discovery via Component abstraction
        // - KV block size auto-discovery
        // - Model configuration loading
        // - NATS event subscription
        // - All the complex initialization logic
        match KvRouter::new(
            component.clone(),
            16,   // Temporary block size - KvRouter will discover the actual one
            None, // Default worker selector
            kv_router_config,
            consumer_uuid,
        )
        .await
        {
            Ok(router) => {
                let discovered_block_size = router.block_size();
                eprintln!(
                    "KV Router created with discovered block size: {}",
                    discovered_block_size
                );

                match KV_ROUTER.get_or_init(async move { router }).await {
                    _ => {
                        // Note: Preprocessor will be created on-demand by query functions
                        eprintln!("Router created with standard discovery system!");
                        eprintln!("Preprocessor will be initialized on first query if needed");

                        eprintln!("Standard KV Router initialization completed successfully!");
                        DynamoLlmResult::OK
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create standard KV router: {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    });

    match result {
        DynamoLlmResult::OK => {
            eprintln!("KV router initialization completed successfully!");
            DynamoLlmResult::OK
        }
        err => {
            eprintln!("KV router initialization failed with result: {:?}", err);
            err
        }
    }
}

// Helper function to initialize preprocessor using Component discovery (no manual etcd operations)
async fn initialize_preprocessor_if_needed() -> anyhow::Result<Arc<OpenAIPreprocessor>> {
    if let Some(preprocessor) = PREPROCESSOR.get() {
        return Ok(preprocessor.clone());
    }

    let component = COMPONENT.get().ok_or_else(|| {
        anyhow::anyhow!("Component not initialized - call dynamo_kv_router_init first")
    })?;

    let drt = DRT
        .get()
        .ok_or_else(|| anyhow::anyhow!("DRT not initialized"))?;

    // Use Component's built-in discovery instead of manual etcd operations
    let Some(etcd_client) = drt.etcd_client() else {
        anyhow::bail!("No etcd client available through DRT");
    };

    // Use the Component's namespace and component info to find the model
    let namespace = component.namespace().name();
    let component_name = component.name();

    // Minimal discovery - let Component handle the complex parts
    let kvs = etcd_client.kv_get_prefix(MODEL_ROOT_PATH).await?;
    let mut matching_entry: Option<ModelEntry> = None;

    for kv in kvs {
        if let Ok(model_entry) = serde_json::from_slice::<ModelEntry>(kv.value()) {
            if model_entry.endpoint_id.namespace == namespace
                && model_entry.endpoint_id.component == component_name
            {
                matching_entry = Some(model_entry);
                break;
            }
        }
    }

    let model_entry = matching_entry.ok_or_else(|| {
        anyhow::anyhow!("No ModelEntry found for {}/{}", namespace, component_name)
    })?;

    // Load MDC using the Component's discovery system
    let mut mdc = model_entry.load_mdc(&etcd_client).await?;
    let _temp_dir = mdc.move_from_nats(drt.nats_client().clone()).await?;

    // Store for future use
    let preprocessor = PREPROCESSOR
        .get_or_try_init(async {
            let preprocessor = OpenAIPreprocessor::new(mdc)?;
            Ok::<_, anyhow::Error>(preprocessor)
        })
        .await?;

    Ok(preprocessor.clone())
}

/// Initialize the KV router with default configuration
/// Block size is auto-discovered from the worker's ModelDeploymentCard
/// This must be called after dynamo_llm_init() and before any router operations
#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_router_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
) -> DynamoLlmResult {
    dynamo_kv_router_init_with_config(namespace_c_str, component_c_str, std::ptr::null())
}

// Below are the bindings used by the Inference Gateway Endpoint Picker when it needs routing.
// The EPP workflow

// // Uses FFI to get (worker_instance_id, tokens) in-process.
// // Use this as an alternative to making the call over HTTP with the k.callFrontEndForWorker
// func (k *KVAwareScorer) callDynamoRouter(
// 	ctx context.Context,
// 	req *schedtypes.LLMRequest,
// ) (string, []int64, error) {
// 	logger := log.FromContext(ctx)

// 	if err := initFFI(); err != nil {
// 		logger.V(logutil.DEFAULT).Error(err, "FFI init failed")
// 		return "", nil, err
// 	}

// 	// contextID should be unique per request. If your framework has one, use it.
// 	// Otherwise fall back to a deterministic hash or a random UUID.
// 	contextID := req.RequestId
// 	if contextID == "" {
// 		contextID = "gaie-epp" // TODO: replace with your request-id/trace-id
// 	}
// 	prompt := req.Prompt

// 	cCtx := C.CString(contextID)
// 	cPrm := C.CString(prompt)
// 	defer C.free(unsafe.Pointer(cCtx))
// 	defer C.free(unsafe.Pointer(cPrm))

// 	var cWorker C.longlong
// 	var cTokens *C.uint
// 	var cCount C.ulong // uintptr_t in header; maps to C.ulong here

// 	cfg := defaultRouterOverride()
// 	rc := C.dynamo_kv_router_query_instance_id_with_config(
// 		cCtx,
// 		cPrm,
// 		cfg,
// 		&cWorker,
// 		&cTokens,
// 		&cCount,
// 	)
// 	if rc != C.OK {
// 		return "", nil, fmt.Errorf("dynamo_kv_router_query_instance_id failed")
// 	}

// 	// Copy tokens into Go memory then free C memory immediately
// 	count := int(uintptr(cCount))
// 	var tokens64 []int64
// 	if count > 0 && cTokens != nil {
// 		src := unsafe.Slice((*uint32)(unsafe.Pointer(cTokens)), count)
// 		tokens64 = make([]int64, count)
// 		for i := 0; i < count; i++ {
// 			tokens64[i] = int64(src[i])
// 		}
// 		C.dynamo_kv_router_free_tokens((*C.uint)(cTokens))
// 	}

// 	workerID := fmt.Sprintf("%d", int64(cWorker))
// 	return workerID, tokens64, nil
// }

/// Query worker instance using Component-based KV routing (no manual etcd operations)
///
/// This function uses the initialized KV router and Component-based tokenization
/// to find the optimal worker instance for the given prompt.
///
/// The Component abstraction handles all discovery automatically - no manual etcd needed!
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_kv_router_query_instance_id(
    context_id_c_str: *const c_char,
    prompt_c_str: *const c_char,
    worker_instance_id_out: *mut i64,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
) -> DynamoLlmResult {
    let context_id = match unsafe { CStr::from_ptr(context_id_c_str) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to convert context_id C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let prompt = match unsafe { CStr::from_ptr(prompt_c_str) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to convert prompt C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let wk = match WORKER.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let result = wk.runtime().secondary().block_on(async {
        let router = match KV_ROUTER.get() {
            Some(router) => router,
            None => {
                eprintln!("KV Router not initialized - call dynamo_kv_router_init first");
                return DynamoLlmResult::ERR;
            }
        };

        // Use Component-based tokenization (no manual etcd operations)
        let preprocessor = match initialize_preprocessor_if_needed().await {
            Ok(preprocessor) => preprocessor,
            Err(e) => {
                eprintln!(
                    "Failed to initialize preprocessor using Component discovery: {:?}",
                    e
                );
                return DynamoLlmResult::ERR;
            }
        };

        // Tokenize the prompt using Component-discovered tokenizer
        let encoding = match preprocessor.tokenize(prompt) {
            Ok(encoding) => encoding,
            Err(e) => {
                eprintln!("Failed to tokenize prompt: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let tokens = encoding.token_ids();
        let num_tokens = tokens.len();

        // Use the Component-based router to find best match
        match router
            .find_best_match(context_id, tokens, None, false)
            .await
        {
            Ok((instance_id, _overlap_amount)) => {
                // Return worker_instance_id
                unsafe {
                    *worker_instance_id_out = instance_id;
                }

                // Return the tokens (copy them to C-managed memory)
                let tokens_copy =
                    unsafe { libc::malloc(num_tokens * std::mem::size_of::<u32>()) } as *mut u32;
                if tokens_copy.is_null() {
                    eprintln!("Failed to allocate memory for tokens");
                    return DynamoLlmResult::ERR;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(tokens.as_ptr(), tokens_copy, num_tokens);
                    *token_ids_out = tokens_copy;
                    *token_count_out = num_tokens;
                }

                tracing::trace!(
                    "Component-based routing: worker_id={}, tokens={:?}",
                    instance_id,
                    tokens
                );

                // Auto-cleanup: Free the request since this is just a query/probe
                router.free(context_id).await;

                DynamoLlmResult::OK
            }
            Err(e) => {
                eprintln!("Failed to find best match using Component router: {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    });

    result
}

/// Query worker instance with config overrides using Component-based KV routing
///
/// This function uses the initialized KV router and Component-based tokenization
/// with optional configuration overrides to find the optimal worker instance.
///
/// Configuration overrides are applied directly to the router without HTTP headers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_kv_router_query_instance_id_with_config(
    context_id_c_str: *const c_char,
    prompt_c_str: *const c_char,
    config_override: *const DynamoRouterConfigOverride,
    worker_instance_id_out: *mut i64,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
) -> DynamoLlmResult {
    let context_id = match unsafe { CStr::from_ptr(context_id_c_str) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to convert context_id C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let prompt = match unsafe { CStr::from_ptr(prompt_c_str) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to convert prompt C string: {:?}", e);
            return DynamoLlmResult::ERR;
        }
    };

    let wk = match WORKER.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let result = wk.runtime().secondary().block_on(async {
        let router = match KV_ROUTER.get() {
            Some(router) => router,
            None => {
                eprintln!("KV Router not initialized - call dynamo_kv_router_init first");
                return DynamoLlmResult::ERR;
            }
        };

        // Use Component-based tokenization (no manual etcd operations)
        let preprocessor = match initialize_preprocessor_if_needed().await {
            Ok(preprocessor) => preprocessor,
            Err(e) => {
                eprintln!(
                    "Failed to initialize preprocessor using Component discovery: {:?}",
                    e
                );
                return DynamoLlmResult::ERR;
            }
        };

        // Tokenize the prompt using Component-discovered tokenizer
        let encoding = match preprocessor.tokenize(prompt) {
            Ok(encoding) => encoding,
            Err(e) => {
                eprintln!("Failed to tokenize prompt: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let tokens = encoding.token_ids();
        let num_tokens = tokens.len();

        // Convert C config override to Rust type
        let router_config_override = if config_override.is_null() {
            None
        } else {
            let config = unsafe { *config_override };
            Some(convert_to_router_config_override(config))
        };

        // Use the Component-based router to find best match with config override
        match router
            .find_best_match(context_id, tokens, router_config_override.as_ref(), false)
            .await
        {
            Ok((instance_id, _overlap_amount)) => {
                // Return worker_instance_id
                unsafe {
                    *worker_instance_id_out = instance_id;
                }

                // Return the tokens (copy them to C-managed memory)
                let tokens_copy =
                    unsafe { libc::malloc(num_tokens * std::mem::size_of::<u32>()) } as *mut u32;
                if tokens_copy.is_null() {
                    eprintln!("Failed to allocate memory for tokens");
                    return DynamoLlmResult::ERR;
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(tokens.as_ptr(), tokens_copy, num_tokens);
                    *token_ids_out = tokens_copy;
                    *token_count_out = num_tokens;
                }

                tracing::trace!(
                    "Component-based routing with config: worker_id={}, tokens={:?}",
                    instance_id,
                    tokens
                );

                // Auto-cleanup: Free the request since this is just a query/probe
                router.free(context_id).await;

                DynamoLlmResult::OK
            }
            Err(e) => {
                eprintln!(
                    "Failed to find best match using Component router with config: {:?}",
                    e
                );
                DynamoLlmResult::ERR
            }
        }
    });

    result
}

/// Free the token array allocated by dynamo_kv_router_query_instance_id
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_kv_router_free_tokens(tokens_ptr: *mut u32) {
    if !tokens_ptr.is_null() {
        unsafe {
            libc::free(tokens_ptr as *mut libc::c_void);
        }
    }
}
