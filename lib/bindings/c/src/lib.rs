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
    model_card::{ModelDeploymentCard, ROOT_PATH as MDC_ROOT_PATH},
    preprocessor::OpenAIPreprocessor,
};
use dynamo_runtime::{
    DistributedRuntime, Worker,
    slug::Slug,
    storage::key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
};
use std::sync::Arc;
static WK: OnceCell<Worker> = OnceCell::new();
static DRT: AsyncOnceCell<DistributedRuntime> = AsyncOnceCell::new();
// [FIXME] shouldn't the publisher be instance passing between API calls?
static KV_PUB: OnceCell<KvEventPublisher> = OnceCell::new();
static KV_ROUTER: AsyncOnceCell<KvRouter> = AsyncOnceCell::new();
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
    )
}

// KV Router per-request override structure for C API
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DynamoRouterConfigOverride {
    pub has_overlap_score_weight: bool,
    pub overlap_score_weight: f64,
    pub has_router_temperature: bool,
    pub router_temperature: f64,
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
/// Initialize the KV router with configuration
/// This must be called after dynamo_llm_init() and before any router operations
#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_router_init_with_config(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    kv_block_size: u32,
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

    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized - call dynamo_llm_init first");
            return DynamoLlmResult::ERR;
        }
    };

    let result = wk.runtime().secondary().block_on(async {
        let drt = match DRT.get() {
            Some(drt) => drt,
            None => {
                eprintln!("Distributed runtime not initialized");
                return DynamoLlmResult::ERR;
            }
        };

        let namespace_obj = match drt.namespace(namespace) {
            Ok(ns) => ns,
            Err(e) => {
                eprintln!("Failed to get namespace: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let component = match namespace_obj.component(component_name.clone()) {
            Ok(comp) => comp,
            Err(e) => {
                eprintln!("Failed to get component: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let kv_router_config = if config.is_null() {
            KvRouterConfig::default()
        } else {
            let config = unsafe { *config };
            convert_to_kv_router_config(config)
        };

        // Check if router is already initialized
        if KV_ROUTER.get().is_some() {
            eprintln!("KV Router already initialized");
            return DynamoLlmResult::ERR;
        }

        // Try to create the router
        match KvRouter::new(
            component.clone(),
            kv_block_size,
            None,
            Some(kv_router_config),
        )
        .await
        {
            Ok(router) => {
                // Try to initialize the static router
                match KV_ROUTER.get_or_init(async move { router }).await {
                    _router_ref => {
                        // Router successfully stored, now initialize preprocessor
                        // Load MDC from etcd to create preprocessor
                        let Some(etcd_client) = drt.etcd_client() else {
                            eprintln!("No etcd client available for loading MDC");
                            return DynamoLlmResult::ERR;
                        };

                        let kvstore: Box<dyn KeyValueStore> =
                            Box::new(EtcdStorage::new(etcd_client.clone()));
                        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
                        let card_key = Slug::from_string(&component_name);

                        match card_store
                            .load::<ModelDeploymentCard>(MDC_ROOT_PATH, &card_key)
                            .await
                        {
                            Ok(Some(mut mdc)) => {
                                // Download any remote files in the MDC
                                if let Err(e) = mdc.move_from_nats(drt.nats_client().clone()).await
                                {
                                    eprintln!("Failed to download MDC files: {:?}", e);
                                    return DynamoLlmResult::ERR;
                                }

                                // Create preprocessor
                                match OpenAIPreprocessor::new(mdc).await {
                                    Ok(preprocessor) => {
                                        match PREPROCESSOR
                                            .get_or_init(async move { preprocessor })
                                            .await
                                        {
                                            _preprocessor_ref => DynamoLlmResult::OK,
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to create preprocessor: {:?}", e);
                                        DynamoLlmResult::ERR
                                    }
                                }
                            }
                            Ok(None) => {
                                eprintln!(
                                    "No ModelDeploymentCard found for component: {}",
                                    component_name
                                );
                                DynamoLlmResult::ERR
                            }
                            Err(e) => {
                                eprintln!("Failed to load ModelDeploymentCard: {:?}", e);
                                DynamoLlmResult::ERR
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create KV router: {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    });

    result
}

/// Initialize the KV router with default configuration (backward compatibility)
/// This must be called after dynamo_llm_init() and before any router operations
#[unsafe(no_mangle)]
pub extern "C" fn dynamo_kv_router_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    kv_block_size: u32,
) -> DynamoLlmResult {
    dynamo_kv_router_init_with_config(
        namespace_c_str,
        component_c_str,
        kv_block_size,
        std::ptr::null(),
    )
}

// Below are the bindings used by the Inference Gateway Endpoint Picker ort any other client which only needs routing.
// The EPP workflow
// func routeRequest(contextID, prompt string) (int64, error) {
//     // Get routing decision
//     workerID, tokensPtr, tokenCount := dynamo_kv_router_query_instance_id(contextID, prompt)

//     // Convert to Go slice for use
//     tokens := convertCArrayToGoSlice(tokensPtr, tokenCount)

//     // Free C memory immediately after copying
//     dynamo_kv_router_free_tokens(tokensPtr)  // ← Essential!

//     // Use the Go slice
//     return sendToWorker(workerID, tokens)
// }

/// Query the best worker instance for a given prompt (stateless probing)
/// This function takes a text prompt, tokenizes it using the same tokenizer as Dynamo,
/// and returns the best worker instance ID along with the tokenized prompt.
///
/// This is a pure query operation that automatically cleans up after itself,
/// making it suitable for probing/routing decisions without lifecycle management.
///
/// Equivalent to: curl -d '{"prompt": "...", "nvext": {"annotations": ["query_instance_id"]}}'
///
/// Returns the worker_instance_id and tokenized prompt
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

    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized");
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

        // First, get or initialize the preprocessor
        let preprocessor = match PREPROCESSOR.get() {
            Some(preprocessor) => preprocessor,
            None => {
                eprintln!("Preprocessor not initialized - call dynamo_kv_router_init first");
                return DynamoLlmResult::ERR;
            }
        };

        // Tokenize the prompt
        let encoding = match preprocessor.tokenize(prompt) {
            Ok(encoding) => encoding,
            Err(e) => {
                eprintln!("Failed to tokenize prompt: {:?}", e);
                return DynamoLlmResult::ERR;
            }
        };

        let tokens = encoding.token_ids();
        let num_tokens = tokens.len();

        // This replicates the exact logic from the if query_instance_id block (no config override):
        match router.find_best_match(context_id, tokens, None).await {
            Ok((instance_id, _overlap_amount)) => {
                // Return worker_instance_id
                unsafe {
                    *worker_instance_id_out = instance_id;
                }

                // Return the tokens (copy them to C-managed memory)
                let tokens_copy = unsafe { libc::malloc(num_tokens * std::mem::size_of::<u32>()) } as *mut u32;
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
                    "Tokens requested in the response through the query_instance_id annotation: {:?}",
                    tokens
                );

                // Auto-cleanup: Free the request since this is just a query/probe
                router.free(context_id).await;

                DynamoLlmResult::OK
            }
            Err(e) => {
                eprintln!("Failed to find best match: {:?}", e);
                DynamoLlmResult::ERR
            }
        }
    });

    result
}

/// Query the best worker instance for a given prompt with configuration overrides
/// This function accepts per-request configuration overrides for fine-tuned routing
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

    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized");
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

        // First, get or initialize the preprocessor
        let preprocessor = match PREPROCESSOR.get() {
            Some(preprocessor) => preprocessor,
            None => {
                eprintln!("Preprocessor not initialized - call dynamo_kv_router_init first");
                return DynamoLlmResult::ERR;
            }
        };

        // Tokenize the prompt
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

        // Find best match with optional config override
        match router.find_best_match(context_id, tokens, router_config_override.as_ref()).await {
            Ok((instance_id, _overlap_amount)) => {
                // Return worker_instance_id
                unsafe {
                    *worker_instance_id_out = instance_id;
                }

                // Return the tokens (copy them to C-managed memory)
                let tokens_copy = unsafe { libc::malloc(num_tokens * std::mem::size_of::<u32>()) } as *mut u32;
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
                    "Tokens requested in the response through the query_instance_id annotation: {:?}",
                    tokens
                );

                // Auto-cleanup: Free the request since this is just a query/probe
                router.free(context_id).await;

                DynamoLlmResult::OK
            }
            Err(e) => {
                eprintln!("Failed to find best match: {:?}", e);
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
