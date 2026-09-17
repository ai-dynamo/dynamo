// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_once_cell::OnceCell as AsyncOnceCell;
use libc::c_char;
use once_cell::sync::OnceCell;
use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};

use dynamo_kv_router::protocols::*;
use dynamo_llm::kv_router::publisher::KvEventPublisher;
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

    if tracing::subscriber::set_global_default(subscriber).is_ok() {
        tracing::debug!("Tracing initialized");
    }
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
/// the namespace_c_str, component_c_str, and endpoint_c_str are passed as pointers to C strings
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_llm_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    endpoint_c_str: *const c_char,
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

    if endpoint_c_str.is_null() {
        tracing::error!("Serving endpoint name is required");
        return DynamoLlmResult::ERR;
    }
    let endpoint = match unsafe { CStr::from_ptr(endpoint_c_str) }.to_str() {
        Ok(value) if !value.trim().is_empty() => value.trim().to_string(),
        Ok(_) => {
            tracing::error!("Serving endpoint name must not be empty");
            return DynamoLlmResult::ERR;
        }
        Err(error) => {
            tracing::error!(?error, "Failed to convert serving endpoint name to UTF-8");
            return DynamoLlmResult::ERR;
        }
    };

    match result {
        Ok(_) => match KV_PUB.get_or_try_init(move || {
            dynamo_create_kv_publisher(namespace, component, endpoint, kv_block_size)
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
    endpoint: String,
    kv_block_size: u32,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!(%namespace, %component, %endpoint, "Creating endpoint-scoped KV publisher");
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(backend.endpoint(endpoint), kv_block_size, None)
        }
        Err(e) => Err(e),
    }
}

fn kv_event_create_stored_block_from_parts(
    block_hash: u64,
    token_ids: *const u32,
    num_tokens: usize,
    kv_block_size: u32,
    lora_name: Option<&str>,
) -> KvCacheStoredBlockData {
    let tokens_hash = compute_block_hash_for_seq(
        unsafe { std::slice::from_raw_parts(token_ids, num_tokens) },
        kv_block_size,
        BlockHashOptions {
            lora_name,
            cache_namespace: None,
            ..Default::default()
        },
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
            kv_params.lora_name.as_deref(),
        ));
    }

    KvCacheEvent {
        data: KvCacheEventData::Stored(KvCacheStoreData {
            blocks,
            parent_hash: kv_params.parent_hash.map(ExternalSequenceBlockHash),
            start_position: None,
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
    pub lora_name: Option<String>,
}

/// # Safety
/// parent_hash is passed as pointer to indicate whether the blocks
/// has a parent hash or not. nullptr is used to represent no parent hash.
/// lora_name is an optional null-terminated C string; pass nullptr for base model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dynamo_kv_event_publish_stored(
    event_id: u64,
    token_ids: *const u32,
    num_block_tokens: *const usize,
    block_ids: *const u64,
    num_blocks: usize,
    parent_hash: *const u64,
    lora_name: *const c_char,
) -> DynamoLlmResult {
    let parent_hash = {
        if parent_hash.is_null() {
            None
        } else {
            Some(unsafe { *parent_hash })
        }
    };
    let lora_name = if lora_name.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(lora_name) }.to_str() {
            Ok(s) => Some(s.to_owned()),
            Err(e) => {
                tracing::error!(error = ?e, "Failed to convert C string to Rust string (lora_name)");
                return DynamoLlmResult::ERR;
            }
        }
    };
    let kv_params = DynamoKvStoredEventParams {
        event_id,
        token_ids,
        num_block_tokens,
        block_ids,
        num_blocks,
        parent_hash,
        lora_name,
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
