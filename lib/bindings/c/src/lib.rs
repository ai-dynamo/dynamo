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

use dynemo_llm::kv_router::{
    indexer::compute_block_hash_for_seq, protocols::*, publisher::KvEventPublisher,
};
use dynemo_runtime::{DistributedRuntime, Worker};
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
pub enum TritonLlmResult {
    OK = 0,
    ERR = 1,
}

/// # Safety
/// the namespace_c_str and component_c_str are passed as pointers to C strings
#[no_mangle]
pub unsafe extern "C" fn triton_llm_init(
    namespace_c_str: *const c_char,
    component_c_str: *const c_char,
    worker_id: i64,
) -> TritonLlmResult {
    initialize_tracing();
    let wk = match WK.get_or_try_init(Worker::from_settings) {
        Ok(wk) => wk.clone(),
        Err(e) => {
            eprintln!("Failed to initialize runtime: {:?}", e);
            return TritonLlmResult::ERR;
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
                Err(TritonLlmResult::ERR)
            }
        }
    });
    let namespace = match unsafe { CStr::from_ptr(namespace_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return TritonLlmResult::ERR;
        }
    };

    let component = match unsafe { CStr::from_ptr(component_c_str) }.to_str() {
        Ok(s) => s.to_string(),
        Err(e) => {
            eprintln!("Failed to convert C string to Rust string: {:?}", e);
            return TritonLlmResult::ERR;
        }
    };

    match result {
        Ok(_) => match KV_PUB
            .get_or_try_init(move || triton_create_kv_publisher(namespace, component, worker_id))
        {
            Ok(_) => TritonLlmResult::OK,
            Err(e) => {
                eprintln!("Failed to initialize distributed runtime: {:?}", e);
                TritonLlmResult::ERR
            }
        },
        Err(e) => e,
    }
}

#[no_mangle]
pub extern "C" fn triton_llm_shutdown() -> TritonLlmResult {
    let wk = match WK.get() {
        Some(wk) => wk,
        None => {
            eprintln!("Runtime not initialized");
            return TritonLlmResult::ERR;
        }
    };

    wk.runtime().shutdown();

    TritonLlmResult::OK
}

#[no_mangle]
pub extern "C" fn triton_llm_load_publisher_create() -> TritonLlmResult {
    TritonLlmResult::OK
}

// instantiate a kv publisher
// this will bring up the task to publish and the channels to await publishing events
// the [`triton_kv_publish_store_event`] call will use a handle to the publisher to send events
// store and the [`triton_kv_event_create_removed`] will create remove events
// these call mus be driving by external c++ threads that are consuming the kv events from the
// c++ executor api

fn triton_create_kv_publisher(
    namespace: String,
    component: String,
    worker_id: i64,
) -> Result<KvEventPublisher, anyhow::Error> {
    tracing::info!("Creating KV Publisher for model: {}", component);
    match DRT
        .get()
        .ok_or(anyhow::Error::msg("Could not get Distributed Runtime"))
    {
        Ok(drt) => {
            let backend = drt.namespace(namespace)?.component(component)?;
            KvEventPublisher::new(drt.clone(), backend, worker_id)
        }
        Err(e) => Err(e),
    }
}

fn kv_event_create_stored_block_from_parts(
    block_hash: u64,
    token_ids: *const u32,
    num_tokens: usize,
    _lora_id: u64,
) -> KvCacheStoredBlockData {
    let tokens_hash =
        compute_block_hash_for_seq(unsafe { std::slice::from_raw_parts(token_ids, num_tokens) })[0];
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash(block_hash),
        tokens_hash,
    }
}
static WARN_COUNT: AtomicU32 = AtomicU32::new(0);

fn kv_event_create_stored_from_parts(
    event_id: u64,
    token_ids: *const u32,
    num_block_tokens: *const usize,
    block_ids: *const u64,
    num_blocks: usize,
    parent_hash: Option<u64>,
    lora_id: u64,
) -> KvCacheEvent {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    for block_idx in 0..num_blocks {
        let block_hash = unsafe { *block_ids.offset(block_idx.try_into().unwrap()) };
        let tokens = unsafe { token_ids.offset(token_offset.try_into().unwrap()) };
        let num_toks = unsafe { *num_block_tokens.offset(block_idx.try_into().unwrap()) };
        // compute hash only apply to full block (KV_BLOCK_SIZE token)
        if num_toks != 64 {
            if WARN_COUNT
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
                    if c < 3 {
                        Some(c + 1)
                    } else {
                        None
                    }
                })
                .is_ok()
            {
                tracing::warn!(
                    "Block size must be 64 tokens to be published. Block size is: {}",
                    num_toks
                );
            }
            break;
        }
        token_offset += num_toks;
        blocks.push(kv_event_create_stored_block_from_parts(
            block_hash, tokens, num_toks, lora_id,
        ));
    }

    KvCacheEvent {
        data: KvCacheEventData::Stored(KvCacheStoreData {
            blocks,
            parent_hash: parent_hash.map(ExternalSequenceBlockHash),
        }),
        event_id,
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

/// # Safety
/// parent_hash is passed as pointer to indicate whether the blocks
/// has a parent hash or not. nullptr is used to represent no parent hash
#[no_mangle]
pub unsafe extern "C" fn triton_kv_event_publish_stored(
    event_id: u64,
    token_ids: *const u32,
    num_block_tokens: *const usize,
    block_ids: *const u64,
    num_blocks: usize,
    parent_hash: *const u64,
    lora_id: u64,
) -> TritonLlmResult {
    let publisher = KV_PUB.get().unwrap();
    let parent_hash = {
        if parent_hash.is_null() {
            None
        } else {
            Some(unsafe { *parent_hash })
        }
    };
    let event = kv_event_create_stored_from_parts(
        event_id,
        token_ids,
        num_block_tokens,
        block_ids,
        num_blocks,
        parent_hash,
        lora_id,
    );
    match publisher.publish(event) {
        Ok(_) => TritonLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing stored kv event {:?}", e);
            TritonLlmResult::ERR
        }
    }
}

#[no_mangle]
pub extern "C" fn triton_kv_event_publish_removed(
    event_id: u64,
    block_ids: *const u64,
    num_blocks: usize,
) -> TritonLlmResult {
    let publisher = KV_PUB.get().unwrap();
    let event = kv_event_create_removed_from_parts(event_id, block_ids, num_blocks);
    match publisher.publish(event) {
        Ok(_) => TritonLlmResult::OK,
        Err(e) => {
            eprintln!("Error publishing removed kv event {:?}", e);
            TritonLlmResult::ERR
        }
    }
}

// #[no_mangle]
// pub extern "C" fn triton_kv_publish_store_event(
//     event_id: u64,
//     token_ids: *const u32,
//     num_tokens: usize,
//     lora_id: u64,
// ) -> TritonLlmResult {
//     // if event.is_null() || token_ids.is_null() {
//     //     return tritonKvErrorType::INVALID_TOKEN_IDS;
//     // }

//     // let tokens = unsafe { std::slice::from_raw_parts(token_ids, num_tokens) }.to_vec();
//     // let new_event = Box::new(KvCacheStoreData {
//     //     event_id,
//     //     lora_id,
//     //     token_ids: tokens,
//     //     block_hashes: Vec::new(),
//     // });

//     // unsafe { *event = Box::into_raw(new_event) };

//     TritonLlmResult::OK
// }

// #[no_mangle]
// pub extern "C" fn triton_kv_event_create_removed(
//     event_id: u64,
//     block_hashes: *const u64,
//     num_hashes: usize,
// ) -> TritonLlmResult {
//     // if event.is_null() || block_hashes.is_null() {
//     //     return -1;
//     // }

//     // let hashes = unsafe { std::slice::from_raw_parts(block_hashes, num_hashes) }.to_vec();
//     // let new_event = Box::new(KvCacheRemoveData {
//     //     event_id,
//     //     lora_id: 0,
//     //     token_ids: Vec::new(),
//     //     block_hashes: hashes,
//     // });

//     // unsafe { *event = Box::into_raw(new_event) };
//     // 0
//     TritonLlmResult::OK
// }

// /// create load publisher object and return a handle
// /// load publisher will instantiate the nats service and tie its stats handler to
// /// a watch channel receiver.  the watch channel sender will be attach to the
// /// handle and calls to [`triton_load_stats_publish`] issue the stats to the watch t
// pub extern "C" fn triton_load_publisher_create() -> *mut LoadPublisher {
//     // let publisher = Box::new(LoadPublisher::new());
//     // Box::into_raw(publisher)
// }

// pub extern "C" fn triton_load_stats_publish(
//     publisher: *mut LoadPublisher,
//     active_slots: u64,
//     total_slots: u64,
//     active_kv: u64,
//     total_kv: u64,
// ) {
//     // let publisher = unsafe { &mut *publisher };
// }
