// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations of one worker request through `Ingress::for_engine`.
//!
//! The request goes the whole worker path: `handle_payload` decodes the
//! request-plane frame, opens a TCP response stream back to a peer, calls the
//! engine and publishes its stream. The test counts every allocation made on
//! the worker's thread from `handle_payload` until every task the request
//! spawned has ended, so late work by the response-stream tasks is not counted
//! against the next request.
//!
//! The peer that stands in for the frontend runs on another thread and is not
//! counted. The test fails when a count rises above its ceiling.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use anyhow::{Error, Result};
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{
        ManyOut, ResponseStream, SingleIn,
        network::{Ingress, PushWorkHandler},
    },
    protocols::annotated::Annotated,
};

mod common;
use common::ingress::{request_payload, start_response_peer};

/// Allocations per request for a one-item and a 16-item response stream. These
/// are ceilings, set to the counts measured on this code; lower them when the
/// worker path gets cheaper.
const MAX_ALLOCATIONS_ONE_ITEM: u64 = 67;
const MAX_ALLOCATIONS_SIXTEEN_ITEMS: u64 = 127;

const WARMUP_REQUESTS: usize = 20;
const MEASURED_REQUESTS: usize = 200;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static BYTES: Cell<u64> = const { Cell::new(0) };
}

fn record(size: usize) {
    // `try_with` because the allocator also runs while thread locals are torn down.
    let _ = COUNTING.try_with(|counting| {
        if counting.get() {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
            let _ = BYTES.try_with(|n| n.set(n.get() + size as u64));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocations and bytes counted on this thread while `COUNTING` was set.
fn take_counts() -> (u64, u64) {
    (ALLOCATIONS.replace(0), BYTES.replace(0))
}

/// Returns its request's text `items` times.
struct EchoEngine {
    items: usize,
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for EchoEngine {
    async fn generate(&self, request: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (text, context) = request.into_parts();
        let items = std::iter::repeat_n(text, self.items).map(Annotated::from_data);
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            context.context(),
        ))
    }
}

/// Runs one request and returns its allocations and bytes, including the
/// response-stream tasks it spawned.
async fn measure_one(ingress: &dyn PushWorkHandler, payload: Bytes, id: String) -> (u64, u64) {
    let metrics = tokio::runtime::Handle::current().metrics();
    let idle_tasks = metrics.num_alive_tasks();
    take_counts();
    COUNTING.set(true);
    ingress
        .handle_payload(payload, Some(id))
        .await
        .expect("worker request failed");
    // The reader, writer and monitor tasks end once the peer closes its side.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while metrics.num_alive_tasks() > idle_tasks {
        assert!(
            std::time::Instant::now() < deadline,
            "response-stream tasks did not end"
        );
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    COUNTING.set(false);
    take_counts()
}

/// Per-request allocations and bytes, after warm-up. Every measured request
/// must cost the same, or the count is not a property of the request.
async fn allocations_per_request(items: usize) -> (u64, u64) {
    let address = start_response_peer();
    let ingress = Ingress::for_engine(Arc::new(EchoEngine { items })).unwrap();
    let mut counts = Vec::with_capacity(MEASURED_REQUESTS);
    for n in 0..WARMUP_REQUESTS + MEASURED_REQUESTS {
        let id = format!("request-{n:06}");
        let payload = request_payload(&address, &id, &"hello");
        let count = measure_one(ingress.as_ref(), payload, id).await;
        if n >= WARMUP_REQUESTS {
            counts.push(count);
        }
    }
    let allocations: Vec<u64> = counts.iter().map(|(n, _)| *n).collect();
    let min = *allocations.iter().min().unwrap();
    let max = *allocations.iter().max().unwrap();
    let bytes = counts.iter().map(|(_, b)| *b).sum::<u64>() / counts.len() as u64;
    eprintln!("{items} item(s): {min}..={max} allocations, {bytes} bytes per request");
    assert_eq!(min, max, "allocations per request vary: {allocations:?}");
    (min, bytes)
}

#[tokio::test]
async fn worker_request_allocations() {
    let (one, _) = allocations_per_request(1).await;
    let (sixteen, _) = allocations_per_request(16).await;
    assert!(
        one <= MAX_ALLOCATIONS_ONE_ITEM,
        "one-item worker request: {one} allocations, ceiling {MAX_ALLOCATIONS_ONE_ITEM}"
    );
    assert!(
        sixteen <= MAX_ALLOCATIONS_SIXTEEN_ITEMS,
        "16-item worker request: {sixteen} allocations, ceiling {MAX_ALLOCATIONS_SIXTEEN_ITEMS}"
    );
}
