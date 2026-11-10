// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap allocations of one worker request through `Ingress::for_engine`.
//!
//! The request goes the whole worker path: `handle_payload` decodes the
//! request-plane frame, opens a TCP response stream back to a peer, calls the
//! engine and publishes its stream. The bench counts every allocation made on
//! the worker's thread from `handle_payload` until every task the request
//! spawned has ended, so late work by the response-stream tasks is not counted
//! against the next request.
//!
//! The peer that stands in for the frontend runs on another thread and is not
//! counted. Run with `cargo bench -p dynamo-runtime --bench ingress_allocations`;
//! it reports the counts and checks nothing.

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

#[path = "../tests/common/ingress.rs"]
mod ingress;
use ingress::{request_payload, start_response_peer};

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

/// Prints the per-request allocations and bytes after warm-up, as the range over
/// the measured requests.
async fn report(items: usize) {
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
    let min = counts.iter().map(|(n, _)| *n).min().unwrap();
    let max = counts.iter().map(|(n, _)| *n).max().unwrap();
    let bytes = counts.iter().map(|(_, b)| *b).sum::<u64>() / counts.len() as u64;
    println!("{items} item(s): {min}..={max} allocations, {bytes} bytes per request");
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        report(1).await;
        report(16).await;
    });
}
