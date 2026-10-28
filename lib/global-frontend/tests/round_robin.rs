// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-robin forwarding over a catalog controlled by the test.
mod common;

use common::{MODEL, chat, chat_chunk, pool, post, scripted_pool_frontend};
use dynamo_global_frontend::PoolSnapshot;
use reqwest::StatusCode;
use std::sync::Arc;
use tokio::sync::watch;

#[tokio::test]
async fn rotates_across_pools_and_tracks_catalog_changes() {
    let blue_sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("blue"));
    let green_sse = format!("data: {}\n\ndata: [DONE]\n\n", chat_chunk("green"));
    let (blue, blue_requests) = scripted_pool_frontend(blue_sse.leak()).await;
    let (green, green_requests) = scripted_pool_frontend(green_sse.leak()).await;
    let blue_pool = pool("blue", MODEL, &blue.url);
    let green_pool = pool("green", MODEL, &green.url);
    let (catalog, receiver) = watch::channel(Arc::new(PoolSnapshot::new(vec![blue_pool.clone()])));
    let global = common::global_frontend(receiver).await;

    async fn send(global: &common::Server, count: usize) {
        for _ in 0..count {
            let response = post(&global.url, "v1/chat/completions", &chat(MODEL, "hi", true)).await;
            assert_eq!(response.status(), StatusCode::OK);
            response.text().await.unwrap();
        }
    }
    send(&global, 1).await;
    catalog.send_replace(Arc::new(PoolSnapshot::new(vec![
        blue_pool,
        green_pool.clone(),
    ])));
    send(&global, 4).await;
    assert_eq!(blue_requests.lock().unwrap().len(), 3);
    assert_eq!(green_requests.lock().unwrap().len(), 2);

    catalog.send_replace(Arc::new(PoolSnapshot::new(vec![green_pool])));
    send(&global, 2).await;
    assert_eq!(blue_requests.lock().unwrap().len(), 3);
    assert_eq!(green_requests.lock().unwrap().len(), 4);
}
