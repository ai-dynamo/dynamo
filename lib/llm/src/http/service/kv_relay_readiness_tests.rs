// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise the HTTP readiness gate with real engine and router ZMQ feeds.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use dynamo_kv_router::{
    config::KvRouterConfig,
    protocols::{KV_EVENT_SUBJECT, WorkerWithDpRank},
};
use dynamo_runtime::{
    CancellationToken, Runtime,
    discovery::{DiscoveryQuery, EventSourceQuery},
    distributed::{DistributedConfig, DistributedRuntime},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait,
    },
};
use futures::SinkExt;
use serde_json::json;
use tmq::AsZmqSocket;
use tokio::sync::Mutex;

use super::service_v2::HttpService;
use crate::{
    discovery::WorkerSet,
    kv_router::publisher::{KvEventPublisher, KvEventSourceConfig},
    local_model::{register_model_card, runtime_config::SGLANG_GENERATE_CAPABILITY},
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    protocols::{
        Annotated,
        common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    },
    worker_type::WorkerType,
};

struct RequestKvEngine {
    sources: Vec<Mutex<Option<tmq::publish::Publish>>>,
    calls: AtomicUsize,
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for RequestKvEngine
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let (request, context) = request.transfer(());
        let rank = request
            .routing
            .as_ref()
            .and_then(|routing| routing.dp_rank)
            .unwrap();
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Each first request emits exactly one native KV event. Re-sending it
        // would conceal a subscription that became ready too early.
        let payload = rmp_serde::to_vec_named(&json!([
            0.0, [{"type":"BlockStored", "block_hashes":[rank as u64 + 10],
                "parent_block_hash":null, "token_ids":request.token_ids,
                "block_size":4, "lora_id":null}], rank
        ]))?;
        self.sources[rank as usize]
            .lock()
            .await
            .as_mut()
            .expect("ready rank must have an engine publisher")
            .send(vec![Vec::new(), 1_u64.to_be_bytes().to_vec(), payload])
            .await?;
        let output = LLMEngineOutput {
            token_ids: vec![42],
            engine_data: Some(json!({"sglang_response":{
                "text":"first-token", "output_ids":[42],
                "meta_info":{"finish_reason":{"type":"length","length":1}}
            }})),
            ..Default::default()
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter([Annotated::from_data(output)])),
            context.context(),
        ))
    }
}

async fn kv_relay_readiness(dp_size: u32) {
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
        .await
        .unwrap();
    let namespace = format!("kv-relay-readiness-{dp_size}");
    let endpoint = drt
        .namespace(&namespace)
        .unwrap()
        .component("backend")
        .unwrap()
        .endpoint("generate");
    let worker_id = drt.discovery().instance_id();
    let mut card = ModelDeploymentCard::with_name_only("kv-relay-model");
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.kv_cache_block_size = 4;
    card.runtime_config.data_parallel_size = dp_size;
    card.runtime_config
        .set_engine_specific(SGLANG_GENERATE_CAPABILITY, true)
        .unwrap();
    card.runtime_config
        .set_engine_specific("require_kv_event_source_readiness", true)
        .unwrap();

    // This is the leader registration that telemetry followers discover. It
    // must exist before the final rank's relay can begin starting.
    endpoint.register_endpoint_instance().await.unwrap();
    register_model_card(&endpoint, &card).await.unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let service = HttpService::builder()
        .port(port)
        .discovery(Some(drt.discovery()))
        .enable_engine_apis(true)
        .build()
        .unwrap();
    let manager = service.state().manager_clone();
    manager.require_model_readiness();
    let router = manager
        .kv_chooser_for_with_worker_role(
            &endpoint,
            4,
            Some(KvRouterConfig {
                skip_initial_worker_wait: true,
                router_event_threads: 1,
                router_track_active_blocks: false,
                ..Default::default()
            }),
            None,
            Some(WorkerType::Aggregated),
            "decode",
            Some(card.name().to_owned()),
            false,
        )
        .await
        .unwrap();
    let mut configs = manager
        .get_or_create_runtime_config_watcher(&endpoint)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !configs.borrow().contains_key(&worker_id) {
            configs.changed().await.unwrap();
        }
    })
    .await
    .expect("the leader must be discoverable without follower relays");

    let zmq_context = tmq::Context::new();
    let mut sources = Vec::new();
    let mut addresses = Vec::new();
    for rank in 0..dp_size {
        if rank == dp_size - 1 {
            let reserved = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            addresses.push(format!("tcp://{}", reserved.local_addr().unwrap()));
            sources.push(Mutex::new(None));
            // Keep the final engine publisher absent until after its relay
            // object exists, distinguishing construction from connection.
            drop(reserved);
            continue;
        }
        let source = tmq::publish::publish(&zmq_context)
            .set_linger(0)
            .bind("tcp://127.0.0.1:*")
            .unwrap();
        addresses.push(source.get_socket().get_last_endpoint().unwrap().unwrap());
        sources.push(Mutex::new(Some(source)));
    }
    let engine = Arc::new(RequestKvEngine {
        sources,
        calls: AtomicUsize::new(0),
    });
    let mut worker_set = WorkerSet::new(namespace.clone(), card.mdcsum().to_owned(), card);
    worker_set.generate_engine = Some(engine.clone());
    worker_set.kv_event_readiness = router.kv_event_readiness();
    worker_set.set_instance_watcher(router.client().instance_avail_watcher());
    assert!(manager.add_worker_set("kv-relay-model", &namespace, worker_set));
    let stop = CancellationToken::new();
    let server_stop = stop.clone();
    let http_task =
        tokio::spawn(async move { service.run_with_listener(server_stop, listener).await });
    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let mut relays = Vec::new();

    for rank in 0..dp_size {
        // For DP=2 this deliberately holds back the follower after the leader
        // is registered and its local feed is connected. DP=1 must obey the
        // same rule for its only feed.
        assert_eq!(
            client
                .get(format!("{base}/health"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(!router.kv_event_sources_ready());
        assert!(configs.borrow().contains_key(&worker_id));
        assert_eq!(
            drt.discovery()
                .list(DiscoveryQuery::Endpoint {
                    namespace: namespace.clone(),
                    component: "backend".into(),
                    endpoint: "generate".into(),
                })
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
        let rejected = client.post(format!("{base}/generate"))
            .header("x-data-parallel-rank", rank.to_string())
            .json(&json!({"input_ids":[11,12,13,14], "sampling_params":{"max_new_tokens":1}, "stream":true}))
            .send().await.unwrap();
        assert_eq!(rejected.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(engine.calls.load(Ordering::SeqCst), 0);

        let relay = KvEventPublisher::new_with_local_indexer_and_worker_id_at(
            endpoint.clone(),
            endpoint.id(),
            Some(worker_id),
            4,
            Some(KvEventSourceConfig::Zmq {
                endpoint: addresses[rank as usize].clone(),
                topic: String::new(),
                image_token_id: None,
                video_token_id: None,
            }),
            false,
            rank,
            None,
        )
        .unwrap();
        if rank == dp_size - 1 {
            assert!(
                tokio::time::timeout(Duration::from_millis(100), relay.wait_ready())
                    .await
                    .is_err(),
                "creating a relay without its engine feed must not mark it ready"
            );
            assert_eq!(
                drt.discovery()
                    .list(DiscoveryQuery::EventSources(
                        EventSourceQuery::endpoint_topic(endpoint.id(), KV_EVENT_SUBJECT)
                    ))
                    .await
                    .unwrap()
                    .len(),
                rank as usize,
                "an unconnected feed must not advertise a ready source"
            );
            assert!(configs.borrow().contains_key(&worker_id));
            assert_eq!(
                drt.discovery()
                    .list(DiscoveryQuery::Endpoint {
                        namespace: namespace.clone(),
                        component: "backend".into(),
                        endpoint: "generate".into(),
                    })
                    .await
                    .unwrap()
                    .len(),
                1
            );
            assert_eq!(
                client
                    .get(format!("{base}/health"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                reqwest::StatusCode::SERVICE_UNAVAILABLE
            );
            let source = tmq::publish::publish(&zmq_context)
                .set_linger(0)
                .bind(&addresses[rank as usize])
                .unwrap();
            *engine.sources[rank as usize].lock().await = Some(source);
        }
        tokio::time::timeout(Duration::from_secs(5), relay.wait_ready())
            .await
            .unwrap()
            .unwrap();
        relays.push(relay);
    }
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if client
                .get(format!("{base}/health"))
                .send()
                .await
                .unwrap()
                .status()
                .is_success()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all connected rank feeds must make the frontend ready");
    assert!(router.kv_event_sources_ready());

    // No warmup events, fixed sleeps, or retrying inference requests are used.
    // The first admitted request on every rank must reach the router indexer.
    for rank in 0..dp_size {
        let tokens = vec![11 + rank * 10, 12, 13, 14];
        let response = client
            .post(format!("{base}/generate"))
            .header("x-data-parallel-rank", rank.to_string())
            .json(
                &json!({"input_ids":tokens, "sampling_params":{"max_new_tokens":1}, "stream":true}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response.text().await.unwrap().contains("first-token"));
        tokio::time::timeout(Duration::from_secs(5), async {
            while router
                .get_overlap_blocks(
                    &tokens,
                    None,
                    WorkerWithDpRank::new(worker_id, rank),
                    None,
                    None,
                )
                .await
                .unwrap()
                != 1
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the first request's single KV event must reach the router");
    }
    assert_eq!(engine.calls.load(Ordering::SeqCst), dp_size as usize);
    stop.cancel();
    http_task.await.unwrap().unwrap();
    drop(relays);
    drt.shutdown();
}

#[tokio::test]
async fn frontend_waits_for_delayed_follower_and_receives_first_request_events() {
    // Single-node and leader/follower startup must obey the same contract.
    for dp_size in [1, 2] {
        kv_relay_readiness(dp_size).await;
    }
}
