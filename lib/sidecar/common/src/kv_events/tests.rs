// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_kv_events::zmq_wire::RawKvEvent;
use dynamo_llm::kv_router::publisher::KvEventSourceConfig;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::stream;
use tokio::sync::mpsc;
use tokio::time::Instant;

use super::*;

#[tokio::test]
async fn subscription_failures_choose_zmq_or_no_feed() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let failing = |code| {
        let attempts = attempts.clone();
        (0..2).map(move |rank| {
            let attempts = attempts.clone();
            let subscribe: Subscribe = Box::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Box::pin(future::ready(Err(tonic::Status::new(code, "injected"))))
            });
            (rank, subscribe)
        })
    };
    let zmq = || {
        let (endpoint, topic) = Default::default();
        Ok(vec![(0, KvEventFeed::Zmq { endpoint, topic })])
    };

    // An engine without the KvEvents service: its ZMQ publishers.
    let feeds = kv_event_feeds(failing(tonic::Code::Unimplemented), zmq).await;
    assert!(matches!(feeds.unwrap()[..], [(0, KvEventFeed::Zmq { .. })]));

    // A stream it lists but cannot serve: one attempt, no ZMQ fallback, no feed.
    attempts.store(0, Ordering::SeqCst);
    let no_fallback = || unreachable!("KV events are off");
    let feeds = kv_event_feeds(failing(tonic::Code::FailedPrecondition), no_fallback).await;
    assert!(feeds.unwrap().is_empty());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn an_ended_stream_is_reopened_at_once_unless_it_delivered_nothing() {
    let drt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = drt.namespace("kv-resubscribe").unwrap();
    let endpoint = endpoint.component("worker").unwrap().endpoint("generate");
    let config = KvEventSourceConfig::Push {
        idempotent_stores: false,
    };
    let publisher =
        KvEventPublisher::new_with_local_indexer(endpoint, 4, Some(config), false, 0, None);
    let publisher = Arc::new(publisher.unwrap());

    let batch = |sequence| -> KvBatchStream {
        let batch = KvEventBatch {
            ts: 0.0,
            events: vec![RawKvEvent::AllBlocksCleared { ownership: None }],
            data_parallel_rank: Some(0),
        };
        stream::iter([Ok((sequence, batch))]).boxed()
    };
    // The reopened streams: one batch, then nothing, then one that stays open.
    let mut reopened = [batch(1), stream::empty().boxed(), stream::pending().boxed()].into_iter();
    let (opened, mut opened_at) = mpsc::unbounded_channel();
    let source = SidecarKvSource {
        dp_rank: 0,
        image_token_id: None,
        idempotent_stores: false,
        feed: KvEventFeed::Grpc {
            stream: batch(0),
            resubscribe: Box::new(move || {
                opened.send(Instant::now()).unwrap();
                Box::pin(future::ready(Ok(reopened.next().unwrap())))
            }),
        },
    };
    let cancel = CancellationToken::new();
    let KvEventSource::Push { on_ready, .. } = source.into_push(cancel.clone()) else {
        unreachable!("sidecar sources are push sources");
    };
    on_ready(publisher.clone()).unwrap();

    let mut opened = async || {
        let opened = tokio::time::timeout(Duration::from_secs(60), opened_at.recv());
        opened.await.expect("the stream was not reopened").unwrap()
    };
    let (first, second, third) = (opened().await, opened().await, opened().await);
    // Each stream that delivered a batch is reopened at once; the empty one after the delay.
    assert_eq!(second - first, Duration::ZERO);
    assert_eq!(third - second, RESUBSCRIBE_DELAY);
    // Both batches were translated: the initial stream's and the reopened stream's.
    assert_eq!(publisher.next_event_id(), 2);
    cancel.cancel();
    drt.shutdown();
}
