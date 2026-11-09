// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::sequence::{RuntimeSequenceSubscriber, SequenceSubscriber};
use dynamo_lora::LoadEstimator;
use dynamo_runtime::{component::Endpoint, traits::DistributedRuntimeProvider};
use std::{sync::Arc, time::Duration};

pub trait LoadEstimatorSubscription {
    fn start_event_subscription(self: Arc<Self>, endpoint: Endpoint)
    -> tokio::task::JoinHandle<()>;
}

impl LoadEstimatorSubscription for LoadEstimator {
    fn start_event_subscription(
        self: Arc<Self>,
        endpoint: Endpoint,
    ) -> tokio::task::JoinHandle<()> {
        let cancel_token = endpoint.drt().child_token();
        tokio::spawn(async move {
            // Durable feed: reconnect on transient errors / stream end with capped backoff,
            // stopping only on cancellation. A failed subscribe must not silently disable KV
            // load tracking for the lifetime of the process.
            let mut backoff = Duration::from_secs(1);
            while !cancel_token.is_cancelled() {
                match subscribe_to_events(&self, &endpoint, &cancel_token).await {
                    Ok(()) => break, // cancelled cleanly
                    Err(e) => {
                        tracing::warn!(
                            "LORA load event subscription error: {e}; reconnecting in {backoff:?}"
                        );
                        tokio::select! {
                            _ = cancel_token.cancelled() => break,
                            _ = tokio::time::sleep(backoff) => {}
                        }
                        backoff = (backoff * 2).min(Duration::from_secs(30));
                    }
                }
            }
            tracing::debug!("LORA load event subscription task exiting");
        })
    }
}

async fn subscribe_to_events(
    estimator: &LoadEstimator,
    endpoint: &Endpoint,
    cancel_token: &tokio_util::sync::CancellationToken,
) -> anyhow::Result<()> {
    let mut subscriber = RuntimeSequenceSubscriber::for_endpoint(endpoint).await?;

    tracing::info!("Started LORA load event subscription");

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => return Ok(()),
            result = subscriber.next_event() => {
                match result {
                    Some(Ok(event)) => {
                        estimator.handle_event(event);
                    }
                    Some(Err(e)) => {
                        tracing::warn!("Error receiving LORA load event: {}", e);
                    }
                    None => anyhow::bail!("LORA load event stream ended"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::ACTIVE_SEQUENCES_SUBJECT;
    use dynamo_kv_router::protocols::{
        ActiveSequenceEvent, ActiveSequenceEventBatch, ActiveSequenceEventData, WorkerWithDpRank,
    };
    use dynamo_runtime::config::environment_names::zmq_broker as broker_env;
    use dynamo_runtime::distributed::DistributedConfig;
    use dynamo_runtime::transports::event_plane::EventPublisher;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    fn lora_event(
        request_id: &str,
        lora_name: Option<&str>,
        data: ActiveSequenceEventData,
    ) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.to_string(),
            worker: WorkerWithDpRank::new(1, 0),
            data,
            router_id: 7,
            lora_name: lora_name.map(str::to_string),
        }
    }

    fn add_request() -> ActiveSequenceEventData {
        ActiveSequenceEventData::AddRequest {
            token_sequence: None,
            track_prefill_tokens: false,
            expected_output_tokens: None,
            prefill_load_hint: None,
        }
    }

    #[tokio::test]
    async fn mixed_replica_batch_updates_lora_load_in_order() {
        temp_env::async_with_vars(
            [
                (broker_env::DYN_ZMQ_BROKER_URL, None::<&str>),
                (broker_env::DYN_ZMQ_BROKER_ENABLED, None::<&str>),
            ],
            async {
                let runtime = Runtime::from_current().expect("create runtime handle");
                let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
                    .await
                    .expect("create distributed runtime");
                let endpoint = drt
                    .namespace(format!("lora-replica-batch-test-{}", uuid::Uuid::new_v4()))
                    .expect("create namespace")
                    .component("worker")
                    .expect("create component")
                    .endpoint("generate");
                let publisher = EventPublisher::for_endpoint(&endpoint, ACTIVE_SEQUENCES_SUBJECT)
                    .await
                    .expect("create publisher");
                let mut subscriber = RuntimeSequenceSubscriber::for_endpoint(&endpoint)
                    .await
                    .expect("create sequence subscriber");
                let batch = ActiveSequenceEventBatch {
                    events: vec![
                        lora_event("alpha", Some("lora-alpha"), add_request()),
                        lora_event(
                            "alpha",
                            Some("lora-alpha"),
                            ActiveSequenceEventData::MarkPrefillCompleted,
                        ),
                        lora_event("beta", Some("lora-beta"), add_request()),
                        lora_event("alpha", Some("lora-alpha"), ActiveSequenceEventData::Free),
                        lora_event("no-lora", None, add_request()),
                    ],
                };

                let events = tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        publisher
                            .publish(&batch)
                            .await
                            .expect("publish active-sequence batch");
                        let receive_batch = async {
                            let mut events = Vec::with_capacity(batch.events.len());
                            while events.len() < batch.events.len() {
                                match subscriber.next_event().await {
                                    Some(Ok(event)) => events.push(event),
                                    Some(Err(error)) => {
                                        panic!("receive active-sequence batch: {error}")
                                    }
                                    None => panic!("active-sequence event stream closed"),
                                }
                            }
                            events
                        };
                        match tokio::time::timeout(Duration::from_millis(100), receive_batch).await
                        {
                            Ok(events) => break events,
                            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
                        }
                    }
                })
                .await
                .expect("subscriber should receive an active-sequence batch");

                assert_eq!(
                    events
                        .iter()
                        .map(|event| event.request_id.as_str())
                        .collect::<Vec<_>>(),
                    vec!["alpha", "alpha", "beta", "alpha", "no-lora"]
                );

                let estimator = LoadEstimator::new();
                for event in events {
                    estimator.handle_event(event);
                }

                let inflight = estimator.get_inflight_counts();
                assert!(!inflight.contains_key("lora-alpha"));
                assert_eq!(inflight.get("lora-beta"), Some(&1));

                let arrivals = estimator.get_raw_arrival_counts();
                assert_eq!(arrivals.get("lora-alpha"), Some(&1));
                assert_eq!(arrivals.get("lora-beta"), Some(&1));
                assert_eq!(arrivals.len(), 2);
                drt.shutdown();
            },
        )
        .await;
    }
}
