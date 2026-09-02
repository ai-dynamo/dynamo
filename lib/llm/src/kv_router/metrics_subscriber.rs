// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-selecting subscription for worker KV load metrics.

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::protocols::ActiveLoad;
use dynamo_runtime::{
    DistributedRuntime,
    component::{Component, Endpoint},
    config::environment_names::event_plane::DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, EventChannelInstanceId,
        EventChannelQuery, EventScope, EventTransport,
    },
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, EventSubscriber, TypedEventSubscriber, uses_direct_zmq},
};
use futures::StreamExt;
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use super::{
    KV_METRICS_SUBJECT,
    metrics::{KvZmqIngressMetrics, KvZmqIngressStream},
};
use crate::direct_zmq_sub_pool::{
    DirectZmqSubPool, DirectZmqSubPoolEvent, endpoints_per_sub_from_env,
};

const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(5);
const SOURCE_JOIN_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_OUTPUT_CAPACITY: usize = 100_000;

fn output_capacity() -> usize {
    std::env::var(DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_OUTPUT_CAPACITY)
}

pub(crate) struct KvMetricsSubscriber {
    inner: KvMetricsSubscriberInner,
}

enum KvMetricsSubscriberInner {
    Standard(TypedEventSubscriber<ActiveLoad>),
    Direct(DirectKvMetricsSubscriber),
}

struct DirectKvMetricsSubscriber {
    receiver: mpsc::Receiver<ActiveLoad>,
    cancellation_token: CancellationToken,
    _supervisor: JoinHandle<()>,
}

impl Drop for DirectKvMetricsSubscriber {
    fn drop(&mut self) {
        self.cancellation_token.cancel();
    }
}

impl KvMetricsSubscriber {
    pub(crate) async fn for_endpoint(endpoint: &Endpoint) -> Result<Self> {
        Self::new(
            endpoint.component(),
            endpoint.drt(),
            endpoint.id(),
            Some(endpoint),
        )
        .await
    }

    pub(crate) async fn for_endpoint_id(
        component: &Component,
        endpoint: &EndpointId,
    ) -> Result<Self> {
        Self::new(component, component.drt(), endpoint.clone(), None).await
    }

    async fn new(
        component: &Component,
        drt: &DistributedRuntime,
        endpoint_id: EndpointId,
        endpoint: Option<&Endpoint>,
    ) -> Result<Self> {
        if uses_direct_zmq(drt.default_event_transport_kind()) {
            return Ok(Self {
                inner: KvMetricsSubscriberInner::Direct(
                    DirectKvMetricsSubscriber::start(component, endpoint_id).await?,
                ),
            });
        }

        let subscriber = match endpoint {
            Some(endpoint) => EventSubscriber::for_endpoint(endpoint, KV_METRICS_SUBJECT).await?,
            None => EventSubscriber::for_endpoint_id(drt, &endpoint_id, KV_METRICS_SUBJECT).await?,
        };
        Ok(Self {
            inner: KvMetricsSubscriberInner::Standard(subscriber.typed::<ActiveLoad>()),
        })
    }

    pub(crate) async fn next(&mut self) -> Option<Result<ActiveLoad>> {
        match &mut self.inner {
            KvMetricsSubscriberInner::Standard(subscriber) => subscriber
                .next()
                .await
                .map(|result| result.map(|(_envelope, load)| load)),
            KvMetricsSubscriberInner::Direct(subscriber) => {
                subscriber.receiver.recv().await.map(Ok)
            }
        }
    }
}

impl DirectKvMetricsSubscriber {
    async fn start(component: &Component, endpoint: EndpointId) -> Result<Self> {
        let endpoints_per_sub = endpoints_per_sub_from_env()?;
        let cancellation_token = component.drt().primary_token().child_token();
        let query = dynamo_runtime::discovery::DiscoveryQuery::EventChannels(
            EventChannelQuery::endpoint_topic(endpoint.clone(), KV_METRICS_SUBJECT),
        );
        let watch_cancel = cancellation_token.child_token();
        let watch = component
            .drt()
            .discovery()
            .list_and_watch(query, Some(watch_cancel.clone()))
            .await?;
        let metrics = KvZmqIngressMetrics::from_component(component);
        let pool_metrics = metrics.clone();
        let pool_observer = Arc::new(move |event: DirectZmqSubPoolEvent| {
            pool_metrics.observe_pool(KvZmqIngressStream::Metrics, event);
        });
        let pool = DirectZmqSubPool::new(
            KV_METRICS_SUBJECT,
            endpoints_per_sub,
            pool_observer,
            cancellation_token.child_token(),
        )?;
        let (sender, receiver) = mpsc::channel(output_capacity());
        let supervisor_cancel = cancellation_token.clone();
        let supervisor = tokio::spawn(run_metrics_supervisor(
            endpoint,
            watch,
            watch_cancel,
            pool,
            sender,
            metrics,
            supervisor_cancel,
        ));

        Ok(Self {
            receiver,
            cancellation_token,
            _supervisor: supervisor,
        })
    }
}

struct MetricsSourceTask {
    endpoint: String,
    generation: u64,
    cancel: CancellationToken,
    handle: JoinHandle<()>,
}

#[allow(clippy::too_many_arguments)]
async fn run_metrics_supervisor(
    endpoint: EndpointId,
    mut discovery_stream: dynamo_runtime::discovery::DiscoveryStream,
    watch_cancel: CancellationToken,
    pool: DirectZmqSubPool,
    output: mpsc::Sender<ActiveLoad>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancellation_token: CancellationToken,
) {
    let expected_scope = EventScope::Endpoint { endpoint };
    let mut sources = HashMap::<u64, MetricsSourceTask>::new();
    let mut next_generation = 1_u64;

    loop {
        let event = tokio::select! {
            biased;
            _ = cancellation_token.cancelled() => break,
            event = discovery_stream.next() => event,
        };
        let Some(event) = event else {
            tracing::warn!(
                topic = KV_METRICS_SUBJECT,
                "Direct-ZMQ KV metrics discovery stream ended"
            );
            break;
        };

        match event {
            Ok(DiscoveryEvent::Added(DiscoveryInstance::EventChannel {
                scope,
                topic,
                instance_id,
                transport,
            })) if scope == expected_scope && topic == KV_METRICS_SUBJECT => {
                let EventTransport::Zmq { endpoint } = transport else {
                    tracing::warn!(
                        publisher_id = instance_id,
                        "Ignoring non-ZMQ KV metrics publisher in direct mode"
                    );
                    continue;
                };
                if sources
                    .get(&instance_id)
                    .is_some_and(|source| source.endpoint == endpoint)
                {
                    continue;
                }
                if let Some(source) = sources.remove(&instance_id) {
                    stop_metrics_source(source, &metrics).await;
                    metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "replacement");
                }
                let generation = next_generation;
                next_generation = next_generation.wrapping_add(1);
                sources.insert(
                    instance_id,
                    spawn_metrics_source(
                        instance_id,
                        endpoint,
                        generation,
                        pool.clone(),
                        output.clone(),
                        metrics.clone(),
                        cancellation_token.child_token(),
                    ),
                );
                metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "started");
            }
            Ok(DiscoveryEvent::Removed(DiscoveryInstanceId::EventChannel(
                EventChannelInstanceId {
                    scope,
                    topic,
                    instance_id,
                },
            ))) if scope == expected_scope && topic == KV_METRICS_SUBJECT => {
                if let Some(source) = sources.remove(&instance_id) {
                    stop_metrics_source(source, &metrics).await;
                    metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "removed");
                }
            }
            Ok(DiscoveryEvent::Added(_))
            | Ok(DiscoveryEvent::ModelTaintsUpdated(_))
            | Ok(DiscoveryEvent::Removed(_)) => {}
            Err(error) => {
                tracing::warn!(%error, topic = KV_METRICS_SUBJECT, "Direct-ZMQ KV metrics discovery failed");
                break;
            }
        }
    }

    watch_cancel.cancel();
    let sources = sources.into_values().collect::<Vec<_>>();
    for source in &sources {
        source.cancel.cancel();
    }
    pool.shutdown().await;
    futures::future::join_all(
        sources
            .into_iter()
            .map(|source| stop_metrics_source(source, &metrics)),
    )
    .await;
}

fn spawn_metrics_source(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    pool: DirectZmqSubPool,
    output: mpsc::Sender<ActiveLoad>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancel: CancellationToken,
) -> MetricsSourceTask {
    let task_endpoint = endpoint.clone();
    let task_cancel = cancel.clone();
    let handle = tokio::spawn(async move {
        run_metrics_source(
            publisher_id,
            task_endpoint,
            generation,
            pool,
            output,
            metrics,
            task_cancel,
        )
        .await;
    });
    MetricsSourceTask {
        endpoint,
        generation,
        cancel,
        handle,
    }
}

async fn run_metrics_source(
    publisher_id: u64,
    endpoint: String,
    generation: u64,
    pool: DirectZmqSubPool,
    output: mpsc::Sender<ActiveLoad>,
    metrics: Arc<KvZmqIngressMetrics>,
    cancel: CancellationToken,
) {
    let codec = Codec::default();
    let mut retry_delay = INITIAL_BACKOFF;
    loop {
        if cancel.is_cancelled() {
            return;
        }
        let registration = pool.register(publisher_id, &endpoint, generation).await;
        let mut registration = match registration {
            Ok(registration) => registration,
            Err(error) => {
                tracing::warn!(%error, publisher_id, %endpoint, "Failed to register direct-ZMQ KV metrics source");
                metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "reconnect");
                if !sleep_or_cancel(retry_delay, &cancel).await {
                    return;
                }
                retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
                continue;
            }
        };
        if cancel.is_cancelled() {
            pool.unregister(registration.group_id, publisher_id, generation)
                .await;
            return;
        }
        retry_delay = INITIAL_BACKOFF;

        let disconnected = loop {
            let envelope = tokio::select! {
                biased;
                _ = cancel.cancelled() => break false,
                _ = registration.disconnected.cancelled() => break true,
                envelope = registration.receiver.recv() => envelope,
            };
            let Some(envelope) = envelope else {
                break true;
            };
            let load = match codec.decode_payload::<ActiveLoad>(&envelope.payload) {
                Ok(load) => load,
                Err(error) => {
                    tracing::warn!(%error, publisher_id, "Failed to decode direct-ZMQ KV metrics payload");
                    metrics.increment_stream_lifecycle(
                        KvZmqIngressStream::Metrics,
                        "payload_decode_error",
                    );
                    continue;
                }
            };
            match output.try_send(load) {
                Ok(()) => metrics.increment_batch(KvZmqIngressStream::Metrics),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "consumer_full")
                }
                Err(mpsc::error::TrySendError::Closed(_)) => break false,
            }
        };

        pool.unregister(registration.group_id, publisher_id, generation)
            .await;
        if !disconnected || output.is_closed() {
            return;
        }
        metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "reconnect");
        if !sleep_or_cancel(retry_delay, &cancel).await {
            return;
        }
        retry_delay = (retry_delay * 2).min(MAX_BACKOFF);
    }
}

async fn stop_metrics_source(mut source: MetricsSourceTask, metrics: &KvZmqIngressMetrics) {
    source.cancel.cancel();
    match tokio::time::timeout(SOURCE_JOIN_TIMEOUT, &mut source.handle).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) if error.is_cancelled() => {}
        Ok(Err(error)) => {
            tracing::warn!(%error, generation = source.generation, "Direct-ZMQ KV metrics source failed during shutdown")
        }
        Err(_) => {
            source.handle.abort();
            let _ = source.handle.await;
            metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "forced_abort");
        }
    }
    metrics.increment_stream_lifecycle(KvZmqIngressStream::Metrics, "stopped");
}

async fn sleep_or_cancel(delay: Duration, cancellation_token: &CancellationToken) -> bool {
    tokio::select! {
        _ = cancellation_token.cancelled() => false,
        _ = tokio::time::sleep(delay) => true,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use dynamo_runtime::{
        DistributedRuntime, Runtime, discovery::EventTransportKind, distributed::DistributedConfig,
        transports::event_plane::EventPublisher,
    };

    use super::*;
    use crate::direct_zmq_sub_pool::ENDPOINTS_PER_SUB_ENV;

    #[tokio::test]
    #[serial_test::serial]
    async fn direct_metrics_fan_in_receives_several_publishers() {
        temp_env::async_with_vars(
            [
                (
                    dynamo_runtime::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_URL,
                    None::<&str>,
                ),
                (
                    dynamo_runtime::config::environment_names::zmq_broker::DYN_ZMQ_BROKER_ENABLED,
                    None::<&str>,
                ),
                (ENDPOINTS_PER_SUB_ENV, Some("64")),
            ],
            async {
                let runtime = Runtime::from_current().expect("create runtime handle");
                let distributed =
                    DistributedRuntime::new(runtime, DistributedConfig::process_local())
                        .await
                        .expect("create distributed runtime");
                let endpoint = distributed
                    .namespace(format!("kv-metrics-fan-in-{}", uuid::Uuid::new_v4()))
                    .expect("create namespace")
                    .component("frontend")
                    .expect("create component")
                    .endpoint("generate");
                let mut subscriber = KvMetricsSubscriber::for_endpoint(&endpoint)
                    .await
                    .expect("create direct metrics subscriber");
                let publisher_a = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    KV_METRICS_SUBJECT,
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create publisher A");
                let publisher_b = EventPublisher::for_endpoint_with_transport(
                    &endpoint,
                    KV_METRICS_SUBJECT,
                    EventTransportKind::Zmq,
                )
                .await
                .expect("create publisher B");

                let mut observed = HashSet::new();
                tokio::time::timeout(Duration::from_secs(5), async {
                    while observed.len() != 2 {
                        publisher_a
                            .publish(&ActiveLoad {
                                worker_id: 1,
                                ..ActiveLoad::default()
                            })
                            .await
                            .expect("publish A");
                        publisher_b
                            .publish(&ActiveLoad {
                                worker_id: 2,
                                ..ActiveLoad::default()
                            })
                            .await
                            .expect("publish B");
                        if let Ok(Some(Ok(load))) =
                            tokio::time::timeout(Duration::from_millis(50), subscriber.next()).await
                        {
                            observed.insert(load.worker_id);
                        }
                        tokio::time::sleep(Duration::from_millis(20)).await;
                    }
                })
                .await
                .expect("receive metrics from both publishers");
                assert_eq!(observed, HashSet::from([1, 2]));

                distributed.shutdown();
            },
        )
        .await;
    }
}
