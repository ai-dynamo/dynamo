// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-selecting subscription for worker KV load metrics.

use anyhow::Result;
use dynamo_kv_router::protocols::ActiveLoad;
use dynamo_runtime::{
    DistributedRuntime,
    component::{Component, Endpoint},
    config::environment_names::event_plane::DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY,
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::{Codec, EventSubscriber, TypedEventSubscriber, uses_direct_zmq},
};
use tokio::{sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

use super::KV_METRICS_SUBJECT;
use crate::{
    direct_zmq_fan_in::{ContinuityMode, start_direct_zmq_fan_in_for_endpoint_id},
    direct_zmq_sub_pool::KV_ZMQ_RCVHWM,
};

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
    async fn start(component: &Component, endpoint_id: EndpointId) -> Result<Self> {
        let cancellation_token = component.drt().primary_token().child_token();
        let (sender, receiver) = mpsc::channel(output_capacity());
        let handler_cancel = cancellation_token.clone();
        let codec = Codec::default();
        let handler =
            move |envelope: dynamo_runtime::transports::event_plane::ValidatedEnvelope| {
                let load = codec.decode_payload::<ActiveLoad>(&envelope.payload)?;
                match sender.try_send(load) {
                    Ok(()) => Ok(()),
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        tracing::warn!(
                            publisher_id = envelope.publisher_id,
                            "Direct-ZMQ KV metrics consumer is full; dropping newest update"
                        );
                        Ok(())
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        handler_cancel.cancel();
                        anyhow::bail!("direct-ZMQ KV metrics consumer closed")
                    }
                }
            };
        let supervisor = start_direct_zmq_fan_in_for_endpoint_id(
            component.clone(),
            endpoint_id,
            KV_METRICS_SUBJECT,
            KV_ZMQ_RCVHWM,
            None,
            ContinuityMode::Disabled,
            cancellation_token.clone(),
            handler,
            |_| {},
        )
        .await?;

        Ok(Self {
            receiver,
            cancellation_token,
            _supervisor: supervisor,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, time::Duration};

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
