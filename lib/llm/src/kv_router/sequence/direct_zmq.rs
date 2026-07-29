// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{ffi::OsString, sync::Arc};

use anyhow::Result;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, SequencePublisher,
    protocols::{ActiveSequenceEventBatch, MAX_REPLICA_BATCH_EVENTS},
};
use dynamo_runtime::{
    component::Endpoint,
    config::parse_bool_opt,
    discovery::EventTransportKind,
    transports::event_plane::{Codec, uses_direct_zmq},
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{
    direct_zmq_fan_in::{ContinuityMode, FanInEvent, start_direct_zmq_fan_in},
    kv_router::{ACTIVE_SEQUENCES_SUBJECT, metrics::ActiveSequenceZmqIngressMetrics},
};

const DIRECT_ZMQ_ENV: &str = "DYN_ROUTER_ACTIVE_SEQUENCE_DIRECT_ZMQ";
const RCVHWM_ENV: &str = "DYN_ROUTER_ACTIVE_SEQUENCE_ZMQ_RCVHWM";
const DEFAULT_RCVHWM: i32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectZmqSequenceConfig {
    enabled: bool,
    pub(super) rcvhwm: i32,
}

impl DirectZmqSequenceConfig {
    pub(super) fn from_env() -> Self {
        Self::from_lookup(|key| std::env::var_os(key))
    }

    fn from_lookup(mut get_env: impl FnMut(&str) -> Option<OsString>) -> Self {
        let enabled = match get_env(DIRECT_ZMQ_ENV) {
            None => true,
            Some(raw) => {
                let value = raw.to_string_lossy();
                match parse_bool_opt(&value) {
                    Some(enabled) => enabled,
                    None => {
                        tracing::warn!(
                            env = DIRECT_ZMQ_ENV,
                            %value,
                            "invalid direct-ZMQ active-sequence setting; using enabled default"
                        );
                        true
                    }
                }
            }
        };

        let rcvhwm = match get_env(RCVHWM_ENV) {
            None => DEFAULT_RCVHWM,
            Some(raw) => match raw.to_string_lossy().trim().parse::<i32>() {
                Ok(value) if value > 0 => value,
                _ => {
                    tracing::warn!(
                        env = RCVHWM_ENV,
                        value = %raw.to_string_lossy(),
                        default = DEFAULT_RCVHWM,
                        "invalid direct-ZMQ active-sequence receive HWM; using default"
                    );
                    DEFAULT_RCVHWM
                }
            },
        };

        Self { enabled, rcvhwm }
    }

    pub(super) fn should_use_direct(self, transport_kind: EventTransportKind) -> bool {
        self.should_use_direct_for_topology(transport_kind, uses_direct_zmq(transport_kind))
    }

    fn should_use_direct_for_topology(
        self,
        transport_kind: EventTransportKind,
        direct_zmq_topology: bool,
    ) -> bool {
        self.enabled && transport_kind == EventTransportKind::Zmq && direct_zmq_topology
    }
}

pub(super) async fn start<P: SequencePublisher + 'static>(
    endpoint: Endpoint,
    tracker: Arc<ActiveSequencesMultiWorker<P>>,
    rcvhwm: i32,
    cancellation_token: CancellationToken,
) -> Result<JoinHandle<()>> {
    let metrics = ActiveSequenceZmqIngressMetrics::from_component(endpoint.component());
    let handler_metrics = metrics.clone();
    let handler = move |envelope: dynamo_runtime::transports::event_plane::ValidatedEnvelope| {
        let codec = Codec::default();
        let batch = codec
            .decode_payload::<ActiveSequenceEventBatch>(&envelope.payload)
            .inspect_err(|_| {
                handler_metrics.record_payload_decode_error();
            })?;
        if batch.events.len() > MAX_REPLICA_BATCH_EVENTS {
            handler_metrics.record_payload_decode_error();
            anyhow::bail!(
                "active-sequence batch contains {} events; maximum is {}",
                batch.events.len(),
                MAX_REPLICA_BATCH_EVENTS
            );
        }
        tracker.apply_replica_batch(batch.events);
        Ok(())
    };
    let observer =
        move |observation: crate::direct_zmq_fan_in::FanInObservation| match observation.event {
            FanInEvent::SourceStarted => metrics.source_started(),
            FanInEvent::SourceStopped => metrics.source_stopped(),
            FanInEvent::Reconnect => metrics.record_reconnect(),
            FanInEvent::Replacement => metrics.record_replacement(),
            FanInEvent::EnvelopeDecodeError => metrics.record_envelope_decode_error(),
            FanInEvent::IdentityMismatch => metrics.record_identity_error(),
            FanInEvent::SequenceGap { missing } => metrics.record_gap(missing),
            FanInEvent::OutOfOrder => metrics.record_out_of_order(),
            FanInEvent::ForcedAbort => metrics.record_forced_abort(),
        };

    start_direct_zmq_fan_in(
        endpoint,
        ACTIVE_SEQUENCES_SUBJECT,
        rcvhwm,
        None,
        ContinuityMode::TrackFromZero,
        cancellation_token,
        handler,
        observer,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use dynamo_kv_router::{
        NoopSequencePublisher,
        protocols::{ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank},
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime, distributed::DistributedConfig,
        transports::event_plane::EventPublisher,
    };

    use super::*;

    fn lookup(direct: Option<&str>, rcvhwm: Option<&str>) -> impl FnMut(&str) -> Option<OsString> {
        let direct = direct.map(OsString::from);
        let rcvhwm = rcvhwm.map(OsString::from);
        move |key| match key {
            DIRECT_ZMQ_ENV => direct.clone(),
            RCVHWM_ENV => rcvhwm.clone(),
            _ => None,
        }
    }

    #[test]
    fn parses_direct_zmq_configuration() {
        assert_eq!(
            DirectZmqSequenceConfig::from_lookup(lookup(None, None)),
            DirectZmqSequenceConfig {
                enabled: true,
                rcvhwm: DEFAULT_RCVHWM,
            }
        );
        assert_eq!(
            DirectZmqSequenceConfig::from_lookup(lookup(Some("false"), Some("37"))),
            DirectZmqSequenceConfig {
                enabled: false,
                rcvhwm: 37,
            }
        );
        for invalid in ["0", "-1", "not-a-number", "2147483648"] {
            assert_eq!(
                DirectZmqSequenceConfig::from_lookup(lookup(None, Some(invalid))).rcvhwm,
                DEFAULT_RCVHWM
            );
        }
        assert!(DirectZmqSequenceConfig::from_lookup(lookup(Some("invalid"), None)).enabled);
    }

    #[test]
    fn selects_only_direct_zmq_and_honors_rollback() {
        let enabled = DirectZmqSequenceConfig::from_lookup(lookup(None, None));
        let rollback = DirectZmqSequenceConfig::from_lookup(lookup(Some("0"), None));

        assert!(enabled.should_use_direct_for_topology(EventTransportKind::Zmq, true));
        assert!(!enabled.should_use_direct_for_topology(EventTransportKind::Zmq, false));
        assert!(!enabled.should_use_direct_for_topology(EventTransportKind::Nats, false));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Zmq, true));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Zmq, false));
        assert!(!rollback.should_use_direct_for_topology(EventTransportKind::Nats, false));
    }

    fn add_event(request_id: impl Into<String>) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: WorkerWithDpRank::new(0, 0),
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: Some(vec![1]),
                track_prefill_tokens: false,
                expected_output_tokens: None,
                prefill_load_hint: None,
            },
            router_id: 99,
            lora_name: None,
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn direct_zmq_batch_ingress_applies_after_publisher_recreation() -> Result<()> {
        let runtime = Runtime::from_current()?;
        let distributed =
            DistributedRuntime::new(runtime, DistributedConfig::process_local()).await?;
        let endpoint = distributed
            .namespace(format!(
                "active-sequence-direct-zmq-{}",
                uuid::Uuid::new_v4()
            ))?
            .component("frontend")?
            .endpoint("generate");
        let tracker = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(0, (0, 1))]),
            true,
            1,
            "test",
        ));
        let worker = WorkerWithDpRank::new(0, 0);
        let cancel = CancellationToken::new();
        let supervisor = start(
            endpoint.clone(),
            tracker.clone(),
            DEFAULT_RCVHWM,
            cancel.clone(),
        )
        .await?;

        for (index, request_id) in ["before-recreation", "after-recreation"]
            .into_iter()
            .enumerate()
        {
            let publisher = EventPublisher::for_endpoint_with_transport(
                &endpoint,
                ACTIVE_SEQUENCES_SUBJECT,
                EventTransportKind::Zmq,
            )
            .await?;
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    publisher
                        .publish(&ActiveSequenceEventBatch {
                            events: vec![add_event(request_id)],
                        })
                        .await
                        .unwrap();
                    if tracker.active_request_counts()[&worker] == index + 1 {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("direct-ZMQ source should apply a warmed publisher batch");
            drop(publisher);
        }

        cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), supervisor)
            .await
            .expect("fan-in supervisor should stop after cancellation")
            .expect("fan-in supervisor should not panic");
        distributed.shutdown();
        Ok(())
    }
}
