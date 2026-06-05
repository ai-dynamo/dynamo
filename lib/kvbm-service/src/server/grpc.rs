// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! tonic implementation of `KvbmService::Register`.
//!
//! Flow:
//! 1. Decode the proto request into a [`RegistrationInstance`] and
//!    validate it against the registry's slot capacity.
//! 2. Reserve a slot via [`Registry::try_register`] (no
//!    `register_total++` yet).
//! 3. Drive [`ServiceContainer::on_register`] with the typed instance.
//!    On `Err`, call [`Registry::rollback_register`] to release the
//!    slot and bump `register_rejected_total`.
//! 4. On success, call [`Registry::commit_register`] (bumps
//!    `register_total`), send `Accepted`, and spawn the watcher +
//!    heartbeat tasks. The watcher fires on either client disconnect
//!    or coordinator-issued force-close.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

use crate::container::{ContainerError, ServiceContainer};
use crate::instance::RegistrationInstance;
use crate::proto;
use crate::proto::v1::kvbm_service_server::KvbmService as KvbmServiceTrait;
use crate::registry::Registry;
use crate::registry::lifecycle::GrpcStreamLifecycle;

const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// tonic service implementation. Cheap to clone — both the `Registry` and
/// the `Arc<dyn ServiceContainer>` are reference-counted.
#[derive(Clone)]
pub struct KvbmServiceGrpc {
    registry: Registry,
    container: Arc<dyn ServiceContainer>,
    heartbeat_interval: Duration,
}

impl KvbmServiceGrpc {
    pub fn new(registry: Registry, container: Arc<dyn ServiceContainer>) -> Self {
        Self::with_heartbeat(registry, container, DEFAULT_HEARTBEAT_INTERVAL)
    }

    pub fn with_heartbeat(
        registry: Registry,
        container: Arc<dyn ServiceContainer>,
        heartbeat_interval: Duration,
    ) -> Self {
        Self {
            registry,
            container,
            heartbeat_interval,
        }
    }
}

#[tonic::async_trait]
impl KvbmServiceTrait for KvbmServiceGrpc {
    type RegisterStream = ReceiverStream<Result<proto::Event, Status>>;

    async fn register(
        &self,
        request: Request<proto::RegisterRequest>,
    ) -> Result<Response<Self::RegisterStream>, Status> {
        let body = request.into_inner();
        let client_id = body.client_id;

        // 1. Decode and validate the registration instance.
        let proto_instance = body
            .instance
            .ok_or_else(|| Status::invalid_argument("instance is required"))?;
        let instance = RegistrationInstance::from_proto(proto_instance)?;
        instance.validate(self.registry.capacity_slots())?;

        // 2. Reserve the slot. The lifecycle is **not** attached yet — the
        //    shutdown coordinator can't broadcast to pre-commit slots, so
        //    `ServerShutdownInitiated` can never land on this stream before
        //    the `Accepted` event we enqueue below.
        let (tx, rx) = mpsc::channel::<Result<proto::Event, Status>>(8);
        let entry = self
            .registry
            .try_register(instance.clone(), client_id.clone())?;

        // 3. Container hook. On error, roll the registry slot back BEFORE
        //    returning — gauges go down, register_rejected_total++, and
        //    register_total stays at zero for this attempt.
        if let Err(err) = self.container.on_register(entry.id, &instance).await {
            warn!(
                registration_id = %entry.id,
                container = self.container.name(),
                error = %err,
                "container rejected registration; rolling back"
            );
            self.registry.rollback_register(entry.id);
            return Err(container_error_to_status(err));
        }

        // 4. Enqueue `Accepted` on the response channel BEFORE the
        //    lifecycle becomes reachable. Once commit_register attaches
        //    the lifecycle below, the shutdown coordinator may broadcast
        //    at any moment — but `Accepted` is already in the queue, so
        //    `ServerShutdownInitiated` (or any other event) is guaranteed
        //    to be delivered after it.
        let accepted = proto::Event {
            kind: Some(proto::event::Kind::Accepted(proto::Accepted {
                registration_id: entry.id.to_string(),
                reserved_slots: entry.reserved_slots,
            })),
        };
        // Infallible at this point — receiver is alive, channel is fresh.
        let _ = tx.send(Ok(accepted)).await;

        // 5. Build the lifecycle and the per-stream cancellation token.
        //    The token is shared with the watcher and heartbeat tasks
        //    below; the lifecycle fires it from `send_shutdown_timed_out`
        //    so the watcher + heartbeat drop their sender clones and the
        //    stream closes.
        let stream_cancel = CancellationToken::new();
        let lifecycle = Arc::new(GrpcStreamLifecycle::new(
            tx.clone(),
            entry.id,
            stream_cancel.clone(),
        )) as Arc<dyn crate::registry::StreamLifecycle>;

        // 6. Commit. Atomically attaches the lifecycle (making it visible
        //    to begin_drain) and bumps `register_total`. If a concurrent
        //    `shutdown_graceful` flipped the registry to Draining during
        //    the container.on_register await, commit fails — the slot has
        //    already been rolled back inline and we return `unavailable`.
        //    The pending `Accepted` event never reaches the client
        //    because we drop `tx` / `rx` on the early return.
        if let Err(err) = self.registry.commit_register(entry.id, lifecycle) {
            warn!(
                registration_id = %entry.id,
                container = self.container.name(),
                error = %err,
                "commit_register failed; aborting registration and balancing container hook"
            );
            // `container.on_register` already returned `Ok` above, so the
            // container is bookkeeping this id as live. Commit failed
            // (typically because shutdown began during our await), and
            // commit_register has already rolled the registry slot back —
            // but the container never heard about it. Balance the pair.
            self.container.on_unregister(entry.id).await;
            return Err(err.into());
        }

        info!(
            registration_id = %entry.id,
            reserved_slots = entry.reserved_slots,
            kind = instance.kind_str(),
            container = self.container.name(),
            "client registered"
        );

        // Watcher task: fires cleanup either when the receiver is dropped
        // (client disconnect) or when the shutdown coordinator force-closes
        // this stream via `stream_cancel`. Single owner of the unregister
        // + container.on_unregister sequencing.
        let cleanup_tx = tx.clone();
        let registry_clone = self.registry.clone();
        let container_clone = self.container.clone();
        let entry_id = entry.id;
        let watcher_cancel = stream_cancel.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = cleanup_tx.closed() => {},
                _ = watcher_cancel.cancelled() => {
                    drop(cleanup_tx);
                }
            }
            if registry_clone.unregister(entry_id) {
                container_clone.on_unregister(entry_id).await;
                info!(registration_id = %entry_id, "client unregistered");
            } else {
                debug!(
                    registration_id = %entry_id,
                    "watcher fired but registration was already gone"
                );
            }
        });

        // Heartbeat task: sends periodic heartbeats; exits when the receiver
        // closes or when the coordinator force-closes this stream.
        let heartbeat_tx = tx.clone();
        let interval = self.heartbeat_interval;
        let heartbeat_cancel = stream_cancel.clone();
        tokio::spawn(async move {
            let mut seq: u64 = 0;
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {}
                    _ = heartbeat_cancel.cancelled() => break,
                }
                seq += 1;
                let hb = proto::Event {
                    kind: Some(proto::event::Kind::Heartbeat(proto::Heartbeat { seq })),
                };
                if heartbeat_tx.send(Ok(hb)).await.is_err() {
                    debug!(registration_id = %entry_id, seq, "heartbeat send failed — client gone");
                    break;
                }
            }
            drop(heartbeat_tx);
        });

        // Drop the original tx so only the heartbeat clone and watcher clone remain.
        drop(tx);

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn container_error_to_status(err: ContainerError) -> Status {
    match err {
        ContainerError::NotReady(msg) => Status::unavailable(msg),
        ContainerError::Rejected(msg) => Status::failed_precondition(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::NoopContainer;
    use crate::metrics::ServiceMetrics;

    fn make_registry(capacity: u32) -> Registry {
        Registry::new(capacity, ServiceMetrics::new())
    }

    fn make_svc(registry: Registry) -> KvbmServiceGrpc {
        KvbmServiceGrpc::with_heartbeat(
            registry,
            Arc::new(NoopContainer),
            Duration::from_secs(3600),
        )
    }

    fn valid_request(tp: u32) -> proto::RegisterRequest {
        proto::RegisterRequest {
            client_id: "test-client".into(),
            instance: Some(proto::RegistrationInstance {
                kind: Some(proto::registration_instance::Kind::Kvbm(
                    proto::KvbmInstance {
                        model_name: "llm".into(),
                        layout_mode: Some(proto::LayoutMode {
                            kind: Some(proto::layout_mode::Kind::UniversalTp1Canonical(vec![
                                1, 2, 3,
                            ])),
                        }),
                        tp_size: tp,
                        block_size: 64,
                        mode: proto::ServiceMode::Kvbm as i32,
                    },
                )),
            }),
        }
    }

    #[tokio::test]
    async fn validation_rejection_returns_invalid_argument() {
        let svc = make_svc(make_registry(8));
        let bad_req = proto::RegisterRequest {
            client_id: "bad-client".into(),
            instance: Some(proto::RegistrationInstance {
                kind: Some(proto::registration_instance::Kind::Kvbm(
                    proto::KvbmInstance {
                        model_name: "llm".into(),
                        layout_mode: Some(proto::LayoutMode {
                            kind: Some(proto::layout_mode::Kind::UniversalTp1Canonical(vec![1])),
                        }),
                        tp_size: 3,
                        block_size: 64,
                        mode: proto::ServiceMode::Kvbm as i32,
                    },
                )),
            }),
        };
        let err = svc.register(Request::new(bad_req)).await.unwrap_err();
        assert_eq!(
            err.code(),
            tonic::Code::InvalidArgument,
            "unexpected: {err}"
        );
    }

    #[tokio::test]
    async fn missing_instance_returns_invalid_argument() {
        let svc = make_svc(make_registry(8));
        let req = proto::RegisterRequest {
            client_id: "c".into(),
            instance: None,
        };
        let err = svc.register(Request::new(req)).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn missing_instance_kind_returns_invalid_argument() {
        let svc = make_svc(make_registry(8));
        let req = proto::RegisterRequest {
            client_id: "c".into(),
            instance: Some(proto::RegistrationInstance { kind: None }),
        };
        let err = svc.register(Request::new(req)).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn successful_register_sends_accepted_and_commits_metric() {
        use tokio_stream::StreamExt;
        let reg = make_registry(8);
        let svc = make_svc(reg.clone());
        let resp = svc.register(Request::new(valid_request(4))).await.unwrap();
        let mut stream = resp.into_inner();
        let first = stream.next().await.unwrap().unwrap();
        match first.kind {
            Some(proto::event::Kind::Accepted(acc)) => {
                assert!(!acc.registration_id.is_empty());
                assert_eq!(acc.reserved_slots, 4);
            }
            other => panic!("expected Accepted, got {other:?}"),
        }
        // Success path bumps register_total exactly once.
        assert_eq!(reg.metrics().register_total.get(), 1);
        assert_eq!(reg.metrics().register_rejected_total.get(), 0);
    }

    #[tokio::test]
    async fn capacity_exhausted_returns_resource_exhausted() {
        let reg = make_registry(4);
        let svc = make_svc(reg);
        let _resp1 = svc.register(Request::new(valid_request(4))).await.unwrap();
        let err = svc
            .register(Request::new(valid_request(4)))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);
    }

    #[tokio::test]
    async fn register_during_drain_returns_unavailable() {
        let reg = make_registry(8);
        reg.begin_drain();
        let svc = make_svc(reg);
        let err = svc
            .register(Request::new(valid_request(4)))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unavailable);
    }

    /// If `commit_register` fails (e.g. concurrent `begin_drain`),
    /// `container.on_register` has already returned `Ok`, so the handler
    /// MUST call `container.on_unregister` to keep the container's
    /// bookkeeping balanced.
    #[tokio::test]
    async fn container_on_unregister_called_when_commit_fails_due_to_draining() {
        use std::sync::atomic::{AtomicU32, Ordering};

        use crate::registry::RegistrationId;

        struct DrainOnRegisterContainer {
            registry: Registry,
            unregisters: AtomicU32,
        }

        #[tonic::async_trait]
        impl ServiceContainer for DrainOnRegisterContainer {
            fn name(&self) -> &str {
                "drain-on-register"
            }
            async fn on_register(
                &self,
                _id: RegistrationId,
                _instance: &RegistrationInstance,
            ) -> Result<(), ContainerError> {
                // Force the registry into Draining while the gRPC handler
                // is still inside the on_register await — commit_register
                // will then fail with ShuttingDown.
                self.registry.begin_drain();
                Ok(())
            }
            async fn on_unregister(&self, _id: RegistrationId) {
                self.unregisters.fetch_add(1, Ordering::SeqCst);
            }
            async fn on_server_shutdown(&self, _grace: Option<Duration>) {}
        }

        let reg = make_registry(8);
        let container = Arc::new(DrainOnRegisterContainer {
            registry: reg.clone(),
            unregisters: AtomicU32::new(0),
        });
        let svc = KvbmServiceGrpc::with_heartbeat(
            reg.clone(),
            container.clone(),
            Duration::from_secs(3600),
        );

        let err = svc
            .register(Request::new(valid_request(2)))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unavailable, "got {err}");

        // The smoking gun: on_register was called, commit failed, so the
        // handler must have invoked on_unregister to balance the pair.
        assert_eq!(
            container.unregisters.load(Ordering::SeqCst),
            1,
            "container.on_unregister must run when commit_register fails after on_register succeeded"
        );

        // Metrics: rejected (from commit's inline rollback), zero committed.
        let m = reg.metrics();
        assert_eq!(m.register_total.get(), 0);
        assert_eq!(m.register_rejected_total.get(), 1);
        assert_eq!(m.unregister_total.get(), 0);
    }

    /// Container rejection rolls back the registry slot and counts as a
    /// rejected registration — not as a successful register + unregister.
    #[tokio::test]
    async fn container_reject_rolls_back_registry() {
        use crate::registry::RegistrationId;

        struct RejectingContainer;
        #[tonic::async_trait]
        impl ServiceContainer for RejectingContainer {
            fn name(&self) -> &str {
                "rejecting"
            }
            async fn on_register(
                &self,
                _id: RegistrationId,
                _instance: &RegistrationInstance,
            ) -> Result<(), ContainerError> {
                Err(ContainerError::Rejected("nope".into()))
            }
            async fn on_unregister(&self, _id: RegistrationId) {}
            async fn on_server_shutdown(&self, _grace: Option<Duration>) {}
        }

        let reg = make_registry(4);
        let svc = KvbmServiceGrpc::with_heartbeat(
            reg.clone(),
            Arc::new(RejectingContainer),
            Duration::from_secs(3600),
        );
        let err = svc
            .register(Request::new(valid_request(2)))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        let snap = reg.snapshot();
        assert_eq!(snap.state, "Empty");
        assert_eq!(snap.used_slots, 0);

        // Metrics: rejection counted, no committed register, no unregister.
        let m = reg.metrics();
        assert_eq!(m.register_total.get(), 0);
        assert_eq!(m.register_rejected_total.get(), 1);
        assert_eq!(m.unregister_total.get(), 0);
        assert_eq!(m.reset_total.get(), 0);
    }
}
