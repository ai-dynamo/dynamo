// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};
use dynamo_kv_router::selector::WorkerSelector;

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    /// Query the best prefill worker without executing a request.
    ///
    /// This query is advisory and does not book scheduler or occupancy state;
    /// concurrent callers may observe the same worker. `policy_class` names the
    /// policy-class family to admit the request under; `None` uses the router's
    /// configured default family. A returned `PrefillQueryOutcome::QueueRejected`
    /// means the class refused admission outright (its
    /// `request_queue_limit_per_worker` is `0`) rather than queuing, so callers
    /// that want per-class prefill shedding must treat it as a shed, not as
    /// "prefill unavailable, fall back to aggregated."
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let binding = self
            .binding
            .load_full()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match &binding.router {
            InnerPrefillRouter::KvRouter(router) => {
                let outcome = router
                    .chooser
                    .find_best_match_details_with_policy_class(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        false,
                        false,
                        lora_name,
                        cache_namespace,
                        priority_jump,
                        strict_priority,
                        policy_class,
                        None,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        Ok(PrefillQueryOutcome::Routed {
                            worker_id: worker.worker_id,
                            dp_rank: Some(worker.dp_rank),
                        })
                    }
                    crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                        Ok(PrefillQueryOutcome::QueueRejected { rejection })
                    }
                }
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                let worker_id = router
                    .peek_next_worker()
                    .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank: None,
                })
            }
        }
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(binding) = self.binding.load_full()
            && let InnerPrefillRouter::KvRouter(router) = &binding.router
        {
            router.chooser.register_workers(worker_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::Duration,
    };

    use async_trait::async_trait;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        component::Instance,
        discovery::{EndpointInstanceId, EventTransportKind},
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        engine::{AsyncEngine, AsyncEngineContext},
        pipeline::{
            AddressedRequest, Context, Error, ManyIn, ManyOut, PushRouter, ResponseStream,
            RouterMode, SingleIn, StreamingDispatch, context::Controller,
        },
        storage::kv,
    };
    use futures::{StreamExt, future::join_all};

    use super::*;
    use crate::{
        discovery::ModelManager,
        protocols::common::{
            FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest,
        },
        session_affinity::SessionAffinityPushRouter,
    };

    type LlmResponse = dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>;

    struct RecordingDispatch {
        worker_ids: Mutex<Vec<u64>>,
        pending_responses: bool,
    }

    impl RecordingDispatch {
        fn completed() -> Self {
            Self {
                worker_ids: Mutex::new(Vec::new()),
                pending_responses: false,
            }
        }

        fn pending() -> Self {
            Self {
                worker_ids: Mutex::new(Vec::new()),
                pending_responses: true,
            }
        }

        fn response_stream(&self) -> ManyOut<LlmResponse> {
            let context: Arc<dyn AsyncEngineContext> = Arc::new(Controller::default());
            if self.pending_responses {
                return ResponseStream::new(Box::pin(futures::stream::pending()), context);
            }
            ResponseStream::new(
                Box::pin(tokio_stream::iter(vec![LlmResponse::from_data(
                    LLMEngineOutput {
                        finish_reason: Some(FinishReason::EoS),
                        ..Default::default()
                    },
                )])),
                context,
            )
        }
    }

    #[async_trait]
    impl StreamingDispatch<PreprocessedRequest, LlmResponse> for RecordingDispatch {
        async fn generate(
            &self,
            request: SingleIn<AddressedRequest<PreprocessedRequest>>,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            let (addressed, _) = request.transfer(());
            let (_, _, instance) = addressed.into_parts();
            self.worker_ids
                .lock()
                .unwrap()
                .push(instance.expect("selected instance").id());
            Ok(self.response_stream())
        }

        async fn generate_bidirectional(
            &self,
            _instance: Instance,
            _address: String,
            _input: ManyIn<PreprocessedRequest>,
        ) -> Result<ManyOut<LlmResponse>, Error> {
            anyhow::bail!("bidirectional dispatch is unused in this test")
        }

        async fn on_instance_removed(&self, _id: &EndpointInstanceId) {}
    }

    fn distributed_config(root: &std::path::Path) -> DistributedConfig {
        DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(root.to_path_buf())),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            event_transport_kind: EventTransportKind::Zmq,
        }
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    async fn query_worker(router: &PrefillRouter) -> u64 {
        match router
            .query_prefill_worker(
                &[1, 2, 3],
                None,
                None,
                None,
                0.0,
                0,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap()
        {
            PrefillQueryOutcome::Routed { worker_id, dp_rank } => {
                assert_eq!(dp_rank, None);
                worker_id
            }
            PrefillQueryOutcome::QueueRejected { .. } => panic!("RR query cannot queue"),
        }
    }

    async fn shared_router(
        runtime: &Runtime,
        discovery_root: &std::path::Path,
        namespace: &str,
        mode: RouterMode,
        dispatch: Arc<RecordingDispatch>,
    ) -> (
        Arc<SessionAffinityPushRouter>,
        Arc<PrefillRouter>,
        Vec<DistributedRuntime>,
        Vec<u64>,
    ) {
        let component = "workers";
        let endpoint_name = "generate";
        let mut worker_runtimes = Vec::new();
        for _ in 0..4 {
            let worker_runtime =
                DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
                    .await
                    .unwrap();
            worker_runtime
                .namespace(namespace.to_string())
                .unwrap()
                .component(component.to_string())
                .unwrap()
                .endpoint(endpoint_name)
                .register_endpoint_instance()
                .await
                .unwrap();
            worker_runtimes.push(worker_runtime);
        }

        let router_runtime =
            DistributedRuntime::new(runtime.clone(), distributed_config(discovery_root))
                .await
                .unwrap();
        let client = router_runtime
            .namespace(namespace.to_string())
            .unwrap()
            .component(component.to_string())
            .unwrap()
            .endpoint(endpoint_name)
            .client()
            .await
            .unwrap();
        let instances = tokio::time::timeout(Duration::from_secs(5), async {
            let mut source = client.instance_source.as_ref().clone();
            loop {
                let instances = source.borrow_and_update().clone();
                if instances.len() == 4 {
                    return instances;
                }
                source
                    .changed()
                    .await
                    .expect("discovery source must remain open");
            }
        })
        .await
        .expect("all four workers must be discovered");
        let mut workers = instances.iter().map(Instance::id).collect::<Vec<_>>();
        workers.sort_unstable();
        let push_router = PushRouter::from_client_with_dispatch(client, mode, dispatch)
            .await
            .unwrap();
        let shared = Arc::new(SessionAffinityPushRouter::new(push_router, None, false).unwrap());
        let prefill = PrefillRouter::disabled(Arc::new(ModelManager::new()), mode, None);
        prefill.binding.store(Some(Arc::new(
            crate::kv_router::prefill_router::PrefillBinding {
                endpoint_id: dynamo_runtime::protocols::EndpointId {
                    namespace: namespace.to_string(),
                    component: component.to_string(),
                    name: endpoint_name.to_string(),
                },
                router: InnerPrefillRouter::SimpleRouter(shared.clone()),
                prefill_router_mode: mode,
            },
        )));
        prefill.lifecycle.store(
            PrefillLifecycleState::Active as u8,
            std::sync::atomic::Ordering::Release,
        );
        worker_runtimes.push(router_runtime);
        (shared, prefill, worker_runtimes, workers)
    }

    #[tokio::test]
    async fn rr_prefill_queries_do_not_advance_shared_dispatch_cursor() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();
        let namespace = "prefill-query-rr";
        let dispatch = Arc::new(RecordingDispatch::completed());
        let (shared_router, prefill_router, worker_runtimes, expected_workers) = shared_router(
            &runtime,
            discovery_root.path(),
            namespace,
            RouterMode::RoundRobin,
            dispatch.clone(),
        )
        .await;

        let concurrent_peeks = join_all((0..16).map(|_| query_worker(&prefill_router))).await;
        assert!(
            concurrent_peeks
                .iter()
                .all(|worker_id| *worker_id == expected_workers[0])
        );

        for expected_worker in &expected_workers {
            assert_eq!(query_worker(&prefill_router).await, *expected_worker);
            assert_eq!(query_worker(&prefill_router).await, *expected_worker);
            let mut stream = shared_router
                .generate(Context::new(request()))
                .await
                .unwrap();
            while stream.next().await.is_some() {}
        }

        assert_eq!(*dispatch.worker_ids.lock().unwrap(), expected_workers);
        drop(worker_runtimes);
        runtime.shutdown();
    }

    #[tokio::test]
    async fn advisory_prefill_query_never_acquires_local_occupancy() {
        let runtime = Runtime::from_current().unwrap();
        let discovery_root = tempfile::tempdir().unwrap();

        for (index, mode) in [
            RouterMode::PowerOfTwoChoices,
            RouterMode::LeastLoaded,
            RouterMode::DeviceAwareWeighted,
        ]
        .into_iter()
        .enumerate()
        {
            let dispatch = Arc::new(RecordingDispatch::pending());
            let (shared, prefill, runtimes, workers) = shared_router(
                &runtime,
                discovery_root.path(),
                &format!("prefill-query-occupancy-{index}"),
                mode,
                dispatch,
            )
            .await;

            let _ = join_all((0..16).map(|_| query_worker(&prefill))).await;
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                0,
                "{mode:?} advisory queries must not acquire occupancy"
            );

            let stream = shared.generate(Context::new(request())).await.unwrap();
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                1,
                "{mode:?} committed dispatch must retain exactly one lease"
            );
            drop(stream);
            assert_eq!(
                workers
                    .iter()
                    .map(|worker| shared.occupancy_for_test(*worker))
                    .sum::<u64>(),
                0,
                "{mode:?} dropping the response stream must release the lease"
            );
            drop(runtimes);
        }

        runtime.shutdown();
    }
}
