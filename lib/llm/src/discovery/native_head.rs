// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::{
    component::{Client, Instance},
    discovery::EndpointInstanceId,
    pipeline::{
        AddressedPushRouter, AddressedRequest, Error, ManyIn, ManyOut, SingleIn, StreamingDispatch,
    },
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::{Model, SystemOneSelectionError, WorkerSet};
use crate::{
    model_card::ModelDeploymentCard,
    protocols::{
        Annotated,
        systemone::head::{HeadEngine, HeadRequest, HeadResponse},
    },
};

#[derive(Clone)]
pub(crate) struct NativeHeadSelection {
    pub(crate) canonical_model: String,
    pub(crate) engine: HeadEngine,
    pub(crate) card: ModelDeploymentCard,
}

impl Model {
    pub(crate) fn get_native_head_selection(
        &self,
    ) -> Result<NativeHeadSelection, SystemOneSelectionError> {
        self.select_worker_set_with_worker_count(
            |workers| {
                workers.card().is_native_head_eligible().then_some(())?;
                Some(NativeHeadSelection {
                    canonical_model: self.name().into(),
                    engine: workers.native_head_engine.clone()?,
                    card: workers.card().clone(),
                })
            },
            WorkerSet::decision_worker_count,
        )
        .ok_or_else(|| {
            if self
                .worker_sets()
                .iter()
                .any(|workers| workers.card().is_native_head_eligible())
            {
                SystemOneSelectionError::Unavailable
            } else {
                SystemOneSelectionError::Unsupported
            }
        })
    }
}

pub(super) struct HeadDispatch {
    inner: Arc<dyn StreamingDispatch<HeadRequest, Annotated<HeadResponse>>>,
    admitted: watch::Receiver<Vec<u64>>,
    cancellation: CancellationToken,
}

impl HeadDispatch {
    pub(super) async fn for_client(
        client: &Client,
        admitted: watch::Receiver<Vec<u64>>,
        cancellation: CancellationToken,
    ) -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            inner: AddressedPushRouter::from_runtime_provider(&client.endpoint).await?,
            admitted,
            cancellation,
        }))
    }

    fn validate(&self, worker: u64) -> Result<(), Error> {
        if self.cancellation.is_cancelled()
            || self.admitted.has_changed().is_err()
            || !self.admitted.borrow().contains(&worker)
        {
            return Err(dynamo_runtime::error::DynamoError::builder()
                .error_type(dynamo_runtime::error::ErrorType::Unavailable)
                .message("selected native decision worker is unavailable")
                .build()
                .into());
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamingDispatch<HeadRequest, Annotated<HeadResponse>> for HeadDispatch {
    async fn generate(
        &self,
        request: SingleIn<AddressedRequest<HeadRequest>>,
    ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
        let timeout = request.establish_timeout();
        let (addressed, envelope) = request.transfer(());
        let (wire, address, instance) = addressed.into_parts();
        let instance = instance
            .ok_or_else(|| anyhow::anyhow!("native decision worker identity is missing"))?;
        self.validate(instance.id())?;
        let addressed = AddressedRequest::with_instance(wire, address, instance)
            .with_establish_timeout(timeout);
        self.inner.generate(envelope.map(|()| addressed)).await
    }

    async fn generate_bidirectional(
        &self,
        _instance: Instance,
        _address: String,
        _input: ManyIn<HeadRequest>,
    ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
        anyhow::bail!("native decisions require a unary request")
    }

    async fn on_instance_removed(&self, id: &EndpointInstanceId) {
        self.inner.on_instance_removed(id).await;
    }
    async fn on_instance_added(&self, id: &EndpointInstanceId) {
        self.inner.on_instance_added(id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{component::TransportType, engine::AsyncEngine, pipeline::Context};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Sink(AtomicUsize);

    #[async_trait::async_trait]
    impl StreamingDispatch<HeadRequest, Annotated<HeadResponse>> for Sink {
        async fn generate(
            &self,
            _request: SingleIn<AddressedRequest<HeadRequest>>,
        ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
            self.0.fetch_add(1, Ordering::Relaxed);
            anyhow::bail!("fixture forwarded")
        }
        async fn generate_bidirectional(
            &self,
            _instance: Instance,
            _address: String,
            _input: ManyIn<HeadRequest>,
        ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
            anyhow::bail!("fixture unary only")
        }
    }

    #[async_trait::async_trait]
    impl AsyncEngine<SingleIn<HeadRequest>, ManyOut<Annotated<HeadResponse>>, Error> for Sink {
        async fn generate(
            &self,
            _request: SingleIn<HeadRequest>,
        ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
            anyhow::bail!("selection fixture does not execute")
        }
    }

    fn wire() -> HeadRequest {
        serde_json::from_value(json!({"decision_head":{"protocol_version":1,"profile":"vllm_clef_flash_joint_decision_v1","model":"model","input":"state","questions":[],"temperature":1,"cache_salt":"salt","max_input_tokens":10,"max_tokenization_work":100,"max_encoding_bytes":1000}})).unwrap()
    }

    fn addressed(id: u64) -> SingleIn<AddressedRequest<HeadRequest>> {
        Context::new(AddressedRequest::with_instance(
            wire(),
            "unused".into(),
            Instance {
                namespace: "test".into(),
                component: "worker".into(),
                endpoint: "generate".into(),
                instance_id: id,
                transport: TransportType::Tcp("unused".into()),
                device_type: None,
                request_plane_codec: None,
            },
        ))
    }

    #[tokio::test]
    async fn final_dispatch_rechecks_original_admission_and_retired_lifecycles() {
        let sink = Arc::new(Sink::default());
        let (tx, rx) = watch::channel(vec![7]);
        let cancellation = CancellationToken::new();
        let dispatch = HeadDispatch {
            inner: sink.clone(),
            admitted: rx,
            cancellation: cancellation.clone(),
        };
        assert!(dispatch.generate(addressed(7)).await.is_err());
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
        tx.send_replace(vec![8]);
        assert!(dispatch.generate(addressed(7)).await.is_err());
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
        cancellation.cancel();
        assert!(dispatch.generate(addressed(8)).await.is_err());
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
        drop(tx);
        assert!(dispatch.validate(8).is_err());
        let (closed_tx, closed_rx) = watch::channel(vec![7]);
        drop(closed_tx);
        let closed = HeadDispatch {
            inner: sink.clone(),
            admitted: closed_rx,
            cancellation: CancellationToken::new(),
        };
        assert!(closed.generate(addressed(7)).await.is_err());
        assert_eq!(sink.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn native_card_selection_is_bound_to_qualified_admitted_assets() {
        use crate::{
            model_type::{ModelInput, ModelType},
            protocols::systemone::head::CLEF_FLASH_PROFILE,
            worker_type::WorkerType,
        };
        let mut card = ModelDeploymentCard::with_name_only("head-model");
        card.model_input = ModelInput::Text;
        card.model_type = ModelType::Classify | ModelType::Pooling;
        card.worker_type = Some(WorkerType::Aggregated);
        let baseline = card.mdcsum().to_owned();
        let mut value = serde_json::to_value(&card).unwrap();
        value["runtime_config"]["runtime_data"][CLEF_FLASH_PROFILE] = json!(true);
        value["runtime_config"]["runtime_data"]["vllm_native_decision_max_candidates"] = json!(26);
        card = serde_json::from_value(value).unwrap();
        assert!(card.is_native_head_eligible());
        assert_ne!(baseline, card.mdcsum());
        let mut workers = WorkerSet::new("ns".into(), card.mdcsum().into(), card);
        workers.native_head_engine = Some(Arc::new(Sink::default()));
        let (tx, rx) = watch::channel(vec![7]);
        workers.set_instance_watcher(rx);
        let model = Model::new("canonical".into());
        model.add_worker_set("ns".into(), Arc::new(workers));
        assert_eq!(
            model.get_native_head_selection().unwrap().canonical_model,
            "canonical"
        );
        tx.send_replace(Vec::new());
        assert!(matches!(
            model.get_native_head_selection(),
            Err(SystemOneSelectionError::Unavailable)
        ));
    }
}
