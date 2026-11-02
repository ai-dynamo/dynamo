// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::decision_profile::VLLM_CAUSAL_DECISION_V1;

use super::*;
use crate::{
    discovery::test_support,
    local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    model_type::{ModelInput, ModelType},
    worker_type::WorkerType,
};

#[tokio::test]
async fn serial_causal_siblings_survive_qualified_membership_churn() {
    assert_serial_transition(false).await;
}

#[tokio::test]
async fn serial_causal_siblings_reject_revocation_and_reopen() {
    assert_serial_transition(true).await;
}

async fn assert_serial_transition(revoke: bool) {
    fn frame(request: &PreprocessedRequest) -> (Value, Annotated<LLMEngineOutput>) {
        let native = request.extra_args.as_ref().unwrap()["vllm_tito"].clone();
        let labels: Vec<u32> =
            serde_json::from_value(native["sampling_params"]["logprob_token_ids"].clone()).unwrap();
        let terminal = super::vllm::terminal_output(&labels, request.token_ids.len());
        (native, terminal)
    }
    crate::generate_format::test_formats::install();
    let mut card = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    card.set_name("test-model");
    card.model_input = ModelInput::Tokens;
    card.model_type = ModelType::Chat;
    card.worker_type = Some(WorkerType::Aggregated);
    card.runtime_config.max_num_seqs = Some(1);
    card.runtime_config.data_parallel_start_rank = 3;
    for capability in [
        VLLM_CAUSAL_DECISION_V1,
        VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
    ] {
        card.runtime_config
            .runtime_data
            .insert(capability.into(), true.into());
    }
    let preprocessor = OpenAIPreprocessor::new(card.clone()).unwrap();
    let resume = Arc::new(Notify::new());
    let mut engine = TestEngine::with_frame(EngineBehavior::PendingSiblings, frame);
    Arc::get_mut(&mut engine).unwrap().resume_sibling = Some(resume.clone());
    let mut workers = WorkerSet::new("cohort".into(), card.mdcsum().into(), card);
    let resources = crate::discovery::test_support::token_resources(engine.clone(), preprocessor);
    let (_ordinary, ordinary_rx) = tokio::sync::watch::channel(vec![41, 42]);
    workers.set_instance_watcher(ordinary_rx);
    let qualification = (Profile::VllmCausalV1, Profile::VllmCausalV1.limits().1);
    let (publisher, fleet) = test_support::fleet(Some(qualification), vec![41]);
    test_support::attach_surface_for_test(&mut workers, resources, fleet).await;
    let selected =
        crate::discovery::test_support::prepared_selection_for_test("test-model".into(), &workers)
            .await
            .unwrap();
    let task = tokio::spawn(dispatch_selection_for_test(
        selected,
        (0..3).map(prepared_noul_branch).collect(),
        "membership-churn".into(),
        Context::new(()).context(),
        Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap(),
        Arc::new(super::super::Metrics::new()),
        canonical_request(3),
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if engine.observations.lock().unwrap().len() == 2 {
                break;
            }
            engine.started.notified().await;
        }
    })
    .await
    .unwrap();
    if revoke {
        test_support::publish(&publisher, None, vec![]);
    } else {
        test_support::publish(&publisher, Some(qualification), vec![41, 42]);
    }
    test_support::publish(&publisher, Some(qualification), vec![41]);
    resume.notify_one();
    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        response.status(),
        if revoke {
            StatusCode::SERVICE_UNAVAILABLE
        } else {
            StatusCode::OK
        }
    );
    let observed = engine.observations.lock().unwrap();
    assert_eq!(observed.len(), if revoke { 2 } else { 3 });
    assert!(
        observed
            .iter()
            .skip(1)
            .all(|branch| branch.routing.backend_instance_id == Some(41))
    );
}
