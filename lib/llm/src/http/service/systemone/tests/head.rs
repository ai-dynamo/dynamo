// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{
    discovery::native_head::NativeHeadSelection,
    protocols::systemone::head::{CLEF_FLASH_PROFILE, HeadRequest, HeadResponse},
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct HeadFixture {
    duplicate: bool,
    pending: bool,
    calls: AtomicUsize,
}

#[async_trait]
impl AsyncEngine<SingleIn<HeadRequest>, ManyOut<Annotated<HeadResponse>>, Error> for HeadFixture {
    async fn generate(
        &self,
        request: SingleIn<HeadRequest>,
    ) -> Result<ManyOut<Annotated<HeadResponse>>, Error> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let context = request.context();
        if self.pending {
            return Ok(ResponseStream::new(Box::pin(stream::pending()), context));
        }
        let outcomes:Vec<_>=request.decision_head.questions.iter().map(|question|json!({"ordinal":question.ordinal,"probabilities":vec![1.0/question.candidates.len() as f64;question.candidates.len()]})).collect();
        let response:HeadResponse=serde_json::from_value(json!({"protocol_version":1,"profile":CLEF_FLASH_PROFILE,"finish_reason":"stop","outcomes":outcomes,"usage":{"input_tokens":4,"output_tokens":0,"cached_tokens":0}})).unwrap();
        let frames = vec![Annotated::from_data(response); if self.duplicate { 2 } else { 1 }];
        Ok(ResponseStream::new(Box::pin(stream::iter(frames)), context))
    }
}

fn request(dialect: dynamo_decisions::Dialect) -> dynamo_decisions::CanonicalRequest {
    let mut request = canonical_request(2).as_ref().clone();
    request.dialect = dialect;
    request.temperature = 1.0;
    request
}

fn selected(engine: Arc<HeadFixture>) -> NativeHeadSelection {
    let mut card = crate::model_card::ModelDeploymentCard::with_name_only("native-model");
    card.runtime_config.context_length = Some(128);
    NativeHeadSelection {
        canonical_model: "native-model".into(),
        engine,
        card,
    }
}

#[tokio::test]
async fn native_http_execution_projects_both_contracts_and_measured_usage() {
    for dialect in [
        dynamo_decisions::Dialect::OpenAi,
        dynamo_decisions::Dialect::Jev,
    ] {
        let service = super::super::service_v2::HttpService::builder()
            .build()
            .unwrap();
        let engine = Arc::new(HeadFixture {
            duplicate: false,
            pending: false,
            calls: AtomicUsize::new(0),
        });
        let state = service.state_clone();
        let admission = state.systemone_admission();
        let initial = admission.available_permits();
        let response = super::super::head::handle(
            state,
            "native-id".into(),
            request(dialect),
            selected(engine.clone()),
            tokio::time::Instant::now() + Duration::from_secs(2),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = response_body(response).await;
        assert_eq!(value["model"], "native-model");
        assert_eq!(value["usage"]["output_tokens"], 0);
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
        assert_eq!(admission.available_permits(), initial);
    }
}

#[tokio::test]
async fn native_multiple_terminal_frames_fail_without_normal_answers() {
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let engine = Arc::new(HeadFixture {
        duplicate: true,
        pending: false,
        calls: AtomicUsize::new(0),
    });
    let response = super::super::head::handle(
        service.state_clone(),
        "duplicate".into(),
        request(dynamo_decisions::Dialect::OpenAi),
        selected(engine),
        tokio::time::Instant::now() + Duration::from_secs(2),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response_body(response).await.get("answers").is_none());
}

#[tokio::test]
async fn native_deadline_drains_local_dispatch_and_releases_admission() {
    let service = super::super::service_v2::HttpService::builder()
        .build()
        .unwrap();
    let state = service.state_clone();
    let admission = state.systemone_admission();
    let initial = admission.available_permits();
    let engine = Arc::new(HeadFixture {
        duplicate: false,
        pending: true,
        calls: AtomicUsize::new(0),
    });
    let response = super::super::head::handle(
        state,
        "timeout".into(),
        request(dynamo_decisions::Dialect::OpenAi),
        selected(engine),
        tokio::time::Instant::now() + Duration::from_millis(30),
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if admission.available_permits() == initial {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
