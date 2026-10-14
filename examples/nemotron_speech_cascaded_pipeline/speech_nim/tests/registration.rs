// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Own process: the runtime TCP server is process-global.
mod support;

use std::time::Duration;

use dynamo_llm::{
    model_card::ModelDeploymentCard,
    model_type::{ModelInput, ModelType},
    worker_type::WorkerType,
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    discovery::{DiscoveryInstance, DiscoveryQuery},
    distributed::DistributedConfig,
    engine::AsyncEngine,
    pipeline::{
        Context, RequestStream,
        network::egress::push_router::{PushRouter, RouterMode},
    },
    protocols::annotated::Annotated,
};
use dynamo_speech_nim_sidecar::{config::Mode, proto::tts::SynthesizeSpeechResponse, worker};
use futures::StreamExt;
use serde_json::{Value, json};
use support::{Server, transcript, within};
use tokio_util::{sync::CancellationToken, task::AbortOnDropHandle};

#[tokio::test]
async fn speech_workers_register_without_assets_and_serve_over_dynamo_tcp() {
    let mut server = Server::start().await;
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let namespace = drt.namespace("speech-nim-example-test").unwrap();
    let asr_endpoint = namespace.component("asr").unwrap().endpoint("generate");
    let tts_endpoint = namespace.component("tts").unwrap().endpoint("generate");
    let mut asr = server.asr_config();
    asr.worker.endpoint = Some("speech-nim-example-test.asr.generate".into());
    let mut tts = server.tts_config();
    tts.worker.endpoint = Some("speech-nim-example-test.tts.generate".into());
    let stop_asr = CancellationToken::new();
    let stop_tts = CancellationToken::new();
    let asr_task = AbortOnDropHandle::new(tokio::spawn(worker::serve(
        Mode::Asr(asr),
        server.connection().await,
        drt.clone(),
        stop_asr.clone(),
    )));
    let tts_task = AbortOnDropHandle::new(tokio::spawn(worker::serve(
        Mode::Tts(tts),
        server.connection().await,
        drt.clone(),
        stop_tts.clone(),
    )));
    let cards = within(async {
        loop {
            let cards = drt
                .discovery()
                .list(DiscoveryQuery::AllModels)
                .await
                .unwrap();
            if cards.len() == 2 {
                break cards;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    for instance in cards {
        let DiscoveryInstance::Model {
            component,
            card_json,
            ..
        } = instance
        else {
            panic!("expected model card")
        };
        let card: ModelDeploymentCard = serde_json::from_value(card_json).unwrap();
        assert_eq!(card.model_input, ModelInput::Text);
        assert_eq!(card.worker_type, Some(WorkerType::Aggregated));
        assert!(card.tokenizer.is_none());
        assert!(card.model_info.is_none());
        if component == "asr" {
            assert_eq!(card.display_name, "nemotron-asr-streaming");
            assert_eq!(card.model_type, ModelType::Realtime);
        } else {
            assert_eq!(card.display_name, "nvidia/magpie-tts-multilingual");
            assert_eq!(card.model_type, ModelType::Audios);
        }
    }
    let client = asr_endpoint.client().await.unwrap();
    within(client.wait_for_instances()).await.unwrap();
    let router = PushRouter::<Value, Annotated<Value>>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let request = Context::new(RequestStream::new(futures::stream::iter(vec![
        json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "nemotron-asr-streaming"}
        }}}}),
        json!({"type": "input_audio_buffer.append", "audio": "AQACAA=="}),
        json!({"type": "input_audio_buffer.commit"}),
    ]).boxed()));
    let output = tokio::spawn(async move {
        router
            .generate(request)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });
    let mut call = within(server.asr.recv()).await.unwrap();
    while within(call.request.get_mut().message())
        .await
        .unwrap()
        .is_some()
    {}
    call.responses
        .send(Ok(transcript("over Dynamo", true)))
        .await
        .unwrap();
    drop(call.responses);
    let output = within(output).await.unwrap();
    let events: Vec<_> = output
        .into_iter()
        .filter_map(|v| v.into_data().unwrap())
        .collect();
    assert_eq!(events.first().unwrap()["type"], "session.updated");
    assert_eq!(events.last().unwrap()["transcript"], "over Dynamo");

    let client = tts_endpoint.client().await.unwrap();
    within(client.wait_for_instances()).await.unwrap();
    let router = PushRouter::<Value, Annotated<Value>>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let output = tokio::spawn(async move {
        router
            .generate(Context::new(
                json!({"input": "Hello", "response_format": "pcm"}),
            ))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });
    let mut call = within(server.tts.recv()).await.unwrap();
    assert_eq!(
        within(call.request.get_mut().message())
            .await
            .unwrap()
            .unwrap()
            .text,
        "Hello"
    );
    call.responses
        .send(Ok(SynthesizeSpeechResponse {
            audio: vec![1, 0, 2, 0],
            ..Default::default()
        }))
        .await
        .unwrap();
    drop(call.responses);
    let output = within(output).await.unwrap();
    let responses: Vec<_> = output
        .into_iter()
        .filter_map(|v| v.into_data().unwrap())
        .collect();
    assert_eq!(responses[0]["data"][0]["b64_json"], "AQACAA==");

    stop_asr.cancel();
    within(asr_task).await.unwrap().unwrap();
    assert!(
        drt.discovery()
            .list(DiscoveryQuery::ComponentModels {
                namespace: "speech-nim-example-test".into(),
                component: "asr".into(),
            })
            .await
            .unwrap()
            .is_empty()
    );
    stop_tts.cancel();
    // Worker includes the normal five-second discovery propagation grace.
    tokio::time::timeout(Duration::from_secs(15), tts_task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    runtime.shutdown();
}
