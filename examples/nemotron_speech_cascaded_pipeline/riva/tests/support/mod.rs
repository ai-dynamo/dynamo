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

use std::{future::Future, time::Duration};

use clap::Parser;
use dynamo_riva_sidecar::{
    client::RivaConnection,
    config::{Args, AsrConfig, Mode, TtsConfig},
    proto::{asr::*, tts::*},
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tokio_util::task::AbortOnDropHandle;
use tonic::{Request, Response, Status, Streaming};

pub async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("test operation timed out")
}

pub struct Call<Input, Output> {
    pub request: Request<Streaming<Input>>,
    pub responses: mpsc::Sender<Result<Output, Status>>,
}

type AsrCall = Call<StreamingRecognizeRequest, StreamingRecognizeResponse>;
type TtsCall = Call<SynthesizeSpeechRequest, SynthesizeSpeechResponse>;

#[derive(Clone)]
struct Riva {
    asr: mpsc::UnboundedSender<AsrCall>,
    tts: mpsc::UnboundedSender<TtsCall>,
}

#[tonic::async_trait]
impl riva_speech_recognition_server::RivaSpeechRecognition for Riva {
    type StreamingRecognizeStream = ReceiverStream<Result<StreamingRecognizeResponse, Status>>;

    async fn streaming_recognize(
        &self,
        request: Request<Streaming<StreamingRecognizeRequest>>,
    ) -> Result<Response<Self::StreamingRecognizeStream>, Status> {
        let (responses, rx) = mpsc::channel(32);
        self.asr
            .send(Call { request, responses })
            .map_err(|_| Status::unavailable("test receiver gone"))?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn recognize(
        &self,
        _: Request<RecognizeRequest>,
    ) -> Result<Response<RecognizeResponse>, Status> {
        Err(Status::unimplemented("unused unary ASR"))
    }

    async fn get_riva_speech_recognition_config(
        &self,
        _: Request<RivaSpeechRecognitionConfigRequest>,
    ) -> Result<Response<RivaSpeechRecognitionConfigResponse>, Status> {
        Err(Status::unimplemented("unused model config"))
    }
}

#[tonic::async_trait]
impl riva_speech_synthesis_server::RivaSpeechSynthesis for Riva {
    type SynthesizeOnlineStream = ReceiverStream<Result<SynthesizeSpeechResponse, Status>>;

    async fn synthesize_online(
        &self,
        request: Request<Streaming<SynthesizeSpeechRequest>>,
    ) -> Result<Response<Self::SynthesizeOnlineStream>, Status> {
        let (responses, rx) = mpsc::channel(32);
        self.tts
            .send(Call { request, responses })
            .map_err(|_| Status::unavailable("test receiver gone"))?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn synthesize(
        &self,
        _: Request<SynthesizeSpeechRequest>,
    ) -> Result<Response<SynthesizeSpeechResponse>, Status> {
        Err(Status::unimplemented("unused unary TTS"))
    }

    async fn get_riva_synthesis_config(
        &self,
        _: Request<RivaSynthesisConfigRequest>,
    ) -> Result<Response<RivaSynthesisConfigResponse>, Status> {
        Err(Status::unimplemented("unused model config"))
    }
}

pub struct Server {
    pub address: String,
    pub asr: mpsc::UnboundedReceiver<AsrCall>,
    pub tts: mpsc::UnboundedReceiver<TtsCall>,
    _task: AbortOnDropHandle<()>,
}

impl Server {
    pub async fn start() -> Self {
        Self::start_with_tls(None).await
    }

    pub async fn start_with_tls(identity: Option<tonic::transport::Identity>) -> Self {
        if identity.is_some() {
            let _ = rustls::crypto::ring::default_provider().install_default();
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let (asr_tx, asr) = mpsc::unbounded_channel();
        let (tts_tx, tts) = mpsc::unbounded_channel();
        let service = Riva {
            asr: asr_tx,
            tts: tts_tx,
        };
        let task = tokio::spawn(async move {
            let mut builder = tonic::transport::Server::builder();
            if let Some(identity) = identity {
                builder = builder
                    .tls_config(tonic::transport::ServerTlsConfig::new().identity(identity))
                    .unwrap();
            }
            builder
                .add_service(
                    riva_speech_recognition_server::RivaSpeechRecognitionServer::new(
                        service.clone(),
                    ),
                )
                .add_service(riva_speech_synthesis_server::RivaSpeechSynthesisServer::new(service))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        Self {
            address,
            asr,
            tts,
            _task: AbortOnDropHandle::new(task),
        }
    }

    pub fn asr_config(&self) -> AsrConfig {
        let Mode::Asr(config) = Args::parse_from([
            "sidecar",
            "asr",
            "--nim-server",
            &self.address,
            "--nim-startup-timeout-s",
            "2",
            "--timeout-s",
            "0.5",
        ])
        .mode
        else {
            unreachable!()
        };
        config
    }

    pub fn tts_config(&self) -> TtsConfig {
        let Mode::Tts(config) = Args::parse_from([
            "sidecar",
            "tts",
            "--nim-server",
            &self.address,
            "--nim-startup-timeout-s",
            "2",
        ])
        .mode
        else {
            unreachable!()
        };
        config
    }

    pub async fn connection(&self) -> RivaConnection {
        within(RivaConnection::connect(
            &self.asr_config().worker.connection,
        ))
        .await
        .unwrap()
    }
}

pub fn transcript(text: &str, is_final: bool) -> StreamingRecognizeResponse {
    StreamingRecognizeResponse {
        results: vec![StreamingRecognitionResult {
            alternatives: vec![SpeechRecognitionAlternative {
                transcript: text.into(),
                ..Default::default()
            }],
            is_final,
            ..Default::default()
        }],
        ..Default::default()
    }
}
