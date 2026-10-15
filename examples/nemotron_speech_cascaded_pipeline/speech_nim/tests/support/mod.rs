// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use clap::Parser;
use dynamo_sidecar_common::GrpcChannelPool;
use dynamo_speech_nim_sidecar::{
    client,
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
struct SpeechNim {
    ready: Arc<AtomicBool>,
    asr: mpsc::UnboundedSender<AsrCall>,
    tts: mpsc::UnboundedSender<TtsCall>,
}

#[tonic::async_trait]
impl riva_speech_recognition_server::RivaSpeechRecognition for SpeechNim {
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
        Ok(Response::new(RivaSpeechRecognitionConfigResponse {
            model_config: if self.ready.load(Ordering::SeqCst) {
                vec![riva_speech_recognition_config_response::Config {
                    model_name: "asr".into(),
                    ..Default::default()
                }]
            } else {
                vec![]
            },
        }))
    }
}

#[tonic::async_trait]
impl riva_speech_synthesis_server::RivaSpeechSynthesis for SpeechNim {
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
        Ok(Response::new(RivaSynthesisConfigResponse {
            model_config: if self.ready.load(Ordering::SeqCst) {
                vec![riva_synthesis_config_response::Config {
                    model_name: "tts".into(),
                    ..Default::default()
                }]
            } else {
                vec![]
            },
        }))
    }
}

pub struct Server {
    pub ready: Arc<AtomicBool>,
    port: u16,
    pub asr: mpsc::UnboundedReceiver<AsrCall>,
    pub tts: mpsc::UnboundedReceiver<TtsCall>,
    _task: AbortOnDropHandle<()>,
}

impl Server {
    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (asr_tx, asr) = mpsc::unbounded_channel();
        let (tts_tx, tts) = mpsc::unbounded_channel();
        let ready = Arc::new(AtomicBool::new(true));
        let service = SpeechNim {
            ready: ready.clone(),
            asr: asr_tx,
            tts: tts_tx,
        };
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
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
            ready,
            port,
            asr,
            tts,
            _task: AbortOnDropHandle::new(task),
        }
    }

    pub fn asr_config(&self) -> AsrConfig {
        let Mode::Asr(config) = Args::parse_from([
            "sidecar",
            "asr",
            "--grpc-endpoint",
            &format!("127.0.0.1:{}", self.port),
            "--grpc-startup-deadline-secs",
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
            "--grpc-endpoint",
            &format!("127.0.0.1:{}", self.port),
            "--grpc-startup-deadline-secs",
            "2",
        ])
        .mode
        else {
            unreachable!()
        };
        config
    }

    pub async fn connection(&self) -> Arc<GrpcChannelPool> {
        within(client::connect(&Mode::Asr(self.asr_config())))
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
