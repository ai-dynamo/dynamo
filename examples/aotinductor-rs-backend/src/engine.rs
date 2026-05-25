// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal AOTInductor backend.
//!
//! The engine maps preprocessed token IDs to a fixed-width float tensor, runs
//! an AOTInductor `.pt2` package through `pierric/aotinductor-rs`, then maps
//! the first output tensor back to token IDs.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use aotinductor::ModelPackage;
use async_trait::async_trait;
use clap::Parser;
use dynamo_backend_common::{
    BackendError, CommonArgs, DynamoError, EngineConfig, ErrorType, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tch::Tensor;

#[derive(clap::Parser, Debug)]
#[command(
    name = env!("CARGO_BIN_NAME"),
    about = "Dynamo Rust backend example backed by a PyTorch AOTInductor package."
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Path to an AOTInductor .pt2 package.
    #[arg(long, value_name = "MODEL_PT2")]
    model_package: PathBuf,

    /// Public-facing model name advertised to Dynamo clients.
    #[arg(long, default_value = "aotinductor-tiny")]
    model_name: String,

    /// HF repo or local tokenizer path used by a Dynamo frontend. The
    /// backend consumes token IDs; this is only model-card metadata.
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model_path: String,

    /// Fixed input tensor width. The example exporter creates a [1, 4] model.
    #[arg(long, default_value_t = 4)]
    input_width: usize,
}

pub struct AotInductorBackend {
    model_name: String,
    model_package: PathBuf,
    input_width: usize,
    runner: Arc<Mutex<Option<ModelRunner>>>,
}

struct ModelRunner {
    tx: mpsc::Sender<ModelCommand>,
    handle: JoinHandle<()>,
}

enum ModelCommand {
    Run {
        input: Vec<f32>,
        input_width: usize,
        reply: mpsc::Sender<Result<Vec<u32>, String>>,
    },
    Shutdown,
}

impl AotInductorBackend {
    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(a) => <Args as Parser>::try_parse_from(a),
            None => <Args as Parser>::try_parse(),
        }
        .map_err(|e| invalid_arg(e.to_string()))?;

        let engine = Self::new(
            args.model_name.clone(),
            args.model_package,
            args.input_width,
        )?;

        let config = WorkerConfig {
            namespace: args.common.namespace,
            component: args.common.component,
            endpoint: args.common.endpoint,
            endpoint_types: args.common.endpoint_types,
            custom_jinja_template: args.common.custom_jinja_template,
            disaggregation_mode: args.common.disaggregation_mode,
            model_name: args.model_path,
            served_model_name: Some(args.model_name),
            ..Default::default()
        };
        Ok((engine, config))
    }

    fn new(
        model_name: String,
        model_package: PathBuf,
        input_width: usize,
    ) -> Result<Self, DynamoError> {
        if input_width == 0 {
            return Err(invalid_arg("--input-width must be greater than zero"));
        }

        Ok(Self {
            model_name,
            model_package,
            input_width,
            runner: Arc::new(Mutex::new(None)),
        })
    }
}

#[async_trait]
impl LLMEngine for AotInductorBackend {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self
            .runner
            .lock()
            .map_err(|_| engine_shutdown("model runner mutex poisoned"))?
            .is_some()
        {
            return Err(engine_shutdown("AOTInductor backend already started"));
        }

        let package_path = self
            .model_package
            .to_str()
            .ok_or_else(|| invalid_arg("model package path must be valid UTF-8"))?;
        let runner = spawn_model_runner(package_path.to_string())?;

        let mut guard = self
            .runner
            .lock()
            .map_err(|_| engine_shutdown("model runner mutex poisoned"))?;
        if guard.is_some() {
            runner.shutdown();
            return Err(engine_shutdown("AOTInductor backend already started"));
        }
        *guard = Some(runner);

        tracing::info!(
            model = %self.model_name,
            package = %self.model_package.display(),
            input_width = self.input_width,
            "AOTInductor backend started"
        );

        Ok(EngineConfig {
            model: self.model_name.clone(),
            served_model_name: Some(self.model_name.clone()),
            context_length: Some(self.input_width as u32),
            ..Default::default()
        })
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let runner = self.runner.clone();
        let input_width = self.input_width;
        let prompt_tokens = request.token_ids.len() as u32;
        let max_tokens = request
            .stop_conditions
            .max_tokens
            .map(|n| n as usize)
            .unwrap_or(input_width);
        let input_tokens = request.token_ids;

        Ok(Box::pin(async_stream::stream! {
            if ctx.is_stopped() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            }

            if max_tokens == 0 {
                yield Ok(LLMEngineOutput::length().with_usage(usage(prompt_tokens, 0)));
                return;
            }

            let all_tokens = match run_model(&runner, input_width, &input_tokens) {
                Ok(tokens) => tokens,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            if ctx.is_stopped() {
                yield Ok(LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0)));
                return;
            }

            let completion: Vec<u32> = all_tokens.into_iter().take(max_tokens).collect();
            let completion_tokens = completion.len() as u32;
            let terminal = if completion_tokens as usize == max_tokens {
                LLMEngineOutput::length()
            } else {
                LLMEngineOutput::stop()
            }
            .with_tokens(completion)
            .with_usage(usage(prompt_tokens, completion_tokens));

            yield Ok(terminal);
        }))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        let mut guard = self
            .runner
            .lock()
            .map_err(|_| engine_shutdown("model runner mutex poisoned"))?;
        if let Some(runner) = guard.take() {
            runner.shutdown();
        }
        Ok(())
    }
}

impl ModelRunner {
    fn run(&self, input: Vec<f32>, input_width: usize) -> Result<Vec<u32>, String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(ModelCommand::Run {
                input,
                input_width,
                reply: reply_tx,
            })
            .map_err(|_| "AOTInductor model thread is not accepting requests".to_string())?;
        reply_rx
            .recv()
            .map_err(|_| "AOTInductor model thread stopped before replying".to_string())?
    }

    fn shutdown(self) {
        let _ = self.tx.send(ModelCommand::Shutdown);
        let _ = self.handle.join();
    }
}

fn spawn_model_runner(package_path: String) -> Result<ModelRunner, DynamoError> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<ModelCommand>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let thread_package_path = package_path.clone();

    let handle = thread::spawn(move || {
        let package = match ModelPackage::new(&thread_package_path) {
            Ok(package) => {
                let _ = ready_tx.send(Ok(()));
                package
            }
            Err(err) => {
                let _ = ready_tx.send(Err(format!(
                    "failed to load AOTInductor package {thread_package_path}: {err}"
                )));
                return;
            }
        };

        while let Ok(command) = cmd_rx.recv() {
            match command {
                ModelCommand::Run {
                    input,
                    input_width,
                    reply,
                } => {
                    let _ = reply.send(run_loaded_model(&package, input, input_width));
                }
                ModelCommand::Shutdown => break,
            }
        }
    });

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(ModelRunner { tx: cmd_tx, handle }),
        Ok(Err(message)) => {
            let _ = handle.join();
            Err(backend_error(BackendError::InvalidArgument, message))
        }
        Err(_) => {
            let _ = handle.join();
            Err(engine_shutdown(format!(
                "AOTInductor model thread exited while loading {package_path}"
            )))
        }
    }
}

fn run_model(
    runner: &Arc<Mutex<Option<ModelRunner>>>,
    input_width: usize,
    token_ids: &[u32],
) -> Result<Vec<u32>, DynamoError> {
    let mut input = vec![0.0_f32; input_width];
    for (dst, src) in input.iter_mut().zip(token_ids.iter().copied()) {
        *dst = src as f32;
    }

    let guard = runner
        .lock()
        .map_err(|_| engine_shutdown("model runner mutex poisoned"))?;
    let runner = guard
        .as_ref()
        .ok_or_else(|| engine_shutdown("generate called before start"))?;
    runner
        .run(input, input_width)
        .map_err(|e| backend_error(BackendError::Unknown, e))
}

fn run_loaded_model(
    package: &ModelPackage,
    input: Vec<f32>,
    input_width: usize,
) -> Result<Vec<u32>, String> {
    let tensor = Tensor::from_slice(&input).reshape([1, input_width as i64]);
    let outputs = package.run(&vec![tensor]);
    let Some(first) = outputs.first() else {
        return Err("AOTInductor package returned no output tensors".to_string());
    };

    let flat = first.reshape([-1]);
    let values: Vec<f32> = (&flat)
        .try_into()
        .map_err(|e| format!("failed to convert AOTInductor output tensor: {e}"))?;

    Ok(values.into_iter().map(float_to_token).collect())
}

fn float_to_token(value: f32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f32 {
        u32::MAX
    } else {
        value.round() as u32
    }
}

fn invalid_arg(message: impl Into<String>) -> DynamoError {
    backend_error(BackendError::InvalidArgument, message)
}

fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend_error(BackendError::EngineShutdown, message)
}

fn backend_error(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message.into())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::testing::{mock_context, run_conformance};
    use futures::StreamExt;

    fn package_path() -> Option<PathBuf> {
        std::env::var_os("AOTINDUCTOR_TEST_PACKAGE").map(PathBuf::from)
    }

    #[tokio::test]
    async fn generate_maps_tiny_model_output_to_tokens() {
        let Some(package) = package_path() else {
            eprintln!("skipping: AOTINDUCTOR_TEST_PACKAGE is not set");
            return;
        };

        let engine = AotInductorBackend::new("aot-test".to_string(), package, 4).unwrap();
        engine.start(0).await.unwrap();

        let request = PreprocessedRequest::builder()
            .model("aot-test".to_string())
            .token_ids(vec![0, 1, 2, 3])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        let stream = engine
            .generate(request, GenerateContext::new(mock_context(), None))
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        let terminal = chunks.into_iter().last().unwrap().unwrap();

        assert_eq!(terminal.token_ids, vec![1, 3, 5, 7]);
        assert!(terminal.finish_reason.is_some());
        engine.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn conformance_when_package_is_available() {
        let Some(package) = package_path() else {
            eprintln!("skipping: AOTINDUCTOR_TEST_PACKAGE is not set");
            return;
        };

        run_conformance(|| {
            AotInductorBackend::new("aot-test".to_string(), package.clone(), 4).unwrap()
        })
        .await
        .unwrap();
    }
}
