// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{net::IpAddr, num::NonZeroUsize, time::Duration};

use clap::{Args as ClapArgs, Parser, Subcommand};
use dynamo_backend_common::{RuntimeConfig, args::parse_worker_namespace};
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportArgs};

#[derive(Parser)]
#[command(about = "Dynamo Speech NIM sidecar for ASR and TTS")]
pub struct Args {
    #[command(subcommand)]
    pub mode: Mode,
}

#[derive(Subcommand)]
pub enum Mode {
    /// Serve OpenAI realtime transcription sessions.
    Asr(AsrConfig),
    /// Serve streaming PCM through /v1/audio/speech.
    Tts(TtsConfig),
}

impl Mode {
    pub fn worker(&self) -> &WorkerArgs {
        match self {
            Self::Asr(config) => &config.worker,
            Self::Tts(config) => &config.worker,
        }
    }
}

#[derive(Clone, ClapArgs)]
pub struct ConnectionConfig {
    /// Plaintext gRPC endpoint of the Speech NIM in this pod or on this host.
    #[arg(long, env = "DYN_SIDECAR_GRPC_ENDPOINT", default_value = "127.0.0.1:50051", value_parser = local_grpc_endpoint)]
    pub grpc_endpoint: GrpcEndpoint,
    #[command(flatten)]
    pub grpc: GrpcTransportArgs,
}

#[derive(Clone, ClapArgs)]
pub struct WorkerArgs {
    #[command(flatten)]
    pub connection: ConnectionConfig,
    #[arg(long, env = "DYN_NAMESPACE", default_value = "dynamo", value_parser = parse_worker_namespace)]
    pub namespace: String,
    /// Defaults to speech-asr-realtime or speech-tts for the selected mode.
    #[arg(long, env = "DYN_COMPONENT")]
    pub component: Option<String>,
    #[arg(long, env = "DYN_ENDPOINT", default_value = "generate")]
    pub endpoint: String,
    #[command(flatten)]
    pub runtime: RuntimeConfig,
}

#[derive(Clone, ClapArgs)]
pub struct AsrConfig {
    #[command(flatten)]
    pub worker: WorkerArgs,
    #[arg(long, default_value = "nemotron-asr-streaming")]
    pub model_name: String,
    #[arg(long, default_value = "en-US")]
    pub language_code: String,
    /// Empty lets the NIM choose its recognition model.
    #[arg(long, default_value = "")]
    pub nim_model: String,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=60_000))]
    pub commit_padding_ms: u32,
    /// Completion deadline after the input turn closes.
    #[arg(long, default_value = "30", value_parser = duration)]
    pub timeout_s: Duration,
    #[arg(long, default_value = "32")]
    pub max_concurrent_turns: NonZeroUsize,
}

#[derive(Clone, ClapArgs)]
pub struct TtsConfig {
    #[command(flatten)]
    pub worker: WorkerArgs,
    #[arg(long, default_value = "nvidia/magpie-tts-multilingual")]
    pub model_name: String,
    #[arg(long, default_value = "Magpie-Multilingual.EN-US.Aria")]
    pub voice: String,
    #[arg(long, default_value = "en-US")]
    pub language_code: String,
    #[arg(long, default_value_t = 24_000, value_parser = clap::value_parser!(i32).range(1..))]
    pub sample_rate_hz: i32,
}

fn duration(value: &str) -> Result<Duration, String> {
    let seconds = value.parse::<f64>().map_err(|_| "expected seconds")?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("seconds must be finite and positive".into());
    }
    Duration::try_from_secs_f64(seconds).map_err(|_| "seconds are out of range".into())
}

fn local_grpc_endpoint(raw: &str) -> Result<GrpcEndpoint, String> {
    let endpoint =
        GrpcEndpoint::parse(raw, "--grpc-endpoint").map_err(|error| error.to_string())?;
    if endpoint.as_str().ends_with(":0") {
        return Err("Speech NIM port must be nonzero".into());
    }
    let host = endpoint.authority_host().trim_matches(['[', ']']);
    if host != "localhost" && !host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback()) {
        return Err(
            "Speech NIM must run in the same pod or on this host; use a loopback address".into(),
        );
    }
    Ok(endpoint)
}

impl WorkerArgs {
    pub fn endpoint_parts(&self, default_component: &str) -> (String, String, String) {
        (
            self.namespace.clone(),
            self.component
                .as_deref()
                .unwrap_or(default_component)
                .into(),
            self.endpoint.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_addressing_honors_environment_and_cli() {
        // Isolate environment overrides from concurrently running tests.
        if std::env::var_os("SPEECH_NIM_CONFIG_TEST_CHILD").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "config::tests::worker_addressing_honors_environment_and_cli",
                ])
                .env("SPEECH_NIM_CONFIG_TEST_CHILD", "1")
                .env("DYN_NAMESPACE", "speech")
                .env("DYN_NAMESPACE_WORKER_SUFFIX", "blue")
                .env("DYN_COMPONENT", "env-asr")
                .env("DYN_ENDPOINT", "transcribe")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        let args = Args::parse_from(["sidecar", "asr"]);
        assert_eq!(
            args.mode.worker().endpoint_parts("speech-asr-realtime"),
            ("speech-blue".into(), "env-asr".into(), "transcribe".into())
        );
        let args = Args::parse_from([
            "sidecar",
            "asr",
            "--namespace",
            "override-blue",
            "--component",
            "cli-asr",
            "--endpoint",
            "generate",
        ]);
        assert_eq!(
            args.mode.worker().endpoint_parts("speech-asr-realtime"),
            ("override-blue".into(), "cli-asr".into(), "generate".into())
        );
    }

    #[test]
    fn connection_and_speech_defaults() {
        let Mode::Asr(config) = Args::parse_from(["sidecar", "asr"]).mode else {
            unreachable!()
        };
        assert_eq!(
            config.worker.connection.grpc_endpoint.as_str(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            config.worker.connection.grpc.config().startup_deadline,
            Duration::from_secs(1800)
        );
        assert_eq!(config.max_concurrent_turns.get(), 32);
        assert_eq!(config.timeout_s, Duration::from_secs(30));
        let Mode::Tts(config) = Args::parse_from([
            "sidecar",
            "tts",
            "--grpc-endpoint",
            "grpc://[::1]:50052",
            "--grpc-connections",
            "2",
            "--grpc-startup-deadline-secs",
            "12",
            "--namespace",
            "speech",
            "--component",
            "custom-tts",
            "--endpoint",
            "speak",
        ])
        .mode
        else {
            unreachable!()
        };
        assert_eq!(
            config.worker.connection.grpc_endpoint.as_str(),
            "http://[::1]:50052"
        );
        assert_eq!(config.worker.connection.grpc.config().connections.get(), 2);
        assert_eq!(
            config.worker.connection.grpc.config().startup_deadline,
            Duration::from_secs(12)
        );
        assert_eq!(
            config.worker.endpoint_parts("speech-tts"),
            ("speech".into(), "custom-tts".into(), "speak".into())
        );
        assert_eq!(config.sample_rate_hz, 24000);
    }

    #[test]
    fn rejects_invalid_limits_and_nonlocal_endpoints() {
        for (mode, option, value) in [
            ("asr", "--grpc-endpoint", "127.0.0.1:0"),
            ("tts", "--grpc-endpoint", "127.0.0.1:65536"),
            ("asr", "--grpc-endpoint", "https://localhost:50051"),
            ("tts", "--grpc-endpoint", "10.0.0.1:50051"),
            ("asr", "--grpc-connections", "0"),
            ("asr", "--grpc-startup-deadline-secs", "0"),
            ("asr", "--timeout-s", "0"),
            ("asr", "--timeout-s", "NaN"),
            ("asr", "--max-concurrent-turns", "0"),
            ("asr", "--commit-padding-ms", "-1"),
            ("tts", "--sample-rate-hz", "0"),
        ] {
            assert!(
                Args::try_parse_from(["sidecar", mode, option, value]).is_err(),
                "{option}={value}"
            );
        }
    }
}
