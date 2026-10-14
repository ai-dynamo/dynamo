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

use std::{num::NonZeroUsize, path::PathBuf, time::Duration};

use clap::{Args as ClapArgs, Parser, Subcommand};
use dynamo_backend_common::RuntimeConfig;

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

#[derive(Clone, ClapArgs)]
pub struct ConnectionConfig {
    #[arg(long, default_value = "localhost:50051")]
    pub nim_server: String,
    #[arg(long, default_value = "600", value_parser = duration)]
    pub nim_startup_timeout_s: Duration,
    #[arg(long)]
    pub nim_use_ssl: bool,
    #[arg(long, hide_env_values = true)]
    pub nim_api_key: Option<String>,
    #[arg(long)]
    pub nim_function_id: Option<String>,
    #[arg(long)]
    pub nim_ssl_root_cert: Option<PathBuf>,
}

#[derive(Clone, ClapArgs)]
pub struct WorkerArgs {
    #[command(flatten)]
    pub connection: ConnectionConfig,
    /// Dynamo namespace.component.endpoint; defaults to the operator namespace.
    #[arg(long)]
    pub endpoint: Option<String>,
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

impl WorkerArgs {
    pub fn endpoint_parts(&self, component: &str) -> anyhow::Result<(String, String, String)> {
        resolve_endpoint(
            self.endpoint.as_deref(),
            component,
            std::env::var("DYN_NAMESPACE").ok().as_deref(),
            std::env::var("DYN_NAMESPACE_WORKER_SUFFIX").ok().as_deref(),
        )
    }
}

fn resolve_endpoint(
    endpoint: Option<&str>,
    component: &str,
    namespace: Option<&str>,
    suffix: Option<&str>,
) -> anyhow::Result<(String, String, String)> {
    if let Some(endpoint) = endpoint {
        let parts: Vec<_> = endpoint.split('.').collect();
        anyhow::ensure!(
            parts.len() == 3 && parts.iter().all(|part| !part.is_empty()),
            "endpoint must be namespace.component.endpoint"
        );
        return Ok((parts[0].into(), parts[1].into(), parts[2].into()));
    }
    let mut namespace = namespace.unwrap_or("dynamo").to_owned();
    if let Some(suffix) = suffix.filter(|s| !s.is_empty()) {
        namespace.push('-');
        namespace.push_str(suffix);
    }
    Ok((namespace, component.into(), "generate".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_suffix_and_explicit_endpoint() {
        for (namespace, suffix, expected) in [
            (None, None, "dynamo"),
            (Some("speech"), None, "speech"),
            (Some("speech"), Some("blue"), "speech-blue"),
        ] {
            assert_eq!(
                resolve_endpoint(None, "asr", namespace, suffix).unwrap(),
                (expected.into(), "asr".into(), "generate".into())
            );
            assert_eq!(
                resolve_endpoint(Some("custom.tts.generate"), "asr", namespace, suffix).unwrap(),
                ("custom".into(), "tts".into(), "generate".into())
            );
        }
        for invalid in ["", "asr", "a.b", "a..c", "a.b.c.d"] {
            assert!(resolve_endpoint(Some(invalid), "asr", None, None).is_err());
        }
    }

    #[test]
    fn connection_and_speech_defaults() {
        let Mode::Asr(config) = Args::parse_from(["sidecar", "asr"]).mode else {
            unreachable!()
        };
        assert_eq!(config.worker.connection.nim_server, "localhost:50051");
        assert_eq!(
            config.worker.connection.nim_startup_timeout_s,
            Duration::from_secs(600)
        );
        assert_eq!(config.max_concurrent_turns.get(), 32);
        assert_eq!(config.timeout_s, Duration::from_secs(30));
        let Mode::Tts(config) = Args::parse_from([
            "sidecar",
            "tts",
            "--nim-use-ssl",
            "--nim-api-key",
            "test-key",
            "--nim-function-id",
            "test-function",
            "--nim-ssl-root-cert",
            "ca.pem",
            "--nim-startup-timeout-s",
            "12.5",
        ])
        .mode
        else {
            unreachable!()
        };
        assert!(config.worker.connection.nim_use_ssl);
        assert_eq!(
            config.worker.connection.nim_api_key.as_deref(),
            Some("test-key")
        );
        assert_eq!(
            config.worker.connection.nim_function_id.as_deref(),
            Some("test-function")
        );
        assert_eq!(
            config.worker.connection.nim_ssl_root_cert,
            Some(PathBuf::from("ca.pem"))
        );
        assert_eq!(
            config.worker.connection.nim_startup_timeout_s,
            Duration::from_millis(12500)
        );
        assert_eq!(config.sample_rate_hz, 24000);
    }

    #[test]
    fn rejects_invalid_limits_before_starting_runtime() {
        for (mode, option, value) in [
            ("asr", "--timeout-s", "0"),
            ("asr", "--timeout-s", "NaN"),
            ("asr", "--nim-startup-timeout-s", "inf"),
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
