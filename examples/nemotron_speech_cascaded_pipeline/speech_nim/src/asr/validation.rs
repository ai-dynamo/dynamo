// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;

use crate::config::AsrConfig;

pub(super) const SAMPLE_RATE: usize = 24_000;
const MAX_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const MAX_AUDIO_BYTES: usize = SAMPLE_RATE * 2 * 60;

pub(super) fn audio(value: &Value, received: &mut usize) -> Result<Vec<u8>, &'static str> {
    let encoded = value.as_str().ok_or("audio must be a base64 string")?;
    // Bound allocation before decoding; the decoded-size check covers padding.
    if encoded.len() > MAX_CHUNK_BYTES.div_ceil(3) * 4 {
        return Err("audio chunk exceeds 4 MiB");
    }
    let audio = STANDARD
        .decode(encoded)
        .map_err(|_| "audio must be valid base64")?;
    if audio.is_empty() {
        return Err("audio chunk is empty");
    }
    if audio.len() > MAX_CHUNK_BYTES {
        return Err("audio chunk exceeds 4 MiB");
    }
    if audio.len() % 2 != 0 {
        return Err("PCM16 audio must be 2-byte aligned");
    }
    *received += audio.len();
    if *received > MAX_AUDIO_BYTES {
        return Err("input audio exceeds 60 seconds");
    }
    Ok(audio)
}

pub(super) fn session(session: &Value, config: &AsrConfig) -> Result<(), String> {
    if !session.is_object() || session["type"] != "transcription" {
        return Err("session.type must be 'transcription'".into());
    }
    let input = &session["audio"]["input"];
    if !input.is_object() {
        return Err("session.audio.input must be an object".into());
    }
    let transcription = &input["transcription"];
    if !transcription.is_object() || transcription["model"] != config.model_name {
        return Err(format!(
            "session transcription model must be '{}'",
            config.model_name
        ));
    }
    let format = &input["format"];
    if !format.is_object() {
        return Err("session.audio.input.format must be an object".into());
    }
    if format["type"] != "audio/pcm" {
        return Err("only audio/pcm input is supported".into());
    }
    if format["rate"].as_f64() != Some(SAMPLE_RATE as f64) {
        return Err(format!("audio/pcm input rate must be {SAMPLE_RATE} Hz"));
    }
    if let Some(language) = transcription
        .get("language")
        .filter(|v| !v.is_null() && **v != "")
    {
        let supported = config.language_code.split('-').next().unwrap_or_default();
        if !language.as_str().is_some_and(|s| {
            s.split('-')
                .next()
                .unwrap_or_default()
                .eq_ignore_ascii_case(supported)
        }) {
            return Err(format!(
                "only {} realtime transcription is supported",
                config.language_code
            ));
        }
    }
    if !transcription["prompt"].is_null() && transcription["prompt"] != "" {
        return Err("transcription prompts are not supported".into());
    }
    if !input["noise_reduction"].is_null() {
        return Err("input audio noise reduction is not supported".into());
    }
    if !input["turn_detection"].is_null() {
        return Err("server turn detection is not supported; use local VAD".into());
    }
    Ok(())
}
