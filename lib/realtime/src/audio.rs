// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::{Engine, prelude::BASE64_STANDARD};
use rubato::{FftFixedInOut, Resampler};

use crate::Error;

pub(crate) const OPENAI_PCM_RATE: usize = 24_000;
const VLLM_PCM_RATE: usize = 16_000;
// Match the existing transcription handler's 100 ms transport block.
const RESAMPLE_BLOCK_MILLISECONDS: usize = 100;

pub(crate) fn decode_pcm16(encoded: &str, limit: usize) -> Result<Vec<u8>, Error> {
    // Bound allocation before base64 decoding, not just after it.
    if encoded.len().div_ceil(4).saturating_mul(3) > limit.saturating_add(2) {
        return Err(Error::Invalid(
            "audio chunk exceeds the configured byte limit".into(),
        ));
    }
    let decoded = BASE64_STANDARD
        .decode(encoded)
        .map_err(|_| Error::Invalid("audio must be valid base64".into()))?;
    if decoded.is_empty() || decoded.len() > limit || decoded.len() % 2 != 0 {
        return Err(Error::Invalid(
            "audio must contain nonempty, bounded PCM16 samples".into(),
        ));
    }
    Ok(decoded)
}

/// Stateful, band-limited 24 → 16 kHz conversion. Delayed filter samples are
/// flushed on commit and trimmed to the exact resampled duration.
pub(crate) struct AsrAudio {
    resampler: FftFixedInOut<f32>,
    pending: Vec<f32>,
    input: Vec<Vec<f32>>,
    output: Vec<Vec<f32>>,
    skip: usize,
    received: usize,
    emitted: usize,
}

impl AsrAudio {
    pub(crate) fn new() -> Result<Self, Error> {
        let resampler = FftFixedInOut::<f32>::new(
            OPENAI_PCM_RATE,
            VLLM_PCM_RATE,
            OPENAI_PCM_RATE * RESAMPLE_BLOCK_MILLISECONDS / 1000,
            1,
        )
        .map_err(|e| Error::Audio(e.to_string()))?;
        Ok(Self {
            input: resampler.input_buffer_allocate(true),
            output: resampler.output_buffer_allocate(true),
            skip: resampler.output_delay(),
            resampler,
            pending: Vec::new(),
            received: 0,
            emitted: 0,
        })
    }

    pub(crate) fn append(&mut self, pcm: &[u8]) -> Result<Vec<u8>, Error> {
        self.received += pcm.len() / 2;
        self.pending.extend(
            pcm.chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
        );
        let mut result = Vec::new();
        let frames = self.resampler.input_frames_next();
        let ready = self.pending.len() / frames * frames;
        for offset in (0..ready).step_by(frames) {
            self.input[0].copy_from_slice(&self.pending[offset..offset + frames]);
            self.process(&mut result, usize::MAX)?;
        }
        self.pending.drain(..ready);
        Ok(result)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u8>, Error> {
        let target = (self.received * VLLM_PCM_RATE).div_ceil(OPENAI_PCM_RATE);
        self.input[0].fill(0.0);
        self.input[0][..self.pending.len()].copy_from_slice(&self.pending);
        self.pending.clear();
        let mut result = Vec::new();
        while self.emitted < target {
            self.process(&mut result, target)?;
            self.input[0].fill(0.0);
        }
        Ok(result)
    }

    fn process(&mut self, result: &mut Vec<u8>, target: usize) -> Result<(), Error> {
        let (_, frames) = self
            .resampler
            .process_into_buffer(&self.input, &mut self.output, None)
            .map_err(|e| Error::Audio(e.to_string()))?;
        let skip = self.skip.min(frames);
        self.skip -= skip;
        let count = (frames - skip).min(target.saturating_sub(self.emitted));
        for sample in &self.output[0][skip..skip + count] {
            let pcm = (sample * 32768.0)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            result.extend_from_slice(&pcm.to_le_bytes());
        }
        self.emitted += count;
        Ok(())
    }
}
