// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small CPU polyphase resampler for the prototype's two PCM rate conversions.
//! Matches scipy.signal.resample_poly's default 10 * max(up, down) Kaiser-5
//! filter construction and zero-padding. ASR uses the baseline's 100 ms blocks.
//! The upstream WS requires PCM16, so an extra float-to-PCM quantization remains.

use anyhow::{Result, ensure};

pub struct AudioConverter {
    input_rate: u32,
    output_rate: u32,
    pending: Vec<i16>,
    received: usize,
}

impl AudioConverter {
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            pending: Vec::new(),
            received: 0,
        }
    }

    pub fn push(&mut self, bytes: &[u8], flush: bool) -> Result<Vec<u8>> {
        ensure!(bytes.len() % 2 == 0, "PCM16 input must be 2-byte aligned");
        ensure!(bytes.len() <= 4 * 1024 * 1024, "audio chunk exceeds 4 MiB");
        self.received += bytes.len() / 2;
        ensure!(
            self.received <= self.input_rate as usize * 60,
            "audio exceeds 60 seconds"
        );
        self.pending.extend(
            bytes
                .chunks_exact(2)
                .map(|s| i16::from_le_bytes([s[0], s[1]])),
        );
        let block = self.input_rate as usize / 10;
        let size = if flush {
            self.pending.len()
        } else {
            self.pending.len() / block * block
        };
        let output = resample(&self.pending[..size], self.input_rate, self.output_rate);
        self.pending.drain(..size);
        Ok(output.into_iter().flat_map(i16::to_le_bytes).collect())
    }
}

fn i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..40 {
        term *= (x * x / 4.0) / f64::from(k * k);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

fn resample(input: &[i16], from: u32, to: u32) -> Vec<i16> {
    if from == to {
        return input.to_vec();
    }
    let (mut a, mut b) = (from, to);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let up = (to / a) as usize;
    let down = (from / a) as usize;
    let max_rate = up.max(down);
    let half = 10 * max_rate;
    let denom = i0(5.0);
    let mut h: Vec<f64> = (0..=2 * half)
        .map(|n| {
            let x = (n as f64 - half as f64) / max_rate as f64;
            let sinc = if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let position = (n as f64 - half as f64) / half as f64;
            sinc / max_rate as f64 * i0(5.0 * (1.0 - position * position).max(0.0).sqrt()) / denom
        })
        .collect();
    let scale = up as f64 / h.iter().sum::<f64>();
    h.iter_mut().for_each(|x| *x *= scale);
    (0..(input.len() * up).div_ceil(down))
        .map(|i| {
            let center = i * down + half;
            let first = center.saturating_sub(2 * half).div_ceil(up);
            let last = (center / up + 1).min(input.len());
            let value: f64 = (first..last)
                .map(|j| f64::from(input[j]) * h[center - j * up])
                .sum();
            value.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_chunk_boundaries_preserve_duration() {
        let mut converter = AudioConverter::new(24000, 16000);
        assert!(converter.push(&vec![0; 2400], false).unwrap().is_empty());
        assert_eq!(converter.push(&vec![0; 2400], false).unwrap().len(), 3200);
        assert_eq!(converter.push(&vec![0; 480], true).unwrap().len(), 320);
        let mut up = AudioConverter::new(16000, 24000);
        assert_eq!(up.push(&vec![0; 3200], false).unwrap().len(), 4800);
        assert!(up.push(&[0], false).is_err());
    }

    #[test]
    fn polyphase_preserves_low_frequency_signal() {
        let input: Vec<i16> = (0..2400)
            .map(|i| {
                (10000.0 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 24000.0).sin()) as i16
            })
            .collect();
        let output = resample(&input, 24000, 16000);
        let mse = output[20..1580]
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let expected = 10000.0
                    * (2.0 * std::f64::consts::PI * 440.0 * (i + 20) as f64 / 16000.0).sin();
                (f64::from(*x) - expected).powi(2)
            })
            .sum::<f64>()
            / 1560.0;
        assert!(mse.sqrt() < 20.0, "RMS error {mse}");
    }
}
