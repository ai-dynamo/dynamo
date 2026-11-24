//! Inkling audio preprocessing.
//!
//! Implements the Inkling feature-extraction pipeline for the model-facing part:
//! audio bytes are decoded to mono f32, resampled to the configured sample
//! rate, non-zero quiet signals are raised to the configured RMS floor,
//! transformed to Slaney-normalized log-mel magnitudes, then quantized to dMel
//! bin ids. The returned encoder input stores those integer bin ids as f32,
//! matching the checkpoint feature-extractor contract and allowing the transport
//! to apply the same configurable floating-point wire dtype as other modalities.

use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
};

#[derive(Debug, Clone)]
pub struct InklingAudioEncoderParams {
    pub sample_rate: usize,
    pub window_size_multiplier: f64,
    pub n_fft: Option<usize>,
    pub n_mels: usize,
    pub num_dmel_bins: usize,
    pub dmel_min_value: f64,
    pub dmel_max_value: f64,
    pub audio_token_duration_s: f64,
    pub audio_rms_norm_floor: f64,
}

impl Default for InklingAudioEncoderParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            window_size_multiplier: 2.0,
            n_fft: None,
            n_mels: 80,
            num_dmel_bins: 16,
            dmel_min_value: -7.0,
            dmel_max_value: 2.0,
            audio_token_duration_s: 0.05,
            audio_rms_norm_floor: 0.01,
        }
    }
}

impl InklingAudioEncoderParams {
    pub fn from_model_config(config: &serde_json::Value) -> Self {
        let mut params = Self::default();
        let Some(audio_config) = config.get("audio_config") else {
            return params;
        };
        if let Some(v) = audio_config.get("n_mel_bins").and_then(|v| v.as_u64()) {
            params.n_mels = v as usize;
        }
        if let Some(v) = audio_config.get("mel_vocab_size").and_then(|v| v.as_u64()) {
            params.num_dmel_bins = v as usize;
        }
        if let Some(v) = audio_config.get("dmel_min_value").and_then(|v| v.as_f64()) {
            params.dmel_min_value = v;
        }
        if let Some(v) = audio_config.get("dmel_max_value").and_then(|v| v.as_f64()) {
            params.dmel_max_value = v;
        }
        if let Some(v) = audio_config
            .get("audio_rms_norm_floor")
            .and_then(|v| v.as_f64())
        {
            params.audio_rms_norm_floor = v;
        }
        params
    }

    fn hop_length(&self) -> Result<usize, TransformError> {
        exact_sample_count(
            self.audio_token_duration_s * self.sample_rate as f64,
            "audio_token_duration_s * sample_rate",
        )
    }

    fn window_size(&self) -> Result<usize, TransformError> {
        exact_sample_count(
            self.audio_token_duration_s * self.window_size_multiplier * self.sample_rate as f64,
            "audio_token_duration_s * window_size_multiplier * sample_rate",
        )
    }
}

#[derive(Debug, Clone)]
pub struct InklingAudioProcessor {
    params: InklingAudioEncoderParams,
}

impl Default for InklingAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingAudioProcessor {
    pub fn new() -> Self {
        Self {
            params: InklingAudioEncoderParams::default(),
        }
    }

    pub fn from_model_config(config: &serde_json::Value) -> Self {
        Self {
            params: InklingAudioEncoderParams::from_model_config(config),
        }
    }

    pub fn preprocess_decoded_clips(
        &self,
        clips: Vec<DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut all_bins = Vec::new();
        let mut token_counts = Vec::with_capacity(clips.len());
        let mut item_sizes = Vec::with_capacity(clips.len());
        let mut tokens_per_item = Vec::with_capacity(clips.len());

        for clip in clips {
            let bins = self.preprocess_decoded(clip)?;
            let shape = bins.shape();
            let num_tokens = shape[0];
            let n_mels = shape[1];
            token_counts.push(num_tokens);
            tokens_per_item.push(num_tokens as i64);
            item_sizes.push((n_mels as u32, num_tokens as u32));
            all_bins.extend(bins.into_raw_vec_and_offset().0);
        }

        let total_tokens: usize = token_counts.iter().sum();
        let encoder_input = Array2::from_shape_vec((total_tokens, self.params.n_mels), all_bins)
            .map_err(|e| {
                TransformError::ShapeError(format!(
                    "failed to create Inkling audio encoder input [{total_tokens}, {}]: {e}",
                    self.params.n_mels
                ))
            })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes).with_extra(
                "tokens_per_item",
                ModelSpecificValue::int_1d(tokens_per_item),
            ),
        )
    }

    fn preprocess_decoded(&self, decoded: DecodedAudio) -> Result<Array2<f32>, TransformError> {
        if decoded.sample_rate == 0 {
            return Err(TransformError::ShapeError(
                "decoded audio sample rate must be positive".to_string(),
            ));
        }
        if decoded.samples.is_empty() {
            return Err(TransformError::ShapeError(
                "decoded audio contains no samples".to_string(),
            ));
        }
        let mut samples = if decoded.sample_rate == self.params.sample_rate {
            decoded.samples
        } else {
            bandlimited_resample(
                &decoded.samples,
                decoded.sample_rate,
                self.params.sample_rate,
            )?
        };
        normalize_audio_rms(&mut samples, self.params.audio_rms_norm_floor)?;
        dmel_bins(&samples, &self.params)
    }
}

impl AudioPreProcessor for InklingAudioProcessor {
    fn preprocess(
        &self,
        clips: &[std::sync::Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_decoded_clips(clips.iter().map(|clip| clip.decoded().clone()).collect())
    }
}

fn exact_sample_count(value: f64, name: &str) -> Result<usize, TransformError> {
    let rounded = value.round();
    if (value - rounded).abs() > 1e-6 {
        return Err(TransformError::ShapeError(format!(
            "{name} must resolve to an integer sample count, got {value}"
        )));
    }
    if rounded <= 0.0 {
        return Err(TransformError::ShapeError(format!(
            "{name} must be positive, got {rounded}"
        )));
    }
    Ok(rounded as usize)
}

fn normalize_audio_rms(samples: &mut [f32], floor: f64) -> Result<(), TransformError> {
    if !floor.is_finite() || floor < 0.0 {
        return Err(TransformError::ShapeError(format!(
            "audio_rms_norm_floor must be finite and non-negative, got {floor}"
        )));
    }
    if floor == 0.0 {
        return Ok(());
    }

    let mean_square = samples
        .iter()
        .map(|&sample| {
            let sample = f64::from(sample);
            sample * sample
        })
        .sum::<f64>()
        / samples.len() as f64;
    let rms = mean_square.sqrt();
    if rms > 0.0 && rms < floor {
        let scale = floor / rms;
        for sample in samples {
            *sample = (f64::from(*sample) * scale) as f32;
        }
    }
    Ok(())
}

fn dmel_bins(
    samples: &[f32],
    params: &InklingAudioEncoderParams,
) -> Result<Array2<f32>, TransformError> {
    if params.n_mels == 0 || params.num_dmel_bins == 0 {
        return Err(TransformError::ShapeError(
            "n_mels and num_dmel_bins must be positive".to_string(),
        ));
    }

    let hop_length = params.hop_length()?;
    let window_size = params.window_size()?;
    let n_fft = params.n_fft.unwrap_or(window_size);
    if n_fft < window_size {
        return Err(TransformError::ShapeError(format!(
            "n_fft ({n_fft}) must be greater than or equal to window_size ({window_size})"
        )));
    }
    if samples.is_empty() {
        return Err(TransformError::ShapeError(
            "audio preprocessing requires at least one sample".to_string(),
        ));
    }

    let right_pad = samples.len().div_ceil(hop_length) * hop_length - samples.len();
    let left_pad = n_fft.saturating_sub(hop_length);
    let padded_len = left_pad + samples.len() + right_pad;
    let mut padded = vec![0.0_f32; padded_len];
    padded[left_pad..left_pad + samples.len()].copy_from_slice(samples);

    let frame_count = (padded_len - n_fft) / hop_length + 1;
    let fft_bins = n_fft / 2 + 1;
    let window = hann_window(window_size);
    let mel_basis = mel_basis(params.sample_rate, n_fft, params.n_mels);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buffer = vec![Complex32::new(0.0, 0.0); n_fft];
    let mut magnitudes = vec![0.0_f32; fft_bins * frame_count];

    for frame in 0..frame_count {
        buffer.fill(Complex32::new(0.0, 0.0));
        let start = frame * hop_length;
        for i in 0..window_size {
            buffer[i].re = padded[start + i] * window[i];
        }
        fft.process(&mut buffer);
        for bin in 0..fft_bins {
            let value = buffer[bin];
            magnitudes[bin * frame_count + frame] =
                (value.re.mul_add(value.re, value.im * value.im))
                    .max(1e-10)
                    .sqrt();
        }
    }

    let mut output = Vec::with_capacity(frame_count * params.n_mels);
    for frame in 0..frame_count {
        for mel in 0..params.n_mels {
            let basis_row = &mel_basis[mel * fft_bins..(mel + 1) * fft_bins];
            let mut value = 0.0_f32;
            for bin in 0..fft_bins {
                value += basis_row[bin] * magnitudes[bin * frame_count + frame];
            }
            let log_mel = f64::from(value.max(1e-10).log10())
                .clamp(params.dmel_min_value, params.dmel_max_value);
            output.push(quantize_dmel(log_mel, params) as f32);
        }
    }

    Array2::from_shape_vec((frame_count, params.n_mels), output).map_err(|e| {
        TransformError::ShapeError(format!(
            "failed to create Inkling dMel bins [{frame_count}, {}]: {e}",
            params.n_mels
        ))
    })
}

fn quantize_dmel(value: f64, params: &InklingAudioEncoderParams) -> usize {
    if params.num_dmel_bins <= 1 {
        return 0;
    }
    let span = params.dmel_max_value - params.dmel_min_value;
    if span <= 0.0 {
        return 0;
    }
    let scaled = (value - params.dmel_min_value) / span * (params.num_dmel_bins - 1) as f64;
    // Nearest-bin selection keeps the lower bin on an exact tie.
    (scaled - 0.5)
        .ceil()
        .clamp(0.0, (params.num_dmel_bins - 1) as f64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_or_invalid_rate_audio() {
        let processor = InklingAudioProcessor::new();
        assert!(processor
            .preprocess_decoded(DecodedAudio {
                samples: Vec::new(),
                sample_rate: 16_000,
            })
            .is_err());
        assert!(processor
            .preprocess_decoded(DecodedAudio {
                samples: vec![0.0],
                sample_rate: 0,
            })
            .is_err());
    }

    #[test]
    fn model_config_overrides_audio_shape_params() {
        let config = serde_json::json!({
            "audio_config": {
                "n_mel_bins": 8,
                "mel_vocab_size": 4,
                "dmel_min_value": -5.0,
                "dmel_max_value": 1.0,
                "audio_rms_norm_floor": 0.02
            }
        });
        let processor = InklingAudioProcessor::from_model_config(&config);
        assert_eq!(processor.params.n_mels, 8);
        assert_eq!(processor.params.num_dmel_bins, 4);
        assert_eq!(processor.params.dmel_min_value, -5.0);
        assert_eq!(processor.params.dmel_max_value, 1.0);
        assert_eq!(processor.params.audio_rms_norm_floor, 0.02);
    }

    fn seeded_signal(scale: f32) -> Vec<f32> {
        (0..1600)
            .map(|i| {
                let raw = (i * 73 + 17 * 977) % 65_536 - 32_768;
                raw as f32 / 32_768.0 * scale
            })
            .collect()
    }

    #[test]
    fn quiet_nonzero_audio_is_normalized_to_rms_floor() {
        let processor = InklingAudioProcessor::new();
        let very_quiet = processor
            .preprocess_decoded(DecodedAudio {
                samples: seeded_signal(0.001),
                sample_rate: 16_000,
            })
            .unwrap();
        let less_quiet = processor
            .preprocess_decoded(DecodedAudio {
                samples: seeded_signal(0.004),
                sample_rate: 16_000,
            })
            .unwrap();

        assert_eq!(very_quiet, less_quiet);
    }

    #[test]
    fn rms_floor_leaves_silence_and_loud_audio_unchanged() {
        let processor = InklingAudioProcessor::new();
        let mut without_floor = processor.clone();
        without_floor.params.audio_rms_norm_floor = 0.0;

        for samples in [vec![0.0; 1600], seeded_signal(1.0)] {
            let with_floor = processor
                .preprocess_decoded(DecodedAudio {
                    samples: samples.clone(),
                    sample_rate: 16_000,
                })
                .unwrap();
            let without_floor = without_floor
                .preprocess_decoded(DecodedAudio {
                    samples,
                    sample_rate: 16_000,
                })
                .unwrap();
            assert_eq!(with_floor, without_floor);
        }
    }

    #[test]
    fn rejects_invalid_rms_floor() {
        let mut processor = InklingAudioProcessor::new();
        processor.params.audio_rms_norm_floor = -0.01;
        let error = processor
            .preprocess_decoded(DecodedAudio {
                samples: vec![0.1; 800],
                sample_rate: 16_000,
            })
            .unwrap_err();
        assert!(error.to_string().contains("audio_rms_norm_floor"));
    }
}
