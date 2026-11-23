//! Qwen3 audio preprocessing using a Whisper-compatible log-mel frontend.

use std::sync::Arc;

use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde_json::Value;

use crate::{
    audio::{
        transforms::{bandlimited_resample, hann_window, mel_basis},
        AudioPreProcessor, DecodedAudio,
    },
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    error::TransformError,
    types::AudioClip,
    vision::PreProcessorConfig,
};

/// Parameters used by the Qwen3 audio frontend.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3AudioParams {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_window: usize,
    pub padding_value: f32,
    pub max_samples: Option<usize>,
}

impl Default for Qwen3AudioParams {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 128,
            n_fft: 400,
            hop_length: 160,
            n_window: 50,
            padding_value: 0.0,
            max_samples: None,
        }
    }
}

impl Qwen3AudioParams {
    pub fn from_configs(model_config: &Value, preprocessor_config: &PreProcessorConfig) -> Self {
        let mut params = Self::default();

        if let Some(value) = preprocessor_config.get_extra::<usize>("sampling_rate") {
            params.sample_rate = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("feature_size") {
            params.n_mels = value;
        } else if let Some(value) = find_model_usize(
            model_config,
            &[
                &["thinker_config", "audio_config", "num_mel_bins"],
                &["audio_config", "num_mel_bins"],
            ],
        ) {
            params.n_mels = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("n_fft") {
            params.n_fft = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<usize>("hop_length") {
            params.hop_length = value;
        }
        if let Some(value) = preprocessor_config
            .get_extra::<usize>("n_window")
            .or_else(|| {
                find_model_usize(
                    model_config,
                    &[
                        &["thinker_config", "audio_config", "n_window"],
                        &["audio_config", "n_window"],
                    ],
                )
            })
        {
            params.n_window = value;
        }
        if let Some(value) = preprocessor_config.get_extra::<f32>("padding_value") {
            params.padding_value = value;
        }
        // Qwen processors set `padding=true, truncation=false`: n_samples is
        // the default padding target, not an input limit. Honor it only when a
        // checkpoint or deployment explicitly enables truncation. A custom
        // max_samples remains available as an operational hard limit.
        params.max_samples = preprocessor_config
            .get_extra::<usize>("max_samples")
            .or_else(|| {
                preprocessor_config
                    .get_extra::<bool>("truncation")
                    .filter(|enabled| *enabled)
                    .and_then(|_| {
                        preprocessor_config
                            .get_extra::<usize>("n_samples")
                            .or_else(|| {
                                preprocessor_config
                                    .get_extra::<usize>("chunk_length")
                                    .and_then(|seconds| seconds.checked_mul(params.sample_rate))
                            })
                    })
            });

        params
    }
}

fn find_model_usize(config: &Value, paths: &[&[&str]]) -> Option<usize> {
    paths.iter().find_map(|path| {
        let mut value = config;
        for key in *path {
            value = value.get(*key)?;
        }
        value.as_u64().and_then(|value| usize::try_from(value).ok())
    })
}

#[derive(Debug, Clone)]
pub struct Qwen3AudioProcessor {
    params: Qwen3AudioParams,
}

impl Default for Qwen3AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3AudioProcessor {
    pub fn new() -> Self {
        Self {
            params: Qwen3AudioParams::default(),
        }
    }

    pub fn with_params(params: Qwen3AudioParams) -> Self {
        Self { params }
    }

    pub fn from_configs(model_config: &Value, preprocessor_config: &PreProcessorConfig) -> Self {
        Self::with_params(Qwen3AudioParams::from_configs(
            model_config,
            preprocessor_config,
        ))
    }

    pub fn params(&self) -> &Qwen3AudioParams {
        &self.params
    }

    pub fn preprocess_decoded_clips(
        &self,
        clips: Vec<DecodedAudio>,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.validate_params()?;
        if clips.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        let mut waveforms = Vec::with_capacity(clips.len());
        for (clip_index, clip) in clips.into_iter().enumerate() {
            if clip.sample_rate == 0 {
                return Err(TransformError::ShapeError(
                    "decoded audio sample rate must be positive".to_string(),
                ));
            }
            if clip.samples.is_empty() {
                return Err(TransformError::ShapeError(
                    "decoded audio contains no samples".to_string(),
                ));
            }
            if clip.samples.iter().any(|sample| !sample.is_finite()) {
                return Err(TransformError::ShapeError(
                    "decoded audio contains a non-finite sample".to_string(),
                ));
            }
            let samples = if clip.sample_rate == self.params.sample_rate {
                clip.samples
            } else {
                bandlimited_resample(&clip.samples, clip.sample_rate, self.params.sample_rate)?
            };
            let mut samples = samples;
            if let Some(max_samples) = self.params.max_samples {
                samples.truncate(max_samples);
            }
            if samples.is_empty() {
                return Err(TransformError::ShapeError(
                    "decoded audio contains no samples after truncation".to_string(),
                ));
            }
            if samples.len() < self.params.hop_length {
                return Err(TransformError::ShapeError(format!(
                    "Qwen3 audio clip at batch index {clip_index} has {} samples after resampling and truncation; at least {} are required",
                    samples.len(),
                    self.params.hop_length
                )));
            }
            waveforms.push(samples);
        }

        let max_samples = waveforms.iter().map(Vec::len).max().unwrap_or(0);
        // Whisper's centered STFT yields floor(samples / hop) + 1 frames and
        // then drops the final frame. This matches padding=True with the batch
        // padded to its longest waveform.
        let max_frames = max_samples / self.params.hop_length;
        if max_frames == 0 {
            return Err(TransformError::ShapeError(format!(
                "Qwen3 audio requires at least {} samples after resampling",
                self.params.hop_length
            )));
        }

        let batch_size = waveforms.len();
        let feature_values = batch_size
            .checked_mul(self.params.n_mels)
            .and_then(|value| value.checked_mul(max_frames))
            .ok_or_else(|| {
                TransformError::ShapeError("Qwen3 audio feature size overflow".to_string())
            })?;
        let mut all_features = Vec::with_capacity(feature_values);
        let mut attention_mask = Vec::with_capacity(batch_size * max_frames);
        let mut feature_lengths = Vec::with_capacity(batch_size);
        let mut token_counts = Vec::with_capacity(batch_size);
        let mut item_sizes = Vec::with_capacity(batch_size);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.params.n_fft);

        for waveform in waveforms {
            let original_samples = waveform.len();
            let feature_length = (original_samples / self.params.hop_length).min(max_frames);
            let mut padded = waveform;
            padded.resize(max_samples, self.params.padding_value);
            let features = whisper_log_mel(&padded, max_frames, &self.params, fft.as_ref())?;
            all_features.extend(features.into_raw_vec_and_offset().0);

            attention_mask.extend((0..max_frames).map(|frame| i64::from(frame < feature_length)));
            feature_lengths.push(feature_length as i64);
            token_counts.push(qwen3_audio_output_length(
                feature_length,
                self.params.n_window,
            ));
            item_sizes.push((self.params.n_mels as u32, feature_length as u32));
        }

        let encoder_input =
            Array3::from_shape_vec((batch_size, self.params.n_mels, max_frames), all_features)
                .map_err(|error| {
                    TransformError::ShapeError(format!(
                "failed to create Qwen3 audio input [{batch_size}, {}, {max_frames}]: {error}",
                self.params.n_mels
            ))
                })?;

        Ok(
            PreprocessedEncoderInputs::new(encoder_input, token_counts, item_sizes)
                .with_extra(
                    "feature_attention_mask",
                    ModelSpecificValue::int_2d(attention_mask, batch_size, max_frames),
                )
                .with_extra(
                    "audio_feature_lengths",
                    ModelSpecificValue::int_1d(feature_lengths),
                ),
        )
    }

    pub fn preprocess_decoded(&self, decoded: DecodedAudio) -> Result<Array2<f32>, TransformError> {
        let output = self.preprocess_decoded_clips(vec![decoded])?;
        output
            .encoder_input
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|error| TransformError::ShapeError(error.to_string()))?
            .index_axis_move(ndarray::Axis(0), 0)
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|error| TransformError::ShapeError(error.to_string()))
    }

    fn validate_params(&self) -> Result<(), TransformError> {
        if self.params.sample_rate == 0
            || self.params.n_mels == 0
            || self.params.n_fft == 0
            || self.params.hop_length == 0
            || self.params.n_window == 0
        {
            return Err(TransformError::ShapeError(
                "Qwen3 audio sample rate, mel bins, FFT size, hop length, and window size must be positive"
                    .to_string(),
            ));
        }
        if self.params.n_fft < self.params.hop_length {
            return Err(TransformError::ShapeError(format!(
                "Qwen3 audio n_fft ({}) must be at least hop_length ({})",
                self.params.n_fft, self.params.hop_length
            )));
        }
        if !self.params.padding_value.is_finite() {
            return Err(TransformError::ShapeError(
                "Qwen3 audio padding_value must be finite".to_string(),
            ));
        }
        if self.params.max_samples == Some(0) {
            return Err(TransformError::ShapeError(
                "Qwen3 audio max_samples must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl AudioPreProcessor for Qwen3AudioProcessor {
    fn preprocess(
        &self,
        clips: &[Arc<AudioClip>],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.preprocess_decoded_clips(clips.iter().map(|clip| clip.decoded().clone()).collect())
    }
}

fn qwen_audio_cnn_output_length(mut input_length: usize) -> usize {
    for _ in 0..3 {
        input_length = input_length.div_ceil(2);
    }
    input_length
}

/// Output tokens produced by Qwen's chunked audio encoder for a log-mel length.
fn qwen3_audio_output_length(input_length: usize, n_window: usize) -> usize {
    debug_assert!(n_window > 0);
    let chunk_size = 2 * n_window;
    let full_windows = input_length / chunk_size;
    let remainder = input_length % chunk_size;
    full_windows * qwen_audio_cnn_output_length(chunk_size)
        + qwen_audio_cnn_output_length(remainder)
}

fn whisper_log_mel(
    samples: &[f32],
    frame_count: usize,
    params: &Qwen3AudioParams,
    fft: &dyn Fft<f32>,
) -> Result<Array2<f32>, TransformError> {
    let center_pad = params.n_fft / 2;
    let padded = reflect_pad(samples, center_pad);
    let fft_bins = params.n_fft / 2 + 1;
    let window = hann_window(params.n_fft);
    let mel_filters = mel_basis(params.sample_rate, params.n_fft, params.n_mels);
    let mut buffer = vec![Complex32::new(0.0, 0.0); params.n_fft];
    let mut output = vec![0.0_f32; params.n_mels * frame_count];

    for frame in 0..frame_count {
        let start = frame * params.hop_length;
        let end = start + params.n_fft;
        let frame_samples = padded.get(start..end).ok_or_else(|| {
            TransformError::ShapeError(format!(
                "Qwen3 audio STFT frame {frame} lies outside padded waveform"
            ))
        })?;
        for index in 0..params.n_fft {
            buffer[index] = Complex32::new(frame_samples[index] * window[index], 0.0);
        }
        fft.process(&mut buffer);

        for mel in 0..params.n_mels {
            let filter = &mel_filters[mel * fft_bins..(mel + 1) * fft_bins];
            let mut value = 0.0_f32;
            for bin in 0..fft_bins {
                let fft_value = buffer[bin];
                let power = fft_value
                    .re
                    .mul_add(fft_value.re, fft_value.im * fft_value.im);
                value = filter[bin].mul_add(power, value);
            }
            output[mel * frame_count + frame] = value.max(1e-10).log10();
        }
    }

    let peak = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = peak - 8.0;
    for value in &mut output {
        *value = (value.max(floor) + 4.0) / 4.0;
    }

    Array2::from_shape_vec((params.n_mels, frame_count), output).map_err(|error| {
        TransformError::ShapeError(format!(
            "failed to create Qwen log-mel input [{}, {frame_count}]: {error}",
            params.n_mels
        ))
    })
}

fn reflect_pad(samples: &[f32], padding: usize) -> Vec<f32> {
    if padding == 0 {
        return samples.to_vec();
    }
    if samples.len() == 1 {
        return vec![samples[0]; samples.len() + 2 * padding];
    }

    let mut padded = Vec::with_capacity(samples.len() + 2 * padding);
    for position in -(padding as isize)..(samples.len() + padding) as isize {
        padded.push(samples[reflect_index(position, samples.len())]);
    }
    padded
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    let last = len as isize - 1;
    while index < 0 || index > last {
        if index < 0 {
            index = -index;
        }
        if index > last {
            index = 2 * last - index;
        }
    }
    index as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(samples: usize) -> DecodedAudio {
        DecodedAudio {
            samples: vec![0.0; samples],
            sample_rate: 16_000,
        }
    }

    #[test]
    fn silence_matches_whisper_normalization() {
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(decoded(1600))
            .unwrap();
        assert_eq!(features.shape(), &[128, 10]);
        assert!(features.iter().all(|value| (*value + 1.5).abs() < 1e-6));
    }

    #[test]
    fn log_mel_matches_numpy_whisper_reference() {
        let samples = (0..1000)
            .map(|index| ((index % 23) as f32 - 11.0) / 32.0)
            .collect();
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(DecodedAudio {
                samples,
                sample_rate: 16_000,
            })
            .unwrap();
        assert_eq!(features.shape(), &[128, 6]);

        for ((mel, frame), expected) in [
            ((0, 0), 0.71467185),
            ((10, 0), 0.735_666_9),
            ((20, 1), 0.36943567),
            ((32, 3), 0.79836977),
            ((64, 5), -0.582_248_9),
            ((100, 5), 0.60166824),
            ((127, 5), 0.52557164),
        ] {
            assert!(
                (features[[mel, frame]] - expected).abs() < 2e-4,
                "log-mel mismatch at ({mel}, {frame}): {} vs {expected}",
                features[[mel, frame]]
            );
        }
        let sum: f64 = features.iter().map(|&value| f64::from(value)).sum();
        assert!((sum - 125.08224487).abs() < 0.02, "feature sum {sum}");
    }

    #[test]
    fn log_mel_boundary_matches_hf_whisper_reference() {
        let samples = (0..1600)
            .map(|index| ((index % 23) as f32 - 11.0) / 32.0)
            .collect();
        let features = Qwen3AudioProcessor::new()
            .preprocess_decoded(DecodedAudio {
                samples,
                sample_rate: 16_000,
            })
            .unwrap();

        assert_eq!(features.shape(), &[128, 10]);
        for (mel, expected) in [(64, 0.017_234_564), (100, 0.601_944_7), (127, 0.525_99)] {
            assert!(
                (features[[mel, 9]] - expected).abs() < 2e-4,
                "last-frame log-mel mismatch at ({mel}, 9): {} vs {expected}",
                features[[mel, 9]]
            );
        }
    }

    #[test]
    fn batches_variable_lengths_with_feature_mask() {
        let output = Qwen3AudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(1000), decoded(801)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[2, 128, 6]);
        assert_eq!(output.feature_token_counts, vec![1, 1]);
        assert_eq!(output.item_sizes, vec![(128, 6), (128, 5)]);
        assert!(matches!(
            output.model_specific.get("audio_feature_lengths"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data == &vec![6, 5] && shape == &vec![2]
        ));
        assert!(matches!(
            output.model_specific.get("feature_attention_mask"),
            Some(ModelSpecificValue::IntTensor { data, shape })
                if data == &vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                    && shape == &vec![2, 6]
        ));
    }

    #[test]
    fn rejects_sub_hop_clip_in_mixed_batch() {
        let error = Qwen3AudioProcessor::new()
            .preprocess_decoded_clips(vec![decoded(320), decoded(159)])
            .unwrap_err();

        match error {
            TransformError::ShapeError(message) => assert_eq!(
                message,
                "Qwen3 audio clip at batch index 1 has 159 samples after resampling and truncation; at least 160 are required"
            ),
            other => panic!("expected a shape error, got {other}"),
        }
    }

    #[test]
    fn accepts_per_clip_hop_length_boundary() {
        let processor = Qwen3AudioProcessor::new();
        let expected_token_counts = vec![
            qwen3_audio_output_length(
                320 / processor.params().hop_length,
                processor.params().n_window,
            ),
            qwen3_audio_output_length(
                160 / processor.params().hop_length,
                processor.params().n_window,
            ),
        ];
        assert_eq!(expected_token_counts, vec![1, 1]);

        let output = processor
            .preprocess_decoded_clips(vec![decoded(320), decoded(160)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[2, 128, 2]);
        assert_eq!(output.feature_token_counts, expected_token_counts);
        assert_eq!(output.item_sizes, vec![(128, 2), (128, 1)]);
    }

    #[test]
    fn reads_preprocessor_and_nested_model_config() {
        let preprocessor = PreProcessorConfig::from_json(
            r#"{"sampling_rate": 16000, "feature_size": 64, "n_fft": 320, "hop_length": 80, "n_samples": 4800000}"#,
        )
        .unwrap();
        let params = Qwen3AudioParams::from_configs(
            &serde_json::json!({
                "thinker_config": {"audio_config": {"num_mel_bins": 96, "n_window": 64}}
            }),
            &preprocessor,
        );
        assert_eq!(params.sample_rate, 16_000);
        assert_eq!(params.n_mels, 64);
        assert_eq!(params.n_fft, 320);
        assert_eq!(params.hop_length, 80);
        assert_eq!(params.n_window, 64);
        assert_eq!(params.max_samples, None);

        let truncating_preprocessor = PreProcessorConfig::from_json(
            r#"{"sampling_rate": 16000, "n_samples": 4800000, "truncation": true}"#,
        )
        .unwrap();
        let params =
            Qwen3AudioParams::from_configs(&serde_json::json!({}), &truncating_preprocessor);
        assert_eq!(params.max_samples, Some(4_800_000));

        let params = Qwen3AudioParams::from_configs(
            &serde_json::json!({
                "thinker_config": {"audio_config": {"num_mel_bins": 96}}
            }),
            &PreProcessorConfig::default(),
        );
        assert_eq!(params.n_mels, 96);
        assert_eq!(params.max_samples, None);
    }

    #[test]
    fn n_samples_is_padding_target_not_implicit_truncation() {
        let preprocessor = PreProcessorConfig::from_json(r#"{"n_samples": 320}"#).unwrap();
        let processor = Qwen3AudioProcessor::from_configs(&serde_json::json!({}), &preprocessor);
        let output = processor
            .preprocess_decoded_clips(vec![decoded(800)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 128, 5]);
        assert_eq!(output.item_sizes, vec![(128, 5)]);
    }

    #[test]
    fn truncates_to_explicit_audio_limit() {
        let processor = Qwen3AudioProcessor::with_params(Qwen3AudioParams {
            max_samples: Some(320),
            ..Default::default()
        });
        let output = processor
            .preprocess_decoded_clips(vec![decoded(800)])
            .unwrap();

        assert_eq!(output.encoder_input.shape(), &[1, 128, 2]);
        assert_eq!(output.feature_token_counts, vec![1]);
        assert_eq!(output.item_sizes, vec![(128, 2)]);
    }

    #[test]
    fn qwen_chunked_encoder_output_lengths_match_reference_formula() {
        assert_eq!(qwen3_audio_output_length(0, 50), 0);
        assert_eq!(qwen3_audio_output_length(1, 50), 1);
        assert_eq!(qwen3_audio_output_length(8, 50), 1);
        assert_eq!(qwen3_audio_output_length(9, 50), 2);
        assert_eq!(qwen3_audio_output_length(99, 50), 13);
        assert_eq!(qwen3_audio_output_length(100, 50), 13);
        assert_eq!(qwen3_audio_output_length(101, 50), 14);
        assert_eq!(qwen3_audio_output_length(3000, 50), 390);
        assert_eq!(qwen3_audio_output_length(17, 4), 3);
    }

    #[test]
    fn reflect_padding_matches_numpy_convention() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0], 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
        );
        assert_eq!(reflect_pad(&[2.0], 2), vec![2.0; 5]);
    }
}
