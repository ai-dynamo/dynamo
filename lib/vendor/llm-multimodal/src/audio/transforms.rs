//! Shared digital-signal-processing transforms for audio frontends.

use std::{f32::consts::PI as PI_F32, f64::consts::PI};

use crate::error::TransformError;

pub(super) fn hann_window(window_size: usize) -> Vec<f32> {
    (0..window_size)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / window_size as f64).cos()) as f32)
        .collect()
}

fn hz_to_mel(frequency: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if frequency >= min_log_hz {
        min_log_mel + (frequency / min_log_hz).ln() / logstep
    } else {
        frequency / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}

/// Build a Slaney-normalized mel filter bank.
pub(super) fn mel_basis(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let fft_bins = n_fft / 2 + 1;
    let mut fft_freqs = Vec::with_capacity(fft_bins);
    for bin in 0..fft_bins {
        fft_freqs.push(bin as f64 * sample_rate as f64 / n_fft as f64);
    }

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate as f64 / 2.0);
    let mut mel_edges = Vec::with_capacity(n_mels + 2);
    for i in 0..n_mels + 2 {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_edges.push(mel_to_hz(mel_min + (mel_max - mel_min) * t));
    }

    let mel_widths: Vec<f64> = mel_edges.windows(2).map(|w| w[1] - w[0]).collect();
    let mut weights = vec![0.0_f32; n_mels * fft_bins];
    for mel in 0..n_mels {
        let enorm = 2.0 / (mel_edges[mel + 2] - mel_edges[mel]);
        for (bin, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_edges[mel]) / mel_widths[mel];
            let upper = (mel_edges[mel + 2] - freq) / mel_widths[mel + 1];
            weights[mel * fft_bins + bin] = lower.min(upper).max(0.0).mul_add(enorm, 0.0) as f32;
        }
    }
    weights
}

/// Match torchaudio's default `functional.resample`: band-limited sinc
/// interpolation with a Hann window, filter width 6, and rolloff 0.99.
pub(super) fn bandlimited_resample(
    samples: &[f32],
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<Vec<f32>, TransformError> {
    const LOWPASS_FILTER_WIDTH: f32 = 6.0;
    const ROLLOFF: f32 = 0.99;

    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(TransformError::ShapeError(
            "audio resampling rates must be positive".to_string(),
        ));
    }
    if samples.is_empty() || src_sample_rate == dst_sample_rate {
        return Ok(samples.to_vec());
    }

    let gcd = greatest_common_divisor(src_sample_rate, dst_sample_rate);
    let orig_freq = src_sample_rate / gcd;
    let new_freq = dst_sample_rate / gcd;
    let base_freq = orig_freq.min(new_freq) as f32 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig_freq as f32 / base_freq).ceil() as usize;
    let kernel_len = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(orig_freq))
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel is too large".into()))?;
    let kernel_values = new_freq
        .checked_mul(kernel_len)
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel size overflow".into()))?;
    let mut kernels = Vec::new();
    kernels.try_reserve_exact(kernel_values).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample kernel: {error}"))
    })?;

    let orig_freq_f32 = orig_freq as f32;
    let new_freq_f32 = new_freq as f32;
    let scale = base_freq / orig_freq_f32;
    for phase in 0..new_freq {
        for kernel_index in 0..kernel_len {
            let idx = (kernel_index as f32 - width as f32) / orig_freq_f32;
            let mut t = (idx - phase as f32 / new_freq_f32) * base_freq;
            t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (t * PI_F32 / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
            let radians = t * PI_F32;
            let sinc = if radians == 0.0 {
                1.0
            } else {
                radians.sin() / radians
            };
            kernels.push(sinc * window * scale);
        }
    }

    let target_len_u128 = (samples.len() as u128 * new_freq as u128).div_ceil(orig_freq as u128);
    let target_len = usize::try_from(target_len_u128).map_err(|_| {
        TransformError::ShapeError("resampled audio length exceeds usize".to_string())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(target_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate resampled audio: {error}"))
    })?;

    for block in 0..samples.len().div_ceil(orig_freq) {
        let input_start = block * orig_freq;
        for phase in 0..new_freq {
            if output.len() == target_len {
                return Ok(output);
            }
            let kernel = &kernels[phase * kernel_len..(phase + 1) * kernel_len];
            let mut value = 0.0_f32;
            for (kernel_index, &coefficient) in kernel.iter().enumerate() {
                let padded_index = input_start + kernel_index;
                if padded_index >= width {
                    let sample_index = padded_index - width;
                    if let Some(&sample) = samples.get(sample_index) {
                        value = sample.mul_add(coefficient, value);
                    }
                }
            }
            output.push(value);
        }
    }

    Ok(output)
}

fn greatest_common_divisor(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

#[cfg(test)]
mod tests {
    use super::bandlimited_resample;
    use crate::error::TransformError;

    #[test]
    fn bandlimited_resample_matches_torchaudio_golden_vector() {
        let input = [0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.125, -0.25];
        // torchaudio 2.11 functional.resample(input, 8000, 16000), using
        // the default Hann-windowed sinc kernel.
        let expected = [
            0.012_859_77,
            0.347_330_45,
            0.230_903_15,
            -0.388_803_3,
            -0.476_089_15,
            0.325_961_17,
            0.724_112_45,
            -0.117_642_37,
            -0.975_495_4,
            -0.540_876_27,
            0.479_874_4,
            0.698_470_23,
            0.138_904_54,
            -0.281_137_47,
            -0.257_479_25,
            -0.095_786_646,
        ];

        let actual = bandlimited_resample(&input, 8_000, 16_000).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "resampled sample {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn bandlimited_resample_uses_torchaudio_output_length_formula() {
        for (input_len, source_rate, destination_rate) in [
            (17, 16_000, 8_000),
            (8, 8_000, 11_025),
            (161, 48_000, 44_100),
        ] {
            let input = vec![0.0; input_len];
            let output = bandlimited_resample(&input, source_rate, destination_rate).unwrap();
            let expected_len =
                (input_len as u128 * destination_rate as u128).div_ceil(source_rate as u128);

            assert_eq!(
                output.len(),
                expected_len as usize,
                "unexpected output length for {source_rate} Hz to {destination_rate} Hz"
            );
        }
    }

    #[test]
    fn bandlimited_resample_handles_empty_and_passthrough_inputs() {
        assert_eq!(
            bandlimited_resample(&[], 8_000, 16_000).unwrap(),
            Vec::<f32>::new()
        );

        let input = [0.25, -0.5, 1.0];
        assert_eq!(bandlimited_resample(&input, 16_000, 16_000).unwrap(), input);
    }

    #[test]
    fn bandlimited_resample_rejects_zero_rates() {
        for (source_rate, destination_rate) in [(0, 16_000), (16_000, 0)] {
            assert!(matches!(
                bandlimited_resample(&[0.0], source_rate, destination_rate),
                Err(TransformError::ShapeError(message))
                    if message == "audio resampling rates must be positive"
            ));
        }
    }
}
