// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::os::fd::AsRawFd;

use anyhow::Result;
use ffmpeg_next::Rational;
use memfile::{CreateOptions, MemFile, Seal};
use ndarray::Array4;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use video_rs::Time;

use super::Decoder;
use crate::preprocessor::media::{
    DecodedMediaData, EncodedMediaData, decoders::DecodedMediaMetadata,
};

/// Small time buffer (seconds) to avoid edge cases when seeking near frame boundaries
const FRAME_TIME_BUFFER_SECS: f64 = 0.001;
const DEFAULT_MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 MB

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoderLimits {
    /// Maximum allowed total allocation of decoded frames in bytes
    #[serde(default)]
    pub max_alloc: Option<u64>,
}

impl Default for VideoDecoderLimits {
    fn default() -> Self {
        Self {
            max_alloc: Some(DEFAULT_MAX_ALLOC),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    #[serde(default)]
    pub(crate) limits: VideoDecoderLimits,

    /// sample N frames per second
    #[serde(default)]
    pub(crate) fps: Option<f64>,
    /// sample at most N frames (used with fps)
    #[serde(default)]
    pub(crate) max_frames: Option<u64>,
    /// sample N frames in total (linspace)
    #[serde(default)]
    pub(crate) num_frames: Option<u64>,
    /// fail if some frames fail to decode
    #[serde(default)]
    pub(crate) strict: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoMetadata {
    pub(crate) source_fps: f64,
    pub(crate) source_duration: f64,
    pub(crate) sampled_timestamps: Vec<f64>,
}

fn get_num_requested_frames(
    config: &VideoDecoder,
    duration_secs: f64,
    frame_rate: f64,
    mut total_frames: u64,
) -> Result<u64> {
    if total_frames == 0 && duration_secs > 0.0 && frame_rate > 0.0 {
        total_frames = (duration_secs * frame_rate) as u64;
    }

    anyhow::ensure!(total_frames > 0, "Cannot determine the video frame count");

    let requested_frames = if let Some(target_fps) = config.fps {
        // fps based sampling
        anyhow::ensure!(duration_secs > 0.0, "Cannot determine the video duration");
        (duration_secs * target_fps) as u64
    } else {
        // frame count based sampling; the last fallback is to decode all frames
        config.num_frames.unwrap_or(total_frames)
    };

    let requested_frames = requested_frames
        .min(config.max_frames.unwrap_or(requested_frames))
        .max(1);

    anyhow::ensure!(
        requested_frames > 0 && requested_frames <= total_frames,
        "Cannot decode {requested_frames} frames from {total_frames} total frames",
    );

    Ok(requested_frames)
}

fn get_target_times(
    requested_frames: u64,
    duration_secs: f64,
    frame_rate: f64,
) -> Result<Vec<Time>> {
    anyhow::ensure!(
        requested_frames > 0,
        "Invalid requested frames {requested_frames}"
    );
    anyhow::ensure!(duration_secs > 0.0, "Invalid duration {duration_secs}");
    anyhow::ensure!(frame_rate > 0.0, "Invalid frame rate {frame_rate}");

    let frame_duration = 1.0 / frame_rate;
    // Add small buffer to avoid edge cases
    // Variable frame rate might not work well here
    let last_frame_time = (duration_secs - frame_duration - FRAME_TIME_BUFFER_SECS).max(0.0);

    if requested_frames == 1 {
        return Ok(vec![Time::from_secs(last_frame_time as f32 / 2.0)]);
    }

    Ok((0..requested_frames)
        .map(|i| {
            let time_secs = (i as f64 * last_frame_time) / (requested_frames as f64 - 1.0);
            Time::from_secs(time_secs.max(0.0) as f32)
        })
        .collect())
}

fn get_frame_timestamp(frame: &ffmpeg_next::frame::Video, time_base: Rational) -> Result<f64> {
    anyhow::ensure!(!frame.is_corrupt(), "Frame is corrupt");

    // get timestamp from frame metadata: best_effort_timestamp or pts from ffmpeg
    let best_effort_pts = frame.timestamp();
    let pts = frame.pts();

    match best_effort_pts.or(pts) {
        Some(ts) => Ok(Time::new(Some(ts), time_base).as_secs() as f64),
        None => anyhow::bail!("No timestamp found (both best_effort_pts and pts are None)"),
    }
}

fn handle_sample_error(
    config: &VideoDecoder,
    target_index: &mut usize,
    target_count: usize,
    error: anyhow::Error,
) -> Result<bool> {
    if config.strict {
        return Err(error);
    }

    tracing::debug!(%error, target_index = *target_index, "Skipping failed video sample");
    *target_index += 1;
    Ok(*target_index == target_count)
}

fn copy_rgb_frame(frame: &ffmpeg_next::frame::Video, output_buffer: &mut [u8]) -> Result<()> {
    let width = frame.width();
    let height = frame.height();
    let row_bytes = width as usize * 3;
    anyhow::ensure!(
        output_buffer.len() == row_bytes * height as usize,
        "Invalid RGB output buffer size"
    );

    let stride = frame.stride(0);
    anyhow::ensure!(stride >= row_bytes, "FFmpeg RGB frame stride is too small");
    let rgb_data = frame.data(0);
    anyhow::ensure!(
        rgb_data.len() >= stride * height as usize,
        "FFmpeg RGB frame data is truncated"
    );
    for row in 0..height as usize {
        let source_offset = row * stride;
        let output_offset = row * row_bytes;
        output_buffer[output_offset..output_offset + row_bytes]
            .copy_from_slice(&rgb_data[source_offset..source_offset + row_bytes]);
    }
    Ok(())
}

fn convert_ffmpeg_frame_to_rgb(
    scaler: &mut ffmpeg_next::software::scaling::Context,
    decoded_frame: &ffmpeg_next::frame::Video,
    rgb_frame: &mut ffmpeg_next::frame::Video,
    output_buffer: &mut [u8],
) -> Result<()> {
    scaler.run(decoded_frame, rgb_frame)?;
    copy_rgb_frame(rgb_frame, output_buffer)
}

fn decode_video(config: &VideoDecoder, bytes: Vec<u8>) -> Result<DecodedMediaData> {
    use ffmpeg_next::codec::context::Context;
    use ffmpeg_next::software::scaling::{Context as ScalingContext, Flags};
    use ffmpeg_next::util::format::pixel::Pixel;

    let mut mem_file = MemFile::create("video", CreateOptions::new().allow_sealing(true))?;
    mem_file.write_all(&bytes)?;
    drop(bytes);
    mem_file.add_seals(Seal::Write | Seal::Shrink | Seal::Grow)?;
    let fd_path = format!("/proc/self/fd/{}", mem_file.as_raw_fd());
    let mut input = ffmpeg_next::format::input(&fd_path)?;

    let (stream_index, stream_time_base, source_duration, source_fps, total_frames, parameters) = {
        let input_stream = input
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| anyhow::anyhow!("FFmpeg could not find a video stream"))?;
        let stream_time_base = input_stream.time_base();
        let frame_rate = input_stream.rate();
        anyhow::ensure!(
            frame_rate.denominator() > 0,
            "Cannot determine the video frame rate"
        );
        (
            input_stream.index(),
            stream_time_base,
            Time::new(Some(input_stream.duration()), stream_time_base).as_secs() as f64,
            (frame_rate.numerator() as f32 / frame_rate.denominator() as f32) as f64,
            input_stream.frames().max(0) as u64,
            input_stream.parameters(),
        )
    };

    // Duration and frame count come from file metadata and might be inaccurate.
    let requested_frames =
        get_num_requested_frames(config, source_duration, source_fps, total_frames)?;
    let target_times = get_target_times(requested_frames, source_duration, source_fps)?;

    let mut decoder_context = Context::new();
    decoder_context.set_time_base(stream_time_base);
    decoder_context.set_parameters(parameters)?;
    let mut decoder = decoder_context.decoder().video()?;
    let decoder_time_base = decoder.time_base();
    let (width, height) = (decoder.width(), decoder.height());
    anyhow::ensure!(
        width > 0 && height > 0,
        "Invalid video dimensions {width}x{height}"
    );

    let max_alloc = config.limits.max_alloc.unwrap_or(u64::MAX);
    anyhow::ensure!(
        (width as u64) * (height as u64) * requested_frames * 3 <= max_alloc,
        "Video dimensions {requested_frames}x{width}x{height}x3 exceed max alloc {max_alloc}"
    );

    let frame_size = width as usize * height as usize * 3;
    let mut all_frames = vec![0u8; requested_frames as usize * frame_size];
    let mut sampled_timestamps = Vec::with_capacity(requested_frames as usize);
    let mut target_index = 0usize;
    let mut decoded_frame = ffmpeg_next::frame::Video::empty();
    let mut rgb_frame = ffmpeg_next::frame::Video::empty();
    let mut scaler = ScalingContext::get(
        decoder.format(),
        width,
        height,
        Pixel::RGB24,
        width,
        height,
        Flags::AREA,
    )?;

    let mut receive_frames = |decoder: &mut ffmpeg_next::decoder::Video,
                              target_index: &mut usize,
                              sampled_timestamps: &mut Vec<f64>|
     -> Result<bool> {
        loop {
            match decoder.receive_frame(&mut decoded_frame) {
                Ok(()) => {
                    let timestamp = match get_frame_timestamp(&decoded_frame, decoder_time_base) {
                        Ok(timestamp) => timestamp,
                        Err(_) => continue,
                    };
                    if timestamp < target_times[*target_index].as_secs() as f64 {
                        continue;
                    }

                    let offset = sampled_timestamps.len() * frame_size;
                    if let Err(error) = convert_ffmpeg_frame_to_rgb(
                        &mut scaler,
                        &decoded_frame,
                        &mut rgb_frame,
                        &mut all_frames[offset..offset + frame_size],
                    ) {
                        let error = anyhow::anyhow!(
                            "FFmpeg RGB conversion error at timestamp {timestamp:.3}s: {error:?}"
                        );
                        if handle_sample_error(config, target_index, target_times.len(), error)? {
                            return Ok(true);
                        }
                        continue;
                    }
                    sampled_timestamps.push(timestamp);
                    *target_index += 1;
                    if *target_index == target_times.len() {
                        return Ok(true);
                    }
                }
                Err(ffmpeg_next::Error::Other {
                    errno: ffmpeg_next::error::EAGAIN,
                })
                | Err(ffmpeg_next::Error::Eof) => return Ok(false),
                Err(error) => {
                    let error = anyhow::anyhow!("FFmpeg frame decode error: {error:?}");
                    if handle_sample_error(config, target_index, target_times.len(), error)? {
                        return Ok(true);
                    }
                    return Ok(false);
                }
            }
        }
    };

    let mut finished = false;
    for (stream, mut packet) in input.packets() {
        if stream.index() != stream_index {
            continue;
        }
        packet.rescale_ts(stream.time_base(), decoder_time_base);
        if let Err(error) = decoder.send_packet(&packet) {
            let error = anyhow::anyhow!("FFmpeg packet decode error: {error:?}");
            if handle_sample_error(config, &mut target_index, target_times.len(), error)? {
                finished = true;
                break;
            }
            continue;
        }
        if receive_frames(&mut decoder, &mut target_index, &mut sampled_timestamps)? {
            finished = true;
            break;
        }
    }

    if !finished {
        match decoder.send_eof() {
            Ok(()) => {
                finished =
                    receive_frames(&mut decoder, &mut target_index, &mut sampled_timestamps)?;
            }
            Err(error) => {
                let error = anyhow::anyhow!("FFmpeg decoder flush error: {error:?}");
                finished =
                    handle_sample_error(config, &mut target_index, target_times.len(), error)?;
            }
        }
    }
    anyhow::ensure!(
        !config.strict || finished,
        "FFmpeg reached end of video after decoding {} of {requested_frames} requested frames",
        sampled_timestamps.len()
    );

    let num_frames_decoded = sampled_timestamps.len();
    anyhow::ensure!(
        num_frames_decoded > 0,
        "Failed to decode any frames, check for video corruption"
    );
    all_frames.truncate(num_frames_decoded * frame_size);

    let shape = (num_frames_decoded, height as usize, width as usize, 3);
    let array = Array4::from_shape_vec(shape, all_frames)?;
    let mut decoded: DecodedMediaData = array.try_into()?;
    decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Video(VideoMetadata {
        source_fps,
        source_duration,
        sampled_timestamps,
    }));
    Ok(decoded)
}

impl Decoder for VideoDecoder {
    fn with_runtime(&self, runtime: Option<&Self>) -> Self {
        match runtime {
            Some(r) => {
                let mut d = r.clone();
                d.limits.clone_from(&self.limits);
                d
            }
            None => self.clone(),
        }
    }

    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        decode_video(self, data.into_bytes()?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::rdma::DataType;
    use super::*;
    use rstest::rstest;
    use video_rs::Location;

    /// Load test video and parse expected dimensions from filename.
    /// Filename format: "{resolution}_{frames}.mp4" (e.g., "240p_10.mp4" -> 320x240, 10 frames)
    fn load_test_video(filename: &str) -> (EncodedMediaData, u32, u32, u32) {
        let path = format!(
            "{}/tests/data/media/{}",
            env!("CARGO_MANIFEST_DIR"),
            filename
        );
        let bytes =
            std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read test video: {}", path));

        let parts: Vec<&str> = filename.strip_suffix(".mp4").unwrap().split('_').collect();
        let resolution = parts[0];
        let frames = parts[1].parse::<u32>().unwrap();

        let (width, height) = match resolution {
            "2p" => (2, 2),
            "240p" => (320, 240),
            "2160p" => (3840, 2160),
            _ => panic!("Unknown resolution: {}", resolution),
        };

        let encoded = EncodedMediaData {
            bytes,
            b64_encoded: false,
        };

        (encoded, width, height, frames)
    }

    fn decode_reference_with_video_rs(
        bytes: &[u8],
        requested_frames: u64,
    ) -> Result<(Vec<u8>, Vec<f64>)> {
        let mut mem_file = MemFile::create("video", CreateOptions::new().allow_sealing(true))?;
        mem_file.write_all(bytes)?;
        mem_file.add_seals(Seal::Write | Seal::Shrink | Seal::Grow)?;
        let location = Location::File(format!("/proc/self/fd/{}", mem_file.as_raw_fd()).into());
        let mut decoder = video_rs::decode::Decoder::new(location)?;
        let target_times = get_target_times(
            requested_frames,
            decoder.duration()?.as_secs() as f64,
            decoder.frame_rate() as f64,
        )?;
        let (width, height) = decoder.size();
        let frame_size = width as usize * height as usize * 3;
        let mut frames = vec![0; requested_frames as usize * frame_size];
        let mut sampled_timestamps = Vec::with_capacity(requested_frames as usize);
        let mut target_index = 0;
        let time_base = decoder.time_base();

        for result in decoder.decode_raw_iter() {
            let frame = result?;
            let timestamp = get_frame_timestamp(&frame, time_base)?;
            if timestamp < target_times[target_index].as_secs() as f64 {
                continue;
            }
            let output = &mut frames[target_index * frame_size..(target_index + 1) * frame_size];
            copy_rgb_frame(&frame, output)?;
            sampled_timestamps.push(timestamp);
            target_index += 1;
            if target_index == target_times.len() {
                break;
            }
        }

        anyhow::ensure!(
            target_index == target_times.len(),
            "video-rs reference decoded {target_index} of {requested_frames} requested frames"
        );
        Ok((frames, sampled_timestamps))
    }

    #[test]
    fn test_decode_video_num_frames() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_10.mp4");

        let requested_frames = 5u64;
        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: None,
            max_frames: None,
            num_frames: Some(requested_frames),
            strict: false,
        };

        let decoded = decoder.decode(encoded_data).unwrap();

        assert_eq!(decoded.tensor_info.shape[0], requested_frames as usize);
        assert_eq!(decoded.tensor_info.shape[1], height as usize);
        assert_eq!(decoded.tensor_info.shape[2], width as usize);
        assert_eq!(decoded.tensor_info.shape[3], 3);
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[test]
    fn test_decode_video_fps_sampling() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_100.mp4");

        let target_fps = 0.5f64;
        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: Some(target_fps),
            max_frames: None,
            num_frames: None,
            strict: false,
        };

        let decoded = decoder.decode(encoded_data).unwrap();

        // fps * duration calculation - video decoder uses duration from file
        // Source file is at 1fps, should get exactly 50 frames
        assert_eq!(decoded.tensor_info.shape[0], 50);
        assert_eq!(decoded.tensor_info.shape[1], height as usize);
        assert_eq!(decoded.tensor_info.shape[2], width as usize);
        assert_eq!(decoded.tensor_info.shape[3], 3);
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[test]
    fn test_sampled_decode_matches_video_rs() {
        use dynamo_memory::actions::Slice;

        let (encoded_data, ..) = load_test_video("240p_10.mp4");
        let bytes = encoded_data.into_bytes().unwrap();
        let requested_frames = 5;
        let decoder: VideoDecoder = serde_json::from_value(serde_json::json!({
            "num_frames": requested_frames,
            "strict": true,
        }))
        .unwrap();

        let decoded = decoder
            .decode(EncodedMediaData::from_bytes(bytes.clone()))
            .unwrap();
        let (reference_frames, reference_timestamps) =
            decode_reference_with_video_rs(&bytes, requested_frames).unwrap();

        let decoded_frames = unsafe { decoded.data.as_slice().unwrap() };
        assert_eq!(decoded_frames, reference_frames);
        let Some(DecodedMediaMetadata::Video(metadata)) = decoded.tensor_info.metadata else {
            panic!("missing video metadata");
        };
        assert_eq!(metadata.sampled_timestamps, reference_timestamps);
    }

    #[test]
    fn test_sample_error_respects_strict_mode() {
        // Two 16x16 frames followed by two 32x32 frames trigger swscale's InputChanged error.
        let path = format!(
            "{}/tests/data/media/dynamic_resolution_4.mp4",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(path).unwrap();
        let decoder = |strict| VideoDecoder {
            num_frames: Some(4),
            strict,
            ..Default::default()
        };

        let decoded = decoder(false)
            .decode(EncodedMediaData::from_bytes(bytes.clone()))
            .unwrap();
        assert_eq!(decoded.tensor_info.shape, vec![2, 16, 16, 3]);
        let Some(DecodedMediaMetadata::Video(metadata)) = decoded.tensor_info.metadata else {
            panic!("missing video metadata");
        };
        assert_eq!(metadata.sampled_timestamps, vec![0.0, 0.5]);

        let error = decoder(true)
            .decode(EncodedMediaData::from_bytes(bytes))
            .unwrap_err()
            .to_string();
        assert!(error.contains("FFmpeg RGB conversion error"), "{error}");
    }

    #[rstest]
    #[case(Some(320 * 240 * 5 * 3), "240p_10.mp4", 5, true, "within limit")]
    #[case(Some(320 * 240 * 2 * 3), "240p_10.mp4", 5, false, "exceeds limit")]
    #[case(Some(2 * 2 * 10 * 3), "2p_10.mp4", 10, true, "exactly at limit")]
    #[case(None, "2160p_10.mp4", 10, true, "no limit")]
    fn test_max_alloc(
        #[case] max_alloc: Option<u64>,
        #[case] video_file: &str,
        #[case] num_frames: u64,
        #[case] should_succeed: bool,
        #[case] test_case: &str,
    ) {
        let (encoded_data, width, height, _) = load_test_video(video_file);

        let decoder = VideoDecoder {
            limits: VideoDecoderLimits { max_alloc },
            fps: None,
            max_frames: None,
            num_frames: Some(num_frames),
            strict: false,
        };

        let result = decoder.decode(encoded_data);

        if should_succeed {
            assert!(
                result.is_ok(),
                "Should decode successfully for case: {test_case}",
            );
            let decoded = result.unwrap();
            assert_eq!(decoded.tensor_info.shape[1], height as usize);
            assert_eq!(decoded.tensor_info.shape[2], width as usize);
            assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
        } else {
            assert!(result.is_err(), "Should fail for case: {}", test_case);
        }
    }

    #[test]
    fn test_conflicting_fps_and_num_frames() {
        let (encoded_data, ..) = load_test_video("240p_10.mp4");

        let decoder = VideoDecoder {
            limits: VideoDecoderLimits::default(),
            fps: Some(2.0f64),
            max_frames: None,
            num_frames: Some(5u64),
            strict: false,
        };

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_err(),
            "Should fail when both fps and num_frames are specified"
        );
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("cannot be specified at the same time"));
    }

    // Unit tests for get_target_times

    #[test]
    fn test_get_target_times() {
        // 10 frames at 1fps over 10s duration
        let times = get_target_times(10u64, 10.0f64, 1.0f64).unwrap();
        assert_eq!(times.len(), 10);

        assert_eq!(times[0].as_secs(), 0.0);

        // Last frame should be less than 9s (10 - 1/1fps - 0.001)
        let last_time = times[9].as_secs();
        assert!(
            last_time < 9.0,
            "Last time should be < 9s, got {}",
            last_time
        );
        assert!(
            last_time > 8.0,
            "Last time should be > 8s, got {}",
            last_time
        );
    }

    #[test]
    fn test_with_runtime_limit_enforcement() {
        let server_limits = VideoDecoderLimits {
            max_alloc: Some(1024),
        };
        let server_config = VideoDecoder {
            limits: server_limits,
            fps: Some(1.0),
            ..Default::default()
        };

        // Runtime config tries to override limits (should be ignored)
        // And sets different FPS (should be accepted)
        let runtime_limits = VideoDecoderLimits {
            max_alloc: Some(999999),
        };
        let runtime_config = VideoDecoder {
            limits: runtime_limits,
            fps: Some(60.0),
            ..Default::default()
        };

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_alloc, Some(1024));

        // Check that other fields are overridden
        assert_eq!(merged.fps, Some(60.0));
    }
}
