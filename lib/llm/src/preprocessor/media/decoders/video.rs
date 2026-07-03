// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::Once;

use anyhow::Result;
use ffmpeg_next::Rational;
use ffmpeg_next::ffi::{AVPixelFormat, av_image_copy_to_buffer};
use memfile::{CreateOptions, MemFile, Seal};
use ndarray::Array4;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use video_rs::frame::RawFrame;
use video_rs::{Location, Time};

use super::Decoder;
use crate::preprocessor::media::{
    DecodedMediaData, EncodedMediaData, decoders::DecodedMediaMetadata,
};

/// Small time buffer (seconds) to avoid edge cases when seeking near frame boundaries
const FRAME_TIME_BUFFER_SECS: f64 = 0.001;
const DEFAULT_MAX_ALLOC: u64 = 512 * 1024 * 1024; // 512 MB
static OPENCV_UNAVAILABLE_WARNING: Once = Once::new();

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum VideoDecoderBackend {
    #[default]
    VideoRs,
    #[serde(rename = "opencv")]
    OpenCv,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    #[serde(default)]
    pub(crate) backend: VideoDecoderBackend,

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

fn get_num_requested_frames_from_metadata(
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
        // frame count based sampling
        // last fallback is to decode all frames
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

fn get_num_requested_frames(
    config: &VideoDecoder,
    decoder: &video_rs::decode::Decoder,
) -> Result<u64> {
    // careful, duration and frames come from file metadata, might be inaccurate
    get_num_requested_frames_from_metadata(
        config,
        decoder.duration()?.as_secs() as f64,
        decoder.frame_rate() as f64,
        decoder.frames().unwrap_or(0),
    )
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

fn get_frame_timestamp(frame: &RawFrame, time_base: Rational) -> Result<f64> {
    anyhow::ensure!(!frame.is_corrupt(), "Frame is corrupt");

    // get timestamp from frame metadata: best_effort_timestamp or pts from ffmpeg
    let best_effort_pts = frame.timestamp();
    let pts = frame.pts();

    match best_effort_pts.or(pts) {
        Some(ts) => Ok(Time::new(Some(ts), time_base).as_secs() as f64),
        None => anyhow::bail!("No timestamp found (both best_effort_pts and pts are None)"),
    }
}

fn decode_frame_at_timestamp(
    decoder: &mut video_rs::decode::Decoder,
    target_time: &Time,
    output_buffer: &mut [u8],
) -> Result<f64> {
    let target_timestamp = target_time.as_secs() as f64;
    let time_base = decoder.time_base();

    // Decode until we reach or pass the target timestamp
    // Caller is responsible for seeking to the appropriate position
    // We use decode_raw_iter to handle timestamps better than video-rs
    for frame_result in decoder.decode_raw_iter() {
        let mut raw_frame =
            frame_result.map_err(|e| anyhow::anyhow!("Frame decode error: {}", e))?;

        let timestamp = match get_frame_timestamp(&raw_frame, time_base) {
            Ok(ts) => ts,
            Err(_) => continue,
        };

        // If we reached the target time or passed it
        if timestamp >= target_timestamp {
            // Copy frame data to provided buffer
            // Adapted from video-rs convert_frame_to_ndarray_rgb24 (private function)
            unsafe {
                let frame_ptr = raw_frame.as_mut_ptr();
                let frame_format = std::mem::transmute::<i32, AVPixelFormat>((*frame_ptr).format);

                let bytes_copied = av_image_copy_to_buffer(
                    output_buffer.as_mut_ptr(),
                    output_buffer.len() as i32,
                    (*frame_ptr).data.as_ptr() as *const *const u8,
                    (*frame_ptr).linesize.as_ptr(),
                    frame_format,
                    raw_frame.width() as i32,
                    raw_frame.height() as i32,
                    1,
                );

                anyhow::ensure!(
                    bytes_copied == output_buffer.len() as i32,
                    "Failed to copy frame data: expected {} bytes, copied {}",
                    output_buffer.len(),
                    bytes_copied
                );
            }

            return Ok(timestamp);
        }
    }

    anyhow::bail!("No frame found for timestamp {target_timestamp:.3}s");
}

impl VideoDecoder {
    fn validate_config(&self) -> Result<()> {
        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        Ok(())
    }

    pub(crate) fn warn_if_unavailable_backend(&self) {
        if self.backend == VideoDecoderBackend::OpenCv && !opencv_backend_available() {
            OPENCV_UNAVAILABLE_WARNING.call_once(|| {
                tracing::warn!(
                    "Video decoder backend `opencv` was selected, but this build does not include \
                     the `media-opencv-video` feature; falling back to `video_rs`"
                );
            });
        }
    }

    fn decode_video_rs(&self, bytes: Vec<u8>) -> Result<DecodedMediaData> {
        // video-rs wants a file path, we use memfile for in-memory file
        let mut mem_file = MemFile::create("video", CreateOptions::new().allow_sealing(true))?;
        mem_file.write_all(&bytes)?;
        mem_file.add_seals(Seal::Write | Seal::Shrink | Seal::Grow)?;
        let fd_path = format!("/proc/self/fd/{}", mem_file.as_raw_fd());
        let location = Location::File(fd_path.into());
        let mut decoder = video_rs::decode::Decoder::new(location)?;

        let requested_frames = get_num_requested_frames(self, &decoder)?;
        let source_duration = decoder.duration()?.as_secs() as f64;
        let source_fps = decoder.frame_rate() as f64;
        let target_times = get_target_times(requested_frames, source_duration, source_fps)?;

        let (width, height) = decoder.size();
        anyhow::ensure!(
            width > 0 && height > 0,
            "Invalid video dimensions {width}x{height}"
        );

        let max_alloc = self.limits.max_alloc.unwrap_or(u64::MAX);
        anyhow::ensure!(
            (width as u64) * (height as u64) * requested_frames * 3 <= max_alloc,
            "Video dimensions {requested_frames}x{width}x{height}x3 exceed max alloc {max_alloc}"
        );

        // Preallocate the buffer for all frames
        let frame_size = width as usize * height as usize * 3;
        let total_size = requested_frames as usize * frame_size;
        let mut all_frames = vec![0u8; total_size];

        let mut sampled_timestamps: Vec<f64> = Vec::with_capacity(requested_frames as usize);
        let mut sequential_mode = false;
        let mut last_successful_time = Time::from_secs(0.0);

        for time in target_times.iter() {
            // Try to seek if not in sequential mode
            if !sequential_mode && let Ok(_) = decoder.seek((time.as_secs() * 1000.0) as i64) {
                sequential_mode = true;
                // Re-establish decoder position at last known good position
                decoder.seek((last_successful_time.as_secs() * 1000.0) as i64)?;
            }

            let offset = sampled_timestamps.len() * frame_size;
            let frame_buffer = &mut all_frames[offset..offset + frame_size];

            match decode_frame_at_timestamp(&mut decoder, time, frame_buffer) {
                Ok(timestamp) => {
                    sampled_timestamps.push(timestamp);
                    last_successful_time = *time;
                }
                Err(error) => {
                    if self.strict {
                        anyhow::bail!(
                            "Frame decode error at timestamp {:.3}s: {}",
                            time.as_secs(),
                            error
                        );
                    }
                    continue;
                }
            }
        }

        video_array_from_frames(
            all_frames,
            sampled_timestamps,
            requested_frames,
            width,
            height,
            source_fps,
            source_duration,
        )
    }

    #[cfg(not(feature = "media-opencv-video"))]
    fn decode_opencv(&self, _bytes: &[u8]) -> Result<Option<DecodedMediaData>> {
        self.warn_if_unavailable_backend();
        Ok(None)
    }

    #[cfg(feature = "media-opencv-video")]
    fn decode_opencv(&self, bytes: &[u8]) -> Result<Option<DecodedMediaData>> {
        Ok(Some(decode_video_with_opencv(self, bytes)?))
    }
}

fn opencv_backend_available() -> bool {
    cfg!(feature = "media-opencv-video")
}

fn video_array_from_frames(
    mut all_frames: Vec<u8>,
    sampled_timestamps: Vec<f64>,
    requested_frames: u64,
    width: u32,
    height: u32,
    source_fps: f64,
    source_duration: f64,
) -> Result<DecodedMediaData> {
    let num_frames_decoded = sampled_timestamps.len();

    anyhow::ensure!(
        num_frames_decoded > 0,
        "Failed to decode any frames, check for video corruption"
    );

    let frame_size = width as usize * height as usize * 3;
    anyhow::ensure!(
        all_frames.len() >= num_frames_decoded * frame_size,
        "Decoded video buffer is smaller than decoded frame metadata"
    );

    // Truncate buffer to actual frames decoded (in case some failed in non-strict mode)
    all_frames.truncate(num_frames_decoded * frame_size);

    let shape = (num_frames_decoded, height as usize, width as usize, 3);
    let array = Array4::from_shape_vec(shape, all_frames)?;
    let mut decoded: DecodedMediaData = array.try_into()?;
    decoded.tensor_info.metadata = Some(DecodedMediaMetadata::Video(VideoMetadata {
        source_fps,
        source_duration,
        sampled_timestamps,
    }));
    anyhow::ensure!(
        num_frames_decoded <= requested_frames as usize,
        "Decoded more frames than requested"
    );
    Ok(decoded)
}

#[cfg(feature = "media-opencv-video")]
fn video_temp_suffix(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && bytes.get(4..8) == Some(b"ftyp") {
        return ".mp4";
    }
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return ".webm";
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"AVI ") {
        return ".avi";
    }
    if bytes.starts_with(b"OggS") {
        return ".ogv";
    }
    if bytes.starts_with(&[0x00, 0x00, 0x01, 0xba]) {
        return ".mpg";
    }
    ".video"
}

#[cfg(feature = "media-opencv-video")]
fn decode_video_with_opencv(config: &VideoDecoder, bytes: &[u8]) -> Result<DecodedMediaData> {
    use opencv::{core::Mat, prelude::*, videoio};

    let mut input_file = tempfile::Builder::new()
        .prefix("dynamo-video-")
        .suffix(video_temp_suffix(bytes))
        .tempfile()?;
    input_file.write_all(bytes)?;
    input_file.flush()?;

    let input = input_file.path().to_str().ok_or_else(|| {
        anyhow::anyhow!(
            "OpenCV video path is not valid UTF-8: {}",
            input_file.path().display()
        )
    })?;

    let mut capture = open_opencv_video_capture(input)?;
    let total_frames = capture.get(videoio::CAP_PROP_FRAME_COUNT)?.round().max(0.0) as u64;
    let source_fps = capture.get(videoio::CAP_PROP_FPS)?;
    let source_duration = if source_fps.is_finite() && source_fps > 0.0 {
        total_frames as f64 / source_fps
    } else {
        0.0
    };

    let requested_frames =
        get_num_requested_frames_from_metadata(config, source_duration, source_fps, total_frames)?;
    let target_times = get_target_times(requested_frames, source_duration, source_fps)?;

    let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)?.round().max(0.0) as u32;
    let height = capture
        .get(videoio::CAP_PROP_FRAME_HEIGHT)?
        .round()
        .max(0.0) as u32;
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
    let total_size = requested_frames as usize * frame_size;
    let mut all_frames = vec![0u8; total_size];
    let mut sampled_timestamps: Vec<f64> = Vec::with_capacity(requested_frames as usize);

    let mut bgr_frame = Mat::default();
    let mut rgb_frame = Mat::default();

    for time in target_times.iter() {
        let frame_index = opencv_target_frame_index(*time, source_fps, total_frames);
        let offset = sampled_timestamps.len() * frame_size;
        let frame_buffer = &mut all_frames[offset..offset + frame_size];

        match decode_opencv_frame_at_index(
            &mut capture,
            &mut bgr_frame,
            &mut rgb_frame,
            frame_index,
            width,
            height,
            frame_buffer,
        ) {
            Ok(()) => sampled_timestamps.push(frame_index as f64 / source_fps),
            Err(error) => {
                if config.strict {
                    anyhow::bail!(
                        "OpenCV frame decode error at timestamp {:.3}s: {}",
                        time.as_secs(),
                        error
                    );
                }
                continue;
            }
        }
    }

    video_array_from_frames(
        all_frames,
        sampled_timestamps,
        requested_frames,
        width,
        height,
        source_fps,
        source_duration,
    )
}

#[cfg(feature = "media-opencv-video")]
fn open_opencv_video_capture(input: &str) -> Result<opencv::videoio::VideoCapture> {
    use opencv::{prelude::*, videoio};

    let capture = videoio::VideoCapture::from_file(input, videoio::CAP_FFMPEG)?;
    if capture.is_opened()? {
        return Ok(capture);
    }

    let capture = videoio::VideoCapture::from_file(input, videoio::CAP_ANY)?;
    if capture.is_opened()? {
        return Ok(capture);
    }

    anyhow::bail!("OpenCV could not open video: {input}")
}

#[cfg(feature = "media-opencv-video")]
fn opencv_target_frame_index(target_time: Time, source_fps: f64, total_frames: u64) -> u64 {
    let index = ((target_time.as_secs() as f64) * source_fps).ceil() as u64;
    index.min(total_frames.saturating_sub(1))
}

#[cfg(feature = "media-opencv-video")]
fn decode_opencv_frame_at_index(
    capture: &mut opencv::videoio::VideoCapture,
    bgr_frame: &mut opencv::core::Mat,
    rgb_frame: &mut opencv::core::Mat,
    frame_index: u64,
    expected_width: u32,
    expected_height: u32,
    output_buffer: &mut [u8],
) -> Result<()> {
    use opencv::{imgproc, prelude::*, videoio};

    anyhow::ensure!(
        capture.set(videoio::CAP_PROP_POS_FRAMES, frame_index as f64)?,
        "OpenCV could not seek to sampled frame {frame_index}"
    );

    anyhow::ensure!(
        capture.read(bgr_frame)? && !bgr_frame.empty(),
        "OpenCV produced no frame at sampled frame {frame_index}"
    );

    let color_code = match bgr_frame.channels() {
        3 => imgproc::COLOR_BGR2RGB,
        4 => imgproc::COLOR_BGRA2RGB,
        channels => anyhow::bail!("OpenCV produced unsupported {channels}-channel frame"),
    };
    imgproc::cvt_color_def(bgr_frame, rgb_frame, color_code)?;

    let decoded_width = u32::try_from(rgb_frame.cols())
        .map_err(|_| anyhow::anyhow!("OpenCV produced invalid frame width"))?;
    let decoded_height = u32::try_from(rgb_frame.rows())
        .map_err(|_| anyhow::anyhow!("OpenCV produced invalid frame height"))?;
    anyhow::ensure!(
        decoded_width == expected_width && decoded_height == expected_height,
        "OpenCV frame dimensions {decoded_width}x{decoded_height} differ from metadata {expected_width}x{expected_height}"
    );

    let rgb_bytes = rgb_frame.data_bytes()?;
    anyhow::ensure!(
        rgb_bytes.len() >= output_buffer.len(),
        "OpenCV produced {} RGB bytes for {decoded_width}x{decoded_height} frame, expected {}",
        rgb_bytes.len(),
        output_buffer.len()
    );
    output_buffer.copy_from_slice(&rgb_bytes[..output_buffer.len()]);
    Ok(())
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
        self.validate_config()?;

        let bytes = data.into_bytes()?;
        if self.backend == VideoDecoderBackend::OpenCv
            && let Some(decoded) = self.decode_opencv(&bytes)?
        {
            return Ok(decoded);
        }

        self.decode_video_rs(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::rdma::DataType;
    use super::*;
    use rstest::rstest;

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

    #[test]
    fn test_default_backend_is_video_rs() {
        assert_eq!(
            VideoDecoder::default().backend,
            VideoDecoderBackend::VideoRs
        );
    }

    #[test]
    fn test_parse_opencv_backend_config() {
        let decoder: VideoDecoder = serde_json::from_value(serde_json::json!({
            "backend": "opencv",
            "num_frames": 1,
        }))
        .unwrap();

        assert_eq!(decoder.backend, VideoDecoderBackend::OpenCv);
        assert_eq!(decoder.num_frames, Some(1));
    }

    #[test]
    fn test_decode_video_opencv_backend_or_fallback() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_10.mp4");

        let decoder = VideoDecoder {
            backend: VideoDecoderBackend::OpenCv,
            limits: VideoDecoderLimits::default(),
            fps: None,
            max_frames: None,
            num_frames: Some(2),
            strict: false,
        };

        let decoded = decoder.decode(encoded_data).unwrap();

        assert_eq!(decoded.tensor_info.shape[0], 2);
        assert_eq!(decoded.tensor_info.shape[1], height as usize);
        assert_eq!(decoded.tensor_info.shape[2], width as usize);
        assert_eq!(decoded.tensor_info.shape[3], 3);
        assert_eq!(decoded.tensor_info.dtype, DataType::UINT8);
    }

    #[test]
    fn test_decode_video_num_frames() {
        let (encoded_data, width, height, _total_frames) = load_test_video("240p_10.mp4");

        let requested_frames = 5u64;
        let decoder = VideoDecoder {
            backend: VideoDecoderBackend::VideoRs,
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
            backend: VideoDecoderBackend::VideoRs,
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
            backend: VideoDecoderBackend::VideoRs,
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
            backend: VideoDecoderBackend::VideoRs,
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
            backend: VideoDecoderBackend::VideoRs,
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
            backend: VideoDecoderBackend::OpenCv,
            limits: runtime_limits,
            fps: Some(60.0),
            ..Default::default()
        };

        let merged = server_config.with_runtime(Some(&runtime_config));

        // Check that server limits are preserved
        assert_eq!(merged.limits.max_alloc, Some(1024));

        // Check that other fields are overridden
        assert_eq!(merged.fps, Some(60.0));
        assert_eq!(merged.backend, VideoDecoderBackend::OpenCv);
    }

    #[cfg(feature = "media-opencv-video")]
    #[test]
    fn test_video_temp_suffix_detects_common_containers() {
        let mut mp4 = vec![0; 12];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert_eq!(video_temp_suffix(&mp4), ".mp4");
        assert_eq!(video_temp_suffix(&[0x1a, 0x45, 0xdf, 0xa3]), ".webm");
        assert_eq!(video_temp_suffix(b"RIFFxxxxAVI "), ".avi");
        assert_eq!(video_temp_suffix(b"OggS"), ".ogv");
        assert_eq!(video_temp_suffix(&[0x00, 0x00, 0x01, 0xba]), ".mpg");
        assert_eq!(video_temp_suffix(b"unknown"), ".video");
    }
}
