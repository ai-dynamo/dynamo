// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::common::{DecodedMediaData, Decoder, EncodedMediaData};
use anyhow::Result;
use ndarray::Array4;
use std::io::Write;
use tempfile::NamedTempFile;
use video_rs;
use video_rs::location::Location;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VideoDecoder {
    // sample N frames per second
    #[serde(default)]
    pub fps: Option<f32>,
    // sample at most N frames (used with fps)
    #[serde(default)]
    pub max_frames: Option<u32>,
    // sample N frames in total (linspace)
    #[serde(default)]
    pub num_frames: Option<u32>,
    // fail if some frames fail to decode
    #[serde(default)]
    pub strict: bool,
    // maximum total size of the sampled frames in pixels
    #[serde(default)]
    pub max_pixels: Option<usize>,
}

impl Decoder for VideoDecoder {
    fn decode(&self, data: EncodedMediaData) -> Result<DecodedMediaData> {
        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        let bytes = data.into_bytes()?;

        // video-rs wants a file path, we use tmpfs / ramdisk
        let mut temp_file = NamedTempFile::with_prefix("video")?;
        temp_file.write_all(&bytes)?;
        temp_file.flush()?;

        let location = Location::File(temp_file.path().to_path_buf());
        let mut decoder = video_rs::decode::Decoder::new(location)?;
        let total_frames = decoder.frames()? as u32; // note: this comes from the metadata and might not be exact

        let requested_frames = if let Some(target_fps) = self.fps {
            let duration = decoder.duration()?.as_secs();
            (duration * target_fps) as u32
        } else {
            self.num_frames.unwrap_or(total_frames)
        };

        let requested_frames = requested_frames.min(self.max_frames.unwrap_or(requested_frames));

        anyhow::ensure!(
            requested_frames > 0 && requested_frames <= total_frames,
            "Cannot decode {requested_frames} frames from {total_frames} total frames",
        );

        let (width, height) = decoder.size();
        anyhow::ensure!(
            width > 0 && height > 0,
            "Invalid video dimensions {width}x{height}"
        );
        let max_pixels = self.max_pixels.unwrap_or(usize::MAX);
        anyhow::ensure!(
            (width as usize) * (height as usize) * (requested_frames as usize) <= max_pixels,
            "Video dimensions {requested_frames}x{width}x{height} exceed max pixels {max_pixels}"
        );

        let mut all_frames =
            Vec::with_capacity(requested_frames as usize * width as usize * height as usize * 3);
        let mut num_frames_decoded = 0;

        let target_indices = if requested_frames == 1 {
            vec![total_frames / 2]
        } else {
            (0..requested_frames)
                .map(|i| (i * (total_frames - 1)) / (requested_frames - 1))
                .collect()
        };

        // Decode all frames sequentially (required for P/B-frames), but only keep target frames
        // TODO: smarter seek-based decoding for better sparse sampling
        for (current_frame_idx, result) in decoder.decode_iter().enumerate() {
            match result {
                Ok((_ts, frame)) => {
                    // Only keep frames at our target indices
                    if target_indices.contains(&(current_frame_idx as u32)) {
                        all_frames.extend_from_slice(frame.as_slice().unwrap());
                        num_frames_decoded += 1;

                        // early exit when we're done
                        if num_frames_decoded >= requested_frames {
                            break;
                        }
                    }
                }
                Err(video_rs::Error::ReadExhausted | video_rs::Error::DecodeExhausted) => {
                    // EOF
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }

        anyhow::ensure!(num_frames_decoded > 0, "Failed to decode any frames");
        if self.strict {
            anyhow::ensure!(
                num_frames_decoded == requested_frames,
                "Failed to decode all requested frames (strict mode)"
            );
        }

        let shape = (
            num_frames_decoded as usize,
            height as usize,
            width as usize,
            3,
        );
        let array = Array4::from_shape_vec(shape, all_frames)?;
        Ok(array.try_into()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn create_encoded_media_data(bytes: Vec<u8>) -> EncodedMediaData {
        EncodedMediaData {
            bytes,
            b64_encoded: false,
        }
    }

    // Load a test video fixture file
    fn load_test_video(filename: &str) -> Vec<u8> {
        let path = format!(
            "{}/tests/data/media/{}",
            env!("CARGO_MANIFEST_DIR"),
            filename
        );
        std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read test video: {}", path))
    }

    // Get expected frame count and dimensions from filename (e.g., "240p_10.mp4" -> 320x240, 10 frames)
    fn parse_video_info(filename: &str) -> (u32, u32, u32) {
        let parts: Vec<&str> = filename.strip_suffix(".mp4").unwrap().split('_').collect();
        let resolution = parts[0];
        let frames = parts[1].parse::<u32>().unwrap();

        let (width, height) = match resolution {
            "2p" => (2, 2),
            "240p" => (320, 240),
            "2160p" => (3840, 2160),
            _ => panic!("Unknown resolution: {}", resolution),
        };

        (width, height, frames)
    }

    #[test]
    fn test_decode_video_all_frames() {
        let video_bytes = load_test_video("240p_10.mp4");
        let (width, height, expected_frames) = parse_video_info("240p_10.mp4");

        let decoder = VideoDecoder::default();
        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);
        assert!(result.is_ok(), "Failed to decode video");

        let decoded = result.unwrap();

        // Should decode all frames
        assert_eq!(
            decoded.shape[0], expected_frames as usize,
            "Should decode all {} frames",
            expected_frames
        );
        assert_eq!(
            decoded.shape[1], height as usize,
            "Height should be {}",
            height
        );
        assert_eq!(
            decoded.shape[2], width as usize,
            "Width should be {}",
            width
        );
        assert_eq!(decoded.shape[3], 3, "Channels should be 3 (RGB)");
        assert_eq!(decoded.dtype, "uint8");
    }

    #[test]
    fn test_decode_video_num_frames() {
        let video_bytes = load_test_video("240p_10.mp4");
        let (width, height, _total_frames) = parse_video_info("240p_10.mp4");

        let requested_frames = 5;
        let decoder = VideoDecoder {
            fps: None,
            max_frames: None,
            num_frames: Some(requested_frames),
            strict: false,
            max_pixels: None,
        };
        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);
        assert!(result.is_ok(), "Failed to decode video with num_frames");

        let decoded = result.unwrap();

        // Should decode exactly the requested number of frames
        assert_eq!(
            decoded.shape[0], requested_frames as usize,
            "Should decode exactly {} frames",
            requested_frames
        );
        assert_eq!(decoded.shape[1], height as usize);
        assert_eq!(decoded.shape[2], width as usize);
        assert_eq!(decoded.shape[3], 3);
        assert_eq!(decoded.dtype, "uint8");
    }

    #[test]
    fn test_decode_video_max_frames() {
        let video_bytes = load_test_video("240p_10.mp4");
        let (width, height, _total_frames) = parse_video_info("240p_10.mp4");

        let max_frames = 3;
        let decoder = VideoDecoder {
            fps: None,
            max_frames: Some(max_frames),
            num_frames: None,
            strict: false,
            max_pixels: None,
        };
        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);
        assert!(result.is_ok(), "Failed to decode video with max_frames");

        let decoded = result.unwrap();
        assert!(
            decoded.shape[0] <= max_frames as usize,
            "Should decode at most {} frames, got {}",
            max_frames,
            decoded.shape[0]
        );
        assert_eq!(decoded.shape[1], height as usize);
        assert_eq!(decoded.shape[2], width as usize);
        assert_eq!(decoded.shape[3], 3);
        assert_eq!(decoded.dtype, "uint8");
    }

    #[test]
    fn test_decode_video_fps_sampling() {
        let video_bytes = load_test_video("240p_100.mp4");
        let (width, height, _total_frames) = parse_video_info("240p_100.mp4");

        // Sample at lower fps from 100-frame video
        let target_fps = 0.5f32;
        let decoder = VideoDecoder {
            fps: Some(target_fps),
            max_frames: None,
            num_frames: None,
            strict: false,
            max_pixels: None,
        };

        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_ok(),
            "Failed to decode video with fps sampling {:?}",
            result.err()
        );

        let decoded = result.unwrap();

        // fps * duration calculation - video decoder uses duration from file
        // Source file is at 1fps, should get exactly 50 frames
        assert_eq!(
            decoded.shape[0], 50,
            "Should decode exactly 50 frames with fps sampling, got {}",
            decoded.shape[0]
        );
        assert_eq!(decoded.shape[1], height as usize);
        assert_eq!(decoded.shape[2], width as usize);
        assert_eq!(decoded.shape[3], 3);
        assert_eq!(decoded.dtype, "uint8");
    }

    // Pixel Limit Tests

    #[rstest]
    #[case(Some(320 * 240 * 5), "240p_10.mp4", 5, true, "within limit")]
    #[case(Some(320 * 240 * 2), "240p_10.mp4", 5, false, "exceeds limit")]
    #[case(Some(2 * 2 * 10), "2p_10.mp4", 10, true, "exactly at limit")]
    #[case(None, "2160p_10.mp4", 10, true, "no limit")]
    fn test_pixel_limits(
        #[case] max_pixels: Option<usize>,
        #[case] video_file: &str,
        #[case] num_frames: u32,
        #[case] should_succeed: bool,
        #[case] test_case: &str,
    ) {
        let video_bytes = load_test_video(video_file);
        let (width, height, _) = parse_video_info(video_file);

        let decoder = VideoDecoder {
            fps: None,
            max_frames: None,
            num_frames: Some(num_frames),
            strict: false,
            max_pixels,
        };

        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);

        if should_succeed {
            assert!(
                result.is_ok(),
                "Should decode successfully for case: {}",
                test_case
            );
            let decoded = result.unwrap();
            assert_eq!(decoded.shape[1], height as usize);
            assert_eq!(decoded.shape[2], width as usize);
            assert_eq!(
                decoded.dtype, "uint8",
                "dtype should be uint8 for case: {}",
                test_case
            );
        } else {
            assert!(result.is_err(), "Should fail for case: {}", test_case);
            let error_msg = result.unwrap_err().to_string();
            assert!(
                error_msg.contains("exceed max pixels"),
                "Error should mention exceeding max pixels for case: {}",
                test_case
            );
        }
    }

    #[test]
    fn test_conflicting_fps_and_num_frames() {
        let video_bytes = load_test_video("240p_10.mp4");

        let decoder = VideoDecoder {
            fps: Some(2.0),
            max_frames: None,
            num_frames: Some(5),
            strict: false,
            max_pixels: None,
        };

        let encoded_data = create_encoded_media_data(video_bytes);

        let result = decoder.decode(encoded_data);
        assert!(
            result.is_err(),
            "Should fail when both fps and num_frames are specified"
        );
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("cannot be specified at the same time"));
    }
}
