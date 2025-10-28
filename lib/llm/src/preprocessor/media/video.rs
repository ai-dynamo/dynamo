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
        let bytes = data.into_bytes()?;

        anyhow::ensure!(
            self.fps.is_none() || self.num_frames.is_none(),
            "fps and num_frames cannot be specified at the same time"
        );

        anyhow::ensure!(
            self.max_frames.is_none() || self.num_frames.is_none(),
            "max_frames and num_frames cannot be specified at the same time"
        );

        // video-rs wants a file path, we use tmpfs / ramdisk
        let mut temp_file = NamedTempFile::with_prefix("video")?;
        temp_file.write_all(&bytes)?;
        temp_file.flush()?;

        let location = Location::File(temp_file.path().to_path_buf());
        let mut decoder = video_rs::decode::Decoder::new(location)?;
        let total_frames = decoder.frames()? as u32;

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

        let step =
            ((total_frames - 1) as f32 / (requested_frames - 1).max(1) as f32).ceil() as usize;
        for i in (0..total_frames)
            .step_by(step)
            .take(requested_frames as usize)
        {
            if decoder.seek_to_frame(i as i64).is_err() {
                continue;
            }
            if let Ok((_ts, frame)) = decoder.decode() {
                all_frames.extend_from_slice(frame.as_slice().unwrap());
                num_frames_decoded += 1;
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
        Ok(array.into())
    }
}
