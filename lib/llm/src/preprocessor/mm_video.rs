// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure-Rust per-video token counting for MM-aware KV routing. Compiled only
//! when the `mm-routing` cargo feature is enabled.
//!
//! Unlike the image path (`lightseek_mm`, backed by the external
//! `llm-multimodal` crate), video counting is implemented in-tree: the
//! upstream crate's `calculate_num_tokens(width, height)` has no temporal
//! parameter (verified through 1.8.0). This module reproduces the HF
//! processor math for the Qwen-VL video families from
//! `preprocessor_config.json` / `video_preprocessor_config.json` directly.
//!
//! The token count depends on the *sampled* frame count `T`, which only the
//! frontend-decoding path knows exactly (the frontend samples frames and
//! ships the decoded tensor to the backend). URL-passthrough video is
//! therefore not routable — callers skip building MM routing info for it.

use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Qwen-VL video families with distinct expansion structure or resize
/// budgets. Qwen2-VL and Qwen2.5-VL share per-frame resize + a single
/// `<|video_pad|>` placeholder; Qwen3-VL budgets the full `T*H*W` volume and
/// interleaves per-frame-group timestamp text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFamily {
    Qwen2Vl,
    Qwen25Vl,
    Qwen3Vl,
}

impl VideoFamily {
    /// Prefer the `model_type` config field; fall back to HF-id/path
    /// substrings, mirroring the image registry's matching convention.
    fn detect(model_id: &str, model_type: Option<&str>) -> Option<Self> {
        match model_type {
            Some("qwen2_vl") => return Some(Self::Qwen2Vl),
            Some("qwen2_5_vl") => return Some(Self::Qwen25Vl),
            Some("qwen3_vl") | Some("qwen3_vl_moe") => return Some(Self::Qwen3Vl),
            _ => {}
        }
        let id = model_id.to_ascii_lowercase().replace('_', "-");
        if id.contains("qwen3-vl") {
            Some(Self::Qwen3Vl)
        } else if id.contains("qwen2.5-vl") || id.contains("qwen2-5-vl") {
            Some(Self::Qwen25Vl)
        } else if id.contains("qwen2-vl") {
            Some(Self::Qwen2Vl)
        } else {
            None
        }
    }

    /// Whether the chat template's `<|vision_start|><|video_pad|><|vision_end|>`
    /// triple is replaced wholesale with per-frame-group segments (Qwen3-VL),
    /// as opposed to expanding the single `<|video_pad|>` token in place.
    pub fn expands_placeholder_triple(&self) -> bool {
        matches!(self, Self::Qwen3Vl)
    }
}

/// Maps `(num_frames, width, height) → num_video_tokens` for a single model
/// using the model's HF video-processing parameters.
pub struct VideoTokenCounter {
    family: VideoFamily,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    min_pixels: usize,
    max_pixels: usize,
    model_id: String,
}

impl VideoTokenCounter {
    /// Returns `Err` when the model family is not a supported Qwen-VL video
    /// family or no processor config is readable. Callers should treat the
    /// error as "video MM-aware routing disabled for this model" rather than
    /// failing the request. Sync filesystem I/O by design — called once per
    /// model at preprocessor construction, like `LightseekMmCounter::try_new`.
    pub fn try_new(model_id: &str, model_type: Option<&str>, model_dir: &Path) -> Result<Self> {
        let family = VideoFamily::detect(model_id, model_type).ok_or_else(|| {
            anyhow!(
                "mm-routing: no video token counter for model_id={:?} model_type={:?}",
                model_id,
                model_type
            )
        })?;

        // Newer Transformers splits video params into
        // `video_preprocessor_config.json`; older releases keep them in
        // `preprocessor_config.json`. Prefer the video-specific file.
        let json = ["video_preprocessor_config.json", "preprocessor_config.json"]
            .iter()
            .find_map(|name| std::fs::read_to_string(model_dir.join(name)).ok())
            .with_context(|| {
                format!(
                    "mm-routing: no readable (video_)preprocessor_config.json under {}",
                    model_dir.display()
                )
            })?;
        let config: serde_json::Value = serde_json::from_str(&json)
            .with_context(|| "mm-routing: failed to parse video processor config")?;

        Self::from_config(family, &config, model_id)
    }

    /// Build from a pre-parsed processor config. Field fallbacks follow the
    /// HF schema across Transformers versions: `patch_size` may be a scalar
    /// or `{height, width}`; the pixel budget is `min_pixels`/`max_pixels`
    /// or `size.shortest_edge`/`size.longest_edge`.
    fn from_config(
        family: VideoFamily,
        config: &serde_json::Value,
        model_id: &str,
    ) -> Result<Self> {
        let scalar = |v: &serde_json::Value| -> Option<usize> {
            v.as_u64().map(|x| x as usize).or_else(|| {
                v.get("height")
                    .or_else(|| v.get("width"))
                    .and_then(|d| d.as_u64())
                    .map(|x| x as usize)
            })
        };
        let field = |name: &str| config.get(name).and_then(&scalar);
        let size_field = |name: &str| {
            config
                .get("size")
                .and_then(|s| s.get(name))
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
        };

        // Per-family HF defaults, used only when the config omits a field.
        let (d_patch, d_min, d_max) = match family {
            // Qwen2VLImageProcessor defaults (also processes Qwen2/2.5-VL video).
            VideoFamily::Qwen2Vl | VideoFamily::Qwen25Vl => (14, 56 * 56, 28 * 28 * 1280),
            // Qwen3VLVideoProcessor defaults; max budgets the T*H*W volume.
            VideoFamily::Qwen3Vl => (16, 65_536, 16_777_216),
        };

        let counter = Self {
            family,
            patch_size: field("patch_size").unwrap_or(d_patch),
            merge_size: field("merge_size").unwrap_or(2),
            temporal_patch_size: field("temporal_patch_size").unwrap_or(2),
            min_pixels: field("min_pixels")
                .or_else(|| size_field("shortest_edge"))
                .unwrap_or(d_min),
            max_pixels: field("max_pixels")
                .or_else(|| size_field("longest_edge"))
                .unwrap_or(d_max),
            model_id: model_id.to_string(),
        };
        if counter.patch_size == 0
            || counter.merge_size == 0
            || counter.temporal_patch_size == 0
            || counter.min_pixels == 0
            || counter.min_pixels > counter.max_pixels
        {
            return Err(anyhow!(
                "mm-routing: invalid video processor config for {model_id}"
            ));
        }
        Ok(counter)
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn family(&self) -> VideoFamily {
        self.family
    }

    /// Dimension alignment factor: dims must be divisible by
    /// `patch_size * merge_size`.
    fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// HF `smart_resize`: round dims to the nearest factor multiple
    /// (Python `round()` is banker's rounding, hence `round_ties_even`),
    /// then rescale into the pixel budget. `volume_frames` is the padded
    /// temporal extent the budget applies to: 1 for per-frame budgets
    /// (Qwen2/2.5-VL), `t_bar` for the Qwen3-VL video volume budget.
    /// HF quirk preserved: the over-budget threshold uses padded frames
    /// while the rescale beta uses the actual pixel volume.
    fn smart_resize(
        &self,
        height: usize,
        width: usize,
        volume_frames: usize,
        beta_frames: usize,
    ) -> Result<(usize, usize)> {
        let factor = self.factor() as f64;
        if height == 0 || width == 0 {
            return Err(anyhow!("mm-routing: zero video frame dimension"));
        }
        let (max_d, min_d) = (height.max(width) as f64, height.min(width) as f64);
        if max_d / min_d > 200.0 {
            return Err(anyhow!("mm-routing: video aspect ratio exceeds 200:1"));
        }

        let mut h_bar =
            ((height as f64 / factor).round_ties_even() as usize).max(1) * self.factor();
        let mut w_bar = ((width as f64 / factor).round_ties_even() as usize).max(1) * self.factor();
        h_bar = h_bar.max(self.factor());
        w_bar = w_bar.max(self.factor());

        let resized = volume_frames as f64 * (h_bar * w_bar) as f64;
        let actual = (beta_frames * height * width) as f64;
        if resized > self.max_pixels as f64 {
            let beta = (actual / self.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor).floor() as usize) * self.factor();
            w_bar = ((width as f64 / beta / factor).floor() as usize) * self.factor();
            h_bar = h_bar.max(self.factor());
            w_bar = w_bar.max(self.factor());
        } else if resized < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / actual).sqrt();
            h_bar = ((height as f64 * beta / factor).ceil() as usize) * self.factor();
            w_bar = ((width as f64 * beta / factor).ceil() as usize) * self.factor();
        }
        Ok((h_bar, w_bar))
    }

    /// `(grid_t, grid_h, grid_w)` for a sampled clip. `grid_t` reflects the
    /// HF frame padding to a multiple of `temporal_patch_size` (the last
    /// frame is repeated), i.e. `ceil(T / temporal_patch_size)`.
    fn grid_thw(
        &self,
        num_frames: usize,
        height: u32,
        width: u32,
    ) -> Result<(usize, usize, usize)> {
        if num_frames == 0 {
            return Err(anyhow!("mm-routing: zero sampled video frames"));
        }
        let grid_t = num_frames.div_ceil(self.temporal_patch_size);
        let (h_bar, w_bar) = match self.family {
            VideoFamily::Qwen2Vl | VideoFamily::Qwen25Vl => {
                // Per-frame pixel budget, applied to each frame identically.
                self.smart_resize(height as usize, width as usize, 1, 1)?
            }
            VideoFamily::Qwen3Vl => {
                // Volume budget over the padded clip.
                let t_bar = grid_t * self.temporal_patch_size;
                self.smart_resize(height as usize, width as usize, t_bar, num_frames)?
            }
        };
        Ok((grid_t, h_bar / self.patch_size, w_bar / self.patch_size))
    }

    /// Total `<|video_pad|>` tokens the backend HF processor emits for this
    /// clip: `grid_t * grid_h * grid_w / merge_size^2`.
    pub fn count_tokens(&self, num_frames: usize, height: u32, width: u32) -> Result<usize> {
        let (t, h, w) = self.grid_thw(num_frames, height, width)?;
        Ok(t * h * w / (self.merge_size * self.merge_size))
    }

    /// Qwen3-VL: `<|video_pad|>` tokens per temporal grid group
    /// (`grid_h * grid_w / merge_size^2`).
    pub fn tokens_per_frame_group(
        &self,
        num_frames: usize,
        height: u32,
        width: u32,
    ) -> Result<usize> {
        let (_, h, w) = self.grid_thw(num_frames, height, width)?;
        Ok(h * w / (self.merge_size * self.merge_size))
    }

    /// Qwen3-VL: one timestamp per temporal grid group, the midpoint of the
    /// group's first and last sampled-frame timestamps. Matches the HF video
    /// processor's grouping of `sampled_timestamps` by `temporal_patch_size`.
    pub fn group_timestamps(&self, sampled_timestamps: &[f64]) -> Vec<f64> {
        sampled_timestamps
            .chunks(self.temporal_patch_size)
            .map(|chunk| {
                let first = chunk[0];
                let last = chunk.last().copied().unwrap_or(first);
                (first + last) / 2.0
            })
            .collect()
    }

    /// Qwen3-VL timestamp text inserted before each frame group, e.g.
    /// `<3.5 seconds>`. One decimal place, matching the HF chat processor's
    /// `f"<{timestamp:.1f} seconds>"`.
    pub fn timestamp_label(timestamp: f64) -> String {
        format!("<{timestamp:.1} seconds>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3(config: &str) -> VideoTokenCounter {
        VideoTokenCounter::from_config(
            VideoFamily::Qwen3Vl,
            &serde_json::from_str(config).unwrap(),
            "Qwen/Qwen3-VL-2B-Instruct",
        )
        .unwrap()
    }

    fn qwen2(config: &str) -> VideoTokenCounter {
        VideoTokenCounter::from_config(
            VideoFamily::Qwen2Vl,
            &serde_json::from_str(config).unwrap(),
            "Qwen/Qwen2-VL-7B-Instruct",
        )
        .unwrap()
    }

    #[test]
    fn family_detection_prefers_model_type() {
        assert_eq!(
            VideoFamily::detect("/models/finetune", Some("qwen2_5_vl")),
            Some(VideoFamily::Qwen25Vl)
        );
        assert_eq!(
            VideoFamily::detect("Qwen/Qwen3-VL-2B-Instruct", None),
            Some(VideoFamily::Qwen3Vl)
        );
        assert_eq!(
            VideoFamily::detect("/models/Qwen2.5-VL-7B/", None),
            Some(VideoFamily::Qwen25Vl)
        );
        assert_eq!(VideoFamily::detect("llava-1.5", None), None);
    }

    #[test]
    fn parses_min_max_pixels_and_size_fallback() {
        let a = qwen2(
            r#"{"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
                          "min_pixels": 3136, "max_pixels": 1003520}"#,
        );
        assert_eq!(
            (a.patch_size, a.min_pixels, a.max_pixels),
            (14, 3136, 1003520)
        );

        let b = qwen3(
            r#"{"patch_size": 16, "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
        );
        assert_eq!((b.min_pixels, b.max_pixels), (65_536, 16_777_216));
    }

    /// Qwen2-VL per-frame budget: a 224x224 clip resizes to itself
    /// (factor 28 aligned, inside [3136, 1003520]); 8 frames with
    /// temporal_patch_size 2 give grid_t=4; tokens = 4*16*16/4.
    #[test]
    fn qwen2_vl_video_token_count() {
        let c = qwen2(
            r#"{"patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
                          "min_pixels": 3136, "max_pixels": 1003520}"#,
        );
        assert_eq!(c.count_tokens(8, 224, 224).unwrap(), 256);
        // Odd frame count pads up: T=5 -> grid_t=3.
        assert_eq!(c.count_tokens(5, 224, 224).unwrap(), 3 * 64);
    }

    /// Qwen3-VL volume budget: 640x360, factor 32 ->
    /// h_bar = round_ties_even(360/32)=11 -> 352, w_bar = 20 -> 640.
    /// T=16 volume 16*352*640 within budget; grid_t=8, grid_h=352/16=22,
    /// grid_w=640/16=40 -> 8*22*40/4 = 1760 tokens, 220 per frame group.
    #[test]
    fn qwen3_vl_video_token_count() {
        let c = qwen3(
            r#"{"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                          "min_pixels": 65536, "max_pixels": 16777216}"#,
        );
        assert_eq!(c.count_tokens(16, 360, 640).unwrap(), 1760);
        assert_eq!(c.tokens_per_frame_group(16, 360, 640).unwrap(), 220);
    }

    /// Over-budget Qwen3-VL clip scales down: threshold uses padded frames,
    /// beta uses actual pixels (HF quirk).
    #[test]
    fn qwen3_vl_volume_downscale() {
        let c = qwen3(
            r#"{"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                          "min_pixels": 65536, "max_pixels": 16777216}"#,
        );
        // 64 frames of 1080x1920: volume 64*1088*1920 = 133.7M > 16.7M budget.
        let n = c.count_tokens(64, 1080, 1920).unwrap();
        let (t, h, w) = c.grid_thw(64, 1080, 1920).unwrap();
        assert_eq!(t, 32);
        // Post-rescale volume must fit the budget.
        assert!(t * 2 * (h * 16) * (w * 16) <= 16_777_216);
        assert_eq!(n, t * h * w / 4);
    }

    #[test]
    fn group_timestamps_midpoint_and_tail() {
        let c = qwen3(r#"{"temporal_patch_size": 2}"#);
        assert_eq!(
            c.group_timestamps(&[0.0, 0.5, 1.0, 1.5, 2.0]),
            vec![0.25, 1.25, 2.0]
        );
    }

    #[test]
    fn timestamp_label_one_decimal() {
        assert_eq!(VideoTokenCounter::timestamp_label(1.25), "<1.2 seconds>");
        assert_eq!(VideoTokenCounter::timestamp_label(3.0), "<3.0 seconds>");
    }

    /// Parity against real HF processor outputs (counts AND the full
    /// expanded token stream). Fixtures are produced by running the pinned
    /// Transformers processors over synthetic clips; regenerate them when
    /// bumping the supported Transformers range. Run with:
    /// `DYN_MM_VIDEO_FIXTURE=a.json;b.json cargo test -p dynamo-llm \
    ///    --no-default-features --features mm-routing hf_fixture_parity \
    ///    -- --ignored --nocapture`
    #[test]
    #[ignore = "requires HF-generated fixture files via DYN_MM_VIDEO_FIXTURE"]
    fn hf_fixture_parity() {
        let paths = std::env::var("DYN_MM_VIDEO_FIXTURE")
            .expect("set DYN_MM_VIDEO_FIXTURE to ';'-separated fixture json paths");
        let u32s = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };
        for path in paths.split(';').filter(|p| !p.is_empty()) {
            let fx: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let model_id = fx["model_id"].as_str().unwrap();
            let dir = std::path::PathBuf::from(fx["local_dir"].as_str().unwrap());
            let counter = VideoTokenCounter::try_new(model_id, None, &dir).unwrap();
            let video_tok = fx["video_token_id"].as_u64().unwrap() as u32;
            let pre_ids = u32s(&fx["pre_expansion_input_ids"]);
            let tokenizer = crate::tokenizers::Tokenizer::from_file(
                dir.join("tokenizer.json").to_str().unwrap(),
            )
            .unwrap();
            for case in fx["cases"].as_array().unwrap() {
                let t = case["num_frames"].as_u64().unwrap() as usize;
                let h = case["height"].as_u64().unwrap() as u32;
                let w = case["width"].as_u64().unwrap() as u32;
                let label = format!("{model_id} T={t} {h}x{w}");
                let expect_n = case["n_video_tokens"].as_u64().unwrap() as usize;
                let grid = u32s(&case["video_grid_thw"]);
                let expect_ids = u32s(&case["expanded_input_ids"]);

                assert_eq!(
                    counter.count_tokens(t, h, w).unwrap(),
                    expect_n,
                    "count {label}"
                );
                let per_group = counter.tokens_per_frame_group(t, h, w).unwrap();
                assert_eq!(per_group, expect_n / grid[0] as usize, "per-group {label}");

                // Rebuild the expansion exactly as the frontend does, filling
                // video positions with video_tok instead of pad_value, so the
                // result must be bit-identical to HF's expanded input_ids.
                let rebuilt = if counter.family().expands_placeholder_triple() {
                    let vs = fx["vision_start_token_id"].as_u64().unwrap() as u32;
                    let ve = fx["vision_end_token_id"].as_u64().unwrap() as u32;
                    let ts: Vec<f64> = case["sampled_timestamps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    let mut repl = Vec::new();
                    for g in counter.group_timestamps(&ts) {
                        let enc = tokenizer
                            .encode(&VideoTokenCounter::timestamp_label(g))
                            .unwrap();
                        repl.extend_from_slice(enc.token_ids());
                        repl.push(vs);
                        repl.extend(std::iter::repeat_n(video_tok, per_group));
                        repl.push(ve);
                    }
                    let mut out = Vec::with_capacity(pre_ids.len() + repl.len());
                    let mut i = 0usize;
                    while i < pre_ids.len() {
                        if i + 2 < pre_ids.len()
                            && pre_ids[i] == vs
                            && pre_ids[i + 1] == video_tok
                            && pre_ids[i + 2] == ve
                        {
                            out.extend_from_slice(&repl);
                            i += 3;
                        } else {
                            out.push(pre_ids[i]);
                            i += 1;
                        }
                    }
                    out
                } else {
                    let mut out = Vec::with_capacity(pre_ids.len() + expect_n);
                    for &tok in &pre_ids {
                        if tok == video_tok {
                            out.extend(std::iter::repeat_n(video_tok, expect_n));
                        } else {
                            out.push(tok);
                        }
                    }
                    out
                };
                assert_eq!(rebuilt, expect_ids, "expanded ids {label}");
            }
            println!(
                "fixture parity ok: {model_id} ({} cases)",
                fx["cases"].as_array().unwrap().len()
            );
        }
    }

    /// End-to-end parity against a REAL vLLM engine's KV events: the routing
    /// stream rebuilt here (pre-expansion prompt ids + VideoTokenCounter +
    /// pad_value fill) must block-hash identically to the engine's
    /// BlockStored events after the production ZMQ normalization
    /// (`decode_event_batch` + `ZmqEventNormalizer` with the video token id).
    /// Capture with `e2e_capture.py`; run with:
    /// `DYN_MM_VIDEO_E2E_CAPTURE=e2e_capture.json cargo test -p dynamo-llm \
    ///    --no-default-features --features mm-routing --lib \
    ///    vllm_e2e_event_parity -- --ignored --nocapture`
    #[test]
    #[ignore = "requires a vLLM KV-event capture via DYN_MM_VIDEO_E2E_CAPTURE"]
    fn vllm_e2e_event_parity() {
        use base64::Engine as _;
        use dynamo_kv_router::protocols::{
            BlockHashOptions, KvCacheEventData, WorkerWithDpRank, compute_block_hash_for_seq,
            pad_value_for_mm_hash,
        };
        use dynamo_kv_router::zmq_wire::{ZmqEventNormalizer, decode_event_batch};

        let path = std::env::var("DYN_MM_VIDEO_E2E_CAPTURE")
            .expect("set DYN_MM_VIDEO_E2E_CAPTURE to the capture json path");
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let u32s = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect()
        };

        let dir = std::path::PathBuf::from(fx["local_dir"].as_str().unwrap());
        let model_id = fx["model_id"].as_str().unwrap();
        let block_size = fx["kv_block_size"].as_u64().unwrap() as usize;
        let mm_hash = fx["mm_hash_u64"].as_u64().unwrap();
        let video_tok = fx["video_token_id"].as_u64().unwrap() as u32;
        let image_tok = fx["image_token_id"].as_u64().map(|x| x as u32);
        let t = fx["num_frames"].as_u64().unwrap() as usize;
        let h = fx["height"].as_u64().unwrap() as u32;
        let w = fx["width"].as_u64().unwrap() as u32;

        // ---- Frontend side: independent rebuild of the routing stream ----
        let counter = VideoTokenCounter::try_new(model_id, None, &dir).unwrap();
        let n = counter.count_tokens(t, h, w).unwrap();
        let pre_ids = u32s(&fx["pre_expansion_input_ids"]);
        assert_eq!(
            pre_ids.iter().filter(|&&x| x == video_tok).count(),
            1,
            "expected exactly one video placeholder in the chat prompt"
        );
        let pad = pad_value_for_mm_hash(mm_hash);
        let mut routing: Vec<u32> = Vec::with_capacity(pre_ids.len() + n);
        for &tok in &pre_ids {
            if tok == video_tok {
                routing.extend(std::iter::repeat_n(pad, n));
            } else {
                routing.push(tok);
            }
        }

        // Cross-check the expansion against the engine's actual prompt: same
        // length, pad exactly where the engine has video_token_id.
        let engine_prompt = u32s(&fx["engine_prompt_token_ids"]);
        assert_eq!(routing.len(), engine_prompt.len(), "expanded prompt length");
        for (i, (&r, &e)) in routing.iter().zip(engine_prompt.iter()).enumerate() {
            if e == video_tok {
                assert_eq!(r, pad, "expected pad at video position {i}");
            } else {
                assert_eq!(r, e, "token mismatch at {i}");
            }
        }

        let frontend_hashes =
            compute_block_hash_for_seq(&routing, block_size as u32, BlockHashOptions::default());

        // ---- Event side: production decode + normalization ----
        let mut normalizer = ZmqEventNormalizer::new(block_size as u32)
            .with_image_token_id(image_tok)
            .with_video_token_id(Some(video_tok));
        let worker = WorkerWithDpRank::from_worker_id(0);
        let mut event_hashes = Vec::new();
        let mut normalized_mm_blocks = 0usize;
        for (i, payload_b64) in fx["raw_event_payloads_b64"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let payload = base64::engine::general_purpose::STANDARD
                .decode(payload_b64.as_str().unwrap())
                .unwrap();
            let batch = decode_event_batch(&payload).unwrap();
            for raw in batch.events {
                let Some(placement) = normalizer.normalize(raw, i as u64, worker) else {
                    continue;
                };
                if let KvCacheEventData::Stored(store) = placement.event.data {
                    for b in store.blocks {
                        if b.mm_extra_info.is_some() {
                            normalized_mm_blocks += 1;
                        }
                        event_hashes.push(b.tokens_hash);
                    }
                }
            }
        }

        // The engine also stores warm-up and generated-token blocks, so the
        // video prompt's whole blocks must appear as a contiguous run inside
        // the stored-hash sequence, in order.
        let prompt_blocks = routing.len() / block_size;
        assert!(prompt_blocks > 0, "prompt shorter than one block");
        assert!(
            event_hashes.len() >= prompt_blocks,
            "engine stored {} blocks, need at least {prompt_blocks}",
            event_hashes.len()
        );
        assert!(
            normalized_mm_blocks > 0,
            "no event block carried mm extra keys — uuid forwarding broken?"
        );
        let expected = &frontend_hashes[..prompt_blocks];
        let found = event_hashes
            .windows(prompt_blocks)
            .any(|run| run == expected);
        assert!(
            found,
            "frontend block hashes not found as a run in {} stored event hashes \
             (first frontend hash {:?}, event hashes {:?})",
            event_hashes.len(),
            expected.first(),
            &event_hashes[..event_hashes.len().min(8)]
        );
        println!(
            "e2e parity ok: {model_id} — {prompt_blocks} prompt blocks match \
             ({} video tokens, {normalized_mm_blocks} mm blocks normalized)",
            n
        );
    }
}
