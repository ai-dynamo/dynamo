use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{ModelMetadata, ModelProcessorSpec, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(super) struct Llama4Spec;

impl Llama4Spec {
    fn patch_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "patch_size"])
            .unwrap_or(14)
    }

    fn tile_size(metadata: &ModelMetadata) -> u32 {
        metadata
            .config_u32(&["vision_config", "image_size"])
            .filter(|v| *v > 0)
            .unwrap_or(336)
    }

    fn pixel_shuffle_ratio(metadata: &ModelMetadata) -> f64 {
        metadata
            .config
            .get("vision_config")
            .and_then(|v| v.get("pixel_shuffle_ratio"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5)
    }

    fn tokens_per_tile(metadata: &ModelMetadata) -> usize {
        let tile = Self::tile_size(metadata) as usize;
        let patch = Self::patch_size(metadata) as usize;
        if patch == 0 {
            return 0;
        }
        let patches = (tile / patch).pow(2);
        // Pixel shuffle reduces spatial dims by ratio, so token count by ratio^2
        let ratio = Self::pixel_shuffle_ratio(metadata);
        let downsample = (1.0 / (ratio * ratio)).round().max(1.0) as usize;
        patches / downsample
    }

    /// Extract per-image `(h_tiles, w_tiles)` from the preprocessor's
    /// `aspect_ratios` tensor.  Falls back to deriving tile counts from
    /// the original image sizes when aspect_ratios are unavailable.
    fn extract_aspect_ratios(
        preprocessed: &PreprocessedEncoderInputs,
        tile_size: usize,
    ) -> Vec<(usize, usize)> {
        if let Some(ModelSpecificValue::IntTensor { data, shape }) =
            preprocessed.model_specific.get("aspect_ratios")
        {
            if shape.len() == 2 && shape[1] == 2 && data.len() == shape[0] * 2 {
                return data
                    .chunks_exact(2)
                    .map(|chunk| (chunk[0] as usize, chunk[1] as usize))
                    .collect();
            }
        }
        // Fallback: derive from original image sizes (height, width).
        preprocessed
            .item_sizes
            .iter()
            .map(|&(h, w)| {
                let h_tiles = (h as usize).div_ceil(tile_size);
                let w_tiles = (w as usize).div_ceil(tile_size);
                (h_tiles, w_tiles)
            })
            .collect()
    }
}

impl ModelProcessorSpec for Llama4Spec {
    fn name(&self) -> &'static str {
        "llama4"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("llama-4")
            || id.contains("llama4")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "llama4")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id("<|image|>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 8)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let patch_token_id = self.placeholder_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        let tokens_per_tile = Self::tokens_per_tile(metadata);
        let tile_size = Self::tile_size(metadata) as usize;

        // Structural token IDs matching HF _prompt_split_image format.
        let image_start_id = metadata.token_id("<|image_start|>")?;
        let image_end_id = metadata.token_id("<|image_end|>")?;
        let image_id = metadata.token_id("<|image|>")?;
        let tile_x_sep_id = metadata.token_id("<|tile_x_separator|>")?;
        let tile_y_sep_id = metadata.token_id("<|tile_y_separator|>")?;

        // Extract aspect_ratios from preprocessor output (computed by
        // get_best_fit, respecting max_patches cap).  This is the Llama 4
        // analog of vLLM's out_mm_kwargs["image"][i]["aspect_ratios"].data.
        let aspect_ratios = Self::extract_aspect_ratios(preprocessed, tile_size);

        Ok(aspect_ratios
            .iter()
            .map(|&(h_tiles, w_tiles)| {
                let num_tiles = h_tiles * w_tiles;

                let mut tokens = Vec::new();

                // <|image_start|>
                tokens.push(image_start_id);

                // Grid tiles with separators (only for multi-tile images)
                if num_tiles > 1 {
                    for _row in 0..h_tiles {
                        for col in 0..w_tiles {
                            tokens.extend(std::iter::repeat_n(patch_token_id, tokens_per_tile));
                            if col < w_tiles - 1 {
                                tokens.push(tile_x_sep_id);
                            }
                        }
                        tokens.push(tile_y_sep_id);
                    }
                }

                // Global/cover tile: <|image|> + <|patch|> * tokens_per_tile
                tokens.push(image_id);
                tokens.extend(std::iter::repeat_n(patch_token_id, tokens_per_tile));

                // <|image_end|>
                tokens.push(image_end_id);

                PromptReplacement::sequence(Modality::Image, &placeholder, tokens)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is [total_tiles, C, H, W] — variable tiles per image.
        // aspect_ratios and patches_per_image are [num_images, ...].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("aspect_ratios".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn llama4_single_tile_token_count() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image|>", 200090),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
        ]);
        let config = json!({
            "model_type": "llama4",
            "image_token_index": 200092,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llama4 spec");
        assert_eq!(spec.name(), "llama4");

        // Single tile (336x336): <|image_start|> <|image|> <|patch|>*144 <|image_end|>
        // = 1 + 1 + 144 + 1 = 147 tokens
        let pp = test_preprocessed_with_aspects(&[ImageSize::new(336, 336)], &[(1, 1)]);
        let replacements = spec.prompt_replacements(&metadata, &pp).unwrap();
        assert_eq!(replacements[0].tokens.len(), 147);
        assert_eq!(replacements[0].tokens[0], 200088); // <|image_start|>
        assert_eq!(replacements[0].tokens[1], 200090); // <|image|>
        assert_eq!(replacements[0].tokens[2], 200092); // <|patch|> (first)
        assert_eq!(replacements[0].tokens[145], 200092); // <|patch|> (last)
        assert_eq!(replacements[0].tokens[146], 200089); // <|image_end|>
    }

    #[test]
    fn llama4_multi_tile_adds_global() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image|>", 200090),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
        ]);
        let config = json!({
            "model_type": "llama4",
            "image_token_index": 200092,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Llama-4-Scout-Vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llama4 spec");

        // 672x672 = 2x2 tiles + 1 global:
        //   <|image_start|>                                  = 1
        //   row0: <|patch|>*144 <|tile_x_sep|> <|patch|>*144 <|tile_y_sep|>  = 290
        //   row1: <|patch|>*144 <|tile_x_sep|> <|patch|>*144 <|tile_y_sep|>  = 290
        //   <|image|> <|patch|>*144                          = 145
        //   <|image_end|>                                    = 1
        //   Total = 1 + 290 + 290 + 145 + 1 = 727
        let pp = test_preprocessed_with_aspects(&[ImageSize::new(672, 672)], &[(2, 2)]);
        let replacements = spec.prompt_replacements(&metadata, &pp).unwrap();
        assert_eq!(replacements[0].tokens.len(), 727);
        // Verify structure: starts with image_start, ends with image_end
        assert_eq!(replacements[0].tokens[0], 200088); // <|image_start|>
        assert_eq!(*replacements[0].tokens.last().unwrap(), 200089); // <|image_end|>
                                                                     // The token before the last patch block is <|image|> (global tile marker)
                                                                     // Position: 1 + 290 + 290 = 581
        assert_eq!(replacements[0].tokens[581], 200090); // <|image|>
    }

    #[test]
    fn llama4_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[
            ("<|image|>", 200090),
            ("<|image_start|>", 200088),
            ("<|image_end|>", 200089),
            ("<|patch|>", 200092),
            ("<|tile_x_separator|>", 200093),
            ("<|tile_y_separator|>", 200094),
        ]);
        let config = json!({
            "model_type": "llama4",
            "image_token_index": 200092,
            "vision_config": {"image_size": 336, "patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llama4 alias");
        assert_eq!(spec.name(), "llama4");
    }
}
