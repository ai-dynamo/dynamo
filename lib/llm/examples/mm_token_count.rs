// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone harness for the MM-routing per-image token-count path: load a
// HF model dir, decode an image header to get (w, h), call the image
// processor's versioned semantic contract, and print the count. Useful for
// cross-checking against the same model's worker output when investigating
// routing-cache mismatches or exact usage.
//
//   cargo run -p dynamo-llm --example mm_token_count --features mm-routing \
//     -- <model_dir> <image_path> <spec> [model_id]

#[cfg(feature = "mm-routing")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use dynamo_llm::local_model::runtime_config::ImageTokenizationSpec;
    use dynamo_llm::preprocessor::lightseek_mm::ExactImageTokenCounter;
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args
        .next()
        .context("usage: mm_token_count <model_dir> <image_path> <spec> [model_id]")?
        .into();
    let image_path: PathBuf = args
        .next()
        .context("usage: mm_token_count <model_dir> <image_path> <spec> [model_id]")?
        .into();
    let spec: ImageTokenizationSpec = args
        .next()
        .context("missing spec: qwen2_vl_v1, qwen3_vl_v1, or moonvit_v1")?
        .parse()
        .map_err(anyhow::Error::msg)?;
    let model_id = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen2.5-VL-3B-Instruct".to_string());

    println!(
        "model_dir = {}\nmodel_id  = {}\nspec      = {}",
        model_dir.display(),
        model_id,
        spec.as_str()
    );

    let counter = ExactImageTokenCounter::try_new(&model_id, spec, &model_dir)?;
    println!("counter for '{}' constructed", counter.model_id());

    let img_bytes = std::fs::read(&image_path)?;
    let (w, h) = image::ImageReader::new(std::io::Cursor::new(&img_bytes))
        .with_guessed_format()?
        .into_dimensions()?;
    println!("image     = {} ({}x{})", image_path.display(), w, h);

    let n = counter.count_tokens(w, h);
    println!("tokens    = {}", n);

    Ok(())
}

#[cfg(not(feature = "mm-routing"))]
fn main() {
    eprintln!("rebuild with --features mm-routing");
    std::process::exit(2);
}
