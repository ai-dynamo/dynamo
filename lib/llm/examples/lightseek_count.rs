// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone harness for the lightseek-mm path: load a HF model dir, decode
// an image header to get (w, h), call lightseek's `calculate_num_tokens`, and
// print the count. Useful for cross-checking against the same model's vLLM
// output when investigating routing-cache mismatches.
//
//   cargo run -p dynamo-llm --example lightseek_count --features lightseek-mm \
//     -- <model_dir> <image_path> [model_id]

#[cfg(feature = "lightseek-mm")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use dynamo_llm::preprocessor::lightseek_mm::LightseekMmCounter;
    use std::path::PathBuf;

    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args
        .next()
        .context("usage: lightseek_count <model_dir> <image_path> [model_id]")?
        .into();
    let image_path: PathBuf = args
        .next()
        .context("usage: lightseek_count <model_dir> <image_path> [model_id]")?
        .into();
    let model_id = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen2.5-VL-3B-Instruct".to_string());

    // model_type from config.json so the registry lookup is robust.
    let cfg_path = model_dir.join("config.json");
    let cfg_json: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&cfg_path)
            .with_context(|| format!("opening {}", cfg_path.display()))?,
    )?;
    let model_type = cfg_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    println!(
        "model_dir = {}\nmodel_id  = {}\nmodel_type= {:?}",
        model_dir.display(),
        model_id,
        model_type
    );

    let counter = LightseekMmCounter::try_new(&model_id, model_type.as_deref(), &model_dir)?;
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

#[cfg(not(feature = "lightseek-mm"))]
fn main() {
    eprintln!("rebuild with --features lightseek-mm");
    std::process::exit(2);
}
