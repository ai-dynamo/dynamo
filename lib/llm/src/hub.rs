// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use hf_hub::api::tokio::ApiBuilder;
use std::path::{Path, PathBuf};

const IGNORED: [&str; 3] = [".gitattributes", "LICENSE", "README.md"];

/// Attempt to download a model from Hugging Face
/// Returns the directory it is in
pub async fn from_hf(name: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let api = ApiBuilder::new().with_progress(true).build()?;
    let model_name = name.display().to_string();

    let repo = api.model(model_name.clone());

    let info = match repo.info().await {
        Ok(info) => info,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to fetch model '{}' from HuggingFace: {}. Is this a valid HuggingFace ID?",
                model_name,
                e
            ));
        }
    };

    if info.siblings.is_empty() {
        return Err(anyhow::anyhow!(
            "Model '{}' exists but contains no downloadable files.",
            model_name
        ));
    }

    let mut p = PathBuf::new();
    let mut files_downloaded = false;

    for sib in info.siblings {
        if IGNORED.contains(&sib.rfilename.as_str()) || is_image(&sib.rfilename) {
            continue;
        }

        match repo.get(&sib.rfilename).await {
            Ok(path) => {
                p = path;
                files_downloaded = true;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to download file '{}' from model '{}': {}",
                    sib.rfilename,
                    model_name,
                    e
                ));
            }
        }
    }

    if !files_downloaded {
        return Err(anyhow::anyhow!(
            "No valid files found for model '{}'.",
            model_name
        ));
    }

    match p.parent() {
        Some(p) => Ok(p.to_path_buf()),
        None => Err(anyhow::anyhow!("Invalid HF cache path: {}", p.display())),
    }
}

fn is_image(s: &str) -> bool {
    s.ends_with(".png")
        || s.ends_with("PNG")
        || s.ends_with(".jpg")
        || s.ends_with("JPG")
        || s.ends_with(".jpeg")
        || s.ends_with("JPEG")
}
