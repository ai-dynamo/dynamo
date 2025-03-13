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

use candle_hf_hub::api::tokio::ApiBuilder;
use std::path::{Path, PathBuf};

const IGNORED: [&str; 3] = [".gitattributes", "LICENSE", "README.md"];

/// Attempt to download a model from Hugging Face
/// Returns the directory it is in
pub async fn from_hf(name: &Path) -> anyhow::Result<PathBuf> {
    let api = ApiBuilder::new().with_progress(true).build()?;

    let repo = api.model(name.display().to_string());
    let info = repo.info().await?;
    let mut p = PathBuf::new();
    for sib in info.siblings {
        if IGNORED.contains(&sib.rfilename.as_str()) || is_image(&sib.rfilename) {
            continue;
        }
        p = repo.get(&sib.rfilename).await?;
    }
    match p.parent() {
        Some(p) => Ok(p.to_path_buf()),
        None => Err(anyhow::anyhow!("Invalid HF cache path: {}", p.display())),
    }
}

fn is_image(s: &str) -> bool {
    s.ends_with(".png") || s.ends_with("PNG") || s.ends_with(".jpg") || s.ends_with("JPG")
}
