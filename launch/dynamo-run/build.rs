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

use std::env;
use std::process::Command;

fn main() {
    if has_cuda_toolkit() && !has_feature("cuda") && is_cuda_engine() {
        println!("cargo:warning=CUDA not enabled, re-run with `--features cuda`");
    }
    if is_mac() && !has_feature("metal") {
        println!("cargo:warning=Metal not enabled, re-run with `--features metal`");
    }
}

fn has_feature(s: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", s.to_uppercase())).is_ok()
}

fn has_cuda_toolkit() -> bool {
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        output.status.success()
    } else {
        false
    }
}

fn is_cuda_engine() -> bool {
    has_feature("mistralrs") || has_feature("llamacpp")
}

#[cfg(target_os = "macos")]
fn is_mac() -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
fn is_mac() -> bool {
    false
}
