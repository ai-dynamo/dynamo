// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::{env, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = env::var("CARGO_MANIFEST_DIR")?;
    let root = Path::new(&manifest)
        .ancestors()
        .nth(7)
        .expect("Dynamo repository root");
    let proto = root.join("deploy/inference-gateway/ext-proc/proto");
    let files = [
        "envoy/config/core/v3/base.proto",
        "envoy/type/v3/http_status.proto",
        "envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto",
        "envoy/service/ext_proc/v3/external_processor.proto",
    ]
    .map(|file| proto.join(file));
    tonic_build::configure().compile_protos(&files, &[proto])?;
    Ok(())
}
