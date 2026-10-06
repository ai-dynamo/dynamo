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

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../../../../deploy/inference-gateway/ext-proc/proto");
    tonic_build::configure()
        .include_file("envoy.rs")
        .compile_protos(
            &[proto.join("envoy/service/ext_proc/v3/external_processor.proto")],
            &[proto],
        )?;
    Ok(())
}
