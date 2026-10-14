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

use clap::Parser;
use dynamo_speech_nim_sidecar::{config::Args, worker};

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let _ = rustls::crypto::ring::default_provider().install_default();
    dynamo_sidecar_common::run_task(|runtime, shutdown| worker::run(args, runtime, shutdown))
}
