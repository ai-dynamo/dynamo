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

//! Example-local speech adapters. Backend RPCs and Dynamo serving contracts
//! are separate modules so the crate can later move to `lib/sidecar`.

pub mod asr;
pub mod client;
pub mod config;
mod events;
pub mod tts;
pub mod worker;

/// Bindings generated from the pinned upstream Riva protocol.
pub mod proto {
    tonic::include_proto!("nvidia.riva");
    pub mod asr {
        tonic::include_proto!("nvidia.riva.asr");
    }
    pub mod tts {
        tonic::include_proto!("nvidia.riva.tts");
    }
}
