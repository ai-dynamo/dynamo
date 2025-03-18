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

use super::*;

// Python bindings for the Dynamo LLM system.
//
// This module exposes several sub-modules to interact with the Dynamo runtime:
// - `backend`: Interfaces for LLM inference and backend resource management.
// - `disagg_router`: Functionality for disaggregated routing of inference requests.
// - `kv`: Key-value caching and indexing utilities for managing model state.
// - `model_card`: Handling model deployment cards and related operations.
// - `preprocessor`: Tools for preprocessing LLM requests prior to execution.

pub mod backend;
pub mod disagg_router;
pub mod kv;
pub mod model_card;
pub mod preprocessor;
