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

use std::time::Duration;

pub mod create;
pub mod model;
pub use model::ModelDeploymentCard;

// TODO: Do these network/publish related model deployment card values belong here or in a
// network module?

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "mdc";

/// Delete model deployment cards that haven't been re-published after this long.
/// Cleans up if the worker stopped.
pub const BUCKET_TTL: Duration = Duration::from_secs(5 * 60);
