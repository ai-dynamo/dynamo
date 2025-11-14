// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod model_manager;
pub use model_manager::{ModelManager, ModelManagerError};

mod watcher;
pub use watcher::{ModelUpdate, ModelWatcher};

mod worker_monitor;
pub use worker_monitor::{KvWorkerMonitor, WorkerLoadState};

/// The root etcd path for KV Router registrations
pub const KV_ROUTERS_ROOT_PATH: &str = "v1/kv-routers";
