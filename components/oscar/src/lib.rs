// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//! Oscar distributed object sharing service for Dynamo runtime.
//!
//! Oscar allows dynamo runtime processes to register and share objects across
//! multiple leases using content-addressable storage with BLAKE3 hashing.

pub mod error;
pub mod hash;
pub mod keys;
pub mod object;
pub mod registry;
pub mod storage;
pub mod v2;
pub mod watcher;

pub use error::{OscarError, OscarResult};
pub use hash::{ContentHash, ObjectHasher};
pub use keys::{LeaseMetadata, LeaseReference, ObjectMetadata, OscarKeyType, OscarKeys};
pub use object::{ObjectInfo, ObjectState, SharedObject};
pub use registry::ObjectRegistry;
pub use storage::StorageBackend;
pub use watcher::ReferenceWatcher;

/// Maximum object size supported by Oscar (32 MiB).
pub const MAX_OBJECT_SIZE: usize = 32 * 1024 * 1024;

/// Etcd key prefix for Oscar objects.
pub const OSCAR_KEY_PREFIX: &str = "dynamo://_internal/oscar";