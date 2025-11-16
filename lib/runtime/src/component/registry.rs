// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use async_once_cell::OnceCell;
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};
use tokio::sync::Mutex;

use crate::component::{Registry, RegistryInner};

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner::default())),
            is_static: false,
        }
    }

    pub fn new_with_static(is_static: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner::default())),
            is_static,
        }
    }

    /// Check if this registry is for a static runtime
    pub fn is_static(&self) -> bool {
        self.is_static
    }
}
