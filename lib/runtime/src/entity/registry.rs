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

use super::{Component, Registry, RegistryInner, Result};
use async_once_cell::OnceCell;
use std::{
    collections::HashMap,
    sync::{Arc, Weak},
};
use tokio::sync::Mutex;

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner::default())),
        }
    }
}

// impl ComponentRegistry {
//     pub fn new() -> Self {
//         Self {
//             clients: Arc::new(Mutex::new(HashMap::new())),
//         }
//     }

//     pub async fn get_or_create(&mut self, component: Component) -> Result<Arc<Client>> {
//         // Lock the clients HashMap for thread-safe access
//         let mut guard = self.clients.lock().await;

//         // Check if the component already exists in the registry
//         if let Some(weak) = guard.get(&component.slug()) {
//             // Attempt to upgrade the Weak pointer
//             if let Some(client) = weak.upgrade() {
//                 return Ok(client);
//             }
//         }

//         // Fallback: Create a new Client
//         let client = component.client().await?;

//         // Insert a Weak reference to the new client into the map
//         guard.insert(component.slug(), Arc::downgrade(&client));

//         Ok(client)
//     }
// }

// #[derive(Clone)]
// pub struct ServiceRegistry {
//     clients: Arc<Mutex<HashMap<String, Arc<Service>>>>,
// }

// impl ServiceRegistry {
//     pub fn new() -> Self {
//         Self {
//             clients: Arc::new(Mutex::new(HashMap::new())),
//         }
//     }

//     pub async fn get_or_create(&mut self, component: Component) -> Result<Arc<Client>> {
//         // Lock the clients HashMap for thread-safe access
//         let mut guard = self.clients.lock().await;

//         // Check if the component already exists in the registry
//         if let Some(weak) = guard.get(&component.slug()) {
//             // Attempt to upgrade the Weak pointer
//             if let Some(client) = weak.upgrade() {
//                 return Ok(client);
//             }
//         }

//         // Fallback: Create a new Client
//         let client = component.client().await?;

//         // Insert a Weak reference to the new client into the map
//         guard.insert(component.slug(), Arc::downgrade(&client));

//         Ok(client)
//     }
// }
