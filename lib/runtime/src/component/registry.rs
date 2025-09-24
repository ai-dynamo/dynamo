// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
