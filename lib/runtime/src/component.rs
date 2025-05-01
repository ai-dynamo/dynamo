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

//! The [Component] module defines the top-level API for building distributed applications.
//!
//! A distributed application consists of a set of [Component] that can host one
//! or more [Endpoint]. Each [Endpoint] is a network-accessible service
//! that can be accessed by other [Component] in the distributed application.
//!
//! A [Component] is made discoverable by registering it with the distributed runtime under
//! a [`Namespace`].
//!
//! A [`Namespace`] is a logical grouping of [Component] that are grouped together.
//!
//! We might extend namespace to include grouping behavior, which would define groups of
//! components that are tightly coupled.
//!
//! A [Component] is the core building block of a distributed application. It is a logical
//! unit of work such as a `Preprocessor` or `SmartRouter` that has a well-defined role in the
//! distributed application.
//!
//! A [Component] can present to the distributed application one or more configuration files
//! which define how that component was constructed/configured and what capabilities it can
//! provide.
//!
//! Other [Component] can write to watching locations within a [Component] etcd
//! path. This allows the [Component] to take dynamic actions depending on the watch
//! triggers.
//!
//! TODO: Top-level Overview of Endpoints/Functions

use crate::{discovery::Lease, service::ServiceSet};

use super::{
    error, traits::*, transports::nats::Slug, utils::Duration, DistributedRuntime, Result, Runtime,
};

use crate::pipeline::network::{ingress::push_endpoint::PushEndpoint, PushWorkHandler};
use async_nats::{
    rustls::quic,
    service::{Service, ServiceExt},
};
use derive_builder::Builder;
use derive_getters::Getters;
use educe::Educe;
use serde::{Deserialize, Serialize};
use service::EndpointStatsHandler;
use std::{collections::HashMap, sync::Arc};
use validator::{Validate, ValidationError};

mod client;
#[allow(clippy::module_inception)]
mod component;
mod endpoint;
mod namespace;
mod registry;
pub mod service;

pub use client::{Client, EndpointSource};

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    NatsTcp(String),
}

#[derive(Default)]
pub struct RegistryInner {
    services: HashMap<String, Service>,
    stats_handlers: HashMap<String, Arc<std::sync::Mutex<HashMap<String, EndpointStatsHandler>>>>,
}

#[derive(Clone)]
pub struct Registry {
    inner: Arc<tokio::sync::Mutex<RegistryInner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentEndpointInfo {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub lease_id: i64,
    pub transport: TransportType,
}

impl ComponentEndpointInfo {
    pub fn id(&self) -> i64 {
        self.lease_id
    }
}

/// A [Component] a discoverable entity in the distributed runtime.
/// You can host [Endpoint] on a [Component] by first creating
/// a [Service] then adding one or more [Endpoint] to the [Service].
///
/// You can also issue a request to a [Component]'s [Endpoint] by creating a [Client].
#[derive(Educe, Builder, Clone)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Component {
    #[builder(private)]
    #[educe(Debug(ignore))]
    drt: DistributedRuntime,

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Name of the component
    #[builder(setter(into))]
    name: String,

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Namespace
    #[builder(setter(into))]
    namespace: Namespace,

    // A static component's endpoints cannot be discovered via etcd, they are
    // fixed at startup time.
    is_static: bool,
}

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.namespace.name(), self.name)
    }
}

impl DistributedRuntimeProvider for Component {
    fn drt(&self) -> &DistributedRuntime {
        &self.drt
    }
}

impl RuntimeProvider for Component {
    fn rt(&self) -> &Runtime {
        self.drt.rt()
    }
}

impl Component {
    pub fn etcd_path(&self) -> String {
        format!("{}/components/{}", self.namespace.name(), self.name)
    }

    pub fn service_name(&self) -> String {
        let service_name = format!("{}_{}", self.namespace.name(), self.name);
        Slug::slugify_unique(&service_name).to_string()
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.namespace.name(), self.name)
    }

    pub fn namespace(&self) -> &Namespace {
        &self.namespace
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn endpoint(&self, endpoint: impl Into<String>) -> Endpoint {
        Endpoint {
            component: self.clone(),
            name: endpoint.into(),
            is_static: self.is_static,
        }
    }

    /// Get keys from etcd on the slug, splitting the endpoints and only returning the
    /// set of unique endpoints.
    pub async fn list_endpoints(&self) -> Vec<Endpoint> {
        unimplemented!("endpoints")
    }

    pub async fn scrape_stats(&self, timeout: Duration) -> Result<ServiceSet> {
        let service_name = self.service_name();
        let service_client = self.drt().service_client();
        service_client
            .collect_services(&service_name, timeout)
            .await
    }

    /// TODO
    ///
    /// This method will scrape the stats for all available services
    /// Returns a stream of `ServiceInfo` objects.
    /// This should be consumed by a `[tokio::time::timeout_at`] because each services
    /// will only respond once, but there is no way to know when all services have responded.
    pub async fn stats_stream(&self) -> Result<()> {
        unimplemented!("collect_stats")
    }

    pub fn service_builder(&self) -> service::ServiceConfigBuilder {
        service::ServiceConfigBuilder::from_component(self.clone())
    }
}

impl ComponentBuilder {
    pub fn from_runtime(drt: DistributedRuntime) -> Self {
        Self::default().drt(drt)
    }
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    component: Component,

    // todo - restrict alphabet
    /// Endpoint name
    name: String,

    is_static: bool,
}

impl DistributedRuntimeProvider for Endpoint {
    fn drt(&self) -> &DistributedRuntime {
        self.component.drt()
    }
}

impl RuntimeProvider for Endpoint {
    fn rt(&self) -> &Runtime {
        self.component.rt()
    }
}

impl Endpoint {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn component(&self) -> &Component {
        &self.component
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.component.path(), self.name)
    }

    pub fn etcd_path(&self) -> String {
        format!("{}/{}", self.component.etcd_path(), self.name)
    }

    pub fn etcd_path_with_id(&self, lease_id: i64) -> String {
        if self.is_static {
            self.etcd_path()
        } else {
            format!("{}:{:x}", self.etcd_path(), lease_id)
        }
    }

    pub fn name_with_id(&self, lease_id: i64) -> String {
        if self.is_static {
            self.name.clone()
        } else {
            format!("{}-{:x}", self.name, lease_id)
        }
    }

    pub fn subject(&self) -> String {
        format!("{}.{}", self.component.service_name(), self.name)
    }

    /// Subject to an instance of the [Endpoint] with a specific lease id
    pub fn subject_to(&self, lease_id: i64) -> String {
        format!(
            "{}.{}",
            self.component.service_name(),
            self.name_with_id(lease_id)
        )
    }

    pub async fn client(&self) -> Result<client::Client> {
        if self.is_static {
            client::Client::new_static(self.clone()).await
        } else {
            client::Client::new_dynamic(self.clone()).await
        }
    }

    pub fn endpoint_builder(&self) -> endpoint::EndpointConfigBuilder {
        endpoint::EndpointConfigBuilder::from_endpoint(self.clone())
    }
}

#[derive(Educe, Builder, Clone, Validate)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Namespace {
    #[builder(private)]
    #[educe(Debug(ignore))]
    runtime: DistributedRuntime,

    #[validate()]
    name: String,

    is_static: bool,
}

impl DistributedRuntimeProvider for Namespace {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl RuntimeProvider for Namespace {
    fn rt(&self) -> &Runtime {
        self.runtime.rt()
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Namespace {
    pub(crate) fn new(runtime: DistributedRuntime, name: String, is_static: bool) -> Result<Self> {
        Ok(NamespaceBuilder::default()
            .runtime(runtime)
            .name(name)
            .is_static(is_static)
            .build()?)
    }

    /// Create a [`Component`] in the namespace who's endpoints can be discovered with etcd
    pub fn component(&self, name: impl Into<String>) -> Result<Component> {
        Ok(ComponentBuilder::from_runtime(self.runtime.clone())
            .name(name)
            .namespace(self.clone())
            .is_static(self.is_static)
            .build()?)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

// Custom validator function
fn validate_allowed_chars(input: &str) -> Result<(), ValidationError> {
    // Define the allowed character set using a regex
    let regex = regex::Regex::new(r"^[a-z0-9-_]+$").unwrap();

    if regex.is_match(input) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_characters"))
    }
}

// TODO - enable restrictions to the character sets allowed for namespaces,
// components, and endpoints.
//
// Put Validate traits on the struct and use the `validate_allowed_chars` method
// to validate the fields.

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use validator::Validate;

//     #[test]
//     fn test_valid_names() {
//         // Valid strings
//         let valid_inputs = vec![
//             "abc",        // Lowercase letters
//             "abc123",     // Letters and numbers
//             "a-b-c",      // Letters with hyphens
//             "a_b_c",      // Letters with underscores
//             "a-b_c-123",  // Mixed valid characters
//             "a",          // Single character
//             "a_b",        // Short valid pattern
//             "123456",     // Only numbers
//             "a---b_c123", // Repeated hyphens/underscores
//         ];

//         for input in valid_inputs {
//             let result = validate_allowed_chars(input);
//             assert!(result.is_ok(), "Expected '{}' to be valid", input);
//         }
//     }

//     #[test]
//     fn test_invalid_names() {
//         // Invalid strings
//         let invalid_inputs = vec![
//             "abc!",     // Invalid character `!`
//             "abc@",     // Invalid character `@`
//             "123$",     // Invalid character `$`
//             "foo.bar",  // Invalid character `.`
//             "foo/bar",  // Invalid character `/`
//             "foo\\bar", // Invalid character `\`
//             "abc#",     // Invalid character `#`
//             "abc def",  // Spaces are not allowed
//             "foo,",     // Invalid character `,`
//             "",         // Empty string
//         ];

//         for input in invalid_inputs {
//             let result = validate_allowed_chars(input);
//             assert!(result.is_err(), "Expected '{}' to be invalid", input);
//         }
//     }

//     // #[test]
//     // fn test_struct_validation_valid() {
//     //     // Struct with valid data
//     //     let valid_data = InputData {
//     //         name: "valid-name_123".to_string(),
//     //     };
//     //     assert!(valid_data.validate().is_ok());
//     // }

//     // #[test]
//     // fn test_struct_validation_invalid() {
//     //     // Struct with invalid data
//     //     let invalid_data = InputData {
//     //         name: "invalid!name".to_string(),
//     //     };
//     //     let result = invalid_data.validate();
//     //     assert!(result.is_err());

//     //     if let Err(errors) = result {
//     //         let error_map = errors.field_errors();
//     //         assert!(error_map.contains_key("name"));
//     //         let name_errors = &error_map["name"];
//     //         assert_eq!(name_errors[0].code, "invalid_characters");
//     //     }
//     // }

//     #[test]
//     fn test_edge_cases() {
//         // Edge cases
//         let edge_inputs = vec![
//             ("-", true),   // Single hyphen
//             ("_", true),   // Single underscore
//             ("a-", true),  // Letter with hyphen
//             ("-", false),  // Repeated hyphens
//             ("-a", false), // Hyphen at the beginning
//             ("a-", false), // Hyphen at the end
//         ];

//         for (input, expected_validity) in edge_inputs {
//             let result = validate_allowed_chars(input);
//             if expected_validity {
//                 assert!(result.is_ok(), "Expected '{}' to be valid", input);
//             } else {
//                 assert!(result.is_err(), "Expected '{}' to be invalid", input);
//             }
//         }
//     }
// }
