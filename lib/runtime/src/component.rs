// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

use crate::{
    config::HealthStatus,
    discovery::Lease,
    metrics::{prometheus_names, MetricsRegistry},
    service::ServiceSet,
    transports::etcd::EtcdPath,
};

use super::{
    error,
    traits::*,
    transports::etcd::{COMPONENT_KEYWORD, ENDPOINT_KEYWORD},
    transports::nats::Slug,
    utils::Duration,
    DistributedRuntime, Result, Runtime,
};

use crate::pipeline::network::{ingress::push_endpoint::PushEndpoint, PushWorkHandler};
use crate::protocols::Endpoint as EndpointId;
use crate::service::ComponentNatsPrometheusMetrics;
use async_nats::{
    rustls::quic,
    service::{Service, ServiceExt},
};
use derive_builder::Builder;
use derive_getters::Getters;
use educe::Educe;
use serde::{Deserialize, Serialize};
use service::EndpointStatsHandler;
use std::{collections::HashMap, hash::Hash, sync::Arc};
use validator::{Validate, ValidationError};

mod client;
#[allow(clippy::module_inception)]
mod component;
mod endpoint;
mod namespace;
mod registry;
pub mod service;

pub use client::{Client, InstanceSource};

/// The root etcd path where each instance registers itself in etcd.
/// An instance is namespace+component+endpoint+lease_id and must be unique.
pub const INSTANCE_ROOT_PATH: &str = "instances";

/// The root etcd path where each namespace is registered in etcd.
pub const ETCD_ROOT_PATH: &str = "dynamo://";

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
pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: i64,
    pub transport: TransportType,
}

impl Instance {
    pub fn id(&self) -> i64 {
        self.instance_id
    }
}

/// A [Component] a discoverable entity in the distributed runtime.
/// You can host [Endpoint] on a [Component] by first creating
/// a [Service] then adding one or more [Endpoint] to the [Service].
///
/// You can also issue a request to a [Component]'s [Endpoint] by creating a [Client].
#[derive(Educe, Builder, Clone, Validate)]
#[educe(Debug)]
#[builder(pattern = "owned")]
pub struct Component {
    #[builder(private)]
    #[educe(Debug(ignore))]
    drt: Arc<DistributedRuntime>,

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Name of the component
    #[builder(setter(into))]
    #[validate(custom(function = "validate_allowed_chars"))]
    name: String,

    /// Additional labels for metrics
    #[builder(default = "Vec::new()")]
    labels: Vec<(String, String)>,

    // todo - restrict the namespace to a-z0-9-_A-Z
    /// Namespace
    #[builder(setter(into))]
    namespace: Namespace,

    // A static component's endpoints cannot be discovered via etcd, they are
    // fixed at startup time.
    is_static: bool,
}

impl Hash for Component {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.namespace.name().hash(state);
        self.name.hash(state);
        self.is_static.hash(state);
    }
}

impl PartialEq for Component {
    fn eq(&self, other: &Self) -> bool {
        self.namespace.name() == other.namespace.name()
            && self.name == other.name
            && self.is_static == other.is_static
    }
}

impl Eq for Component {}

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

impl MetricsRegistry for Component {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchy(&self) -> Vec<String> {
        [
            self.namespace.parent_hierarchy(),
            vec![self.namespace.basename()],
        ]
        .concat()
    }
}

impl Component {
    /// The component part of an instance path in etcd.
    pub fn etcd_root(&self) -> String {
        let ns = self.namespace.name();
        let cp = &self.name;
        format!("{INSTANCE_ROOT_PATH}/{ns}/{cp}")
    }

    pub fn service_name(&self) -> String {
        let service_name = format!("{}_{}", self.namespace.name(), self.name);
        Slug::slugify(&service_name).to_string()
    }

    pub fn path(&self) -> String {
        format!("{}/{}", self.namespace.name(), self.name)
    }

    pub fn etcd_path(&self) -> EtcdPath {
        EtcdPath::new_component(&self.namespace.name(), &self.name)
            .expect("Component name and namespace should be valid")
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
            labels: Vec::new(),
        }
    }

    pub async fn list_instances(&self) -> anyhow::Result<Vec<Instance>> {
        let Some(etcd_client) = self.drt.etcd_client() else {
            return Ok(vec![]);
        };
        let mut out = vec![];
        // The extra slash is important to only list exact component matches, not substrings.
        for kv in etcd_client
            .kv_get_prefix(format!("{}/", self.etcd_root()))
            .await?
        {
            let val = match serde_json::from_slice::<Instance>(kv.value()) {
                Ok(val) => val,
                Err(err) => {
                    anyhow::bail!(
                        "Error converting etcd response to Instance: {err}. {}",
                        kv.value_str()?
                    );
                }
            };
            out.push(val);
        }
        Ok(out)
    }

    /// Scrape ServiceSet, which contains NATS stats as well as user defined stats
    /// embedded in data field of ServiceInfo.
    pub async fn scrape_stats(&self, timeout: Duration) -> Result<ServiceSet> {
        let service_name = self.service_name();
        let service_client = self.drt().service_client();
        service_client
            .collect_services(&service_name, timeout)
            .await
    }

    /// Add Prometheus metrics for this component's service stats.
    ///
    /// Uses a channel to synchronize with the spawned async task, ensuring
    /// metrics are updated before the callback returns.
    pub fn add_metrics_callback(&self) -> Result<()> {
        let component_metrics = ComponentNatsPrometheusMetrics::new(self)?;

        let component_clone = self.clone();
        let mut hierarchies = self.parent_hierarchy();
        hierarchies.push(self.hierarchy());
        debug_assert_eq!(
            hierarchies.last().cloned().unwrap_or_default(),
            self.service_name()
        ); // it happens that in component, hierarchy and service name are the same

        // Register a metrics callback that scrapes component statistics
        let metrics_callback = Arc::new(move || {
            // Timeout for scraping metrics from components (in milliseconds)
            // This value is also used by KV Router metrics aggregator (300ms) and other components
            const METRICS_SCRAPE_TIMEOUT_MS: u64 = 300;

            // Get the current Tokio runtime handle
            let handle = tokio::runtime::Handle::try_current()
                .map_err(|err| anyhow::anyhow!("No Tokio runtime handle available: {}", err))?;

            let m = component_metrics.clone();
            let c = component_clone.clone();

            // Create a channel to synchronize with the spawned task
            let (tx, rx) = std::sync::mpsc::channel::<anyhow::Result<()>>();

            let timeout = std::time::Duration::from_millis(METRICS_SCRAPE_TIMEOUT_MS);
            handle.spawn(async move {
                let result = match c.scrape_stats(timeout).await {
                    Ok(service_set) => {
                        m.update_from_service_set(&service_set);
                        Ok(())
                    }
                    Err(err) => {
                        // Reset metrics on failure
                        m.reset_to_zeros();
                        Err(anyhow::anyhow!("Failed to scrape stats: {}", err))
                    }
                };

                // Send the result back to the waiting thread
                // If send fails, the receiver has already given up waiting
                let _ = tx.send(result);
            });

            // Wait for the spawned task to complete (with a timeout to prevent hanging)
            // Add 100ms buffer to the scrape timeout to account for processing overhead
            let recv_timeout = std::time::Duration::from_millis(METRICS_SCRAPE_TIMEOUT_MS + 100);
            match rx.recv_timeout(recv_timeout) {
                Ok(result) => result, // Return the actual result from scraping
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    component_metrics.reset_to_zeros();
                    Err(anyhow::anyhow!("Metrics collection timed out"))
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    component_metrics.reset_to_zeros();
                    Err(anyhow::anyhow!("Metrics collection task failed"))
                }
            }
        });

        self.drt()
            .register_metrics_callback(hierarchies, metrics_callback);

        Ok(())
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
    pub fn from_runtime(drt: Arc<DistributedRuntime>) -> Self {
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

    /// Additional labels for metrics
    labels: Vec<(String, String)>,
}

impl Hash for Endpoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.component.hash(state);
        self.name.hash(state);
        self.is_static.hash(state);
    }
}

impl PartialEq for Endpoint {
    fn eq(&self, other: &Self) -> bool {
        self.component == other.component
            && self.name == other.name
            && self.is_static == other.is_static
    }
}

impl Eq for Endpoint {}

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

impl MetricsRegistry for Endpoint {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchy(&self) -> Vec<String> {
        [
            self.component.parent_hierarchy(),
            vec![self.component.basename()],
        ]
        .concat()
    }
}

impl Endpoint {
    pub fn id(&self) -> EndpointId {
        EndpointId {
            namespace: self.component.namespace().name().to_string(),
            component: self.component.name().to_string(),
            name: self.name().to_string(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn component(&self) -> &Component {
        &self.component
    }

    // todo(ryan): deprecate this as we move to Discovery traits and Component Identifiers
    pub fn path(&self) -> String {
        format!(
            "{}/{}/{}",
            self.component.path(),
            ENDPOINT_KEYWORD,
            self.name
        )
    }

    /// The endpoint part of an instance path in etcd
    pub fn etcd_root(&self) -> String {
        let component_path = self.component.etcd_root();
        let endpoint_name = &self.name;
        format!("{component_path}/{endpoint_name}")
    }

    /// The endpoint as an EtcdPath object
    pub fn etcd_path(&self) -> EtcdPath {
        EtcdPath::new_endpoint(
            &self.component.namespace().name(),
            &self.component.name(),
            &self.name,
        )
        .expect("Endpoint name and component name should be valid")
    }

    /// The fully path of an instance in etcd
    pub fn etcd_path_with_lease_id(&self, lease_id: i64) -> String {
        let endpoint_root = self.etcd_root();
        if self.is_static {
            endpoint_root
        } else {
            format!("{endpoint_root}:{lease_id:x}")
        }
    }

    /// The endpoint as an EtcdPath object with lease ID
    pub fn etcd_path_object_with_lease_id(&self, lease_id: i64) -> EtcdPath {
        if self.is_static {
            self.etcd_path()
        } else {
            EtcdPath::new_endpoint_with_lease(
                &self.component.namespace().name(),
                &self.component.name(),
                &self.name,
                lease_id,
            )
            .expect("Endpoint name and component name should be valid")
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

#[derive(Builder, Clone, Validate)]
#[builder(pattern = "owned")]
pub struct Namespace {
    #[builder(private)]
    runtime: Arc<DistributedRuntime>,

    #[validate(custom(function = "validate_allowed_chars"))]
    name: String,

    is_static: bool,

    #[builder(default = "None")]
    parent: Option<Arc<Namespace>>,

    /// Additional labels for metrics
    #[builder(default = "Vec::new()")]
    labels: Vec<(String, String)>,
}

impl DistributedRuntimeProvider for Namespace {
    fn drt(&self) -> &DistributedRuntime {
        &self.runtime
    }
}

impl std::fmt::Debug for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Namespace {{ name: {}; is_static: {}; parent: {:?} }}",
            self.name, self.is_static, self.parent
        )
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
            .runtime(Arc::new(runtime))
            .name(name)
            .is_static(is_static)
            .build()?)
    }

    /// Create a [`Component`] in the namespace who's endpoints can be discovered with etcd
    pub fn component(&self, name: impl Into<String>) -> Result<Component> {
        let component = ComponentBuilder::from_runtime(self.runtime.clone())
            .name(name)
            .namespace(self.clone())
            .is_static(self.is_static)
            .build()?;

        // Register the metrics callback for this component.
        // If registration fails, log a warning but do not propagate the error,
        // as metrics are not mission critical and should not block component creation.
        if let Err(err) = component.add_metrics_callback() {
            tracing::warn!(
                "Failed to add metrics callback for component '{}': {}",
                component.service_name(),
                err
            );
        }

        Ok(component)
    }

    /// Create a [`Namespace`] in the parent namespace
    pub fn namespace(&self, name: impl Into<String>) -> Result<Namespace> {
        Ok(NamespaceBuilder::default()
            .runtime(self.runtime.clone())
            .name(name.into())
            .is_static(self.is_static)
            .parent(Some(Arc::new(self.clone())))
            .build()?)
    }

    pub fn etcd_path(&self) -> String {
        format!("{}{}", ETCD_ROOT_PATH, self.name())
    }

    pub fn name(&self) -> String {
        match &self.parent {
            Some(parent) => format!("{}.{}", parent.name(), self.name),
            None => self.name.clone(),
        }
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
