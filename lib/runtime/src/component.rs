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

use std::fmt;

use crate::{
    config::HealthStatus,
    discovery::Lease,
    metrics::{MetricsRegistry, prometheus_names},
    service::ServiceSet,
    transports::etcd::{ETCD_ROOT_PATH, EtcdPath},
};

use super::{
    DistributedRuntime, Result, Runtime, error,
    traits::*,
    transports::etcd::{COMPONENT_KEYWORD, ENDPOINT_KEYWORD},
    transports::nats::Slug,
    utils::Duration,
};

use crate::pipeline::network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint};
use crate::protocols::EndpointId;
use crate::service::ComponentNatsServerPrometheusMetrics;
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

/// The root key-value path where each instance registers itself in.
/// An instance is namespace+component+endpoint+lease_id and must be unique.
pub const INSTANCE_ROOT_PATH: &str = "v1/instances";

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    NatsTcp(String),
}

#[derive(Default)]
pub struct RegistryInner {
    services: HashMap<String, Service>,
    stats_handlers: HashMap<String, Arc<parking_lot::Mutex<HashMap<String, EndpointStatsHandler>>>>,
}

#[derive(Clone)]
pub struct Registry {
    inner: Arc<tokio::sync::Mutex<RegistryInner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub fn endpoint_id(&self) -> EndpointId {
        EndpointId {
            namespace: self.namespace.clone(),
            component: self.component.clone(),
            name: self.endpoint.clone(),
        }
    }
}

impl fmt::Display for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}",
            self.namespace, self.component, self.endpoint, self.instance_id
        )
    }
}

/// Sort by string name
impl std::cmp::Ord for Instance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_string().cmp(&other.to_string())
    }
}

impl PartialOrd for Instance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Since Ord is fully implemented, the comparison is always total.
        Some(self.cmp(other))
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
    /// The component part of an instance path in key-value store.
    pub fn instance_root(&self) -> String {
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn labels(&self) -> &[(String, String)] {
        &self.labels
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
        let client = self.drt.store();
        let Some(bucket) = client.get_bucket(&self.instance_root()).await? else {
            return Ok(vec![]);
        };
        let entries = bucket.entries().await?;
        let mut instances = Vec::with_capacity(entries.len());
        for (name, bytes) in entries.into_iter() {
            let val = match serde_json::from_slice::<Instance>(&bytes) {
                Ok(val) => val,
                Err(err) => {
                    anyhow::bail!("Error converting storage response to Instance: {err}. {name}",);
                }
            };
            instances.push(val);
        }
        instances.sort();
        Ok(instances)
    }

    /// Scrape ServiceSet, which contains NATS stats as well as user defined stats
    /// embedded in data field of ServiceInfo.
    pub async fn scrape_stats(&self, timeout: Duration) -> Result<ServiceSet> {
        // Debug: scraping stats for component
        let service_name = self.service_name();
        let Some(service_client) = self.drt().service_client() else {
            anyhow::bail!("ServiceSet is gathered via NATS, do not call this in non-NATS setups.");
        };
        service_client
            .collect_services(&service_name, timeout)
            .await
    }

    /// Add Prometheus metrics for this component's NATS service stats.
    ///
    /// Starts a background task that periodically requests service statistics from NATS
    /// and updates the corresponding Prometheus metrics. The first scrape happens immediately,
    /// then subsequent scrapes occur at a fixed interval of 9.8 seconds (MAX_WAIT_MS),
    /// which should be near or smaller than typical Prometheus scraping intervals to ensure
    /// metrics are fresh when Prometheus collects them.
    pub fn start_scraping_nats_service_component_metrics(&self) -> Result<()> {
        const MAX_WAIT_MS: std::time::Duration = std::time::Duration::from_millis(9800); // Should be <= Prometheus scrape interval

        // If there is another component with the same service name, this will fail.
        let component_metrics = ComponentNatsServerPrometheusMetrics::new(self)?;

        let component_clone = self.clone();
        let mut hierarchies = self.parent_hierarchy();
        hierarchies.push(self.hierarchy());
        debug_assert!(
            hierarchies
                .last()
                .map(|x| x.as_str())
                .unwrap_or_default()
                .eq_ignore_ascii_case(&self.service_name())
        ); // it happens that in component, hierarchy and service name are the same

        // Start a background task that scrapes stats every 5 seconds
        let m = component_metrics.clone();
        let c = component_clone.clone();

        // Use the DRT's runtime handle to spawn the background task.
        // We cannot use regular `tokio::spawn` here because:
        // 1. This method may be called from contexts without an active Tokio runtime
        //    (e.g., tests that create a DRT in a blocking context)
        // 2. Tests often create a temporary runtime just to build the DRT, then drop it
        // 3. `tokio::spawn` requires being called from within a runtime context
        // By using the DRT's own runtime handle, we ensure the task runs in the
        // correct runtime that will persist for the lifetime of the component.
        c.drt().runtime().secondary().spawn(async move {
            let timeout = std::time::Duration::from_millis(500);
            let mut interval = tokio::time::interval(MAX_WAIT_MS);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                match c.scrape_stats(timeout).await {
                    Ok(service_set) => {
                        m.update_from_service_set(&service_set);
                    }
                    Err(err) => {
                        tracing::error!(
                            "Background scrape failed for {}: {}",
                            c.service_name(),
                            err
                        );
                        m.reset_to_zeros();
                    }
                }

                interval.tick().await;
            }
        });

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

    pub async fn add_stats_service(&mut self) -> anyhow::Result<()> {
        let service_name = self.service_name();

        // Pre-check to save cost of creating the service, but don't hold the lock
        if self
            .drt
            .component_registry
            .inner
            .lock()
            .await
            .services
            .contains_key(&service_name)
        {
            anyhow::bail!("Service {service_name} already exists");
        }

        let Some(nats_client) = self.drt.nats_client() else {
            anyhow::bail!("Cannot create NATS service without NATS.");
        };
        let description = None;
        let (nats_service, stats_reg) =
            service::build_nats_service(nats_client, self, description).await?;

        let mut guard = self.drt.component_registry.inner.lock().await;
        if !guard.services.contains_key(&service_name) {
            // Normal case
            guard.services.insert(service_name.clone(), nats_service);
            guard.stats_handlers.insert(service_name.clone(), stats_reg);
            drop(guard);
        } else {
            drop(guard);
            let _ = nats_service.stop().await;
            return Err(anyhow::anyhow!(
                "Service create race for {service_name}, now already exists"
            ));
        }

        // Register metrics callback. CRITICAL: Never fail service creation for metrics issues.
        if let Err(err) = self.start_scraping_nats_service_component_metrics() {
            tracing::debug!(service_name, error = %err, "Metrics registration failed");
        }
        Ok(())
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
        let component_path = self.component.instance_root();
        let endpoint_name = &self.name;
        format!("{component_path}/{endpoint_name}")
    }

    /// The endpoint as an EtcdPath object
    pub fn etcd_path(&self) -> EtcdPath {
        EtcdPath::new_endpoint(
            &self.component.namespace().name(),
            self.component.name(),
            &self.name,
        )
        .expect("Endpoint name and component name should be valid")
    }

    /// The fully path of an instance in etcd
    pub fn etcd_path_with_lease_id(&self, lease_id: i64) -> String {
        format!("{INSTANCE_ROOT_PATH}/{}", self.unique_path(lease_id))
    }

    /// Full path of this endpoint with forward slash separators, including lease id
    pub fn unique_path(&self, lease_id: i64) -> String {
        let ns = self.component.namespace().name();
        let cp = self.component.name();
        let ep = self.name();
        format!("{ns}/{cp}/{ep}/{lease_id:x}")
    }

    /// The endpoint as an EtcdPath object with lease ID
    pub fn etcd_path_object_with_lease_id(&self, lease_id: i64) -> EtcdPath {
        if self.is_static {
            self.etcd_path()
        } else {
            EtcdPath::new_endpoint_with_lease(
                &self.component.namespace().name(),
                self.component.name(),
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
        Ok(ComponentBuilder::from_runtime(self.runtime.clone())
            .name(name)
            .namespace(self.clone())
            .is_static(self.is_static)
            .build()?)
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
        format!("{ETCD_ROOT_PATH}{}", self.name())
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
