// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    distributed::RequestPlaneMode,
    metrics::{MetricsHierarchy, MetricsRegistry, prometheus_names},
    service::ServiceClient,
    service::ServiceSet,
};

use super::{DistributedRuntime, Runtime, traits::*, transports::nats::Slug, utils::Duration};

use crate::pipeline::network::{PushWorkHandler, ingress::push_endpoint::PushEndpoint};
use crate::protocols::EndpointId;
use async_nats::{
    rustls::quic,
    service::{Service, ServiceExt},
};
use derive_builder::Builder;
use derive_getters::Getters;
use educe::Educe;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, hash::Hash, sync::Arc};
use validator::{Validate, ValidationError};

mod client;
#[allow(clippy::module_inception)]
mod component;
mod endpoint;
mod namespace;
mod registry;
pub mod service;

pub use client::Client;
pub use endpoint::build_transport_type;

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TransportType {
    #[serde(rename = "nats_tcp")]
    Nats(String),
    Http(String),
    Tcp(String),
}

#[derive(Default)]
pub struct RegistryInner {
    pub(crate) services: HashMap<String, Service>,
}

#[derive(Clone)]
pub struct Registry {
    pub(crate) inner: Arc<tokio::sync::Mutex<RegistryInner>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Instance {
    pub component: String,
    pub endpoint: String,
    pub namespace: String,
    pub instance_id: u64,
    pub transport: TransportType,
}

impl Instance {
    pub fn id(&self) -> u64 {
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
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct Component {
    #[builder(private)]
    #[educe(Debug(ignore))]
    drt: Arc<DistributedRuntime>,

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

    /// This hierarchy's own metrics registry
    #[builder(default = "crate::MetricsRegistry::new()")]
    metrics_registry: crate::MetricsRegistry,
}

impl Hash for Component {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.namespace.name().hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Component {
    fn eq(&self, other: &Self) -> bool {
        self.namespace.name() == other.namespace.name() && self.name == other.name
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

impl MetricsHierarchy for Component {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        let mut parents = vec![];

        // Get all ancestors of namespace (DRT, parent namespaces, etc.)
        parents.extend(self.namespace.parent_hierarchies());

        // Add namespace itself
        parents.push(&self.namespace as &dyn MetricsHierarchy);

        parents
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
    }
}

impl Component {
    pub fn service_name(&self) -> String {
        let service_name = format!("{}_{}", self.namespace.name(), self.name);
        Slug::slugify(&service_name).to_string()
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
            labels: Vec::new(),
            metrics_registry: crate::MetricsRegistry::new(),
        }
    }

    pub async fn list_instances(&self) -> anyhow::Result<Vec<Instance>> {
        let discovery = self.drt.discovery();

        let discovery_query = crate::discovery::DiscoveryQuery::ComponentEndpoints {
            namespace: self.namespace.name(),
            component: self.name.clone(),
        };

        let discovery_instances = discovery.list(discovery_query).await?;

        // Extract Instance from DiscoveryInstance::Endpoint wrapper
        let mut instances: Vec<Instance> = discovery_instances
            .into_iter()
            .filter_map(|di| match di {
                crate::discovery::DiscoveryInstance::Endpoint(instance) => Some(instance),
                _ => None, // Ignore all other variants (ModelCard, etc.)
            })
            .collect();

        instances.sort();
        Ok(instances)
    }
}

impl ComponentBuilder {
    pub fn from_runtime(drt: Arc<DistributedRuntime>) -> Self {
        Self::default().drt(drt)
    }

    pub fn build(self) -> Result<Component, anyhow::Error> {
        let component = self.build_internal()?;
        // If this component is using NATS, register the NATS service and wait for completion.
        // This prevents a race condition where serve_endpoint() tries to look up the service
        // before it's registered in the component registry.
        let drt = component.drt();
        if drt.request_plane().is_nats() {
            let mut rx = drt.register_nats_service(component.clone());
            // Wait synchronously for the NATS service registration to complete.
            // Uses block_in_place() to safely call blocking_recv() from async contexts.
            // This temporarily moves the current task off the runtime thread to allow
            // blocking without deadlocking the runtime.
            let result = tokio::task::block_in_place(|| rx.blocking_recv());
            match result {
                Some(Ok(())) => {
                    tracing::debug!(
                        component = component.service_name(),
                        "NATS service registration completed"
                    );
                }
                Some(Err(e)) => {
                    return Err(anyhow::anyhow!(
                        "NATS service registration failed for component '{}': {}",
                        component.service_name(),
                        e
                    ));
                }
                None => {
                    return Err(anyhow::anyhow!(
                        "NATS service registration channel closed unexpectedly for component '{}'",
                        component.service_name()
                    ));
                }
            }
        }
        Ok(component)
    }
}

#[derive(Debug, Clone)]
pub struct Endpoint {
    component: Component,

    // todo - restrict alphabet
    /// Endpoint name
    name: String,

    /// Additional labels for metrics
    labels: Vec<(String, String)>,

    /// This hierarchy's own metrics registry
    metrics_registry: crate::MetricsRegistry,
}

impl Hash for Endpoint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.component.hash(state);
        self.name.hash(state);
    }
}

impl PartialEq for Endpoint {
    fn eq(&self, other: &Self) -> bool {
        self.component == other.component && self.name == other.name
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

impl MetricsHierarchy for Endpoint {
    fn basename(&self) -> String {
        self.name.clone()
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        let mut parents = vec![];

        // Get all ancestors of component (DRT, Namespace, etc.)
        parents.extend(self.component.parent_hierarchies());

        // Add component itself
        parents.push(&self.component as &dyn MetricsHierarchy);

        parents
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
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

    pub async fn client(&self) -> anyhow::Result<client::Client> {
        client::Client::new(self.clone()).await
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

    #[builder(default = "None")]
    parent: Option<Arc<Namespace>>,

    /// Additional labels for metrics
    #[builder(default = "Vec::new()")]
    labels: Vec<(String, String)>,

    /// This hierarchy's own metrics registry
    #[builder(default = "crate::MetricsRegistry::new()")]
    metrics_registry: crate::MetricsRegistry,
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
            "Namespace {{ name: {}; parent: {:?} }}",
            self.name, self.parent
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
    pub(crate) fn new(runtime: DistributedRuntime, name: String) -> anyhow::Result<Self> {
        Ok(NamespaceBuilder::default()
            .runtime(Arc::new(runtime))
            .name(name)
            .build()?)
    }

    /// Create a [`Component`] in the namespace who's endpoints can be discovered with etcd
    pub fn component(&self, name: impl Into<String>) -> anyhow::Result<Component> {
        ComponentBuilder::from_runtime(self.runtime.clone())
            .name(name)
            .namespace(self.clone())
            .build()
    }

    /// Create a [`Namespace`] in the parent namespace
    pub fn namespace(&self, name: impl Into<String>) -> anyhow::Result<Namespace> {
        Ok(NamespaceBuilder::default()
            .runtime(self.runtime.clone())
            .name(name.into())
            .parent(Some(Arc::new(self.clone())))
            .build()?)
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
