// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::BTreeMap, time::Instant};

use dynamo_runtime::component::Instance;

use crate::{input::InputEvent, metrics::MetricsSnapshot};

/// Events emitted by background tasks to mutate the UI state.
#[derive(Debug)]
pub enum AppEvent {
    /// Replace the entire discovery snapshot with the provided instances.
    DiscoverySnapshot(Vec<Instance>),
    /// A new instance has registered.
    InstanceUp(Instance),
    /// An existing instance has been deregistered.
    InstanceDown {
        namespace: String,
        component: String,
        endpoint: String,
        instance_id: u64,
    },
    /// Updated statistics from NATS client.
    Nats(NatsSnapshot),
    /// Updated metrics scraped from Prometheus endpoint.
    Metrics(MetricsSnapshot),
    /// Informational status message.
    Info(String),
    /// Error message.
    Error(String),
}

/// Snapshot of summary statistics from the NATS client.
#[derive(Debug, Clone)]
pub struct NatsSnapshot {
    pub in_bytes: u64,
    pub out_bytes: u64,
    pub in_messages: u64,
    pub out_messages: u64,
    pub connects: u64,
    pub connection_state: NatsConnectionState,
    pub last_updated: Instant,
}

impl Default for NatsSnapshot {
    fn default() -> Self {
        Self {
            in_bytes: 0,
            out_bytes: 0,
            in_messages: 0,
            out_messages: 0,
            connects: 0,
            connection_state: NatsConnectionState::Disconnected,
            last_updated: Instant::now(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatsConnectionState {
    Connected,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusColumn {
    Namespaces,
    Components,
    Endpoints,
}

impl Default for FocusColumn {
    fn default() -> Self {
        FocusColumn::Namespaces
    }
}

#[derive(Debug, Default)]
pub struct App {
    namespaces: BTreeMap<String, NamespaceState>,
    selection: Selection,
    last_error: Option<String>,
    last_info: Option<String>,
    last_update: Option<Instant>,
    nats: Option<NatsSnapshot>,
    metrics: Option<MetricsSnapshot>,
}

#[derive(Debug, Default, Clone)]
pub struct Selection {
    pub focus: FocusColumn,
    pub namespace_index: usize,
    pub component_index: usize,
    pub endpoint_index: usize,
}

#[derive(Debug, Clone)]
pub struct NamespaceState {
    pub components: BTreeMap<String, ComponentState>,
}

#[derive(Debug, Clone)]
pub struct ComponentState {
    pub endpoints: BTreeMap<String, EndpointState>,
}

#[derive(Debug, Clone)]
pub struct EndpointState {
    pub instances: BTreeMap<u64, InstanceState>,
    pub status: EndpointStatus,
    pub updated_at: Instant,
}

#[derive(Debug, Clone)]
pub struct InstanceState {
    pub instance: Instance,
    pub last_seen: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointStatus {
    Unknown,
    Provisioning,
    Ready,
    Offline,
}

impl Default for EndpointStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

impl App {
    pub fn apply_event(&mut self, event: AppEvent) {
        self.last_update = Some(Instant::now());

        match event {
            AppEvent::DiscoverySnapshot(instances) => {
                self.rebuild_tree(instances);
                self.selection.namespace_index = 0;
                self.selection.component_index = 0;
                self.selection.endpoint_index = 0;
            }
            AppEvent::InstanceUp(instance) => {
                self.insert_instance(instance);
            }
            AppEvent::InstanceDown {
                namespace,
                component,
                endpoint,
                instance_id,
            } => {
                self.remove_instance(&namespace, &component, &endpoint, instance_id);
            }
            AppEvent::Nats(snapshot) => {
                self.nats = Some(snapshot);
            }
            AppEvent::Metrics(snapshot) => {
                self.metrics = Some(snapshot);
            }
            AppEvent::Info(message) => {
                self.last_info = Some(message);
                self.last_error = None;
            }
            AppEvent::Error(message) => {
                self.last_error = Some(message);
            }
        }

        self.clamp_selection();
    }

    pub fn handle_input(&mut self, input: InputEvent) -> bool {
        match input {
            InputEvent::Quit => return true,
            InputEvent::MoveUp => self.move_selection(-1),
            InputEvent::MoveDown => self.move_selection(1),
            InputEvent::MoveLeft => self.move_focus(-1),
            InputEvent::MoveRight => self.move_focus(1),
            InputEvent::Refresh => {
                self.last_info = Some("Refreshing...".to_string());
            }
        }
        false
    }

    pub fn namespaces_iter(&self) -> impl Iterator<Item = (&String, &NamespaceState)> {
        self.namespaces.iter()
    }

    pub fn namespace_at(&self, index: usize) -> Option<(&String, &NamespaceState)> {
        self.namespaces_iter().nth(index)
    }

    pub fn component_at(
        &self,
        namespace_index: usize,
        component_index: usize,
    ) -> Option<(&String, &ComponentState)> {
        let (_, namespace) = self.namespace_at(namespace_index)?;
        namespace.components.iter().nth(component_index)
    }

    pub fn endpoint_at(
        &self,
        namespace_index: usize,
        component_index: usize,
        endpoint_index: usize,
    ) -> Option<(&String, &EndpointState)> {
        let (_, component) = self.component_at(namespace_index, component_index)?;
        component.endpoints.iter().nth(endpoint_index)
    }

    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    pub fn last_error(&self) -> Option<&String> {
        self.last_error.as_ref()
    }

    pub fn last_info(&self) -> Option<&String> {
        self.last_info.as_ref()
    }

    pub fn last_update(&self) -> Option<Instant> {
        self.last_update
    }

    pub fn nats_snapshot(&self) -> Option<&NatsSnapshot> {
        self.nats.as_ref()
    }

    pub fn metrics_snapshot(&self) -> Option<&MetricsSnapshot> {
        self.metrics.as_ref()
    }

    pub fn namespace_count(&self) -> usize {
        self.namespaces.len()
    }

    pub fn component_count(&self) -> usize {
        if let Some((_, namespace)) = self.namespace_at(self.selection.namespace_index) {
            return namespace.components.len();
        }
        0
    }

    pub fn endpoint_count(&self) -> usize {
        if let Some((_, component)) = self.component_at(
            self.selection.namespace_index,
            self.selection.component_index,
        ) {
            return component.endpoints.len();
        }
        0
    }

    fn rebuild_tree(&mut self, instances: Vec<Instance>) {
        self.namespaces.clear();
        for instance in instances {
            self.insert_instance(instance);
        }
    }

    fn insert_instance(&mut self, instance: Instance) {
        let namespace = self
            .namespaces
            .entry(instance.namespace.clone())
            .or_insert_with(|| NamespaceState {
                components: BTreeMap::new(),
            });

        let component = namespace
            .components
            .entry(instance.component.clone())
            .or_insert_with(|| ComponentState {
                endpoints: BTreeMap::new(),
            });

        let endpoint = component
            .endpoints
            .entry(instance.endpoint.clone())
            .or_insert_with(|| EndpointState {
                instances: BTreeMap::new(),
                status: EndpointStatus::Provisioning,
                updated_at: Instant::now(),
            });

        endpoint.updated_at = Instant::now();
        endpoint
            .instances
            .insert(
                instance.instance_id,
                InstanceState {
                    instance,
                    last_seen: Instant::now(),
                },
            )
            .map(|_| ());

        endpoint.status = EndpointStatus::Ready;
    }

    fn remove_instance(
        &mut self,
        namespace_name: &str,
        component_name: &str,
        endpoint_name: &str,
        instance_id: u64,
    ) {
        if let Some(namespace) = self.namespaces.get_mut(namespace_name) {
            if let Some(component) = namespace.components.get_mut(component_name) {
                if let Some(endpoint) = component.endpoints.get_mut(endpoint_name) {
                    endpoint.instances.remove(&instance_id);
                    endpoint.updated_at = Instant::now();
                    if endpoint.instances.is_empty() {
                        endpoint.status = EndpointStatus::Offline;
                    }
                }
            }
        }
    }

    fn move_selection(&mut self, delta: isize) {
        match self.selection.focus {
            FocusColumn::Namespaces => {
                let count = self.namespace_count();
                Self::shift_index(&mut self.selection.namespace_index, count, delta);
                self.selection.component_index = 0;
                self.selection.endpoint_index = 0;
            }
            FocusColumn::Components => {
                let count = self.component_count();
                Self::shift_index(&mut self.selection.component_index, count, delta);
                self.selection.endpoint_index = 0;
            }
            FocusColumn::Endpoints => {
                let count = self.endpoint_count();
                Self::shift_index(&mut self.selection.endpoint_index, count, delta);
            }
        }
    }

    fn move_focus(&mut self, delta: isize) {
        use FocusColumn::*;
        self.selection.focus = match (self.selection.focus, delta) {
            (Namespaces, d) if d > 0 && self.component_count() > 0 => Components,
            (Components, d) if d > 0 && self.endpoint_count() > 0 => Endpoints,
            (Components, d) if d < 0 => Namespaces,
            (Endpoints, d) if d < 0 => Components,
            (focus, _) => focus,
        };
    }

    fn shift_index(index: &mut usize, len: usize, delta: isize) {
        if len == 0 {
            *index = 0;
            return;
        }

        let idx = *index as isize + delta;
        if idx < 0 {
            *index = len - 1;
        } else {
            *index = (idx as usize) % len;
        }
    }

    fn clamp_selection(&mut self) {
        if self.namespace_count() == 0 {
            self.selection.namespace_index = 0;
            self.selection.component_index = 0;
            self.selection.endpoint_index = 0;
            return;
        }

        if self.selection.namespace_index >= self.namespace_count() {
            self.selection.namespace_index = self.namespace_count() - 1;
        }

        let component_count = self.component_count();
        if component_count == 0 {
            self.selection.component_index = 0;
            self.selection.endpoint_index = 0;
            if matches!(
                self.selection.focus,
                FocusColumn::Components | FocusColumn::Endpoints
            ) {
                self.selection.focus = FocusColumn::Namespaces;
            }
        } else if self.selection.component_index >= component_count {
            self.selection.component_index = component_count - 1;
        }

        let endpoint_count = self.endpoint_count();
        if endpoint_count == 0 {
            self.selection.endpoint_index = 0;
            if matches!(self.selection.focus, FocusColumn::Endpoints) {
                self.selection.focus = FocusColumn::Components;
            }
        } else if self.selection.endpoint_index >= endpoint_count {
            self.selection.endpoint_index = endpoint_count - 1;
        }
    }

    pub fn endpoint_health_summary(&self, endpoint: &EndpointState) -> EndpointHealthSummary {
        let instance_count = endpoint.instances.len();
        let last_seen = endpoint.instances.values().map(|inst| inst.last_seen).max();

        EndpointHealthSummary {
            status: endpoint.status,
            instances: instance_count,
            last_seen,
        }
    }

    pub fn component_health_summary(&self, component: &ComponentState) -> ComponentHealthSummary {
        let mut status = EndpointStatus::Offline;
        let mut total_instances = 0;

        for endpoint in component.endpoints.values() {
            total_instances += endpoint.instances.len();
            status = aggregate_status(status, endpoint.status);
        }

        ComponentHealthSummary {
            status,
            active_endpoints: component
                .endpoints
                .values()
                .filter(|ep| !ep.instances.is_empty())
                .count(),
            total_endpoints: component.endpoints.len(),
            instances: total_instances,
        }
    }
}

pub fn aggregate_status(current: EndpointStatus, next: EndpointStatus) -> EndpointStatus {
    use EndpointStatus::*;
    match (current, next) {
        (Ready, Ready) => Ready,
        (Ready, _) | (_, Ready) => Ready,
        (Provisioning, _) | (_, Provisioning) => Provisioning,
        (Unknown, status) => status,
        (status, Unknown) => status,
        (Offline, Offline) => Offline,
    }
}

#[derive(Debug, Clone)]
pub struct EndpointHealthSummary {
    pub status: EndpointStatus,
    pub instances: usize,
    pub last_seen: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct ComponentHealthSummary {
    pub status: EndpointStatus,
    pub active_endpoints: usize,
    pub total_endpoints: usize,
    pub instances: usize,
}

impl EndpointStatus {
    pub fn display_name(self) -> &'static str {
        match self {
            EndpointStatus::Unknown => "Unknown",
            EndpointStatus::Provisioning => "Provisioning",
            EndpointStatus::Ready => "Ready",
            EndpointStatus::Offline => "Offline",
        }
    }

    pub fn as_color(self) -> ratatui::style::Color {
        use ratatui::style::Color;
        match self {
            EndpointStatus::Unknown => Color::Gray,
            EndpointStatus::Provisioning => Color::Yellow,
            EndpointStatus::Ready => Color::Green,
            EndpointStatus::Offline => Color::Red,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::TransportType;

    fn mk_instance(namespace: &str, component: &str, endpoint: &str, instance_id: u64) -> Instance {
        Instance {
            namespace: namespace.to_string(),
            component: component.to_string(),
            endpoint: endpoint.to_string(),
            instance_id,
            transport: TransportType::NatsTcp(format!("{component}.{endpoint}.{instance_id}")),
        }
    }

    #[test]
    fn discovery_snapshot_populates_tree() {
        let mut app = App::default();
        let instances = vec![
            mk_instance("ns-a", "frontend", "http", 1),
            mk_instance("ns-a", "frontend", "http", 2),
            mk_instance("ns-a", "router", "grpc", 3),
            mk_instance("ns-b", "worker", "engine", 4),
        ];
        app.apply_event(AppEvent::DiscoverySnapshot(instances.clone()));

        assert_eq!(app.namespace_count(), 2);
        let (_, component) = app.component_at(0, 0).expect("component exists");
        assert_eq!(component.endpoints.len(), 1);

        let summary = app.component_health_summary(component);
        assert_eq!(summary.instances, 2);

        let (_, endpoint) = app.endpoint_at(0, 0, 0).expect("endpoint exists");
        assert_eq!(endpoint.instances.len(), 2);
        assert_eq!(endpoint.status, EndpointStatus::Ready);
    }

    #[test]
    fn instance_down_marks_endpoint_offline() {
        let mut app = App::default();
        let instance = mk_instance("ns", "comp", "ep", 1);
        app.apply_event(AppEvent::DiscoverySnapshot(vec![instance.clone()]));

        app.apply_event(AppEvent::InstanceDown {
            namespace: instance.namespace.clone(),
            component: instance.component.clone(),
            endpoint: instance.endpoint.clone(),
            instance_id: instance.instance_id,
        });

        let (_, endpoint) = app
            .endpoint_at(0, 0, 0)
            .expect("endpoint should remain tracked");
        assert_eq!(endpoint.status, EndpointStatus::Offline);
        assert!(endpoint.instances.is_empty());
    }
}
