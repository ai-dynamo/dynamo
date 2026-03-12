// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Application state and event loop.
//!
//! The `App` struct holds all UI state and handles events from data sources
//! and keyboard input. The event loop uses `tokio::select!` to multiplex
//! across all event channels.

use crate::input::{Action, map_key};
use crate::model::{Namespace, NatsStats, PrometheusMetrics};
use crate::sources::AppEvent;

/// Which pane currently has keyboard focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneFocus {
    Namespaces,
    Components,
    Endpoints,
}

impl PaneFocus {
    pub fn next(self) -> Self {
        match self {
            PaneFocus::Namespaces => PaneFocus::Components,
            PaneFocus::Components => PaneFocus::Endpoints,
            PaneFocus::Endpoints => PaneFocus::Namespaces,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            PaneFocus::Namespaces => PaneFocus::Endpoints,
            PaneFocus::Components => PaneFocus::Namespaces,
            PaneFocus::Endpoints => PaneFocus::Components,
        }
    }
}

/// Application state.
pub struct App {
    /// Discovered namespaces (hierarchical tree).
    pub namespaces: Vec<Namespace>,

    /// NATS connection and message statistics.
    pub nats_stats: NatsStats,

    /// Prometheus metrics (if configured).
    pub metrics: PrometheusMetrics,

    /// Currently focused pane.
    pub focus: PaneFocus,

    /// Selected index in the namespaces list.
    pub ns_index: usize,

    /// Selected index in the components list.
    pub comp_index: usize,

    /// Selected index in the endpoints list.
    pub ep_index: usize,

    /// Whether the app should quit.
    pub should_quit: bool,

    /// Last error messages from sources (for display).
    pub errors: Vec<String>,

    /// Whether we're still waiting for the initial ETCD snapshot.
    pub loading: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            namespaces: Vec::new(),
            nats_stats: NatsStats::default(),
            metrics: PrometheusMetrics::default(),
            focus: PaneFocus::Namespaces,
            ns_index: 0,
            comp_index: 0,
            ep_index: 0,
            should_quit: false,
            errors: Vec::new(),
            loading: true,
        }
    }

    /// Handle an incoming event and update state accordingly.
    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::DiscoveryUpdate(namespaces) => {
                self.loading = false;
                self.namespaces = namespaces;
                // Clamp selection indices
                self.clamp_indices();
            }
            AppEvent::NatsUpdate(stats) => {
                self.nats_stats = stats;
            }
            AppEvent::MetricsUpdate(metrics) => {
                self.metrics = metrics;
            }
            AppEvent::SourceError { source, message } => {
                let msg = format!("[{}] {}", source, message);
                // Keep only the last 5 errors
                if self.errors.len() >= 5 {
                    self.errors.remove(0);
                }
                self.errors.push(msg);
            }
            AppEvent::Input(key_event) => {
                if let Some(action) = map_key(key_event) {
                    self.handle_action(action);
                }
            }
            AppEvent::Tick => {
                // No-op for now; UI refresh is driven by render calls
            }
        }
    }

    /// Handle a mapped action.
    pub fn handle_action(&mut self, action: Action) {
        match action {
            Action::MoveUp => self.move_selection(-1),
            Action::MoveDown => self.move_selection(1),
            Action::FocusLeft => self.focus = self.focus.prev(),
            Action::FocusRight => self.focus = self.focus.next(),
            Action::CycleFocus => self.focus = self.focus.next(),
            Action::Refresh => {
                // Sources auto-refresh; this is a UI hint
                self.errors.clear();
            }
            Action::Quit => {
                self.should_quit = true;
            }
        }
    }

    /// Move the selection in the focused pane by `delta` (+1 = down, -1 = up).
    fn move_selection(&mut self, delta: i32) {
        match self.focus {
            PaneFocus::Namespaces => {
                let len = self.namespaces.len();
                if len > 0 {
                    self.ns_index = wrap_index(self.ns_index, delta, len);
                    // Reset child selections when parent changes
                    self.comp_index = 0;
                    self.ep_index = 0;
                }
            }
            PaneFocus::Components => {
                if let Some(ns) = self.selected_namespace() {
                    let len = ns.components.len();
                    if len > 0 {
                        self.comp_index = wrap_index(self.comp_index, delta, len);
                        self.ep_index = 0;
                    }
                }
            }
            PaneFocus::Endpoints => {
                if let Some(comp) = self.selected_component() {
                    let len = comp.endpoints.len();
                    if len > 0 {
                        self.ep_index = wrap_index(self.ep_index, delta, len);
                    }
                }
            }
        }
    }

    /// Get the currently selected namespace (if any).
    pub fn selected_namespace(&self) -> Option<&Namespace> {
        self.namespaces.get(self.ns_index)
    }

    /// Get the currently selected component (if any).
    pub fn selected_component(&self) -> Option<&crate::model::Component> {
        self.selected_namespace()
            .and_then(|ns| ns.components.get(self.comp_index))
    }

    /// Get the currently selected endpoint (if any).
    #[allow(dead_code)]
    pub fn selected_endpoint(&self) -> Option<&crate::model::Endpoint> {
        self.selected_component()
            .and_then(|comp| comp.endpoints.get(self.ep_index))
    }

    /// Clamp all selection indices to valid bounds.
    fn clamp_indices(&mut self) {
        let ns_len = self.namespaces.len();
        if ns_len == 0 {
            self.ns_index = 0;
            self.comp_index = 0;
            self.ep_index = 0;
            return;
        }
        self.ns_index = self.ns_index.min(ns_len - 1);

        let comp_len = self.namespaces[self.ns_index].components.len();
        if comp_len == 0 {
            self.comp_index = 0;
            self.ep_index = 0;
            return;
        }
        self.comp_index = self.comp_index.min(comp_len - 1);

        let ep_len = self.namespaces[self.ns_index].components[self.comp_index]
            .endpoints
            .len();
        if ep_len == 0 {
            self.ep_index = 0;
            return;
        }
        self.ep_index = self.ep_index.min(ep_len - 1);
    }
}

/// Wrap an index by delta within [0, len), wrapping around at boundaries.
fn wrap_index(current: usize, delta: i32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let new = current as i32 + delta;
    if new < 0 {
        len - 1
    } else if new >= len as i32 {
        0
    } else {
        new as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn make_test_namespaces() -> Vec<Namespace> {
        vec![
            Namespace {
                name: "ns-a".into(),
                components: vec![
                    Component {
                        name: "backend".into(),
                        endpoints: vec![
                            Endpoint {
                                name: "generate".into(),
                                instance_count: 2,
                                status: HealthStatus::Ready,
                            },
                            Endpoint {
                                name: "health".into(),
                                instance_count: 1,
                                status: HealthStatus::Ready,
                            },
                        ],
                        instance_count: 3,
                        status: HealthStatus::Ready,
                        models: vec!["llama-7b".into()],
                    },
                    Component {
                        name: "frontend".into(),
                        endpoints: vec![Endpoint {
                            name: "http".into(),
                            instance_count: 1,
                            status: HealthStatus::Ready,
                        }],
                        instance_count: 1,
                        status: HealthStatus::Ready,
                        models: vec![],
                    },
                ],
            },
            Namespace {
                name: "ns-b".into(),
                components: vec![Component {
                    name: "worker".into(),
                    endpoints: vec![],
                    instance_count: 0,
                    status: HealthStatus::Offline,
                    models: vec![],
                }],
            },
        ]
    }

    #[test]
    fn test_app_initial_state() {
        let app = App::new();
        assert!(app.namespaces.is_empty());
        assert!(app.loading);
        assert!(!app.should_quit);
        assert_eq!(app.focus, PaneFocus::Namespaces);
    }

    #[test]
    fn test_discovery_update() {
        let mut app = App::new();
        let namespaces = make_test_namespaces();
        app.handle_event(AppEvent::DiscoveryUpdate(namespaces.clone()));

        assert!(!app.loading);
        assert_eq!(app.namespaces.len(), 2);
        assert_eq!(app.selected_namespace().unwrap().name, "ns-a");
    }

    #[test]
    fn test_navigation_down_up() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));

        assert_eq!(app.ns_index, 0);
        app.handle_action(Action::MoveDown);
        assert_eq!(app.ns_index, 1);
        app.handle_action(Action::MoveUp);
        assert_eq!(app.ns_index, 0);
    }

    #[test]
    fn test_navigation_wraps() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));

        // Wrap at top
        app.handle_action(Action::MoveUp);
        assert_eq!(app.ns_index, 1); // wraps to last

        // Wrap at bottom
        app.handle_action(Action::MoveDown);
        assert_eq!(app.ns_index, 0); // wraps to first
    }

    #[test]
    fn test_focus_cycling() {
        let mut app = App::new();
        assert_eq!(app.focus, PaneFocus::Namespaces);

        app.handle_action(Action::FocusRight);
        assert_eq!(app.focus, PaneFocus::Components);

        app.handle_action(Action::FocusRight);
        assert_eq!(app.focus, PaneFocus::Endpoints);

        app.handle_action(Action::FocusRight);
        assert_eq!(app.focus, PaneFocus::Namespaces); // wraps

        app.handle_action(Action::FocusLeft);
        assert_eq!(app.focus, PaneFocus::Endpoints);
    }

    #[test]
    fn test_tab_cycles_focus() {
        let mut app = App::new();
        app.handle_action(Action::CycleFocus);
        assert_eq!(app.focus, PaneFocus::Components);
    }

    #[test]
    fn test_component_navigation() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.focus = PaneFocus::Components;

        assert_eq!(app.comp_index, 0);
        assert_eq!(app.selected_component().unwrap().name, "backend");

        app.handle_action(Action::MoveDown);
        assert_eq!(app.comp_index, 1);
        assert_eq!(app.selected_component().unwrap().name, "frontend");
    }

    #[test]
    fn test_endpoint_navigation() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.focus = PaneFocus::Endpoints;

        assert_eq!(app.selected_endpoint().unwrap().name, "generate");
        app.handle_action(Action::MoveDown);
        assert_eq!(app.selected_endpoint().unwrap().name, "health");
    }

    #[test]
    fn test_child_selection_resets_on_parent_change() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));

        // Select second component
        app.focus = PaneFocus::Components;
        app.handle_action(Action::MoveDown);
        assert_eq!(app.comp_index, 1);

        // Move to next namespace — comp_index should reset
        app.focus = PaneFocus::Namespaces;
        app.handle_action(Action::MoveDown);
        assert_eq!(app.comp_index, 0);
    }

    #[test]
    fn test_quit_action() {
        let mut app = App::new();
        assert!(!app.should_quit);
        app.handle_action(Action::Quit);
        assert!(app.should_quit);
    }

    #[test]
    fn test_key_event_input() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));

        let key = KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE);
        app.handle_event(AppEvent::Input(key));
        assert_eq!(app.ns_index, 1);
    }

    #[test]
    fn test_source_error_handling() {
        let mut app = App::new();
        for i in 0..7 {
            app.handle_event(AppEvent::SourceError {
                source: "test".into(),
                message: format!("error {}", i),
            });
        }
        // Only last 5 errors kept
        assert_eq!(app.errors.len(), 5);
        assert!(app.errors[0].contains("error 2"));
    }

    #[test]
    fn test_nats_update() {
        let mut app = App::new();
        let stats = NatsStats {
            connected: true,
            server_id: "test-server".into(),
            msgs_in: 100,
            msgs_out: 50,
            bytes_in: 10240,
            bytes_out: 5120,
            streams: vec![],
        };
        app.handle_event(AppEvent::NatsUpdate(stats));
        assert!(app.nats_stats.connected);
        assert_eq!(app.nats_stats.msgs_in, 100);
    }

    #[test]
    fn test_metrics_update() {
        let mut app = App::new();
        let metrics = PrometheusMetrics {
            available: true,
            ttft_p50_ms: Some(42.0),
            ..Default::default()
        };
        app.handle_event(AppEvent::MetricsUpdate(metrics));
        assert!(app.metrics.available);
        assert_eq!(app.metrics.ttft_p50_ms, Some(42.0));
    }

    #[test]
    fn test_clamp_indices_empty() {
        let mut app = App::new();
        app.ns_index = 5;
        app.handle_event(AppEvent::DiscoveryUpdate(vec![]));
        assert_eq!(app.ns_index, 0);
    }

    #[test]
    fn test_clamp_indices_shrink() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.ns_index = 1;
        // Remove the second namespace
        let smaller = vec![make_test_namespaces().remove(0)];
        app.handle_event(AppEvent::DiscoveryUpdate(smaller));
        assert_eq!(app.ns_index, 0); // clamped
    }

    #[test]
    fn test_wrap_index() {
        assert_eq!(wrap_index(0, 1, 3), 1);
        assert_eq!(wrap_index(2, 1, 3), 0); // wrap forward
        assert_eq!(wrap_index(0, -1, 3), 2); // wrap backward
        assert_eq!(wrap_index(0, 0, 0), 0); // empty
    }

    #[test]
    fn test_pane_focus_cycle() {
        assert_eq!(PaneFocus::Namespaces.next(), PaneFocus::Components);
        assert_eq!(PaneFocus::Components.next(), PaneFocus::Endpoints);
        assert_eq!(PaneFocus::Endpoints.next(), PaneFocus::Namespaces);
        assert_eq!(PaneFocus::Namespaces.prev(), PaneFocus::Endpoints);
    }

    #[test]
    fn test_refresh_clears_errors() {
        let mut app = App::new();
        app.errors.push("some error".into());
        app.handle_action(Action::Refresh);
        assert!(app.errors.is_empty());
    }

    #[test]
    fn test_navigate_empty_components_pane() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        // ns-b has worker with no endpoints
        app.ns_index = 1;
        app.focus = PaneFocus::Endpoints;
        // Should not panic when navigating empty endpoints list
        app.handle_action(Action::MoveDown);
        assert_eq!(app.ep_index, 0);
        app.handle_action(Action::MoveUp);
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_navigate_no_namespaces() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(vec![]));
        // Should not panic with empty namespaces
        app.handle_action(Action::MoveDown);
        assert_eq!(app.ns_index, 0);
        app.focus = PaneFocus::Components;
        app.handle_action(Action::MoveDown);
        assert_eq!(app.comp_index, 0);
        app.focus = PaneFocus::Endpoints;
        app.handle_action(Action::MoveDown);
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_selected_returns_none_when_empty() {
        let app = App::new();
        assert!(app.selected_namespace().is_none());
        assert!(app.selected_component().is_none());
        assert!(app.selected_endpoint().is_none());
    }

    #[test]
    fn test_selected_endpoint_returns_correct() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        let ep = app.selected_endpoint().unwrap();
        assert_eq!(ep.name, "generate");
        assert_eq!(ep.instance_count, 2);
    }

    #[test]
    fn test_tick_event_is_noop() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        let ns_before = app.ns_index;
        let focus_before = app.focus;
        app.handle_event(AppEvent::Tick);
        // Tick should not change any state
        assert_eq!(app.ns_index, ns_before);
        assert_eq!(app.focus, focus_before);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_rapid_sequential_discovery_updates() {
        let mut app = App::new();
        // Simulate rapid updates (e.g., multiple ETCD watches firing)
        for i in 0..10 {
            let ns = vec![Namespace {
                name: format!("ns-{}", i),
                components: vec![],
            }];
            app.handle_event(AppEvent::DiscoveryUpdate(ns));
        }
        // Should have the last update
        assert_eq!(app.namespaces.len(), 1);
        assert_eq!(app.namespaces[0].name, "ns-9");
    }

    #[test]
    fn test_unmapped_key_does_not_change_state() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        let ns_before = app.ns_index;
        app.handle_event(AppEvent::Input(key));
        assert_eq!(app.ns_index, ns_before);
    }

    #[test]
    fn test_clamp_indices_with_empty_components() {
        let mut app = App::new();
        app.comp_index = 5;
        app.ep_index = 3;
        // Namespace with no components
        let ns = vec![Namespace {
            name: "ns".into(),
            components: vec![],
        }];
        app.handle_event(AppEvent::DiscoveryUpdate(ns));
        assert_eq!(app.comp_index, 0);
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_clamp_indices_with_empty_endpoints() {
        let mut app = App::new();
        app.ep_index = 5;
        let ns = vec![Namespace {
            name: "ns".into(),
            components: vec![Component {
                name: "comp".into(),
                endpoints: vec![],
                instance_count: 0,
                status: HealthStatus::Offline,
                models: vec![],
            }],
        }];
        app.handle_event(AppEvent::DiscoveryUpdate(ns));
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_multiple_error_sources() {
        let mut app = App::new();
        app.handle_event(AppEvent::SourceError {
            source: "etcd".into(),
            message: "connection refused".into(),
        });
        app.handle_event(AppEvent::SourceError {
            source: "nats".into(),
            message: "timeout".into(),
        });
        assert_eq!(app.errors.len(), 2);
        assert!(app.errors[0].contains("[etcd]"));
        assert!(app.errors[1].contains("[nats]"));
    }

    #[test]
    fn test_discovery_update_clears_loading() {
        let mut app = App::new();
        assert!(app.loading);
        app.handle_event(AppEvent::DiscoveryUpdate(vec![]));
        assert!(!app.loading);
    }

    #[test]
    fn test_component_navigation_wraps() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.focus = PaneFocus::Components;
        // ns-a has 2 components (backend, frontend)
        app.handle_action(Action::MoveUp); // wrap from 0 to 1
        assert_eq!(app.comp_index, 1);
        app.handle_action(Action::MoveDown); // wrap from 1 to 0
        assert_eq!(app.comp_index, 0);
    }

    #[test]
    fn test_endpoint_navigation_wraps() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.focus = PaneFocus::Endpoints;
        // backend has 2 endpoints (generate, health)
        app.handle_action(Action::MoveUp); // wrap from 0 to 1
        assert_eq!(app.ep_index, 1);
        app.handle_action(Action::MoveDown); // wrap from 1 to 0
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_component_nav_resets_ep_index() {
        let mut app = App::new();
        app.handle_event(AppEvent::DiscoveryUpdate(make_test_namespaces()));
        app.focus = PaneFocus::Endpoints;
        app.handle_action(Action::MoveDown); // ep_index = 1
        assert_eq!(app.ep_index, 1);

        // Switch component — ep_index should reset
        app.focus = PaneFocus::Components;
        app.handle_action(Action::MoveDown); // comp_index = 1
        assert_eq!(app.ep_index, 0);
    }

    #[test]
    fn test_nats_stats_default() {
        let stats = NatsStats::default();
        assert!(!stats.connected);
        assert_eq!(stats.msgs_in, 0);
        assert_eq!(stats.msgs_out, 0);
        assert_eq!(stats.bytes_in, 0);
        assert_eq!(stats.bytes_out, 0);
        assert!(stats.streams.is_empty());
    }

    #[test]
    fn test_metrics_default() {
        let metrics = PrometheusMetrics::default();
        assert!(!metrics.available);
        assert!(metrics.ttft_p50_ms.is_none());
        assert!(metrics.tpot_p50_ms.is_none());
        assert!(metrics.requests_inflight.is_none());
        assert!(metrics.tokens_per_sec.is_none());
    }

    #[test]
    fn test_nats_update_with_streams() {
        use crate::model::StreamInfo;
        let mut app = App::new();
        let stats = NatsStats {
            connected: true,
            server_id: "test".into(),
            msgs_in: 0,
            msgs_out: 0,
            bytes_in: 0,
            bytes_out: 0,
            streams: vec![
                StreamInfo {
                    name: "audit-events".into(),
                    consumer_count: 3,
                    message_count: 1500,
                },
                StreamInfo {
                    name: "kv-events".into(),
                    consumer_count: 1,
                    message_count: 42,
                },
            ],
        };
        app.handle_event(AppEvent::NatsUpdate(stats));
        assert_eq!(app.nats_stats.streams.len(), 2);
        assert_eq!(app.nats_stats.streams[0].name, "audit-events");
        assert_eq!(app.nats_stats.streams[0].consumer_count, 3);
        assert_eq!(app.nats_stats.streams[1].message_count, 42);
    }

    #[test]
    fn test_full_metrics_update() {
        let mut app = App::new();
        let metrics = PrometheusMetrics {
            available: true,
            ttft_p50_ms: Some(42.0),
            ttft_p99_ms: Some(180.0),
            tpot_p50_ms: Some(8.0),
            tpot_p99_ms: Some(25.0),
            requests_inflight: Some(12),
            requests_queued: Some(3),
            tokens_per_sec: Some(450.5),
        };
        app.handle_event(AppEvent::MetricsUpdate(metrics));
        assert!(app.metrics.available);
        assert_eq!(app.metrics.ttft_p50_ms, Some(42.0));
        assert_eq!(app.metrics.ttft_p99_ms, Some(180.0));
        assert_eq!(app.metrics.tpot_p50_ms, Some(8.0));
        assert_eq!(app.metrics.tpot_p99_ms, Some(25.0));
        assert_eq!(app.metrics.requests_inflight, Some(12));
        assert_eq!(app.metrics.requests_queued, Some(3));
        assert!((app.metrics.tokens_per_sec.unwrap() - 450.5).abs() < 0.01);
    }
}
