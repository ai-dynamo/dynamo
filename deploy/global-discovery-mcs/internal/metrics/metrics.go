// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package metrics defines the dynamo_mcs_registry_* Prometheus collectors of
// mcs-registry. Counters are pre-initialised for every known label value so
// that they appear on the first scrape. gRPC server metrics come from the
// go-grpc-middleware Prometheus provider and are registered by internal/server.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

// Namespace prefixes every metric name.
const Namespace = "dynamo_mcs_registry"

// Watch kinds, the values of the kind label.
const (
	KindCluster       = "cluster"
	KindServiceImport = "service_import"
)

// Kinds lists every watch kind.
var Kinds = []string{KindCluster, KindServiceImport}

// Watch event types, the values of the type label.
var EventTypes = []string{"snapshot_start", "upsert", "delete", "snapshot_end", "bookmark"}

// Watch resync reasons.
const (
	// ResyncLagged: the watcher's queue overflowed.
	ResyncLagged = "lagged"
	// ResyncRevisionUnknown: the resume revision is malformed or from another
	// view epoch.
	ResyncRevisionUnknown = "revision_unknown"
	// ResyncRevisionExpired: the resume revision is older than the history.
	ResyncRevisionExpired = "revision_expired"
)

// ResyncReasons lists every resync reason.
var ResyncReasons = []string{ResyncLagged, ResyncRevisionUnknown, ResyncRevisionExpired}

// Watch rejection reasons.
const (
	RejectTooManyWatchers = "too_many_watchers"
	RejectShuttingDown    = "shutting_down"
)

// Lease kinds of lease_expirations_total.
const (
	LeaseCluster = "cluster"
)

// Record kinds of compaction_deleted_total.
const (
	RecordCluster     = "cluster"
	RecordExport      = "export"
	RecordEndpointSet = "endpoint_set"
)

// Store operations of the store metrics.
var StoreOperations = []string{
	"put_cluster", "renew_cluster_lease", "get_cluster", "delete_cluster",
	"put_export", "delete_export", "put_endpoint_set", "delete_endpoint_set",
	"list_cluster_sets", "load_cluster_set", "ping",
}

// Store results of store_request_duration_seconds: ok, conditional (the
// write's condition refused it, or the record was not found) and error.
var StoreResults = []string{"ok", "conditional", "error"}

// Conditional failure reasons of store_conditional_failures_total, keyed by
// the operations that can produce them.
var ConditionalFailures = map[string][]string{
	"renew_cluster_lease": {"lease_epoch_mismatch", "not_found"},
	"put_export":          {"stale_generation"},
	"put_endpoint_set":    {"stale_generation", "too_large"},
}

// Metrics holds every mcs-registry collector.
type Metrics struct {
	Clusters       *prometheus.GaugeVec
	Exports        *prometheus.GaugeVec
	Endpoints      *prometheus.GaugeVec
	ServiceImports *prometheus.GaugeVec

	LeaseExpirations *prometheus.CounterVec

	Watchers        *prometheus.GaugeVec
	WatchEventsSent *prometheus.CounterVec
	WatchResyncs    *prometheus.CounterVec
	WatchRejected   *prometheus.CounterVec

	ViewRefreshDuration prometheus.Histogram
	ViewRefreshFailures prometheus.Counter
	ViewAge             prometheus.Gauge

	StoreRequestDuration     *prometheus.HistogramVec
	StoreConditionalFailures *prometheus.CounterVec

	CompactionDeleted *prometheus.CounterVec

	BuildInfo *prometheus.GaugeVec
}

// New creates the collectors, registers them on reg and pre-initialises the
// counters.
func New(reg prometheus.Registerer) *Metrics {
	m := &Metrics{
		Clusters: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "clusters",
			Help: "Registered clusters by lease state (active, expired).",
		}, []string{"clusterset", "state"}),
		Exports: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "exports",
			Help: "ServiceExport records.",
		}, []string{"clusterset"}),
		Endpoints: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "endpoints",
			Help: "Endpoints in stored endpoint sets by readiness (ready, not_ready).",
		}, []string{"clusterset", "condition"}),
		ServiceImports: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "service_imports",
			Help: "Merged ServiceImports by conflict (true, false).",
		}, []string{"clusterset", "conflict"}),
		LeaseExpirations: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "lease_expirations_total",
			Help: "Leases seen expiring by this replica, by kind (cluster).",
		}, []string{"kind"}),
		Watchers: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "watchers",
			Help: "Open watch streams by kind.",
		}, []string{"kind"}),
		WatchEventsSent: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "watch_events_sent_total",
			Help: "Watch events sent by kind and event type.",
		}, []string{"kind", "type"}),
		WatchResyncs: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "watch_resyncs_total",
			Help: "Watch snapshots sent instead of deltas, by kind and reason (lagged, revision_unknown, revision_expired).",
		}, []string{"kind", "reason"}),
		WatchRejected: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "watch_rejected_total",
			Help: "Watch streams refused, by reason (too_many_watchers, shutting_down).",
		}, []string{"reason"}),
		ViewRefreshDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: Namespace, Name: "view_refresh_duration_seconds",
			Help:    "Duration of one clusterset refresh from the store.",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		}),
		ViewRefreshFailures: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: Namespace, Name: "view_refresh_failures_total",
			Help: "Failed clusterset refreshes and clusterset discoveries.",
		}),
		ViewAge: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "view_age_seconds",
			Help: "Age of the oldest clusterset view since its last successful refresh.",
		}),
		StoreRequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: Namespace, Name: "store_request_duration_seconds",
			Help:    "Store request duration by operation and result (ok, conditional, error).",
			Buckets: []float64{.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5},
		}, []string{"operation", "result"}),
		StoreConditionalFailures: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "store_conditional_failures_total",
			Help: "Store writes refused by their condition, by operation and reason.",
		}, []string{"operation", "reason"}),
		CompactionDeleted: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: Namespace, Name: "compaction_deleted_total",
			Help: "Records removed by compaction, by kind (cluster, export, endpoint_set).",
		}, []string{"kind"}),
		BuildInfo: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: Namespace, Name: "build_info",
			Help: "Build metadata of mcs-registry; always 1.",
		}, []string{"version", "git_hash"}),
	}
	reg.MustRegister(
		m.Clusters, m.Exports, m.Endpoints, m.ServiceImports,
		m.LeaseExpirations,
		m.Watchers, m.WatchEventsSent, m.WatchResyncs, m.WatchRejected,
		m.ViewRefreshDuration, m.ViewRefreshFailures, m.ViewAge,
		m.StoreRequestDuration, m.StoreConditionalFailures,
		m.CompactionDeleted, m.BuildInfo,
	)
	m.init()
	return m
}

func (m *Metrics) init() {
	for _, k := range []string{LeaseCluster} {
		m.LeaseExpirations.WithLabelValues(k)
	}
	for _, k := range Kinds {
		m.Watchers.WithLabelValues(k)
		for _, t := range EventTypes {
			m.WatchEventsSent.WithLabelValues(k, t)
		}
		for _, r := range ResyncReasons {
			m.WatchResyncs.WithLabelValues(k, r)
		}
	}
	for _, r := range []string{RejectTooManyWatchers, RejectShuttingDown} {
		m.WatchRejected.WithLabelValues(r)
	}
	for _, op := range StoreOperations {
		for _, res := range StoreResults {
			m.StoreRequestDuration.WithLabelValues(op, res)
		}
	}
	for op, reasons := range ConditionalFailures {
		for _, r := range reasons {
			m.StoreConditionalFailures.WithLabelValues(op, r)
		}
	}
	for _, k := range []string{RecordCluster, RecordExport, RecordEndpointSet} {
		m.CompactionDeleted.WithLabelValues(k)
	}
}

// SetBuildInfo sets the build_info series.
func (m *Metrics) SetBuildInfo(version, gitHash string) {
	m.BuildInfo.Reset()
	m.BuildInfo.WithLabelValues(version, gitHash).Set(1)
}

// ClusterSetGauges are the per-clusterset object counts.
type ClusterSetGauges struct {
	ActiveClusters, ExpiredClusters int
	Exports                         int
	ReadyEndpoints, NotReadyEnds    int
	Imports, ConflictedImports      int
}

// SetClusterSet sets the gauges of one clusterset.
func (m *Metrics) SetClusterSet(cs string, g ClusterSetGauges) {
	m.Clusters.WithLabelValues(cs, "active").Set(float64(g.ActiveClusters))
	m.Clusters.WithLabelValues(cs, "expired").Set(float64(g.ExpiredClusters))
	m.Exports.WithLabelValues(cs).Set(float64(g.Exports))
	m.Endpoints.WithLabelValues(cs, "ready").Set(float64(g.ReadyEndpoints))
	m.Endpoints.WithLabelValues(cs, "not_ready").Set(float64(g.NotReadyEnds))
	m.ServiceImports.WithLabelValues(cs, "true").Set(float64(g.ConflictedImports))
	m.ServiceImports.WithLabelValues(cs, "false").Set(float64(g.Imports - g.ConflictedImports))
}

// DeleteClusterSet removes the gauges of a clusterset the view dropped.
func (m *Metrics) DeleteClusterSet(cs string) {
	labels := prometheus.Labels{"clusterset": cs}
	for _, g := range []*prometheus.GaugeVec{m.Clusters, m.Exports, m.Endpoints, m.ServiceImports} {
		g.DeletePartialMatch(labels)
	}
}
