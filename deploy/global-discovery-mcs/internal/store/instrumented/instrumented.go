// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package instrumented wraps a store.Store with the dynamo_mcs_registry_store_*
// metrics and an OpenTelemetry child span per call (mcs.store.<operation>).
package instrumented

import (
	"context"
	"errors"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

const tracerName = "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"

// Store is an instrumented store.Store.
type Store struct {
	next    store.Store
	metrics *metrics.Metrics
	tracer  trace.Tracer
}

var _ store.Store = (*Store)(nil)

// New wraps next. m may be nil to record spans only.
func New(next store.Store, m *metrics.Metrics) *Store {
	return &Store{next: next, metrics: m, tracer: otel.Tracer(tracerName)}
}

// Unwrap returns the wrapped store.
func (s *Store) Unwrap() store.Store { return s.next }

// conditionalReason classifies an error a write's condition produced.
func conditionalReason(err error) (string, bool) {
	switch {
	case errors.Is(err, store.ErrStaleGeneration{}):
		return "stale_generation", true
	case errors.Is(err, store.ErrLeaseEpochMismatch):
		return "lease_epoch_mismatch", true
	case errors.Is(err, store.ErrTooLarge):
		return "too_large", true
	case errors.Is(err, store.ErrNotFound):
		return "not_found", true
	}
	return "", false
}

func (s *Store) start(ctx context.Context, op string, attrs ...attribute.KeyValue) (context.Context, func(error)) {
	ctx, span := s.tracer.Start(ctx, "mcs.store."+op, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	begin := time.Now()
	return ctx, func(err error) {
		result := "ok"
		if err != nil {
			reason, conditional := conditionalReason(err)
			switch {
			case conditional && op == "get_cluster":
				// A miss is an answer, not a failure.
			case conditional:
				result = "conditional"
				span.SetAttributes(attribute.String("mcs.store.condition", reason))
				if s.metrics != nil {
					s.metrics.StoreConditionalFailures.WithLabelValues(op, reason).Inc()
				}
			default:
				result = "error"
				span.RecordError(err)
				span.SetStatus(codes.Error, err.Error())
				span.SetAttributes(attribute.Bool("error", true))
			}
		}
		if s.metrics != nil {
			s.metrics.StoreRequestDuration.WithLabelValues(op, result).Observe(time.Since(begin).Seconds())
		}
		span.End()
	}
}

func clusterAttrs(k store.ClusterKey) []attribute.KeyValue {
	return []attribute.KeyValue{attribute.String("dynamo.clusterset", k.ClusterSet), attribute.String("dynamo.cluster.id", k.Cluster)}
}

func exportAttrs(k store.ExportKey) []attribute.KeyValue {
	return append(clusterAttrs(k.ClusterKey()), attribute.String("k8s.namespace.name", k.Namespace), attribute.String("k8s.service.name", k.Name))
}

// PutCluster implements store.Store.
func (s *Store) PutCluster(ctx context.Context, rec store.ClusterRecord, now time.Time) (out store.ClusterRecord, err error) {
	ctx, done := s.start(ctx, "put_cluster", clusterAttrs(rec.Key)...)
	defer func() { done(err) }()
	return s.next.PutCluster(ctx, rec, now)
}

// RenewClusterLease implements store.Store.
func (s *Store) RenewClusterLease(ctx context.Context, key store.ClusterKey, epoch uint64, now time.Time) (out store.ClusterRecord, err error) {
	ctx, done := s.start(ctx, "renew_cluster_lease", clusterAttrs(key)...)
	defer func() { done(err) }()
	return s.next.RenewClusterLease(ctx, key, epoch, now)
}

// GetCluster implements store.Store.
func (s *Store) GetCluster(ctx context.Context, key store.ClusterKey) (out store.ClusterRecord, err error) {
	ctx, done := s.start(ctx, "get_cluster", clusterAttrs(key)...)
	defer func() { done(err) }()
	return s.next.GetCluster(ctx, key)
}

// DeleteCluster implements store.Store.
func (s *Store) DeleteCluster(ctx context.Context, key store.ClusterKey) (err error) {
	ctx, done := s.start(ctx, "delete_cluster", clusterAttrs(key)...)
	defer func() { done(err) }()
	return s.next.DeleteCluster(ctx, key)
}

// PutExport implements store.Store.
func (s *Store) PutExport(ctx context.Context, rec store.ExportRecord) (err error) {
	ctx, done := s.start(ctx, "put_export", exportAttrs(rec.Key)...)
	defer func() { done(err) }()
	return s.next.PutExport(ctx, rec)
}

// DeleteExport implements store.Store.
func (s *Store) DeleteExport(ctx context.Context, key store.ExportKey) (err error) {
	ctx, done := s.start(ctx, "delete_export", exportAttrs(key)...)
	defer func() { done(err) }()
	return s.next.DeleteExport(ctx, key)
}

// PutEndpointSet implements store.Store.
func (s *Store) PutEndpointSet(ctx context.Context, rec store.EndpointSetRecord) (err error) {
	ctx, done := s.start(ctx, "put_endpoint_set", exportAttrs(rec.Key)...)
	defer func() { done(err) }()
	return s.next.PutEndpointSet(ctx, rec)
}

// DeleteEndpointSet implements store.Store.
func (s *Store) DeleteEndpointSet(ctx context.Context, key store.ExportKey) (err error) {
	ctx, done := s.start(ctx, "delete_endpoint_set", exportAttrs(key)...)
	defer func() { done(err) }()
	return s.next.DeleteEndpointSet(ctx, key)
}

// ListClusterSets implements store.Store.
func (s *Store) ListClusterSets(ctx context.Context) (out []string, err error) {
	ctx, done := s.start(ctx, "list_cluster_sets")
	defer func() { done(err) }()
	return s.next.ListClusterSets(ctx)
}

// LoadClusterSet implements store.Store.
func (s *Store) LoadClusterSet(ctx context.Context, clusterSet string) (out store.ClusterSetSnapshot, err error) {
	ctx, done := s.start(ctx, "load_cluster_set", attribute.String("dynamo.clusterset", clusterSet))
	defer func() { done(err) }()
	return s.next.LoadClusterSet(ctx, clusterSet)
}

// Ping implements store.Store.
func (s *Store) Ping(ctx context.Context) (err error) {
	ctx, done := s.start(ctx, "ping")
	defer func() { done(err) }()
	return s.next.Ping(ctx)
}

// Close implements store.Store.
func (s *Store) Close() error { return s.next.Close() }
