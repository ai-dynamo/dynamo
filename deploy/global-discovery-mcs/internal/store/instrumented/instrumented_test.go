// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package instrumented

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/storetest"
)

func TestConformance(t *testing.T) {
	storetest.Run(t, func(t *testing.T) store.Store { return New(memory.New(), nil) })
}

func TestMetricsAndSpans(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	m := metrics.New(prometheus.NewRegistry())
	st := New(memory.New(), m)
	assert.NotNil(t, st.Unwrap())
	ctx := context.Background()
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	key := store.ClusterKey{ClusterSet: "cs", Cluster: "a"}

	_, err := st.GetCluster(ctx, key)
	require.ErrorIs(t, err, store.ErrNotFound)
	_, err = st.PutCluster(ctx, store.ClusterRecord{Key: key, Lease: store.Lease{TTL: time.Minute}}, now)
	require.NoError(t, err)
	_, err = st.RenewClusterLease(ctx, key, 1, now)
	require.ErrorIs(t, err, store.ErrLeaseEpochMismatch)
	ek := store.ExportKey{ClusterSet: "cs", Cluster: "a", Namespace: "ns", Name: "svc"}
	require.NoError(t, st.PutExport(ctx, store.ExportRecord{Key: ek, Generation: 2}))
	require.ErrorIs(t, st.PutExport(ctx, store.ExportRecord{Key: ek, Generation: 1}), store.ErrStaleGeneration{})
	require.NoError(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek, Generation: 1}))
	require.ErrorIs(t, st.PutEndpointSet(ctx, store.EndpointSetRecord{Key: ek, Generation: 2, Endpoints: make([]store.Endpoint, 1001)}), store.ErrTooLarge)
	require.NoError(t, st.DeleteEndpointSet(ctx, ek))
	require.NoError(t, st.DeleteExport(ctx, ek))
	_, err = st.ListClusterSets(ctx)
	require.NoError(t, err)
	_, err = st.LoadClusterSet(ctx, "cs")
	require.NoError(t, err)
	require.NoError(t, st.Ping(ctx))
	require.NoError(t, st.DeleteCluster(ctx, key))
	require.NoError(t, st.Close())
	require.ErrorIs(t, st.Ping(ctx), store.ErrUnavailable)

	assert.Equal(t, 1.0, testutil.ToFloat64(m.StoreConditionalFailures.WithLabelValues("put_export", "stale_generation")))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.StoreConditionalFailures.WithLabelValues("put_endpoint_set", "too_large")))
	assert.Equal(t, 1.0, testutil.ToFloat64(m.StoreConditionalFailures.WithLabelValues("renew_cluster_lease", "lease_epoch_mismatch")))
	assert.Equal(t, 1, testutil.CollectAndCount(m.StoreRequestDuration.WithLabelValues("delete_export", "ok").(prometheus.Histogram)))

	spans := rec.Ended()
	require.NotEmpty(t, spans)
	assert.Equal(t, "mcs.store.get_cluster", spans[0].Name())
	last := spans[len(spans)-1]
	assert.Equal(t, "mcs.store.ping", last.Name())
	assert.Equal(t, "Error", last.Status().Code.String())
}

func TestConditionalReason(t *testing.T) {
	_, ok := conditionalReason(errors.New("x"))
	assert.False(t, ok)
	r, ok := conditionalReason(store.ErrNotFound)
	assert.True(t, ok)
	assert.Equal(t, "not_found", r)
}
