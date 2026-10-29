// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
)

// faultyStore fails the operations named in fail.
type faultyStore struct {
	store.Store
	mu   sync.Mutex
	fail map[string]error
	// loads counts LoadClusterSet calls.
	loads int
}

func (s *faultyStore) err(op string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fail[op]
}

func (s *faultyStore) set(op string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.fail == nil {
		s.fail = map[string]error{}
	}
	s.fail[op] = err
}

func (s *faultyStore) ListClusterSets(ctx context.Context) ([]string, error) {
	if err := s.err("list"); err != nil {
		return nil, err
	}
	return s.Store.ListClusterSets(ctx)
}

func (s *faultyStore) LoadClusterSet(ctx context.Context, name string) (store.ClusterSetSnapshot, error) {
	s.mu.Lock()
	s.loads++
	s.mu.Unlock()
	if err := s.err("load"); err != nil {
		return store.ClusterSetSnapshot{}, err
	}
	return s.Store.LoadClusterSet(ctx, name)
}

func (s *faultyStore) GetCluster(ctx context.Context, key store.ClusterKey) (store.ClusterRecord, error) {
	if err := s.err("get"); err != nil {
		return store.ClusterRecord{}, err
	}
	return s.Store.GetCluster(ctx, key)
}

func (s *faultyStore) DeleteCluster(ctx context.Context, key store.ClusterKey) error {
	if err := s.err("delete"); err != nil {
		return err
	}
	return s.Store.DeleteCluster(ctx, key)
}

func (s *faultyStore) loadCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loads
}

func seed(t *testing.T, st store.Store, now time.Time) {
	t.Helper()
	ctx := context.Background()
	_, err := st.PutCluster(ctx, clusterRec("a", "us-east-1", now), now)
	require.NoError(t, err)
	require.NoError(t, st.PutExport(ctx, exportRec("a", "ns", "web", 1, now, nil, nil, httpPort)))
	require.NoError(t, st.PutEndpointSet(ctx, endpointSet("a", "ns", "web", 1, true)))
}

func TestRefresherSync(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	seed(t, st, t0)
	r := NewRefresher(f.view, st, RefresherOptions{Metrics: f.metrics})

	require.NoError(t, r.Sync(context.Background()))
	importOf(t, f.view, "web")
	ok, _ := f.view.Ready(t0, 6*time.Second)
	assert.True(t, ok)

	st.set("list", store.ErrUnavailable)
	assert.ErrorIs(t, r.Sync(context.Background()), store.ErrUnavailable)
	st.set("list", nil)
	st.set("load", store.ErrUnavailable)
	assert.ErrorIs(t, r.RefreshAll(context.Background()), store.ErrUnavailable)
	assert.Equal(t, 2.0, testutilValue(f.metrics.ViewRefreshFailures))
}

func TestRefresherRun(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	seed(t, st, t0)
	r := NewRefresher(f.view, st, RefresherOptions{
		RefreshInterval: 2 * time.Second, LeaseEvalInterval: time.Second, DiscoveryInterval: 30 * time.Second,
		Metrics: f.metrics,
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		r.Run(ctx)
		close(done)
	}()
	f.clock.BlockUntil(3)
	importOf(t, f.view, "web")

	// A write from another replica appears after one refresh interval.
	require.NoError(t, st.PutExport(context.Background(), exportRec("a", "ns", "api", 1, t0, nil, nil, httpPort)))
	f.clock.Advance(2 * time.Second)
	require.Eventually(t, func() bool {
		_, ok := f.view.Get(cs, KindServiceImport, ImportKey("ns", "api"))
		return ok
	}, 5*time.Second, 10*time.Millisecond)

	// Leases are evaluated between refreshes, without a store read.
	st.set("load", store.ErrUnavailable)
	f.clock.Advance(31 * time.Second)
	require.Eventually(t, func() bool {
		o, _ := f.view.Get(cs, KindServiceImport, ImportKey("ns", "web"))
		return len(o.Full.(*mcsv1.ServiceImport).GetEndpoints()) == 0
	}, 5*time.Second, 10*time.Millisecond)

	// A clusterset created by a watcher is loaded at once.
	st.set("load", nil)
	before := st.loadCount()
	f.view.Ensure("fresh")
	require.Eventually(t, func() bool { return st.loadCount() > before }, 5*time.Second, 10*time.Millisecond)

	cancel()
	<-done
}

func TestRefresherRetriesDiscovery(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	seed(t, st, t0)
	st.set("list", store.ErrUnavailable)
	r := NewRefresher(f.view, st, RefresherOptions{RefreshInterval: time.Second, DiscoveryInterval: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		r.Run(ctx)
		close(done)
	}()
	f.clock.BlockUntil(3)
	assert.False(t, f.view.Discovered())
	st.set("list", nil)
	f.clock.Advance(time.Second)
	require.Eventually(t, func() bool {
		ok, _ := f.view.Ready(f.clock.Now(), time.Minute)
		return ok
	}, 5*time.Second, 10*time.Millisecond)
	cancel()
	<-done
}

func TestCompaction(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	ctx := context.Background()
	old := t0.Add(-25 * time.Hour)
	seed(t, st, old)
	// A live cluster with an export, an orphan export of an unregistered
	// cluster and a stale orphan endpoint set.
	_, err := st.PutCluster(ctx, clusterRec("b", "us-west-2", t0), t0)
	require.NoError(t, err)
	require.NoError(t, st.PutExport(ctx, exportRec("ghost", "ns", "web", 1, old, nil, nil, httpPort)))
	orphanSet := endpointSet("b", "ns", "gone", 1, true)
	orphanSet.UpdatedAt = old
	require.NoError(t, st.PutEndpointSet(ctx, orphanSet))
	require.NoError(t, st.PutExport(ctx, exportRec("b", "dyn", "fe", 1, old, map[string]string{"tier": "gold"}, nil, httpPort)))

	r := NewRefresher(f.view, st, RefresherOptions{})
	require.NoError(t, r.Sync(ctx))
	c := NewCompactor(f.view, st, CompactorOptions{Retention: 24 * time.Hour, LeaseSkew: 2 * time.Second, Metrics: f.metrics})
	require.NoError(t, c.Compact(ctx))

	snap, err := st.LoadClusterSet(ctx, cs)
	require.NoError(t, err)
	assert.Len(t, snap.Clusters, 1, "a expired more than the retention ago")
	assert.Equal(t, "b", snap.Clusters[0].Key.Cluster)
	require.Len(t, snap.Exports, 1, "the orphan export and a's export are gone")
	assert.Equal(t, "fe", snap.Exports[0].Key.Name)
	assert.Empty(t, snap.EndpointSets)
	assert.Equal(t, snap.Clusters, f.view.Records(cs).Clusters, "the view follows")

	for kind, n := range map[string]float64{
		metrics.RecordCluster: 1, metrics.RecordExport: 1, metrics.RecordEndpointSet: 1,
	} {
		assert.Equal(t, n, testutilValue(f.metrics.CompactionDeleted.WithLabelValues(kind)), kind)
	}
	// Idempotent.
	require.NoError(t, c.Compact(ctx))
	assert.Equal(t, 1.0, testutilValue(f.metrics.CompactionDeleted.WithLabelValues(metrics.RecordCluster)))
}

func TestCompactionRechecksBeforeDelete(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	ctx := context.Background()
	old := t0.Add(-25 * time.Hour)
	seed(t, st, old)
	require.NoError(t, NewRefresher(f.view, st, RefresherOptions{}).Sync(ctx))

	// The controller came back after the view was loaded.
	_, err := st.PutCluster(ctx, clusterRec("a", "us-east-1", t0), t0)
	require.NoError(t, err)
	c := NewCompactor(f.view, st, CompactorOptions{Retention: 24 * time.Hour})
	require.NoError(t, c.Compact(ctx))
	_, err = st.GetCluster(ctx, store.ClusterKey{ClusterSet: cs, Cluster: "a"})
	require.NoError(t, err)
	rec, ok := f.view.ClusterRecord(store.ClusterKey{ClusterSet: cs, Cluster: "a"})
	require.True(t, ok)
	assert.Equal(t, t0, rec.Lease.RenewedAt)

	// Errors are reported and the records kept.
	f.clock.Advance(26 * time.Hour)
	st.set("get", store.ErrUnavailable)
	assert.ErrorIs(t, c.Compact(ctx), store.ErrUnavailable)
	st.set("get", nil)
	st.set("delete", store.ErrUnavailable)
	assert.ErrorIs(t, c.Compact(ctx), store.ErrUnavailable)
	st.set("delete", nil)

	// A cluster already deleted elsewhere leaves the view too.
	require.NoError(t, st.DeleteCluster(ctx, store.ClusterKey{ClusterSet: cs, Cluster: "a"}))
	require.NoError(t, c.Compact(ctx))
	_, ok = f.view.ClusterRecord(store.ClusterKey{ClusterSet: cs, Cluster: "a"})
	assert.False(t, ok)
}

func TestCompactorRun(t *testing.T) {
	f := newFixture(t)
	st := &faultyStore{Store: memory.New()}
	seed(t, st, t0.Add(-48*time.Hour))
	require.NoError(t, NewRefresher(f.view, st, RefresherOptions{}).Sync(context.Background()))
	c := NewCompactor(f.view, st, CompactorOptions{Interval: time.Minute})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		c.Run(ctx)
		close(done)
	}()
	f.clock.BlockUntil(1)
	f.clock.Advance(time.Minute)
	require.Eventually(t, func() bool {
		_, err := st.GetCluster(context.Background(), store.ClusterKey{ClusterSet: cs, Cluster: "a"})
		return errors.Is(err, store.ErrNotFound)
	}, 5*time.Second, 10*time.Millisecond)
	cancel()
	<-done
}
