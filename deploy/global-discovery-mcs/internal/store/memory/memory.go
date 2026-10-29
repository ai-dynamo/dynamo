// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package memory is an in-process store.Store. It has the conditional
// semantics of the DynamoDB store and backs development, single-replica
// deployments, the k3d end-to-end tests and unit tests. Nothing is persisted.
package memory

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Store is an in-memory store.Store. It is safe for concurrent use.
type Store struct {
	mu          sync.RWMutex
	closed      bool
	clusterSets map[string]*clusterSet
}

// clusterSet holds the records of one clusterset.
type clusterSet struct {
	clusters  map[string]store.ClusterRecord
	exports   map[store.ExportKey]store.ExportRecord
	endpoints map[store.ExportKey]store.EndpointSetRecord
}

var _ store.Store = (*Store)(nil)

// New returns an empty store.
func New() *Store {
	return &Store{clusterSets: map[string]*clusterSet{}}
}

// read runs fn under the read lock after the context and closed checks.
func (s *Store) read(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return store.ErrUnavailable
	}
	return fn()
}

// write runs fn under the write lock after the context and closed checks.
func (s *Store) write(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return store.ErrUnavailable
	}
	return fn()
}

// set returns the clusterset, creating it when create is true; otherwise it
// returns nil for an unknown clusterset.
func (s *Store) set(name string, create bool) *clusterSet {
	cs := s.clusterSets[name]
	if cs == nil && create {
		cs = &clusterSet{
			clusters:  map[string]store.ClusterRecord{},
			exports:   map[store.ExportKey]store.ExportRecord{},
			endpoints: map[store.ExportKey]store.EndpointSetRecord{},
		}
		s.clusterSets[name] = cs
	}
	return cs
}

// PutCluster implements store.ClusterStore.
func (s *Store) PutCluster(ctx context.Context, rec store.ClusterRecord, now time.Time) (store.ClusterRecord, error) {
	var out store.ClusterRecord
	err := s.write(ctx, func() error {
		cs := s.set(rec.Key.ClusterSet, true)
		stored, exists := cs.clusters[rec.Key.Cluster]
		out = rec.Clone()
		out.Lease = store.Lease{
			Epoch:     store.NextEpoch(stored.Lease.Epoch, now),
			TTL:       rec.Lease.TTL,
			RenewedAt: now,
		}
		out.Generation = stored.Generation + 1
		out.RegisteredAt = now
		if exists {
			out.RegisteredAt = stored.RegisteredAt
		}
		out.UpdatedAt = now
		cs.clusters[rec.Key.Cluster] = out
		out = out.Clone()
		return nil
	})
	return out, err
}

// RenewClusterLease implements store.ClusterStore.
func (s *Store) RenewClusterLease(ctx context.Context, key store.ClusterKey, epoch uint64, now time.Time) (store.ClusterRecord, error) {
	var out store.ClusterRecord
	err := s.write(ctx, func() error {
		cs := s.set(key.ClusterSet, false)
		if cs == nil {
			return store.ErrNotFound
		}
		stored, ok := cs.clusters[key.Cluster]
		if !ok {
			return store.ErrNotFound
		}
		if epoch != stored.Lease.Epoch {
			return store.ErrLeaseEpochMismatch
		}
		stored.Lease.RenewedAt = now
		cs.clusters[key.Cluster] = stored
		out = stored.Clone()
		return nil
	})
	return out, err
}

// GetCluster implements store.ClusterStore.
func (s *Store) GetCluster(ctx context.Context, key store.ClusterKey) (store.ClusterRecord, error) {
	var out store.ClusterRecord
	err := s.read(ctx, func() error {
		cs := s.set(key.ClusterSet, false)
		if cs == nil {
			return store.ErrNotFound
		}
		stored, ok := cs.clusters[key.Cluster]
		if !ok {
			return store.ErrNotFound
		}
		out = stored.Clone()
		return nil
	})
	return out, err
}

// DeleteCluster implements store.ClusterStore.
func (s *Store) DeleteCluster(ctx context.Context, key store.ClusterKey) error {
	return s.write(ctx, func() error {
		cs := s.set(key.ClusterSet, false)
		if cs == nil {
			return nil
		}
		delete(cs.clusters, key.Cluster)
		for k := range cs.exports {
			if k.Cluster == key.Cluster {
				delete(cs.exports, k)
			}
		}
		for k := range cs.endpoints {
			if k.Cluster == key.Cluster {
				delete(cs.endpoints, k)
			}
		}
		return nil
	})
}

// PutExport implements store.ExportStore.
func (s *Store) PutExport(ctx context.Context, rec store.ExportRecord) error {
	return s.write(ctx, func() error {
		cs := s.set(rec.Key.ClusterSet, true)
		if stored, ok := cs.exports[rec.Key]; ok && rec.Generation <= stored.Generation {
			return store.ErrStaleGeneration{Current: stored.Generation}
		}
		cs.exports[rec.Key] = rec.Clone()
		return nil
	})
}

// DeleteExport implements store.ExportStore.
func (s *Store) DeleteExport(ctx context.Context, key store.ExportKey) error {
	return s.write(ctx, func() error {
		if cs := s.set(key.ClusterSet, false); cs != nil {
			delete(cs.exports, key)
			delete(cs.endpoints, key)
		}
		return nil
	})
}

// PutEndpointSet implements store.EndpointStore.
func (s *Store) PutEndpointSet(ctx context.Context, rec store.EndpointSetRecord) error {
	return s.write(ctx, func() error {
		if len(rec.Endpoints) > store.MaxEndpointsPerSet {
			return store.ErrTooLarge
		}
		cs := s.set(rec.Key.ClusterSet, true)
		if stored, ok := cs.endpoints[rec.Key]; ok && rec.Generation <= stored.Generation {
			return store.ErrStaleGeneration{Current: stored.Generation}
		}
		cs.endpoints[rec.Key] = rec.Clone()
		return nil
	})
}

// DeleteEndpointSet implements store.EndpointStore.
func (s *Store) DeleteEndpointSet(ctx context.Context, key store.ExportKey) error {
	return s.write(ctx, func() error {
		if cs := s.set(key.ClusterSet, false); cs != nil {
			delete(cs.endpoints, key)
		}
		return nil
	})
}

// ListClusterSets implements store.Store.
func (s *Store) ListClusterSets(ctx context.Context) ([]string, error) {
	var out []string
	err := s.read(ctx, func() error {
		out = make([]string, 0, len(s.clusterSets))
		for name := range s.clusterSets {
			out = append(out, name)
		}
		sort.Strings(out)
		return nil
	})
	return out, err
}

// LoadClusterSet implements store.Store.
func (s *Store) LoadClusterSet(ctx context.Context, clusterSet string) (store.ClusterSetSnapshot, error) {
	snap := store.ClusterSetSnapshot{ClusterSet: clusterSet}
	err := s.read(ctx, func() error {
		cs := s.set(clusterSet, false)
		if cs == nil {
			return nil
		}
		for _, r := range cs.clusters {
			snap.Clusters = append(snap.Clusters, r.Clone())
		}
		for _, r := range cs.exports {
			snap.Exports = append(snap.Exports, r.Clone())
		}
		for _, r := range cs.endpoints {
			snap.EndpointSets = append(snap.EndpointSets, r.Clone())
		}
		return nil
	})
	if err != nil {
		return store.ClusterSetSnapshot{}, err
	}
	snap.Sort()
	return snap, nil
}

// Ping implements store.Store. It fails only after Close.
func (s *Store) Ping(ctx context.Context) error {
	return s.read(ctx, func() error { return nil })
}

// Close implements store.Store. Later calls return store.ErrUnavailable.
// Close is idempotent.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.clusterSets = map[string]*clusterSet{}
	return nil
}
