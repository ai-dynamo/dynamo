// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"sync"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// clusterSet is the state of one clusterset. Every field is guarded by mu.
type clusterSet struct {
	mu   sync.Mutex
	name string

	clusters  map[string]store.ClusterRecord
	exports   map[store.ExportKey]store.ExportRecord
	endpoints map[store.ExportKey]store.EndpointSetRecord

	local localWrites

	clusterLease map[string]registry.LeaseState

	objects    [numKinds]map[string]*Object
	importMemo map[string]memoEntry
	epCache    map[store.ExportKey]endpointCache
	index      indexes

	seq  uint64
	ring *ring
	subs map[*subscriber]struct{}

	discovered    bool
	refreshed     bool
	lastRefresh   time.Time
	lastLoadStart time.Time
}

func newClusterSet(name string, history int) *clusterSet {
	cs := &clusterSet{
		name:         name,
		clusters:     map[string]store.ClusterRecord{},
		exports:      map[store.ExportKey]store.ExportRecord{},
		endpoints:    map[store.ExportKey]store.EndpointSetRecord{},
		local:        newLocalWrites(),
		clusterLease: map[string]registry.LeaseState{},
		importMemo:   map[string]memoEntry{},
		epCache:      map[store.ExportKey]endpointCache{},
		ring:         newRing(history),
		subs:         map[*subscriber]struct{}{},
	}
	for k := range cs.objects {
		cs.objects[k] = map[string]*Object{}
	}
	return cs
}

func (cs *clusterSet) setDiscovered(d bool) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.discovered = d
}

// idle reports whether the clusterset can be dropped: no records, no
// watcher and no local write within window.
func (cs *clusterSet) idle(now time.Time, window time.Duration) bool {
	return len(cs.clusters) == 0 && len(cs.exports) == 0 && len(cs.endpoints) == 0 && len(cs.subs) == 0 &&
		!cs.local.recent(now.Add(-window))
}

// mark records a local write or delete of one record.
type mark[R any] struct {
	at      time.Time
	deleted bool
	// prev is the version a local delete removed, nil when unknown.
	prev *R
}

// localWrites tracks the records written through this replica recently.
type localWrites struct {
	clusters  map[string]mark[store.ClusterRecord]
	exports   map[store.ExportKey]mark[store.ExportRecord]
	endpoints map[store.ExportKey]mark[store.EndpointSetRecord]
	// clusterDeletes are cascading cluster deletions.
	clusterDeletes map[string]time.Time
}

func newLocalWrites() localWrites {
	return localWrites{
		clusters:       map[string]mark[store.ClusterRecord]{},
		exports:        map[store.ExportKey]mark[store.ExportRecord]{},
		endpoints:      map[store.ExportKey]mark[store.EndpointSetRecord]{},
		clusterDeletes: map[string]time.Time{},
	}
}

func (l *localWrites) recent(cutoff time.Time) bool {
	for _, m := range l.clusters {
		if m.at.After(cutoff) {
			return true
		}
	}
	for _, m := range l.exports {
		if m.at.After(cutoff) {
			return true
		}
	}
	for _, m := range l.endpoints {
		if m.at.After(cutoff) {
			return true
		}
	}
	for _, at := range l.clusterDeletes {
		if at.After(cutoff) {
			return true
		}
	}
	return false
}

// Version order of each record type: newer reports whether a is a later
// version of the same record than b.

func newerCluster(a, b store.ClusterRecord) bool {
	if a.Lease.Epoch != b.Lease.Epoch {
		return a.Lease.Epoch > b.Lease.Epoch
	}
	if !a.Lease.RenewedAt.Equal(b.Lease.RenewedAt) {
		return a.Lease.RenewedAt.After(b.Lease.RenewedAt)
	}
	return a.UpdatedAt.After(b.UpdatedAt)
}

func newerExport(a, b store.ExportRecord) bool { return a.Generation > b.Generation }

func newerEndpointSet(a, b store.EndpointSetRecord) bool { return a.Generation > b.Generation }

// reconcile merges the local marks into the snapshot records next, given the
// current records cur: within the window a local write wins unless the
// snapshot holds a newer version, and a local delete wins unless the
// snapshot holds a version newer than the one deleted. Marks older than
// cutoff are dropped.
func reconcile[K comparable, R any](next, cur map[K]R, marks map[K]mark[R], cutoff time.Time, newer func(a, b R) bool) {
	for k, m := range marks {
		if !m.at.After(cutoff) {
			delete(marks, k)
			continue
		}
		snapRec, inSnap := next[k]
		if m.deleted {
			if inSnap && m.prev != nil && newer(snapRec, *m.prev) {
				continue
			}
			delete(next, k)
			continue
		}
		local, ok := cur[k]
		if !ok {
			continue
		}
		if !inSnap || newer(local, snapRec) {
			next[k] = local
		}
	}
}

// replace swaps in the records of a snapshot, keeping recent local writes.
func (cs *clusterSet) replace(snap store.ClusterSetSnapshot, cutoff time.Time) {
	clusters := make(map[string]store.ClusterRecord, len(snap.Clusters))
	for _, r := range snap.Clusters {
		clusters[r.Key.Cluster] = r
	}
	exports := make(map[store.ExportKey]store.ExportRecord, len(snap.Exports))
	for _, r := range snap.Exports {
		exports[r.Key] = r
	}
	endpoints := make(map[store.ExportKey]store.EndpointSetRecord, len(snap.EndpointSets))
	for _, r := range snap.EndpointSets {
		endpoints[r.Key] = r
	}

	// A recent cascading cluster delete hides the cluster's records from a
	// snapshot that may predate it, except records written locally after
	// the delete.
	for cluster, at := range cs.local.clusterDeletes {
		if !at.After(cutoff) {
			delete(cs.local.clusterDeletes, cluster)
			continue
		}
		if m, ok := cs.local.clusters[cluster]; !ok || !m.at.After(at) || m.deleted {
			delete(clusters, cluster)
		}
		for k := range exports {
			if m, ok := cs.local.exports[k]; k.Cluster == cluster && (!ok || !m.at.After(at) || m.deleted) {
				delete(exports, k)
			}
		}
		for k := range endpoints {
			if m, ok := cs.local.endpoints[k]; k.Cluster == cluster && (!ok || !m.at.After(at) || m.deleted) {
				delete(endpoints, k)
			}
		}
	}

	reconcile(clusters, cs.clusters, cs.local.clusters, cutoff, newerCluster)
	reconcile(exports, cs.exports, cs.local.exports, cutoff, newerExport)
	reconcile(endpoints, cs.endpoints, cs.local.endpoints, cutoff, newerEndpointSet)

	cs.clusters, cs.exports, cs.endpoints = clusters, exports, endpoints
}

// leaseStates evaluates every cluster lease at now.
func (cs *clusterSet) leaseStates(now time.Time, skew time.Duration) map[string]registry.LeaseState {
	l := make(map[string]registry.LeaseState, len(cs.clusters))
	for name, r := range cs.clusters {
		l[name] = registry.EvaluateLease(r.Lease.RenewedAt, r.Lease.TTL, now, skew)
	}
	return l
}

// leasesChanged reports whether any lease state differs from the last
// evaluation.
func (cs *clusterSet) leasesChanged(now time.Time, skew time.Duration) bool {
	return !mapsEqual(cs.leaseStates(now, skew), cs.clusterLease)
}

func mapsEqual[K comparable, V comparable](a, b map[K]V) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return false
		}
	}
	return true
}
