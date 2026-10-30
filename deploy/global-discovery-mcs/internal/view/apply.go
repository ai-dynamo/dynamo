// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"fmt"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// The Apply methods are the write-through ("ApplyLocal") path: an API
// handler calls one after each successful store write with the record the
// store returned, so the handling replica serves the write at once. Each
// call derives the affected objects and publishes the changes to watchers.
//
// The store orders the writes of one key by version (lease epoch and
// renewal for clusters, generation for exports and endpoint sets), but the
// view application after the write is not atomic with it: two handlers for
// one key can reach the view in reverse order, and a refresh can load the
// newer version first. Each Apply therefore compares the record with the
// held one under the clusterset lock and drops an older version, using the
// same ordering the snapshot path (clusterSet.replace) uses, so the view and
// the watch stream never move backwards.

// ApplyLocal applies records that a handler stored through this replica, in
// order: store.ClusterRecord, store.ExportRecord or store.EndpointSetRecord.
// It returns an error for any other type. Deletions use the typed
// Apply*Delete methods.
func (v *View) ApplyLocal(records ...any) error {
	for _, r := range records {
		switch rec := r.(type) {
		case store.ClusterRecord:
			v.ApplyCluster(rec)
		case store.ExportRecord:
			v.ApplyExport(rec)
		case store.EndpointSetRecord:
			v.ApplyEndpointSet(rec)
		default:
			return fmt.Errorf("view: cannot apply %T", r)
		}
	}
	return nil
}

// apply runs fn on the clusterset under its lock and recomputes it when fn
// reports a change.
func (v *View) apply(clusterSet string, fn func(cs *clusterSet) bool) {
	cs := v.ensure(clusterSet)
	now := v.clock.Now()
	cs.mu.Lock()
	defer cs.mu.Unlock()
	if fn(cs) {
		v.recompute(cs, now)
	}
}

// applyDelete runs fn on the clusterset under its lock and recomputes it.
func (v *View) applyDelete(name string, fn func(cs *clusterSet)) {
	v.apply(name, func(cs *clusterSet) bool {
		fn(cs)
		return true
	})
}

// put stores rec under key unless the held record is a newer version, and
// marks it as a local write. It reports whether rec was stored.
func put[K comparable, R any](recs map[K]R, marks map[K]mark[R], key K, rec R, now time.Time, newer func(a, b R) bool) bool {
	if held, ok := recs[key]; ok && newer(held, rec) {
		return false
	}
	recs[key] = rec
	marks[key] = mark[R]{at: now}
	return true
}

// ApplyCluster applies a registered or renewed cluster. A version older
// than the held one (lower lease epoch, or an earlier renewal of the same
// epoch) is ignored.
func (v *View) ApplyCluster(rec store.ClusterRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) bool {
		return put(cs.clusters, cs.local.clusters, rec.Key.Cluster, rec.Clone(), v.clock.Now(), newerCluster)
	})
}

// ApplyClusterDelete applies a cluster deletion and its cascade to the
// cluster's exports and endpoint sets.
func (v *View) ApplyClusterDelete(key store.ClusterKey) {
	v.applyDelete(key.ClusterSet, func(cs *clusterSet) {
		now := v.clock.Now()
		if prev, ok := cs.clusters[key.Cluster]; ok {
			cs.local.clusters[key.Cluster] = mark[store.ClusterRecord]{at: now, deleted: true, prev: &prev}
		} else {
			cs.local.clusters[key.Cluster] = mark[store.ClusterRecord]{at: now, deleted: true}
		}
		delete(cs.clusters, key.Cluster)
		for k := range cs.exports {
			if k.Cluster == key.Cluster {
				delete(cs.exports, k)
				delete(cs.local.exports, k)
			}
		}
		for k := range cs.endpoints {
			if k.Cluster == key.Cluster {
				delete(cs.endpoints, k)
				delete(cs.local.endpoints, k)
			}
		}
		cs.local.clusterDeletes[key.Cluster] = now
	})
}

// ApplyExport applies a stored export. A generation older than the held
// one is ignored.
func (v *View) ApplyExport(rec store.ExportRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) bool {
		return put(cs.exports, cs.local.exports, rec.Key, rec.Clone(), v.clock.Now(), newerExport)
	})
}

// ApplyExportDelete applies an export deletion and its endpoint set's.
func (v *View) ApplyExportDelete(key store.ExportKey) {
	v.applyDelete(key.ClusterSet, func(cs *clusterSet) {
		now := v.clock.Now()
		cs.local.exports[key] = deleteMark(cs.exports, key, now)
		cs.local.endpoints[key] = deleteMark(cs.endpoints, key, now)
		delete(cs.exports, key)
		delete(cs.endpoints, key)
	})
}

// ApplyEndpointSet applies a stored endpoint set. A generation older than
// the held one is ignored.
func (v *View) ApplyEndpointSet(rec store.EndpointSetRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) bool {
		return put(cs.endpoints, cs.local.endpoints, rec.Key, rec.Clone(), v.clock.Now(), newerEndpointSet)
	})
}

// ApplyEndpointSetDelete applies an endpoint set deletion.
func (v *View) ApplyEndpointSetDelete(key store.ExportKey) {
	v.applyDelete(key.ClusterSet, func(cs *clusterSet) {
		cs.local.endpoints[key] = deleteMark(cs.endpoints, key, v.clock.Now())
		delete(cs.endpoints, key)
	})
}

func deleteMark[K comparable, R any](recs map[K]R, key K, now time.Time) mark[R] {
	if prev, ok := recs[key]; ok {
		return mark[R]{at: now, deleted: true, prev: &prev}
	}
	return mark[R]{at: now, deleted: true}
}
