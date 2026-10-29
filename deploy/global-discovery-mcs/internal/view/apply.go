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

// apply runs fn on the clusterset under its lock and recomputes it.
func (v *View) apply(clusterSet string, fn func(cs *clusterSet)) {
	cs := v.ensure(clusterSet)
	now := v.clock.Now()
	cs.mu.Lock()
	defer cs.mu.Unlock()
	fn(cs)
	v.recompute(cs, now)
}

// ApplyCluster applies a registered or renewed cluster.
func (v *View) ApplyCluster(rec store.ClusterRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) {
		cs.clusters[rec.Key.Cluster] = rec.Clone()
		cs.local.clusters[rec.Key.Cluster] = mark[store.ClusterRecord]{at: v.clock.Now()}
	})
}

// ApplyClusterDelete applies a cluster deletion and its cascade to the
// cluster's exports and endpoint sets.
func (v *View) ApplyClusterDelete(key store.ClusterKey) {
	v.apply(key.ClusterSet, func(cs *clusterSet) {
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

// ApplyExport applies a stored export.
func (v *View) ApplyExport(rec store.ExportRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) {
		cs.exports[rec.Key] = rec.Clone()
		cs.local.exports[rec.Key] = mark[store.ExportRecord]{at: v.clock.Now()}
	})
}

// ApplyExportDelete applies an export deletion and its endpoint set's.
func (v *View) ApplyExportDelete(key store.ExportKey) {
	v.apply(key.ClusterSet, func(cs *clusterSet) {
		now := v.clock.Now()
		cs.local.exports[key] = deleteMark(cs.exports, key, now)
		cs.local.endpoints[key] = deleteMark(cs.endpoints, key, now)
		delete(cs.exports, key)
		delete(cs.endpoints, key)
	})
}

// ApplyEndpointSet applies a stored endpoint set.
func (v *View) ApplyEndpointSet(rec store.EndpointSetRecord) {
	v.apply(rec.Key.ClusterSet, func(cs *clusterSet) {
		cs.endpoints[rec.Key] = rec.Clone()
		cs.local.endpoints[rec.Key] = mark[store.EndpointSetRecord]{at: v.clock.Now()}
	})
}

// ApplyEndpointSetDelete applies an endpoint set deletion.
func (v *View) ApplyEndpointSetDelete(key store.ExportKey) {
	v.apply(key.ClusterSet, func(cs *clusterSet) {
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
