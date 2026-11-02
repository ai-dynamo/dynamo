// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"cmp"
	"slices"
	"time"
)

// NextEpoch returns the lease epoch a registration mints:
// max(stored + 1, now in unix milliseconds). Epochs therefore increase across
// re-registrations even when the clock of the handling replica is behind.
func NextEpoch(stored uint64, now time.Time) uint64 {
	next := stored + 1
	if ms := now.UnixMilli(); ms > 0 && uint64(ms) > next {
		next = uint64(ms)
	}
	return next
}

// Sort orders every slice of the snapshot by key.
func (s *ClusterSetSnapshot) Sort() {
	slices.SortFunc(s.Clusters, func(a, b ClusterRecord) int { return CompareClusterKeys(a.Key, b.Key) })
	slices.SortFunc(s.Exports, func(a, b ExportRecord) int { return CompareExportKeys(a.Key, b.Key) })
	slices.SortFunc(s.EndpointSets, func(a, b EndpointSetRecord) int { return CompareExportKeys(a.Key, b.Key) })
}

// CompareClusterKeys orders cluster keys by clusterset, then cluster.
func CompareClusterKeys(a, b ClusterKey) int {
	return cmp.Or(cmp.Compare(a.ClusterSet, b.ClusterSet), cmp.Compare(a.Cluster, b.Cluster))
}

// CompareExportKeys orders export keys by clusterset, cluster, namespace,
// then name.
func CompareExportKeys(a, b ExportKey) int {
	return cmp.Or(
		cmp.Compare(a.ClusterSet, b.ClusterSet),
		cmp.Compare(a.Cluster, b.Cluster),
		cmp.Compare(a.Namespace, b.Namespace),
		cmp.Compare(a.Name, b.Name),
	)
}
