// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package store

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestNextEpoch(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	ms := uint64(now.UnixMilli())
	tests := []struct {
		name   string
		stored uint64
		now    time.Time
		want   uint64
	}{
		{"first registration takes the clock", 0, now, ms},
		{"clock ahead of stored", ms - 5000, now, ms},
		{"same millisecond", ms, now, ms + 1},
		{"clock behind stored", ms + 60000, now, ms + 60001},
		{"zero time", 0, time.Time{}, 1},
		{"before the unix epoch", 9, time.Unix(-10, 0), 10},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, NextEpoch(tt.stored, tt.now), tt.name)
	}
}

func TestLeaseExpiresAt(t *testing.T) {
	renewed := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	l := Lease{Epoch: 1, TTL: 30 * time.Second, RenewedAt: renewed}
	assert.Equal(t, renewed.Add(30*time.Second), l.ExpiresAt())
}

func TestKeyHelpers(t *testing.T) {
	ek := ExportKey{ClusterSet: "cs", Cluster: "c1", Namespace: "ns", Name: "svc"}
	assert.Equal(t, ClusterKey{ClusterSet: "cs", Cluster: "c1"}, ek.ClusterKey())
}

func TestSnapshotSort(t *testing.T) {
	snap := ClusterSetSnapshot{
		Clusters: []ClusterRecord{{Key: ClusterKey{"cs", "b"}}, {Key: ClusterKey{"cs", "a"}}},
		Exports: []ExportRecord{
			{Key: ExportKey{"cs", "b", "ns", "a"}},
			{Key: ExportKey{"cs", "a", "ns", "b"}},
			{Key: ExportKey{"cs", "a", "ns", "a"}},
			{Key: ExportKey{"cs", "a", "a", "z"}},
		},
		EndpointSets: []EndpointSetRecord{{Key: ExportKey{"cs", "b", "ns", "a"}}, {Key: ExportKey{"cs", "a", "ns", "a"}}},
	}
	snap.Sort()

	assert.Equal(t, []ClusterKey{{"cs", "a"}, {"cs", "b"}}, []ClusterKey{snap.Clusters[0].Key, snap.Clusters[1].Key})
	var exports []ExportKey
	for _, e := range snap.Exports {
		exports = append(exports, e.Key)
	}
	assert.Equal(t, []ExportKey{{"cs", "a", "a", "z"}, {"cs", "a", "ns", "a"}, {"cs", "a", "ns", "b"}, {"cs", "b", "ns", "a"}}, exports)
	assert.Equal(t, "a", snap.EndpointSets[0].Key.Cluster)
	assert.Negative(t, CompareClusterKeys(ClusterKey{"a", "z"}, ClusterKey{"b", "a"}))
	assert.Zero(t, CompareExportKeys(ExportKey{"a", "b", "c", "d"}, ExportKey{"a", "b", "c", "d"}))
	assert.Positive(t, CompareExportKeys(ExportKey{"a", "b", "c", "e"}, ExportKey{"a", "b", "c", "d"}))
}

func TestCloneIsDeep(t *testing.T) {
	cluster := ClusterRecord{Zones: []string{"z"}, Labels: map[string]string{"k": "v"}}
	cc := cluster.Clone()
	cc.Zones[0], cc.Labels["k"] = "x", "x"
	assert.Equal(t, "z", cluster.Zones[0])
	assert.Equal(t, "v", cluster.Labels["k"])

	export := ExportRecord{Spec: ExportSpec{
		Annotations:           map[string]string{"a": "1"},
		Labels:                map[string]string{"l": "1"},
		Ports:                 []ServicePort{{Port: 1}},
		SessionAffinityConfig: &SessionAffinityConfig{ClientIP: &ClientIPConfig{TimeoutSeconds: 1}},
		IPFamilies:            []IPFamily{IPFamilyIPv4},
	}}
	ec := export.Clone()
	ec.Spec.Annotations["a"], ec.Spec.Labels["l"] = "x", "x"
	ec.Spec.Ports[0].Port = 2
	ec.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds = 2
	ec.Spec.IPFamilies[0] = IPFamilyIPv6
	assert.Equal(t, "1", export.Spec.Annotations["a"])
	assert.Equal(t, "1", export.Spec.Labels["l"])
	assert.Equal(t, int32(1), export.Spec.Ports[0].Port)
	assert.Equal(t, int32(1), export.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
	assert.Equal(t, IPFamilyIPv4, export.Spec.IPFamilies[0])

	noClientIP := ExportSpec{SessionAffinityConfig: &SessionAffinityConfig{}}.Clone()
	assert.Nil(t, noClientIP.SessionAffinityConfig.ClientIP)

	eps := EndpointSetRecord{Endpoints: []Endpoint{{Addresses: []string{"a"}, Ports: []EndpointPort{{Port: 1}}, Attributes: map[string]string{"k": "v"}}}}
	epc := eps.Clone()
	epc.Endpoints[0].Addresses[0] = "x"
	epc.Endpoints[0].Ports[0].Port = 2
	epc.Endpoints[0].Attributes["k"] = "x"
	assert.Equal(t, Endpoint{Addresses: []string{"a"}, Ports: []EndpointPort{{Port: 1}}, Attributes: map[string]string{"k": "v"}}, eps.Endpoints[0])
	assert.Nil(t, EndpointSetRecord{}.Clone().Endpoints)
}
