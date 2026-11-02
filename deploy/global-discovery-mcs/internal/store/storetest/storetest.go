// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package storetest is the conformance suite every store.Store
// implementation must pass. An implementation's test calls Run with a
// factory:
//
//	func TestConformance(t *testing.T) {
//		storetest.Run(t, func(t *testing.T) store.Store { return memory.New() })
//	}
//
// Every subtest takes a new store from the factory and uses clusterset names
// derived from its own name, so a factory may hand out stores that share a
// backend (for example one DynamoDB table). The suite closes each store when
// its subtest ends.
//
// Records must round-trip exactly, except that empty and nil maps and slices
// are equivalent and times may come back in UTC.
package storetest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Factory returns a new store for one subtest.
type Factory func(t *testing.T) store.Store

// t0 is the base server time of every test; it has no monotonic reading.
var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

// Run runs the conformance suite against the stores newStore returns.
func Run(t *testing.T, newStore Factory) {
	tests := []struct {
		name string
		fn   func(t *testing.T, st store.Store, cs string)
	}{
		{"ClusterRegisterNew", clusterRegisterNew},
		{"ClusterReregisterMintsNewEpoch", clusterReregisterMintsNewEpoch},
		{"ClusterReregisterSameMillisecond", clusterReregisterSameMillisecond},
		{"ClusterReregisterClockBehind", clusterReregisterClockBehind},
		{"ClusterHeartbeat", clusterHeartbeat},
		{"ClusterHeartbeatEpochMismatch", clusterHeartbeatEpochMismatch},
		{"ClusterHeartbeatUnknown", clusterHeartbeatUnknown},
		{"ClusterGetUnknown", clusterGetUnknown},
		{"ClusterRecordsAreCopies", clusterRecordsAreCopies},
		{"ClusterDeleteCascades", clusterDeleteCascades},
		{"ClusterDeleteIdempotent", clusterDeleteIdempotent},
		{"ClusterDeleteUnregisteredRemovesRecords", clusterDeleteUnregistered},
		{"ExportPutAndLoad", exportPutAndLoad},
		{"ExportStaleGeneration", exportStaleGeneration},
		{"ExportNewerGenerationReplaces", exportNewerGenerationReplaces},
		{"ExportDeleteCascadesEndpointSet", exportDeleteCascades},
		{"ExportDeleteIdempotent", exportDeleteIdempotent},
		{"ExportDeleteForgetsGeneration", exportDeleteForgetsGeneration},
		{"ExportKeysAreIndependent", exportKeysAreIndependent},
		{"EndpointSetReplace", endpointSetReplace},
		{"EndpointSetStaleGeneration", endpointSetStaleGeneration},
		{"EndpointSetTooLarge", endpointSetTooLarge},
		{"EndpointSetEmptyAndOrphan", endpointSetEmptyAndOrphan},
		{"EndpointSetDeleteIdempotent", endpointSetDeleteIdempotent},
		{"LoadClusterSetComplete", loadClusterSetComplete},
		{"LoadClusterSetUnknown", loadClusterSetUnknown},
		{"LoadClusterSetReturnsCopies", loadClusterSetReturnsCopies},
		{"ListClusterSets", listClusterSets},
		{"Ping", ping},
		{"CanceledContext", canceledContext},
		{"ConcurrentRegistrations", concurrentRegistrations},
		{"ConcurrentGenerations", concurrentGenerations},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			st := newStore(t)
			t.Cleanup(func() { _ = st.Close() })
			tt.fn(t, st, clusterSetName(t, "a"))
		})
	}
}

// clusterSetName returns a DNS label unique to the running subtest.
func clusterSetName(t *testing.T, suffix string) string {
	sum := sha256.Sum256([]byte(t.Name()))
	return "cs-" + hex.EncodeToString(sum[:])[:12] + "-" + suffix
}

func ctx(t *testing.T) context.Context {
	return t.Context()
}

// Fixtures.

func clusterRecord(cs, cluster string) store.ClusterRecord {
	return store.ClusterRecord{
		Key:          store.ClusterKey{ClusterSet: cs, Cluster: cluster},
		Region:       "us-east-1",
		Zones:        []string{"us-east-1a", "us-east-1b"},
		Provider:     "eks",
		Labels:       map[string]string{"env": "test"},
		AgentID:      "mcs-controller-0",
		AgentVersion: "v0.1.0",
		Lease:        store.Lease{TTL: 30 * time.Second},
	}
}

func exportKey(cs, cluster, ns, name string) store.ExportKey {
	return store.ExportKey{ClusterSet: cs, Cluster: cluster, Namespace: ns, Name: name}
}

func exportRecord(key store.ExportKey, gen uint64) store.ExportRecord {
	return store.ExportRecord{
		Key: key,
		Spec: store.ExportSpec{
			Annotations:             map[string]string{"mcs.dynamo.nvidia.com/dial-scheme": "https"},
			Labels:                  map[string]string{"app": "router"},
			ExportCreationTimestamp: t0.Add(-time.Hour),
			Ports: []store.ServicePort{
				{Name: "grpc", Protocol: store.ProtocolTCP, AppProtocol: "grpc", Port: 50071},
				{Name: "quic", Protocol: store.ProtocolUDP, Port: 50072},
			},
			Type:                  store.ServiceTypeClusterSetIP,
			SessionAffinity:       store.SessionAffinityClientIP,
			SessionAffinityConfig: &store.SessionAffinityConfig{ClientIP: &store.ClientIPConfig{TimeoutSeconds: 10800}},
			IPFamilies:            []store.IPFamily{store.IPFamilyIPv4},
			InternalTrafficPolicy: store.InternalTrafficPolicyCluster,
			TrafficDistribution:   "PreferClose",
		},
		Generation: gen,
		UpdatedAt:  t0,
	}
}

func endpointSetRecord(key store.ExportKey, gen uint64, n int) store.EndpointSetRecord {
	rec := store.EndpointSetRecord{Key: key, Generation: gen, UpdatedAt: t0}
	for i := range n {
		rec.Endpoints = append(rec.Endpoints, store.Endpoint{
			Addresses:   []string{fmt.Sprintf("10.%d.%d.%d", i/65536%256, i/256%256, i%256)},
			AddressType: store.AddressTypeIPv4,
			Conditions:  store.EndpointConditions{Ready: i%2 == 0, Serving: true, Terminating: i%2 == 1},
			Hostname:    fmt.Sprintf("pod-%d", i),
			NodeName:    "node-1",
			Zone:        "us-east-1a",
			Ports:       []store.EndpointPort{{Name: "grpc", Protocol: store.ProtocolTCP, AppProtocol: "grpc", Port: 50071}},
			Attributes:  map[string]string{"pod": fmt.Sprintf("pod-%d", i)},
		})
	}
	return rec
}

func ms(t time.Time) uint64 {
	return uint64(t.UnixMilli())
}

// Comparison.

// normalize returns a deep copy of v with empty maps and slices set to nil
// and times converted to UTC without a monotonic reading.
func normalize[T any](v T) T {
	rv := reflect.ValueOf(&v).Elem()
	normalizeValue(rv)
	return v
}

var timeType = reflect.TypeFor[time.Time]()

func normalizeValue(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer:
		if !v.IsNil() {
			cp := reflect.New(v.Elem().Type())
			cp.Elem().Set(v.Elem())
			normalizeValue(cp.Elem())
			v.Set(cp)
		}
	case reflect.Struct:
		if v.Type() == timeType {
			v.Set(reflect.ValueOf(v.Interface().(time.Time).UTC().Round(0)))
			return
		}
		for _, field := range v.Fields() {
			if field.CanSet() {
				normalizeValue(field)
			}
		}
	case reflect.Slice:
		if v.Len() == 0 {
			v.Set(reflect.Zero(v.Type()))
			return
		}
		cp := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		reflect.Copy(cp, v)
		for i := range cp.Len() {
			normalizeValue(cp.Index(i))
		}
		v.Set(cp)
	case reflect.Map:
		if v.Len() == 0 {
			v.Set(reflect.Zero(v.Type()))
			return
		}
		cp := reflect.MakeMapWithSize(v.Type(), v.Len())
		iter := v.MapRange()
		for iter.Next() {
			val := reflect.New(iter.Value().Type()).Elem()
			val.Set(iter.Value())
			normalizeValue(val)
			cp.SetMapIndex(iter.Key(), val)
		}
		v.Set(cp)
	}
}

func assertRecord[T any](t *testing.T, want, got T, msgAndArgs ...any) bool {
	t.Helper()
	return assert.Equal(t, normalize(want), normalize(got), msgAndArgs...)
}

func load(t *testing.T, st store.Store, cs string) store.ClusterSetSnapshot {
	t.Helper()
	snap, err := st.LoadClusterSet(ctx(t), cs)
	require.NoError(t, err)
	return snap
}

func requireStale(t *testing.T, err error, current uint64) {
	t.Helper()
	require.Error(t, err)
	assert.ErrorIs(t, err, store.ErrStaleGeneration{})
	got, ok := store.StaleGeneration(err)
	require.True(t, ok, "error %v is not ErrStaleGeneration", err)
	assert.Equal(t, current, got)
}

// Clusters.

func clusterRegisterNew(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	got, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)

	want := rec.Clone()
	want.Lease = store.Lease{Epoch: ms(t0), TTL: 30 * time.Second, RenewedAt: t0}
	want.Generation = 1
	want.RegisteredAt = t0
	want.UpdatedAt = t0
	assertRecord(t, want, got)

	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assertRecord(t, want, stored)
}

func clusterReregisterMintsNewEpoch(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	first, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)

	rec.Region = "us-west-2"
	rec.Labels = map[string]string{"env": "prod"}
	rec.Lease.TTL = time.Minute
	t1 := t0.Add(5 * time.Second)
	second, err := st.PutCluster(ctx(t), rec, t1)
	require.NoError(t, err)

	assert.Equal(t, ms(t1), second.Lease.Epoch)
	assert.Greater(t, second.Lease.Epoch, first.Lease.Epoch)
	assert.Equal(t, time.Minute, second.Lease.TTL)
	assert.True(t, second.Lease.RenewedAt.Equal(t1))
	assert.Equal(t, uint64(2), second.Generation)
	assert.True(t, second.RegisteredAt.Equal(t0), "RegisteredAt is kept")
	assert.True(t, second.UpdatedAt.Equal(t1))
	assert.Equal(t, "us-west-2", second.Region)
	assert.Equal(t, map[string]string{"env": "prod"}, second.Labels)

	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assertRecord(t, second, stored)
}

func clusterReregisterSameMillisecond(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	first, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)
	second, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)
	assert.Equal(t, first.Lease.Epoch+1, second.Lease.Epoch)
}

func clusterReregisterClockBehind(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	first, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)
	second, err := st.PutCluster(ctx(t), rec, t0.Add(-time.Minute))
	require.NoError(t, err)
	assert.Equal(t, first.Lease.Epoch+1, second.Lease.Epoch)
}

func clusterHeartbeat(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	reg, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)

	t1 := t0.Add(10 * time.Second)
	got, err := st.RenewClusterLease(ctx(t), rec.Key, reg.Lease.Epoch, t1)
	require.NoError(t, err)

	want := reg.Clone()
	want.Lease.RenewedAt = t1
	assertRecord(t, want, got, "a heartbeat changes only Lease.RenewedAt")

	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assertRecord(t, want, stored)
}

func clusterHeartbeatEpochMismatch(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	first, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)
	second, err := st.PutCluster(ctx(t), rec, t0.Add(time.Second))
	require.NoError(t, err)

	for _, epoch := range []uint64{0, first.Lease.Epoch, second.Lease.Epoch + 1} {
		_, err = st.RenewClusterLease(ctx(t), rec.Key, epoch, t0.Add(2*time.Second))
		assert.ErrorIs(t, err, store.ErrLeaseEpochMismatch, "epoch %d", epoch)
	}

	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assertRecord(t, second, stored, "a rejected heartbeat changes nothing")
}

func clusterHeartbeatUnknown(t *testing.T, st store.Store, cs string) {
	_, err := st.RenewClusterLease(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "nope"}, 1, t0)
	assert.ErrorIs(t, err, store.ErrNotFound)

	_, err = st.PutCluster(ctx(t), clusterRecord(cs, "c1"), t0)
	require.NoError(t, err)
	_, err = st.RenewClusterLease(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "nope"}, ms(t0), t0)
	assert.ErrorIs(t, err, store.ErrNotFound)
}

func clusterGetUnknown(t *testing.T, st store.Store, cs string) {
	_, err := st.GetCluster(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "c1"})
	assert.ErrorIs(t, err, store.ErrNotFound)

	_, err = st.PutCluster(ctx(t), clusterRecord(cs, "c1"), t0)
	require.NoError(t, err)
	_, err = st.GetCluster(ctx(t), store.ClusterKey{ClusterSet: clusterSetName(t, "other"), Cluster: "c1"})
	assert.ErrorIs(t, err, store.ErrNotFound, "clustersets are separate")
}

func clusterRecordsAreCopies(t *testing.T, st store.Store, cs string) {
	rec := clusterRecord(cs, "c1")
	got, err := st.PutCluster(ctx(t), rec, t0)
	require.NoError(t, err)

	rec.Labels["env"] = "mutated-input"
	rec.Zones[0] = "mutated-input"
	got.Labels["env"] = "mutated-output"
	got.Zones[0] = "mutated-output"

	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assert.Equal(t, "test", stored.Labels["env"])
	assert.Equal(t, "us-east-1a", stored.Zones[0])

	stored.Labels["env"] = "mutated-get"
	again, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assert.Equal(t, "test", again.Labels["env"])
}

// populateCluster writes a registration and two exports with endpoint sets
// for one cluster.
func populateCluster(t *testing.T, st store.Store, cs, cluster string) {
	t.Helper()
	_, err := st.PutCluster(ctx(t), clusterRecord(cs, cluster), t0)
	require.NoError(t, err)
	for _, name := range []string{"router", "frontend"} {
		key := exportKey(cs, cluster, "llm", name)
		require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 1)))
		require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, 2)))
	}
}

func clusterDeleteCascades(t *testing.T, st store.Store, cs string) {
	other := clusterSetName(t, "b")
	populateCluster(t, st, cs, "c1")
	populateCluster(t, st, cs, "c2")
	populateCluster(t, st, other, "c1")
	before := load(t, st, cs)
	otherBefore := load(t, st, other)

	require.NoError(t, st.DeleteCluster(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "c1"}))

	_, err := st.GetCluster(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "c1"})
	assert.ErrorIs(t, err, store.ErrNotFound)
	_, err = st.RenewClusterLease(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "c1"}, ms(t0), t0)
	assert.ErrorIs(t, err, store.ErrNotFound)

	after := load(t, st, cs)
	want := store.ClusterSetSnapshot{ClusterSet: cs}
	for _, r := range before.Clusters {
		if r.Key.Cluster != "c1" {
			want.Clusters = append(want.Clusters, r)
		}
	}
	for _, r := range before.Exports {
		if r.Key.Cluster != "c1" {
			want.Exports = append(want.Exports, r)
		}
	}
	for _, r := range before.EndpointSets {
		if r.Key.Cluster != "c1" {
			want.EndpointSets = append(want.EndpointSets, r)
		}
	}
	assert.Len(t, want.Exports, 2)
	assertRecord(t, want, after, "only the deleted cluster's records are gone")
	assertRecord(t, otherBefore, load(t, st, other), "other clustersets are untouched")
}

func clusterDeleteIdempotent(t *testing.T, st store.Store, cs string) {
	key := store.ClusterKey{ClusterSet: cs, Cluster: "c1"}
	require.NoError(t, st.DeleteCluster(ctx(t), key), "unknown clusterset")
	populateCluster(t, st, cs, "c1")
	require.NoError(t, st.DeleteCluster(ctx(t), key))
	require.NoError(t, st.DeleteCluster(ctx(t), key))
	require.NoError(t, st.DeleteCluster(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "never"}))

	snap := load(t, st, cs)
	assert.Empty(t, snap.Clusters)
	assert.Empty(t, snap.Exports)
	assert.Empty(t, snap.EndpointSets)
}

func clusterDeleteUnregistered(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 1)))
	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, 1)))
	orphan := exportKey(cs, "c1", "llm", "orphan")
	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(orphan, 1, 1)))

	require.NoError(t, st.DeleteCluster(ctx(t), store.ClusterKey{ClusterSet: cs, Cluster: "c1"}))
	snap := load(t, st, cs)
	assert.Empty(t, snap.Exports)
	assert.Empty(t, snap.EndpointSets)
}

// Exports.

func exportPutAndLoad(t *testing.T, st store.Store, cs string) {
	rec := exportRecord(exportKey(cs, "c1", "llm", "router"), 5)
	require.NoError(t, st.PutExport(ctx(t), rec))
	snap := load(t, st, cs)
	require.Len(t, snap.Exports, 1)
	assertRecord(t, rec, snap.Exports[0])
}

func exportStaleGeneration(t *testing.T, st store.Store, cs string) {
	rec := exportRecord(exportKey(cs, "c1", "llm", "router"), 5)
	require.NoError(t, st.PutExport(ctx(t), rec))

	for _, gen := range []uint64{0, 4, 5} {
		stale := exportRecord(rec.Key, gen)
		stale.Spec.Type = store.ServiceTypeHeadless
		requireStale(t, st.PutExport(ctx(t), stale), 5)
	}
	snap := load(t, st, cs)
	require.Len(t, snap.Exports, 1)
	assertRecord(t, rec, snap.Exports[0], "a stale put changes nothing")
}

func exportNewerGenerationReplaces(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 5)))

	next := exportRecord(key, 6)
	next.Spec = store.ExportSpec{Type: store.ServiceTypeHeadless, Ports: []store.ServicePort{{Name: "http", Protocol: store.ProtocolTCP, Port: 8000}}}
	next.UpdatedAt = t0.Add(time.Second)
	require.NoError(t, st.PutExport(ctx(t), next))

	snap := load(t, st, cs)
	require.Len(t, snap.Exports, 1)
	assertRecord(t, next, snap.Exports[0], "the spec is replaced, not merged")
}

func exportDeleteCascades(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	keep := exportKey(cs, "c1", "llm", "frontend")
	for _, k := range []store.ExportKey{key, keep} {
		require.NoError(t, st.PutExport(ctx(t), exportRecord(k, 1)))
		require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(k, 1, 3)))
	}

	require.NoError(t, st.DeleteExport(ctx(t), key))
	snap := load(t, st, cs)
	require.Len(t, snap.Exports, 1)
	assert.Equal(t, keep, snap.Exports[0].Key)
	require.Len(t, snap.EndpointSets, 1)
	assert.Equal(t, keep, snap.EndpointSets[0].Key)
}

func exportDeleteIdempotent(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.DeleteExport(ctx(t), key))
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 1)))
	require.NoError(t, st.DeleteExport(ctx(t), key))
	require.NoError(t, st.DeleteExport(ctx(t), key))
	assert.Empty(t, load(t, st, cs).Exports)
}

func exportDeleteForgetsGeneration(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 100)))
	require.NoError(t, st.DeleteExport(ctx(t), key))
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 1)), "a deleted record has no generation")
}

func exportKeysAreIndependent(t *testing.T, st store.Store, cs string) {
	other := clusterSetName(t, "b")
	keys := []store.ExportKey{
		exportKey(cs, "c1", "llm", "router"),
		exportKey(cs, "c2", "llm", "router"),
		exportKey(cs, "c1", "other", "router"),
		exportKey(cs, "c1", "llm", "router-2"),
		exportKey(other, "c1", "llm", "router"),
	}
	for _, k := range keys {
		require.NoError(t, st.PutExport(ctx(t), exportRecord(k, 7)), "%+v", k)
	}
	assert.Len(t, load(t, st, cs).Exports, 4)
	assert.Len(t, load(t, st, other).Exports, 1)
}

// Endpoint sets.

func endpointSetReplace(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, 5)))

	next := endpointSetRecord(key, 2, 2)
	next.UpdatedAt = t0.Add(time.Second)
	require.NoError(t, st.PutEndpointSet(ctx(t), next))

	snap := load(t, st, cs)
	require.Len(t, snap.EndpointSets, 1)
	assertRecord(t, next, snap.EndpointSets[0], "the set is replaced as a whole")
}

func endpointSetStaleGeneration(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	rec := endpointSetRecord(key, 10, 3)
	require.NoError(t, st.PutEndpointSet(ctx(t), rec))
	requireStale(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 10, 1)), 10)
	requireStale(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 9, 1)), 10)

	snap := load(t, st, cs)
	require.Len(t, snap.EndpointSets, 1)
	assertRecord(t, rec, snap.EndpointSets[0])
}

func endpointSetTooLarge(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	err := st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, store.MaxEndpointsPerSet+1))
	assert.ErrorIs(t, err, store.ErrTooLarge)
	assert.Empty(t, load(t, st, cs).EndpointSets)

	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, store.MaxEndpointsPerSet)))
	snap := load(t, st, cs)
	require.Len(t, snap.EndpointSets, 1)
	assert.Len(t, snap.EndpointSets[0].Endpoints, store.MaxEndpointsPerSet)
}

func endpointSetEmptyAndOrphan(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	rec := endpointSetRecord(key, 1, 0)
	require.NoError(t, st.PutEndpointSet(ctx(t), rec), "an endpoint set may exist without its export")
	snap := load(t, st, cs)
	assert.Empty(t, snap.Exports)
	require.Len(t, snap.EndpointSets, 1)
	assertRecord(t, rec, snap.EndpointSets[0])
}

func endpointSetDeleteIdempotent(t *testing.T, st store.Store, cs string) {
	key := exportKey(cs, "c1", "llm", "router")
	require.NoError(t, st.DeleteEndpointSet(ctx(t), key))
	require.NoError(t, st.PutExport(ctx(t), exportRecord(key, 1)))
	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(key, 1, 1)))
	require.NoError(t, st.DeleteEndpointSet(ctx(t), key))
	require.NoError(t, st.DeleteEndpointSet(ctx(t), key))

	snap := load(t, st, cs)
	assert.Empty(t, snap.EndpointSets)
	assert.Len(t, snap.Exports, 1, "deleting the endpoint set keeps the export")
}

// Snapshots.

func loadClusterSetComplete(t *testing.T, st store.Store, cs string) {
	other := clusterSetName(t, "b")
	want := map[string]*store.ClusterSetSnapshot{
		cs:    {ClusterSet: cs},
		other: {ClusterSet: other},
	}
	// Written out of key order so that the snapshot must sort.
	for _, c := range []string{"c3", "c1", "c2"} {
		for _, set := range []string{cs, other} {
			snap := want[set]
			reg, err := st.PutCluster(ctx(t), clusterRecord(set, c), t0)
			require.NoError(t, err)
			snap.Clusters = append(snap.Clusters, reg)

			for _, name := range []string{"svc-b", "svc-a"} {
				key := exportKey(set, c, "ns", name)
				exp := exportRecord(key, 3)
				require.NoError(t, st.PutExport(ctx(t), exp))
				snap.Exports = append(snap.Exports, exp)
				eps := endpointSetRecord(key, 4, 3)
				require.NoError(t, st.PutEndpointSet(ctx(t), eps))
				snap.EndpointSets = append(snap.EndpointSets, eps)
			}
			orphan := endpointSetRecord(exportKey(set, c, "ns", "orphan"), 1, 1)
			require.NoError(t, st.PutEndpointSet(ctx(t), orphan))
			snap.EndpointSets = append(snap.EndpointSets, orphan)
		}
	}
	// An export of a cluster that never registered belongs to the snapshot
	// too.
	unregistered := exportRecord(exportKey(cs, "c0", "ns", "svc"), 1)
	require.NoError(t, st.PutExport(ctx(t), unregistered))
	want[cs].Exports = append(want[cs].Exports, unregistered)

	for set, snap := range want {
		snap.Sort()
		got := load(t, st, set)
		assertRecord(t, *snap, got, "snapshot of %s", set)
		assert.Len(t, got.Clusters, 3)
		assert.GreaterOrEqual(t, len(got.Exports), 6)
		assert.Len(t, got.EndpointSets, 9)
	}
}

func loadClusterSetUnknown(t *testing.T, st store.Store, cs string) {
	snap := load(t, st, cs)
	assert.Equal(t, cs, snap.ClusterSet)
	assert.Empty(t, snap.Clusters)
	assert.Empty(t, snap.Exports)
	assert.Empty(t, snap.EndpointSets)
}

func loadClusterSetReturnsCopies(t *testing.T, st store.Store, cs string) {
	populateCluster(t, st, cs, "c1")
	first := load(t, st, cs)
	want := normalize(first)

	first.Clusters[0].Labels["env"] = "mutated"
	first.Exports[0].Spec.Ports[0].Port = 1
	first.Exports[0].Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds = 1
	first.EndpointSets[0].Endpoints[0].Addresses[0] = "192.0.2.1"
	first.EndpointSets[0].Endpoints[0].Attributes["pod"] = "mutated"

	assertRecord(t, want, load(t, st, cs))
}

func listClusterSets(t *testing.T, st store.Store, cs string) {
	names := []string{clusterSetName(t, "c"), clusterSetName(t, "a"), clusterSetName(t, "b")}
	_, err := st.PutCluster(ctx(t), clusterRecord(names[0], "c1"), t0)
	require.NoError(t, err)
	require.NoError(t, st.PutExport(ctx(t), exportRecord(exportKey(names[1], "c1", "ns", "svc"), 1)))
	require.NoError(t, st.PutEndpointSet(ctx(t), endpointSetRecord(exportKey(names[2], "c1", "ns", "svc"), 1, 1)))

	got, err := st.ListClusterSets(ctx(t))
	require.NoError(t, err)
	assert.True(t, slices.IsSorted(got), "sorted: %v", got)
	for _, n := range names {
		assert.Contains(t, got, n)
	}
}

// Store lifecycle and concurrency.

func ping(t *testing.T, st store.Store, _ string) {
	assert.NoError(t, st.Ping(ctx(t)))
}

func canceledContext(t *testing.T, st store.Store, cs string) {
	c, cancel := context.WithCancel(ctx(t))
	cancel()
	ck := store.ClusterKey{ClusterSet: cs, Cluster: "c1"}
	ek := exportKey(cs, "c1", "ns", "svc")

	calls := map[string]func() error{
		"PutCluster": func() error { _, err := st.PutCluster(c, clusterRecord(cs, "c1"), t0); return err },
		"RenewClusterLease": func() error {
			_, err := st.RenewClusterLease(c, ck, 1, t0)
			return err
		},
		"GetCluster":        func() error { _, err := st.GetCluster(c, ck); return err },
		"DeleteCluster":     func() error { return st.DeleteCluster(c, ck) },
		"PutExport":         func() error { return st.PutExport(c, exportRecord(ek, 1)) },
		"DeleteExport":      func() error { return st.DeleteExport(c, ek) },
		"PutEndpointSet":    func() error { return st.PutEndpointSet(c, endpointSetRecord(ek, 1, 1)) },
		"DeleteEndpointSet": func() error { return st.DeleteEndpointSet(c, ek) },
		"ListClusterSets":   func() error { _, err := st.ListClusterSets(c); return err },
		"LoadClusterSet":    func() error { _, err := st.LoadClusterSet(c, cs); return err },
		"Ping":              func() error { return st.Ping(c) },
	}
	for name, call := range calls {
		assert.ErrorIs(t, call(), context.Canceled, name)
	}
	assert.Empty(t, load(t, st, cs).Clusters, "nothing was written")
}

func concurrentRegistrations(t *testing.T, st store.Store, cs string) {
	const n = 16
	rec := clusterRecord(cs, "c1")
	epochs := make([]uint64, n)
	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := range n {
		wg.Go(func() {
			got, err := st.PutCluster(ctx(t), rec, t0)
			errs[i] = err
			epochs[i] = got.Lease.Epoch
		})
	}
	wg.Wait()
	for _, err := range errs {
		require.NoError(t, err)
	}

	seen := map[uint64]bool{}
	for _, e := range epochs {
		assert.False(t, seen[e], "epoch %d minted twice", e)
		seen[e] = true
	}
	stored, err := st.GetCluster(ctx(t), rec.Key)
	require.NoError(t, err)
	assert.Equal(t, slices.Max(epochs), stored.Lease.Epoch)
	assert.Equal(t, ms(t0)+n-1, stored.Lease.Epoch)
	assert.Equal(t, uint64(n), stored.Generation)
}

func concurrentGenerations(t *testing.T, st store.Store, cs string) {
	const n = 32
	key := exportKey(cs, "c1", "ns", "svc")
	var wg sync.WaitGroup
	errs := make([]error, n+1)
	for gen := n; gen >= 1; gen-- {
		wg.Go(func() {
			errs[gen] = st.PutExport(ctx(t), exportRecord(key, uint64(gen)))
			if errs[gen] == nil {
				errs[gen] = st.PutEndpointSet(ctx(t), endpointSetRecord(key, uint64(gen), 1))
			}
		})
	}
	wg.Wait()

	require.NoError(t, errs[n], "the highest generation always wins")
	for gen := 1; gen < n; gen++ {
		if errs[gen] != nil {
			assert.ErrorIs(t, errs[gen], store.ErrStaleGeneration{}, "generation %d", gen)
		}
	}
	snap := load(t, st, cs)
	require.Len(t, snap.Exports, 1)
	assert.Equal(t, uint64(n), snap.Exports[0].Generation)
	require.Len(t, snap.EndpointSets, 1)
	assert.Equal(t, uint64(n), snap.EndpointSets[0].Generation)
}
