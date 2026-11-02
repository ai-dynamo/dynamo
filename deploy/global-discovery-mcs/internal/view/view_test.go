// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

const cs = "prod"

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

type fixture struct {
	clock   *clock.Fake
	metrics *metrics.Metrics
	view    *View
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{clock: clock.NewFake(t0), metrics: metrics.New(prometheus.NewRegistry())}
	f.view = New(Options{
		Clock:            f.clock,
		LeaseSkew:        2 * time.Second,
		WatchHistory:     100,
		LocalWriteWindow: 2 * time.Second,
		Epoch:            "e1",
		Metrics:          f.metrics,
	})
	return f
}

func clusterRec(name, region string, renewed time.Time) store.ClusterRecord {
	return store.ClusterRecord{
		Key:          store.ClusterKey{ClusterSet: cs, Cluster: name},
		Region:       region,
		Zones:        []string{region + "a"},
		Lease:        store.Lease{Epoch: uint64(renewed.UnixMilli()), TTL: 30 * time.Second, RenewedAt: renewed},
		Generation:   1,
		RegisteredAt: renewed,
		UpdatedAt:    renewed,
	}
}

func exportKey(cluster, ns, name string) store.ExportKey {
	return store.ExportKey{ClusterSet: cs, Cluster: cluster, Namespace: ns, Name: name}
}

func exportRec(cluster, ns, name string, gen uint64, created time.Time, labels, annotations map[string]string, ports ...store.ServicePort) store.ExportRecord {
	return store.ExportRecord{
		Key: exportKey(cluster, ns, name),
		Spec: store.ExportSpec{
			Annotations:             annotations,
			Labels:                  labels,
			ExportCreationTimestamp: created,
			Ports:                   ports,
			Type:                    store.ServiceTypeClusterSetIP,
			SessionAffinity:         store.SessionAffinityNone,
			IPFamilies:              []store.IPFamily{store.IPFamilyIPv4},
		},
		Generation: gen,
		UpdatedAt:  created,
	}
}

func endpointSet(cluster, ns, name string, gen uint64, ready ...bool) store.EndpointSetRecord {
	rec := store.EndpointSetRecord{Key: exportKey(cluster, ns, name), Generation: gen, UpdatedAt: t0}
	for i, r := range ready {
		rec.Endpoints = append(rec.Endpoints, store.Endpoint{
			Addresses:   []string{fmt.Sprintf("10.%d.0.%d", len(cluster), i+1)},
			AddressType: store.AddressTypeIPv4,
			Conditions:  store.EndpointConditions{Ready: r, Serving: r},
			Hostname:    fmt.Sprintf("%s-%d", name, i),
			Zone:        "z1",
		})
	}
	return rec
}

// importOf returns the import ns/name.
func importOf(t *testing.T, v *View, name string) *mcsv1.ServiceImport {
	t.Helper()
	o, ok := v.Get(cs, KindServiceImport, ImportKey("ns", name))
	require.True(t, ok, "import ns/%s", name)
	return o.Full.(*mcsv1.ServiceImport)
}

var httpPort = store.ServicePort{Name: "http", Protocol: store.ProtocolTCP, Port: 8000}

func baseSnapshot() store.ClusterSetSnapshot {
	snap := store.ClusterSetSnapshot{
		ClusterSet: cs,
		Clusters:   []store.ClusterRecord{clusterRec("a", "us-east-1", t0), clusterRec("b", "us-west-2", t0)},
		Exports: []store.ExportRecord{
			exportRec("a", "ns", "web", 1, t0.Add(-time.Hour), map[string]string{"app": "web"}, nil, httpPort),
			exportRec("b", "ns", "web", 1, t0, map[string]string{"app": "web"}, nil, httpPort),
		},
		EndpointSets: []store.EndpointSetRecord{
			endpointSet("a", "ns", "web", 5, true, false),
			endpointSet("b", "ns", "web", 6, true),
		},
	}
	snap.Sort()
	return snap
}

func TestReplaceDerivesImportsAndClusters(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)

	si := importOf(t, f.view, "web")
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, si.GetType())
	assert.Nil(t, si.GetConflict())
	require.Len(t, si.GetClusters(), 2)
	assert.Equal(t, "a", si.GetClusters()[0].GetCluster())
	assert.Equal(t, "us-east-1", si.GetClusters()[0].GetRegion())
	assert.True(t, si.GetClusters()[0].GetParticipating())
	assert.Equal(t, int32(2), si.GetClusters()[0].GetEndpointCount())
	assert.Equal(t, int32(1), si.GetClusters()[0].GetReadyEndpointCount())
	require.Len(t, si.GetEndpoints(), 2)
	assert.Equal(t, uint64(5), si.GetEndpoints()[0].GetGeneration())
	assert.Len(t, si.GetEndpoints()[0].GetEndpoints(), 2)
	assert.Equal(t, "web", si.GetLabels()["app"])
	require.Len(t, si.GetClusters()[0].GetConditions(), 3)
	assert.Equal(t, t0, si.GetClusters()[0].GetConditions()[0].GetLastTransitionAt().AsTime())

	o, ok := f.view.Get(cs, KindCluster, "a")
	require.True(t, ok)
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_ACTIVE, o.Full.(*mcsv1.Cluster).GetLease().GetState())

	assert.Equal(t, 2.0, testutilValue(f.metrics.Endpoints.WithLabelValues(cs, "ready")))
	assert.Equal(t, 1.0, testutilValue(f.metrics.Endpoints.WithLabelValues(cs, "not_ready")))
	assert.Equal(t, 2.0, testutilValue(f.metrics.Clusters.WithLabelValues(cs, "active")))

	rev := f.view.Revision(cs)
	f.view.Replace(baseSnapshot(), t0)
	assert.Equal(t, rev, f.view.Revision(cs), "an unchanged snapshot publishes nothing")
	assert.Equal(t, "e1:3", rev, "two clusters and one import")
}

// A cluster whose Service names a port differently from the winning export
// contributes its endpoints under the merged port name, with its own target
// port; a port the merged import lacks is dropped.
func TestImportEndpointPortsUseTheMergedNames(t *testing.T) {
	f := newFixture(t)
	snap := baseSnapshot()
	web := store.ServicePort{Name: "web", Protocol: store.ProtocolTCP, Port: 8000}
	metrics := store.ServicePort{Name: "metrics", Protocol: store.ProtocolTCP, Port: 9090}
	snap.Exports[1].Spec.Ports = []store.ServicePort{web, metrics}
	withPorts := func(rec *store.EndpointSetRecord, ports ...store.EndpointPort) {
		for i := range rec.Endpoints {
			rec.Endpoints[i].Ports = ports
		}
	}
	withPorts(&snap.EndpointSets[0], store.EndpointPort{Name: "http", Protocol: store.ProtocolTCP, Port: 8080})
	withPorts(&snap.EndpointSets[1],
		store.EndpointPort{Name: "web", Protocol: store.ProtocolTCP, Port: 9090},
		store.EndpointPort{Name: "metrics", Protocol: store.ProtocolTCP, Port: 9091},
		store.EndpointPort{Name: "debug", Protocol: store.ProtocolTCP, Port: 6060})
	snap.Sort()
	f.view.Replace(snap, t0)

	si := importOf(t, f.view, "web")
	require.Len(t, si.GetPorts(), 2, "the union by port and protocol")
	assert.Equal(t, "http", si.GetPorts()[0].GetName(), "the oldest export names 8000/TCP")
	assert.Equal(t, "metrics", si.GetPorts()[1].GetName())
	byCluster := map[string]*mcsv1.ClusterEndpoints{}
	for _, ce := range si.GetEndpoints() {
		byCluster[ce.GetCluster()] = ce
	}
	a := byCluster["a"].GetEndpoints()[0].GetPorts()
	require.Len(t, a, 1)
	assert.Equal(t, "http", a[0].GetName())
	assert.Equal(t, int32(8080), a[0].GetPort())
	b := byCluster["b"].GetEndpoints()[0].GetPorts()
	require.Len(t, b, 2, "the port without a Service port is dropped")
	assert.Equal(t, "http", b[0].GetName(), "renamed to the merged port")
	assert.Equal(t, int32(9090), b[0].GetPort(), "with its own target port")
	assert.Equal(t, "metrics", b[1].GetName())
	assert.Equal(t, int32(9091), b[1].GetPort())
	assert.Equal(t, int32(1), byCluster["b"].GetEndpointCount())
	assert.NotNil(t, si.GetConflict(), "the port sets differ")

	// The lite form and the full form agree on the renamed ports' owner.
	o, _ := f.view.Get(cs, KindServiceImport, ImportKey("ns", "web"))
	assert.Empty(t, o.Lite.(*mcsv1.ServiceImport).GetEndpoints()[0].GetEndpoints())

	// A Service port the merged import lacks drops its endpoint port: cluster
	// b's metrics is replaced by a port a's export does not have, named like
	// a's http port but on another number, which the duplicate-name rule skips.
	snap.Exports[1].Spec.Ports = []store.ServicePort{web, {Name: "http", Protocol: store.ProtocolTCP, Port: 7000}}
	snap.Exports[1].Generation++
	withPorts(&snap.EndpointSets[1],
		store.EndpointPort{Name: "web", Protocol: store.ProtocolTCP, Port: 9090},
		store.EndpointPort{Name: "http", Protocol: store.ProtocolTCP, Port: 7070})
	snap.EndpointSets[1].Generation++
	f.view.Replace(snap, t0.Add(time.Second))
	si = importOf(t, f.view, "web")
	for _, ce := range si.GetEndpoints() {
		byCluster[ce.GetCluster()] = ce
	}
	b = byCluster["b"].GetEndpoints()[0].GetPorts()
	require.Len(t, b, 1)
	assert.Equal(t, "http", b[0].GetName())
	assert.Equal(t, int32(9090), b[0].GetPort())
}

func TestLeaseExpiryExcludesCluster(t *testing.T) {
	f := newFixture(t)
	snap := baseSnapshot()
	snap.Clusters[1].Lease.RenewedAt = t0.Add(-40 * time.Second)
	snap.Exports = append(snap.Exports, exportRec("b", "ns", "router", 1, t0,
		map[string]string{"app": "router"}, nil, store.ServicePort{Name: "grpc", Port: 50071}))
	snap.EndpointSets = append(snap.EndpointSets, endpointSet("b", "ns", "router", 1, true))
	f.view.Replace(snap, t0)

	si := importOf(t, f.view, "web")
	require.Len(t, si.GetEndpoints(), 1, "b's lease is expired")
	assert.Equal(t, "a", si.GetEndpoints()[0].GetCluster())
	assert.False(t, si.GetClusters()[1].GetParticipating())
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_EXPIRED, si.GetClusters()[1].GetLeaseState())
	router := importOf(t, f.view, "router")
	assert.Empty(t, router.GetEndpoints(), "an import whose only cluster expired has no endpoints")
	require.Len(t, router.GetClusters(), 1)
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_EXPIRED, router.GetClusters()[0].GetLeaseState())

	// a expires too once the clock passes ttl + skew.
	f.clock.Advance(31 * time.Second)
	f.view.EvaluateLeases()
	assert.Len(t, importOf(t, f.view, "web").GetEndpoints(), 1)
	f.clock.Advance(2 * time.Second)
	f.view.EvaluateLeases()
	si = importOf(t, f.view, "web")
	assert.Empty(t, si.GetEndpoints())
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, si.GetType(), "the shape stays")
	assert.Equal(t, 1.0, testutilValue(f.metrics.LeaseExpirations.WithLabelValues(metrics.LeaseCluster)))

	// A renewal brings a back.
	f.view.ApplyCluster(clusterRec("a", "us-east-1", f.clock.Now()))
	assert.Len(t, importOf(t, f.view, "web").GetEndpoints(), 1)
}

func TestConflictAndTransitionTimes(t *testing.T) {
	f := newFixture(t)
	snap := baseSnapshot()
	snap.Exports[1].Spec.Type = store.ServiceTypeHeadless
	f.view.Replace(snap, t0)
	si := importOf(t, f.view, "web")
	require.NotNil(t, si.GetConflict())
	assert.Equal(t, api.ConditionReasonTypeConflict, si.GetConflict().GetReason())
	assert.Equal(t, "a", si.GetConflict().GetWinningCluster())
	assert.NotNil(t, si.GetClusters()[1].GetConflict())

	// The conflict is resolved later: the Conflict condition transitions,
	// Valid and Ready keep their times.
	f.clock.Advance(time.Minute)
	f.view.ApplyExport(exportRec("b", "ns", "web", 2, t0, map[string]string{"app": "web"}, nil, httpPort))
	si = importOf(t, f.view, "web")
	assert.Nil(t, si.GetConflict())
	conds := si.GetClusters()[1].GetConditions()
	assert.Equal(t, t0, conds[0].GetLastTransitionAt().AsTime())
	assert.Equal(t, t0.Add(time.Minute), conds[2].GetLastTransitionAt().AsTime())
}

func TestLocalWritesSurviveStaleSnapshot(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	loadStart := f.clock.Now()

	f.clock.Advance(100 * time.Millisecond)
	newer := exportRec("a", "ns", "web", 2, t0.Add(-time.Hour), map[string]string{"app": "web", "v": "2"}, nil, httpPort)
	f.view.ApplyExport(newer)
	f.view.ApplyExportDelete(exportKey("b", "ns", "web"))
	added := exportRec("a", "ns", "new", 1, t0, nil, nil, httpPort)
	f.view.ApplyExport(added)

	// The snapshot was loaded before the writes.
	f.view.Replace(baseSnapshot(), loadStart)
	rec, ok := f.view.ExportRecord(exportKey("a", "ns", "web"))
	require.True(t, ok)
	assert.Equal(t, uint64(2), rec.Generation)
	_, ok = f.view.ExportRecord(exportKey("b", "ns", "web"))
	assert.False(t, ok, "the local delete wins over the stale snapshot")
	_, ok = f.view.ExportRecord(exportKey("a", "ns", "new"))
	assert.True(t, ok)

	// A snapshot with a newer version wins even within the window.
	snap := baseSnapshot()
	snap.Exports[0].Generation = 3
	snap.Exports[1].Generation = 9
	f.view.Replace(snap, loadStart)
	rec, _ = f.view.ExportRecord(exportKey("a", "ns", "web"))
	assert.Equal(t, uint64(3), rec.Generation)
	rec, ok = f.view.ExportRecord(exportKey("b", "ns", "web"))
	require.True(t, ok, "a newer version than the deleted one was written elsewhere")
	assert.Equal(t, uint64(9), rec.Generation)

	// After the window the store is the truth again.
	f.clock.Advance(5 * time.Second)
	f.view.Replace(baseSnapshot(), f.clock.Now())
	_, ok = f.view.ExportRecord(exportKey("a", "ns", "new"))
	assert.False(t, ok)
}

func TestClusterDeleteCascades(t *testing.T) {
	f := newFixture(t)
	snap := baseSnapshot()
	f.view.Replace(snap, t0)
	require.Len(t, importOf(t, f.view, "web").GetClusters(), 2)

	loadStart := f.clock.Now()
	f.clock.Advance(10 * time.Millisecond)
	f.view.ApplyClusterDelete(store.ClusterKey{ClusterSet: cs, Cluster: "b"})
	f.view.Replace(snap, loadStart)
	_, ok := f.view.ClusterRecord(store.ClusterKey{ClusterSet: cs, Cluster: "b"})
	assert.False(t, ok)
	_, ok = f.view.ExportRecord(exportKey("b", "ns", "web"))
	assert.False(t, ok)
	assert.Len(t, importOf(t, f.view, "web").GetClusters(), 1)
	_, ok = f.view.EndpointSetGeneration(exportKey("b", "ns", "web"))
	assert.False(t, ok, "the cascade covers endpoint sets")

	// A re-registration after the delete is kept.
	f.clock.Advance(10 * time.Millisecond)
	f.view.ApplyCluster(clusterRec("b", "us-west-2", f.clock.Now()))
	f.view.Replace(snap, loadStart)
	_, ok = f.view.ClusterRecord(store.ClusterKey{ClusterSet: cs, Cluster: "b"})
	assert.True(t, ok)
}

// labelledSnapshot is cluster a exporting two Services of namespace dyn,
// with prefixed labels, and their endpoint sets.
func labelledSnapshot() store.ClusterSetSnapshot {
	labels := map[string]string{api.LabelExport: "true", "tier": "gold"}
	snap := store.ClusterSetSnapshot{
		ClusterSet: cs,
		Clusters:   []store.ClusterRecord{clusterRec("a", "us-east-1", t0)},
		Exports: []store.ExportRecord{
			exportRec("a", "dyn", "frontend", 1, t0.Add(-time.Minute), labels, map[string]string{"models": "llama-3"}, httpPort),
			exportRec("a", "dyn", "relay", 1, t0, labels, nil, store.ServicePort{Name: "grpc", Port: 9000}),
		},
		EndpointSets: []store.EndpointSetRecord{
			endpointSet("a", "dyn", "frontend", 1, true, true, false),
			endpointSet("a", "dyn", "relay", 1, true),
		},
	}
	snap.Sort()
	return snap
}

func TestListFiltersAndIndexes(t *testing.T) {
	f := newFixture(t)
	snap := labelledSnapshot()
	snap.Clusters = append(snap.Clusters, clusterRec("b", "us-west-2", t0))
	snap.Exports = append(snap.Exports, exportRec("b", "other", "svc", 1, t0, map[string]string{"app": "x"}, nil, httpPort))
	f.view.Replace(snap, t0)

	names := func(f ImportFilter, v *View) []string {
		msgs, _ := v.List(cs, f)
		var out []string
		for _, m := range msgs {
			out = append(out, m.(*mcsv1.ServiceImport).GetName())
		}
		return out
	}
	assert.Equal(t, []string{"frontend", "relay", "svc"}, names(ImportFilter{}, f.view))
	assert.Equal(t, []string{"frontend", "relay"}, names(ImportFilter{Selector: map[string]string{"tier": "gold"}}, f.view))
	assert.Equal(t, []string{"frontend", "relay"}, names(ImportFilter{Selector: map[string]string{"tier": "gold", api.LabelExport: "true"}}, f.view))
	assert.Empty(t, names(ImportFilter{Selector: map[string]string{"tier": "gold", "app": "x"}}, f.view))
	assert.Empty(t, names(ImportFilter{Selector: map[string]string{"tier": "silver"}}, f.view))
	assert.Equal(t, []string{"svc"}, names(ImportFilter{Namespace: "other"}, f.view))

	msgs, rev := f.view.List(cs, ImportFilter{Selector: map[string]string{"app": "x"}})
	require.Len(t, msgs, 1)
	assert.Equal(t, "svc", msgs[0].(*mcsv1.ServiceImport).GetName())
	assert.Equal(t, f.view.Revision(cs), rev)
	msgs, _ = f.view.List(cs, ImportFilter{Namespace: "dyn"})
	assert.Len(t, msgs, 2)
	msgs, _ = f.view.List(cs, ImportFilter{Namespace: "dyn", Endpoints: true})
	assert.NotEmpty(t, msgs[0].(*mcsv1.ServiceImport).GetEndpoints()[0].GetEndpoints())
	msgs, _ = f.view.List(cs, ImportFilter{Namespace: "dyn"})
	assert.Empty(t, msgs[0].(*mcsv1.ServiceImport).GetEndpoints()[0].GetEndpoints())

	msgs, _ = f.view.List(cs, ClusterFilter{})
	assert.Len(t, msgs, 2)
	f.clock.Advance(time.Hour)
	f.view.EvaluateLeases()
	msgs, _ = f.view.List(cs, ClusterFilter{})
	assert.Empty(t, msgs)
	msgs, _ = f.view.List(cs, ClusterFilter{IncludeExpired: true})
	assert.Len(t, msgs, 2)

	msgs, rev = f.view.List("unknown", ClusterFilter{})
	assert.Empty(t, msgs)
	assert.Equal(t, "e1:0", rev)
}

// Lite objects must equal their full objects without endpoint lists, so
// that a new proto field is not silently dropped from Lite.
func TestLiteMatchesFull(t *testing.T) {
	f := newFixture(t)
	snap := labelledSnapshot()
	snap.Exports[0].Spec.Annotations = map[string]string{"note": "x"}
	snap.Exports[0].Spec.SessionAffinity = store.SessionAffinityClientIP
	snap.Exports[0].Spec.SessionAffinityConfig = &store.SessionAffinityConfig{ClientIP: &store.ClientIPConfig{TimeoutSeconds: 60}}
	f.view.Replace(snap, t0)
	msgs, _ := f.view.List(cs, ImportFilter{Endpoints: true})
	require.NotEmpty(t, msgs)
	lites, _ := f.view.List(cs, ImportFilter{})
	for _, m := range msgs {
		full := proto.Clone(m).(*mcsv1.ServiceImport)
		for _, ce := range full.GetEndpoints() {
			ce.Endpoints = nil
		}
		found := false
		for _, l := range lites {
			if proto.Equal(l, full) {
				found = true
			}
		}
		assert.True(t, found, "%s", full.GetName())
	}
}

func TestAccessors(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)

	exps := f.view.Exports(store.ClusterKey{ClusterSet: cs, Cluster: "a"})
	require.Len(t, exps, 1)
	assert.Equal(t, uint64(5), exps[0].EndpointsGeneration)
	assert.Equal(t, int32(2), exps[0].EndpointCount)
	assert.Equal(t, int32(1), exps[0].ReadyEndpointCount)
	assert.Nil(t, f.view.Exports(store.ClusterKey{ClusterSet: "x", Cluster: "a"}))

	gen, ok := f.view.EndpointSetGeneration(exportKey("b", "ns", "web"))
	assert.True(t, ok)
	assert.Equal(t, uint64(6), gen)
	_, ok = f.view.EndpointSetGeneration(store.ExportKey{ClusterSet: "x"})
	assert.False(t, ok)
	_, ok = f.view.ExportRecord(store.ExportKey{ClusterSet: "x"})
	assert.False(t, ok)
	_, ok = f.view.ClusterRecord(store.ClusterKey{ClusterSet: "x"})
	assert.False(t, ok)
	_, ok = f.view.Get("x", KindServiceImport, "ns/web")
	assert.False(t, ok)
	assert.Equal(t, "e1:0", f.view.Revision("x"))

	recs := f.view.Records(cs)
	assert.Len(t, recs.Clusters, 2)
	assert.Len(t, recs.Exports, 2)
	assert.Len(t, recs.EndpointSets, 2)
	assert.Empty(t, f.view.Records("x").Clusters)

	f.view.SetDiscovered([]string{cs})
	sums := f.view.Summaries()
	require.Len(t, sums, 1)
	assert.Equal(t, Summary{ClusterSet: cs, Clusters: 2, ActiveClusters: 2, Exports: 2}, sums[0])
	assert.Equal(t, t0, f.view.Now())
}

func TestReadinessAndDiscovery(t *testing.T) {
	f := newFixture(t)
	ok, why := f.view.Ready(t0, 6*time.Second)
	assert.False(t, ok)
	assert.Contains(t, why, "not discovered")

	f.view.SetDiscovered([]string{cs})
	ok, why = f.view.Ready(t0, 6*time.Second)
	assert.False(t, ok)
	assert.Contains(t, why, "not loaded")

	f.view.Replace(baseSnapshot(), t0)
	ok, _ = f.view.Ready(t0, 6*time.Second)
	assert.True(t, ok)
	assert.Equal(t, 5*time.Second, f.view.Age(t0.Add(5*time.Second)))
	ok, why = f.view.Ready(t0.Add(7*time.Second), 6*time.Second)
	assert.False(t, ok)
	assert.Contains(t, why, "old")

	// A clusterset created by a request does not count for readiness and
	// is dropped once idle and unknown to the store.
	f.view.Ensure("adhoc")
	select {
	case <-f.view.Kicked():
	default:
		t.Fatal("a new clusterset kicks the refresher")
	}
	ok, _ = f.view.Ready(t0, 6*time.Second)
	assert.True(t, ok)
	assert.Equal(t, []string{"adhoc", cs}, f.view.ClusterSetNames())
	f.view.SetDiscovered([]string{cs})
	assert.Equal(t, []string{cs}, f.view.ClusterSetNames())

	// A clusterset with records stays tracked when the store forgets it.
	f.view.SetDiscovered(nil)
	assert.Equal(t, []string{cs}, f.view.ClusterSetNames())
}

func TestRevisions(t *testing.T) {
	assert.Equal(t, "1761480000000-ab:42", Revision("1761480000000-ab", 42))
	epoch, seq, err := ParseRevision("1761480000000-ab:42")
	require.NoError(t, err)
	assert.Equal(t, "1761480000000-ab", epoch)
	assert.Equal(t, uint64(42), seq)
	for _, bad := range []string{"", "42", ":1", "e:x", "e:-1"} {
		_, _, err := ParseRevision(bad)
		assert.Error(t, err, bad)
	}
	e := NewEpoch(t0)
	assert.Regexp(t, `^\d+-[0-9a-f]{8}$`, e)
	assert.NotEqual(t, e, NewEpoch(t0))
	assert.NotEmpty(t, New(Options{}).Epoch())
	assert.Equal(t, "service_import", KindServiceImport.String())
}

func TestApplyLocal(t *testing.T) {
	f := newFixture(t)
	require.NoError(t, f.view.ApplyLocal(
		clusterRec("a", "us-east-1", t0),
		exportRec("a", "ns", "web", 1, t0, nil, nil, httpPort),
		endpointSet("a", "ns", "web", 1, true),
	))
	si := importOf(t, f.view, "web")
	assert.Len(t, si.GetEndpoints(), 1)
	assert.ErrorContains(t, f.view.ApplyLocal("nope"), "cannot apply string")
}

func TestReplaceIgnoresOlderLoads(t *testing.T) {
	f := newFixture(t)
	newer := baseSnapshot()
	newer.Exports[0].Generation = 2
	f.view.Replace(newer, t0.Add(time.Second))
	f.view.Replace(baseSnapshot(), t0)
	rec, ok := f.view.ExportRecord(exportKey("a", "ns", "web"))
	require.True(t, ok)
	assert.Equal(t, uint64(2), rec.Generation, "a load that began earlier does not win")
}

// Two handlers for one key can apply their successful writes to the view
// in reverse order, and a refresh can load the newer version before the
// older handler resumes. Neither may move the view or the watch stream
// backwards.
func TestApplyRejectsOlderVersions(t *testing.T) {
	f := newFixture(t)
	f.view.Replace(baseSnapshot(), t0)
	h := newHub(f, HubOptions{})
	w := startWatch(t, h, ImportFilter{Endpoints: true}, "")
	_, rev := w.snapshot(t)
	assert.Equal(t, "e1:3", rev)

	// Endpoint sets: generation 9 is applied, then the older 7.
	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 9, true))
	ev := w.next(t)
	assert.Equal(t, "e1:4", ev.Revision)
	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 7, true, true))
	gen, ok := f.view.EndpointSetGeneration(exportKey("a", "ns", "web"))
	require.True(t, ok)
	assert.Equal(t, uint64(9), gen)
	assert.Equal(t, "e1:4", f.view.Revision(cs), "an older application publishes nothing")
	// The same version applied twice is idempotent and publishes nothing.
	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 9, true))
	assert.Equal(t, "e1:4", f.view.Revision(cs))

	// Exports: generation 3 then 2.
	f.view.ApplyExport(exportRec("a", "ns", "web", 3, t0.Add(-time.Hour), map[string]string{"app": "web", "v": "3"}, nil, httpPort))
	ev = w.next(t)
	assert.Equal(t, "e1:5", ev.Revision)
	f.view.ApplyExport(exportRec("a", "ns", "web", 2, t0.Add(-time.Hour), map[string]string{"app": "web", "v": "2"}, nil, httpPort))
	rec, ok := f.view.ExportRecord(exportKey("a", "ns", "web"))
	require.True(t, ok)
	assert.Equal(t, uint64(3), rec.Generation)
	assert.Equal(t, "3", importOf(t, f.view, "web").GetLabels()["v"])
	assert.Equal(t, "e1:5", f.view.Revision(cs))

	// Clusters: a renewal older than the held one, and a lower lease epoch,
	// are ignored; a newer renewal and a re-registration are applied.
	held := clusterRec("a", "us-east-1", t0)
	older := held
	older.Lease.RenewedAt = t0.Add(-10 * time.Second)
	f.view.ApplyCluster(older)
	got, ok := f.view.ClusterRecord(held.Key)
	require.True(t, ok)
	assert.Equal(t, t0, got.Lease.RenewedAt)
	lowerEpoch := clusterRec("a", "us-east-1", t0.Add(time.Minute))
	lowerEpoch.Lease.Epoch = held.Lease.Epoch - 1
	f.view.ApplyCluster(lowerEpoch)
	got, _ = f.view.ClusterRecord(held.Key)
	assert.Equal(t, held.Lease.Epoch, got.Lease.Epoch)
	assert.Equal(t, t0, got.Lease.RenewedAt)
	assert.Equal(t, "e1:5", f.view.Revision(cs), "older cluster versions publish nothing")
	renewed := held
	renewed.Lease.RenewedAt = t0.Add(10 * time.Second)
	f.view.ApplyCluster(renewed)
	got, _ = f.view.ClusterRecord(held.Key)
	assert.Equal(t, t0.Add(10*time.Second), got.Lease.RenewedAt)
	reregistered := clusterRec("a", "us-east-1", t0.Add(time.Minute))
	f.view.ApplyCluster(reregistered)
	got, _ = f.view.ClusterRecord(held.Key)
	assert.Equal(t, reregistered.Lease.Epoch, got.Lease.Epoch)
	assert.Equal(t, "e1:7", f.view.Revision(cs), "the renewal and the re-registration each change the Cluster object")

	// A refresh loaded the newer write before the older handler resumed.
	snap := baseSnapshot()
	snap.EndpointSets[0].Generation = 12
	f.view.Replace(snap, f.clock.Now())
	ev = w.next(t)
	assert.Equal(t, uint64(12), ev.Object.(*mcsv1.ServiceImport).GetEndpoints()[0].GetGeneration())
	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 11, true))
	gen, _ = f.view.EndpointSetGeneration(exportKey("a", "ns", "web"))
	assert.Equal(t, uint64(12), gen)
	assert.Equal(t, ev.Revision, f.view.Revision(cs))

	// The next real change is the next event: nothing was queued for the
	// older applications.
	f.view.ApplyEndpointSet(endpointSet("a", "ns", "web", 13, true))
	ev = w.next(t)
	assert.Equal(t, uint64(13), ev.Object.(*mcsv1.ServiceImport).GetEndpoints()[0].GetGeneration())
	assert.Equal(t, "e1:9", ev.Revision)
	w.cancel()
}
