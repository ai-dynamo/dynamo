// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/types/known/durationpb"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

// labelledExport exports namespace/name from cluster with two ready
// endpoints.
func labelledExport(t *testing.T, e *env, cluster, ns, name string, labels, annotations map[string]string, ports ...*mcsv1.ServicePort) {
	t.Helper()
	s := spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0, labels, ports...)
	s.Annotations = annotations
	_, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: cluster, Namespace: ns, Name: name, Spec: s,
	})
	require.NoError(t, err)
	_, err = e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: cluster, Namespace: ns, Name: name, Endpoints: endpoints(true, true),
	})
	require.NoError(t, err)
}

func TestServerInfo(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	info, err := e.info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)
	assert.Equal(t, "us-east-1", info.GetRegion())
	assert.Equal(t, "v1.2.3", info.GetVersion())
	assert.Equal(t, "e1", info.GetViewEpoch())
	assert.Equal(t, t0, info.GetServerTime().AsTime())
	require.Len(t, info.GetEndpoints(), 1)
	assert.True(t, info.GetEndpoints()[0].GetPrimary())
}

func TestValidation(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	cases := []struct {
		name   string
		call   func() error
		fields []string
	}{
		{"cluster names", func() error {
			_, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: "Prod", Cluster: ""})
			return err
		}, []string{"clusterset", "cluster"}},
		{"region label value", func() error {
			_, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs, Cluster: "a", Region: "us east/1"})
			return err
		}, []string{"region"}},
		{"negative ttl", func() error {
			_, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs, Cluster: "a", LeaseTtl: durationpb.New(-time.Second)})
			return err
		}, []string{"lease_ttl"}},
		{"heartbeat epoch", func() error {
			_, err := e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a"})
			return err
		}, []string{"epoch"}},
		{"export spec", func() error {
			_, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
				Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web",
				Spec: &mcsv1.ServiceExportSpec{
					Ports:                 []*mcsv1.ServicePort{{Port: 0}, {Port: 80, Protocol: 9}},
					IpFamilies:            []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_UNSPECIFIED},
					SessionAffinityConfig: &mcsv1.SessionAffinityConfig{ClientIp: &mcsv1.ClientIPConfig{TimeoutSeconds: -1}},
					Labels:                map[string]string{"": "x"},
					Annotations:           map[string]string{"a": string(make([]byte, mcsapi.MaxAnnotationBytes))},
				},
			})
			return err
		}, []string{"spec.type", "spec.ports[0].port", "spec.ports[1].protocol", "spec.ip_families[0]",
			"spec.session_affinity_config.client_ip.timeout_seconds", "spec.labels", "spec.annotations"}},
		{"missing spec", func() error {
			_, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web"})
			return err
		}, []string{"spec"}},
		{"endpoints", func() error {
			_, err := e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
				Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web",
				Endpoints: []*mcsv1.Endpoint{{Ports: []*mcsv1.EndpointPort{{Port: 70000}}}},
			})
			return err
		}, []string{"endpoints[0].addresses", "endpoints[0].ports[0].port"}},
		{"list imports", func() error {
			_, err := e.imports.ListServiceImports(ctx(t), &mcsv1.ListServiceImportsRequest{Clusterset: cs, Namespace: "Bad"})
			return err
		}, []string{"namespace"}},
		{"get import", func() error {
			_, err := e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs})
			return err
		}, []string{"namespace", "name"}},
		{"clusters", func() error {
			_, err := e.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{})
			return err
		}, []string{"clusterset"}},
		{"get cluster", func() error {
			_, err := e.clusters.GetCluster(ctx(t), &mcsv1.GetClusterRequest{Clusterset: cs})
			return err
		}, []string{"cluster"}},
		{"deregister cluster", func() error {
			_, err := e.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs})
			return err
		}, []string{"cluster"}},
		{"exports", func() error {
			_, err := e.exports.ListServiceExports(ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: cs})
			return err
		}, []string{"cluster"}},
		{"delete export", func() error {
			_, err := e.exports.DeleteServiceExport(ctx(t), &mcsv1.DeleteServiceExportRequest{Clusterset: cs})
			return err
		}, []string{"cluster", "namespace", "name"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := requireError(t, tc.call(), codes.InvalidArgument, mcsapi.ReasonInvalidArgument)
			fields := badRequestFields(st)
			for _, f := range tc.fields {
				assert.Contains(t, fields, f)
			}
		})
	}
}

func TestStoreUnavailable(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	e.register(t, "a", "us-east-1")
	require.NoError(t, e.store.Close())
	_, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs, Cluster: "a"})
	requireError(t, err, codes.Unavailable, mcsapi.ReasonStoreUnavailable)
	_, err = e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: "b", Namespace: "ns", Name: "web",
		Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0, nil, httpPort),
	})
	requireError(t, err, codes.Unavailable, mcsapi.ReasonStoreUnavailable)
	for _, call := range []func() error{
		func() error {
			_, err := e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: 1})
			return err
		},
		func() error {
			_, err := e.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs, Cluster: "a"})
			return err
		},
		func() error {
			_, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
				Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web",
				Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0, nil, httpPort),
			})
			return err
		},
		func() error {
			_, err := e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web"})
			return err
		},
		func() error {
			_, err := e.exports.DeleteServiceExport(ctx(t), &mcsv1.DeleteServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web"})
			return err
		},
	} {
		requireError(t, call(), codes.Unavailable, mcsapi.ReasonStoreUnavailable)
	}
}

func TestWatches(t *testing.T) {
	e := newEnv(t, view.HubOptions{BookmarkInterval: 5 * time.Second, MaxWatchers: 4})
	e.register(t, "a", "us-east-1")

	t.Log("A filtered import watch starts with an empty snapshot")
	web, err := e.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, Namespace: "ns"})
	require.NoError(t, err)
	ev, err := web.Recv()
	require.NoError(t, err)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START, ev.GetType())
	ev, err = web.Recv()
	require.NoError(t, err)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END, ev.GetType())
	rev := ev.GetRevision()

	t.Log("An export sends an upsert, then the bookmark interval a bookmark")
	labelledExport(t, e, "a", "ns", "web", nil, nil, httpPort)
	ev, err = web.Recv()
	require.NoError(t, err)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.GetType())
	assert.Equal(t, "web", ev.GetServiceImport().GetName())
	for ev.GetType() == mcsv1.EventType_EVENT_TYPE_UPSERT {
		e.clock.BlockUntil(1)
		e.clock.Advance(5 * time.Second)
		ev, err = web.Recv()
		require.NoError(t, err)
	}
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_BOOKMARK, ev.GetType())

	t.Log("Resuming from the snapshot revision replays the upsert")
	resumed, err := e.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, Namespace: "ns", ResumeRevision: rev})
	require.NoError(t, err)
	ev, err = resumed.Recv()
	require.NoError(t, err)
	assert.Equal(t, mcsv1.EventType_EVENT_TYPE_UPSERT, ev.GetType())

	t.Log("A cluster watch lists the registered cluster")
	clusters, err := e.clusters.WatchClusters(ctx(t), &mcsv1.WatchClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	_, err = clusters.Recv()
	require.NoError(t, err)
	ev2, err := clusters.Recv()
	require.NoError(t, err)
	assert.Equal(t, "a", ev2.GetCluster().GetCluster())

	t.Log("A watch with endpoints carries the endpoint lists")
	imports, err := e.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, IncludeEndpoints: true})
	require.NoError(t, err)
	for {
		ev3, err := imports.Recv()
		require.NoError(t, err)
		if ev3.GetType() == mcsv1.EventType_EVENT_TYPE_UPSERT && ev3.GetServiceImport().GetEndpoints()[0].GetEndpointCount() > 0 {
			assert.Len(t, ev3.GetServiceImport().GetEndpoints()[0].GetEndpoints(), 2)
			break
		}
	}

	t.Log("The fifth watcher is refused")
	refused, err := e.clusters.WatchClusters(ctx(t), &mcsv1.WatchClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	_, err = refused.Recv()
	requireError(t, err, codes.ResourceExhausted, mcsapi.ReasonTooManyWatchers)

	t.Log("Shutdown ends every watch with SHUTTING_DOWN")
	e.hub.Close()
	for {
		_, err = web.Recv()
		if err != nil {
			break
		}
	}
	requireError(t, err, codes.Unavailable, mcsapi.ReasonShuttingDown)
	assert.NotEqual(t, io.EOF, err)
}

func TestWatchValidation(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	c, err := e.clusters.WatchClusters(ctx(t), &mcsv1.WatchClustersRequest{})
	require.NoError(t, err)
	_, err = c.Recv()
	requireError(t, err, codes.InvalidArgument, mcsapi.ReasonInvalidArgument)
	i, err := e.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, Namespace: "BAD"})
	require.NoError(t, err)
	_, err = i.Recv()
	requireError(t, err, codes.InvalidArgument, mcsapi.ReasonInvalidArgument)
	i, err = e.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{
		Clusterset: cs, Selector: &mcsv1.LabelSelector{MatchLabels: map[string]string{"": "x"}},
	})
	require.NoError(t, err)
	_, err = i.Recv()
	requireError(t, err, codes.InvalidArgument, mcsapi.ReasonInvalidArgument)
}
