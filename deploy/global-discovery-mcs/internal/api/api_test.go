// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

const cs = "prod"

var t0 = time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)

type env struct {
	clock    *clock.Fake
	store    *memory.Store
	view     *view.View
	hub      *view.Hub
	clusters mcsv1.ClusterServiceClient
	exports  mcsv1.ServiceExportServiceClient
	imports  mcsv1.ServiceImportServiceClient
	info     mcsv1.ServerInfoServiceClient
}

func newEnv(t *testing.T, hubOpts view.HubOptions) *env {
	t.Helper()
	e := &env{clock: clock.NewFake(t0), store: memory.New()}
	m := metrics.New(prometheus.NewRegistry())
	e.view = view.New(view.Options{Clock: e.clock, LeaseSkew: 2 * time.Second, Epoch: "e1", Metrics: m})
	hubOpts.Clock = e.clock
	hubOpts.Metrics = m
	e.hub = view.NewHub(e.view, hubOpts)
	srv := New(Options{
		Store:     e.store,
		View:      e.view,
		Hub:       e.hub,
		Clock:     e.clock,
		Region:    "us-east-1",
		Version:   "v1.2.3",
		Endpoints: []*mcsv1.ServerEndpoint{{Region: "us-east-1", GrpcUrl: "https://registry.use1:443", Primary: true}},
	})
	lis := bufconn.Listen(1 << 20)
	gs := grpc.NewServer()
	srv.Register(gs)
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(func() {
		e.hub.Close()
		gs.Stop()
	})
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	e.clusters = mcsv1.NewClusterServiceClient(conn)
	e.exports = mcsv1.NewServiceExportServiceClient(conn)
	e.imports = mcsv1.NewServiceImportServiceClient(conn)
	e.info = mcsv1.NewServerInfoServiceClient(conn)
	return e
}

func ctx(t *testing.T) context.Context { return t.Context() }

// requireError checks the gRPC code and the ErrorInfo reason of err and
// returns its status.
func requireError(t *testing.T, err error, code codes.Code, reason string) *status.Status {
	t.Helper()
	require.Error(t, err)
	st, ok := status.FromError(err)
	require.True(t, ok)
	require.Equal(t, code, st.Code(), st.Message())
	var info *errdetails.ErrorInfo
	for _, d := range st.Details() {
		if i, ok := d.(*errdetails.ErrorInfo); ok {
			info = i
		}
	}
	require.NotNil(t, info, "ErrorInfo detail")
	assert.Equal(t, mcsapi.ErrorDomain, info.GetDomain())
	assert.Equal(t, reason, info.GetReason())
	return st
}

func badRequestFields(st *status.Status) []string {
	var fields []string
	for _, d := range st.Details() {
		if br, ok := d.(*errdetails.BadRequest); ok {
			for _, v := range br.GetFieldViolations() {
				fields = append(fields, v.GetField())
			}
		}
	}
	return fields
}

func (e *env) register(t *testing.T, cluster, region string) *mcsv1.Cluster {
	t.Helper()
	resp, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{
		Clusterset: cs, Cluster: cluster, Region: region, Zones: []string{region + "a"},
	})
	require.NoError(t, err)
	return resp.GetCluster()
}

func spec(typ mcsv1.ServiceType, created time.Time, labels map[string]string, ports ...*mcsv1.ServicePort) *mcsv1.ServiceExportSpec {
	return &mcsv1.ServiceExportSpec{
		Type:                    typ,
		Labels:                  labels,
		ExportCreationTimestamp: timestamppb.New(created),
		Ports:                   ports,
		IpFamilies:              []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV4},
	}
}

func endpoints(ready ...bool) []*mcsv1.Endpoint {
	var out []*mcsv1.Endpoint
	for i, r := range ready {
		out = append(out, &mcsv1.Endpoint{
			Addresses:  []string{"10.0.0." + string(rune('1'+i))},
			Conditions: &mcsv1.EndpointConditions{Ready: r, Serving: r},
			Ports:      []*mcsv1.EndpointPort{{Name: "http", Port: 8080}},
		})
	}
	return out
}

var httpPort = &mcsv1.ServicePort{Name: "http", Port: 80}

func TestClusterFlow(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	resp, err := e.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{
		Clusterset: cs, Cluster: "a", Region: "us-east-1", Provider: "eks", Labels: map[string]string{"env": "prod"},
		AgentId: "ctrl-0", AgentVersion: "v1", LeaseTtl: durationpb.New(5 * time.Second),
	})
	require.NoError(t, err)
	c := resp.GetCluster()
	assert.Equal(t, 10*time.Second, c.GetLease().GetTtl().AsDuration(), "clamped to the minimum")
	assert.Equal(t, uint64(t0.UnixMilli()), c.GetLease().GetEpoch())
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_ACTIVE, c.GetLease().GetState())
	assert.Equal(t, t0, resp.GetServerTime().AsTime())
	epoch := c.GetLease().GetEpoch()

	e.clock.Advance(3 * time.Second)
	hb, err := e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: epoch})
	require.NoError(t, err)
	assert.Equal(t, t0.Add(3*time.Second), hb.GetLease().GetRenewedAt().AsTime())

	_, err = e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: epoch + 1})
	requireError(t, err, codes.Aborted, mcsapi.ReasonLeaseEpochMismatch)
	_, err = e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "zz", Epoch: 1})
	requireError(t, err, codes.NotFound, mcsapi.ReasonClusterNotRegistered)

	// Re-registration mints a new epoch and keeps registered_at.
	e.clock.Advance(time.Second)
	again := e.register(t, "a", "us-east-1")
	assert.Greater(t, again.GetLease().GetEpoch(), epoch)
	assert.Equal(t, t0, again.GetRegisteredAt().AsTime())
	assert.Equal(t, 30*time.Second, again.GetLease().GetTtl().AsDuration(), "the default TTL")
	e.register(t, "b", "us-west-2")

	got, err := e.clusters.GetCluster(ctx(t), &mcsv1.GetClusterRequest{Clusterset: cs, Cluster: "a"})
	require.NoError(t, err)
	assert.Equal(t, "us-east-1", got.GetCluster().GetRegion())
	_, err = e.clusters.GetCluster(ctx(t), &mcsv1.GetClusterRequest{Clusterset: cs, Cluster: "zz"})
	requireError(t, err, codes.NotFound, mcsapi.ReasonClusterNotRegistered)

	list, err := e.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	assert.Len(t, list.GetClusters(), 2)
	assert.NotEmpty(t, list.GetRevision())

	// b's lease expires; a renews.
	e.clock.Advance(31 * time.Second)
	_, err = e.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: again.GetLease().GetEpoch()})
	require.NoError(t, err)
	e.clock.Advance(2 * time.Second)
	e.view.EvaluateLeases()
	list, err = e.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	require.Len(t, list.GetClusters(), 1)
	list, err = e.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{Clusterset: cs, IncludeExpired: true})
	require.NoError(t, err)
	assert.Len(t, list.GetClusters(), 2)

	sets, err := e.clusters.ListClusterSets(ctx(t), &mcsv1.ListClusterSetsRequest{})
	require.NoError(t, err)
	require.Len(t, sets.GetClustersets(), 1)
	assert.Equal(t, int32(2), sets.GetClustersets()[0].GetClusterCount())
	assert.Equal(t, int32(1), sets.GetClustersets()[0].GetActiveClusterCount())

	_, err = e.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs, Cluster: "b"})
	require.NoError(t, err)
	_, err = e.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs, Cluster: "b"})
	require.NoError(t, err, "idempotent")
	_, err = e.clusters.GetCluster(ctx(t), &mcsv1.GetClusterRequest{Clusterset: cs, Cluster: "b"})
	requireError(t, err, codes.NotFound, mcsapi.ReasonClusterNotRegistered)
}

func TestExportImportFlow(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	_, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 1,
		Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0, nil, httpPort),
	})
	requireError(t, err, codes.FailedPrecondition, mcsapi.ReasonClusterNotRegistered)

	e.register(t, "a", "us-east-1")
	e.register(t, "b", "us-west-2")
	labels := map[string]string{"app": "web"}
	put, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 10,
		Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0.Add(-time.Hour), labels, httpPort),
	})
	require.NoError(t, err)
	assert.Equal(t, uint64(10), put.GetExport().GetGeneration())
	assert.Equal(t, mcsv1.Protocol_PROTOCOL_TCP, put.GetExport().GetSpec().GetPorts()[0].GetProtocol(), "TCP by default")
	assert.Equal(t, mcsv1.SessionAffinity_SESSION_AFFINITY_NONE, put.GetExport().GetSpec().GetSessionAffinity())

	// b exports a headless Service later: a type conflict that a wins.
	_, err = e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: "b", Namespace: "ns", Name: "web",
		Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_HEADLESS, t0, labels, httpPort),
	})
	require.NoError(t, err, "generation 0 is assigned by the server")

	// A stale generation reports the stored one.
	_, err = e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 9,
		Spec: spec(mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, t0.Add(-time.Hour), labels, httpPort),
	})
	st := requireError(t, err, codes.FailedPrecondition, mcsapi.ReasonStaleGeneration)
	var pf *errdetails.PreconditionFailure
	var info *errdetails.ErrorInfo
	for _, d := range st.Details() {
		switch x := d.(type) {
		case *errdetails.PreconditionFailure:
			pf = x
		case *errdetails.ErrorInfo:
			info = x
		}
	}
	require.NotNil(t, pf)
	assert.Equal(t, "10", info.GetMetadata()[mcsapi.MetadataCurrentGeneration])
	assert.Equal(t, "generation", pf.GetViolations()[0].GetSubject())

	rep, err := e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 5, Endpoints: endpoints(true, false),
	})
	require.NoError(t, err)
	assert.Equal(t, int32(2), rep.GetEndpointCount())
	assert.Equal(t, int32(1), rep.GetReadyEndpointCount())
	_, err = e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 5, Endpoints: endpoints(true),
	})
	requireError(t, err, codes.FailedPrecondition, mcsapi.ReasonStaleGeneration)
	_, err = e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "b", Namespace: "ns", Name: "web", Endpoints: endpoints(true),
	})
	require.NoError(t, err)

	got, err := e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "web", IncludeEndpoints: true})
	require.NoError(t, err)
	si := got.GetServiceImport()
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, si.GetType())
	require.NotNil(t, si.GetConflict())
	assert.Equal(t, mcsapi.ConditionReasonTypeConflict, si.GetConflict().GetReason())
	assert.Equal(t, "a", si.GetConflict().GetWinningCluster())
	require.Len(t, si.GetEndpoints(), 2)
	assert.Len(t, si.GetEndpoints()[0].GetEndpoints(), 2)
	assert.Equal(t, mcsv1.AddressType_ADDRESS_TYPE_IPV4, si.GetEndpoints()[0].GetEndpoints()[0].GetAddressType(), "inferred")
	require.Len(t, si.GetClusters(), 2)
	assert.Equal(t, mcsapi.ConditionConflict, si.GetClusters()[1].GetConditions()[2].GetType())
	assert.Equal(t, mcsv1.ConditionStatus_CONDITION_STATUS_TRUE, si.GetClusters()[1].GetConditions()[2].GetStatus())

	got, err = e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "web"})
	require.NoError(t, err)
	assert.Empty(t, got.GetServiceImport().GetEndpoints()[0].GetEndpoints())
	assert.Equal(t, int32(2), got.GetServiceImport().GetEndpoints()[0].GetEndpointCount())
	_, err = e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "nope"})
	requireError(t, err, codes.NotFound, mcsapi.ReasonNotFound)

	list, err := e.imports.ListServiceImports(ctx(t), &mcsv1.ListServiceImportsRequest{
		Clusterset: cs, Selector: &mcsv1.LabelSelector{MatchLabels: map[string]string{"app": "web"}},
	})
	require.NoError(t, err)
	assert.Len(t, list.GetServiceImports(), 1)
	list, err = e.imports.ListServiceImports(ctx(t), &mcsv1.ListServiceImportsRequest{Clusterset: cs, Namespace: "other"})
	require.NoError(t, err)
	assert.Empty(t, list.GetServiceImports())

	exps, err := e.exports.ListServiceExports(ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: cs, Cluster: "a"})
	require.NoError(t, err)
	require.Len(t, exps.GetExports(), 1)
	assert.Equal(t, uint64(5), exps.GetExports()[0].GetEndpointsGeneration())
	assert.Equal(t, int32(1), exps.GetExports()[0].GetReadyEndpointCount())

	tooMany := make([]*mcsv1.Endpoint, mcsapi.MaxEndpointsPerExport+1)
	_, err = e.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web", Generation: 99, Endpoints: tooMany,
	})
	requireError(t, err, codes.InvalidArgument, mcsapi.ReasonTooManyEndpoints)

	// Deregistering b hides its endpoints and its export.
	_, err = e.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs, Cluster: "b"})
	require.NoError(t, err)
	got, err = e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "web", IncludeEndpoints: true})
	require.NoError(t, err)
	require.Len(t, got.GetServiceImport().GetEndpoints(), 1)
	assert.Equal(t, "a", got.GetServiceImport().GetEndpoints()[0].GetCluster())
	assert.Nil(t, got.GetServiceImport().GetConflict())

	_, err = e.exports.DeleteServiceExport(ctx(t), &mcsv1.DeleteServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "web"})
	require.NoError(t, err)
	_, err = e.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "web"})
	requireError(t, err, codes.NotFound, mcsapi.ReasonNotFound)
}

func TestExportCreationTimestampDefault(t *testing.T) {
	e := newEnv(t, view.HubOptions{})
	e.register(t, "a", "us-east-1")
	s := spec(mcsv1.ServiceType_SERVICE_TYPE_HEADLESS, t0, nil, httpPort)
	s.ExportCreationTimestamp = nil
	put, err := e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "x", Spec: s})
	require.NoError(t, err)
	assert.Equal(t, t0, put.GetExport().GetSpec().GetExportCreationTimestamp().AsTime())
	e.clock.Advance(time.Minute)
	put, err = e.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "x", Spec: s})
	require.NoError(t, err)
	assert.Equal(t, t0, put.GetExport().GetSpec().GetExportCreationTimestamp().AsTime(), "kept from the stored export")
}
