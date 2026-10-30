// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/durationpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

const cs = "prod"

// clock is an adjustable test clock.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func newClock() *clock { return &clock{now: time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)} }

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

type clients struct {
	clusters mcsv1.ClusterServiceClient
	exports  mcsv1.ServiceExportServiceClient
	imports  mcsv1.ServiceImportServiceClient
	info     mcsv1.ServerInfoServiceClient
}

func start(t *testing.T, opts ...fakeregistry.Option) (*fakeregistry.Server, clients) {
	t.Helper()
	srv := fakeregistry.New(opts...)
	run := srv.ServeBufconn()
	t.Cleanup(run.Stop)
	conn, err := grpc.NewClient("passthrough:///fakeregistry",
		grpc.WithContextDialer(run.Dialer()),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return srv, clients{
		clusters: mcsv1.NewClusterServiceClient(conn),
		exports:  mcsv1.NewServiceExportServiceClient(conn),
		imports:  mcsv1.NewServiceImportServiceClient(conn),
		info:     mcsv1.NewServerInfoServiceClient(conn),
	}
}

func ctx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return c
}

func errorInfo(t *testing.T, err error) *errdetails.ErrorInfo {
	t.Helper()
	st, ok := status.FromError(err)
	require.True(t, ok, "not a status error: %v", err)
	for _, d := range st.Details() {
		if info, ok := d.(*errdetails.ErrorInfo); ok {
			return info
		}
	}
	t.Fatalf("no ErrorInfo in %v", err)
	return nil
}

func TestClusterLeaseLifecycle(t *testing.T) {
	clk := newClock()
	_, c := start(t, fakeregistry.WithClock(clk.Now), fakeregistry.WithRefreshInterval(0))

	reg, err := c.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{
		Clusterset: cs, Cluster: "a", Region: "us-east-1", LeaseTtl: durationpb.New(time.Second),
	})
	require.NoError(t, err)
	lease := reg.GetCluster().GetLease()
	assert.Equal(t, uint64(clk.Now().UnixMilli()), lease.GetEpoch())
	assert.Equal(t, 10*time.Second, lease.GetTtl().AsDuration(), "TTL is clamped to 10 s")
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_ACTIVE, lease.GetState())
	assert.Equal(t, uint64(1), reg.GetCluster().GetGeneration())

	_, err = c.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: lease.GetEpoch() + 1})
	assert.Equal(t, codes.Aborted, status.Code(err))
	assert.Equal(t, api.ReasonLeaseEpochMismatch, errorInfo(t, err).GetReason())
	assert.Equal(t, api.ErrorDomain, errorInfo(t, err).GetDomain())

	_, err = c.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "nope", Epoch: 1})
	assert.Equal(t, codes.NotFound, status.Code(err))
	assert.Equal(t, api.ReasonClusterNotRegistered, errorInfo(t, err).GetReason())

	clk.Advance(5 * time.Second)
	hb, err := c.clusters.HeartbeatCluster(ctx(t), &mcsv1.HeartbeatClusterRequest{Clusterset: cs, Cluster: "a", Epoch: lease.GetEpoch()})
	require.NoError(t, err)
	assert.Equal(t, clk.Now(), hb.GetLease().GetRenewedAt().AsTime())

	clk.Advance(13 * time.Second)
	list, err := c.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	assert.Empty(t, list.GetClusters(), "expired clusters are hidden")
	list, err = c.clusters.ListClusters(ctx(t), &mcsv1.ListClustersRequest{Clusterset: cs, IncludeExpired: true})
	require.NoError(t, err)
	require.Len(t, list.GetClusters(), 1)
	assert.Equal(t, mcsv1.LeaseState_LEASE_STATE_EXPIRED, list.GetClusters()[0].GetLease().GetState())

	// Re-registration mints a new epoch and keeps registered_at.
	clk.Advance(time.Second)
	again, err := c.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs, Cluster: "a"})
	require.NoError(t, err)
	assert.Greater(t, again.GetCluster().GetLease().GetEpoch(), lease.GetEpoch())
	assert.Equal(t, reg.GetCluster().GetRegisteredAt().AsTime(), again.GetCluster().GetRegisteredAt().AsTime())
	assert.Equal(t, uint64(2), again.GetCluster().GetGeneration())

	sets, err := c.clusters.ListClusterSets(ctx(t), &mcsv1.ListClusterSetsRequest{})
	require.NoError(t, err)
	require.Len(t, sets.GetClustersets(), 1)
	assert.Equal(t, int32(1), sets.GetClustersets()[0].GetActiveClusterCount())

	_, err = c.clusters.DeregisterCluster(ctx(t), &mcsv1.DeregisterClusterRequest{Clusterset: cs, Cluster: "a"})
	require.NoError(t, err)
	_, err = c.clusters.GetCluster(ctx(t), &mcsv1.GetClusterRequest{Clusterset: cs, Cluster: "a"})
	assert.Equal(t, codes.NotFound, status.Code(err))

	_, err = c.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs})
	assert.Equal(t, codes.InvalidArgument, status.Code(err))
}

func TestExportGenerationsAndImports(t *testing.T) {
	clk := newClock()
	srv, c := start(t, fakeregistry.WithClock(clk.Now), fakeregistry.WithRefreshInterval(0))

	put := func(cluster string, gen uint64, created time.Time, ports ...int32) error {
		spec := &mcsv1.ServiceExportSpec{
			Type:                    mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
			ExportCreationTimestamp: timestamppb.New(created),
			Labels:                  map[string]string{"owner": cluster},
		}
		for _, p := range ports {
			spec.Ports = append(spec.Ports, &mcsv1.ServicePort{Name: "p", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: p})
		}
		_, err := c.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
			Clusterset: cs, Cluster: cluster, Namespace: "ns", Name: "svc", Spec: spec, Generation: gen,
		})
		return err
	}

	err := put("a", 1, clk.Now(), 80)
	assert.Equal(t, codes.FailedPrecondition, status.Code(err))
	assert.Equal(t, api.ReasonClusterNotRegistered, errorInfo(t, err).GetReason())

	srv.EnsureCluster(cs, "a", "us-east-1")
	srv.EnsureCluster(cs, "b", "us-west-2")
	require.NoError(t, put("a", 5, clk.Now(), 80))
	require.NoError(t, put("b", 1, clk.Now().Add(-time.Hour), 80, 443))

	err = put("a", 5, clk.Now(), 80)
	require.Equal(t, codes.FailedPrecondition, status.Code(err))
	info := errorInfo(t, err)
	assert.Equal(t, api.ReasonStaleGeneration, info.GetReason())
	assert.Equal(t, "5", info.GetMetadata()[api.MetadataCurrentGeneration])
	st, _ := status.FromError(err)
	var pf *errdetails.PreconditionFailure
	for _, d := range st.Details() {
		if v, ok := d.(*errdetails.PreconditionFailure); ok {
			pf = v
		}
	}
	require.NotNil(t, pf)
	assert.Equal(t, api.ReasonStaleGeneration, pf.GetViolations()[0].GetType())

	_, err = c.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc", Generation: 1,
		Endpoints: fakeregistry.ReadyEndpoints("10.0.0.1", "10.0.0.2"),
	})
	require.NoError(t, err)
	_, err = c.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc", Generation: 1,
	})
	assert.Equal(t, api.ReasonStaleGeneration, errorInfo(t, err).GetReason())
	_, err = c.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
		Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc", Generation: 9,
		Endpoints: make([]*mcsv1.Endpoint, api.MaxEndpointsPerExport+1),
	})
	assert.Equal(t, api.ReasonTooManyEndpoints, errorInfo(t, err).GetReason())
	srv.SetEndpoints(cs, "b", "ns", "svc", fakeregistry.ReadyEndpoints("10.1.0.1"))

	exports, err := c.exports.ListServiceExports(ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: cs, Cluster: "a"})
	require.NoError(t, err)
	require.Len(t, exports.GetExports(), 1)
	assert.Equal(t, int32(2), exports.GetExports()[0].GetReadyEndpointCount())
	assert.Equal(t, uint64(1), exports.GetExports()[0].GetEndpointsGeneration())

	imp, err := c.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "svc", IncludeEndpoints: true})
	require.NoError(t, err)
	si := imp.GetServiceImport()
	assert.Equal(t, "b", si.GetLabels()["owner"], "the oldest export wins")
	require.Len(t, si.GetPorts(), 2, "ports are unioned")
	require.Len(t, si.GetEndpoints(), 2)
	assert.Equal(t, "a", si.GetEndpoints()[0].GetCluster())
	assert.Len(t, si.GetEndpoints()[0].GetEndpoints(), 2)
	assert.Nil(t, si.GetConflict())

	// Without include_endpoints only the counts remain.
	list, err := c.imports.ListServiceImports(ctx(t), &mcsv1.ListServiceImportsRequest{Clusterset: cs, Namespace: "ns"})
	require.NoError(t, err)
	require.Len(t, list.GetServiceImports(), 1)
	assert.Empty(t, list.GetServiceImports()[0].GetEndpoints()[0].GetEndpoints())
	assert.Equal(t, int32(2), list.GetServiceImports()[0].GetEndpoints()[0].GetReadyEndpointCount())
	list, err = c.imports.ListServiceImports(ctx(t), &mcsv1.ListServiceImportsRequest{
		Clusterset: cs, Selector: &mcsv1.LabelSelector{MatchLabels: map[string]string{"owner": "a"}},
	})
	require.NoError(t, err)
	assert.Empty(t, list.GetServiceImports())

	// When b's lease expires, a wins and b's endpoints leave the import.
	clk.Advance(20 * time.Second)
	srv.EnsureCluster(cs, "a", "")
	clk.Advance(15 * time.Second)
	imp, err = c.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "svc"})
	require.NoError(t, err)
	si = imp.GetServiceImport()
	assert.Equal(t, "a", si.GetLabels()["owner"])
	require.Len(t, si.GetEndpoints(), 1)
	require.Len(t, si.GetClusters(), 2)
	assert.False(t, si.GetClusters()[1].GetParticipating())
	assert.Equal(t, api.ConditionReasonClusterLeaseExpired, si.GetClusters()[1].GetConditions()[0].GetReason())

	_, err = c.exports.DeleteServiceExport(ctx(t), &mcsv1.DeleteServiceExportRequest{Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc"})
	require.NoError(t, err)
	srv.DeleteExport(cs, "b", "ns", "svc")
	_, err = c.imports.GetServiceImport(ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: cs, Namespace: "ns", Name: "svc"})
	assert.Equal(t, codes.NotFound, status.Code(err))
}

// modelSpec is the export spec of a Service labelled with the model it
// serves.
func modelSpec(model string) *mcsv1.ServiceExportSpec {
	spec := fakeregistry.ServiceSpec(fakeregistry.TCPPort("http", 8000))
	spec.Labels["model"] = model
	return spec
}

func recvImports(t *testing.T, w grpc.ServerStreamingClient[mcsv1.WatchServiceImportsResponse], n int) []*mcsv1.WatchServiceImportsResponse {
	t.Helper()
	var out []*mcsv1.WatchServiceImportsResponse
	for range n {
		ev, err := w.Recv()
		require.NoError(t, err)
		out = append(out, ev)
	}
	return out
}

func types(evs []*mcsv1.WatchServiceImportsResponse) []mcsv1.EventType {
	var out []mcsv1.EventType
	for _, ev := range evs {
		out = append(out, ev.GetType())
	}
	return out
}

const (
	start_  = mcsv1.EventType_EVENT_TYPE_SNAPSHOT_START
	upsert  = mcsv1.EventType_EVENT_TYPE_UPSERT
	del     = mcsv1.EventType_EVENT_TYPE_DELETE
	end     = mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END
	bookmrk = mcsv1.EventType_EVENT_TYPE_BOOKMARK
)

func TestWatchSnapshotDeltasFiltersAndResume(t *testing.T) {
	srv, c := start(t, fakeregistry.WithBookmarkInterval(0))
	srv.PutExport(cs, "a", "llm", "blue", modelSpec("llama-3"), fakeregistry.ReadyEndpoints("10.0.0.1"))
	llama := &mcsv1.LabelSelector{MatchLabels: map[string]string{"model": "llama-3"}}

	wctx, cancel := context.WithCancel(ctx(t))
	w, err := c.imports.WatchServiceImports(wctx, &mcsv1.WatchServiceImportsRequest{Clusterset: cs, Selector: llama})
	require.NoError(t, err)
	evs := recvImports(t, w, 3)
	assert.Equal(t, []mcsv1.EventType{start_, upsert, end}, types(evs))
	assert.Empty(t, evs[1].GetRevision())
	assert.Equal(t, srv.Revision(), evs[2].GetRevision())

	t.Log("A non-matching import is not sent; an import that stops matching is a DELETE")
	srv.PutExport(cs, "a", "llm", "red", modelSpec("mistral"), fakeregistry.ReadyEndpoints("10.0.0.2"))
	srv.PutExport(cs, "a", "llm", "blue", modelSpec("mistral"), nil)
	evs = recvImports(t, w, 1)
	assert.Equal(t, del, evs[0].GetType())
	assert.Equal(t, "blue", evs[0].GetServiceImport().GetName())
	resume := evs[0].GetRevision()
	assert.NotEmpty(t, resume)
	cancel()

	t.Log("Changes while disconnected are replayed on resume")
	srv.PutExport(cs, "a", "llm", "blue", modelSpec("llama-3"), nil)
	w, err = c.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, Selector: llama, ResumeRevision: resume})
	require.NoError(t, err)
	evs = recvImports(t, w, 1)
	assert.Equal(t, upsert, evs[0].GetType())
	assert.Equal(t, "blue", evs[0].GetServiceImport().GetName())

	t.Log("A deleted export is a DELETE carrying the last import")
	srv.DeleteExport(cs, "a", "llm", "blue")
	evs = recvImports(t, w, 1)
	assert.Equal(t, del, evs[0].GetType())
	assert.Equal(t, "llama-3", evs[0].GetServiceImport().GetLabels()["model"])

	t.Log("An unknown revision yields a new snapshot")
	w, err = c.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs, ResumeRevision: "other:5"})
	require.NoError(t, err)
	evs = recvImports(t, w, 3)
	assert.Equal(t, []mcsv1.EventType{start_, upsert, end}, types(evs))
	assert.Equal(t, "red", evs[1].GetServiceImport().GetName())
}

func TestWatchBookmarksLagAndClose(t *testing.T) {
	srv, c := start(t, fakeregistry.WithBookmarkInterval(20*time.Millisecond), fakeregistry.WithWatchQueue(1))
	w, err := c.imports.WatchServiceImports(ctx(t), &mcsv1.WatchServiceImportsRequest{Clusterset: cs})
	require.NoError(t, err)
	ev, err := w.Recv()
	require.NoError(t, err)
	assert.Equal(t, start_, ev.GetType())
	ev, err = w.Recv()
	require.NoError(t, err)
	assert.Equal(t, end, ev.GetType())
	ev, err = w.Recv()
	require.NoError(t, err)
	assert.Equal(t, bookmrk, ev.GetType())
	assert.Equal(t, srv.Revision(), ev.GetRevision())
	assert.Equal(t, 1, srv.WatcherCount())

	// Overflow the one-event queue with one commit that removes many
	// imports at once: the watcher is re-snapshotted.
	for i := range 50 {
		srv.PutExport(cs, "a", "llm", fmt.Sprintf("r%d", i), fakeregistry.ServiceSpec(fakeregistry.TCPPort("grpc", 50071)), nil)
	}
	srv.RemoveCluster(cs, "a")
	sawSnapshot := false
	for !sawSnapshot {
		ev, err = w.Recv()
		require.NoError(t, err)
		sawSnapshot = ev.GetType() == start_
	}
	for ev.GetType() != end {
		ev, err = w.Recv()
		require.NoError(t, err)
	}

	srv.CloseWatches(nil)
	for {
		_, err = w.Recv()
		if err != nil {
			break
		}
	}
	assert.Equal(t, codes.Unavailable, status.Code(err))
	assert.Equal(t, api.ReasonShuttingDown, errorInfo(t, err).GetReason())
	assert.Eventually(t, func() bool { return srv.WatcherCount() == 0 }, 5*time.Second, 10*time.Millisecond)
}

func TestWatchLeaseExpiryOnRefresh(t *testing.T) {
	clk := newClock()
	srv, c := start(t, fakeregistry.WithClock(clk.Now), fakeregistry.WithRefreshInterval(0), fakeregistry.WithBookmarkInterval(0))
	srv.EnsureCluster(cs, "a", "")
	w, err := c.clusters.WatchClusters(ctx(t), &mcsv1.WatchClustersRequest{Clusterset: cs})
	require.NoError(t, err)
	for range 3 {
		_, err = w.Recv()
		require.NoError(t, err)
	}
	clk.Advance(time.Minute)
	srv.Refresh()
	ev, err := w.Recv()
	require.NoError(t, err)
	assert.Equal(t, del, ev.GetType(), "an expired cluster leaves a watch without include_expired")
}

func TestFailNextCallCountServerInfoAndServeLocal(t *testing.T) {
	srv := fakeregistry.New()
	run, err := srv.ServeLocal()
	require.NoError(t, err)
	defer run.Stop()
	assert.NotZero(t, run.Port())
	assert.Equal(t, "http://"+run.Addr(), run.URL())
	conn, err := grpc.NewClient("passthrough:///"+run.Addr(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	defer conn.Close()
	info := mcsv1.NewServerInfoServiceClient(conn)

	srv.FailNext("GetServerInfo", codes.Unavailable)
	srv.FailNext(mcsv1.ServerInfoService_GetServerInfo_FullMethodName, codes.Internal)
	_, err = info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	assert.Equal(t, codes.Unavailable, status.Code(err))
	_, err = info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	assert.Equal(t, codes.Internal, status.Code(err))
	got, err := info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)
	assert.Equal(t, fakeregistry.DefaultRegion, got.GetRegion())
	assert.Equal(t, srv.ViewEpoch(), got.GetViewEpoch())
	assert.Equal(t, 3, srv.CallCount("GetServerInfo"))

	srv.SetServerInfo(&mcsv1.GetServerInfoResponse{
		Region:    "us-east-1",
		Endpoints: []*mcsv1.ServerEndpoint{{Region: "us-west-2", GrpcUrl: "https://registry.us-west-2.example:443"}},
	})
	got, err = info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)
	assert.Equal(t, "us-east-1", got.GetRegion())
	assert.Len(t, got.GetEndpoints(), 1)
	assert.NotNil(t, got.GetServerTime())
	srv.ResetCallCounts()
	assert.Zero(t, srv.CallCount("GetServerInfo"))

	// A restarted listener on the same address serves the same state.
	addr := run.Addr()
	run.Stop()
	run2, err := srv.ServeAddr(addr)
	require.NoError(t, err)
	defer run2.GracefulStop()
	// The first call may still go out on the connection the stop closed.
	assert.Eventually(t, func() bool {
		_, err = info.GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{}, grpc.WaitForReady(true))
		return err == nil
	}, 5*time.Second, 10*time.Millisecond)
}

func TestHealthToggle(t *testing.T) {
	srv := fakeregistry.New()
	run := srv.ServeBufconn()
	t.Cleanup(run.Stop)
	conn, err := grpc.NewClient("passthrough:///fakeregistry",
		grpc.WithContextDialer(run.Dialer()),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	hc := healthpb.NewHealthClient(conn)
	check := func(service string) healthpb.HealthCheckResponse_ServingStatus {
		t.Helper()
		resp, err := hc.Check(ctx(t), &healthpb.HealthCheckRequest{Service: service})
		require.NoError(t, err)
		return resp.GetStatus()
	}
	services := []string{"", mcsv1.ClusterService_ServiceDesc.ServiceName, mcsv1.ServiceExportService_ServiceDesc.ServiceName,
		mcsv1.ServiceImportService_ServiceDesc.ServiceName, mcsv1.ServerInfoService_ServiceDesc.ServiceName}
	for _, s := range services {
		assert.Equal(t, healthpb.HealthCheckResponse_SERVING, check(s), "serving by default: %q", s)
	}
	srv.SetServing(false)
	for _, s := range services {
		assert.Equal(t, healthpb.HealthCheckResponse_NOT_SERVING, check(s), "%q", s)
	}
	_, err = hc.Check(ctx(t), &healthpb.HealthCheckRequest{Service: "other"})
	assert.Equal(t, codes.NotFound, status.Code(err), "unknown services are not reported on")
	// RPCs are still served; a second listener shares the status.
	_, err = mcsv1.NewServerInfoServiceClient(conn).GetServerInfo(ctx(t), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)
	run2, err := srv.ServeLocal()
	require.NoError(t, err)
	t.Cleanup(run2.Stop)
	conn2, err := grpc.NewClient("passthrough:///"+run2.Addr(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn2.Close() })
	resp, err := healthpb.NewHealthClient(conn2).Check(ctx(t), &healthpb.HealthCheckRequest{})
	require.NoError(t, err)
	assert.Equal(t, healthpb.HealthCheckResponse_NOT_SERVING, resp.GetStatus())
	srv.SetServing(true)
	assert.Equal(t, healthpb.HealthCheckResponse_SERVING, check(""))
}
