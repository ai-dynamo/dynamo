// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api_test

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// recorder implements every registry service. Each call records the RPC name
// and request; watches send one SNAPSHOT_END event.
type recorder struct {
	mcsv1.UnimplementedClusterServiceServer
	mcsv1.UnimplementedServiceExportServiceServer
	mcsv1.UnimplementedServiceImportServiceServer
	mcsv1.UnimplementedServerInfoServiceServer

	mu  sync.Mutex
	rpc string
	req proto.Message
}

func (r *recorder) record(rpc string, req proto.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.rpc, r.req = rpc, req
}

func (r *recorder) last() (string, proto.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.rpc, r.req
}

func (r *recorder) RegisterCluster(_ context.Context, req *mcsv1.RegisterClusterRequest) (*mcsv1.RegisterClusterResponse, error) {
	r.record("RegisterCluster", req)
	return &mcsv1.RegisterClusterResponse{Cluster: &mcsv1.Cluster{Clusterset: req.Clusterset, Cluster: req.Cluster, AgentVersion: "v1"}}, nil
}

func (r *recorder) HeartbeatCluster(_ context.Context, req *mcsv1.HeartbeatClusterRequest) (*mcsv1.HeartbeatClusterResponse, error) {
	r.record("HeartbeatCluster", req)
	return &mcsv1.HeartbeatClusterResponse{Lease: &mcsv1.Lease{Epoch: req.Epoch, State: mcsv1.LeaseState_LEASE_STATE_ACTIVE}}, nil
}

func (r *recorder) DeregisterCluster(_ context.Context, req *mcsv1.DeregisterClusterRequest) (*mcsv1.DeregisterClusterResponse, error) {
	r.record("DeregisterCluster", req)
	return &mcsv1.DeregisterClusterResponse{}, nil
}

func (r *recorder) GetCluster(_ context.Context, req *mcsv1.GetClusterRequest) (*mcsv1.GetClusterResponse, error) {
	r.record("GetCluster", req)
	return &mcsv1.GetClusterResponse{Cluster: &mcsv1.Cluster{
		Clusterset: req.Clusterset,
		Cluster:    req.Cluster,
		AgentId:    "controller-0",
		Lease:      &mcsv1.Lease{Epoch: 1761480000000, State: mcsv1.LeaseState_LEASE_STATE_ACTIVE},
		Generation: 3,
	}}, nil
}

func (r *recorder) ListClusters(_ context.Context, req *mcsv1.ListClustersRequest) (*mcsv1.ListClustersResponse, error) {
	r.record("ListClusters", req)
	return &mcsv1.ListClustersResponse{}, nil
}

func (r *recorder) WatchClusters(req *mcsv1.WatchClustersRequest, s grpc.ServerStreamingServer[mcsv1.WatchClustersResponse]) error {
	r.record("WatchClusters", req)
	return s.Send(&mcsv1.WatchClustersResponse{Type: mcsv1.EventType_EVENT_TYPE_SNAPSHOT_END, Revision: "7:42"})
}

func (r *recorder) ListClusterSets(_ context.Context, req *mcsv1.ListClusterSetsRequest) (*mcsv1.ListClusterSetsResponse, error) {
	r.record("ListClusterSets", req)
	return &mcsv1.ListClusterSetsResponse{}, nil
}

func (r *recorder) PutServiceExport(_ context.Context, req *mcsv1.PutServiceExportRequest) (*mcsv1.PutServiceExportResponse, error) {
	r.record("PutServiceExport", req)
	return &mcsv1.PutServiceExportResponse{}, nil
}

func (r *recorder) DeleteServiceExport(_ context.Context, req *mcsv1.DeleteServiceExportRequest) (*mcsv1.DeleteServiceExportResponse, error) {
	r.record("DeleteServiceExport", req)
	return &mcsv1.DeleteServiceExportResponse{}, nil
}

func (r *recorder) ListServiceExports(_ context.Context, req *mcsv1.ListServiceExportsRequest) (*mcsv1.ListServiceExportsResponse, error) {
	r.record("ListServiceExports", req)
	return &mcsv1.ListServiceExportsResponse{}, nil
}

func (r *recorder) ReplaceEndpoints(_ context.Context, req *mcsv1.ReplaceEndpointsRequest) (*mcsv1.ReplaceEndpointsResponse, error) {
	r.record("ReplaceEndpoints", req)
	return &mcsv1.ReplaceEndpointsResponse{}, nil
}

func (r *recorder) GetServiceImport(_ context.Context, req *mcsv1.GetServiceImportRequest) (*mcsv1.GetServiceImportResponse, error) {
	r.record("GetServiceImport", req)
	return &mcsv1.GetServiceImportResponse{}, nil
}

func (r *recorder) ListServiceImports(_ context.Context, req *mcsv1.ListServiceImportsRequest) (*mcsv1.ListServiceImportsResponse, error) {
	r.record("ListServiceImports", req)
	return &mcsv1.ListServiceImportsResponse{}, nil
}

func (r *recorder) WatchServiceImports(req *mcsv1.WatchServiceImportsRequest, s grpc.ServerStreamingServer[mcsv1.WatchServiceImportsResponse]) error {
	r.record("WatchServiceImports", req)
	if err := s.Send(&mcsv1.WatchServiceImportsResponse{
		Type: mcsv1.EventType_EVENT_TYPE_UPSERT, ServiceImport: &mcsv1.ServiceImport{Namespace: "llm", Name: "router"}, Revision: "7:41",
	}); err != nil {
		return err
	}
	return s.Send(&mcsv1.WatchServiceImportsResponse{Type: mcsv1.EventType_EVENT_TYPE_BOOKMARK, Revision: "7:42"})
}

func (r *recorder) GetServerInfo(_ context.Context, req *mcsv1.GetServerInfoRequest) (*mcsv1.GetServerInfoResponse, error) {
	r.record("GetServerInfo", req)
	return &mcsv1.GetServerInfoResponse{Region: "us-east-1"}, nil
}

// newGateway serves the recorder over an in-memory gRPC connection and
// returns a grpc-gateway mux dialling it, as the server's REST listener does.
func newGateway(t *testing.T) (*recorder, http.Handler) {
	t.Helper()
	rec := &recorder{}
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	mcsv1.RegisterClusterServiceServer(srv, rec)
	mcsv1.RegisterServiceExportServiceServer(srv, rec)
	mcsv1.RegisterServiceImportServiceServer(srv, rec)
	mcsv1.RegisterServerInfoServiceServer(srv, rec)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })

	ctx := t.Context()
	mux := runtime.NewServeMux()
	require.NoError(t, mcsv1.RegisterClusterServiceHandler(ctx, mux, conn))
	require.NoError(t, mcsv1.RegisterServiceExportServiceHandler(ctx, mux, conn))
	require.NoError(t, mcsv1.RegisterServiceImportServiceHandler(ctx, mux, conn))
	require.NoError(t, mcsv1.RegisterServerInfoServiceHandler(ctx, mux, conn))
	return rec, mux
}

// TestRESTMapping pins the REST paths of the API to their RPCs and checks
// that path, query and body fields reach the request.
func TestRESTMapping(t *testing.T) {
	rec, mux := newGateway(t)
	const base = "/v1/clustersets/cs1"

	tests := []struct {
		method, path, body string
		rpc                string
		want               proto.Message
	}{
		{"PUT", base + "/clusters/c1", `{"region":"us-east-1","leaseTtl":"45s"}`, "RegisterCluster",
			&mcsv1.RegisterClusterRequest{Clusterset: "cs1", Cluster: "c1", Region: "us-east-1", LeaseTtl: durationpbSeconds(45)}},
		{"POST", base + "/clusters/c1:heartbeat", `{"epoch":"1761480000000"}`, "HeartbeatCluster",
			&mcsv1.HeartbeatClusterRequest{Clusterset: "cs1", Cluster: "c1", Epoch: 1761480000000}},
		{"DELETE", base + "/clusters/c1", "", "DeregisterCluster",
			&mcsv1.DeregisterClusterRequest{Clusterset: "cs1", Cluster: "c1"}},
		{"GET", base + "/clusters/c1", "", "GetCluster",
			&mcsv1.GetClusterRequest{Clusterset: "cs1", Cluster: "c1"}},
		{"GET", base + "/clusters?includeExpired=true", "", "ListClusters",
			&mcsv1.ListClustersRequest{Clusterset: "cs1", IncludeExpired: true}},
		{"GET", base + "/clusters:watch?resumeRevision=7:40", "", "WatchClusters",
			&mcsv1.WatchClustersRequest{Clusterset: "cs1", ResumeRevision: "7:40"}},
		{"GET", "/v1/clustersets", "", "ListClusterSets",
			&mcsv1.ListClusterSetsRequest{}},
		{"PUT", base + "/clusters/c1/exports/llm/router", `{"generation":"5","spec":{"type":"SERVICE_TYPE_HEADLESS"}}`, "PutServiceExport",
			&mcsv1.PutServiceExportRequest{Clusterset: "cs1", Cluster: "c1", Namespace: "llm", Name: "router", Generation: 5,
				Spec: &mcsv1.ServiceExportSpec{Type: mcsv1.ServiceType_SERVICE_TYPE_HEADLESS}}},
		{"DELETE", base + "/clusters/c1/exports/llm/router", "", "DeleteServiceExport",
			&mcsv1.DeleteServiceExportRequest{Clusterset: "cs1", Cluster: "c1", Namespace: "llm", Name: "router"}},
		{"GET", base + "/clusters/c1/exports", "", "ListServiceExports",
			&mcsv1.ListServiceExportsRequest{Clusterset: "cs1", Cluster: "c1"}},
		{"PUT", base + "/clusters/c1/exports/llm/router/endpoints", `{"generation":"6","endpoints":[{"addresses":["10.0.0.1"],"conditions":{"ready":true}}]}`, "ReplaceEndpoints",
			&mcsv1.ReplaceEndpointsRequest{Clusterset: "cs1", Cluster: "c1", Namespace: "llm", Name: "router", Generation: 6,
				Endpoints: []*mcsv1.Endpoint{{Addresses: []string{"10.0.0.1"}, Conditions: &mcsv1.EndpointConditions{Ready: true}}}}},
		{"GET", base + "/imports/llm/router?includeEndpoints=true", "", "GetServiceImport",
			&mcsv1.GetServiceImportRequest{Clusterset: "cs1", Namespace: "llm", Name: "router", IncludeEndpoints: true}},
		{"GET", base + "/imports?namespace=llm", "", "ListServiceImports",
			&mcsv1.ListServiceImportsRequest{Clusterset: "cs1", Namespace: "llm"}},
		{"GET", base + "/imports:watch?includeEndpoints=true", "", "WatchServiceImports",
			&mcsv1.WatchServiceImportsRequest{Clusterset: "cs1", IncludeEndpoints: true}},
		{"GET", "/v1/serverinfo", "", "GetServerInfo",
			&mcsv1.GetServerInfoRequest{}},
	}
	for _, tt := range tests {
		t.Run(tt.rpc, func(t *testing.T) {
			resp := serve(t, mux, tt.method, tt.path, tt.body)
			require.Equal(t, http.StatusOK, resp.Code, resp.Body.String())
			rpc, req := rec.last()
			assert.Equal(t, tt.rpc, rpc)
			assert.True(t, proto.Equal(tt.want, req), "got %v, want %v", req, tt.want)
		})
	}
}

// TestJSONShape checks the proto3 JSON mapping clients rely on.
func TestJSONShape(t *testing.T) {
	_, mux := newGateway(t)
	resp := serve(t, mux, "GET", "/v1/clustersets/cs1/clusters/c1", "")
	require.Equal(t, http.StatusOK, resp.Code)
	body := resp.Body.String()
	assert.Contains(t, body, `"agentId":"controller-0"`, "lowerCamelCase field names")
	assert.Contains(t, body, `"state":"LEASE_STATE_ACTIVE"`, "enums as strings")
	assert.Contains(t, body, `"epoch":"1761480000000"`, "64-bit integers as strings")
	assert.Contains(t, body, `"generation":"3"`)
}

// TestWatchStreamShape checks that watches stream newline-delimited
// {"result": {...}} objects over REST.
func TestWatchStreamShape(t *testing.T) {
	_, mux := newGateway(t)
	resp := serve(t, mux, "GET", "/v1/clustersets/cs1/imports:watch", "")
	require.Equal(t, http.StatusOK, resp.Code)
	lines := strings.Split(strings.TrimSpace(resp.Body.String()), "\n")
	require.Len(t, lines, 2, resp.Body.String())
	assert.True(t, strings.HasPrefix(lines[0], `{"result":{`), lines[0])
	assert.Contains(t, lines[0], `"type":"EVENT_TYPE_UPSERT"`)
	assert.Contains(t, lines[0], `"serviceImport":{`)
	assert.Contains(t, lines[0], `"name":"router"`)
	assert.Contains(t, lines[1], `"type":"EVENT_TYPE_BOOKMARK"`)
	assert.Contains(t, lines[1], `"revision":"7:42"`)
}

// TestUnknownPathIsNotFound guards against accidental catch-all routes.
func TestUnknownPathIsNotFound(t *testing.T) {
	_, mux := newGateway(t)
	assert.Equal(t, http.StatusNotFound, serve(t, mux, "GET", "/v1/clustersets/cs1/nope", "").Code)
	assert.Equal(t, http.StatusNotImplemented, serve(t, mux, "PUT", "/v1/clustersets/cs1/imports/llm/router", "{}").Code)
}

func serve(t *testing.T, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	req := httptest.NewRequestWithContext(t.Context(), method, path, r)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	resp := httptest.NewRecorder()
	h.ServeHTTP(resp, req)
	return resp
}

func durationpbSeconds(s int64) *durationpb.Duration {
	return durationpb.New(time.Duration(s) * time.Second)
}
