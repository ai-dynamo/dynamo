// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry_test

import (
	"context"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/api"
	regclock "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

// startRegistry serves mcs-registry's handlers (internal/api over the memory
// store) at a fixed clock, the reference the fake is checked against.
func startRegistry(t *testing.T, now time.Time) clients {
	t.Helper()
	clk := regclock.NewFake(now)
	m := metrics.New(prometheus.NewRegistry())
	v := view.New(view.Options{Clock: clk, LeaseSkew: 2 * time.Second, Epoch: "e1", Metrics: m})
	hub := view.NewHub(v, view.HubOptions{Clock: clk, Metrics: m})
	srv := mcsapi.New(mcsapi.Options{Store: memory.New(), View: v, Hub: hub, Clock: clk, Region: "us-east-1", Version: "test"})
	lis := bufconn.Listen(1 << 20)
	gs := grpc.NewServer()
	srv.Register(gs)
	go func() { _ = gs.Serve(lis) }()
	t.Cleanup(func() {
		hub.Close()
		gs.Stop()
	})
	conn, err := grpc.NewClient("passthrough:///mcs-registry",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) { return lis.DialContext(ctx) }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return clients{
		clusters: mcsv1.NewClusterServiceClient(conn),
		exports:  mcsv1.NewServiceExportServiceClient(conn),
		imports:  mcsv1.NewServiceImportServiceClient(conn),
		info:     mcsv1.NewServerInfoServiceClient(conn),
	}
}

// TestGenerationsMatchRegistry runs the same generation cases against the
// fake and against mcs-registry: an omitted generation is assigned
// (max(stored + 1, unix milliseconds)) and returned, so repeated writes
// without one are accepted with increasing generations; a writer's own
// generation must exceed the stored one.
func TestGenerationsMatchRegistry(t *testing.T) {
	now := time.Date(2026, 10, 26, 12, 0, 0, 0, time.UTC)
	ms := uint64(now.UnixMilli())
	impls := []struct {
		name  string
		start func(t *testing.T) clients
	}{
		{"fakeregistry", func(t *testing.T) clients {
			_, c := start(t, fakeregistry.WithClock(func() time.Time { return now }), fakeregistry.WithRefreshInterval(0))
			return c
		}},
		{"mcs-registry", func(t *testing.T) clients { return startRegistry(t, now) }},
	}
	for _, impl := range impls {
		t.Run(impl.name, func(t *testing.T) {
			c := impl.start(t)
			_, err := c.clusters.RegisterCluster(ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: cs, Cluster: "a", Region: "us-east-1"})
			require.NoError(t, err)
			put := func(gen uint64) (uint64, error) {
				resp, err := c.exports.PutServiceExport(ctx(t), &mcsv1.PutServiceExportRequest{
					Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc", Generation: gen,
					Spec: fakeregistry.ServiceSpec(fakeregistry.TCPPort("http", 80)),
				})
				return resp.GetExport().GetGeneration(), err
			}
			replace := func(gen uint64) (uint64, error) {
				resp, err := c.exports.ReplaceEndpoints(ctx(t), &mcsv1.ReplaceEndpointsRequest{
					Clusterset: cs, Cluster: "a", Namespace: "ns", Name: "svc", Generation: gen,
					Endpoints: fakeregistry.ReadyEndpoints("10.0.0.1"),
				})
				return resp.GetGeneration(), err
			}
			stale := func(t *testing.T, err error, current uint64) {
				t.Helper()
				require.Equal(t, codes.FailedPrecondition, status.Code(err), "%v", err)
				info := errorInfo(t, err)
				assert.Equal(t, api.ReasonStaleGeneration, info.GetReason())
				assert.Equal(t, strconv.FormatUint(current, 10), info.GetMetadata()[api.MetadataCurrentGeneration])
			}

			t.Run("export", func(t *testing.T) {
				gen, err := put(0)
				require.NoError(t, err, "a first write without a generation is accepted")
				assert.Equal(t, ms, gen, "assigned from the server clock")
				gen, err = put(0)
				require.NoError(t, err, "a second write without a generation is accepted")
				assert.Equal(t, ms+1, gen, "stored + 1 when the clock has not moved")

				_, err = put(ms + 1)
				stale(t, err, ms+1)
				_, err = put(1)
				stale(t, err, ms+1)

				gen, err = put(ms + 10)
				require.NoError(t, err)
				assert.Equal(t, ms+10, gen, "a writer's generation is stored as sent")
				gen, err = put(0)
				require.NoError(t, err)
				assert.Equal(t, ms+11, gen, "assignment continues above the writer's generation")
			})

			t.Run("endpoints", func(t *testing.T) {
				gen, err := replace(0)
				require.NoError(t, err)
				assert.Equal(t, ms, gen)
				gen, err = replace(0)
				require.NoError(t, err)
				assert.Equal(t, ms+1, gen)

				_, err = replace(ms + 1)
				stale(t, err, ms+1)

				gen, err = replace(ms + 5)
				require.NoError(t, err)
				assert.Equal(t, ms+5, gen)
			})

			list, err := c.exports.ListServiceExports(ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: cs, Cluster: "a"})
			require.NoError(t, err)
			require.Len(t, list.GetExports(), 1)
			assert.Equal(t, ms+11, list.GetExports()[0].GetGeneration())
			assert.Equal(t, ms+5, list.GetExports()[0].GetEndpointsGeneration())
		})
	}
}
