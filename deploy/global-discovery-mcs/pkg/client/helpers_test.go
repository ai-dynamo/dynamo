// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"io"
	"log/slog"
	"net"
	"os"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

const testClusterset = "prod"

func TestMain(m *testing.M) {
	minResolveInterval = 10 * time.Millisecond
	os.Exit(m.Run())
}

// fastBackoff keeps retries and reconnects quick in tests.
var fastBackoff = Backoff{Initial: 5 * time.Millisecond, Max: 50 * time.Millisecond, Multiplier: 2, Jitter: 0.2}

var quietLogger = slog.New(slog.NewTextHandler(io.Discard, nil))

// fakeResolver resolves host names from a map and counts lookups.
type fakeResolver struct {
	mu      sync.Mutex
	hosts   map[string][]string
	lookups map[string]int
}

func newFakeResolver(hosts map[string]string) *fakeResolver {
	f := &fakeResolver{hosts: map[string][]string{}, lookups: map[string]int{}}
	for h, ip := range hosts {
		f.hosts[h] = []string{ip}
	}
	return f
}

func (f *fakeResolver) LookupHost(_ context.Context, host string) ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.lookups[host]++
	if ips, ok := f.hosts[host]; ok {
		return slices.Clone(ips), nil
	}
	return nil, &net.DNSError{Err: "no such host", Name: host, IsNotFound: true}
}

func (f *fakeResolver) set(host string, ips ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.hosts[host] = ips
}

func (f *fakeResolver) count(host string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lookups[host]
}

// fakeServer is a fakeregistry server reporting region in GetServerInfo,
// served on 127.0.0.1.
type fakeServer struct {
	srv  *fakeregistry.Server
	run  *fakeregistry.Running
	addr string
}

func startFake(t *testing.T, region string, opts ...fakeregistry.Option) *fakeServer {
	t.Helper()
	srv := fakeregistry.New(append([]fakeregistry.Option{fakeregistry.WithRegion(region)}, opts...)...)
	run, err := srv.ServeLocal()
	require.NoError(t, err)
	f := &fakeServer{srv: srv, run: run, addr: run.Addr()}
	t.Cleanup(func() { f.run.Stop() })
	return f
}

func (f *fakeServer) port() string {
	_, p, _ := net.SplitHostPort(f.addr)
	return p
}

// url returns http://<host>:<port of this server>.
func (f *fakeServer) url(host string) string {
	return "http://" + net.JoinHostPort(host, f.port())
}

// canonical returns the canonical client URL for host on this server.
func (f *fakeServer) canonical(host string) string {
	return "http://" + net.JoinHostPort(host, f.port())
}

func (f *fakeServer) stop() { f.run.Stop() }

// restart serves the same state on the same address again.
func (f *fakeServer) restart(t *testing.T) {
	t.Helper()
	var err error
	require.Eventually(t, func() bool {
		var run *fakeregistry.Running
		run, err = f.srv.ServeAddr(f.addr)
		if err == nil {
			f.run = run
		}
		return err == nil
	}, 5*time.Second, 20*time.Millisecond, "restart on %s: %v", f.addr, err)
}

func newTestClient(t *testing.T, cfg Config) *Client {
	t.Helper()
	cfg.AllowInsecureHTTP = true
	if cfg.Backoff == (Backoff{}) {
		cfg.Backoff = fastBackoff
	}
	if cfg.Logger == nil {
		cfg.Logger = quietLogger
	}
	if cfg.ConnectTimeout == 0 {
		cfg.ConnectTimeout = time.Second
	}
	if cfg.RPCTimeout == 0 {
		cfg.RPCTimeout = 5 * time.Second
	}
	c, err := New(context.Background(), cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func testCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// region returns the region of the replica answering GetServerInfo.
func region(t *testing.T, c *Client) string {
	t.Helper()
	info, err := c.ServerInfo().Get(testCtx(t))
	require.NoError(t, err)
	return info.GetRegion()
}

// metricValue returns the value of a counter or gauge (or the sample
// count of a histogram) with the given labels, and whether it exists.
func metricValue(t *testing.T, reg *prometheus.Registry, name string, labels map[string]string) (float64, bool) {
	t.Helper()
	families, err := reg.Gather()
	require.NoError(t, err)
	for _, mf := range families {
		if mf.GetName() != name {
			continue
		}
	metrics:
		for _, m := range mf.GetMetric() {
			got := map[string]string{}
			for _, lp := range m.GetLabel() {
				got[lp.GetName()] = lp.GetValue()
			}
			for k, v := range labels {
				if got[k] != v {
					continue metrics
				}
			}
			switch {
			case m.Counter != nil:
				return m.GetCounter().GetValue(), true
			case m.Gauge != nil:
				return m.GetGauge().GetValue(), true
			case m.Histogram != nil:
				return float64(m.GetHistogram().GetSampleCount()), true
			}
		}
	}
	return 0, false
}

// exportSpec is the export spec of a Service with a grpc port.
func exportSpec(port int) *mcsv1.ServiceExportSpec {
	return fakeregistry.ServiceSpec(fakeregistry.TCPPort("grpc", int32(port)))
}
