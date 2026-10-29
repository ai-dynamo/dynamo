// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/fakeregistry"
)

// TestFailoverAndReturnToPrimary stops the primary replica, expects the
// client to move to the secondary with DNS re-resolved, then restarts the
// primary and expects the client and its watch to return.
func TestFailoverAndReturnToPrimary(t *testing.T) {
	primary := startFake(t, "primary")
	secondary := startFake(t, "secondary")
	dns := newFakeResolver(map[string]string{"primary.registry.test": "127.0.0.1", "secondary.registry.test": "127.0.0.1"})
	reg := prometheus.NewRegistry()
	c := newTestClient(t, Config{
		URLs:                 []string{primary.url("primary.registry.test"), secondary.url("secondary.registry.test")},
		Resolver:             dns,
		PrimaryRetryInterval: 100 * time.Millisecond,
		Registerer:           reg,
	})
	var mu sync.Mutex
	var reconnects []string
	c.OnReconnect(func(url string) {
		mu.Lock()
		defer mu.Unlock()
		reconnects = append(reconnects, url)
	})
	primaryURL, secondaryURL := primary.canonical("primary.registry.test"), secondary.canonical("secondary.registry.test")

	assert.Equal(t, "primary", region(t, c))
	assert.Equal(t, primaryURL, c.CurrentURL())
	v, _ := metricValue(t, reg, MetricConnected, map[string]string{"url": primaryURL})
	assert.Equal(t, 1.0, v)

	// A watch follows the client.
	loop := NewWatchLoop(c, WatchLoopConfig[*mcsv1.ServiceImport]{
		Name:  WatchServiceImports,
		Watch: c.ServiceImports().Watcher(&mcsv1.WatchServiceImportsRequest{Clusterset: testClusterset}),
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = loop.Run(ctx)
	}()
	defer func() {
		cancel()
		<-done
	}()
	require.Eventually(t, func() bool { return primary.srv.WatcherCount() == 1 }, 5*time.Second, 10*time.Millisecond)

	lookups := dns.count("primary.registry.test")
	primary.stop()
	require.Eventually(t, func() bool {
		info, err := c.ServerInfo().Get(testCtx(t))
		return err == nil && info.GetRegion() == "secondary"
	}, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, secondaryURL, c.CurrentURL())
	assert.Greater(t, dns.count("primary.registry.test"), lookups, "the reconnect re-resolved the primary")
	require.Eventually(t, func() bool { return secondary.srv.WatcherCount() == 1 }, 5*time.Second, 10*time.Millisecond)
	v, _ = metricValue(t, reg, MetricEndpointFailoversTotal, nil)
	assert.Equal(t, 1.0, v)
	v, _ = metricValue(t, reg, MetricConnected, map[string]string{"url": secondaryURL})
	assert.Equal(t, 1.0, v)
	v, _ = metricValue(t, reg, MetricConnected, map[string]string{"url": primaryURL})
	assert.Equal(t, 0.0, v)

	// The primary comes back: the timer probes it and returns.
	primary.restart(t)
	require.Eventually(t, func() bool { return c.CurrentURL() == primaryURL }, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, "primary", region(t, c))
	require.Eventually(t, func() bool { return primary.srv.WatcherCount() == 1 }, 5*time.Second, 10*time.Millisecond,
		"the watch moved back to the primary")
	require.Eventually(t, func() bool { return secondary.srv.WatcherCount() == 0 }, 5*time.Second, 10*time.Millisecond)
	v, _ = metricValue(t, reg, MetricEndpointFailoversTotal, nil)
	assert.Equal(t, 2.0, v)

	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reconnects) >= 2 && reconnects[len(reconnects)-1] == primaryURL
	}, 5*time.Second, 10*time.Millisecond)
	mu.Lock()
	assert.Contains(t, reconnects, secondaryURL)
	mu.Unlock()

	// Connected to the primary, the timer does nothing.
	assert.False(t, c.returnToPrimary(testCtx(t)))
}

// TestDNSReResolution moves the only registry name to another address, as a
// Route53 failover does, and expects the client to follow on reconnect.
func TestDNSReResolution(t *testing.T) {
	a := fakeregistry.New(fakeregistry.WithRegion("a"))
	b := fakeregistry.New(fakeregistry.WithRegion("b"))
	runA, runB := a.ServeBufconn(), b.ServeBufconn()
	defer runB.Stop()
	defer runA.Stop()
	listeners := map[string]func(context.Context, string) (net.Conn, error){
		"10.0.0.1:80": runA.Dialer(),
		"10.0.0.2:80": runB.Dialer(),
	}
	dialer := func(ctx context.Context, addr string) (net.Conn, error) {
		if d, ok := listeners[addr]; ok {
			return d(ctx, addr)
		}
		return nil, &net.OpError{Op: "dial", Net: "tcp", Err: os.ErrNotExist}
	}
	dns := newFakeResolver(map[string]string{"registry.test": "10.0.0.1"})
	c := newTestClient(t, Config{URLs: []string{"http://registry.test"}, Resolver: dns, Dialer: dialer, PrimaryRetryInterval: -1})

	assert.Equal(t, "a", region(t, c))
	before := dns.count("registry.test")
	assert.GreaterOrEqual(t, before, 1)

	dns.set("registry.test", "10.0.0.2")
	runA.Stop()
	require.Eventually(t, func() bool {
		info, err := c.ServerInfo().Get(testCtx(t))
		return err == nil && info.GetRegion() == "b"
	}, 10*time.Second, 20*time.Millisecond)
	assert.Greater(t, dns.count("registry.test"), before)
	assert.Len(t, c.URLs(), 1, "no fallbacks were advertised")
}

// TestRetryOnUnavailable checks the retry policy of unary calls.
func TestRetryOnUnavailable(t *testing.T) {
	f := startFake(t, "r")
	dns := newFakeResolver(map[string]string{"registry.test": "127.0.0.1"})
	c := newTestClient(t, Config{URLs: []string{f.url("registry.test")}, Resolver: dns, MaxAttempts: 4})
	req := &mcsv1.ListServiceImportsRequest{Clusterset: testClusterset}

	_, err := c.ServiceImports().List(testCtx(t), req)
	require.NoError(t, err)
	f.srv.ResetCallCounts()

	for range 3 {
		f.srv.FailNext("ListServiceImports", codes.Unavailable)
	}
	_, err = c.ServiceImports().List(testCtx(t), req)
	require.NoError(t, err, "three UNAVAILABLE then success within four attempts")
	assert.Equal(t, 4, f.srv.CallCount("ListServiceImports"))

	f.srv.ResetCallCounts()
	for range 5 {
		f.srv.FailNext("ListServiceImports", codes.Unavailable)
	}
	_, err = c.ServiceImports().List(testCtx(t), req)
	assert.Equal(t, codes.Unavailable, status.Code(err))
	assert.Equal(t, 4, f.srv.CallCount("ListServiceImports"))
	f.srv.ResetCallCounts()
	_, _ = c.ServiceImports().List(testCtx(t), req) // drains the fifth failure

	f.srv.ResetCallCounts()
	f.srv.FailNext("ListServiceImports", codes.Internal)
	_, err = c.ServiceImports().List(testCtx(t), req)
	assert.Equal(t, codes.Internal, status.Code(err), "only UNAVAILABLE is retried")
	assert.Equal(t, 1, f.srv.CallCount("ListServiceImports"))

	single := newTestClient(t, Config{URLs: []string{f.url("registry.test")}, Resolver: dns, MaxAttempts: 1})
	_, err = single.ServiceImports().List(testCtx(t), req)
	require.NoError(t, err)
	f.srv.ResetCallCounts()
	f.srv.FailNext("ListServiceImports", codes.Unavailable)
	_, err = single.ServiceImports().List(testCtx(t), req)
	assert.Equal(t, codes.Unavailable, status.Code(err))
	assert.Equal(t, 1, f.srv.CallCount("ListServiceImports"))
}

// TestFallbacksFromServerInfo learns a fallback from GetServerInfo and
// fails over to it.
func TestFallbacksFromServerInfo(t *testing.T) {
	primary := startFake(t, "primary")
	fallback := startFake(t, "fallback")
	primary.srv.SetServerInfo(&mcsv1.GetServerInfoResponse{
		Region: "primary",
		Endpoints: []*mcsv1.ServerEndpoint{
			{Region: "primary", GrpcUrl: primary.url("primary.registry.test"), Primary: true},
			{Region: "fallback", GrpcUrl: fallback.url("fallback.registry.test")},
			{Region: "tls", GrpcUrl: "https://tls.registry.test"},
			{Region: "broken", GrpcUrl: "::"},
			{Region: "empty"},
		},
	})
	dns := newFakeResolver(map[string]string{"primary.registry.test": "127.0.0.1", "fallback.registry.test": "127.0.0.1"})
	c := newTestClient(t, Config{URLs: []string{primary.url("primary.registry.test")}, Resolver: dns, PrimaryRetryInterval: -1})

	assert.Equal(t, "primary", region(t, c))
	want := []string{primary.canonical("primary.registry.test"), fallback.canonical("fallback.registry.test")}
	require.Eventually(t, func() bool { return assert.ObjectsAreEqual(want, c.URLs()) }, 5*time.Second, 10*time.Millisecond)
	require.Eventually(t, func() bool { return dns.count("fallback.registry.test") > 0 }, 5*time.Second, 10*time.Millisecond)

	primary.stop()
	require.Eventually(t, func() bool {
		info, err := c.ServerInfo().Get(testCtx(t))
		return err == nil && info.GetRegion() == "fallback"
	}, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, fallback.canonical("fallback.registry.test"), c.CurrentURL())
}

// TestTLSServerNamePerURL serves TLS for one name and checks that each URL
// is verified against its own host name.
func TestTLSServerNamePerURL(t *testing.T) {
	pki := newTestPKI(t, "registry.test")
	srv := fakeregistry.New(fakeregistry.WithRegion("tls"))
	gs := grpc.NewServer(append(srv.ServerOptions(), grpc.Creds(credentials.NewTLS(pki.serverTLS)))...)
	srv.Register(gs)
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	go func() { _ = gs.Serve(lis) }()
	defer gs.Stop()
	_, port, _ := net.SplitHostPort(lis.Addr().String())
	dns := newFakeResolver(map[string]string{"registry.test": "127.0.0.1", "other.test": "127.0.0.1"})

	c := newTestClient(t, Config{URLs: []string{"https://registry.test:" + port}, Resolver: dns, CAFile: pki.caFile})
	assert.Equal(t, "tls", region(t, c))

	wrong := newTestClient(t, Config{
		URLs: []string{"https://other.test:" + port}, Resolver: dns, CAFile: pki.caFile, MaxAttempts: 1,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err = wrong.ServerInfo().Get(ctx)
	assert.Error(t, err, "the certificate does not cover other.test")
}

// TestMetricsAndDeadline checks the metrics, their pre-initialisation and
// the default deadline of unary calls.
func TestMetricsAndDeadline(t *testing.T) {
	f := startFake(t, "m", fakeregistry.WithBookmarkInterval(0))
	dns := newFakeResolver(map[string]string{"registry.test": "127.0.0.1"})
	reg := prometheus.NewRegistry()
	var deadlines []time.Duration
	var mu sync.Mutex
	capture := grpc.WithChainUnaryInterceptor(func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if d, ok := ctx.Deadline(); ok && method != mcsv1.ServerInfoService_GetServerInfo_FullMethodName {
			mu.Lock()
			deadlines = append(deadlines, time.Until(d))
			mu.Unlock()
		}
		return invoker(ctx, method, req, reply, cc, opts...)
	})
	c := newTestClient(t, Config{
		URLs: []string{f.url("registry.test")}, Resolver: dns, Registerer: reg,
		RPCTimeout: 3 * time.Second, DialOptions: []grpc.DialOption{capture},
	})
	url := f.canonical("registry.test")

	for _, labels := range []map[string]string{
		{"rpc": "RegisterCluster", "code": "OK"},
		{"rpc": "WatchServiceImports", "code": "Unavailable"},
		{"rpc": "GetServerInfo", "code": "Unauthenticated"},
	} {
		v, ok := metricValue(t, reg, MetricRequestsTotal, labels)
		assert.True(t, ok, "pre-initialised %v", labels)
		assert.Zero(t, v)
	}
	for _, w := range []string{WatchClusters, WatchServiceImports} {
		_, ok := metricValue(t, reg, MetricWatchReconnectsTotal, map[string]string{"watch": w})
		assert.True(t, ok, w)
	}
	_, ok := metricValue(t, reg, MetricEndpointFailoversTotal, nil)
	assert.True(t, ok)
	var v float64
	_, ok = metricValue(t, reg, MetricConnected, map[string]string{"url": url})
	assert.True(t, ok)

	_, err := c.ServiceImports().List(context.Background(), &mcsv1.ListServiceImportsRequest{Clusterset: testClusterset})
	require.NoError(t, err)
	_, err = c.ServiceImports().Get(context.Background(), &mcsv1.GetServiceImportRequest{Clusterset: testClusterset, Namespace: "ns", Name: "nope"})
	assert.Equal(t, codes.NotFound, status.Code(err))
	explicit, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	_, err = c.ServiceImports().List(explicit, &mcsv1.ListServiceImportsRequest{Clusterset: testClusterset})
	require.NoError(t, err)
	mu.Lock()
	require.Len(t, deadlines, 3)
	assert.LessOrEqual(t, deadlines[0], 3*time.Second)
	assert.Greater(t, deadlines[0], 2*time.Second)
	assert.Greater(t, deadlines[2], 50*time.Second, "an explicit deadline is kept")
	mu.Unlock()

	v, _ = metricValue(t, reg, MetricRequestsTotal, map[string]string{"rpc": "ListServiceImports", "code": "OK"})
	assert.Equal(t, 2.0, v)
	v, _ = metricValue(t, reg, MetricRequestsTotal, map[string]string{"rpc": "GetServiceImport", "code": "NotFound"})
	assert.Equal(t, 1.0, v)
	v, _ = metricValue(t, reg, MetricRequestDurationSeconds, map[string]string{"rpc": "ListServiceImports"})
	assert.Equal(t, 2.0, v)
	v, _ = metricValue(t, reg, MetricConnected, map[string]string{"url": url})
	assert.Equal(t, 1.0, v)

	// A stream counts once, when it ends.
	sctx, scancel := context.WithCancel(context.Background())
	stream, err := c.ServiceImports().Watch(sctx, &mcsv1.WatchServiceImportsRequest{Clusterset: testClusterset})
	require.NoError(t, err)
	_, err = stream.Recv()
	require.NoError(t, err)
	scancel()
	for err == nil {
		_, err = stream.Recv()
	}
	v, _ = metricValue(t, reg, MetricRequestsTotal, map[string]string{"rpc": "WatchServiceImports", "code": "Canceled"})
	assert.Equal(t, 1.0, v)
	f.srv.FailNext("WatchServiceImports", codes.Unavailable)
	stream, err = c.ServiceImports().Watch(context.Background(), &mcsv1.WatchServiceImportsRequest{Clusterset: testClusterset})
	require.NoError(t, err)
	_, err = stream.Recv()
	assert.Equal(t, codes.Unavailable, status.Code(err))
	v, _ = metricValue(t, reg, MetricRequestsTotal, map[string]string{"rpc": "WatchServiceImports", "code": "Unavailable"})
	assert.Equal(t, 1.0, v)

	// A second client on the same registry shares the collectors.
	other := newTestClient(t, Config{URLs: []string{f.url("registry.test")}, Resolver: dns, Registerer: reg})
	_, err = other.ServiceImports().List(context.Background(), &mcsv1.ListServiceImportsRequest{Clusterset: testClusterset})
	require.NoError(t, err)
	v, _ = metricValue(t, reg, MetricRequestsTotal, map[string]string{"rpc": "ListServiceImports", "code": "OK"})
	assert.Equal(t, 3.0, v)

	// A conflicting collector fails New.
	conflict := prometheus.NewRegistry()
	conflict.MustRegister(prometheus.NewCounter(prometheus.CounterOpts{Name: MetricEndpointFailoversTotal, Help: "x", ConstLabels: prometheus.Labels{"x": "y"}}))
	_, err = New(context.Background(), Config{URLs: []string{"https://registry.test"}, Registerer: conflict})
	assert.Error(t, err)

	require.NoError(t, c.Close())
	require.NoError(t, c.Close(), "Close is idempotent")
}

// testPKI is a CA and a server certificate for some DNS names.
type testPKI struct {
	caFile    string
	serverTLS *tls.Config
}

func newTestPKI(t *testing.T, dnsNames ...string) testPKI {
	t.Helper()
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	caTmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "registry test CA"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}
	caDER, err := x509.CreateCertificate(rand.Reader, caTmpl, caTmpl, &caKey.PublicKey, caKey)
	require.NoError(t, err)
	caCert, err := x509.ParseCertificate(caDER)
	require.NoError(t, err)
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: dnsNames[0]},
		DNSNames:     dnsNames,
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, caCert, &key.PublicKey, caKey)
	require.NoError(t, err)
	caFile := filepath.Join(t.TempDir(), "ca.pem")
	require.NoError(t, os.WriteFile(caFile, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER}), 0o600))
	return testPKI{
		caFile: caFile,
		serverTLS: &tls.Config{
			MinVersion:   tls.VersionTLS12,
			Certificates: []tls.Certificate{{Certificate: [][]byte{der}, PrivateKey: key}},
		},
	}
}
