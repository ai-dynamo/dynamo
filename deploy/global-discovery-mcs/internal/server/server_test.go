// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bufio"
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/auth"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store/memory"
)

func testConfig() *config.Config {
	c := config.Default()
	c.GRPCListenAddress = "127.0.0.1:0"
	c.HTTPListenAddress = "127.0.0.1:0"
	c.AdminListenAddress = "127.0.0.1:0"
	c.Region = "us-east-1"
	c.RefreshInterval = 100 * time.Millisecond
	c.LeaseEvalInterval = 100 * time.Millisecond
	c.BookmarkInterval = time.Second
	c.ShutdownDrain = 50 * time.Millisecond
	c.ShutdownTimeout = 2 * time.Second
	c.AdvertiseEndpoints = config.AdvertiseEndpoints{{Region: "us-east-1", GRPCURL: "https://registry.example:443", Primary: true}}
	return c
}

type running struct {
	app    *App
	cancel context.CancelFunc
	done   chan error
	logs   *observer.ObservedLogs
}

func start(t *testing.T, cfg *config.Config, opts AppOptions) *running {
	t.Helper()
	if opts.Store == nil {
		opts.Store = memory.New()
	}
	core, logs := observer.New(zapcore.DebugLevel)
	opts.Logger = zap.New(core)
	app, err := NewApp(cfg, opts)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	r := &running{app: app, cancel: cancel, done: make(chan error, 1), logs: logs}
	go func() { r.done <- app.Server.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-r.done:
			if err != nil {
				t.Logf("server stopped: %v", err)
			}
		case <-time.After(10 * time.Second):
			t.Error("server did not stop")
		}
	})
	require.Eventually(t, func() bool {
		resp, err := http.Get("http://" + app.Server.AdminAddr() + "/readyz")
		if err != nil {
			return false
		}
		_ = resp.Body.Close()
		return resp.StatusCode == http.StatusOK
	}, 10*time.Second, 20*time.Millisecond)
	return r
}

func dial(t *testing.T, addr string, creds credentials.TransportCredentials) *grpc.ClientConn {
	t.Helper()
	if creds == nil {
		creds = insecure.NewCredentials()
	}
	conn, err := grpc.NewClient("passthrough:///"+addr, grpc.WithTransportCredentials(creds))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func doJSON(t *testing.T, client *http.Client, method, url, body string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	require.NoError(t, err)
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	var out map[string]any
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&out))
	return resp.StatusCode, out
}

func TestServeGRPCRESTAndAdmin(t *testing.T) {
	r := start(t, testConfig(), AppOptions{})
	conn := dial(t, r.app.Server.GRPCAddr(), nil)
	ctx := t.Context()

	health := healthpb.NewHealthClient(conn)
	for _, svc := range []string{"", mcsv1.ServiceImportService_ServiceDesc.ServiceName, mcsv1.ClusterService_ServiceDesc.ServiceName} {
		resp, err := health.Check(ctx, &healthpb.HealthCheckRequest{Service: svc})
		require.NoError(t, err)
		assert.Equal(t, healthpb.HealthCheckResponse_SERVING, resp.GetStatus(), svc)
	}

	clusters := mcsv1.NewClusterServiceClient(conn)
	md := metadata.Pairs("x-request-id", "req-1")
	reg, err := clusters.RegisterCluster(metadata.NewOutgoingContext(ctx, md), &mcsv1.RegisterClusterRequest{Clusterset: "prod", Cluster: "a", Region: "us-east-1"})
	require.NoError(t, err)

	base := "http://" + r.app.Server.HTTPAddr()
	code, body := doJSON(t, http.DefaultClient, http.MethodPut, base+"/v1/clustersets/prod/clusters/b", `{"region":"us-west-2","leaseTtl":"20s"}`)
	require.Equal(t, http.StatusOK, code, body)
	cluster := body["cluster"].(map[string]any)
	assert.Equal(t, "us-west-2", cluster["region"])
	assert.Equal(t, "20s", cluster["lease"].(map[string]any)["ttl"])

	// REST errors carry the ErrorInfo.
	code, body = doJSON(t, http.DefaultClient, http.MethodPost, base+"/v1/clustersets/prod/clusters/a:heartbeat",
		fmt.Sprintf(`{"epoch":"%d"}`, reg.GetCluster().GetLease().GetEpoch()+1))
	assert.Equal(t, http.StatusConflict, code)
	details := body["details"].([]any)
	info := details[0].(map[string]any)
	assert.Equal(t, mcsapi.ReasonLeaseEpochMismatch, info["reason"])
	assert.Equal(t, mcsapi.ErrorDomain, info["domain"])

	req, err := http.NewRequest(http.MethodPut, base+"/v1/clustersets/prod/clusters/a/exports/llm/frontend",
		strings.NewReader(`{"spec":{"type":"SERVICE_TYPE_CLUSTER_SET_IP","ports":[{"name":"http","port":8000}]}}`))
	require.NoError(t, err)
	req.Header.Set("X-Request-Id", "rest-1")
	resp0, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	code = resp0.StatusCode
	_ = resp0.Body.Close()
	require.Equal(t, http.StatusOK, code)
	code, body = doJSON(t, http.DefaultClient, http.MethodGet, base+"/v1/clustersets/prod/imports?namespace=llm", "")
	require.Equal(t, http.StatusOK, code)
	assert.Len(t, body["serviceImports"], 1)
	code, body = doJSON(t, http.DefaultClient, http.MethodGet, base+"/v1/serverinfo", "")
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, "us-east-1", body["region"])
	assert.Equal(t, r.app.View.Epoch(), body["viewEpoch"])

	admin := "http://" + r.app.Server.AdminAddr()
	for path, want := range map[string]string{
		"/healthz":      "ok",
		"/readyz":       "ok",
		"/metrics":      "dynamo_mcs_registry_grpc_server_handled_total",
		"/debug/pprof/": "goroutine",
	} {
		resp, err := http.Get(admin + path)
		require.NoError(t, err)
		b, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode, path)
		assert.Contains(t, string(b), want, path)
	}
	resp, err := http.Get(admin + "/metrics")
	require.NoError(t, err)
	b, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	for _, m := range []string{"dynamo_mcs_registry_build_info", "dynamo_mcs_registry_clusters", "dynamo_mcs_registry_store_request_duration_seconds", "go_goroutines"} {
		assert.Contains(t, string(b), m)
	}

	// The access log carries the request fields.
	require.Eventually(t, func() bool {
		for _, e := range r.logs.FilterMessage("finished call").All() {
			f := e.ContextMap()
			if f["grpc.method"] == "RegisterCluster" && f["request_id"] == "req-1" && f["cluster"] == "a" {
				return true
			}
		}
		return false
	}, 5*time.Second, 10*time.Millisecond)
	require.Eventually(t, func() bool {
		for _, e := range r.logs.FilterMessage("finished call").All() {
			if f := e.ContextMap(); f["grpc.method"] == "PutServiceExport" && f["request_id"] == "rest-1" {
				return true
			}
		}
		return false
	}, 5*time.Second, 10*time.Millisecond, "the REST gateway forwards x-request-id")
}

// The REST gateway streams watches as newline-delimited JSON.
func TestRESTWatchStream(t *testing.T) {
	r := start(t, testConfig(), AppOptions{})
	gw := httptest.NewServer(r.app.Server.gateway.Handler)
	t.Cleanup(gw.Close)

	code, body := doJSON(t, gw.Client(), http.MethodPut, gw.URL+"/v1/clustersets/prod/clusters/a", `{"region":"us-east-1"}`)
	require.Equal(t, http.StatusOK, code, body)
	exportSpec := `{"spec":{"type":"SERVICE_TYPE_CLUSTER_SET_IP","ports":[{"name":"http","port":8000}]}}`
	code, body = doJSON(t, gw.Client(), http.MethodPut, gw.URL+"/v1/clustersets/prod/clusters/a/exports/llm/blue", exportSpec)
	require.Equal(t, http.StatusOK, code, body)

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, gw.URL+"/v1/clustersets/prod/imports:watch", nil)
	require.NoError(t, err)
	resp, err := gw.Client().Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	lines := bufio.NewScanner(resp.Body)
	next := func() map[string]any {
		t.Helper()
		require.True(t, lines.Scan(), "stream ended: %v", lines.Err())
		var ev map[string]any
		require.NoError(t, json.Unmarshal(lines.Bytes(), &ev), lines.Text())
		return ev["result"].(map[string]any)
	}
	start := next()
	assert.Equal(t, map[string]any{"type": "EVENT_TYPE_SNAPSHOT_START"}, start, "unpopulated fields are omitted")
	up := next()
	assert.Equal(t, "EVENT_TYPE_UPSERT", up["type"])
	assert.Equal(t, "blue", up["serviceImport"].(map[string]any)["name"])
	end := next()
	assert.Equal(t, "EVENT_TYPE_SNAPSHOT_END", end["type"])
	assert.NotEmpty(t, end["revision"])

	code, _ = doJSON(t, gw.Client(), http.MethodPut, gw.URL+"/v1/clustersets/prod/clusters/a/exports/llm/green", exportSpec)
	require.Equal(t, http.StatusOK, code)
	ev := next()
	assert.Equal(t, "EVENT_TYPE_UPSERT", ev["type"])
	assert.Equal(t, "green", ev["serviceImport"].(map[string]any)["name"])
	assert.Equal(t, "EVENT_TYPE_BOOKMARK", next()["type"], "a bookmark every bookmark interval")
}

func TestShutdownSequence(t *testing.T) {
	cfg := testConfig()
	cfg.ShutdownDrain = 300 * time.Millisecond
	r := start(t, cfg, AppOptions{})
	conn := dial(t, r.app.Server.GRPCAddr(), nil)
	watch, err := mcsv1.NewServiceImportServiceClient(conn).WatchServiceImports(t.Context(), &mcsv1.WatchServiceImportsRequest{Clusterset: "prod"})
	require.NoError(t, err)
	_, err = watch.Recv()
	require.NoError(t, err)

	r.cancel()
	// During the drain the replica reports not ready but still serves.
	require.Eventually(t, func() bool {
		resp, err := http.Get("http://" + r.app.Server.AdminAddr() + "/readyz")
		if err != nil {
			return false
		}
		_ = resp.Body.Close()
		return resp.StatusCode == http.StatusServiceUnavailable
	}, 5*time.Second, 10*time.Millisecond)
	hc, err := healthpb.NewHealthClient(conn).Check(t.Context(), &healthpb.HealthCheckRequest{})
	require.NoError(t, err)
	assert.Equal(t, healthpb.HealthCheckResponse_NOT_SERVING, hc.GetStatus())

	for {
		_, err = watch.Recv()
		if err != nil {
			break
		}
	}
	st, _ := status.FromError(err)
	assert.Equal(t, codes.Unavailable, st.Code())
	assert.Contains(t, st.Message(), "shutting down")
	require.NoError(t, <-r.done)
	r.done <- nil
}

func TestNotReadyWhenStoreFails(t *testing.T) {
	st := memory.New()
	r := start(t, testConfig(), AppOptions{Store: st})
	require.NoError(t, st.Close())
	require.Eventually(t, func() bool {
		ok, why := r.app.Server.Ready()
		return !ok && strings.Contains(why, "store")
	}, 5*time.Second, 10*time.Millisecond)
	conn := dial(t, r.app.Server.GRPCAddr(), nil)
	hc, err := healthpb.NewHealthClient(conn).Check(t.Context(), &healthpb.HealthCheckRequest{Service: mcsv1.ServiceImportService_ServiceDesc.ServiceName})
	require.NoError(t, err)
	assert.Equal(t, healthpb.HealthCheckResponse_NOT_SERVING, hc.GetStatus())
}

type denyImports struct{}

func (denyImports) Authenticate(_ context.Context, method string) (auth.Principal, error) {
	if strings.HasPrefix(method, "/"+mcsv1.ServiceImportService_ServiceDesc.ServiceName+"/") {
		return auth.Principal{}, errors.New("imports are off limits")
	}
	return auth.Principal{Subject: "test"}, nil
}

func TestAuthenticatorAndShutdownGuard(t *testing.T) {
	r := start(t, testConfig(), AppOptions{Authenticator: denyImports{}})
	conn := dial(t, r.app.Server.GRPCAddr(), nil)
	_, err := mcsv1.NewServiceImportServiceClient(conn).ListServiceImports(t.Context(), &mcsv1.ListServiceImportsRequest{Clusterset: "prod"})
	assert.Equal(t, codes.Unauthenticated, status.Code(err))
	_, err = mcsv1.NewServerInfoServiceClient(conn).GetServerInfo(t.Context(), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)

	r.app.Server.stopping.Store(true)
	_, err = mcsv1.NewServerInfoServiceClient(conn).GetServerInfo(t.Context(), &mcsv1.GetServerInfoRequest{})
	assert.Equal(t, codes.Unavailable, status.Code(err))
	stream, err := mcsv1.NewClusterServiceClient(conn).WatchClusters(t.Context(), &mcsv1.WatchClustersRequest{Clusterset: "prod"})
	require.NoError(t, err)
	_, err = stream.Recv()
	assert.Equal(t, codes.Unavailable, status.Code(err))
	r.app.Server.stopping.Store(false)
}

func selfSigned(t *testing.T) (certFile, keyFile string, pool *x509.CertPool) {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	require.NoError(t, err)
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "mcs-test"},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		IPAddresses:  []net.IP{net.ParseIP("127.0.0.1")},
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	require.NoError(t, err)
	keyDER, err := x509.MarshalECPrivateKey(key)
	require.NoError(t, err)
	dir := t.TempDir()
	certFile, keyFile = filepath.Join(dir, "tls.crt"), filepath.Join(dir, "tls.key")
	require.NoError(t, os.WriteFile(certFile, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	require.NoError(t, os.WriteFile(keyFile, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}), 0o600))
	cert, err := x509.ParseCertificate(der)
	require.NoError(t, err)
	pool = x509.NewCertPool()
	pool.AddCert(cert)
	return certFile, keyFile, pool
}

func TestTLS(t *testing.T) {
	cfg := testConfig()
	var pool *x509.CertPool
	cfg.TLSCertFile, cfg.TLSKeyFile, pool = selfSigned(t)
	r := start(t, cfg, AppOptions{})
	tlsConfig := &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}

	conn := dial(t, r.app.Server.GRPCAddr(), credentials.NewTLS(tlsConfig))
	info, err := mcsv1.NewServerInfoServiceClient(conn).GetServerInfo(t.Context(), &mcsv1.GetServerInfoRequest{})
	require.NoError(t, err)
	assert.Equal(t, "us-east-1", info.GetRegion())

	client := &http.Client{Transport: &http.Transport{TLSClientConfig: tlsConfig}}
	code, body := doJSON(t, client, http.MethodGet, "https://"+r.app.Server.HTTPAddr()+"/v1/serverinfo", "")
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, "us-east-1", body["region"])

	plain := dial(t, r.app.Server.GRPCAddr(), nil)
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	_, err = mcsv1.NewServerInfoServiceClient(plain).GetServerInfo(ctx, &mcsv1.GetServerInfoRequest{})
	assert.Error(t, err, "plaintext is refused on a TLS listener")
}

func TestNewErrors(t *testing.T) {
	cfg := testConfig()
	cfg.TLSCertFile, cfg.TLSKeyFile = "/nonexistent/tls.crt", "/nonexistent/tls.key"
	_, err := NewApp(cfg, AppOptions{Store: memory.New()})
	assert.ErrorContains(t, err, "load TLS key pair")

	busy, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer busy.Close()
	for _, set := range []func(c *config.Config){
		func(c *config.Config) { c.GRPCListenAddress = busy.Addr().String() },
		func(c *config.Config) { c.HTTPListenAddress = busy.Addr().String() },
		func(c *config.Config) { c.AdminListenAddress = busy.Addr().String() },
	} {
		cfg := testConfig()
		set(cfg)
		_, err := NewApp(cfg, AppOptions{Store: memory.New()})
		assert.ErrorContains(t, err, "listen")
	}
}

func TestLoggingHelpers(t *testing.T) {
	assert.Equal(t, logging.LevelDebug, codeToLevel(codes.OK))
	assert.Equal(t, logging.LevelInfo, codeToLevel(codes.FailedPrecondition))
	assert.Equal(t, logging.LevelWarn, codeToLevel(codes.Unavailable))
	assert.Equal(t, logging.LevelError, codeToLevel(codes.Internal))
	assert.Equal(t, zapcore.DebugLevel, zapLevel(logging.LevelDebug))
	assert.Equal(t, zapcore.InfoLevel, zapLevel(logging.LevelInfo))
	assert.Equal(t, zapcore.WarnLevel, zapLevel(logging.LevelWarn))
	assert.Equal(t, zapcore.ErrorLevel, zapLevel(logging.LevelError))

	core, logs := observer.New(zapcore.InfoLevel)
	l := zapLogger(zap.New(core))
	l.Log(context.Background(), logging.LevelDebug, "hidden")
	l.Log(context.Background(), logging.LevelInfo, "shown", "k", 1, 2, "v")
	require.Equal(t, 1, logs.Len())
	assert.Equal(t, map[string]any{"k": int64(1), "2": "v"}, logs.All()[0].ContextMap())

	f := fieldsFromRequest(context.Background(), &mcsv1.ListServiceImportsRequest{Clusterset: "prod", Namespace: "ns"})
	assert.Equal(t, logging.Fields{"clusterset", "prod", "namespace", "ns"}, f)
	f = fieldsFromRequest(context.Background(), &mcsv1.PutServiceExportRequest{Clusterset: "prod", Cluster: "a", Namespace: "ns", Name: "svc"})
	assert.Equal(t, logging.Fields{"clusterset", "prod", "cluster", "a", "namespace", "ns", "name", "svc"}, f)

	s := &Server{log: zap.NewNop()}
	assert.Equal(t, codes.Internal, status.Code(s.recovered(context.Background(), "boom")))
}

func TestSetupTracing(t *testing.T) {
	prev, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(prev)
		otel.SetTextMapPropagator(prevProp)
	})
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
	shutdown, err := SetupTracing(t.Context(), "mcs-test", "v0")
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = shutdown(ctx)
}

func TestOTelStatsHandler(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() {
		otel.SetTracerProvider(prev)
		otel.SetTextMapPropagator(prevProp)
	})

	cfg := testConfig()
	cfg.OTelEnabled = true
	r := start(t, cfg, AppOptions{})
	conn := dial(t, r.app.Server.GRPCAddr(), nil)
	_, err := mcsv1.NewClusterServiceClient(conn).RegisterCluster(t.Context(), &mcsv1.RegisterClusterRequest{Clusterset: "prod", Cluster: "a"})
	require.NoError(t, err)
	watch, err := mcsv1.NewServiceImportServiceClient(conn).WatchServiceImports(t.Context(), &mcsv1.WatchServiceImportsRequest{Clusterset: "prod"})
	require.NoError(t, err)
	_, err = watch.Recv()
	require.NoError(t, err)

	attrs := func(name string) map[string]string {
		for _, s := range rec.Ended() {
			if s.Name() == name {
				out := map[string]string{}
				for _, a := range s.Attributes() {
					out[string(a.Key)] = a.Value.String()
				}
				return out
			}
		}
		return nil
	}
	// A REST call joins the caller's trace.
	gw := httptest.NewServer(r.app.Server.gateway.Handler)
	t.Cleanup(gw.Close)
	req, err := http.NewRequest(http.MethodGet, gw.URL+"/v1/serverinfo", nil)
	require.NoError(t, err)
	const traceID = "4bf92f3577b34da6a3ce929d0e0e4736"
	req.Header.Set("traceparent", "00-"+traceID+"-00f067aa0ba902b7-01")
	resp, err := gw.Client().Do(req)
	require.NoError(t, err)
	_ = resp.Body.Close()
	require.Eventually(t, func() bool {
		for _, s := range rec.Ended() {
			if s.Name() == mcsv1.ServerInfoService_ServiceDesc.ServiceName+"/GetServerInfo" {
				return s.SpanContext().TraceID().String() == traceID
			}
		}
		return false
	}, 5*time.Second, 10*time.Millisecond)

	registerSpan := mcsv1.ClusterService_ServiceDesc.ServiceName + "/RegisterCluster"
	require.Eventually(t, func() bool { return attrs(registerSpan) != nil }, 5*time.Second, 10*time.Millisecond)
	a := attrs(registerSpan)
	assert.Equal(t, "prod", a["dynamo.clusterset"])
	assert.Equal(t, "a", a["dynamo.cluster.id"])
	assert.NotNil(t, attrs("mcs.store.put_cluster"), "store child span")
}
