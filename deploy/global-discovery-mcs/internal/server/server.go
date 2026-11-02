// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package server runs the mcs-registry listeners:
//
//   - gRPC (h2c, or TLS when configured), the NLB target, with keepalive,
//     message size limits, the gRPC health service per service name and the
//     interceptor chain: Prometheus (go-grpc-middleware provider), zap
//     logging, request fields on logs and spans, the shutdown guard,
//     authentication and panic recovery, plus the otelgrpc stats handler
//     when OpenTelemetry is enabled;
//   - the REST gateway (grpc-gateway), which dials the gRPC server over a
//     private loopback listener so that server streams work;
//   - admin: /metrics, /healthz, /readyz and /debug/pprof.
//
// Readiness requires a store ping, one completed refresh of every clusterset
// the store knows and a view younger than three refresh intervals. Shutdown:
// not ready and health NOT_SERVING, drain so the load balancer deregisters
// the replica, end watches with UNAVAILABLE SHUTTING_DOWN, stop the REST
// gateway and the gRPC server gracefully within the shutdown timeout, then
// stop the background loops.
package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	grpcprom "github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/recovery"
	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/auth"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/view"
)

// Keepalive settings. The server pings an idle connection after a minute,
// well inside the 350 s NLB idle timeout, and lets clients ping every 10 s
// even without an active stream.
var (
	keepaliveParams = keepalive.ServerParameters{Time: 60 * time.Second, Timeout: 20 * time.Second}
	keepalivePolicy = keepalive.EnforcementPolicy{MinTime: 10 * time.Second, PermitWithoutStream: true}
)

// Options configures a Server.
type Options struct {
	Config    *config.Config
	Store     store.Store
	View      *view.View
	Hub       *view.Hub
	Refresher *view.Refresher
	Compactor *view.Compactor
	API       *api.Server
	Metrics   *metrics.Metrics
	// Registry receives the gRPC server metrics and is served on /metrics.
	Registry *prometheus.Registry
	// Authenticator defaults to auth.NoopAuthenticator.
	Authenticator auth.Authenticator
	Clock         clock.Clock
	Logger        *zap.Logger
}

// Server owns the listeners and the shutdown sequence.
type Server struct {
	opts Options
	cfg  *config.Config
	log  *zap.Logger

	grpc    *grpc.Server
	health  *health.Server
	gateway *http.Server
	admin   *http.Server

	grpcLis, loopLis, httpLis, adminLis net.Listener

	// draining is set at the start of shutdown: not ready, health
	// NOT_SERVING.
	draining atomic.Bool
	// stopping is set after the drain: new calls fail with SHUTTING_DOWN.
	stopping atomic.Bool

	readyMu   sync.Mutex
	ready     bool
	notReady  string
	everReady atomic.Bool
}

// startupRetry is how often readiness is evaluated before the replica is
// ready for the first time.
const startupRetry = 250 * time.Millisecond

// New builds the server and binds its listeners, so that the addresses are
// known before Run (listen addresses may use port 0).
func New(opts Options) (*Server, error) {
	if opts.Authenticator == nil {
		opts.Authenticator = auth.NoopAuthenticator{}
	}
	if opts.Clock == nil {
		opts.Clock = clock.Real()
	}
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	s := &Server{opts: opts, cfg: opts.Config, log: opts.Logger, notReady: "starting"}
	if err := s.listen(); err != nil {
		s.closeListeners()
		return nil, err
	}
	if err := s.build(); err != nil {
		s.closeListeners()
		return nil, err
	}
	return s, nil
}

func (s *Server) listen() error {
	var tlsConfig *tls.Config
	if s.cfg.TLSCertFile != "" {
		cert, err := tls.LoadX509KeyPair(s.cfg.TLSCertFile, s.cfg.TLSKeyFile)
		if err != nil {
			return fmt.Errorf("load TLS key pair: %w", err)
		}
		tlsConfig = &tls.Config{Certificates: []tls.Certificate{cert}, MinVersion: tls.VersionTLS12}
	}
	var err error
	if s.grpcLis, err = net.Listen("tcp", s.cfg.GRPCListenAddress); err != nil {
		return fmt.Errorf("listen gRPC: %w", err)
	}
	if s.loopLis, err = net.Listen("tcp", "127.0.0.1:0"); err != nil {
		return fmt.Errorf("listen gateway loopback: %w", err)
	}
	if s.httpLis, err = net.Listen("tcp", s.cfg.HTTPListenAddress); err != nil {
		return fmt.Errorf("listen REST: %w", err)
	}
	if s.adminLis, err = net.Listen("tcp", s.cfg.AdminListenAddress); err != nil {
		return fmt.Errorf("listen admin: %w", err)
	}
	if tlsConfig != nil {
		// TLS terminates on the listener, so gRPC itself runs without
		// transport credentials on both the public and the loopback
		// listener.
		grpcTLS := tlsConfig.Clone()
		grpcTLS.NextProtos = []string{"h2"}
		s.grpcLis = tls.NewListener(s.grpcLis, grpcTLS)
		httpTLS := tlsConfig.Clone()
		httpTLS.NextProtos = []string{"h2", "http/1.1"}
		s.httpLis = tls.NewListener(s.httpLis, httpTLS)
	}
	return nil
}

func (s *Server) closeListeners() {
	for _, l := range []net.Listener{s.grpcLis, s.loopLis, s.httpLis, s.adminLis} {
		if l != nil {
			_ = l.Close()
		}
	}
}

// GRPCAddr returns the bound gRPC address.
func (s *Server) GRPCAddr() string { return s.grpcLis.Addr().String() }

// HTTPAddr returns the bound REST gateway address.
func (s *Server) HTTPAddr() string { return s.httpLis.Addr().String() }

// AdminAddr returns the bound admin address.
func (s *Server) AdminAddr() string { return s.adminLis.Addr().String() }

// shutdownGuard refuses calls once the replica stops serving.
func (s *Server) shutdownGuardUnary(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	if s.stopping.Load() && !strings.HasPrefix(info.FullMethod, "/grpc.health.v1.Health/") {
		return nil, api.ShuttingDownError()
	}
	return handler(ctx, req)
}

func (s *Server) shutdownGuardStream(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	if s.stopping.Load() && !strings.HasPrefix(info.FullMethod, "/grpc.health.v1.Health/") {
		return api.ShuttingDownError()
	}
	return handler(srv, ss)
}

func (s *Server) recovered(_ context.Context, p any) error {
	s.log.Error("panic in gRPC handler", zap.Any("panic", p), zap.Stack("stack"))
	return api.InternalError("internal error")
}

func (s *Server) build() error {
	srvMetrics := grpcprom.NewServerMetrics(
		grpcprom.WithServerCounterOptions(grpcprom.WithNamespace(metrics.Namespace)),
		grpcprom.WithServerHandlingTimeHistogram(grpcprom.WithHistogramNamespace(metrics.Namespace)),
	)
	if s.opts.Registry != nil {
		if err := s.opts.Registry.Register(srvMetrics); err != nil {
			return fmt.Errorf("register gRPC metrics: %w", err)
		}
	}
	logger := zapLogger(s.log.With(zap.String("component", "grpc")))
	logOpts := []logging.Option{
		logging.WithLogOnEvents(logging.FinishCall),
		logging.WithLevels(codeToLevel),
	}
	recoveryOpt := recovery.WithRecoveryHandlerContext(s.recovered)
	serverOpts := []grpc.ServerOption{
		grpc.KeepaliveParams(keepaliveParams),
		grpc.KeepaliveEnforcementPolicy(keepalivePolicy),
		grpc.MaxRecvMsgSize(s.cfg.MaxMessageBytes),
		grpc.MaxSendMsgSize(s.cfg.MaxMessageBytes),
		grpc.ChainUnaryInterceptor(
			srvMetrics.UnaryServerInterceptor(),
			logging.UnaryServerInterceptor(logger, logOpts...),
			requestFieldsUnary,
			s.shutdownGuardUnary,
			auth.UnaryServerInterceptor(s.opts.Authenticator, auth.DefaultExempt...),
			recovery.UnaryServerInterceptor(recoveryOpt),
		),
		grpc.ChainStreamInterceptor(
			srvMetrics.StreamServerInterceptor(),
			logging.StreamServerInterceptor(logger, logOpts...),
			requestFieldsStream,
			s.shutdownGuardStream,
			auth.StreamServerInterceptor(s.opts.Authenticator, auth.DefaultExempt...),
			recovery.StreamServerInterceptor(recoveryOpt),
		),
	}
	if s.cfg.OTelEnabled {
		serverOpts = append(serverOpts, grpc.StatsHandler(otelgrpc.NewServerHandler()))
	}
	s.grpc = grpc.NewServer(serverOpts...)
	s.opts.API.Register(s.grpc)
	s.health = health.NewServer()
	healthpb.RegisterHealthServer(s.grpc, s.health)
	s.setHealth(healthpb.HealthCheckResponse_NOT_SERVING)
	srvMetrics.InitializeMetrics(s.grpc)

	gw, err := s.gatewayHandler()
	if err != nil {
		return err
	}
	s.gateway = &http.Server{Handler: gw, ReadHeaderTimeout: 10 * time.Second}
	s.admin = &http.Server{Handler: s.adminHandler(), ReadHeaderTimeout: 10 * time.Second}
	return nil
}

func (s *Server) setHealth(st healthpb.HealthCheckResponse_ServingStatus) {
	s.health.SetServingStatus("", st)
	for _, name := range api.ServiceNames {
		s.health.SetServingStatus(name, st)
	}
}

// gatewayHandler returns the REST gateway: a grpc-gateway mux whose
// handlers call the gRPC server through the loopback listener.
func (s *Server) gatewayHandler() (http.Handler, error) {
	conn, err := grpc.NewClient("passthrough:///"+s.loopLis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(s.cfg.MaxMessageBytes), grpc.MaxCallSendMsgSize(s.cfg.MaxMessageBytes)),
	)
	if err != nil {
		return nil, fmt.Errorf("dial gateway loopback: %w", err)
	}
	// Proto3 canonical JSON: lowerCamelCase names, enum names, unpopulated
	// fields omitted; unknown input fields are ignored so that newer
	// clients work against older servers.
	mux := runtime.NewServeMux(
		runtime.WithMarshalerOption(runtime.MIMEWildcard, &runtime.HTTPBodyMarshaler{
			Marshaler: &runtime.JSONPb{
				EmitUnpopulated: false,
				DiscardUnknown:  true,
			},
		}),
		runtime.WithIncomingHeaderMatcher(gatewayHeader),
	)
	ctx := context.Background()
	for _, register := range []func(context.Context, *runtime.ServeMux, *grpc.ClientConn) error{
		mcsv1.RegisterClusterServiceHandler,
		mcsv1.RegisterServiceExportServiceHandler,
		mcsv1.RegisterServiceImportServiceHandler,
		mcsv1.RegisterServerInfoServiceHandler,
	} {
		if err := register(ctx, mux, conn); err != nil {
			return nil, fmt.Errorf("register gateway handler: %w", err)
		}
	}
	return mux, nil
}

// gatewayHeader forwards the W3C trace context and x-request-id of REST
// requests to the gRPC call, so that the server span joins the caller's
// trace and the access log carries the request id.
func gatewayHeader(key string) (string, bool) {
	switch k := strings.ToLower(key); k {
	case "traceparent", "tracestate", "baggage", "x-request-id":
		return k, true
	}
	return runtime.DefaultHeaderMatcher(key)
}

func (s *Server) adminHandler() http.Handler {
	mux := http.NewServeMux()
	reg := s.opts.Registry
	if reg == nil {
		reg = prometheus.NewRegistry()
	}
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{Registry: reg}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		ok, why := s.Ready()
		if !ok {
			http.Error(w, "not ready: "+why, http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return mux
}

// Ready returns the last readiness evaluation and, when not ready, why.
func (s *Server) Ready() (bool, string) {
	if s.draining.Load() {
		return false, "shutting down"
	}
	s.readyMu.Lock()
	defer s.readyMu.Unlock()
	return s.ready, s.notReady
}

// evaluateReadiness pings the store and checks the view's freshness, then
// publishes the result to /readyz and the gRPC health service.
func (s *Server) evaluateReadiness(ctx context.Context) {
	var ready bool
	var why string
	pingCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	err := s.opts.Store.Ping(pingCtx)
	cancel()
	if err != nil {
		ready, why = false, "store: "+err.Error()
	} else {
		ready, why = s.opts.View.Ready(s.opts.Clock.Now(), 3*s.cfg.RefreshInterval)
	}
	s.readyMu.Lock()
	changed := ready != s.ready
	s.ready, s.notReady = ready, why
	s.readyMu.Unlock()
	if ready {
		s.everReady.Store(true)
	}
	if s.draining.Load() {
		return
	}
	if changed {
		if ready {
			s.log.Info("replica ready")
		} else {
			s.log.Warn("replica not ready", zap.String("reason", why))
		}
	}
	if ready {
		s.setHealth(healthpb.HealthCheckResponse_SERVING)
	} else {
		s.setHealth(healthpb.HealthCheckResponse_NOT_SERVING)
	}
}

// Run serves until ctx is done, then shuts down gracefully. It returns the
// first serving error, if any.
func (s *Server) Run(ctx context.Context) error {
	loops, stopLoops := context.WithCancel(context.Background())
	defer stopLoops()
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	serve := func(name string, fn func() error) {
		wg.Go(func() {
			if err := fn(); err != nil && !errors.Is(err, http.ErrServerClosed) && !errors.Is(err, grpc.ErrServerStopped) {
				errs <- fmt.Errorf("%s: %w", name, err)
			}
		})
	}
	serve("gRPC", func() error { return s.grpc.Serve(s.grpcLis) })
	serve("gateway loopback", func() error { return s.grpc.Serve(s.loopLis) })
	serve("REST", func() error { return s.gateway.Serve(s.httpLis) })
	serve("admin", func() error { return s.admin.Serve(s.adminLis) })

	var loopsWG sync.WaitGroup
	loopsWG.Add(3)
	go func() {
		defer loopsWG.Done()
		s.opts.Refresher.Run(loops)
	}()
	go func() {
		defer loopsWG.Done()
		if s.opts.Compactor != nil {
			s.opts.Compactor.Run(loops)
		}
	}()
	go func() {
		defer loopsWG.Done()
		s.readinessLoop(loops)
	}()
	s.log.Info("mcs-registry serving", zap.String("grpc", s.GRPCAddr()), zap.String("http", s.HTTPAddr()),
		zap.String("admin", s.AdminAddr()), zap.Bool("tls", s.cfg.TLSCertFile != ""))

	var serveErr error
	select {
	case <-ctx.Done():
	case serveErr = <-errs:
		s.log.Error("listener failed; shutting down", zap.Error(serveErr))
	}
	s.shutdown()
	stopLoops()
	loopsWG.Wait()
	wg.Wait()
	return serveErr
}

func (s *Server) readinessLoop(ctx context.Context) {
	s.evaluateReadiness(ctx)
	t := s.opts.Clock.NewTicker(s.cfg.RefreshInterval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			s.evaluateReadiness(ctx)
		case <-time.After(startupRetry):
			// Until the replica is ready for the first time, retry quickly
			// so that it does not wait a full interval after the first
			// refresh.
			if !s.everReady.Load() && !s.draining.Load() {
				s.evaluateReadiness(ctx)
			}
		}
	}
}

// shutdown runs the graceful shutdown sequence: not ready, drain, end the
// watches, stop the servers.
func (s *Server) shutdown() {
	s.log.Info("shutting down: draining", zap.Duration("drain", s.cfg.ShutdownDrain))
	s.draining.Store(true)
	s.health.Shutdown()
	if s.cfg.ShutdownDrain > 0 {
		t := s.opts.Clock.NewTimer(s.cfg.ShutdownDrain)
		<-t.C()
	}

	s.stopping.Store(true)
	s.opts.Hub.Close()

	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.ShutdownTimeout)
	defer cancel()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := s.gateway.Shutdown(ctx); err != nil {
			s.log.Warn("REST gateway did not stop gracefully", zap.Error(err))
			_ = s.gateway.Close()
		}
	}()
	go func() {
		defer wg.Done()
		done := make(chan struct{})
		go func() {
			s.grpc.GracefulStop()
			close(done)
		}()
		select {
		case <-done:
		case <-ctx.Done():
			s.log.Warn("gRPC server did not stop within the shutdown timeout; closing connections",
				zap.Duration("timeout", s.cfg.ShutdownTimeout))
			s.grpc.Stop()
			<-done
		}
	}()
	wg.Wait()
	adminCtx, adminCancel := context.WithTimeout(context.Background(), time.Second)
	defer adminCancel()
	_ = s.admin.Shutdown(adminCtx)
	s.log.Info("mcs-registry stopped")
}
