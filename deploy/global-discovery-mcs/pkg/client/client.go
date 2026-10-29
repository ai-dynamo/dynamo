// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

var clientSeq atomic.Uint64

// Client is a registry client: one gRPC ClientConn with endpoint failover,
// typed sub-clients, metrics and reconnect hooks. It is safe for concurrent
// use. Close it when done.
type Client struct {
	cfg            Config
	targets        []target
	creds          credentials.TransportCredentials
	resolverScheme string
	conn           *grpc.ClientConn
	metrics        *metrics
	logger         *slog.Logger

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	reconnectCh  chan struct{}
	serverInfoCh chan struct{}

	mu               sync.Mutex
	res              *registryResolver
	fallbacks        []target
	unhealthy        map[string]*unhealthyURL
	hooks            []func(url string)
	current          *connInfo
	lastURL          string
	everConnected    bool
	pendingReconnect string
	watchRestart     chan struct{}
	closeOnce        sync.Once
	closeErr         error

	clusters       *ClusterClient
	serviceExports *ServiceExportClient
	serviceImports *ServiceImportClient
	serverInfo     *ServerInfoClient
}

// New validates cfg and returns a Client. It does not wait for a
// connection: the client connects in the background, and calls wait for it
// within their deadline. ctx only bounds construction.
func New(ctx context.Context, cfg Config) (*Client, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cfg, targets, err := normalize(cfg)
	if err != nil {
		return nil, err
	}
	creds, err := transportCredentials(cfg, targets[0].scheme)
	if err != nil {
		return nil, err
	}
	urls := make([]string, len(targets))
	for i, t := range targets {
		urls[i] = t.url
	}
	m, err := newMetrics(cfg.Registerer, urls)
	if err != nil {
		return nil, fmt.Errorf("mcs client: register metrics: %w", err)
	}
	c := &Client{
		cfg:            cfg,
		targets:        targets,
		creds:          creds,
		resolverScheme: "mcs-client-" + strconv.FormatUint(clientSeq.Add(1), 10),
		metrics:        m,
		logger:         cfg.Logger,
		unhealthy:      map[string]*unhealthyURL{},
		reconnectCh:    make(chan struct{}, 1),
		serverInfoCh:   make(chan struct{}, 1),
		watchRestart:   make(chan struct{}),
	}
	c.ctx, c.cancel = context.WithCancel(context.Background())
	opts := append(c.baseDialOptions(),
		grpc.WithResolvers(&resolverBuilder{c: c}),
		grpc.WithStatsHandler(&connTracker{c: c}),
		grpc.WithDefaultServiceConfig(serviceConfig(cfg)),
		grpc.WithChainUnaryInterceptor(c.unaryInterceptor),
		grpc.WithChainStreamInterceptor(c.streamInterceptor),
	)
	opts = append(opts, cfg.DialOptions...)
	conn, err := grpc.NewClient(c.resolverScheme+":///registry", opts...)
	if err != nil {
		c.cancel()
		return nil, fmt.Errorf("mcs client: create client: %w", err)
	}
	c.conn = conn
	c.clusters = &ClusterClient{c: mcsv1.NewClusterServiceClient(conn)}
	c.serviceExports = &ServiceExportClient{c: mcsv1.NewServiceExportServiceClient(conn)}
	c.serviceImports = &ServiceImportClient{c: mcsv1.NewServiceImportServiceClient(conn)}
	c.serverInfo = &ServerInfoClient{c: mcsv1.NewServerInfoServiceClient(conn)}
	c.wg.Add(3)
	go c.hookLoop()
	go c.serverInfoLoop()
	go c.primaryLoop()
	conn.Connect()
	return c, nil
}

// baseDialOptions are shared by the main connection and primary probes.
func (c *Client) baseDialOptions() []grpc.DialOption {
	bc := backoff.Config{
		BaseDelay:  c.cfg.Backoff.Initial,
		Multiplier: c.cfg.Backoff.Multiplier,
		Jitter:     c.cfg.Backoff.Jitter,
		MaxDelay:   c.cfg.Backoff.Max,
	}
	return []grpc.DialOption{
		grpc.WithTransportCredentials(c.creds),
		grpc.WithContextDialer(c.dial),
		grpc.WithUserAgent(c.cfg.UserAgent),
		grpc.WithConnectParams(grpc.ConnectParams{Backoff: bc, MinConnectTimeout: c.cfg.ConnectTimeout}),
	}
}

// serviceConfig returns the default service config: pick_first in address
// order and, unless MaxAttempts is 1, a retry policy on UNAVAILABLE for
// every unary registry method. Watches are not retried by gRPC; WatchLoop
// reconnects them.
func serviceConfig(cfg Config) string {
	sc := map[string]any{
		"loadBalancingConfig": []any{map[string]any{"pick_first": map[string]any{"shuffleAddressList": false}}},
	}
	if cfg.MaxAttempts > 1 {
		var names []map[string]string
		for _, sd := range serviceDescs {
			for _, m := range sd.Methods {
				names = append(names, map[string]string{"service": sd.ServiceName, "method": m.MethodName})
			}
		}
		seconds := func(d time.Duration) string { return strconv.FormatFloat(d.Seconds(), 'f', -1, 64) + "s" }
		sc["methodConfig"] = []any{map[string]any{
			"name": names,
			"retryPolicy": map[string]any{
				"maxAttempts":          cfg.MaxAttempts,
				"initialBackoff":       seconds(cfg.Backoff.Initial),
				"maxBackoff":           seconds(cfg.Backoff.Max),
				"backoffMultiplier":    cfg.Backoff.Multiplier,
				"retryableStatusCodes": []string{"UNAVAILABLE"},
			},
		}}
	}
	b, err := json.Marshal(sc)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// Close closes the connection and stops the background work. Running
// watches fail; stop them first.
func (c *Client) Close() error {
	c.closeOnce.Do(func() {
		c.cancel()
		c.closeErr = c.conn.Close()
		c.wg.Wait()
	})
	return c.closeErr
}

// Conn returns the underlying connection, for generated clients the
// sub-clients do not cover.
func (c *Client) Conn() *grpc.ClientConn { return c.conn }

// Clusters returns the ClusterService client.
func (c *Client) Clusters() *ClusterClient { return c.clusters }

// ServiceExports returns the ServiceExportService client.
func (c *Client) ServiceExports() *ServiceExportClient { return c.serviceExports }

// ServiceImports returns the ServiceImportService client.
func (c *Client) ServiceImports() *ServiceImportClient { return c.serviceImports }

// ServerInfo returns the ServerInfoService client.
func (c *Client) ServerInfo() *ServerInfoClient { return c.serverInfo }

// OnReconnect registers fn to run after every connection the client
// establishes after its first one, with the URL it connected to. Hooks run
// one at a time on a background goroutine; reconnects that happen while
// hooks run are coalesced into one call with the latest URL. Writers use it
// to resend their state, readers to resync.
func (c *Client) OnReconnect(fn func(url string)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.hooks = append(c.hooks, fn)
}

// CurrentURL returns the canonical URL (scheme://host:port) of the
// connection in use, or "" while disconnected.
func (c *Client) CurrentURL() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.current == nil {
		return ""
	}
	return c.current.url
}

// URLs returns the canonical configured URLs followed by the fallbacks
// learned from GetServerInfo, in dial order.
func (c *Client) URLs() []string {
	targets := c.allTargets()
	urls := make([]string, 0, len(targets))
	for _, t := range targets {
		urls = append(urls, t.url)
	}
	return urls
}

func (c *Client) allTargets() []target {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append(slices.Clone(c.targets), c.fallbacks...)
}

func (c *Client) knownURLsLocked() []string {
	urls := make([]string, 0, len(c.targets)+len(c.fallbacks))
	for _, t := range c.targets {
		urls = append(urls, t.url)
	}
	for _, t := range c.fallbacks {
		urls = append(urls, t.url)
	}
	return urls
}

func (c *Client) setResolver(r *registryResolver) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.res = r
}

func (c *Client) clearResolver(r *registryResolver) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.res == r {
		c.res = nil
	}
}

func (c *Client) resolverInstance() *registryResolver {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.res
}

// watchRestarted returns a channel closed when the client returns to its
// primary URL, so that WatchLoops move their streams too.
func (c *Client) watchRestarted() <-chan struct{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.watchRestart
}

func (c *Client) restartWatches() {
	c.mu.Lock()
	defer c.mu.Unlock()
	close(c.watchRestart)
	c.watchRestart = make(chan struct{})
}

func (c *Client) hookLoop() {
	defer c.wg.Done()
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-c.reconnectCh:
		}
		c.mu.Lock()
		url, hooks := c.pendingReconnect, slices.Clone(c.hooks)
		c.mu.Unlock()
		for _, h := range hooks {
			h(url)
		}
	}
}

func (c *Client) serverInfoLoop() {
	defer c.wg.Done()
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-c.serverInfoCh:
			c.refreshFallbacks(c.ctx)
		}
	}
}

// refreshFallbacks asks the connected replica for the registry endpoints of
// every region and keeps the gRPC URLs that are new, valid and of the
// client's scheme as fallbacks after the configured URLs.
func (c *Client) refreshFallbacks(ctx context.Context) {
	info, err := c.serverInfo.Get(ctx)
	if err != nil {
		c.logger.Debug("GetServerInfo failed; keeping the fallback URLs", "error", err)
		return
	}
	seen := map[string]bool{}
	for _, t := range c.targets {
		seen[t.url] = true
	}
	var fallbacks []target
	for _, ep := range info.GetEndpoints() {
		if ep.GetGrpcUrl() == "" {
			continue
		}
		t, err := parseTarget(ep.GetGrpcUrl(), c.cfg.AllowInsecureHTTP)
		if err != nil || t.scheme != c.targets[0].scheme {
			c.logger.Debug("ignoring advertised registry endpoint", "url", ep.GetGrpcUrl(), "error", err)
			continue
		}
		if seen[t.url] {
			continue
		}
		seen[t.url] = true
		fallbacks = append(fallbacks, t)
	}
	c.mu.Lock()
	changed := !slices.Equal(fallbacks, c.fallbacks)
	if changed {
		c.fallbacks = fallbacks
		for _, t := range fallbacks {
			if c.current == nil || c.current.url != t.url {
				c.metrics.connected.WithLabelValues(t.url).Set(0)
			}
		}
	}
	r := c.res
	c.mu.Unlock()
	if changed && r != nil {
		c.logger.Info("registry fallback URLs updated", "fallbacks", len(fallbacks))
		r.ResolveNow(resolver.ResolveNowOptions{})
	}
}

func (c *Client) primaryLoop() {
	defer c.wg.Done()
	if c.cfg.PrimaryRetryInterval <= 0 {
		return
	}
	t := time.NewTicker(c.cfg.PrimaryRetryInterval)
	defer t.Stop()
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-t.C:
			c.returnToPrimary(c.ctx)
		}
	}
}

// returnToPrimary re-resolves the URLs and, when the client is connected to
// another URL than the primary and the primary reports SERVING to a health
// probe, moves the connection and the watches back to the primary. A
// primary within its UNAVAILABLE backoff is not probed. It reports whether
// it moved.
func (c *Client) returnToPrimary(ctx context.Context) bool {
	r := c.resolverInstance()
	if r == nil {
		return false
	}
	r.ResolveNow(resolver.ResolveNowOptions{})
	cur, primary := c.CurrentURL(), c.targets[0]
	if cur == "" || cur == primary.url {
		return false
	}
	if c.isUnhealthy(primary.url) {
		c.logger.Debug("primary registry URL answered UNAVAILABLE recently; not probing it", "url", primary.url)
		return false
	}
	addrs, err := c.addresses(ctx, []target{primary})
	if err != nil {
		c.logger.Debug("primary registry URL does not resolve", "url", primary.url, "error", err)
		return false
	}
	if err := c.probe(ctx, addrs); err != nil {
		c.logger.Debug("primary registry URL is not serving yet", "url", primary.url, "error", err)
		return false
	}
	c.logger.Info("returning to the primary registry URL", "from", cur, "to", primary.url)
	r.returnTo(addrs)
	c.restartWatches()
	return true
}

// probe checks, over a separate connection to addrs, that the gRPC health
// service (grpc.health.v1.Health/Check) reports SERVING for every registry
// service. A reachable replica answers NOT_SERVING while its view is stale
// or its store is unreachable, and while it shuts down; answering
// GetServerInfo would not tell these apart from a ready replica.
func (c *Client) probe(ctx context.Context, addrs []resolver.Address) error {
	mb := manual.NewBuilderWithScheme(c.resolverScheme + "-probe")
	mb.InitialState(resolver.State{Addresses: addrs})
	conn, err := grpc.NewClient(mb.Scheme()+":///probe", append(c.baseDialOptions(), grpc.WithResolvers(mb))...)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()
	ctx, cancel := context.WithTimeout(ctx, c.cfg.ConnectTimeout)
	defer cancel()
	hc := healthpb.NewHealthClient(conn)
	for _, sd := range serviceDescs {
		resp, err := hc.Check(ctx, &healthpb.HealthCheckRequest{Service: sd.ServiceName}, grpc.WaitForReady(true))
		if err != nil {
			return fmt.Errorf("health check of %s: %w", sd.ServiceName, err)
		}
		if st := resp.GetStatus(); st != healthpb.HealthCheckResponse_SERVING {
			return fmt.Errorf("%s reports %s", sd.ServiceName, st)
		}
	}
	return nil
}
