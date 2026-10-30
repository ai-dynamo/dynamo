// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package fakeregistry

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// Defaults of the fake server. They match the mcs-registry defaults except
// DefaultHistory, which is smaller.
const (
	// DefaultBookmarkInterval is the interval between BOOKMARK events.
	DefaultBookmarkInterval = 5 * time.Second
	// DefaultRefreshInterval is how often a serving listener re-evaluates
	// leases and broadcasts the resulting changes.
	DefaultRefreshInterval = time.Second
	// DefaultHistory is the number of events kept per clusterset and watch
	// kind for resume.
	DefaultHistory = 1000
	// DefaultWatchQueue is the per-watcher event queue. A watcher that falls
	// further behind is re-snapshotted.
	DefaultWatchQueue = 1024
	// DefaultRegion is the region GetServerInfo reports unless
	// SetServerInfo or WithRegion changes it.
	DefaultRegion = "fake-region"
	// Version is the version GetServerInfo reports by default.
	Version = "fakeregistry"
)

// Option configures a Server.
type Option func(*Server)

// WithClock sets the server clock used for leases and timestamps. Tests
// that need lease expiry pass an adjustable clock and call Refresh after
// moving it.
func WithClock(now func() time.Time) Option {
	return func(s *Server) { s.now = now }
}

// WithBookmarkInterval sets the interval between BOOKMARK events; zero or
// negative disables bookmarks.
func WithBookmarkInterval(d time.Duration) Option {
	return func(s *Server) { s.bookmarkInterval = d }
}

// WithRefreshInterval sets how often a serving listener re-evaluates leases;
// zero or negative disables the background refresh (call Refresh instead).
func WithRefreshInterval(d time.Duration) Option {
	return func(s *Server) { s.refreshInterval = d }
}

// WithViewEpoch sets the view epoch, the first half of every revision.
// Two servers with the same epoch accept each other's revisions only if
// their sequence numbers line up, so tests simulating separate replicas
// should leave the default (unique per Server).
func WithViewEpoch(epoch string) Option {
	return func(s *Server) { s.viewEpoch = epoch }
}

// WithHistory sets the number of events kept per clusterset and watch kind
// for resume.
func WithHistory(n int) Option {
	return func(s *Server) { s.historySize = n }
}

// WithWatchQueue sets the per-watcher event queue length.
func WithWatchQueue(n int) Option {
	return func(s *Server) { s.queueSize = n }
}

// WithRegion sets the region reported by GetServerInfo and used for
// clusters registered without one by the seeding helpers.
func WithRegion(region string) Option {
	return func(s *Server) { s.region = region }
}

var epochSeq struct {
	sync.Mutex
	last int64
}

// newViewEpoch returns a view epoch unique within the process: the current
// unix milliseconds, bumped when two servers start in the same millisecond.
func newViewEpoch() string {
	epochSeq.Lock()
	defer epochSeq.Unlock()
	ms := time.Now().UnixMilli()
	if ms <= epochSeq.last {
		ms = epochSeq.last + 1
	}
	epochSeq.last = ms
	return strconv.FormatInt(ms, 10)
}

// Server is the in-memory registry. Its zero value is not usable; call New. All
// methods are safe for concurrent use. One Server can serve several
// listeners at once; they share its state like replicas sharing a store
// and a view.
type Server struct {
	now              func() time.Time
	bookmarkInterval time.Duration
	refreshInterval  time.Duration
	historySize      int
	queueSize        int
	viewEpoch        string
	region           string

	// health is shared by every listener, like the state.
	health *health.Server

	mu       sync.Mutex
	seq      uint64
	sets     map[string]*clusterSet
	subs     map[*subscriber]struct{}
	failures map[string][]error
	calls    map[string]int
	info     *mcsv1.GetServerInfoResponse
}

// serviceNames are the fully qualified names of the registry services, the
// names the health service reports on besides the server-wide "".
var serviceNames = []string{
	mcsv1.ClusterService_ServiceDesc.ServiceName,
	mcsv1.ServiceExportService_ServiceDesc.ServiceName,
	mcsv1.ServiceImportService_ServiceDesc.ServiceName,
	mcsv1.ServerInfoService_ServiceDesc.ServiceName,
}

// New returns an empty Server.
func New(opts ...Option) *Server {
	s := &Server{
		now:              time.Now,
		bookmarkInterval: DefaultBookmarkInterval,
		refreshInterval:  DefaultRefreshInterval,
		historySize:      DefaultHistory,
		queueSize:        DefaultWatchQueue,
		region:           DefaultRegion,
		health:           health.NewServer(),
		sets:             map[string]*clusterSet{},
		subs:             map[*subscriber]struct{}{},
		failures:         map[string][]error{},
		calls:            map[string]int{},
	}
	for _, o := range opts {
		o(s)
	}
	s.SetServing(true)
	if s.viewEpoch == "" {
		s.viewEpoch = newViewEpoch()
	}
	if s.historySize < 1 {
		s.historySize = 1
	}
	if s.queueSize < 1 {
		s.queueSize = 1
	}
	s.info = &mcsv1.GetServerInfoResponse{Region: s.region, Version: Version}
	return s
}

// ViewEpoch returns the view epoch of the server's revisions.
func (s *Server) ViewEpoch() string { return s.viewEpoch }

// Revision returns the current revision, the revision a List would report.
func (s *Server) Revision() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.revision(s.seq)
}

func (s *Server) revision(seq uint64) string {
	return s.viewEpoch + ":" + strconv.FormatUint(seq, 10)
}

// parseRevision returns the sequence number of a revision of this server.
func (s *Server) parseRevision(rev string) (uint64, bool) {
	epoch, seq, ok := strings.Cut(rev, ":")
	if !ok || epoch != s.viewEpoch {
		return 0, false
	}
	n, err := strconv.ParseUint(seq, 10, 64)
	return n, err == nil
}

// Register registers the four services and the gRPC health service on r.
// When r is a grpc.Server created by the caller, pass ServerOptions to
// grpc.NewServer so that FailNext and CallCount work.
func (s *Server) Register(r grpc.ServiceRegistrar) {
	mcsv1.RegisterClusterServiceServer(r, &clusterService{s: s})
	mcsv1.RegisterServiceExportServiceServer(r, &exportService{s: s})
	mcsv1.RegisterServiceImportServiceServer(r, &importService{s: s})
	mcsv1.RegisterServerInfoServiceServer(r, &serverInfoService{s: s})
	healthpb.RegisterHealthServer(r, s.health)
}

// SetServing sets what the gRPC health service (grpc.health.v1.Health)
// reports for the server and for each registry service: SERVING, the
// default, or NOT_SERVING, as mcs-registry reports while its view is stale,
// its store is unreachable or it is shutting down. The RPCs are served
// either way; the registry client's failback probe stays away from a
// NOT_SERVING replica. Combine it with FailNext or CloseWatches to make
// calls fail.
func (s *Server) SetServing(serving bool) {
	st := healthpb.HealthCheckResponse_SERVING
	if !serving {
		st = healthpb.HealthCheckResponse_NOT_SERVING
	}
	s.health.SetServingStatus("", st)
	for _, name := range serviceNames {
		s.health.SetServingStatus(name, st)
	}
}

// ServerOptions returns the interceptors that implement FailNext and
// CallCount.
func (s *Server) ServerOptions() []grpc.ServerOption {
	return []grpc.ServerOption{
		grpc.ChainUnaryInterceptor(s.unaryInterceptor),
		grpc.ChainStreamInterceptor(s.streamInterceptor),
	}
}

func (s *Server) unaryInterceptor(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	if err := s.intercept(info.FullMethod); err != nil {
		return nil, err
	}
	return handler(ctx, req)
}

func (s *Server) streamInterceptor(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	if err := s.intercept(info.FullMethod); err != nil {
		return err
	}
	return handler(srv, ss)
}

func (s *Server) intercept(fullMethod string) error {
	m := methodName(fullMethod)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls[m]++
	if q := s.failures[m]; len(q) > 0 {
		s.failures[m] = q[1:]
		return q[0]
	}
	return nil
}

// methodName returns the RPC name of a full method name or of a bare name:
// "/dynamo.mcs.v1.ServiceImportService/ListServiceImports" and
// "ListServiceImports" both yield "ListServiceImports".
func methodName(m string) string {
	if _, after, ok := strings.CutLast(m, "/"); ok {
		return after
	}
	return m
}

// FailNext makes the next call of method fail with code. method is an RPC name
// such as "ListServiceImports" or a full method name. Calls queue: two calls of
// FailNext fail the next two calls. A failed call still counts in CallCount.
// For a watch the stream fails before any event is sent.
func (s *Server) FailNext(method string, code codes.Code) {
	s.FailNextWith(method, status.Errorf(code, "fakeregistry: injected %s failure of %s", code, methodName(method)))
}

// FailNextWith makes the next call of method fail with err, which should be
// a status error.
func (s *Server) FailNextWith(method string, err error) {
	m := methodName(method)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failures[m] = append(s.failures[m], err)
}

// CallCount returns how many calls of method the server received, including
// calls failed by FailNext. method is an RPC name or a full method name.
func (s *Server) CallCount(method string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls[methodName(method)]
}

// ResetCallCounts sets every call count to zero.
func (s *Server) ResetCallCounts() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = map[string]int{}
}

// SetServerInfo sets the response of GetServerInfo. ServerTime is filled at
// call time and an empty ViewEpoch with the server's view epoch.
func (s *Server) SetServerInfo(info *mcsv1.GetServerInfoResponse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if info == nil {
		info = &mcsv1.GetServerInfoResponse{}
	}
	s.info = proto.CloneOf(info)
}

func (s *Server) serverInfo() *mcsv1.GetServerInfoResponse {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := proto.CloneOf(s.info)
	if info.ViewEpoch == "" {
		info.ViewEpoch = s.viewEpoch
	}
	info.ServerTime = timestamppb.New(s.now())
	return info
}

// WatcherCount returns the number of open watch streams.
func (s *Server) WatcherCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.subs)
}

// CloseWatches ends every open watch stream with err, or with UNAVAILABLE
// and reason SHUTTING_DOWN when err is nil, as a draining replica does.
func (s *Server) CloseWatches(err error) {
	if err == nil {
		err = errShuttingDown()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for sub := range s.subs {
		select {
		case sub.kill <- err:
		default:
		}
	}
}

// Refresh re-evaluates every lease at the server clock and broadcasts the
// resulting changes, as the server's lease loop does. Serving listeners
// call it every refresh interval.
func (s *Server) Refresh() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, cs := range s.sets {
		s.commitLocked(cs)
	}
}

// Running is a Server serving one listener.
type Running struct {
	gs   *grpc.Server
	lis  net.Listener
	buf  *bufconn.Listener
	done chan struct{}
	stop chan struct{}
	once sync.Once
}

// Serve serves the Server on lis until Stop.
func (s *Server) Serve(lis net.Listener) *Running {
	gs := grpc.NewServer(s.ServerOptions()...)
	s.Register(gs)
	r := &Running{gs: gs, lis: lis, done: make(chan struct{}), stop: make(chan struct{})}
	if b, ok := lis.(*bufconn.Listener); ok {
		r.buf = b
	}
	go func() {
		defer close(r.done)
		_ = gs.Serve(lis)
	}()
	if s.refreshInterval > 0 {
		go func() {
			t := time.NewTicker(s.refreshInterval)
			defer t.Stop()
			for {
				select {
				case <-r.stop:
					return
				case <-t.C:
					s.Refresh()
				}
			}
		}()
	}
	return r
}

// ServeBufconn serves the Server on an in-process bufconn listener. Dial it
// with Dialer.
func (s *Server) ServeBufconn() *Running {
	return s.Serve(bufconn.Listen(1 << 20))
}

// ServeLocal serves the Server on 127.0.0.1 with a free port.
func (s *Server) ServeLocal() (*Running, error) {
	return s.ServeAddr("127.0.0.1:0")
}

// ServeAddr serves the Server on a TCP address, for example to restart a
// stopped listener on its previous address.
func (s *Server) ServeAddr(addr string) (*Running, error) {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("fakeregistry: listen on %s: %w", addr, err)
	}
	return s.Serve(lis), nil
}

// Addr returns the listen address, host:port for TCP and "bufconn" for a
// bufconn listener.
func (r *Running) Addr() string {
	if r.buf != nil {
		return "bufconn"
	}
	return r.lis.Addr().String()
}

// Port returns the TCP port, or 0 for a bufconn listener.
func (r *Running) Port() int {
	if a, ok := r.lis.Addr().(*net.TCPAddr); ok {
		return a.Port
	}
	return 0
}

// URL returns the plaintext URL of the listener, http://<Addr>.
func (r *Running) URL() string {
	return "http://" + r.Addr()
}

// Dialer returns a dialer that connects to this listener whatever address
// it is given, for grpc.WithContextDialer or a client Config.Dialer.
func (r *Running) Dialer() func(context.Context, string) (net.Conn, error) {
	if r.buf != nil {
		return func(ctx context.Context, _ string) (net.Conn, error) {
			return r.buf.DialContext(ctx)
		}
	}
	addr := r.Addr()
	return func(ctx context.Context, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, "tcp", addr)
	}
}

// Stop closes the listener and every connection, failing open calls and
// watches, like a crashed replica. It is idempotent.
func (r *Running) Stop() {
	r.once.Do(func() {
		close(r.stop)
		r.gs.Stop()
		<-r.done
	})
}

// GracefulStop stops accepting connections and waits for open calls,
// including watches, to finish. Combine it with Server.CloseWatches to end
// watches. It is idempotent with Stop.
func (r *Running) GracefulStop() {
	r.once.Do(func() {
		close(r.stop)
		r.gs.GracefulStop()
		<-r.done
	})
}
