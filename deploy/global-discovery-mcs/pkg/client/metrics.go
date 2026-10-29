// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// Metric names. Every client in a process shares them; the label sets are
// bounded by the API (rpc, code), the configured and advertised URLs (url)
// and the watch names (watch).
const (
	MetricRequestsTotal          = "dynamo_mcs_client_requests_total"
	MetricRequestDurationSeconds = "dynamo_mcs_client_request_duration_seconds"
	MetricEndpointFailoversTotal = "dynamo_mcs_client_endpoint_failovers_total"
	MetricConnected              = "dynamo_mcs_client_connected"
	MetricWatchReconnectsTotal   = "dynamo_mcs_client_watch_reconnects_total"
)

// Watch names used as metric labels; a WatchLoop may use any other bounded
// name.
const (
	WatchClusters       = "clusters"
	WatchServiceImports = "service_imports"
)

// serviceDescs are the registry services; their methods label the RPC metrics
// and the unary ones get the retry policy.
var serviceDescs = []*grpc.ServiceDesc{
	&mcsv1.ClusterService_ServiceDesc,
	&mcsv1.ServiceExportService_ServiceDesc,
	&mcsv1.ServiceImportService_ServiceDesc,
	&mcsv1.ServerInfoService_ServiceDesc,
}

// rpcNames returns the names of every registry RPC.
func rpcNames() []string {
	var names []string
	for _, sd := range serviceDescs {
		for _, m := range sd.Methods {
			names = append(names, m.MethodName)
		}
		for _, s := range sd.Streams {
			names = append(names, s.StreamName)
		}
	}
	return names
}

// rpcName returns the RPC name of a full method name.
func rpcName(fullMethod string) string {
	return fullMethod[strings.LastIndex(fullMethod, "/")+1:]
}

type metrics struct {
	requests        *prometheus.CounterVec
	duration        *prometheus.HistogramVec
	failovers       prometheus.Counter
	connected       *prometheus.GaugeVec
	watchReconnects *prometheus.CounterVec
}

// register registers c with reg, returning the collector already
// registered under the same descriptors when another client registered it
// first.
func register[C prometheus.Collector](reg prometheus.Registerer, c C) (C, error) {
	if reg == nil {
		return c, nil
	}
	if err := reg.Register(c); err != nil {
		if are, ok := errors.AsType[prometheus.AlreadyRegisteredError](err); ok {
			if existing, ok := are.ExistingCollector.(C); ok {
				return existing, nil
			}
		}
		return c, err
	}
	return c, nil
}

func newMetrics(reg prometheus.Registerer, urls []string) (*metrics, error) {
	m := &metrics{
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: MetricRequestsTotal,
			Help: "Registry client calls by RPC and gRPC status code. A watch counts once, when it ends.",
		}, []string{"rpc", "code"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    MetricRequestDurationSeconds,
			Help:    "Duration of registry client unary calls, retries included.",
			Buckets: prometheus.DefBuckets,
		}, []string{"rpc"}),
		failovers: prometheus.NewCounter(prometheus.CounterOpts{
			Name: MetricEndpointFailoversTotal,
			Help: "Connections established to a different registry URL than the previous connection.",
		}),
		connected: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: MetricConnected,
			Help: "1 for the registry URL the client is connected to, 0 for the others.",
		}, []string{"url"}),
		watchReconnects: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: MetricWatchReconnectsTotal,
			Help: "Watch streams reopened after an error or the end of a stream.",
		}, []string{"watch"}),
	}
	var err error
	if m.requests, err = register(reg, m.requests); err != nil {
		return nil, err
	}
	if m.duration, err = register(reg, m.duration); err != nil {
		return nil, err
	}
	if m.failovers, err = register(reg, m.failovers); err != nil {
		return nil, err
	}
	if m.connected, err = register(reg, m.connected); err != nil {
		return nil, err
	}
	if m.watchReconnects, err = register(reg, m.watchReconnects); err != nil {
		return nil, err
	}
	for _, rpc := range rpcNames() {
		for c := codes.OK; c <= codes.Unauthenticated; c++ {
			m.requests.WithLabelValues(rpc, c.String())
		}
	}
	for _, w := range []string{WatchClusters, WatchServiceImports} {
		m.watchReconnects.WithLabelValues(w)
	}
	for _, u := range urls {
		m.connected.WithLabelValues(u).Set(0)
	}
	return m, nil
}

// unaryInterceptor applies RPCTimeout to calls without a deadline, records
// the call metrics and the health of the endpoint that answered. gRPC
// retries happen inside the invoker, so a call counts once whatever its
// attempts, and the peer is that of the last attempt.
func (c *Client) unaryInterceptor(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	if c.cfg.RPCTimeout > 0 {
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, c.cfg.RPCTimeout)
			defer cancel()
		}
	}
	var p peer.Peer
	start := time.Now()
	err := invoker(ctx, method, req, reply, cc, append(opts, grpc.Peer(&p))...)
	rpc := rpcName(method)
	c.metrics.requests.WithLabelValues(rpc, status.Code(err).String()).Inc()
	c.metrics.duration.WithLabelValues(rpc).Observe(time.Since(start).Seconds())
	c.observeEndpoint(p.Addr, status.Code(err))
	return err
}

// streamInterceptor counts a stream once, with the status it ends with,
// and records the health of the endpoint that served it. gRPC fills the
// peer when the stream finishes, before RecvMsg returns its error.
func (c *Client) streamInterceptor(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	rpc := rpcName(method)
	p := &peer.Peer{}
	s, err := streamer(ctx, desc, cc, method, append(opts, grpc.Peer(p))...)
	if err != nil {
		c.metrics.requests.WithLabelValues(rpc, status.Code(err).String()).Inc()
		return nil, err
	}
	return &meteredStream{ClientStream: s, report: func(err error) {
		code := codes.OK
		if !errors.Is(err, io.EOF) {
			code = status.Code(err)
		}
		c.metrics.requests.WithLabelValues(rpc, code.String()).Inc()
		c.observeEndpoint(p.Addr, code)
	}}, nil
}

type meteredStream struct {
	grpc.ClientStream
	once   sync.Once
	report func(error)
}

func (s *meteredStream) RecvMsg(m any) error {
	err := s.ClientStream.RecvMsg(m)
	if err != nil {
		s.once.Do(func() { s.report(err) })
	}
	return err
}
