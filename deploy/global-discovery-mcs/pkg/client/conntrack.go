// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"net"
	"sync/atomic"

	"google.golang.org/grpc/stats"
)

// urlAddr is the remote address of a connection made by the client's
// dialer. It carries the registry URL the address was resolved from, so the
// stats handler can attribute the connection.
type urlAddr struct {
	net.Addr
	url string
}

// trackedConn reports a urlAddr as its remote address.
type trackedConn struct {
	net.Conn
	remote net.Addr
}

func (c *trackedConn) RemoteAddr() net.Addr { return c.remote }

// dial dials an address produced by the resolver.
func (c *Client) dial(ctx context.Context, addr string) (net.Conn, error) {
	url, hostport := decodeAddr(addr)
	var conn net.Conn
	var err error
	if c.cfg.Dialer != nil {
		conn, err = c.cfg.Dialer(ctx, hostport)
	} else {
		var d net.Dialer
		conn, err = d.DialContext(ctx, "tcp", hostport)
	}
	if err != nil {
		return nil, err
	}
	return &trackedConn{Conn: conn, remote: &urlAddr{Addr: conn.RemoteAddr(), url: url}}, nil
}

// connInfo identifies one transport of the client's ClientConn.
type connInfo struct {
	id  uint64
	url string
}

type connKey struct{}

// connTracker is a stats.Handler that follows the transports of the
// client's ClientConn: which URL the client is connected to, failovers and
// reconnect hooks.
type connTracker struct {
	c   *Client
	seq atomic.Uint64
}

func (t *connTracker) TagRPC(ctx context.Context, _ *stats.RPCTagInfo) context.Context { return ctx }

func (t *connTracker) HandleRPC(context.Context, stats.RPCStats) {}

func (t *connTracker) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context {
	ci := &connInfo{id: t.seq.Add(1)}
	if a, ok := info.RemoteAddr.(*urlAddr); ok {
		ci.url = a.url
	}
	return context.WithValue(ctx, connKey{}, ci)
}

func (t *connTracker) HandleConn(ctx context.Context, s stats.ConnStats) {
	ci, _ := ctx.Value(connKey{}).(*connInfo)
	if ci == nil || ci.url == "" {
		return
	}
	switch s.(type) {
	case *stats.ConnBegin:
		t.c.connBegin(ci)
	case *stats.ConnEnd:
		t.c.connEnd(ci)
	}
}

// connBegin records a new transport. Every transport after the first runs
// the OnReconnect hooks; one to another URL than the previous counts as a
// failover.
func (c *Client) connBegin(ci *connInfo) {
	c.mu.Lock()
	first := !c.everConnected
	prev := c.lastURL
	c.everConnected = true
	c.lastURL = ci.url
	c.current = ci
	for _, u := range c.knownURLsLocked() {
		c.metrics.connected.WithLabelValues(u).Set(0)
	}
	c.metrics.connected.WithLabelValues(ci.url).Set(1)
	if !first {
		c.pendingReconnect = ci.url
	}
	c.mu.Unlock()

	switch {
	case first:
		c.logger.Info("connected to the registry", "url", ci.url)
	case prev != ci.url:
		c.metrics.failovers.Inc()
		c.logger.Warn("registry endpoint changed", "from", prev, "to", ci.url)
	default:
		c.logger.Info("reconnected to the registry", "url", ci.url)
	}
	if !first {
		signal(c.reconnectCh)
	}
	signal(c.serverInfoCh)
}

func (c *Client) connEnd(ci *connInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.current == ci {
		c.current = nil
		c.metrics.connected.WithLabelValues(ci.url).Set(0)
	}
}

// signal does a non-blocking send on a channel of capacity one.
func signal(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}
