// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"net"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/resolver"
)

// unhealthyURL is a registry URL whose connected endpoint answered
// UNAVAILABLE. The resolver leaves its addresses out of the list it pushes
// to pick_first until the backoff expires, so the client moves to the next
// URL instead of reconnecting to a replica that is up but not serving.
type unhealthyURL struct {
	// failures counts the consecutive UNAVAILABLE answers; it grows the
	// backoff and is reset by a successful call after the backoff.
	failures int
	until    time.Time
}

// unhealthyFor returns how long the n-th consecutive failure (0 for the
// first) marks a URL unhealthy: the client's Backoff delay, but at least
// ConnectTimeout, so that the mark outlasts the move to another endpoint
// and a resolution during the move cannot bring pick_first back.
func (c *Client) unhealthyFor(n int) time.Duration {
	return max(c.cfg.Backoff.Delay(n), c.cfg.ConnectTimeout)
}

// observeEndpoint records the outcome of a call answered by the endpoint at
// addr: UNAVAILABLE marks its URL unhealthy, OK clears the mark's failure
// count. Calls that never reached an endpoint (no peer) are ignored; gRPC
// is already reconnecting.
func (c *Client) observeEndpoint(addr net.Addr, code codes.Code) {
	ua, ok := addr.(*urlAddr)
	if !ok || ua.url == "" {
		return
	}
	switch code {
	case codes.OK:
		c.markAvailable(ua.url)
	case codes.Unavailable:
		c.markUnavailable(ua.url)
	}
}

// markUnavailable marks url unhealthy for unhealthyFor its consecutive
// failures and has the resolver push the addresses again without it, which
// makes pick_first drop a connection to url when another URL is available.
func (c *Client) markUnavailable(url string) {
	now := time.Now()
	c.mu.Lock()
	u := c.unhealthy[url]
	if u == nil {
		u = &unhealthyURL{}
		c.unhealthy[url] = u
	}
	u.until = now.Add(c.unhealthyFor(u.failures))
	u.failures++
	r, until := c.res, u.until
	c.mu.Unlock()
	c.logger.Warn("registry endpoint answered UNAVAILABLE; skipping it in the failover order", "url", url, "for", until.Sub(now))
	if r != nil {
		r.refilter()
	}
}

// markAvailable clears the failure count of url after a successful call,
// so that a later failure starts the backoff from the beginning again. A
// backoff in progress is kept: pick_first stays where it is.
func (c *Client) markAvailable(url string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if u := c.unhealthy[url]; u != nil && u.failures != 0 && !time.Now().Before(u.until) {
		delete(c.unhealthy, url)
	}
}

// isUnhealthy reports whether url is within its UNAVAILABLE backoff.
func (c *Client) isUnhealthy(url string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	u := c.unhealthy[url]
	return u != nil && time.Now().Before(u.until)
}

// healthyAddresses returns addrs without those of URLs within their
// UNAVAILABLE backoff. When that would leave none, every address is kept:
// a replica that answered UNAVAILABLE is still better than no replica.
func (c *Client) healthyAddresses(addrs []resolver.Address) []resolver.Address {
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.unhealthy) == 0 {
		return addrs
	}
	healthy := make([]resolver.Address, 0, len(addrs))
	for _, a := range addrs {
		url, _ := decodeAddr(a.Addr)
		if u := c.unhealthy[url]; u != nil && now.Before(u.until) {
			continue
		}
		healthy = append(healthy, a)
	}
	if len(healthy) == 0 {
		return addrs
	}
	return healthy
}
