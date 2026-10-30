// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/status"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
)

// TestFailbackRequiresServing fails the client over to the secondary, brings
// the primary back reachable but NOT_SERVING, and expects the client to stay
// on the secondary until the primary reports SERVING.
func TestFailbackRequiresServing(t *testing.T) {
	primary := startFake(t, "primary")
	secondary := startFake(t, "secondary")
	dns := newFakeResolver(map[string]string{"primary.registry.test": "127.0.0.1", "secondary.registry.test": "127.0.0.1"})
	c := newTestClient(t, Config{
		URLs:                 []string{primary.url("primary.registry.test"), secondary.url("secondary.registry.test")},
		Resolver:             dns,
		PrimaryRetryInterval: 50 * time.Millisecond,
	})
	primaryURL, secondaryURL := primary.canonical("primary.registry.test"), secondary.canonical("secondary.registry.test")
	require.Equal(t, "primary", region(t, c))

	primary.srv.SetServing(false)
	primary.stop()
	require.Eventually(t, func() bool {
		info, err := c.ServerInfo().Get(testCtx(t))
		return err == nil && info.GetRegion() == "secondary"
	}, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, secondaryURL, c.CurrentURL())

	// Reachable again, but its health service answers NOT_SERVING: the
	// probe fails and the client stays where it is.
	primary.restart(t)
	assert.Never(t, func() bool { return c.CurrentURL() == primaryURL }, 500*time.Millisecond, 20*time.Millisecond,
		"no return to a primary that reports NOT_SERVING")
	assert.False(t, c.returnToPrimary(testCtx(t)))
	assert.Equal(t, "secondary", region(t, c))

	primary.srv.SetServing(true)
	require.Eventually(t, func() bool { return c.CurrentURL() == primaryURL }, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, "primary", region(t, c))
}

// TestUnavailableSkipsEndpoint has the connected primary answer a call with
// UNAVAILABLE and expects the client to move to the secondary although the
// primary is reachable, then to return once the backoff has expired.
func TestUnavailableSkipsEndpoint(t *testing.T) {
	primary := startFake(t, "primary")
	secondary := startFake(t, "secondary")
	dns := newFakeResolver(map[string]string{"primary.registry.test": "127.0.0.1", "secondary.registry.test": "127.0.0.1"})
	reg := prometheus.NewRegistry()
	c := newTestClient(t, Config{
		URLs:                 []string{primary.url("primary.registry.test"), secondary.url("secondary.registry.test")},
		Resolver:             dns,
		PrimaryRetryInterval: -1,
		MaxAttempts:          1,
		Registerer:           reg,
	})
	primaryURL, secondaryURL := primary.canonical("primary.registry.test"), secondary.canonical("secondary.registry.test")
	require.Equal(t, "primary", region(t, c))
	req := &mcsv1.ListServiceImportsRequest{Clusterset: testClusterset}

	primary.srv.FailNext("ListServiceImports", codes.Unavailable)
	_, err := c.ServiceImports().List(testCtx(t), req)
	require.Equal(t, codes.Unavailable, status.Code(err))
	assert.True(t, c.isUnhealthy(primaryURL))

	require.Eventually(t, func() bool { return c.CurrentURL() == secondaryURL }, 10*time.Second, 20*time.Millisecond,
		"the client leaves a reachable endpoint that answered UNAVAILABLE")
	assert.Equal(t, "secondary", region(t, c))
	v, _ := metricValue(t, reg, MetricEndpointFailoversTotal, nil)
	assert.Equal(t, 1.0, v)
	// The mark lasts at least ConnectTimeout (1 s here) and no probe runs
	// before it expires; then the primary serves, so a probe moves the
	// client back.
	assert.False(t, c.returnToPrimary(testCtx(t)), "the primary is skipped while marked")
	require.Eventually(t, func() bool { return !c.isUnhealthy(primaryURL) }, 5*time.Second, 5*time.Millisecond)
	require.Eventually(t, func() bool { return c.returnToPrimary(testCtx(t)) }, 5*time.Second, 20*time.Millisecond)
	require.Eventually(t, func() bool { return c.CurrentURL() == primaryURL }, 10*time.Second, 20*time.Millisecond)
	assert.Equal(t, "primary", region(t, c))
	require.Eventually(t, func() bool { return !c.isUnhealthy(primaryURL) }, time.Second, 5*time.Millisecond)
}

// TestHealthyAddresses checks the address filter and the backoff growth of
// the endpoint health bookkeeping without a connection.
func TestHealthyAddresses(t *testing.T) {
	c := &Client{
		cfg:       Config{Backoff: Backoff{Initial: 50 * time.Millisecond, Max: 200 * time.Millisecond, Multiplier: 2, Jitter: 0}},
		logger:    quietLogger,
		unhealthy: map[string]*unhealthyURL{},
	}
	a := resolver.Address{Addr: encodeAddr("http://a:80", "10.0.0.1:80")}
	b := resolver.Address{Addr: encodeAddr("http://b:80", "10.0.0.2:80")}
	all := []resolver.Address{a, b}

	assert.Equal(t, all, c.healthyAddresses(all), "nothing marked")
	c.observeEndpoint(&urlAddr{url: "http://a:80"}, codes.Internal)
	assert.Equal(t, all, c.healthyAddresses(all), "only UNAVAILABLE marks")
	c.observeEndpoint(nil, codes.Unavailable)
	assert.Equal(t, all, c.healthyAddresses(all), "no peer, no mark")

	c.observeEndpoint(&urlAddr{url: "http://a:80"}, codes.Unavailable)
	assert.True(t, c.isUnhealthy("http://a:80"))
	assert.Equal(t, []resolver.Address{b}, c.healthyAddresses(all))
	first := c.unhealthy["http://a:80"].until

	c.observeEndpoint(&urlAddr{url: "http://b:80"}, codes.Unavailable)
	assert.Equal(t, all, c.healthyAddresses(all), "every URL unhealthy: all are kept")

	// A success within the backoff keeps the mark; the next failure grows it.
	c.observeEndpoint(&urlAddr{url: "http://a:80"}, codes.OK)
	assert.True(t, c.isUnhealthy("http://a:80"))
	c.observeEndpoint(&urlAddr{url: "http://a:80"}, codes.Unavailable)
	assert.Equal(t, 2, c.unhealthy["http://a:80"].failures)
	assert.Greater(t, c.unhealthy["http://a:80"].until.Sub(first), 40*time.Millisecond, "the second failure backs off longer")

	// Expired marks do not filter, and a success after expiry clears them.
	c.unhealthy["http://a:80"].until = time.Now().Add(-time.Millisecond)
	c.unhealthy["http://b:80"].until = time.Now().Add(-time.Millisecond)
	assert.False(t, c.isUnhealthy("http://a:80"))
	assert.Equal(t, all, c.healthyAddresses(all))
	c.observeEndpoint(&urlAddr{url: "http://a:80"}, codes.OK)
	assert.NotContains(t, c.unhealthy, "http://a:80")
	assert.Contains(t, c.unhealthy, "http://b:80")
}
