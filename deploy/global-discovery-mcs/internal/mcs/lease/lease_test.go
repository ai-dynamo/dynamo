// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lease_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/lease"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

type harness struct {
	fakeReg *mcstest.Registry
	lease   *lease.Lease
	reg     *prometheus.Registry

	mu   sync.Mutex
	regs []bool
}

func (h *harness) registrations() []bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]bool(nil), h.regs...)
}

func start(t *testing.T, deregister bool) (*harness, context.CancelFunc, chan error) {
	t.Helper()
	h := &harness{fakeReg: mcstest.StartRegistry(t), reg: prometheus.NewRegistry()}
	m := metrics.New()
	require.NoError(t, m.Register(h.reg))
	h.lease = lease.New(lease.Options{
		Registry:             h.fakeReg.Client,
		Identity:             mcstest.Identity(),
		TTL:                  300 * time.Millisecond,
		AgentID:              "pod-0",
		AgentVersion:         "test",
		DeregisterOnShutdown: deregister,
		Metrics:              m,
		Logger:               testr.New(t),
		OnRegistered: func(first bool) {
			h.mu.Lock()
			defer h.mu.Unlock()
			h.regs = append(h.regs, first)
		},
		Backoff: client.Backoff{Initial: 10 * time.Millisecond, Max: 50 * time.Millisecond, Multiplier: 2},
	})
	assert.True(t, h.lease.NeedLeaderElection())
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- h.lease.Start(ctx) }()
	t.Cleanup(func() {
		cancel()
		if err := <-done; err != nil {
			t.Logf("lease stopped: %v", err)
		}
	})
	return h, cancel, done
}

func (h *harness) cluster(t *testing.T) (*mcsv1.Cluster, error) {
	t.Helper()
	resp, err := h.fakeReg.Client.Clusters().Get(mcstest.Ctx(t), &mcsv1.GetClusterRequest{Clusterset: mcstest.ClusterSet, Cluster: mcstest.Cluster})
	return resp.GetCluster(), err
}

func TestRegisterAndHeartbeat(t *testing.T) {
	h, _, _ := start(t, false)
	require.NoError(t, h.lease.WaitRegistered(mcstest.Ctx(t)))
	assert.True(t, h.lease.Registered())
	c, err := h.cluster(t)
	require.NoError(t, err)
	assert.Equal(t, h.lease.Epoch(), c.GetLease().GetEpoch())
	assert.Equal(t, "pod-0", c.GetAgentId())
	assert.Equal(t, mcstest.Region, c.GetRegion())
	require.Eventually(t, func() bool { return h.fakeReg.Server.CallCount("HeartbeatCluster") >= 3 }, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, []bool{true}, h.registrations())
	assert.Equal(t, 1.0, mcstest.MetricValue(t, h.reg, "dynamo_mcs_controller_cluster_registered", nil))
	assert.Equal(t, 1.0, mcstest.MetricValue(t, h.reg, "dynamo_mcs_controller_registrations_total", map[string]string{"result": "success"}))
}

func TestReregistersWhenTheLeaseIsLost(t *testing.T) {
	h, _, _ := start(t, false)
	require.NoError(t, h.lease.WaitRegistered(mcstest.Ctx(t)))
	first := h.lease.Epoch()

	// Another registration takes the lease: ABORTED LEASE_EPOCH_MISMATCH.
	_, err := h.fakeReg.Client.Clusters().Register(mcstest.Ctx(t), &mcsv1.RegisterClusterRequest{Clusterset: mcstest.ClusterSet, Cluster: mcstest.Cluster})
	require.NoError(t, err)
	h.lease.Kick()
	require.Eventually(t, func() bool { return len(h.registrations()) == 2 }, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, []bool{true, false}, h.registrations())
	assert.Greater(t, h.lease.Epoch(), first)
	assert.Equal(t, 1.0, mcstest.MetricValue(t, h.reg, "dynamo_mcs_controller_heartbeat_failures_total", map[string]string{"reason": "epoch_mismatch"}))

	// The registry forgets the cluster: NOT_FOUND.
	h.fakeReg.Server.RemoveCluster(mcstest.ClusterSet, mcstest.Cluster)
	h.lease.Kick()
	require.Eventually(t, func() bool { return len(h.registrations()) == 3 }, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, 1.0, mcstest.MetricValue(t, h.reg, "dynamo_mcs_controller_heartbeat_failures_total", map[string]string{"reason": "not_registered"}))
	_, err = h.cluster(t)
	require.NoError(t, err)
}

func TestTransientFailuresKeepTheLease(t *testing.T) {
	h, _, _ := start(t, false)
	require.NoError(t, h.lease.WaitRegistered(mcstest.Ctx(t)))
	for range 4 {
		h.fakeReg.Server.FailNext("HeartbeatCluster", codes.Internal)
	}
	h.lease.Kick()
	require.Eventually(t, func() bool {
		return mcstest.MetricValue(t, h.reg, "dynamo_mcs_controller_heartbeat_failures_total", map[string]string{"reason": "error"}) >= 4
	}, 5*time.Second, 10*time.Millisecond)
	assert.True(t, h.lease.Registered())
	assert.Equal(t, []bool{true}, h.registrations())
}

func TestRegistrationRetries(t *testing.T) {
	srv := mcstest.StartRegistry(t)
	for range 4 {
		srv.Server.FailNext("RegisterCluster", codes.Internal)
	}
	l := lease.New(lease.Options{Registry: srv.Client, Identity: mcstest.Identity(), Logger: testr.New(t),
		Backoff: client.Backoff{Initial: 5 * time.Millisecond, Max: 20 * time.Millisecond}})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- l.Start(ctx) }()
	require.NoError(t, l.WaitRegistered(mcstest.Ctx(t)))
	assert.Equal(t, 5, srv.Server.CallCount("RegisterCluster"))
	cancel()
	require.NoError(t, <-done)
	assert.False(t, l.Registered())

	waitCtx, waitCancel := context.WithCancel(context.Background())
	waitCancel()
	assert.ErrorIs(t, l.WaitRegistered(waitCtx), context.Canceled)
}

func TestDeregisterOnShutdown(t *testing.T) {
	for _, deregister := range []bool{false, true} {
		h, cancel, done := start(t, deregister)
		require.NoError(t, h.lease.WaitRegistered(mcstest.Ctx(t)))
		cancel()
		require.NoError(t, <-done)
		done <- nil // for the cleanup
		_, err := h.cluster(t)
		if deregister {
			assert.Equal(t, codes.NotFound, status.Code(err))
		} else {
			assert.NoError(t, err)
		}
	}
}

func TestDeregisterFailureIsLogged(t *testing.T) {
	h, cancel, done := start(t, true)
	require.NoError(t, h.lease.WaitRegistered(mcstest.Ctx(t)))
	h.fakeReg.Server.FailNext("DeregisterCluster", codes.Internal)
	cancel()
	require.NoError(t, <-done)
	done <- nil
	_, err := h.cluster(t)
	assert.NoError(t, err, "the lease expires instead")
}
