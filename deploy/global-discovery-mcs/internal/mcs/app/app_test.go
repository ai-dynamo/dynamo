// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package app

import (
	"context"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
)

func testConfig(url string) config.Config {
	return config.Config{
		RegistryURLs: []string{url}, RegistryAllowInsecureHTTP: true, IdentitySource: config.IdentitySourceFlags,
		ClusterSetID: mcstest.ClusterSet, ClusterID: mcstest.Cluster, LeaseTTL: config.DefaultLeaseTTL,
		WriteClusterProperties: true, EndpointsPerSlice: 100, MaxConcurrentReconciles: 2,
		MetricsBindAddress: "0", HealthProbeBindAddress: "0",
	}
}

// unreachable is a REST configuration of an API server that never answers;
// the manager does not dial it before Start.
var unreachable = &rest.Config{Host: "http://127.0.0.1:1", Timeout: 100 * time.Millisecond}

func skipNames(o *ctrl.Options) { o.Controller.SkipNameValidation = new(true) }

func TestNewWiresEverything(t *testing.T) {
	fakeReg := mcstest.StartRegistry(t)
	node := &corev1.Node{Name: "n1", Labels: map[string]string{identity.LabelRegion: "us-east-1"}}
	c := mcstest.FakeClient(t, node)
	reg := prometheus.NewRegistry()
	a, err := New(context.Background(), testConfig(fakeReg.Running.URL()), unreachable, Options{
		Registry: reg, Logger: testr.New(t), ManagerOptions: skipNames, Client: c, AgentID: "pod-0",
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = a.Registry.Close() })
	assert.Equal(t, "us-east-1", a.Identity.Region)
	assert.NotNil(t, a.Lease)
	assert.NotNil(t, a.Exports)
	assert.NotNil(t, a.Importer)
	assert.NotNil(t, a.Resyncer)
	v, err := identity.GetClusterProperty(context.Background(), c, api.ClusterPropertyClusterID)
	require.NoError(t, err)
	assert.Equal(t, mcstest.Cluster, v, "--write-cluster-properties")

	// Ready once the client connected; this replica is not the leader.
	require.Eventually(t, func() bool { return a.ReadyCheck(nil) == nil }, 5*time.Second, 10*time.Millisecond)
	families, err := reg.Gather()
	require.NoError(t, err)
	var names []string
	for _, f := range families {
		names = append(names, f.GetName())
	}
	assert.Contains(t, names, "dynamo_mcs_controller_exports")
	assert.Contains(t, names, "dynamo_mcs_client_requests_total")

	// The OnReconnect hook triggers a resync.
	fakeReg.Running.Stop()
	_, err = a.Registry.ServerInfo().Get(context.Background())
	assert.Error(t, err)
}

func TestNewErrors(t *testing.T) {
	fakeReg := mcstest.StartRegistry(t)
	cfg := testConfig(fakeReg.Running.URL())

	_, err := New(context.Background(), cfg, unreachable, Options{Logger: testr.New(t), ManagerOptions: skipNames})
	require.ErrorContains(t, err, "resolve cluster identity")

	taken := mcstest.FakeClient(t)
	require.NoError(t, identity.WriteClusterProperties(context.Background(), taken, identity.Identity{ClusterSet: "other", Cluster: "x"}))
	_, err = New(context.Background(), cfg, unreachable, Options{Logger: testr.New(t), ManagerOptions: skipNames, Client: taken})
	require.ErrorContains(t, err, "refusing to overwrite")

	bad := cfg
	bad.WriteClusterProperties = false
	bad.RegistryAllowInsecureHTTP = false
	_, err = New(context.Background(), bad, unreachable, Options{Logger: testr.New(t), ManagerOptions: skipNames, Client: mcstest.FakeClient(t)})
	require.ErrorContains(t, err, "create registry client")

	reg := prometheus.NewRegistry()
	_, err = New(context.Background(), cfg, unreachable, Options{Registry: reg, ManagerOptions: skipNames, Client: mcstest.FakeClient(t)})
	require.NoError(t, err)
	_, err = New(context.Background(), cfg, unreachable, Options{Registry: reg, ManagerOptions: skipNames, Client: mcstest.FakeClient(t)})
	require.ErrorContains(t, err, "register metrics")

	_, err = New(context.Background(), cfg, &rest.Config{Host: "::not a url"}, Options{ManagerOptions: skipNames})
	require.Error(t, err)
}

func TestHostname(t *testing.T) {
	t.Setenv("HOSTNAME", "pod-7")
	assert.Equal(t, "pod-7", hostname())
	t.Setenv("HOSTNAME", "")
	assert.NotEmpty(t, hostname())
}

type electedManager struct {
	manager.Manager
	elected chan struct{}
}

func (m electedManager) Elected() <-chan struct{} { return m.elected }

func TestReadyCheckOnTheLeader(t *testing.T) {
	fakeReg := mcstest.StartRegistry(t)
	a, err := New(context.Background(), testConfig(fakeReg.Running.URL()), unreachable, Options{
		ManagerOptions: skipNames, Client: mcstest.FakeClient(t),
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = a.Registry.Close() })
	require.Eventually(t, func() bool { return a.ReadyCheck(nil) == nil }, 5*time.Second, 10*time.Millisecond)
	elected := make(chan struct{})
	close(elected)
	a.Manager = electedManager{Manager: a.Manager, elected: elected}
	require.ErrorContains(t, a.ReadyCheck(nil), "registered")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- a.Lease.Start(ctx) }()
	require.NoError(t, a.Lease.WaitRegistered(mcstest.Ctx(t)))
	assert.NoError(t, a.ReadyCheck(nil))
	cancel()
	require.NoError(t, <-done)
}

func TestReadyCheckBeforeTheFirstConnection(t *testing.T) {
	a, err := New(context.Background(), testConfig("http://127.0.0.1:1"), unreachable, Options{
		ManagerOptions: skipNames, Client: mcstest.FakeClient(t),
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = a.Registry.Close() })
	require.ErrorContains(t, a.ReadyCheck(nil), "not connected")
}
