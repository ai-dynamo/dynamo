// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"flag"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
)

func parse(t *testing.T, env map[string]string, args ...string) (*Config, error) {
	t.Helper()
	var c Config
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	c.bindFlags(fs, func(k string) (string, bool) {
		v, ok := env[k]
		return v, ok
	})
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	return &c, c.Validate()
}

func TestDefaults(t *testing.T) {
	c, err := parse(t, map[string]string{RegistryURLsEnv: "https://registry.example.internal, https://registry.b.example.internal", PodNamespaceEnv: "mcs-system"},
		"--clusterset-id=prod", "--cluster-id=a")
	require.NoError(t, err)
	assert.Equal(t, []string{"https://registry.example.internal", "https://registry.b.example.internal"}, c.RegistryURLs)
	assert.Equal(t, "mcs-system", c.OperatorNamespace)
	assert.Equal(t, IdentitySourceAuto, c.IdentitySource)
	assert.Equal(t, DefaultLeaseTTL, c.LeaseTTL)
	assert.Equal(t, DefaultEndpointsPerSlice, c.EndpointsPerSlice)
	assert.Equal(t, DefaultMaxConcurrentReconciles, c.MaxConcurrentReconciles)
	assert.True(t, c.LeaderElect)
	assert.False(t, c.CreateNamespaces)
	assert.False(t, c.DeregisterOnShutdown)
	assert.False(t, c.WriteClusterProperties)
	assert.Equal(t, DefaultHealthProbeBindAddress, c.HealthProbeBindAddress)
	assert.True(t, c.ImportTargetRef)
	assert.Equal(t, DefaultResyncPeriod, c.ExportResyncPeriod)
	assert.Equal(t, DefaultResyncPeriod, c.ImportResyncPeriod)
	assert.True(t, c.WatchesNamespace("anything"))
}

func TestFlags(t *testing.T) {
	ca := filepath.Join(t.TempDir(), "ca.crt")
	require.NoError(t, os.WriteFile(ca, []byte("x"), 0o600))
	c, err := parse(t, map[string]string{RegistryURLsEnv: "https://ignored.example"},
		"--registry-url=http://a.example:8080,http://b.example", "--registry-url", "http://c.example",
		"--registry-allow-insecure-http", "--registry-tls-ca-file="+ca,
		"--clusterset-id=prod", "--cluster-id=a", "--cluster-identity-source=flags", "--region=us-west-2",
		"--cluster-labels=env=prod, tier=gpu", "--lease-ttl=45s", "--watch-namespaces=b,a,b",
		"--create-namespaces", "--adopt-unmanaged-imports", "--import-target-ref=false",
		"--export-resync-period=0", "--import-resync-period=2m",
		"--write-cluster-properties", "--deregister-on-shutdown", "--endpoints-per-slice=50",
		"--max-concurrent-reconciles=8", "--metrics-bind-address=0", "--health-probe-bind-address=:9000",
		"--leader-elect=false", "--otel-enabled")
	require.NoError(t, err)
	assert.Equal(t, []string{"http://a.example:8080", "http://b.example", "http://c.example"}, c.RegistryURLs)
	assert.Equal(t, map[string]string{"env": "prod", "tier": "gpu"}, c.ClusterLabels)
	assert.Equal(t, []string{"a", "b"}, c.WatchNamespaces)
	assert.Equal(t, 45*time.Second, c.LeaseTTL)
	assert.True(t, c.CreateNamespaces && c.AdoptUnmanagedImports && c.WriteClusterProperties && c.DeregisterOnShutdown && c.OTelEnabled)
	assert.Equal(t, 50, c.EndpointsPerSlice)
	assert.Equal(t, 8, c.MaxConcurrentReconciles)
	assert.False(t, c.ImportTargetRef)
	assert.Zero(t, c.ExportResyncPeriod)
	assert.Equal(t, 2*time.Minute, c.ImportResyncPeriod)
	assert.True(t, c.WatchesNamespace("a"))
	assert.False(t, c.WatchesNamespace("c"))
	assert.Equal(t, "env=prod,tier=gpu", (*labelMap)(&c.ClusterLabels).String())
	assert.Equal(t, "a,b", (*namespaceList)(&c.WatchNamespaces).String())
}

func TestInvalidLabelFlag(t *testing.T) {
	_, err := parse(t, nil, "--cluster-labels=novalue")
	require.Error(t, err)
}

func TestValidate(t *testing.T) {
	base := []string{"--registry-url=https://registry.example", "--leader-elect=false"}
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"no url", []string{"--leader-elect=false"}, "--registry-url"},
		{"bad scheme", []string{"--registry-url=ftp://x", "--leader-elect=false"}, "scheme must be https or http"},
		{"no host", []string{"--registry-url=https://", "--leader-elect=false"}, "has no host"},
		{"plaintext", []string{"--registry-url=http://x", "--leader-elect=false"}, "--registry-allow-insecure-http"},
		{"mixed schemes", []string{"--registry-url=https://x,http://y", "--registry-allow-insecure-http", "--leader-elect=false"}, "every URL must use the scheme"},
		{"unparsable url", []string{"--registry-url=https://x:bad:port", "--leader-elect=false"}, "--registry-url"},
		{"ca file", append(base, "--registry-tls-ca-file=/does/not/exist"), "--registry-tls-ca-file"},
		{"identity source", append(base, "--cluster-identity-source=dns"), "--cluster-identity-source"},
		{"flags source needs ids", append(base, "--cluster-identity-source=flags", "--cluster-id=a"), "requires --clusterset-id"},
		{"cluster id", append(base, "--cluster-id=Not_A_Label"), "--cluster-id"},
		{"region", append(base, "--region=us east"), "--region"},
		{"label key", append(base, "--cluster-labels=bad key=v"), "--cluster-labels key"},
		{"label value", append(base, "--cluster-labels=k=bad value"), "--cluster-labels value"},
		{"ttl", append(base, "--lease-ttl=1s"), "--lease-ttl"},
		{"namespace", append(base, "--watch-namespaces=Bad"), "--watch-namespaces"},
		{"per slice", append(base, "--endpoints-per-slice=0"), "--endpoints-per-slice"},
		{"per slice max", append(base, "--endpoints-per-slice=1001"), "--endpoints-per-slice"},
		{"workers", append(base, "--max-concurrent-reconciles=0"), "--max-concurrent-reconciles"},
		{"export resync", append(base, "--export-resync-period=10ms"), "--export-resync-period"},
		{"import resync", append(base, "--import-resync-period=-1s"), "--import-resync-period"},
		{"health", append(base, "--health-probe-bind-address="), "--health-probe-bind-address"},
		{"leader namespace", []string{"--registry-url=https://registry.example"}, "--operator-namespace"},
		{"operator namespace", append(base, "--operator-namespace=Bad"), "--operator-namespace"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parse(t, nil, tt.args...)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.want)
		})
	}
	t.Run("too many labels", func(t *testing.T) {
		c := Config{RegistryURLs: []string{"https://x"}, LeaseTTL: DefaultLeaseTTL, EndpointsPerSlice: 1, MaxConcurrentReconciles: 1,
			HealthProbeBindAddress: ":1", IdentitySource: IdentitySourceAuto, ClusterLabels: map[string]string{}}
		for i := range api.MaxLabels + 1 {
			c.ClusterLabels["k"+string(rune('a'+i%26))+string(rune('a'+i/26))] = "v"
		}
		require.ErrorContains(t, c.Validate(), "at most")
	})
}

func TestBindFlagsReadsEnvironment(t *testing.T) {
	t.Setenv(RegistryURLsEnv, "https://env.example")
	t.Setenv(PodNamespaceEnv, "ns-from-env")
	var c Config
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	c.BindFlags(fs)
	require.NoError(t, fs.Parse(nil))
	assert.Equal(t, []string{"https://env.example"}, c.RegistryURLs)
	assert.Equal(t, "ns-from-env", c.OperatorNamespace)
	assert.Equal(t, "https://env.example", fs.Lookup("registry-url").Value.String())
}

func TestSchemeAndManagerOptions(t *testing.T) {
	s, err := NewScheme()
	require.NoError(t, err)
	for _, gvk := range []schema.GroupVersionKind{
		{Version: "v1", Kind: "Service"},
		{Group: "discovery.k8s.io", Version: "v1", Kind: "EndpointSlice"},
		{Group: mcsv1beta1.GroupName, Version: "v1beta1", Kind: "ServiceExport"},
		{Group: mcsv1beta1.GroupName, Version: "v1beta1", Kind: "ServiceImport"},
	} {
		assert.True(t, s.Recognizes(gvk), gvk.String())
	}
	assert.False(t, s.Recognizes(ClusterPropertyGVK), "ClusterProperty stays unstructured")

	c := Config{MetricsBindAddress: "0", HealthProbeBindAddress: ":8081", LeaderElect: true, OperatorNamespace: "mcs-system"}
	opts := c.ManagerOptions(s)
	assert.Equal(t, api.LeaderElectionID, opts.LeaderElectionID)
	assert.Equal(t, "mcs-system", opts.LeaderElectionNamespace)
	assert.True(t, opts.LeaderElection)
	assert.True(t, opts.LeaderElectionReleaseOnCancel)
	assert.Equal(t, "0", opts.Metrics.BindAddress)
	assert.Nil(t, opts.Cache.DefaultNamespaces)
	assert.NotNil(t, opts.Cache.DefaultTransform)

	c.WatchNamespaces = []string{"a", "b"}
	opts = c.ManagerOptions(s)
	assert.Equal(t, map[string]cache.Config{"a": {}, "b": {}}, opts.Cache.DefaultNamespaces)
}

func TestURLListString(t *testing.T) {
	var l *urlList
	assert.Equal(t, "", l.String())
	var nl *namespaceList
	assert.Equal(t, "", nl.String())
	var lm *labelMap
	assert.Equal(t, "", lm.String())
}
