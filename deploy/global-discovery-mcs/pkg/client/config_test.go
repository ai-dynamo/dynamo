// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseTarget(t *testing.T) {
	tests := []struct {
		raw       string
		allowHTTP bool
		want      target
		wantErr   string
	}{
		{raw: "https://REGISTRY.example.internal", want: target{url: "https://registry.example.internal:443", scheme: "https", host: "registry.example.internal", port: "443", serverName: "REGISTRY.example.internal"}},
		{raw: "https://registry.example.internal:8443/", want: target{url: "https://registry.example.internal:8443", scheme: "https", host: "registry.example.internal", port: "8443", serverName: "registry.example.internal:8443"}},
		{raw: "http://10.0.0.1", allowHTTP: true, want: target{url: "http://10.0.0.1:80", scheme: "http", host: "10.0.0.1", port: "80", serverName: "10.0.0.1"}},
		{raw: "https://[::1]:9090", want: target{url: "https://[::1]:9090", scheme: "https", host: "::1", port: "9090", serverName: "[::1]:9090"}},
		{raw: "http://registry:9090", wantErr: "AllowInsecureHTTP"},
		{raw: "dns:///registry:9090", wantErr: "scheme"},
		{raw: "registry:9090", wantErr: "scheme"},
		{raw: "https://", wantErr: "no host"},
		{raw: "https://user:pw@registry", wantErr: "credentials"},
		{raw: "https://registry/v1", wantErr: "path"},
		{raw: "https://registry?x=1", wantErr: "path"},
		{raw: "https://registry:0", wantErr: "port"},
		{raw: "https://registry:99999", wantErr: "port"},
		{raw: "://bad", wantErr: "invalid URL"},
	}
	for _, tt := range tests {
		got, err := parseTarget(tt.raw, tt.allowHTTP)
		if tt.wantErr != "" {
			assert.ErrorContains(t, err, tt.wantErr, tt.raw)
			continue
		}
		require.NoError(t, err, tt.raw)
		assert.Equal(t, tt.want, got, tt.raw)
	}
}

func TestNormalize(t *testing.T) {
	_, _, err := normalize(Config{})
	assert.ErrorContains(t, err, "no URLs")
	_, _, err = normalize(Config{URLs: []string{"https://a", "http://b"}, AllowInsecureHTTP: true})
	assert.ErrorContains(t, err, "every URL must use the scheme https")

	cfg, targets, err := normalize(Config{URLs: []string{"https://a", "https://b:443", "https://A"}, MaxAttempts: 9})
	require.NoError(t, err)
	require.Len(t, targets, 2, "duplicates are dropped")
	assert.Equal(t, "https://a:443", targets[0].url)
	assert.Equal(t, DefaultConnectTimeout, cfg.ConnectTimeout)
	assert.Equal(t, DefaultRPCTimeout, cfg.RPCTimeout)
	assert.Equal(t, DefaultPrimaryRetryInterval, cfg.PrimaryRetryInterval)
	assert.Equal(t, DefaultBackoff, cfg.Backoff)
	assert.Equal(t, maxRetryAttempts, cfg.MaxAttempts)
	assert.Equal(t, DefaultUserAgent, cfg.UserAgent)
	assert.NotNil(t, cfg.Resolver)
	assert.NotNil(t, cfg.Logger)

	cfg, _, err = normalize(Config{URLs: []string{"https://a"}, RPCTimeout: -1, PrimaryRetryInterval: -1})
	require.NoError(t, err)
	assert.Negative(t, cfg.RPCTimeout)
	assert.Negative(t, cfg.PrimaryRetryInterval)
	assert.Equal(t, DefaultMaxAttempts, cfg.MaxAttempts)

	_, err = New(context.Background(), Config{})
	assert.Error(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = New(ctx, Config{URLs: []string{"https://a"}})
	assert.ErrorIs(t, err, context.Canceled)
}

func TestBackoffDelay(t *testing.T) {
	b := Backoff{Initial: 100 * time.Millisecond, Max: time.Second, Multiplier: 2, Jitter: 0.5}
	for n, base := range []time.Duration{100, 200, 400, 800, 1000, 1000} {
		base *= time.Millisecond
		for range 20 {
			d := b.Delay(n)
			assert.GreaterOrEqual(t, d, base/2, "attempt %d", n)
			assert.LessOrEqual(t, d, min(base*3/2, time.Second), "attempt %d", n)
		}
	}
	assert.Equal(t, time.Millisecond, Backoff{Initial: time.Millisecond, Max: time.Second, Multiplier: 3, Jitter: 0}.Delay(-1))
	assert.Equal(t, DefaultBackoff, Backoff{}.withDefaults())
	assert.Equal(t, Backoff{Initial: time.Second, Max: time.Second, Multiplier: 2, Jitter: 0.5},
		Backoff{Initial: time.Second, Max: time.Millisecond, Multiplier: 0.5, Jitter: 3}.withDefaults())
}

func TestServiceConfig(t *testing.T) {
	var sc struct {
		LoadBalancingConfig []map[string]map[string]bool `json:"loadBalancingConfig"`
		MethodConfig        []struct {
			Name        []map[string]string `json:"name"`
			RetryPolicy struct {
				MaxAttempts          int      `json:"maxAttempts"`
				InitialBackoff       string   `json:"initialBackoff"`
				MaxBackoff           string   `json:"maxBackoff"`
				BackoffMultiplier    float64  `json:"backoffMultiplier"`
				RetryableStatusCodes []string `json:"retryableStatusCodes"`
			} `json:"retryPolicy"`
		} `json:"methodConfig"`
	}
	cfg := Config{MaxAttempts: 4, Backoff: DefaultBackoff}
	require.NoError(t, json.Unmarshal([]byte(serviceConfig(cfg)), &sc))
	require.Len(t, sc.LoadBalancingConfig, 1)
	assert.Equal(t, map[string]bool{"shuffleAddressList": false}, sc.LoadBalancingConfig[0]["pick_first"])
	require.Len(t, sc.MethodConfig, 1)
	rp := sc.MethodConfig[0].RetryPolicy
	assert.Equal(t, 4, rp.MaxAttempts)
	assert.Equal(t, "0.2s", rp.InitialBackoff)
	assert.Equal(t, "30s", rp.MaxBackoff)
	assert.Equal(t, 2.0, rp.BackoffMultiplier)
	assert.Equal(t, []string{"UNAVAILABLE"}, rp.RetryableStatusCodes)
	methods := map[string]bool{}
	for _, n := range sc.MethodConfig[0].Name {
		methods[n["method"]] = true
	}
	assert.True(t, methods["ListServiceImports"])
	assert.True(t, methods["GetServerInfo"])
	assert.False(t, methods["WatchServiceImports"], "streams are not retried by gRPC")
	assert.True(t, methods["ReplaceEndpoints"])
	assert.Len(t, methods, 13)

	sc.MethodConfig = nil
	require.NoError(t, json.Unmarshal([]byte(serviceConfig(Config{MaxAttempts: 1})), &sc))
	assert.Empty(t, sc.MethodConfig)
}

func TestTransportCredentials(t *testing.T) {
	creds, err := transportCredentials(Config{}, "http")
	require.NoError(t, err)
	assert.Equal(t, "insecure", creds.Info().SecurityProtocol)

	creds, err = transportCredentials(Config{}, "https")
	require.NoError(t, err)
	assert.Equal(t, "tls", creds.Info().SecurityProtocol)

	dir := t.TempDir()
	_, err = transportCredentials(Config{CAFile: filepath.Join(dir, "missing.pem")}, "https")
	assert.ErrorContains(t, err, "read CA file")
	bad := filepath.Join(dir, "bad.pem")
	require.NoError(t, os.WriteFile(bad, []byte("not pem"), 0o600))
	_, err = transportCredentials(Config{CAFile: bad}, "https")
	assert.ErrorContains(t, err, "no PEM certificate")

	pki := newTestPKI(t, "registry.test")
	_, err = transportCredentials(Config{CAFile: pki.caFile, TLS: &tls.Config{RootCAs: x509.NewCertPool()}}, "https")
	assert.ErrorContains(t, err, "mutually exclusive")
	_, err = transportCredentials(Config{CAFile: pki.caFile}, "https")
	require.NoError(t, err)
}
