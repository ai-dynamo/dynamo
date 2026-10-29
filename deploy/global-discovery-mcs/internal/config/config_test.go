// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"bytes"
	"flag"
	"io"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func env(vars map[string]string) func(string) (string, bool) {
	return func(name string) (string, bool) {
		v, ok := vars[name]
		return v, ok
	}
}

func TestDefaultsAreValid(t *testing.T) {
	c, err := Parse(nil, env(nil), io.Discard)
	require.NoError(t, err)
	assert.Equal(t, Default(), c)

	assert.Equal(t, ":9090", c.GRPCListenAddress)
	assert.Equal(t, ":8080", c.HTTPListenAddress)
	assert.Equal(t, ":9091", c.AdminListenAddress)
	assert.Equal(t, StoreMemory, c.Store)
	assert.Equal(t, 2*time.Second, c.RefreshInterval)
	assert.Equal(t, time.Second, c.LeaseEvalInterval)
	assert.Equal(t, 30*time.Second, c.ClusterSetDiscoveryInterval)
	assert.Equal(t, 2*time.Second, c.LeaseSkew)
	assert.Equal(t, 24*time.Hour, c.Retention)
	assert.Equal(t, 5*time.Second, c.BookmarkInterval)
	assert.Equal(t, 10000, c.WatchHistory)
	assert.Equal(t, 1024, c.WatchQueueSize)
	assert.Equal(t, 5000, c.MaxWatchers)
	assert.Equal(t, 4<<20, c.MaxMessageBytes)
	assert.Equal(t, 5*time.Second, c.ShutdownDrain)
	assert.Equal(t, 25*time.Second, c.ShutdownTimeout)
	assert.False(t, c.OTelEnabled)
}

func TestFlags(t *testing.T) {
	c, err := Parse([]string{
		"--grpc-listen-address=127.0.0.1:19090",
		"--http-listen-address", "127.0.0.1:18080",
		"--admin-listen-address=127.0.0.1:19091",
		"--region=eu-west-1",
		"--store=dynamodb",
		"--dynamodb-table=mcs-test",
		"--dynamodb-endpoint=http://localhost:8000",
		"--refresh-interval=500ms",
		"--lease-eval-interval=250ms",
		"--clusterset-discovery-interval=10s",
		"--lease-skew=0s",
		"--retention=1h",
		"--compaction-interval=1m",
		"--bookmark-interval=10s",
		"--watch-history=100",
		"--watch-queue-size=16",
		"--max-watchers=10",
		"--max-message-bytes=8388608",
		`--advertise-endpoints=[{"region":"us-east-1","grpcUrl":"https://registry.us-east-1.example:443","httpUrl":"https://registry.us-east-1.example","primary":true},{"region":"us-west-2","grpcUrl":"https://registry.us-west-2.example:443"}]`,
		"--tls-cert-file=/tls/tls.crt",
		"--tls-key-file=/tls/tls.key",
		"--shutdown-drain=0s",
		"--shutdown-timeout=5s",
		"--otel-enabled",
		"--log-level=debug",
	}, env(nil), io.Discard)
	require.NoError(t, err)

	assert.Equal(t, &Config{
		GRPCListenAddress:           "127.0.0.1:19090",
		HTTPListenAddress:           "127.0.0.1:18080",
		AdminListenAddress:          "127.0.0.1:19091",
		Region:                      "eu-west-1",
		Store:                       StoreDynamoDB,
		DynamoDBTable:               "mcs-test",
		DynamoDBEndpoint:            "http://localhost:8000",
		RefreshInterval:             500 * time.Millisecond,
		LeaseEvalInterval:           250 * time.Millisecond,
		ClusterSetDiscoveryInterval: 10 * time.Second,
		LeaseSkew:                   0,
		Retention:                   time.Hour,
		CompactionInterval:          time.Minute,
		BookmarkInterval:            10 * time.Second,
		WatchHistory:                100,
		WatchQueueSize:              16,
		MaxWatchers:                 10,
		MaxMessageBytes:             8 << 20,
		AdvertiseEndpoints: AdvertiseEndpoints{
			{Region: "us-east-1", GRPCURL: "https://registry.us-east-1.example:443", HTTPURL: "https://registry.us-east-1.example", Primary: true},
			{Region: "us-west-2", GRPCURL: "https://registry.us-west-2.example:443"},
		},
		TLSCertFile:     "/tls/tls.crt",
		TLSKeyFile:      "/tls/tls.key",
		ShutdownDrain:   0,
		ShutdownTimeout: 5 * time.Second,
		OTelEnabled:     true,
		LogLevel:        "debug",
	}, c)
}

func TestEnvFallback(t *testing.T) {
	c, err := Parse([]string{"--max-watchers=7"}, env(map[string]string{
		"MCS_REFRESH_INTERVAL":    "3s",
		"MCS_MAX_WATCHERS":        "9",
		"MCS_OTEL_ENABLED":        "true",
		"MCS_STORE":               "dynamodb",
		"MCS_ADVERTISE_ENDPOINTS": `[{"region":"us-east-1","httpUrl":"http://registry:8080"}]`,
		"MCS_REGION":              "us-east-2",
		"AWS_REGION":              "us-west-1",
		"REFRESH_INTERVAL":        "9s",
	}), io.Discard)
	require.NoError(t, err)

	assert.Equal(t, 3*time.Second, c.RefreshInterval)
	assert.Equal(t, 7, c.MaxWatchers, "the command line wins over the environment")
	assert.True(t, c.OTelEnabled)
	assert.Equal(t, StoreDynamoDB, c.Store)
	assert.Equal(t, AdvertiseEndpoints{{Region: "us-east-1", HTTPURL: "http://registry:8080"}}, c.AdvertiseEndpoints)
	assert.Equal(t, "us-east-2", c.Region, "MCS_REGION wins over AWS_REGION")
}

func TestRegionFallsBackToAWSRegion(t *testing.T) {
	c, err := Parse(nil, env(map[string]string{"AWS_REGION": "us-west-1"}), io.Discard)
	require.NoError(t, err)
	assert.Equal(t, "us-west-1", c.Region)

	c, err = Parse([]string{"--region=ap-south-1"}, env(map[string]string{"AWS_REGION": "us-west-1"}), io.Discard)
	require.NoError(t, err)
	assert.Equal(t, "ap-south-1", c.Region)
}

func TestEnvName(t *testing.T) {
	assert.Equal(t, "MCS_REFRESH_INTERVAL", EnvName("refresh-interval"))
	assert.Equal(t, "MCS_GRPC_LISTEN_ADDRESS", EnvName("grpc-listen-address"))
	assert.Equal(t, "MCS_REGION", EnvName("region"))
}

func TestEveryFlagHasAnEnvVar(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	Default().RegisterFlags(fs)
	n := 0
	fs.VisitAll(func(f *flag.Flag) {
		n++
		assert.Regexp(t, `^MCS_[A-Z0-9_]+$`, EnvName(f.Name))
		assert.NotEmpty(t, f.Usage, f.Name)
	})
	assert.Equal(t, 25, n)
}

func TestParseErrors(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		env     map[string]string
		wantErr string
	}{
		{"unknown flag", []string{"--nope"}, nil, "flag provided but not defined: -nope"},
		{"bad duration flag", []string{"--refresh-interval=soon"}, nil, "invalid value"},
		{"positional argument", []string{"serve"}, nil, "unexpected arguments: serve"},
		{"bad env duration", nil, map[string]string{"MCS_RETENTION": "forever"}, "MCS_RETENTION"},
		{"bad env int", nil, map[string]string{"MCS_MAX_WATCHERS": "many"}, "MCS_MAX_WATCHERS"},
		{"bad env bool", nil, map[string]string{"MCS_OTEL_ENABLED": "maybe"}, "MCS_OTEL_ENABLED"},
		{"bad advertise JSON", []string{"--advertise-endpoints={"}, nil, "want a JSON array"},
		{"unknown advertise field", []string{`--advertise-endpoints=[{"region":"a","grpc_url":"https://x"}]`}, nil, "unknown field"},
		{"invalid result", []string{"--store=etcd"}, nil, `--store "etcd"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(tt.args, env(tt.env), io.Discard)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestHelp(t *testing.T) {
	var out bytes.Buffer
	_, err := Parse([]string{"-h"}, env(nil), &out)
	assert.ErrorIs(t, err, flag.ErrHelp)
	assert.Contains(t, out.String(), "-refresh-interval")
	assert.Contains(t, out.String(), "-advertise-endpoints")
}

func TestAdvertiseEndpointsValue(t *testing.T) {
	var a AdvertiseEndpoints
	assert.Empty(t, a.String())
	require.NoError(t, a.Set(` [{"region":"r","grpcUrl":"https://g","primary":true}] `))
	assert.Equal(t, `[{"region":"r","grpcUrl":"https://g","primary":true}]`, a.String())
	require.NoError(t, a.Set(""))
	assert.Nil(t, a)
	var nilPtr *AdvertiseEndpoints
	assert.Empty(t, nilPtr.String())
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(c *Config)
		wantErr string
	}{
		{"grpc address without port", func(c *Config) { c.GRPCListenAddress = "localhost" }, "--grpc-listen-address"},
		{"empty admin address", func(c *Config) { c.AdminListenAddress = "" }, "--admin-listen-address"},
		{"duplicate address", func(c *Config) { c.HTTPListenAddress = ":9090" }, "--grpc-listen-address and --http-listen-address both use"},
		{"unknown store", func(c *Config) { c.Store = "etcd" }, "--store"},
		{"dynamodb without table", func(c *Config) { c.Store, c.DynamoDBTable = StoreDynamoDB, "" }, "--dynamodb-table is required"},
		{"dynamodb endpoint without scheme", func(c *Config) { c.DynamoDBEndpoint = "localhost:8000" }, "--dynamodb-endpoint"},
		{"dynamodb endpoint without host", func(c *Config) { c.DynamoDBEndpoint = "http://" }, "host is required"},
		{"dynamodb endpoint unparsable", func(c *Config) { c.DynamoDBEndpoint = "http://[::1" }, "--dynamodb-endpoint"},
		{"refresh too fast", func(c *Config) { c.RefreshInterval = time.Millisecond }, "--refresh-interval"},
		{"lease eval zero", func(c *Config) { c.LeaseEvalInterval = 0 }, "--lease-eval-interval"},
		{"discovery too fast", func(c *Config) { c.ClusterSetDiscoveryInterval = time.Millisecond }, "--clusterset-discovery-interval"},
		{"retention too short", func(c *Config) { c.Retention = time.Minute }, "--retention"},
		{"compaction zero", func(c *Config) { c.CompactionInterval = 0 }, "--compaction-interval"},
		{"bookmark zero", func(c *Config) { c.BookmarkInterval = 0 }, "--bookmark-interval"},
		{"bookmark beyond idle timeout", func(c *Config) { c.BookmarkInterval = 6 * time.Minute }, "want at most 5m0s"},
		{"negative skew", func(c *Config) { c.LeaseSkew = -time.Second }, "--lease-skew"},
		{"huge skew", func(c *Config) { c.LeaseSkew = time.Hour }, "--lease-skew"},
		{"negative drain", func(c *Config) { c.ShutdownDrain = -time.Second }, "--shutdown-drain"},
		{"shutdown timeout zero", func(c *Config) { c.ShutdownTimeout = 0 }, "--shutdown-timeout"},
		{"watch history zero", func(c *Config) { c.WatchHistory = 0 }, "--watch-history"},
		{"queue zero", func(c *Config) { c.WatchQueueSize = 0 }, "--watch-queue-size"},
		{"no watchers", func(c *Config) { c.MaxWatchers = 0 }, "--max-watchers"},
		{"tiny messages", func(c *Config) { c.MaxMessageBytes = 1024 }, "--max-message-bytes"},
		{"cert without key", func(c *Config) { c.TLSCertFile = "/tls.crt" }, "must be set together"},
		{"key without cert", func(c *Config) { c.TLSKeyFile = "/tls.key" }, "must be set together"},
		{"advertise without region", func(c *Config) {
			c.AdvertiseEndpoints = AdvertiseEndpoints{{GRPCURL: "https://g"}}
		}, "region is required"},
		{"advertise without URL", func(c *Config) {
			c.AdvertiseEndpoints = AdvertiseEndpoints{{Region: "r"}}
		}, "grpcUrl or httpUrl is required"},
		{"advertise bad grpc URL", func(c *Config) {
			c.AdvertiseEndpoints = AdvertiseEndpoints{{Region: "r", GRPCURL: "dns:///registry:443"}}
		}, "[0].grpcUrl"},
		{"advertise bad http URL", func(c *Config) {
			c.AdvertiseEndpoints = AdvertiseEndpoints{{Region: "r", HTTPURL: "registry:8080"}}
		}, "[0].httpUrl"},
		{"two primaries", func(c *Config) {
			c.AdvertiseEndpoints = AdvertiseEndpoints{
				{Region: "a", GRPCURL: "https://a", Primary: true},
				{Region: "b", GRPCURL: "https://b", Primary: true},
			}
		}, "at most one entry may be primary"},
		{"bad log level", func(c *Config) { c.LogLevel = "trace" }, "--log-level"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := Default()
			tt.mutate(c)
			err := c.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

func TestValidateReportsEveryError(t *testing.T) {
	c := Default()
	c.Store = "etcd"
	c.MaxWatchers = 0
	c.LogLevel = "loud"
	err := c.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "--store")
	assert.Contains(t, err.Error(), "--max-watchers")
	assert.Contains(t, err.Error(), "--log-level")
}
