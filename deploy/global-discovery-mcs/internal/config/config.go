// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package config holds the mcs-registry flags. Every flag can also be set
// through an environment variable named MCS_ plus the flag name in upper case
// with dashes replaced by underscores, for example MCS_REFRESH_INTERVAL for
// --refresh-interval. A flag on the command line wins over the environment.
package config

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
)

// EnvPrefix prefixes the environment variable of every flag.
const EnvPrefix = "MCS_"

// Store kinds.
const (
	StoreMemory   = "memory"
	StoreDynamoDB = "dynamodb"
)

// Config is the mcs-registry configuration.
type Config struct {
	// GRPCListenAddress serves gRPC (h2c or TLS); the NLB targets it.
	GRPCListenAddress string
	// HTTPListenAddress serves the REST/JSON gateway, which dials the gRPC
	// listener over loopback.
	HTTPListenAddress string
	// AdminListenAddress serves /metrics, /healthz, /readyz and
	// /debug/pprof.
	AdminListenAddress string
	// Region is reported by GetServerInfo. It falls back to AWS_REGION.
	Region string

	// Store selects the backend: StoreMemory or StoreDynamoDB.
	Store string
	// DynamoDBTable is the Global Table name.
	DynamoDBTable string
	// DynamoDBEndpoint overrides the DynamoDB endpoint, for dynamodb-local.
	DynamoDBEndpoint string

	// RefreshInterval is how often each clusterset is reloaded from the
	// store.
	RefreshInterval time.Duration
	// LeaseEvalInterval is how often leases are re-evaluated between
	// refreshes.
	LeaseEvalInterval time.Duration
	// ClusterSetDiscoveryInterval is how often the store's clustersets are
	// listed to find new ones.
	ClusterSetDiscoveryInterval time.Duration
	// LeaseSkew is the clock skew allowance before a lease counts as expired.
	LeaseSkew time.Duration
	// Retention keeps the records of an expired cluster this long before
	// compaction removes them.
	Retention time.Duration
	// CompactionInterval is how often compaction runs.
	CompactionInterval time.Duration

	// BookmarkInterval is the interval of watch BOOKMARK events.
	BookmarkInterval time.Duration
	// WatchHistory is the number of events kept per clusterset for resume.
	WatchHistory int
	// WatchQueueSize is the per-watcher event queue; an overflowing watcher
	// is re-snapshotted.
	WatchQueueSize int
	// MaxWatchers caps concurrent watch streams per replica.
	MaxWatchers int
	// MaxMessageBytes caps gRPC message sizes.
	MaxMessageBytes int

	// AdvertiseEndpoints are returned by GetServerInfo as client fallbacks.
	AdvertiseEndpoints AdvertiseEndpoints

	// TLSCertFile and TLSKeyFile enable TLS on the gRPC and REST listeners.
	TLSCertFile string
	TLSKeyFile  string

	// ShutdownDrain is how long the replica reports not ready before it
	// stops accepting work, so the load balancer deregisters it.
	ShutdownDrain time.Duration
	// ShutdownTimeout bounds the graceful stop of the gRPC server.
	ShutdownTimeout time.Duration

	// OTelEnabled turns on OpenTelemetry tracing, configured through the
	// standard OTEL_* environment variables.
	OTelEnabled bool
	// LogLevel is debug, info, warn or error.
	LogLevel string
}

// AdvertiseEndpoint is one entry of --advertise-endpoints. The JSON names
// match dynamo.mcs.v1.ServerEndpoint.
type AdvertiseEndpoint struct {
	Region  string `json:"region"`
	GRPCURL string `json:"grpcUrl,omitempty"`
	HTTPURL string `json:"httpUrl,omitempty"`
	Primary bool   `json:"primary,omitempty"`
}

// AdvertiseEndpoints is a flag.Value holding a JSON array of AdvertiseEndpoint,
// for example
// [{"region":"us-east-1","grpcUrl":"https://registry.us-east-1.example:443","primary":true}].
type AdvertiseEndpoints []AdvertiseEndpoint

// String implements flag.Value.
func (a *AdvertiseEndpoints) String() string {
	if a == nil || len(*a) == 0 {
		return ""
	}
	b, err := json.Marshal([]AdvertiseEndpoint(*a))
	if err != nil {
		return ""
	}
	return string(b)
}

// Set implements flag.Value. An empty value clears the list.
func (a *AdvertiseEndpoints) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		*a = nil
		return nil
	}
	dec := json.NewDecoder(strings.NewReader(value))
	dec.DisallowUnknownFields()
	var eps []AdvertiseEndpoint
	if err := dec.Decode(&eps); err != nil {
		return fmt.Errorf("want a JSON array of {region, grpcUrl, httpUrl, primary}: %w", err)
	}
	*a = eps
	return nil
}

// Default returns the default configuration.
func Default() *Config {
	return &Config{
		GRPCListenAddress:           ":9090",
		HTTPListenAddress:           ":8080",
		AdminListenAddress:          ":9091",
		Store:                       StoreMemory,
		DynamoDBTable:               "global-discovery-mcs",
		RefreshInterval:             2 * time.Second,
		LeaseEvalInterval:           time.Second,
		ClusterSetDiscoveryInterval: 30 * time.Second,
		LeaseSkew:                   2 * time.Second,
		Retention:                   24 * time.Hour,
		CompactionInterval:          10 * time.Minute,
		BookmarkInterval:            5 * time.Second,
		WatchHistory:                10000,
		WatchQueueSize:              1024,
		MaxWatchers:                 5000,
		MaxMessageBytes:             api.MaxMessageBytes,
		ShutdownDrain:               5 * time.Second,
		ShutdownTimeout:             25 * time.Second,
		LogLevel:                    "info",
	}
}

// RegisterFlags registers every flag on fs, bound to c. The current values
// of c are the defaults.
func (c *Config) RegisterFlags(fs *flag.FlagSet) {
	fs.StringVar(&c.GRPCListenAddress, "grpc-listen-address", c.GRPCListenAddress, "gRPC listen address")
	fs.StringVar(&c.HTTPListenAddress, "http-listen-address", c.HTTPListenAddress, "REST gateway listen address")
	fs.StringVar(&c.AdminListenAddress, "admin-listen-address", c.AdminListenAddress, "admin listen address for metrics, health and pprof")
	fs.StringVar(&c.Region, "region", c.Region, "region reported by GetServerInfo; falls back to AWS_REGION")
	fs.StringVar(&c.Store, "store", c.Store, "store backend: memory or dynamodb")
	fs.StringVar(&c.DynamoDBTable, "dynamodb-table", c.DynamoDBTable, "DynamoDB table name")
	fs.StringVar(&c.DynamoDBEndpoint, "dynamodb-endpoint", c.DynamoDBEndpoint, "DynamoDB endpoint override, for dynamodb-local")
	fs.DurationVar(&c.RefreshInterval, "refresh-interval", c.RefreshInterval, "interval between full reloads of each clusterset")
	fs.DurationVar(&c.LeaseEvalInterval, "lease-eval-interval", c.LeaseEvalInterval, "interval between lease evaluations")
	fs.DurationVar(&c.ClusterSetDiscoveryInterval, "clusterset-discovery-interval", c.ClusterSetDiscoveryInterval, "interval between listings of the store's clustersets")
	fs.DurationVar(&c.LeaseSkew, "lease-skew", c.LeaseSkew, "clock skew allowance before a lease expires")
	fs.DurationVar(&c.Retention, "retention", c.Retention, "how long records of an expired cluster are kept")
	fs.DurationVar(&c.CompactionInterval, "compaction-interval", c.CompactionInterval, "interval between compaction runs")
	fs.DurationVar(&c.BookmarkInterval, "bookmark-interval", c.BookmarkInterval, "interval between watch bookmarks")
	fs.IntVar(&c.WatchHistory, "watch-history", c.WatchHistory, "watch events kept per clusterset for resume")
	fs.IntVar(&c.WatchQueueSize, "watch-queue-size", c.WatchQueueSize, "per-watcher event queue length")
	fs.IntVar(&c.MaxWatchers, "max-watchers", c.MaxWatchers, "maximum concurrent watch streams")
	fs.IntVar(&c.MaxMessageBytes, "max-message-bytes", c.MaxMessageBytes, "maximum gRPC message size in bytes")
	fs.Var(&c.AdvertiseEndpoints, "advertise-endpoints", `JSON array of {"region","grpcUrl","httpUrl","primary"} returned by GetServerInfo`)
	fs.StringVar(&c.TLSCertFile, "tls-cert-file", c.TLSCertFile, "TLS certificate for the gRPC and REST listeners")
	fs.StringVar(&c.TLSKeyFile, "tls-key-file", c.TLSKeyFile, "TLS private key for the gRPC and REST listeners")
	fs.DurationVar(&c.ShutdownDrain, "shutdown-drain", c.ShutdownDrain, "time reported not ready before shutdown")
	fs.DurationVar(&c.ShutdownTimeout, "shutdown-timeout", c.ShutdownTimeout, "graceful stop timeout of the gRPC server")
	fs.BoolVar(&c.OTelEnabled, "otel-enabled", c.OTelEnabled, "enable OpenTelemetry tracing (OTEL_* environment)")
	fs.StringVar(&c.LogLevel, "log-level", c.LogLevel, "log level: debug, info, warn or error")
}

// EnvName returns the environment variable of a flag.
func EnvName(flagName string) string {
	return EnvPrefix + strings.ToUpper(strings.ReplaceAll(flagName, "-", "_"))
}

// ApplyEnv sets every flag of fs that was not set on the command line from
// its environment variable, when that variable is present.
func ApplyEnv(fs *flag.FlagSet, lookupEnv func(string) (string, bool)) error {
	set := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { set[f.Name] = true })
	var errs []error
	fs.VisitAll(func(f *flag.Flag) {
		if set[f.Name] {
			return
		}
		name := EnvName(f.Name)
		value, ok := lookupEnv(name)
		if !ok {
			return
		}
		if err := fs.Set(f.Name, value); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", name, err))
		}
	})
	return errors.Join(errs...)
}

// Parse parses the command-line arguments (without the program name),
// applies the environment fallbacks and validates the result. Usage goes to
// output. It returns flag.ErrHelp when -h or --help is given.
func Parse(args []string, lookupEnv func(string) (string, bool), output io.Writer) (*Config, error) {
	c := Default()
	fs := flag.NewFlagSet("mcs-registry", flag.ContinueOnError)
	fs.SetOutput(output)
	c.RegisterFlags(fs)
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if fs.NArg() > 0 {
		return nil, fmt.Errorf("unexpected arguments: %s", strings.Join(fs.Args(), " "))
	}
	if err := ApplyEnv(fs, lookupEnv); err != nil {
		return nil, err
	}
	if c.Region == "" {
		if region, ok := lookupEnv("AWS_REGION"); ok {
			c.Region = region
		}
	}
	if err := c.Validate(); err != nil {
		return nil, err
	}
	return c, nil
}

// Validate reports every invalid setting.
func (c *Config) Validate() error {
	var errs []error
	add := func(format string, args ...any) { errs = append(errs, fmt.Errorf(format, args...)) }

	addrs := map[string]string{}
	for _, a := range []struct{ flag, value string }{
		{"grpc-listen-address", c.GRPCListenAddress},
		{"http-listen-address", c.HTTPListenAddress},
		{"admin-listen-address", c.AdminListenAddress},
	} {
		if _, _, err := net.SplitHostPort(a.value); err != nil {
			add("--%s %q: want host:port", a.flag, a.value)
			continue
		}
		if other, dup := addrs[a.value]; dup {
			add("--%s and --%s both use %q", other, a.flag, a.value)
		}
		addrs[a.value] = a.flag
	}

	switch c.Store {
	case StoreMemory:
	case StoreDynamoDB:
		if c.DynamoDBTable == "" {
			add("--dynamodb-table is required with --store=%s", StoreDynamoDB)
		}
	default:
		add("--store %q: want %s or %s", c.Store, StoreMemory, StoreDynamoDB)
	}
	if c.DynamoDBEndpoint != "" {
		if err := checkURL(c.DynamoDBEndpoint); err != nil {
			add("--dynamodb-endpoint: %v", err)
		}
	}

	positive := []struct {
		flag  string
		value time.Duration
		min   time.Duration
	}{
		{"refresh-interval", c.RefreshInterval, 100 * time.Millisecond},
		{"lease-eval-interval", c.LeaseEvalInterval, 100 * time.Millisecond},
		{"clusterset-discovery-interval", c.ClusterSetDiscoveryInterval, time.Second},
		{"retention", c.Retention, 5 * time.Minute},
		{"compaction-interval", c.CompactionInterval, time.Second},
		{"bookmark-interval", c.BookmarkInterval, time.Second},
		{"shutdown-timeout", c.ShutdownTimeout, time.Second},
	}
	for _, d := range positive {
		if d.value < d.min {
			add("--%s %s: want at least %s", d.flag, d.value, d.min)
		}
	}
	// The bookmark keeps idle streams under the 350 s NLB idle timeout.
	if c.BookmarkInterval > 5*time.Minute {
		add("--bookmark-interval %s: want at most 5m0s", c.BookmarkInterval)
	}
	if c.LeaseSkew < 0 || c.LeaseSkew > time.Minute {
		add("--lease-skew %s: want between 0s and 1m0s", c.LeaseSkew)
	}
	if c.ShutdownDrain < 0 {
		add("--shutdown-drain %s: want at least 0s", c.ShutdownDrain)
	}

	for _, n := range []struct {
		flag  string
		value int
		min   int
	}{
		{"watch-history", c.WatchHistory, 1},
		{"watch-queue-size", c.WatchQueueSize, 1},
		{"max-watchers", c.MaxWatchers, 1},
		{"max-message-bytes", c.MaxMessageBytes, 1 << 20},
	} {
		if n.value < n.min {
			add("--%s %d: want at least %d", n.flag, n.value, n.min)
		}
	}

	if (c.TLSCertFile == "") != (c.TLSKeyFile == "") {
		add("--tls-cert-file and --tls-key-file must be set together")
	}

	primaries := 0
	for i, ep := range c.AdvertiseEndpoints {
		if ep.Region == "" {
			add("--advertise-endpoints[%d]: region is required", i)
		}
		if ep.GRPCURL == "" && ep.HTTPURL == "" {
			add("--advertise-endpoints[%d]: grpcUrl or httpUrl is required", i)
		}
		for _, u := range []struct{ field, value string }{{"grpcUrl", ep.GRPCURL}, {"httpUrl", ep.HTTPURL}} {
			if u.value == "" {
				continue
			}
			if err := checkURL(u.value); err != nil {
				add("--advertise-endpoints[%d].%s: %v", i, u.field, err)
			}
		}
		if ep.Primary {
			primaries++
		}
	}
	if primaries > 1 {
		add("--advertise-endpoints: at most one entry may be primary, got %d", primaries)
	}

	switch c.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		add("--log-level %q: want debug, info, warn or error", c.LogLevel)
	}
	return errors.Join(errs...)
}

func checkURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("%q: want an http or https URL", raw)
	}
	if u.Host == "" {
		return fmt.Errorf("%q: host is required", raw)
	}
	return nil
}
