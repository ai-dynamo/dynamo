// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"math/rand/v2"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// Defaults applied by New to zero Config fields.
const (
	DefaultConnectTimeout       = 5 * time.Second
	DefaultRPCTimeout           = 10 * time.Second
	DefaultPrimaryRetryInterval = 5 * time.Minute
	DefaultMaxAttempts          = 4
	DefaultUserAgent            = "global-discovery-mcs-client-go"

	// maxRetryAttempts is the gRPC limit on retryPolicy.maxAttempts.
	maxRetryAttempts = 5
)

// DefaultBackoff is the backoff of unary retries, connection attempts and
// watch reconnects.
var DefaultBackoff = Backoff{Initial: 200 * time.Millisecond, Max: 30 * time.Second, Multiplier: 2.0, Jitter: 0.5}

// Backoff is an exponential backoff with jitter.
type Backoff struct {
	// Initial is the first delay.
	Initial time.Duration
	// Max caps every delay.
	Max time.Duration
	// Multiplier grows the delay after each attempt.
	Multiplier float64
	// Jitter randomises each delay by up to this fraction in either
	// direction, for example 0.5 for [0.5d, 1.5d].
	Jitter float64
}

// withDefaults returns DefaultBackoff for the zero Backoff, and otherwise
// fills unset or invalid fields; a Jitter of 0 in a non-zero Backoff
// disables jitter.
func (b Backoff) withDefaults() Backoff {
	if b == (Backoff{}) {
		return DefaultBackoff
	}
	if b.Initial <= 0 {
		b.Initial = DefaultBackoff.Initial
	}
	if b.Max <= 0 {
		b.Max = DefaultBackoff.Max
	}
	if b.Max < b.Initial {
		b.Max = b.Initial
	}
	if b.Multiplier < 1 {
		b.Multiplier = DefaultBackoff.Multiplier
	}
	if b.Jitter < 0 || b.Jitter > 1 {
		b.Jitter = DefaultBackoff.Jitter
	}
	return b
}

// Delay returns the delay before retry attempt n (0 for the first retry):
// Initial * Multiplier^n capped at Max, then jittered and capped again.
func (b Backoff) Delay(n int) time.Duration {
	b = b.withDefaults()
	d := float64(b.Initial) * math.Pow(b.Multiplier, float64(max(n, 0)))
	d = min(d, float64(b.Max))
	if b.Jitter > 0 {
		d *= 1 + b.Jitter*(2*rand.Float64()-1)
	}
	return time.Duration(min(d, float64(b.Max)))
}

// Resolver resolves host names to addresses. *net.Resolver implements it.
type Resolver interface {
	LookupHost(ctx context.Context, host string) (addrs []string, err error)
}

// Config configures a Client. Zero fields take the defaults above.
type Config struct {
	// URLs is the ordered list of registry endpoints, for example
	// https://registry.prod.example.internal followed by per-region fallbacks.
	// Every URL uses the same scheme; http requires AllowInsecureHTTP.
	// Ports default to 443 for https and 80 for http.
	URLs []string
	// CAFile is a PEM bundle of the CAs that sign the registry certificates.
	// It is mutually exclusive with TLS.RootCAs.
	CAFile string
	// TLS is the base TLS configuration for https URLs. ServerName is
	// normally left empty: the client sets it per URL.
	TLS *tls.Config
	// AllowInsecureHTTP permits plaintext http:// URLs, for development and
	// in-VPC deployments behind a TLS-terminating load balancer.
	AllowInsecureHTTP bool
	// ConnectTimeout bounds one connection attempt and one DNS lookup.
	ConnectTimeout time.Duration
	// RPCTimeout is the deadline of unary calls whose context has none.
	// Negative disables it.
	RPCTimeout time.Duration
	// PrimaryRetryInterval is how often a client connected to another URL
	// probes the primary (the first URL) and returns to it when it serves.
	// Negative disables the return.
	PrimaryRetryInterval time.Duration
	// Backoff applies to unary retries, connection attempts and watch
	// reconnects.
	Backoff Backoff
	// MaxAttempts is the number of attempts of a unary call failing with
	// UNAVAILABLE, the first included. 1 disables retries; gRPC caps it at 5.
	MaxAttempts int
	// UserAgent prefixes the gRPC user agent.
	UserAgent string
	// Registerer receives the dynamo_mcs_client_* metrics. Nil registers
	// nothing.
	Registerer prometheus.Registerer
	// Resolver resolves the URL host names on every reconnect. Nil uses
	// net.DefaultResolver.
	Resolver Resolver

	// Dialer, when set, replaces the TCP dialer. It receives host:port
	// with the resolved IP. Tests use it for bufconn listeners.
	Dialer func(ctx context.Context, addr string) (net.Conn, error)
	// DialOptions are appended to the client's dial options, for example
	// an OpenTelemetry stats handler. They must not set the dialer, the
	// resolver, the transport credentials or the default service config.
	DialOptions []grpc.DialOption
	// Logger receives failover and watch logs. Nil uses slog.Default().
	Logger *slog.Logger
}

// target is one parsed registry URL.
type target struct {
	// url is the canonical form scheme://host:port, used as the metrics
	// label and to deduplicate.
	url    string
	scheme string
	host   string
	port   string
	// serverName is the TLS server name and HTTP/2 authority: the host and
	// port as written in the URL.
	serverName string
}

func parseTarget(raw string, allowHTTP bool) (target, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return target{}, fmt.Errorf("mcs client: invalid URL %q: %w", raw, err)
	}
	var port string
	switch u.Scheme {
	case "https":
		port = "443"
	case "http":
		if !allowHTTP {
			return target{}, fmt.Errorf("mcs client: plaintext URL %q requires AllowInsecureHTTP", raw)
		}
		port = "80"
	default:
		return target{}, fmt.Errorf("mcs client: URL %q: scheme must be https or http", raw)
	}
	switch {
	case u.Hostname() == "":
		return target{}, fmt.Errorf("mcs client: URL %q has no host", raw)
	case u.User != nil:
		return target{}, fmt.Errorf("mcs client: URL %q must not carry credentials", raw)
	case u.Path != "" && u.Path != "/", u.RawQuery != "", u.Fragment != "":
		return target{}, fmt.Errorf("mcs client: URL %q must not have a path, query or fragment", raw)
	}
	if p := u.Port(); p != "" {
		n, err := strconv.Atoi(p)
		if err != nil || n < 1 || n > 65535 {
			return target{}, fmt.Errorf("mcs client: URL %q has an invalid port", raw)
		}
		port = p
	}
	host := strings.ToLower(u.Hostname())
	return target{
		url:        u.Scheme + "://" + net.JoinHostPort(host, port),
		scheme:     u.Scheme,
		host:       host,
		port:       port,
		serverName: u.Host,
	}, nil
}

// normalize validates cfg, applies defaults and parses the URLs.
func normalize(cfg Config) (Config, []target, error) {
	if len(cfg.URLs) == 0 {
		return cfg, nil, errors.New("mcs client: no URLs configured")
	}
	var targets []target
	seen := map[string]bool{}
	for _, raw := range cfg.URLs {
		t, err := parseTarget(raw, cfg.AllowInsecureHTTP)
		if err != nil {
			return cfg, nil, err
		}
		if len(targets) > 0 && t.scheme != targets[0].scheme {
			return cfg, nil, fmt.Errorf("mcs client: URL %q: every URL must use the scheme %s", raw, targets[0].scheme)
		}
		if seen[t.url] {
			continue
		}
		seen[t.url] = true
		targets = append(targets, t)
	}
	if cfg.ConnectTimeout <= 0 {
		cfg.ConnectTimeout = DefaultConnectTimeout
	}
	if cfg.RPCTimeout == 0 {
		cfg.RPCTimeout = DefaultRPCTimeout
	}
	if cfg.PrimaryRetryInterval == 0 {
		cfg.PrimaryRetryInterval = DefaultPrimaryRetryInterval
	}
	cfg.Backoff = cfg.Backoff.withDefaults()
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = DefaultMaxAttempts
	}
	cfg.MaxAttempts = min(cfg.MaxAttempts, maxRetryAttempts)
	if cfg.UserAgent == "" {
		cfg.UserAgent = DefaultUserAgent
	}
	if cfg.Resolver == nil {
		cfg.Resolver = net.DefaultResolver
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return cfg, targets, nil
}

// transportCredentials returns TLS credentials for https and insecure ones
// for http.
func transportCredentials(cfg Config, scheme string) (credentials.TransportCredentials, error) {
	if scheme == "http" {
		return insecure.NewCredentials(), nil
	}
	tc := &tls.Config{MinVersion: tls.VersionTLS12}
	if cfg.TLS != nil {
		tc = cfg.TLS.Clone()
	}
	if cfg.CAFile != "" {
		if tc.RootCAs != nil {
			return nil, errors.New("mcs client: CAFile and TLS.RootCAs are mutually exclusive")
		}
		pem, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("mcs client: read CA file: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return nil, fmt.Errorf("mcs client: CA file %s holds no PEM certificate", cfg.CAFile)
		}
		tc.RootCAs = pool
	}
	return credentials.NewTLS(tc), nil
}
