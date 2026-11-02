// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package lease keeps the cluster registered with the registry:
// RegisterCluster, then HeartbeatCluster every TTL/3 with jitter. A heartbeat
// refused with ABORTED LEASE_EPOCH_MISMATCH or NOT_FOUND registers again and
// asks for a full resend of the exports, since the registry may have dropped
// them.
package lease

import (
	"context"
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/durationpb"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/tracing"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

// DeregisterTimeout bounds DeregisterCluster at shutdown.
const DeregisterTimeout = 5 * time.Second

// Jitter is the relative jitter of the heartbeat interval.
const Jitter = 0.1

// Options configure a Lease.
type Options struct {
	Registry *client.Client
	Identity identity.Identity
	// TTL is the requested lease TTL; heartbeats run every TTL/3.
	TTL time.Duration
	// AgentID and AgentVersion identify the controller in the registration.
	AgentID      string
	AgentVersion string
	// DeregisterOnShutdown deregisters the cluster when Start returns.
	DeregisterOnShutdown bool
	Metrics              *metrics.Metrics
	Logger               logr.Logger
	// OnRegistered runs after every successful registration; first is true for
	// the first one. A later registration means the registry lost or replaced
	// the lease, so the exports must be resent.
	OnRegistered func(first bool)
	// Backoff spaces failed registrations and heartbeats; zero fields take
	// 1 s to TTL/3.
	Backoff client.Backoff
}

// Lease is a leader-only manager.Runnable that holds the cluster lease.
type Lease struct {
	opts Options

	kick       chan struct{}
	registered atomic.Bool
	epoch      atomic.Uint64

	mu      sync.Mutex
	waiters chan struct{}
	first   bool
}

// New returns a Lease. Start it with the manager.
func New(opts Options) *Lease {
	if opts.TTL <= 0 {
		opts.TTL = 30 * time.Second
	}
	if opts.Backoff.Initial <= 0 {
		opts.Backoff.Initial = time.Second
	}
	if opts.Backoff.Max <= 0 {
		opts.Backoff.Max = opts.TTL / 3
	}
	if opts.Backoff.Multiplier < 1 {
		opts.Backoff.Multiplier = 2
	}
	if opts.Backoff.Jitter == 0 {
		opts.Backoff.Jitter = 0.2
	}
	if opts.Metrics == nil {
		opts.Metrics = metrics.New()
	}
	return &Lease{opts: opts, kick: make(chan struct{}, 1), waiters: make(chan struct{}), first: true}
}

// NeedLeaderElection makes the lease leader-only: one writer per cluster.
func (l *Lease) NeedLeaderElection() bool { return true }

// Registered reports whether the cluster holds a lease.
func (l *Lease) Registered() bool { return l.registered.Load() }

// Epoch returns the current lease epoch, 0 before the first registration.
func (l *Lease) Epoch() uint64 { return l.epoch.Load() }

// Kick runs the next heartbeat (or registration) now, for example after a
// write failed with CLUSTER_NOT_REGISTERED.
func (l *Lease) Kick() {
	select {
	case l.kick <- struct{}{}:
	default:
	}
}

// WaitRegistered blocks until the cluster is registered or ctx is done.
func (l *Lease) WaitRegistered(ctx context.Context) error {
	for {
		l.mu.Lock()
		ch := l.waiters
		l.mu.Unlock()
		if l.Registered() {
			return nil
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (l *Lease) setRegistered(v bool) {
	l.registered.Store(v)
	if v {
		l.opts.Metrics.ClusterRegistered.Set(1)
		l.mu.Lock()
		close(l.waiters)
		l.waiters = make(chan struct{})
		l.mu.Unlock()
		return
	}
	l.opts.Metrics.ClusterRegistered.Set(0)
}

// Start registers the cluster and heartbeats until ctx is done. It
// deregisters on the way out when DeregisterOnShutdown is set.
func (l *Lease) Start(ctx context.Context) error {
	log := l.opts.Logger.WithValues("clusterset", l.opts.Identity.ClusterSet, "cluster", l.opts.Identity.Cluster)
	defer func() {
		wasRegistered := l.Registered()
		l.setRegistered(false)
		if l.opts.DeregisterOnShutdown && wasRegistered {
			l.deregister(log)
		}
	}()
	interval := l.opts.TTL / 3
	failures := 0
	// A few failures are a degraded, recoverable state; persistent ones
	// need attention.
	logRetry := func(msg string, err error, delay time.Duration) {
		if failures >= 3 {
			log.Error(err, msg, "delay", delay.String(), "failures", failures)
			return
		}
		log.Info(msg, "error", err.Error(), "delay", delay.String())
	}
	for {
		var delay time.Duration
		if !l.Registered() {
			if err := l.register(ctx); err != nil {
				if ctx.Err() != nil {
					return nil
				}
				delay = l.opts.Backoff.Delay(failures)
				failures++
				logRetry("Registering the cluster with the registry failed; retrying", err, delay)
			} else {
				failures = 0
				delay = jittered(interval)
			}
		} else {
			switch err := l.heartbeat(ctx); {
			case err == nil:
				failures = 0
				delay = jittered(interval)
			case ctx.Err() != nil:
				return nil
			case lost(err):
				log.Info("The registry refused the heartbeat; registering again", "error", err.Error())
				l.setRegistered(false)
				continue
			default:
				delay = min(l.opts.Backoff.Delay(failures), interval)
				failures++
				logRetry("Cluster heartbeat failed; retrying", err, delay)
			}
		}
		t := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			t.Stop()
			return nil
		case <-l.kick:
			t.Stop()
		case <-t.C:
		}
	}
}

func (l *Lease) register(ctx context.Context) (err error) {
	id := l.opts.Identity
	ctx, span := tracing.Start(ctx, "register_cluster", id.ClusterSet, id.Cluster, "", "")
	defer func() { tracing.End(span, err) }()
	resp, err := l.opts.Registry.Clusters().Register(ctx, &mcsv1.RegisterClusterRequest{
		Clusterset:   id.ClusterSet,
		Cluster:      id.Cluster,
		Region:       id.Region,
		Zones:        id.Zones,
		Provider:     id.Provider,
		Labels:       id.Labels,
		AgentId:      l.opts.AgentID,
		AgentVersion: l.opts.AgentVersion,
		LeaseTtl:     durationpb.New(l.opts.TTL),
	})
	if err != nil {
		l.opts.Metrics.Registrations.WithLabelValues(metrics.ResultError).Inc()
		return err
	}
	l.opts.Metrics.Registrations.WithLabelValues(metrics.ResultSuccess).Inc()
	l.epoch.Store(resp.GetCluster().GetLease().GetEpoch())
	l.mu.Lock()
	first := l.first
	l.first = false
	l.mu.Unlock()
	l.setRegistered(true)
	l.opts.Logger.Info("Cluster registered with the registry", "clusterset", id.ClusterSet, "cluster", id.Cluster,
		"region", id.Region, "epoch", l.Epoch(), "ttl", l.opts.TTL.String(), "first", first)
	if l.opts.OnRegistered != nil {
		l.opts.OnRegistered(first)
	}
	return nil
}

func (l *Lease) heartbeat(ctx context.Context) (err error) {
	id := l.opts.Identity
	ctx, span := tracing.Start(ctx, "heartbeat_cluster", id.ClusterSet, id.Cluster, "", "")
	defer func() { tracing.End(span, err) }()
	_, err = l.opts.Registry.Clusters().Heartbeat(ctx, &mcsv1.HeartbeatClusterRequest{
		Clusterset: id.ClusterSet,
		Cluster:    id.Cluster,
		Epoch:      l.Epoch(),
	})
	switch {
	case err == nil:
	case client.IsLeaseEpochMismatch(err) || status.Code(err) == codes.Aborted:
		l.opts.Metrics.HeartbeatFailures.WithLabelValues(metrics.HeartbeatEpochMismatch).Inc()
	case client.IsClusterNotRegistered(err) || status.Code(err) == codes.NotFound:
		l.opts.Metrics.HeartbeatFailures.WithLabelValues(metrics.HeartbeatNotRegistered).Inc()
	default:
		l.opts.Metrics.HeartbeatFailures.WithLabelValues(metrics.HeartbeatError).Inc()
	}
	return err
}

func (l *Lease) deregister(log logr.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), DeregisterTimeout)
	defer cancel()
	id := l.opts.Identity
	_, err := l.opts.Registry.Clusters().Deregister(ctx, &mcsv1.DeregisterClusterRequest{Clusterset: id.ClusterSet, Cluster: id.Cluster})
	if err != nil {
		log.Error(err, "Deregistering the cluster from the registry failed; its lease will expire")
		return
	}
	log.Info("Cluster deregistered from the registry")
}

// lost reports whether a heartbeat error means the lease is gone: another
// epoch holds it or the registry does not know the cluster.
func lost(err error) bool {
	if client.IsLeaseEpochMismatch(err) || client.IsClusterNotRegistered(err) {
		return true
	}
	switch status.Code(err) {
	case codes.Aborted, codes.NotFound:
		return true
	}
	return false
}

// jittered returns d varied by +-Jitter.
func jittered(d time.Duration) time.Duration {
	return time.Duration(float64(d) * (1 + Jitter*(2*rand.Float64()-1)))
}
