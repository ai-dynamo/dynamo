// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package view

import (
	"context"
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/otel"
	"go.uber.org/zap"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/clock"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// CompactorOptions configures a Compactor.
type CompactorOptions struct {
	Clock clock.Clock
	// Retention keeps the records of a dead cluster this long after its
	// lease expired, so a returning writer keeps its export creation
	// timestamps.
	Retention time.Duration
	// RetireClusters lets compaction delete a cluster whose lease expired
	// more than Retention ago, with everything it owns. Off by default: an
	// expired cluster is only filtered from imports at read time, its
	// records stay, and an operator retires it with mcs-admin
	// delete-cluster. Every replica deletes on its own observation of the
	// lease, so a replica that has not yet seen a heartbeat or a
	// re-registration can cascade over a live cluster; see
	// docs/consistency.md.
	RetireClusters bool
	// Interval is how often compaction runs.
	Interval time.Duration
	// LeaseSkew is the clock skew allowance of lease evaluation.
	LeaseSkew time.Duration
	Metrics   *metrics.Metrics
	Logger    *zap.Logger
}

// Compactor removes records past retention from the store and the view:
//
//   - exports of an unregistered cluster, and endpoint sets without an
//     export, not updated for Retention;
//   - with RetireClusters, a cluster whose lease expired more than
//     Retention ago, with everything it owns (store.DeleteCluster).
//
// Every replica compacts. Deletes are idempotent, so concurrent compactors
// agree. The store has no conditional delete; a cluster is re-read just
// before its deletion to narrow the race with a returning controller, which
// recovers anyway by registering again on NOT_FOUND and resending its
// exports. That re-read does not make the deletion safe, which is why
// retirement is opt-in.
type Compactor struct {
	view  *View
	store store.Store
	opts  CompactorOptions
}

// NewCompactor returns a compactor of v and st.
func NewCompactor(v *View, st store.Store, opts CompactorOptions) *Compactor {
	if opts.Clock == nil {
		opts.Clock = v.clock
	}
	if opts.Logger == nil {
		opts.Logger = zap.NewNop()
	}
	if opts.Retention <= 0 {
		opts.Retention = 24 * time.Hour
	}
	if opts.Interval <= 0 {
		opts.Interval = 10 * time.Minute
	}
	return &Compactor{view: v, store: st, opts: opts}
}

// Run compacts every Interval until ctx is done.
func (c *Compactor) Run(ctx context.Context) {
	t := c.opts.Clock.NewTicker(c.opts.Interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C():
			if err := c.Compact(ctx); err != nil && ctx.Err() == nil {
				c.opts.Logger.Warn("compaction failed", zap.Error(err))
			}
		}
	}
}

func (c *Compactor) deleted(kind string, n int) {
	if c.opts.Metrics != nil && n > 0 {
		c.opts.Metrics.CompactionDeleted.WithLabelValues(kind).Add(float64(n))
	}
}

// dead reports whether a lease expired more than the retention ago.
func (c *Compactor) dead(l store.Lease, now time.Time) bool {
	return now.After(l.ExpiresAt().Add(c.opts.LeaseSkew).Add(c.opts.Retention))
}

// Compact runs one compaction pass over every tracked clusterset.
func (c *Compactor) Compact(ctx context.Context) error {
	ctx, span := otel.Tracer(tracerName).Start(ctx, "mcs.view.compact")
	defer span.End()
	var errs []error
	for _, name := range c.view.ClusterSetNames() {
		if err := c.compactClusterSet(ctx, name); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (c *Compactor) compactClusterSet(ctx context.Context, cs string) error {
	now := c.opts.Clock.Now()
	snap := c.view.Records(cs)
	log := c.opts.Logger.With(zap.String("clusterset", cs))
	var errs []error

	registered := map[string]bool{}
	gone := map[string]bool{}
	for _, r := range snap.Clusters {
		registered[r.Key.Cluster] = true
		if !c.opts.RetireClusters || !c.dead(r.Lease, now) {
			continue
		}
		current, err := c.store.GetCluster(ctx, r.Key)
		switch {
		case errors.Is(err, store.ErrNotFound):
			c.view.ApplyClusterDelete(r.Key)
			delete(registered, r.Key.Cluster)
			gone[r.Key.Cluster] = true
			continue
		case err != nil:
			errs = append(errs, fmt.Errorf("get cluster %s: %w", r.Key.Cluster, err))
			continue
		case !c.dead(current.Lease, now):
			c.view.ApplyCluster(current)
			continue
		}
		if err := c.store.DeleteCluster(ctx, r.Key); err != nil {
			errs = append(errs, fmt.Errorf("delete cluster %s: %w", r.Key.Cluster, err))
			continue
		}
		c.view.ApplyClusterDelete(r.Key)
		delete(registered, r.Key.Cluster)
		gone[r.Key.Cluster] = true
		c.deleted(metrics.RecordCluster, 1)
		log.Info("compacted expired cluster", zap.String("cluster", r.Key.Cluster),
			zap.Time("lease_expired_at", r.Lease.ExpiresAt()))
	}

	old := func(t time.Time) bool { return now.After(t.Add(c.opts.Retention)) }
	exports := map[store.ExportKey]bool{}
	for _, r := range snap.Exports {
		if gone[r.Key.Cluster] {
			continue
		}
		exports[r.Key] = true
		if registered[r.Key.Cluster] || !old(r.UpdatedAt) {
			continue
		}
		if err := c.store.DeleteExport(ctx, r.Key); err != nil {
			errs = append(errs, fmt.Errorf("delete export %s/%s/%s: %w", r.Key.Cluster, r.Key.Namespace, r.Key.Name, err))
			continue
		}
		c.view.ApplyExportDelete(r.Key)
		delete(exports, r.Key)
		c.deleted(metrics.RecordExport, 1)
	}
	for _, r := range snap.EndpointSets {
		if gone[r.Key.Cluster] || exports[r.Key] || !old(r.UpdatedAt) {
			continue
		}
		if err := c.store.DeleteEndpointSet(ctx, r.Key); err != nil {
			errs = append(errs, fmt.Errorf("delete endpoint set %s/%s/%s: %w", r.Key.Cluster, r.Key.Namespace, r.Key.Name, err))
			continue
		}
		c.view.ApplyEndpointSetDelete(r.Key)
		c.deleted(metrics.RecordEndpointSet, 1)
	}
	return errors.Join(errs...)
}
