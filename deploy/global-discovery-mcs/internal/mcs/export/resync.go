// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

// Resync garbage-collects and resends: it deletes the registry export records
// of this cluster that have no local ServiceExport, forgets what was written,
// and reconciles every local ServiceExport so that each is written again.
// reason labels dynamo_mcs_controller_resync_attempts_total and, once the
// resync completes, dynamo_mcs_controller_resyncs_total.
//
// The registry is listed before the cache: a record the registry holds was
// written by a reconcile that saw its ServiceExport, so the later cache
// list includes that ServiceExport unless it was deleted since. Listing the
// cache first would let a ServiceExport created and exported between the
// two lists look like an orphan.
func (r *Reconciler) Resync(ctx context.Context, reason string) error {
	r.opts.Metrics.ResyncAttempts.WithLabelValues(reason).Inc()
	r.ResetSent()
	id := r.opts.Identity
	resp, err := r.opts.Registry.ServiceExports().List(ctx, &mcsv1.ListServiceExportsRequest{Clusterset: id.ClusterSet, Cluster: id.Cluster})
	if err != nil {
		return fmt.Errorf("list registry exports: %w", err)
	}
	var list mcsv1beta1.ServiceExportList
	if err := r.opts.Client.List(ctx, &list); err != nil {
		return fmt.Errorf("list ServiceExports: %w", err)
	}
	local := map[types.NamespacedName]bool{}
	for _, se := range list.Items {
		if r.opts.WatchesNamespace(se.Namespace) {
			local[types.NamespacedName{Namespace: se.Namespace, Name: se.Name}] = true
		}
	}
	for _, e := range resp.GetExports() {
		key := types.NamespacedName{Namespace: e.GetNamespace(), Name: e.GetName()}
		if local[key] {
			continue
		}
		if err := r.deleteRecord(ctx, key); err != nil {
			return err
		}
		r.opts.Metrics.GCDeletions.WithLabelValues(metrics.GCExport).Inc()
		logr.FromContextOrDiscard(ctx).Info("Deleted a registry export without a local ServiceExport", "namespace", key.Namespace, "name", key.Name)
	}
	for key := range local {
		select {
		case r.events <- genericEvent(key):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	r.opts.Metrics.Resyncs.WithLabelValues(reason).Inc()
	return nil
}

// ResyncerOptions configure a Resyncer.
type ResyncerOptions struct {
	Reconciler *Reconciler
	// Lease gates resyncs on a registered cluster.
	Lease interface {
		WaitRegistered(ctx context.Context) error
	}
	Logger logr.Logger
	// Backoff spaces failed resyncs; zero fields take 1 s to 30 s.
	Backoff client.Backoff
	// Period is the interval of the periodic resync, measured from the end
	// of the previous resync; 0 disables it.
	Period time.Duration
}

// Resyncer is a leader-only manager.Runnable that runs Resync once the
// cluster is registered, again after every Trigger, and every Period.
type Resyncer struct {
	opts    ResyncerOptions
	trigger chan string
}

// NewResyncer returns a Resyncer.
func NewResyncer(opts ResyncerOptions) *Resyncer {
	if opts.Backoff.Initial <= 0 {
		opts.Backoff.Initial = time.Second
	}
	if opts.Backoff.Max <= 0 {
		opts.Backoff.Max = 30 * time.Second
	}
	if opts.Backoff.Multiplier < 1 {
		opts.Backoff.Multiplier = 2
	}
	return &Resyncer{opts: opts, trigger: make(chan string, 1)}
}

// NeedLeaderElection makes the resyncer leader-only.
func (s *Resyncer) NeedLeaderElection() bool { return true }

// Trigger asks for a resync with reason. Triggers coalesce; the first
// pending reason is kept.
func (s *Resyncer) Trigger(reason string) {
	select {
	case s.trigger <- reason:
	default:
	}
}

// drain discards a pending trigger: the resync about to run covers it.
func (s *Resyncer) drain() {
	select {
	case <-s.trigger:
	default:
	}
}

// Start runs the startup resync, then a resync per trigger and per Period,
// until ctx is done. Failed resyncs are retried with backoff. Before the
// startup resync the reconciler releases ServiceExports in namespaces no
// longer watched (SweepUnwatched).
func (s *Resyncer) Start(ctx context.Context) error {
	reason := metrics.ResyncStartup
	ctx = logr.NewContext(ctx, s.opts.Logger)
	var periodic *time.Timer
	if s.opts.Period > 0 {
		periodic = time.NewTimer(s.opts.Period)
		defer periodic.Stop()
	}
	swept := false
	for {
		for attempt := 0; ; attempt++ {
			if s.opts.Lease != nil {
				if err := s.opts.Lease.WaitRegistered(ctx); err != nil {
					return nil
				}
			}
			// A trigger that arrived while this resync was pending, for
			// example a reconnect on a replica that was not the leader yet, is
			// covered by the resync about to run.
			s.drain()
			if !swept {
				if err := s.opts.Reconciler.SweepUnwatched(ctx); err != nil {
					if ctx.Err() != nil {
						return nil
					}
					delay := s.opts.Backoff.Delay(attempt)
					s.opts.Logger.Info("Sweep of namespaces no longer watched failed; retrying", "error", err.Error(), "delay", delay.String())
					select {
					case <-ctx.Done():
						return nil
					case <-time.After(delay):
					}
					continue
				}
				swept = true
			}
			err := s.opts.Reconciler.Resync(ctx, reason)
			if err == nil {
				s.opts.Logger.Info("Exports resynced with the registry", "reason", reason)
				break
			}
			if ctx.Err() != nil {
				return nil
			}
			delay := s.opts.Backoff.Delay(attempt)
			s.opts.Logger.Info("Resync with the registry failed; retrying", "reason", reason, "error", err.Error(), "delay", delay.String())
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(delay):
			}
		}
		if periodic != nil {
			if !periodic.Stop() {
				select {
				case <-periodic.C:
				default:
				}
			}
			periodic.Reset(s.opts.Period)
		}
		select {
		case <-ctx.Done():
			return nil
		case reason = <-s.trigger:
		case <-timerC(periodic):
			reason = metrics.ResyncPeriodic
		}
	}
}

// timerC returns the channel of t, or nil, which never fires, for no timer.
func timerC(t *time.Timer) <-chan time.Time {
	if t == nil {
		return nil
	}
	return t.C
}
