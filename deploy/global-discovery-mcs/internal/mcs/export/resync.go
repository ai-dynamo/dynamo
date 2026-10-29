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
// reason labels dynamo_mcs_controller_resyncs_total.
func (r *Reconciler) Resync(ctx context.Context, reason string) error {
	r.opts.Metrics.Resyncs.WithLabelValues(reason).Inc()
	r.ResetSent()
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
	id := r.opts.Identity
	resp, err := r.opts.Registry.ServiceExports().List(ctx, &mcsv1.ListServiceExportsRequest{Clusterset: id.ClusterSet, Cluster: id.Cluster})
	if err != nil {
		return fmt.Errorf("list registry exports: %w", err)
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
}

// Resyncer is a leader-only manager.Runnable that runs Resync once the
// cluster is registered, and again after every Trigger.
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

// Start runs the startup resync, then a resync per trigger, until ctx is
// done. Failed resyncs are retried with backoff.
func (s *Resyncer) Start(ctx context.Context) error {
	reason := metrics.ResyncStartup
	ctx = logr.NewContext(ctx, s.opts.Logger)
	for {
		for attempt := 0; ; attempt++ {
			if s.opts.Lease != nil {
				if err := s.opts.Lease.WaitRegistered(ctx); err != nil {
					return nil
				}
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
		select {
		case <-ctx.Done():
			return nil
		case reason = <-s.trigger:
		}
	}
}
