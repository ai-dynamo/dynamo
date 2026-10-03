/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Deprecated: Package namespace_scope implements the lease-based coordination mechanism for the
// deprecated namespace-restricted operator mode. It will be removed in a future release.
package namespace_scope

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	coordinationv1 "k8s.io/api/coordination/v1"
	k8sErrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// LeaseName is the well-known name for namespace scope marker leases
	LeaseName = "dynamo-operator-namespace-scope"
)

// Deprecated: LeaseManager manages the namespace scope marker lease for the deprecated
// namespace-restricted operator mode.
type LeaseManager struct {
	client          kubernetes.Interface
	namespace       string
	leaseDuration   time.Duration
	renewInterval   time.Duration
	holderIdentity  string
	operatorVersion string
	stopCh          chan struct{}
	errCh           chan error
	wg              sync.WaitGroup
	failureCount    int
	maxFailures     int
	logger          logr.Logger

	// renewCancel is nil until Start creates the renewal context.
	renewCancel  context.CancelFunc
	expiryMu     sync.Mutex
	expiresAt    time.Time
	leaseUpdated chan struct{}
}

// NewLeaseManager creates a new lease manager for namespace scope marking
func NewLeaseManager(config *rest.Config, namespace string, operatorVersion string, leaseDuration time.Duration, renewInterval time.Duration) (*LeaseManager, error) {
	// Validate inputs
	if leaseDuration <= 0 {
		return nil, fmt.Errorf("lease duration must be greater than zero, got %v", leaseDuration)
	}
	if renewInterval <= 0 {
		return nil, fmt.Errorf("renew interval must be greater than zero, got %v", renewInterval)
	}
	if renewInterval >= leaseDuration {
		return nil, fmt.Errorf("renew interval (%v) must be less than lease duration (%v)", renewInterval, leaseDuration)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// Identify this manager instance so failed acquisition cleanup cannot delete another holder.
	holderIdentity := fmt.Sprintf("namespace-restricted-operator-%s-%s", operatorVersion, uuid.NewUUID())

	// Calculate max failures with buffer to ensure operator exits BEFORE lease expires
	// This prevents split-brain: if we allow failures for the full lease duration,
	// the lease expires at exactly the same time we exit, creating a race condition.
	//
	// Strategy: Subtract 1 renewal interval as a safety buffer
	// Example: 30s lease / 10s renewal = 3 intervals
	//          maxFailures = 3 - 1 = 2 → operator exits after 20s of failures
	//          This leaves 10s buffer before lease expires at 30s
	rawMaxFailures := int(leaseDuration / renewInterval)
	maxFailures := rawMaxFailures - 1
	if maxFailures < 1 {
		maxFailures = 1 // Always allow at least 1 failure for transient issues
	}

	return &LeaseManager{
		client:          client,
		namespace:       namespace,
		leaseDuration:   leaseDuration,
		renewInterval:   renewInterval,
		holderIdentity:  holderIdentity,
		operatorVersion: operatorVersion,
		stopCh:          make(chan struct{}),
		maxFailures:     maxFailures,
	}, nil
}

// Errors returns a channel that will receive fatal errors from the lease manager
// Callers should monitor this channel and take appropriate action (e.g., exit to prevent split-brain)
func (lm *LeaseManager) Errors() <-chan error {
	return lm.errCh
}

// Start creates the lease and begins renewal loop
func (lm *LeaseManager) Start(ctx context.Context) error {
	lm.logger = log.FromContext(ctx).WithValues("component", "namespace-scope-lease", "namespace", lm.namespace)

	// Initialize error channel
	lm.errCh = make(chan error, 1) // buffered to avoid blocking
	lm.leaseUpdated = make(chan struct{}, 1)

	// Own the request context so Stop can interrupt an in-flight renewal.
	renewCtx, cancelRenew := context.WithCancel(ctx)
	lm.renewCancel = cancelRenew

	lm.logger.Info("Starting namespace scope marker lease manager",
		"leaseName", LeaseName,
		"leaseDuration", lm.leaseDuration,
		"renewInterval", lm.renewInterval,
		"holderIdentity", lm.holderIdentity,
		"maxFailures", lm.maxFailures)

	// Create or update the lease initially
	if err := lm.createOrUpdateLease(renewCtx); err != nil {
		cancelRenew()
		return fmt.Errorf("failed to create initial lease: %w", err)
	}

	lm.logger.Info("Namespace scope marker lease created successfully")

	// Start renewal loop in background
	lm.wg.Add(1)
	go lm.renewalLoop(renewCtx)

	return nil
}

// Stop stops the lease renewal loop and releases the lease
func (lm *LeaseManager) Stop(ctx context.Context) error {
	lm.logger.Info("Stopping namespace scope marker lease manager")

	// Cancel requests before waiting, including when acquisition failed before the loop started.
	close(lm.stopCh)
	if lm.renewCancel != nil {
		lm.renewCancel()
	}

	// Bound the wait even if a client transport ignores cancellation. Never race deletion
	// against a renewal that has not finished; TTL expiry remains the fallback in that case.
	renewDone := make(chan struct{})
	go func() {
		lm.wg.Wait()
		close(renewDone)
	}()
	select {
	case <-renewDone:
	case <-ctx.Done():
		return fmt.Errorf("waiting for lease renewal to stop: %w", ctx.Err())
	}

	// A failed create/update may have committed. Read ownership even on that path,
	// and condition deletion on this exact version so a concurrent takeover is preserved.
	lease, err := lm.client.CoordinationV1().Leases(lm.namespace).Get(ctx, LeaseName, metav1.GetOptions{})
	if k8sErrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("reading lease ownership before cleanup: %w", err)
	}
	if lease.Spec.HolderIdentity == nil || *lease.Spec.HolderIdentity != lm.holderIdentity {
		return nil
	}
	err = lm.client.CoordinationV1().Leases(lm.namespace).Delete(ctx, LeaseName, metav1.DeleteOptions{
		Preconditions: &metav1.Preconditions{UID: &lease.UID, ResourceVersion: &lease.ResourceVersion},
	})
	if err != nil {
		// If lease is already deleted (TTL expiry, manual cleanup, etc.), that's fine
		// The goal is achieved - the lease is gone
		if k8sErrors.IsNotFound(err) {
			lm.logger.Info("Namespace scope marker lease already deleted")
			return nil
		}
		// Real failure - return the error
		lm.logger.Error(err, "Failed to delete lease on shutdown")
		return err
	}

	lm.logger.Info("Namespace scope marker lease deleted successfully")
	return nil
}

// createOrUpdateLease creates or updates the namespace scope marker lease
func (lm *LeaseManager) createOrUpdateLease(ctx context.Context) error {
	now := metav1.NewMicroTime(time.Now())

	// Kubernetes stores whole seconds; round up so sub-second configurations never
	// publish an already-expired lease, and account for this exact TTL in Guard.
	leaseDurationSeconds := int32(lm.leaseDuration / time.Second)
	if lm.leaseDuration%time.Second != 0 {
		leaseDurationSeconds++
	}

	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: lm.namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       &lm.holderIdentity,
			LeaseDurationSeconds: &leaseDurationSeconds,
			AcquireTime:          &now,
			RenewTime:            &now,
		},
	}

	// Try to get existing lease
	existingLease, err := lm.client.CoordinationV1().Leases(lm.namespace).Get(ctx, LeaseName, metav1.GetOptions{})
	if err != nil {
		if !k8sErrors.IsNotFound(err) {
			return fmt.Errorf("failed to get lease: %w", err)
		}
		// Lease doesn't exist, create it
		_, err = lm.client.CoordinationV1().Leases(lm.namespace).Create(ctx, lease, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to create lease: %w", err)
		}
		lm.recordRenewal(now.Time, leaseDurationSeconds)
		lm.logger.Info("Created namespace scope marker lease")
		return nil
	}

	// Lease exists, update it
	existingLease.Spec.HolderIdentity = &lm.holderIdentity
	existingLease.Spec.LeaseDurationSeconds = &leaseDurationSeconds
	existingLease.Spec.RenewTime = &now

	_, err = lm.client.CoordinationV1().Leases(lm.namespace).Update(ctx, existingLease, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update lease: %w", err)
	}

	lm.recordRenewal(now.Time, leaseDurationSeconds)
	lm.logger.V(1).Info("Refreshed namespace scope marker lease")
	return nil
}

// renewalLoop continuously renews the lease until stopped
func (lm *LeaseManager) renewalLoop(ctx context.Context) {
	defer lm.wg.Done()

	ticker := time.NewTicker(lm.renewInterval)
	defer ticker.Stop()

	for {
		select {
		case <-lm.stopCh:
			lm.logger.Info("Lease renewal loop stopped")
			return
		case <-ctx.Done():
			lm.logger.Info("Context cancelled, stopping lease renewal loop")
			return
		case <-ticker.C:
			// Use createOrUpdateLease instead of renewLease for self-healing
			// If the lease is manually deleted, it will be automatically recreated
			if err := lm.createOrUpdateLease(ctx); err != nil {
				lm.failureCount++
				lm.logger.Error(err, "Failed to create/update lease, will retry",
					"failureCount", lm.failureCount,
					"maxFailures", lm.maxFailures,
					"nextRetry", lm.renewInterval)

				// Warn when approaching max failures
				if lm.failureCount == lm.maxFailures-1 {
					lm.logger.Error(nil, "WARNING: One more lease renewal failure will cause operator shutdown to prevent split-brain",
						"failureCount", lm.failureCount,
						"maxFailures", lm.maxFailures)
				}

				// After max consecutive failures, signal fatal error to prevent split-brain
				if lm.failureCount >= lm.maxFailures {
					fatalErr := fmt.Errorf("lease renewal failed %d consecutive times (max: %d), operator must exit to prevent split-brain with cluster-wide operator", lm.failureCount, lm.maxFailures)
					lm.logger.Error(fatalErr, "FATAL: Max lease renewal failures exceeded")

					// Send error to channel (non-blocking)
					select {
					case lm.errCh <- fatalErr:
					default:
						// Error already sent, don't block
					}
					return
				}
			} else {
				// Success: reset failure counter
				if lm.failureCount > 0 {
					lm.logger.Info("Lease renewal recovered after failures",
						"previousFailures", lm.failureCount)
					lm.failureCount = 0
				}
			}
		}
	}
}

// recordRenewal records the expiry encoded in the successful request, including time
// spent waiting for its response. The notification coalesces updates; the mutex holds the latest.
func (lm *LeaseManager) recordRenewal(renewedAt time.Time, durationSeconds int32) {
	lm.expiryMu.Lock()
	lm.expiresAt = renewedAt.Add(time.Duration(durationSeconds) * time.Second)
	lm.expiryMu.Unlock()

	// Calls made without Start have no watcher; a nil channel simply takes default.
	select {
	case lm.leaseUpdated <- struct{}{}:
	default:
	}
}

// leaseDeadlines reserves two safety margins before the confirmed lease expires:
// stop waiting for renewal first, then finish work and cleanup before the final margin.
func (lm *LeaseManager) leaseDeadlines() (renewal, shutdown time.Time) {
	lm.expiryMu.Lock()
	defer lm.expiryMu.Unlock()
	margin := min(time.Second, lm.leaseDuration/10)
	return lm.expiresAt.Add(-2 * margin), lm.expiresAt.Add(-margin)
}
