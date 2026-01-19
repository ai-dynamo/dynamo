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
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// LeaseName is the well-known name for namespace scope marker leases
	LeaseName = "dynamo-operator-namespace-scope"
)

// LeaseManager manages the namespace scope marker lease
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

	// Create holder identity with operator version
	// No need for pod name since there's only one operator instance in namespace-restricted mode
	holderIdentity := fmt.Sprintf("namespace-restricted-operator-%s", operatorVersion)

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

	lm.logger.Info("Starting namespace scope marker lease manager",
		"leaseName", LeaseName,
		"leaseDuration", lm.leaseDuration,
		"renewInterval", lm.renewInterval,
		"holderIdentity", lm.holderIdentity,
		"maxFailures", lm.maxFailures)

	// Create or update the lease initially
	if err := lm.createOrUpdateLease(ctx); err != nil {
		return fmt.Errorf("failed to create initial lease: %w", err)
	}

	lm.logger.Info("Namespace scope marker lease created successfully")

	// Start renewal loop in background
	lm.wg.Add(1)
	go lm.renewalLoop(ctx)

	return nil
}

// Stop stops the lease renewal loop and releases the lease
func (lm *LeaseManager) Stop(ctx context.Context) error {
	lm.logger.Info("Stopping namespace scope marker lease manager")

	// Signal renewal loop to stop
	close(lm.stopCh)

	// Wait for renewal loop to complete to avoid race condition
	// where we delete the lease while a renewal is in progress
	lm.wg.Wait()

	// Delete the lease to signal we're no longer managing this namespace
	err := lm.client.CoordinationV1().Leases(lm.namespace).Delete(ctx, LeaseName, metav1.DeleteOptions{})
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
	leaseDurationSeconds := int32(lm.leaseDuration.Seconds())

	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      LeaseName,
			Namespace: lm.namespace,
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       &lm.holderIdentity,
			LeaseDurationSeconds: &leaseDurationSeconds,
			AcquireTime:          &now,
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
