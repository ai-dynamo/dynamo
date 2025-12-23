/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/go-logr/logr"
	coordinationv1 "k8s.io/api/coordination/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	k8sCache "k8s.io/client-go/tools/cache"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// LeaseWatcher watches for namespace scope marker leases and maintains
// an exclusion list for the cluster-wide operator.
// It implements the ExcludedNamespacesInterface from controller_common.
type LeaseWatcher struct {
	excludedNamespaces sync.Map // map[string]*coordinationv1.Lease (namespace -> lease object)
	informerFactory    informers.SharedInformerFactory
	logger             logr.Logger
}

// NewLeaseWatcher creates a new lease watcher for cluster-wide operator
func NewLeaseWatcher(config *rest.Config) (*LeaseWatcher, error) {
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	// Create informer factory for all namespaces
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	return &LeaseWatcher{
		informerFactory: informerFactory,
	}, nil
}

// Contains checks if a namespace is in the exclusion list AND if its lease is still valid.
// This method implements the ExcludedNamespacesInterface.
// It automatically removes expired leases, preventing stale leases from blocking reconciliation.
func (lw *LeaseWatcher) Contains(namespace string) bool {
	value, exists := lw.excludedNamespaces.Load(namespace)
	if !exists {
		return false
	}

	lease, ok := value.(*coordinationv1.Lease)
	if !ok {
		// Should never happen, but clean up if it does
		lw.logger.Error(nil, "Invalid lease object type in exclusion map",
			"namespace", namespace,
			"type", fmt.Sprintf("%T", value))
		lw.excludedNamespaces.Delete(namespace)
		return false
	}

	// Check if lease has expired (critical for handling stale leases after crashes)
	if lw.isLeaseExpired(lease) {
		lw.logger.Info("Lease expired during Contains check, resuming cluster-wide processing",
			"namespace", namespace,
			"renewTime", lease.Spec.RenewTime,
			"leaseDuration", lease.Spec.LeaseDurationSeconds)
		lw.removeExcludedNamespace(namespace)
		return false
	}

	return true
}

// Start starts watching for namespace scope marker leases
func (lw *LeaseWatcher) Start(ctx context.Context) error {
	lw.logger = log.FromContext(ctx).WithValues("component", "namespace-scope-lease-watcher")

	lw.logger.Info("Starting namespace scope marker lease watcher")

	// Get the lease informer
	leaseInformer := lw.informerFactory.Coordination().V1().Leases().Informer()

	// Add event handler for namespace scope marker leases
	_, err := leaseInformer.AddEventHandler(k8sCache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			if lease := lw.extractLease(obj); lease != nil {
				lw.handleLeaseAdd(lease)
			}
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			if lease := lw.extractLease(newObj); lease != nil {
				lw.handleLeaseUpdate(lease)
			}
		},
		DeleteFunc: func(obj interface{}) {
			if lease := lw.extractLease(obj); lease != nil {
				lw.handleLeaseDelete(lease)
			}
		},
	})
	if err != nil {
		return err
	}

	// Start informers
	lw.informerFactory.Start(ctx.Done())

	// Wait for cache sync
	if !k8sCache.WaitForCacheSync(ctx.Done(), leaseInformer.HasSynced) {
		err := errors.New("failed to sync lease informer cache")
		lw.logger.Error(err, "Lease watcher cache sync failed")
		return err
	}

	lw.logger.Info("Namespace scope marker lease watcher started and cache synced")
	return nil
}

// extractLease safely extracts a Lease from an event object, handling tombstones.
// Returns nil if the object is not a valid Lease.
func (lw *LeaseWatcher) extractLease(obj any) *coordinationv1.Lease {
	// Handle DeletedFinalStateUnknown tombstones
	if tombstone, ok := obj.(k8sCache.DeletedFinalStateUnknown); ok {
		obj = tombstone.Obj
	}

	// Type assert to Lease
	lease, ok := obj.(*coordinationv1.Lease)
	if !ok {
		lw.logger.V(1).Info("Received non-Lease object in lease watcher, ignoring",
			"objectType", fmt.Sprintf("%T", obj))
		return nil
	}

	return lease
}

// handleLeaseAdd handles lease creation events
func (lw *LeaseWatcher) handleLeaseAdd(lease *coordinationv1.Lease) {
	// Only process namespace scope marker leases
	if !lw.isNamespaceScopeMarker(lease) {
		return
	}

	// Don't add if already expired (defensive - shouldn't happen in practice)
	if lw.isLeaseExpired(lease) {
		lw.logger.V(1).Info("Ignoring already-expired lease on add",
			"namespace", lease.Namespace)
		return
	}

	lw.addExcludedNamespace(lease)
}

// handleLeaseUpdate handles lease update events (renewals)
func (lw *LeaseWatcher) handleLeaseUpdate(lease *coordinationv1.Lease) {
	// Only process namespace scope marker leases
	if !lw.isNamespaceScopeMarker(lease) {
		return
	}

	// If lease expired, remove from exclusion list
	// This handles the critical case where namespace-scoped operator crashes
	// without deleting its lease - we detect expiry and resume processing
	if lw.isLeaseExpired(lease) {
		lw.logger.Info("Lease expired on update, resuming cluster-wide processing",
			"namespace", lease.Namespace,
			"renewTime", lease.Spec.RenewTime,
			"leaseDuration", lease.Spec.LeaseDurationSeconds)
		lw.removeExcludedNamespace(lease.Namespace)
		return
	}

	// Lease still valid - update with fresh lease object (refreshes RenewTime)
	lw.addExcludedNamespace(lease)
}

// handleLeaseDelete handles lease deletion/expiration events
func (lw *LeaseWatcher) handleLeaseDelete(lease *coordinationv1.Lease) {
	// Only process namespace scope marker leases
	if !lw.isNamespaceScopeMarker(lease) {
		return
	}

	lw.removeExcludedNamespace(lease.Namespace)
}

// addExcludedNamespace adds a namespace to the exclusion list
func (lw *LeaseWatcher) addExcludedNamespace(lease *coordinationv1.Lease) {
	holderIdentity := ""
	if lease.Spec.HolderIdentity != nil {
		holderIdentity = *lease.Spec.HolderIdentity
	}

	// Store the full lease object so Contains() can check TTL on every access
	lw.excludedNamespaces.Store(lease.Namespace, lease)
	lw.logger.Info("Excluding namespace from cluster-wide operator processing",
		"namespace", lease.Namespace,
		"holderIdentity", holderIdentity)
}

// removeExcludedNamespace removes a namespace from the exclusion list
func (lw *LeaseWatcher) removeExcludedNamespace(namespace string) {
	lw.excludedNamespaces.Delete(namespace)
	lw.logger.Info("Resuming namespace processing in cluster-wide operator",
		"namespace", namespace,
		"reason", "namespace-restricted operator lease expired or deleted")
}

// isNamespaceScopeMarker checks if a lease is a namespace scope marker
func (lw *LeaseWatcher) isNamespaceScopeMarker(lease *coordinationv1.Lease) bool {
	// A lease is a namespace scope marker if it has the well-known name
	// Labels are added for observability/filtering but not required for identification
	return lease.Name == LeaseName
}

// isLeaseExpired checks if a lease has exceeded its TTL.
// This is critical for handling stale leases when the namespace-scoped operator
// crashes without gracefully deleting its lease.
func (lw *LeaseWatcher) isLeaseExpired(lease *coordinationv1.Lease) bool {
	if lease.Spec.RenewTime == nil || lease.Spec.LeaseDurationSeconds == nil {
		// Missing required fields - treat as expired for safety
		lw.logger.V(1).Info("Lease missing RenewTime or LeaseDurationSeconds, treating as expired",
			"namespace", lease.Namespace,
			"hasRenewTime", lease.Spec.RenewTime != nil,
			"hasLeaseDuration", lease.Spec.LeaseDurationSeconds != nil)
		return true
	}

	expiryTime := lease.Spec.RenewTime.Add(
		time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second,
	)
	return time.Now().After(expiryTime)
}
