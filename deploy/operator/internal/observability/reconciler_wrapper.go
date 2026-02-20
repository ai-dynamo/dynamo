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

package observability

import (
	"context"
	"time"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// ObservedReconciler wraps any reconciler and automatically records metrics
// for reconciliation duration, results, and errors.
type ObservedReconciler struct {
	reconcile.Reconciler
	resourceType string
}

// NewObservedReconciler creates a new ObservedReconciler wrapper
func NewObservedReconciler(r reconcile.Reconciler, resourceType string) *ObservedReconciler {
	return &ObservedReconciler{
		Reconciler:   r,
		resourceType: resourceType,
	}
}

// Reconcile wraps the underlying reconciler's Reconcile method with metrics collection
func (m *ObservedReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	startTime := time.Now()
	result, err := m.Reconciler.Reconcile(ctx, req)
	duration := time.Since(startTime)

	// Determine if a requeue is happening
	//nolint:staticcheck // SA1019: result.Requeue is deprecated but still supported by controller-runtime
	requeue := result.Requeue || result.RequeueAfter > 0

	// Record reconciliation metrics
	RecordReconciliation(
		m.resourceType,
		req.Namespace,
		err,
		requeue,
		duration,
	)

	return result, err
}
