/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"fmt"

	ctrl "sigs.k8s.io/controller-runtime"
)

// SetupOptions selects the runtime provider registered with the Engine Group controller.
type SetupOptions struct {
	RuntimeProvider RuntimeProvider
	GroveEnabled    bool
}

// Setup registers the Engine Group subsystem with the operator manager.
func Setup(mgr ctrl.Manager, opts SetupOptions) error {
	// Resolve the production provider unless the caller supplies another implementation.
	provider := opts.RuntimeProvider
	if provider == nil {
		provider = newEngineGroupRuntimeProvider(mgr.GetClient())
	}

	// Register reconciliation and watches together at the subsystem composition root.
	if err := (&Reconciler{
		Client:          mgr.GetClient(),
		RuntimeProvider: provider,
	}).SetupWithManager(mgr, opts.GroveEnabled); err != nil {
		return fmt.Errorf("unable to create DynamoGraphDeploymentEngineGroup controller: %w", err)
	}
	return nil
}
