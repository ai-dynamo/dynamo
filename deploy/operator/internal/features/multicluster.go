// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"context"

	"k8s.io/client-go/rest"
)

// MultiClusterExport exports the PoolRelay and frontend Services of each DGD
// marked as a Global Router pool through the Multi-Cluster Services (MCS) API.
//
// Owner: @atchernych
// Experimental since: v1.7.0
// Beta since: N/A
// GA since: N/A
// Configuration: multiCluster.mcsImplementation, and the
// nvidia.com/global-router-pool annotation on each DGD
// Auto-detection: N/A; API availability is verified when an implementation is configured
// Requires: an MCS implementation, such as Karmada, serving
// multicluster.x-k8s.io/v1alpha1 ServiceExport resources
// Default: false
const MultiClusterExport Name = "multiClusterExport"

// resolveMultiClusterExport verifies API availability for a configured MCS
// implementation. A nil config retains the native discovery-unavailable error.
func resolveMultiClusterExport(ctx context.Context, config *rest.Config) (bool, error) {
	// Every supported implementation reads the SIG Multicluster ServiceExport.
	available, err := detectAPIAvailability(ctx, config, "multicluster.x-k8s.io", "v1alpha1", "serviceexports")
	if err != nil {
		return false, err
	}

	return resolve(new(true), available,
		"multiCluster.mcsImplementation is set but the multicluster.x-k8s.io/v1alpha1 ServiceExport API was not detected in the cluster")
}
