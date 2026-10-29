// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"context"
	"errors"

	"k8s.io/client-go/rest"
)

// MultiClusterExport exports the PoolRelay and frontend Services of each DGD
// that is a Global Router pool through the Multi-Cluster Services (MCS) API.
//
// Owner: @atchernych
// Experimental since: v1.7.0
// Beta since: N/A
// GA since: N/A
// Configuration: multiCluster.mcsImplementation, and a DynamoPoolRelay named
// after each pool's DGD
// Auto-detection: N/A; API availability is verified when an implementation is configured
// Requires: an MCS implementation, such as Karmada, serving
// multicluster.x-k8s.io/v1alpha1 or multicluster.x-k8s.io/v1beta1 ServiceExport resources
// Default: false
const MultiClusterExport Name = "multiClusterExport"

// serviceExportVersions lists the multicluster.x-k8s.io versions that the
// operator reads and writes ServiceExports in, preferred first. The SIG
// Multicluster CRDs serve both and define the metadata the operator writes
// alike; v1beta1 is used where the cluster no longer serves v1alpha1.
var serviceExportVersions = [...]string{"v1alpha1", "v1beta1"}

// resolveMultiClusterExport verifies API availability for a configured MCS
// implementation and returns the first version in serviceExportVersions at
// which the cluster serves ServiceExports, or an error when it serves them at
// neither. A nil config retains the native discovery-unavailable error.
func resolveMultiClusterExport(ctx context.Context, config *rest.Config) (string, error) {
	// Every supported implementation reads the SIG Multicluster ServiceExport.
	for _, version := range serviceExportVersions {
		available, err := detectAPIAvailability(ctx, config, "multicluster.x-k8s.io", version, "serviceexports")
		if err != nil {
			return "", err
		}
		if available {
			return version, nil
		}
	}

	return "", errors.New("multiCluster.mcsImplementation is set but neither the multicluster.x-k8s.io/v1alpha1 " +
		"nor the multicluster.x-k8s.io/v1beta1 ServiceExport API was detected in the cluster")
}
