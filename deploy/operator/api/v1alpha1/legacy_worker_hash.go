/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// ComputeDGDWorkersSpecHash computes the worker hash used by the
// v1alpha1 DGD controller. Controller reconciliation may use this to compare
// v1/v2 worker generations, but API conversion must not derive controller
// rollout state from it.
//
// Keep this in the api/v1alpha1 package so internal controller helpers can
// reproduce the legacy hash without duplicating the old algorithm.
func ComputeDGDWorkersSpecHash(dgd *DynamoGraphDeployment) (string, error) {
	if dgd == nil {
		return "", fmt.Errorf("nil DynamoGraphDeployment")
	}

	var workerNames []string
	for name, spec := range dgd.Spec.Services {
		if spec != nil && isV1alpha1WorkerComponent(spec.ComponentType) {
			workerNames = append(workerNames, name)
		}
	}
	sort.Strings(workerNames)

	hashInputs := make(map[string]DynamoComponentDeploymentSharedSpec)
	for _, name := range workerNames {
		hashInputs[name] = stripV1alpha1NonPodTemplateFields(dgd.Spec.Services[name])
	}

	data, err := json.Marshal(hashInputs)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:8], nil
}

func isV1alpha1WorkerComponent(componentType string) bool {
	return componentType == "worker" || componentType == "prefill" || componentType == "decode"
}

func stripV1alpha1NonPodTemplateFields(spec *DynamoComponentDeploymentSharedSpec) DynamoComponentDeploymentSharedSpec {
	stripped := *spec

	stripped.Annotations = nil
	stripped.Labels = nil
	stripped.ServiceName = ""
	stripped.ComponentType = ""
	stripped.SubComponentType = ""
	stripped.DynamoNamespace = nil
	stripped.Replicas = nil
	stripped.Autoscaling = nil //nolint:staticcheck // SA1019: intentionally matching the old v1alpha1 worker hash
	stripped.ScalingAdapter = nil
	stripped.Ingress = nil
	stripped.ModelRef = nil
	stripped.EPPConfig = nil

	return stripped
}
