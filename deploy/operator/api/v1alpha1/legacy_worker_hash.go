/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
)

const (
	AnnotationDGDLegacyWorkerHash = "nvidia.com/dgd-legacy-worker-hash"
	annCurrentWorkerHash          = "nvidia.com/current-worker-hash"
)

func computeLegacyDGDWorkersSpecHash(dgd *DynamoGraphDeployment) (string, error) {
	var workerNames []string
	for name, spec := range dgd.Spec.Services {
		if spec != nil && isLegacyWorkerComponent(spec.ComponentType) {
			workerNames = append(workerNames, name)
		}
	}
	sort.Strings(workerNames)

	hashInputs := make(map[string]DynamoComponentDeploymentSharedSpec)
	for _, name := range workerNames {
		hashInputs[name] = stripLegacyNonPodTemplateFields(dgd.Spec.Services[name])
	}

	data, err := json.Marshal(hashInputs)
	if err != nil {
		return "", err
	}

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:8], nil
}

func isLegacyWorkerComponent(componentType string) bool {
	return componentType == "worker" || componentType == "prefill" || componentType == "decode"
}

func stripLegacyNonPodTemplateFields(spec *DynamoComponentDeploymentSharedSpec) DynamoComponentDeploymentSharedSpec {
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
