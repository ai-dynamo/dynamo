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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	AnnotationDGDLegacyWorkerHash = "nvidia.com/dgd-legacy-worker-hash"
	annCurrentWorkerHash          = "nvidia.com/current-worker-hash"
)

// GetDGDLegacyWorkerHash returns the preserved v1alpha1 worker hash carried on
// converted v1beta1 DGD objects.
func GetDGDLegacyWorkerHash(obj metav1.Object) string {
	if obj == nil {
		return ""
	}
	return obj.GetAnnotations()[AnnotationDGDLegacyWorkerHash]
}

// ClearDGDLegacyWorkerHash removes the preserved v1alpha1 worker hash from a
// DGD object after the v1beta1 controller has consumed it.
func ClearDGDLegacyWorkerHash(obj metav1.Object) {
	if obj == nil || obj.GetAnnotations() == nil {
		return
	}
	annotations := obj.GetAnnotations()
	delete(annotations, AnnotationDGDLegacyWorkerHash)
	obj.SetAnnotations(annotations)
}

// ComputeDGDWorkersSpecHash computes the worker hash used by the
// v1alpha1 DGD controller before v1beta1 conversion. ConvertTo stores this
// value on the v1beta1 hub object so the v1beta1 controller can migrate
// existing v1alpha1 hashes without rolling workers.
//
// Keep this in the api/v1alpha1 package so conversion can call it without an
// api/v1alpha1 -> internal/dynamo -> api/v1alpha1 import cycle.
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
