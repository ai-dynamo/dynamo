// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package shared renders resources common to component workload lowerings.
package shared

import (
	"maps"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// ServiceParams supplies resolved component identity and authored Service metadata.
type ServiceParams struct {
	ServiceName     string
	Namespace       string
	DGDName         string // empty for standalone components
	ComponentName   string
	ComponentType   string
	DynamoNamespace string
	Labels          map[string]string
	Annotations     map[string]string
	IsK8sDiscovery  bool
}

// RenderService returns a base Service for a lowering to extend with ports and other spec fields.
// The returned metadata and selector maps do not alias the inputs.
func RenderService(params ServiceParams) *corev1.Service {
	// Unresolved optional identity must preserve authored metadata rather than erase it.
	labels := make(map[string]string, len(params.Labels)+4)
	maps.Copy(labels, params.Labels)
	labels[consts.KubeLabelDynamoComponent] = params.ComponentName
	labels[consts.KubeLabelDynamoNamespace] = params.DynamoNamespace
	if params.ComponentType != "" {
		labels[consts.KubeLabelDynamoComponentType] = params.ComponentType
	}
	if params.DGDName != "" {
		labels[consts.KubeLabelDynamoGraphDeploymentName] = params.DGDName
	}
	if params.IsK8sDiscovery {
		labels[consts.KubeLabelDynamoDiscoveryBackend] = consts.DiscoveryBackendKubernetes
		labels[consts.KubeLabelDynamoDiscoveryEnabled] = consts.KubeLabelValueTrue
	}

	// Keep authored annotations independent so lowerings can extend the rendered object.
	annotations := make(map[string]string, len(params.Annotations))
	maps.Copy(annotations, params.Annotations)

	return &corev1.Service{
		// Service names must be DNS-1035 labels; model names can contain dots and capitals.
		Name:        strings.ToLower(strings.ReplaceAll(params.ServiceName, ".", "-")),
		Namespace:   params.Namespace,
		Labels:      labels,
		Annotations: annotations,
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				consts.KubeLabelDynamoComponent:     params.ComponentName,
				consts.KubeLabelDynamoComponentType: params.ComponentType,
				consts.KubeLabelDynamoNamespace:     params.DynamoNamespace,
			},
		},
	}
}
