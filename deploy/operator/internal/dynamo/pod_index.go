// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ComponentPodIndex indexes Pods by their DGD name and component name labels.
const ComponentPodIndex = ".metadata.dgdComponent"

// ComponentPodIndexValues extracts the ComponentPodIndex value of a Pod.
func ComponentPodIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	labels := pod.GetLabels()
	dgdName := labels[consts.KubeLabelDynamoGraphDeploymentName]
	componentName := labels[consts.KubeLabelDynamoComponent]
	if dgdName == "" || componentName == "" {
		return nil
	}
	return []string{ComponentPodIndexValue(dgdName, componentName)}
}

// ComponentPodIndexValue is the ComponentPodIndex key of one DGD component.
func ComponentPodIndexValue(dgdName, componentName string) string {
	// Both inputs are label values, which cannot contain '/', so this encoding
	// is collision-free.
	return dgdName + "/" + componentName
}
