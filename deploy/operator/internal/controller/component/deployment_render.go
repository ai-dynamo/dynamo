// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package component

import (
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Render constructs the desired Deployment from a resolved Pod template and update strategy.
// dcd and template must be non-nil. Inputs are not modified.
func Render(dcd *v1beta1.DynamoComponentDeployment, metadata metav1.ObjectMeta, template *corev1.PodTemplateSpec, strategy appsv1.DeploymentStrategy) *appsv1.Deployment {
	metadata = *metadata.DeepCopy()
	metadata.Name = dcd.Name
	metadata.Namespace = dcd.Namespace
	template = template.DeepCopy()
	if template.Labels == nil {
		template.Labels = make(map[string]string)
	}
	template.Labels[consts.KubeLabelDynamoSelector] = dcd.Name

	// Own the replica value while preserving Kubernetes defaulting when unspecified.
	replicas := dcd.Spec.Replicas
	if replicas != nil {
		replicas = new(*replicas)
	}
	return &appsv1.Deployment{
		ObjectMeta: metadata,
		Spec: appsv1.DeploymentSpec{
			Replicas: replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{consts.KubeLabelDynamoSelector: dcd.Name}},
			Template: *template,
			Strategy: *strategy.DeepCopy(),
		},
	}
}
