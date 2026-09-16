// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestDynamoSidecarConversion(t *testing.T) {
	for _, kind := range []string{"DGD", "DCD"} {
		for _, name := range []*string{nil, ptr.To("dynamo"), ptr.To("")} {
			t.Run(kind+"/"+ptr.Deref(name, "absent"), func(t *testing.T) {
				t.Log("Construct a beta component with native runtime and independent engine configuration")
				component := v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, DynamoSidecar: name,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers:     []corev1.Container{{Name: "main", Image: "engine:latest"}},
						InitContainers: []corev1.Container{{Name: "dynamo", Image: "runtime:1.5.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)}},
					}},
				}
				original := component.DeepCopy()
				var restored v1beta1.DynamoComponentDeploymentSharedSpec

				t.Log("Round trip through alpha and retain a live init-container image edit")
				if kind == "DGD" {
					hub := &v1beta1.DynamoGraphDeployment{Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}}}
					spoke := &DynamoGraphDeployment{}
					require.NoError(t, spoke.ConvertFrom(hub))
					spoke.Spec.Services["worker"].ExtraPodSpec.PodSpec.InitContainers[0].Image = "runtime:1.6.0"
					out := &v1beta1.DynamoGraphDeployment{}
					require.NoError(t, spoke.ConvertTo(out))
					restored = out.Spec.Components[0]
					require.Equal(t, *original, hub.Spec.Components[0])

					t.Log("Removing a component in alpha must not resurrect its beta-only selector")
					delete(spoke.Spec.Services, "worker")
					out = &v1beta1.DynamoGraphDeployment{}
					require.NoError(t, spoke.ConvertTo(out))
					require.Empty(t, out.Spec.Components)
				} else {
					hub := &v1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{Name: "worker"}, Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: component}}
					spoke := &DynamoComponentDeployment{}
					require.NoError(t, spoke.ConvertFrom(hub))
					spoke.Spec.ExtraPodSpec.PodSpec.InitContainers[0].Image = "runtime:1.6.0"
					out := &v1beta1.DynamoComponentDeployment{}
					require.NoError(t, spoke.ConvertTo(out))
					restored = out.Spec.DynamoComponentDeploymentSharedSpec
					require.Equal(t, *original, hub.Spec.DynamoComponentDeploymentSharedSpec)
				}
				require.Equal(t, name, restored.DynamoSidecar)
				require.Equal(t, "runtime:1.6.0", restored.PodTemplate.Spec.InitContainers[0].Image)

				t.Log("Beta rename and removal must survive subsequent alpha round trips")
				for _, replacement := range []*string{ptr.To("renamed"), nil} {
					restored.DynamoSidecar = replacement
					hub := &v1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{Name: "worker"}, Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: restored}}
					require.Equal(t, replacement, dcdRoundTripFromV1beta1(t, hub).Spec.DynamoSidecar)
				}
			})
		}
	}
}
