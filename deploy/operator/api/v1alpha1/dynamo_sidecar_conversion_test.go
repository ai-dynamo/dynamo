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
		for _, mutation := range []string{"image", "rename", "remove"} {
			t.Run(kind+"/"+mutation, func(t *testing.T) {
				t.Log("Construct a beta component whose runtime init container selects sidecar mode")
				component := v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers:     []corev1.Container{{Name: "main", Image: "engine:latest"}},
						InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.5.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)}},
					}},
				}
				original := component.DeepCopy()
				hubDGD := &v1beta1.DynamoGraphDeployment{Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}}}
				hubDCD := &v1beta1.DynamoComponentDeployment{ObjectMeta: metav1.ObjectMeta{Name: "worker"}, Spec: v1beta1.DynamoComponentDeploymentSpec{DynamoComponentDeploymentSharedSpec: component}}
				spokeDGD := &DynamoGraphDeployment{}
				spokeDCD := &DynamoComponentDeployment{}
				var alpha *DynamoComponentDeploymentSharedSpec
				if kind == "DGD" {
					require.NoError(t, spokeDGD.ConvertFrom(hubDGD))
					alpha = spokeDGD.Spec.Services["worker"]
				} else {
					require.NoError(t, spokeDCD.ConvertFrom(hubDCD))
					alpha = &spokeDCD.Spec.DynamoComponentDeploymentSharedSpec
				}

				t.Log("Change live alpha init containers without clearing conversion annotations")
				switch mutation {
				case "image":
					alpha.ExtraPodSpec.PodSpec.InitContainers[0].Image = "runtime:1.6.0"
				case "rename":
					alpha.ExtraPodSpec.PodSpec.InitContainers[0].Name = "setup"
				case "remove":
					alpha.ExtraPodSpec.PodSpec.InitContainers = nil
				}

				t.Log("Convert back and retain the live mode change without mutating the original")
				var restored v1beta1.DynamoComponentDeploymentSharedSpec
				if kind == "DGD" {
					out := &v1beta1.DynamoGraphDeployment{}
					require.NoError(t, spokeDGD.ConvertTo(out))
					restored = out.Spec.Components[0]
					require.Equal(t, *original, hubDGD.Spec.Components[0])
				} else {
					out := &v1beta1.DynamoComponentDeployment{}
					require.NoError(t, spokeDCD.ConvertTo(out))
					restored = out.Spec.DynamoComponentDeploymentSharedSpec
					require.Equal(t, *original, hubDCD.Spec.DynamoComponentDeploymentSharedSpec)
				}
				require.Equal(t, alpha.ExtraPodSpec.PodSpec.InitContainers, restored.PodTemplate.Spec.InitContainers)
				require.Equal(t, original.PodTemplate.Spec.Containers, restored.PodTemplate.Spec.Containers)
			})
		}
	}
}
