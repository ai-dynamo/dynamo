// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestWorkerHashOmitsFullyShadowedGeneralEnvironment(t *testing.T) {
	for _, roleTemplates := range []bool{false, true} {
		for _, runtimeConsumes := range []bool{false, true} {
			t.Run(fmt.Sprintf("roles=%t/runtime-consumes=%t", roleTemplates, runtimeConsumes), func(t *testing.T) {
				t.Log("Build semantic-origin workers whose main overrides the general definition")
				component := v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: "main", Image: "runtime:1.8.0", Command: []string{"python3", "-m", "dynamo.vllm"},
						Env: []corev1.EnvVar{{Name: "MODEL_ROOT", Value: "/local"}},
					}}}},
				}
				if runtimeConsumes {
					component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.8.0", RestartPolicy: new(corev1.ContainerRestartPolicyAlways)}}
				}
				if roleTemplates {
					component.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
					component.Roles = []v1beta1.ComponentRoleSpec{{Name: "leader", PodTemplate: component.PodTemplate.DeepCopy()}, {Name: "worker", PodTemplate: component.PodTemplate.DeepCopy()}}
					component.PodTemplate = nil
				}
				dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
					Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
					Spec:        v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Env: []corev1.EnvVar{{Name: "MODEL_ROOT", Value: "/general"}}, Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component}},
				}
				original := dgd.DeepCopy()
				before := mustComputeBetaDGDWorkersSpecHash(t, dgd)

				t.Log("Changing a hidden default must not create a worker generation")
				changed := dgd.DeepCopy()
				changed.Spec.Env[0].Value = "/different"
				after := mustComputeBetaDGDWorkersSpecHash(t, changed)
				if runtimeConsumes {
					require.NotEqual(t, before, after, "the runtime still consumes this default")
				} else {
					require.Equal(t, before, after)
				}
				require.Equal(t, original, dgd)

				t.Log("The persisted child still retains the complete general source list")
				children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
				require.NoError(t, err)
				require.Equal(t, original.Spec.Env, children["worker"].Spec.Env)
			})
		}
	}
}
