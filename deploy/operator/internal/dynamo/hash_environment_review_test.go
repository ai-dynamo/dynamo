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
		for _, test := range []struct {
			name             string
			runtime          bool
			runtimeOverrides bool
			regularHelper    bool
		}{
			{name: "main only"},
			{name: "runtime consumes", runtime: true},
			{name: "runtime overrides", runtime: true, runtimeOverrides: true},
			{name: "regular helper named runtime", regularHelper: true},
		} {
			t.Run(fmt.Sprintf("roles=%t/%s", roleTemplates, test.name), func(t *testing.T) {
				t.Log("Build semantic-origin workers whose main overrides the general definition")
				component := v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: "main", Image: "runtime:1.8.0", Command: []string{"python3", "-m", "dynamo.vllm"},
						Env: []corev1.EnvVar{{Name: "MODEL_ROOT", Value: "/local"}},
					}}}},
				}
				if test.runtime {
					component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.8.0", RestartPolicy: new(corev1.ContainerRestartPolicyAlways)}}
					if test.runtimeOverrides {
						component.PodTemplate.Spec.InitContainers[0].Env = []corev1.EnvVar{{Name: "MODEL_ROOT", Value: "/runtime"}}
					}
				}
				if test.regularHelper {
					component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{Name: "runtime", Image: "helper:1.8.0"})
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
				if test.runtime && !test.runtimeOverrides {
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

func TestWorkerHashKeepsDefaultsForGeneratedMain(t *testing.T) {
	for _, omittedTemplate := range []bool{false, true} {
		t.Run(fmt.Sprintf("omitted-template=%t", omittedTemplate), func(t *testing.T) {
			t.Log("An unconfigured main is generated later and still consumes general defaults")
			spec := &v1beta1.DynamoComponentDeploymentSpec{Env: []corev1.EnvVar{{Name: "MODEL_ROOT", Value: "/general"}}}
			if !omittedTemplate {
				spec.PodTemplate = &corev1.PodTemplateSpec{}
			}
			original := spec.DeepCopy()
			require.Equal(t, spec.Env, workerHashGeneralEnvironment(spec))
			require.Equal(t, original, spec)
		})
	}
}
