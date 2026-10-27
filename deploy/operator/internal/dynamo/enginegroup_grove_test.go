// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGroveEngineGroupWorldRendering(t *testing.T) {
	for _, backend := range []BackendFramework{BackendFrameworkSGLang, BackendFrameworkVLLM} {
		t.Run(string(backend), func(t *testing.T) {
			t.Log("Declare a user-owned launch template and resolved allocation geometry")
			component := &api.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: api.ComponentTypeWorker, Replicas: new(int32(3)),
				EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2},
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"team": "serving"}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: "main", Image: "runtime:1.6.0", Command: []string{"python3", "-m", "dynamo." + string(backend)},
						Args: []string{"--model-path", "model"},
						Env:  []corev1.EnvVar{{Name: "CUSTOM_LEADER", Value: "$(DYNAMO_LEADER_ADDRESS)"}},
						Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
							consts.KubeResourceGPUNvidia: resource.MustParse("4"),
						}},
						ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"check-allocation"}}}},
					}}},
				},
			}
			dgd := &api.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "test"},
				Spec: api.DynamoGraphDeploymentSpec{BackendFramework: string(backend),
					Labels: map[string]string{"parent-label": "retained", "team": "parent"}},
			}
			originalComponent, originalDGD := component.DeepCopy(), dgd.DeepCopy()
			geometry := enginegroup.ResolvedProfileGeometry{Backend: string(backend), GPUsPerReplica: 4, PodsPerReplica: 1}
			renderer := groveEngineGroupRenderer{
				operatorConfig: &config.OperatorConfiguration{}, runtimeConfig: &controller_common.RuntimeConfig{},
				reader: fake.NewClientBuilder().Build(), secretsRetriever: &mockSecretsRetriever{},
			}

			t.Log("Render one member clique without changing the declared command or probes")
			clique, world, err := renderer.renderWorld(t.Context(), dgd, component, geometry, 2, "", nil, nil, nil)
			require.NoError(t, err)
			main := clique.Spec.PodSpec.Containers[0]
			declared := component.PodTemplate.Spec.Containers[0]
			require.Equal(t, declared.Command, main.Command)
			require.Equal(t, declared.Args, main.Args)
			require.Equal(t, declared.ReadinessProbe, main.ReadinessProbe)
			require.Nil(t, main.LivenessProbe)
			require.Nil(t, main.StartupProbe)
			require.Equal(t, resource.MustParse("4"), main.Resources.Limits[consts.KubeResourceGPUNvidia])
			require.Equal(t, []corev1.EnvVar{
				{Name: consts.DynamoRankEnvVar, Value: "$(" + groveconstants.EnvVarPodIndex + ")"},
				{Name: consts.DynamoLeaderAddressEnvVar, Value: "$(" + groveconstants.EnvVarPodCliqueName + ")-0.$(" + groveconstants.EnvVarHeadlessService + ")"},
			}, main.Env[:2])

			t.Log("Keep world count, allocation count and metadata ownership distinct")
			require.Equal(t, int32(2), clique.Spec.Replicas)
			require.Equal(t, new(int32(2)), clique.Spec.MinAvailable)
			require.Equal(t, new(int32(3)), world.Replicas)
			require.Equal(t, []string{clique.Name}, world.CliqueNames)
			require.Equal(t, EngineGroupNameForComponent(dgd.Name, component.ComponentName, 0), clique.Labels[consts.KubeLabelDynamoEngineGroup])
			require.Equal(t, consts.KubeLabelDynamoScaleRepresentativeYes, clique.Labels[consts.KubeLabelDynamoScaleRepresentative])
			require.Equal(t, "serving", clique.Labels["team"])
			require.Equal(t, "retained", clique.Labels["parent-label"])
			require.Equal(t, originalComponent, component)
			require.Equal(t, originalDGD, dgd)
		})
	}
}
