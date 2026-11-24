// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDisaggregatedSetPolicyEnvironmentMatchesDCD(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", ""} {
		for _, runtimeLayout := range []string{"main", "sidecars", "leader-sidecar"} {
			t.Run(fmt.Sprintf("origin=%s/runtime=%s", origin, runtimeLayout), func(t *testing.T) {
				t.Log("Normalize a multinode worker whose graph env conflicts with an absent policy setting")
				template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model=/models"},
				}}}}
				component := v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
					Replicas: new(int32(1)), Multinode: &v1beta1.MultinodeSpec{NodeCount: 2}, PodTemplate: template,
				}
				if runtimeLayout != "main" {
					restart := corev1.ContainerRestartPolicyAlways
					template.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.7.0", RestartPolicy: &restart}}
					component.PodTemplate = nil
					component.Roles = []v1beta1.ComponentRoleSpec{
						{Name: "leader", PodTemplate: template.DeepCopy()},
						{Name: "worker", PodTemplate: template.DeepCopy()},
					}
					if runtimeLayout == "leader-sidecar" {
						component.Roles[1].PodTemplate.Spec.InitContainers = nil
					}
				}
				dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", UID: "graph-uid",
					Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
						Env: []corev1.EnvVar{{Name: consts.EnvKvTransferPreferredWeight, Value: "0.75"}, {Name: "GLOBAL", Value: "retained"}},
						Experimental: &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{
							LabelKey: "topology.kubernetes.io/zone", Domain: "zone", Enforcement: v1beta1.KvTransferEnforcementRequired,
						}},
					},
				}
				if origin != "" {
					dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin}
				}
				normalized, err := dynamo.NormalizeDynamoGraphDeploymentComponents(dgd, nil, nil, dynamo.RollingUpdateContext{})
				require.NoError(t, err)
				children, err := dynamo.GenerateDynamoComponentsDeploymentsFromNormalized(dgd, normalized, dynamo.RollingUpdateContext{})
				require.NoError(t, err)
				child := children["worker"]
				child.OwnerReferences = []metav1.OwnerReference{*dgdControllerOwnerReference(dgd)}

				t.Log("Render through the real direct-DS and materialized-DCD adapters")
				scheme := runtime.NewScheme()
				require.NoError(t, corev1.AddToScheme(scheme))
				renderer := newDCDWorkloadRenderer(fake.NewClientBuilder().WithScheme(scheme).Build(),
					&configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}, nil)
				dcdLeader, dcdWorker, err := renderer.renderMultinodePodTemplateSpecs(t.Context(), child)
				require.NoError(t, err)
				dsLeader, dsWorker, err := renderer.renderMultinodePodTemplateSpecsForDGDComponent(
					t.Context(), dgd, normalized["worker"], "worker", child.Name,
					dynamo.GetDynamoNamespace(dgd, normalized["worker"]), dynamo.BackendFrameworkVLLM, nil)
				require.NoError(t, err)

				t.Log("Policy env belongs to runtimes, while native engines retain graph env")
				for index, pod := range []*corev1.PodTemplateSpec{dsLeader, dsWorker, dcdLeader, dcdWorker} {
					require.Contains(t, pod.Spec.Containers[0].Env, corev1.EnvVar{Name: "GLOBAL", Value: "retained"})
					runtime := &pod.Spec.Containers[0]
					if runtimeLayout != "main" {
						require.Contains(t, pod.Spec.Containers[0].Env, corev1.EnvVar{Name: consts.EnvKvTransferPreferredWeight, Value: "0.75"})
						for _, variable := range pod.Spec.Containers[0].Env {
							require.NotContains(t, []string{consts.EnvKvTransferDomain, consts.EnvKvTransferEnforcement, consts.EnvTopologyEnabled, consts.EnvTopologyMountPath}, variable.Name)
						}
						if runtimeLayout == "leader-sidecar" && index%2 == 1 {
							require.Empty(t, pod.Spec.InitContainers)
							continue
						}
						require.Len(t, pod.Spec.InitContainers, 1)
						runtime = &pod.Spec.InitContainers[0]
					}
					for _, variable := range runtime.Env {
						require.NotEqual(t, consts.EnvKvTransferPreferredWeight, variable.Name)
					}
					require.Contains(t, runtime.Env, corev1.EnvVar{Name: consts.EnvKvTransferEnforcement, Value: "required"})
					require.Contains(t, runtime.Env, corev1.EnvVar{Name: "GLOBAL", Value: "retained"})
				}
				require.Equal(t, dcdLeader.Spec.Containers[0].Env, dsLeader.Spec.Containers[0].Env)
				require.Equal(t, dcdLeader.Spec.InitContainers, dsLeader.Spec.InitContainers)
				require.Equal(t, dcdWorker.Spec.Containers[0].Env, dsWorker.Spec.Containers[0].Env)
				require.Equal(t, dcdWorker.Spec.InitContainers, dsWorker.Spec.InitContainers)
			})
		}
	}
}
