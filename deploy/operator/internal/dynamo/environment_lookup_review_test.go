// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestDeploymentConfigAliasSurvivesUnrelatedPendingBindings(t *testing.T) {
	t.Log("Known deployment config must survive an unrelated cycle removed by topology binding")
	component := v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		Multinode: &v1beta1.MultinodeSpec{NodeCount: 2},
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "engine:1.8.0", Command: []string{"python3", "-m", "dynamo.vllm"},
			Env: []corev1.EnvVar{{Name: "Z_CONFIG", Value: `{"worker":{"ServiceArgs":{"workers":4,"resources":{"cpu":"2"}}}}`},
				{Name: consts.DynamoRankEnvVar, Value: "$(PENDING)"}},
		}}}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{component},
			Env: []corev1.EnvVar{{Name: consts.DynamoDeploymentConfigEnvVar, Value: "$(Z_CONFIG)"}, {Name: "PENDING", Value: "$(DYNAMO_RANK)"}},
		},
	}
	original := dgd.DeepCopy()
	children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
	require.NoError(t, err)
	require.EqualValues(t, 4, *children["worker"].Spec.Replicas)
	main := GetMainContainer(&children["worker"].Spec.DynamoComponentDeploymentSharedSpec)
	require.Equal(t, resource.MustParse("2"), main.Resources.Requests[corev1.ResourceCPU])

	t.Log("Final rendering applies owned topology bindings before validating the complete graph")
	pod, err := GeneratePodSpecForComponent(&component, BackendFrameworkVLLM, nil, dgd, RoleLeader, 2,
		&configv1alpha1.OperatorConfiguration{}, consts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(0))
	require.NoError(t, err)
	resolved := semanticenv.Replay(pod.Containers[0].Env)
	require.True(t, resolved[consts.DynamoDeploymentConfigEnvVar].Known)
	require.JSONEq(t, component.PodTemplate.Spec.Containers[0].Env[0].Value, resolved[consts.DynamoDeploymentConfigEnvVar].Value)
	require.Equal(t, original, dgd)
}
