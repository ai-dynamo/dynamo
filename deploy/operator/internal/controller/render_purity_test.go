// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	componentdeployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/component"
	disaggregatedset "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/disaggregatedset"
	grove "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/grove"
	leaderworkerset "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/leaderworkerset"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestResourceRenderersOwnGeneratedMetadata(t *testing.T) {
	t.Log("Prepare caller-owned metadata and Pod templates with a conflicting selector")
	metadata := metav1.ObjectMeta{Labels: map[string]string{consts.KubeLabelDynamoSelector: "authored"}}
	template := &corev1.PodTemplateSpec{
		ObjectMeta: *metadata.DeepCopy(),
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "runtime:1.6.0"}}},
	}
	dcd := &v1beta1.DynamoComponentDeployment{Name: "graph-worker", Namespace: "default"}
	dcd.Spec.Replicas = new(int32(3))
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "default"}
	originalMetadata, originalTemplate := metadata.DeepCopy(), template.DeepCopy()

	t.Log("Render Deployment and Grove identities without modifying authored metadata")
	deployment := componentdeployment.Render(dcd, metadata, template, appsv1.DeploymentStrategy{})
	groveMetadata := grove.RenderPodMetadata(dgd.Name, dcd.Name, metadata)
	require.EqualValues(t, 3, *deployment.Spec.Replicas)
	require.Equal(t, dcd.Name, deployment.Spec.Template.Labels[consts.KubeLabelDynamoSelector])
	require.Equal(t, dcd.Name, groveMetadata.Labels[consts.KubeLabelDynamoSelector])
	require.Equal(t, originalMetadata, &metadata)
	require.Equal(t, originalTemplate, template)

	t.Log("Render multinode templates without the Deployment-only selector")
	lws := leaderworkerset.Render(dcd, leaderworkerset.Name(dcd.Name), metadata, template, template)
	require.NotContains(t, lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Labels, consts.KubeLabelDynamoSelector)
	require.NotContains(t, lws.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels, consts.KubeLabelDynamoSelector)
	require.Equal(t, originalTemplate, template)

	t.Log("Mutate rendered objects and verify that caller-owned replicas, maps and nested templates remain intact")
	*deployment.Spec.Replicas = 7
	deployment.Spec.Template.Spec.Containers[0].Image = "changed"
	lws.Spec.LeaderWorkerTemplate.LeaderTemplate.Spec.Containers[0].Image = "changed"
	groveMetadata.Labels[consts.KubeLabelDynamoSelector] = "changed"
	require.EqualValues(t, 3, *dcd.Spec.Replicas)
	require.Equal(t, originalTemplate, template)
	require.Equal(t, originalMetadata, &metadata)

	t.Log("Keep native DisaggregatedSet role maps independent from the rendered object")
	roles := []any{map[string]any{"name": "prefill", "spec": map[string]any{"replicas": int64(1)}}}
	originalRoles := runtime.DeepCopyJSONValue(roles)
	ds := disaggregatedset.Render(dgd, roles)
	ds.Object["spec"].(map[string]any)["roles"].([]any)[0].(map[string]any)["name"] = "changed"
	require.Equal(t, originalRoles, roles)

	t.Log("Preserve unspecified replicas for Kubernetes defaulting")
	dcd.Spec.Replicas = nil
	deployment = componentdeployment.Render(dcd, metadata, template, appsv1.DeploymentStrategy{})
	require.Nil(t, deployment.Spec.Replicas)
}
