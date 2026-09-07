//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestClusterGroveScalerUsesScaleSubresource(t *testing.T) {
	env := clusterTestEnv.RunT(t)
	ctx := context.Background()

	t.Log("Create a real Grove PodClique through the cluster API server")
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: env.Namespace()},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
				},
			},
		},
	})
	podClique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker", Namespace: env.Namespace()},
		Spec: grovev1alpha1.PodCliqueSpec{
			RoleName: "worker",
			PodSpec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "worker", Image: "example.invalid/worker:test"}},
			},
			Replicas: 1,
		},
	}
	require.NoError(t, env.Client().Create(ctx, podClique))

	t.Log("Scale the PodClique through the production controller-runtime subresource path")
	require.NoError(t, newGroveScaler(env.Client()).Reconcile(ctx, dgd, nil))

	t.Log("Read the PodClique back and verify the API server applied the scale update")
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(podClique), podClique))
	require.Equal(t, int32(3), podClique.Spec.Replicas)
}
