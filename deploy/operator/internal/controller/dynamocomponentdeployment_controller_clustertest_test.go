//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
)

func TestClusterDynamoComponentDeploymentCreatesDeploymentManifest(t *testing.T) {
	env := clusterTestEnv.RunT(t)

	t.Log("Block ReplicaSets and Pods and start only the DCD reconciler")
	env.BlockWorkloads()
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{}},
			},
		})
	})

	t.Log("Create a single-node decode component with a production replica count")
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "deployment-manifest", Namespace: env.Namespace()},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:      ptr.To[int32](2),
				PodTemplate:   clusterTestPodTemplate("registry.example/dynamo-worker:test"),
			},
		},
	}
	if err := env.Client().Create(t.Context(), dcd); err != nil {
		t.Fatalf("create deployment DCD: %v", err)
	}

	t.Log("Match the Deployment while the quota prevents its ReplicaSet from being actuated")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dynamocomponentdeployment-deployment.yaml")
}

func TestClusterDynamoComponentDeploymentCreatesLeaderWorkerSetManifest(t *testing.T) {
	env := clusterTestEnv.RunT(t)

	t.Log("Block Pods and start only the DCD reconciler with LWS enabled")
	env.BlockWorkloads()
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{LWS: true}},
			},
		})
	})

	t.Log("Create a two-node decode component with two LWS replicas")
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "lws-manifest", Namespace: env.Namespace()},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				Replicas:      ptr.To[int32](2),
				Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				PodTemplate:   clusterTestLWSPodTemplate("registry.example/dynamo-worker:test"),
			},
		},
	}
	if err := env.Client().Create(t.Context(), dcd); err != nil {
		t.Fatalf("create LWS DCD: %v", err)
	}

	t.Log("Match the LeaderWorkerSet without running the LWS controller")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dynamocomponentdeployment-lws.yaml")
}
