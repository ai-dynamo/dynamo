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
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
	clusterTestReadInput(t, env.Namespace(), "testdata/dcd/deployment/input.yaml", dcd)
	if err := env.Client().Create(t.Context(), dcd); err != nil {
		t.Fatalf("create deployment DCD: %v", err)
	}

	t.Log("Match the Deployment while the quota prevents its ReplicaSet from being actuated")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dcd/deployment/output.yaml")
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
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
	clusterTestReadInput(t, env.Namespace(), "testdata/dcd/lws/input.yaml", dcd)
	if err := env.Client().Create(t.Context(), dcd); err != nil {
		t.Fatalf("create LWS DCD: %v", err)
	}

	t.Log("Match the LeaderWorkerSet without running the LWS controller")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dcd/lws/output.yaml")
}
