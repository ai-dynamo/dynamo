//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
)

func TestClusterDynamoGraphDeploymentCreatesComponentAndDeploymentManifests(t *testing.T) {
	env := clusterTestEnv.RunT(t)

	t.Log("Block downstream workload actuation and start the DGD-to-DCD controller chain")
	env.BlockWorkloads()
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{}}
	env.StartManager(func(mgr ctrl.Manager) error {
		if err := SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
		}); err != nil {
			return err
		}
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
		})
	})

	t.Log("Create a non-Grove graph with independently named frontend and decode components")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	clusterTestReadInput(t, env.Namespace(), "testdata/dgd/components/input.yaml", dgd)
	if err := env.Client().Create(t.Context(), dgd); err != nil {
		t.Fatalf("create DGD: %v", err)
	}

	t.Log("Match the DCDs and their terminal Deployments in one complete manifest contract")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dgd/components/output.yaml")
}

func TestClusterDynamoGraphDeploymentCreatesGroveManifest(t *testing.T) {
	env := clusterTestEnv.RunT(t)

	t.Log("Block Pods and start only the DGD reconciler with Grove enabled")
	env.BlockWorkloads()
	operatorConfig := clusterTestRestrictedConfig(env.Namespace())
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
			},
		})
	})

	t.Log("Create a Grove graph containing single-node and multinode components")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	clusterTestReadInput(t, env.Namespace(), "testdata/dgd/grove/input.yaml", dgd)
	if err := env.Client().Create(t.Context(), dgd); err != nil {
		t.Fatalf("create Grove DGD: %v", err)
	}

	t.Log("Match the complete PodCliqueSet produced without running Grove controllers")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dgd/grove/output.yaml")
}

func clusterTestRestrictedConfig(namespace string) *configv1alpha1.OperatorConfiguration {
	config := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(config)
	config.Namespace.Restricted = namespace
	config.GPU.DiscoveryEnabled = ptr.To(false)
	return config
}
