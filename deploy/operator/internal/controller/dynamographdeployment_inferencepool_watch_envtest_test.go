//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

func TestDGDInferencePoolRecoversFromDeletionAndSpecDrift(t *testing.T) {
	ctx := context.Background()
	t.Log("Start the DGD controller with the InferencePool API installed")
	env := sharedEnv.ForTest(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			Config: config, RuntimeConfig: runtimeConfig,
		})
	})

	t.Log("Create an EPP graph and wait for its owned InferencePool")
	dgd := &v1beta1.DynamoGraphDeployment{
		Name: "pool-watch", Namespace: env.Namespace(),
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "epp", ComponentType: v1beta1.ComponentTypeEPP,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: consts.MainContainerName, Image: "registry.example/dynamo-frontend:1.6.0",
				}}}},
			}},
		},
	}
	require.NoError(t, env.Client().Create(ctx, dgd))
	key := client.ObjectKey{Name: epp.GetPoolName(dgd.Name), Namespace: dgd.Namespace}
	pool := &gaiev1.InferencePool{}
	require.Eventually(t, func() bool {
		return env.Client().Get(ctx, key, pool) == nil && metav1.IsControlledBy(pool, dgd)
	}, 10*time.Second, 100*time.Millisecond)
	originalUID := pool.UID
	desired := pool.Spec.DeepCopy()

	t.Log("Let initial child creation events settle before deleting only the pool")
	require.Eventually(t, func() bool {
		current := &v1beta1.DynamoGraphDeployment{}
		return env.Client().Get(ctx, client.ObjectKeyFromObject(dgd), current) == nil &&
			current.Status.ObservedGeneration == current.Generation
	}, 10*time.Second, 100*time.Millisecond)
	time.Sleep(time.Second)
	require.NoError(t, env.Client().Delete(ctx, pool))

	t.Log("Verify deletion alone recreates a pool with the original desired spec")
	require.Eventually(t, func() bool {
		return env.Client().Get(ctx, key, pool) == nil && pool.UID != originalUID && metav1.IsControlledBy(pool, dgd)
	}, 10*time.Second, 100*time.Millisecond)
	require.Equal(t, *desired, pool.Spec)

	t.Log("Change the picker reference without changing the DGD or its workloads")
	pool.Spec.EndpointPickerRef.Name = "wrong-picker"
	require.NoError(t, env.Client().Update(ctx, pool))

	t.Log("Verify the pool spec converges back to the graph's desired picker")
	require.Eventually(t, func() bool {
		return env.Client().Get(ctx, key, pool) == nil && pool.Spec.EndpointPickerRef.Name == desired.EndpointPickerRef.Name
	}, 10*time.Second, 100*time.Millisecond)
}

func TestDGDControllerStartsWithoutInferencePoolAPI(t *testing.T) {
	t.Log("Start an environment with only the operator CRDs")
	env := operatorenv.New(operatorenv.Options{
		CRDDirectoryPaths: []string{filepath.Join("..", "..", "config", "crd", "bases")},
		SetupWebhooks:     setupProductionWebhooks,
	}).RunT(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()

	t.Log("Start and synchronize the DGD controller without an InferencePool watch")
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{},
		})
	})
}
