//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestDCDOwnedServiceIsRecreatedAfterDeletion(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	runtimeConfig := &commoncontroller.RuntimeConfig{}

	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoComponentDeployment(mgr, DynamoComponentDeploymentSetupOptions{
			SetupOptions: SetupOptions{Config: config, RuntimeConfig: runtimeConfig},
		})
	})

	replicas := int32(1)
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "service-watch", Namespace: env.Namespace()},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "frontend",
				ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
				Replicas:      &replicas,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:  consts.MainContainerName,
					Image: "registry.example/dynamo-frontend:test",
				}}}},
			},
		},
	}
	require.NoError(t, env.Client().Create(ctx, dcd))

	serviceKey := client.ObjectKey{Namespace: dcd.Namespace, Name: dcd.Name}
	original := waitForDCDOwnedService(t, ctx, env.Client(), serviceKey, dcd.UID, "")
	require.NoError(t, env.Client().Delete(ctx, original))

	waitForDCDOwnedService(t, ctx, env.Client(), serviceKey, dcd.UID, original.UID)
}

func waitForDCDOwnedService(
	t *testing.T,
	ctx context.Context,
	kubeClient client.Client,
	key client.ObjectKey,
	dcdUID types.UID,
	previousUID types.UID,
) *corev1.Service {
	t.Helper()
	dynamotesting.Eventually(t, func() (bool, string) {
		service := &corev1.Service{}
		if err := kubeClient.Get(ctx, key, service); err != nil {
			return false, fmt.Sprintf("get Service: %v", err)
		}
		if previousUID != "" && service.UID == previousUID {
			return false, "Service has not been recreated yet"
		}
		owner := metav1.GetControllerOf(service)
		if owner == nil || owner.Kind != "DynamoComponentDeployment" || owner.UID != dcdUID {
			return false, fmt.Sprintf("Service controller owner = %#v", owner)
		}
		return true, "DCD-owned Service exists"
	}, 10*time.Second, 100*time.Millisecond, "DCD-owned Service was not reconciled")

	service := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, key, service))
	return service
}
