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
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

func TestDCDOwnedServiceIsRecreatedAfterDeletion(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	runtimeConfig := &commoncontroller.RuntimeConfig{}

	replicas := int32(1)
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		Name: "service-watch", Namespace: env.Namespace(),
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

	t.Log("Reject initial-list and unrelated events while initializing through the real cached reconciler")
	serviceKey := client.ObjectKey{Namespace: dcd.Namespace, Name: dcd.Name}
	var cachedClient client.Client
	var testReconciler *DynamoComponentDeploymentReconciler
	env.StartManager(func(mgr ctrl.Manager) error {
		cachedClient = mgr.GetClient()
		r := &DynamoComponentDeploymentReconciler{
			Client: mgr.GetClient(), Recorder: mgr.GetEventRecorder("dynamocomponentdeployment"), Config: config, RuntimeConfig: runtimeConfig,
		}
		testReconciler = r
		controller, err := r.controllerBuilder(mgr)
		if err != nil {
			return err
		}

		// Only deliberate Service changes or its later deletion may enqueue.
		return controller.WithEventFilter(predicate.Funcs{
			CreateFunc:  func(event.CreateEvent) bool { return false },
			GenericFunc: func(event.GenericEvent) bool { return false },
			UpdateFunc: func(e event.UpdateEvent) bool {
				service, ok := e.ObjectNew.(*corev1.Service)
				if !ok || client.ObjectKeyFromObject(service) != serviceKey {
					return false
				}
				labelDrift := e.ObjectOld.GetLabels()[consts.KubeLabelDynamoComponent] != "wrong" && service.Labels[consts.KubeLabelDynamoComponent] == "wrong"
				annotationWakeup := e.ObjectOld.GetAnnotations()["example.com/identity-retry"] != "wake" && service.Annotations["example.com/identity-retry"] == "wake"
				return labelDrift || annotationWakeup
			},
			DeleteFunc: func(e event.DeleteEvent) bool {
				service, ok := e.Object.(*corev1.Service)
				return ok && client.ObjectKeyFromObject(service) == serviceKey
			},
		}).Complete(r)
	})

	dynamotesting.Eventually(t, func() (bool, string) {
		if _, err := testReconciler.Reconcile(ctx, ctrl.Request{NamespacedName: client.ObjectKeyFromObject(dcd)}); err != nil {
			return false, err.Error()
		}
		observed := &corev1.Service{}
		if err := env.Client().Get(ctx, serviceKey, observed); err != nil {
			return false, err.Error()
		}
		return observed.Labels[consts.KubeLabelDynamoComponent] == "frontend", "waiting for initial reconciliation"
	}, 10*time.Second, 100*time.Millisecond, "initial direct reconciliation did not finish")

	original := waitForDCDOwnedService(t, ctx, env.Client(), serviceKey, dcd.UID, "")

	t.Log("Metadata-only Service drift requeues its DCD and repairs component identity")
	original.Labels[consts.KubeLabelDynamoComponent] = "wrong"
	original.Labels["example.com/team"] = "external"
	require.NoError(t, env.Client().Update(ctx, original))
	dynamotesting.Eventually(t, func() (bool, string) {
		service := &corev1.Service{}
		if err := env.Client().Get(ctx, serviceKey, service); err != nil {
			return false, err.Error()
		}
		return service.Labels[consts.KubeLabelDynamoComponent] == "frontend" && service.Labels["example.com/team"] == "external", "waiting for Service identity repair"
	}, 10*time.Second, 100*time.Millisecond, "metadata-only Service changes did not trigger identity repair")

	t.Log("An annotation-only resource-version change can wake an outstanding identity repair")
	service := &corev1.Service{}
	require.NoError(t, env.Client().Get(ctx, serviceKey, service))
	delete(service.Labels, consts.KubeLabelDynamoNamespace)
	require.NoError(t, env.Client().Update(ctx, service))
	dynamotesting.Eventually(t, func() (bool, string) {
		observed := &corev1.Service{}
		if err := cachedClient.Get(ctx, serviceKey, observed); err != nil {
			return false, err.Error()
		}
		_, present := observed.Labels[consts.KubeLabelDynamoNamespace]
		return !present, "waiting for the cache to observe the filtered-out label edit"
	}, 10*time.Second, 100*time.Millisecond, "cache did not observe the Service edit")
	require.NoError(t, env.Client().Get(ctx, serviceKey, service))
	service.Annotations["example.com/identity-retry"] = "wake"
	require.NoError(t, env.Client().Update(ctx, service))
	dynamotesting.Eventually(t, func() (bool, string) {
		observed := &corev1.Service{}
		if err := env.Client().Get(ctx, serviceKey, observed); err != nil {
			return false, err.Error()
		}
		return observed.Labels[consts.KubeLabelDynamoNamespace] == service.Spec.Selector[consts.KubeLabelDynamoNamespace], "waiting for annotation-triggered identity repair"
	}, 10*time.Second, 100*time.Millisecond, "annotation-only Service update did not trigger identity repair")

	t.Log("Deleting the Service still recreates the DCD-owned endpoint")
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
