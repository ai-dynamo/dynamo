/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commonController "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
)

func testDRAClaimComponent(objectName string, template bool) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	podClaim := corev1.PodResourceClaim{Name: "accelerators"}
	if template {
		podClaim.ResourceClaimTemplateName = ptr.To(objectName)
	} else {
		podClaim.ResourceClaimName = ptr.To(objectName)
	}

	return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				ResourceClaims: []corev1.PodResourceClaim{podClaim},
				Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName,
					Resources: corev1.ResourceRequirements{
						Claims: []corev1.ResourceClaim{{Name: "accelerators"}},
					},
				}},
			},
		},
	}
}

func TestMapResourceClaimToDCDRequests(t *testing.T) {
	t.Log("Create one matching and one unrelated component deployment")
	matching := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "matching", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: testDRAClaimComponent("gpu-claim", false),
		},
	}
	unrelated := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: testDRAClaimComponent("gpu-claim", false),
		},
	}
	unrelated.Spec.Multinode = nil
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	reconciler := &DynamoComponentDeploymentReconciler{
		Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(matching, unrelated).Build(),
	}

	t.Log("Map a ResourceClaim event to its dependent component deployment")
	requests := reconciler.mapResourceClaimToDCDRequests(context.Background(), &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-claim", Namespace: "default"},
	})

	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "default", Name: "matching"}}}, requests)
}

func TestMapResourceClaimTemplateToDGDRequests(t *testing.T) {
	t.Log("Create one matching and one unrelated graph deployment")
	matching := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "matching", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "frontend"},
				testDRAClaimComponent("gpu-template", true),
			},
		},
	}
	unrelated := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				testDRAClaimComponent("gpu-template", true),
			},
		},
	}
	unrelated.Spec.Components[0].Multinode = nil
	lws := matching.DeepCopy()
	lws.Name = "lws"
	lws.Annotations = map[string]string{commonconsts.KubeAnnotationEnableGrove: commonconsts.KubeLabelValueFalse}
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(scheme).WithObjects(matching, unrelated, lws).Build(),
		RuntimeConfig: &commonController.RuntimeConfig{Gate: features.Gates{Grove: true}},
	}

	t.Log("Map a ResourceClaimTemplate event to its dependent graph deployment")
	requests := reconciler.mapResourceClaimTemplateToDGDRequests(context.Background(), &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-template", Namespace: "default"},
	})

	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "default", Name: "matching"}}}, requests)
}

func TestMapDeviceClassToDCDRequests(t *testing.T) {
	t.Log("Create DRA-backed component deployments inside and outside the restricted namespace")
	draBacked := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "dra-backed", Namespace: "dra-workloads"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: testDRAClaimComponent("gpu-template", true),
		},
	}
	draBackedOutsideScope := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "outside-scope", Namespace: "other-workloads"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: testDRAClaimComponent("gpu-template", true),
		},
	}
	scalarGPU := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "scalar-gpu", Namespace: "dra-workloads"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
			},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	reconciler := &DynamoComponentDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(scheme).WithObjects(draBacked, draBackedOutsideScope, scalarGPU).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "dra-workloads"}},
		RuntimeConfig: &commonController.RuntimeConfig{},
	}

	t.Log("Map a cluster-scoped DeviceClass event to every DRA-backed component deployment")
	requests := reconciler.mapDeviceClassToDCDRequests(context.Background(), &resourcev1.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu.example.com"},
	})

	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "dra-workloads", Name: "dra-backed"}}}, requests)
}

func TestMapDeviceClassToDGDRequests(t *testing.T) {
	t.Log("Create DRA-backed graph deployments inside and outside the restricted namespace")
	draBacked := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "dra-backed", Namespace: "dra-workloads"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "frontend"},
				testDRAClaimComponent("gpu-template", true),
			},
		},
	}
	draBackedOutsideScope := draBacked.DeepCopy()
	draBackedOutsideScope.Name = "outside-scope"
	draBackedOutsideScope.Namespace = "other-workloads"
	unrelated := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "unrelated", Namespace: "dra-workloads"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "frontend"}},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(scheme).WithObjects(draBacked, draBackedOutsideScope, unrelated).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "dra-workloads"}},
		RuntimeConfig: &commonController.RuntimeConfig{Gate: features.Gates{Grove: true}},
	}

	t.Log("Map a cluster-scoped DeviceClass event to every DRA-backed graph deployment")
	requests := reconciler.mapDeviceClassToDGDRequests(context.Background(), &resourcev1.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu.example.com"},
	})

	assert.Equal(t, []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: "dra-workloads", Name: "dra-backed"}}}, requests)
}

func TestDeploymentEventFilterAllowsClusterScopedDeviceClasses(t *testing.T) {
	filter := deploymentEventFilter(
		&configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "dra-workloads"}},
		&commonController.RuntimeConfig{},
	)

	t.Log("Allow cluster-scoped DeviceClass events so their mapper can select in-scope deployments")
	assert.True(t, filter.Create(event.CreateEvent{Object: &resourcev1.DeviceClass{}}))

	t.Log("Continue filtering namespaced dependency events at their source")
	assert.True(t, filter.Create(event.CreateEvent{Object: &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: "dra-workloads"}}}))
	assert.False(t, filter.Create(event.CreateEvent{Object: &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: "other-workloads"}}}))
}
