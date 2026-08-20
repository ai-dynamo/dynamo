/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// elasticEPArgs is the command line that marks a component as an elastic-EP Ray launch:
// elastic EP only works on the Ray data-parallel backend, so both flags are required.
const elasticEPArgs = "python3 -m dynamo.vllm --enable-elastic-ep --data-parallel-backend ray"

// elasticEPComponentName is the one component every case in this file builds, so the
// assertions can derive the expected Service name from it.
const elasticEPComponentName = "worker"

// newElasticEPComponent builds a worker component whose main container carries the given
// command line, so a test can describe eligibility by its command line and scale alone.
func newElasticEPComponent(args string) v1beta1.DynamoComponentDeploymentSharedSpec {
	return v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: elasticEPComponentName,
		ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Args: []string{args}},
				},
			},
		},
	}
}

func TestIsSinglePodElasticEPLeader(t *testing.T) {
	tests := []struct {
		name      string
		component func() v1beta1.DynamoComponentDeploymentSharedSpec
		want      bool
	}{
		{
			name: "single-pod elastic-EP leader qualifies",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				return newElasticEPComponent(elasticEPArgs)
			},
			want: true,
		},
		{
			name: "an explicit single replica qualifies",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				component := newElasticEPComponent(elasticEPArgs)
				component.Replicas = ptr.To(int32(1))
				return component
			},
			want: true,
		},
		{
			name: "replicas > 1 is excluded because every replica runs its own Ray head",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				component := newElasticEPComponent(elasticEPArgs)
				component.Replicas = ptr.To(int32(2))
				return component
			},
			want: false,
		},
		{
			name: "multinode is excluded because its worker pods share the component labels",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				component := newElasticEPComponent(elasticEPArgs)
				component.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
				return component
			},
			want: false,
		},
		{
			name: "elastic EP on a non-Ray data-parallel backend is excluded",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				return newElasticEPComponent("python3 -m dynamo.vllm --enable-elastic-ep")
			},
			want: false,
		},
		{
			name: "a plain component is excluded",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				return newElasticEPComponent("python3 -m dynamo.vllm")
			},
			want: false,
		},
		{
			name: "a component without a main container is excluded",
			component: func() v1beta1.DynamoComponentDeploymentSharedSpec {
				return v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: elasticEPComponentName,
					ComponentType: v1beta1.ComponentTypeWorker,
				}
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Classify the component against the single-pod elastic-EP leader gate")
			component := tt.component()
			got := dynamo.IsSinglePodElasticEPLeader(&component)

			t.Log("Verify only a component that renders as exactly one Ray head qualifies")
			if got != tt.want {
				t.Errorf("dynamo.IsSinglePodElasticEPLeader = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGroveStableResourcesReconcilerElasticEPLeaderServiceLifecycle(t *testing.T) {
	tests := []struct {
		name          string
		mutate        func(component *v1beta1.DynamoComponentDeploymentSharedSpec)
		wantEmitted   bool
		wantSelectors map[string]string
	}{
		{
			name:        "emits the leader Service for a single-pod elastic-EP component",
			mutate:      func(*v1beta1.DynamoComponentDeploymentSharedSpec) {},
			wantEmitted: true,
			wantSelectors: map[string]string{
				commonconsts.KubeLabelDynamoComponent:     elasticEPComponentName,
				commonconsts.KubeLabelDynamoComponentType: string(v1beta1.ComponentTypeWorker),
			},
		},
		{
			name: "does not emit for replicas > 1",
			mutate: func(component *v1beta1.DynamoComponentDeploymentSharedSpec) {
				component.Replicas = ptr.To(int32(2))
			},
			wantEmitted: false,
		},
		{
			name: "does not emit for a multinode component",
			mutate: func(component *v1beta1.DynamoComponentDeploymentSharedSpec) {
				component.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
			},
			wantEmitted: false,
		},
		{
			name: "does not emit for a component that is not an elastic-EP Ray launch",
			mutate: func(component *v1beta1.DynamoComponentDeploymentSharedSpec) {
				component.PodTemplate.Spec.Containers[0].Args = []string{"python3 -m dynamo.vllm"}
			},
			wantEmitted: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a DGD with one worker component and apply the case's shape")
			component := newElasticEPComponent(elasticEPArgs)
			tt.mutate(&component)
			dgd := newElasticEPTestDGD(component)

			t.Log("Reconcile the Grove stable resources")
			ctx := context.Background()
			reconciler, kubeClient := newElasticEPTestStableResourcesReconciler(t, dgd)
			if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
				t.Fatalf("Reconcile returned an error: %v", err)
			}

			t.Log("Verify the leader Service exists only for the single-pod elastic-EP shape")
			service := &corev1.Service{}
			err := kubeClient.Get(ctx, types.NamespacedName{
				Name:      dynamo.ElasticEPLeaderServiceName(dynamo.GetDCDResourceName(dgd, elasticEPComponentName, "")),
				Namespace: dgd.Namespace,
			}, service)
			if tt.wantEmitted {
				if err != nil {
					t.Fatalf("expected the leader Service to exist, got error: %v", err)
				}
				for key, want := range tt.wantSelectors {
					if got := service.Spec.Selector[key]; got != want {
						t.Errorf("Selector[%q] = %q, want %q", key, got, want)
					}
				}
				// The controller reference is what lets the DGD's Owns(&corev1.Service{})
				// watch map a deleted Service back to its DGD and recreate it.
				if !metav1.IsControlledBy(service, dgd) {
					t.Errorf("leader Service is not controlled by the DGD, so the owned-Service watch cannot recreate it: %v", service.OwnerReferences)
				}
				return
			}
			if !apierrors.IsNotFound(err) {
				t.Fatalf("expected the leader Service to be absent, got service %q with error %v", service.Name, err)
			}
		})
	}
}

func TestGroveStableResourcesReconcilerDeletesElasticEPLeaderServiceWhenEligibilityIsRemoved(t *testing.T) {
	t.Log("Reconcile a single-pod elastic-EP component so the leader Service is created")
	ctx := context.Background()
	dgd := newElasticEPTestDGD(newElasticEPComponent(elasticEPArgs))
	reconciler, kubeClient := newElasticEPTestStableResourcesReconciler(t, dgd)
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("first Reconcile returned an error: %v", err)
	}

	serviceKey := types.NamespacedName{
		Name:      dynamo.ElasticEPLeaderServiceName(dynamo.GetDCDResourceName(dgd, elasticEPComponentName, "")),
		Namespace: dgd.Namespace,
	}
	if err := kubeClient.Get(ctx, serviceKey, &corev1.Service{}); err != nil {
		t.Fatalf("expected the leader Service to exist after the first reconcile: %v", err)
	}

	t.Log("Drop elastic EP from the component and reconcile again")
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Args = []string{"python3 -m dynamo.vllm"}
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("second Reconcile returned an error: %v", err)
	}

	t.Log("Verify the now-stale leader Service was deleted rather than left pointing at the component")
	err := kubeClient.Get(ctx, serviceKey, &corev1.Service{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected the leader Service to be deleted, got error %v", err)
	}
}

func TestGroveStableResourcesReconcilerLeavesAnUnownedNameCollisionAlone(t *testing.T) {
	t.Log("Pre-create a Service this DGD does not own, at the name the leader Service would take")
	ctx := context.Background()
	dgd := newElasticEPTestDGD(newElasticEPComponent("python3 -m dynamo.vllm"))
	serviceName := dynamo.ElasticEPLeaderServiceName(dynamo.GetDCDResourceName(dgd, elasticEPComponentName, ""))
	unowned := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: dgd.Namespace,
			Labels:    map[string]string{"owner": "somebody-else"},
		},
		Spec: corev1.ServiceSpec{Ports: []corev1.ServicePort{{Name: "http", Port: 80}}},
	}
	reconciler, kubeClient := newElasticEPTestStableResourcesReconciler(t, dgd, unowned)

	t.Log("Reconcile a component that does not qualify, so it takes the delete path")
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("Reconcile returned an error: %v", err)
	}

	t.Log("Verify the unowned Service survived: the operator must not delete what it never created")
	survivor := &corev1.Service{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dgd.Namespace}, survivor); err != nil {
		t.Fatalf("expected the unowned Service to survive, got error: %v", err)
	}
	if survivor.Labels["owner"] != "somebody-else" {
		t.Errorf("unowned Service was modified: labels = %v", survivor.Labels)
	}
}

func TestGroveStableResourcesReconcilerDeletesTheExactOwnershipCheckedService(t *testing.T) {
	t.Log("Seed a DGD-owned leader Service carrying a known UID")
	ctx := context.Background()
	dgd := newElasticEPTestDGD(newElasticEPComponent("python3 -m dynamo.vllm"))
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	serviceName := dynamo.ElasticEPLeaderServiceName(dynamo.GetDCDResourceName(dgd, elasticEPComponentName, ""))
	owned := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      serviceName,
			Namespace: dgd.Namespace,
			UID:       types.UID("leader-service-uid"),
		},
	}
	if err := ctrl.SetControllerReference(dgd, owned, scheme); err != nil {
		t.Fatalf("failed to set the controller reference: %v", err)
	}

	t.Log("Reconcile a component that no longer qualifies, so it takes the delete path")
	var deleteOptions client.DeleteOptions
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, owned).
		WithInterceptorFuncs(interceptor.Funcs{
			Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
				for _, opt := range opts {
					opt.ApplyToDelete(&deleteOptions)
				}
				return c.Delete(ctx, obj, opts...)
			},
		}).
		Build()
	reconciler := newGroveStableResourcesReconciler(
		kubeClient,
		events.NewFakeRecorder(100),
		&configv1alpha1.OperatorConfiguration{},
	)
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("Reconcile returned an error: %v", err)
	}

	t.Log("Verify the owned Service was deleted, pinned by UID to the object the ownership check read")
	err := kubeClient.Get(ctx, types.NamespacedName{Name: serviceName, Namespace: dgd.Namespace}, &corev1.Service{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected the owned Service to be deleted, got error %v", err)
	}
	if deleteOptions.Preconditions == nil || deleteOptions.Preconditions.UID == nil {
		t.Fatal("delete carried no UID precondition, so a same-name replacement could be removed instead")
	}
	if got := *deleteOptions.Preconditions.UID; got != owned.UID {
		t.Errorf("precondition UID = %q, want %q", got, owned.UID)
	}
}

func TestGroveStableResourcesReconcilerConvergesLeaderServiceAnnotations(t *testing.T) {
	t.Log("Reconcile a single-pod elastic-EP component so the leader Service is created")
	ctx := context.Background()
	dgd := newElasticEPTestDGD(newElasticEPComponent(elasticEPArgs))
	dgd.Spec.Annotations = map[string]string{"example.com/note": "before"}
	reconciler, kubeClient := newElasticEPTestStableResourcesReconciler(t, dgd)
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("first Reconcile returned an error: %v", err)
	}

	t.Log("Edit the DGD annotation and reconcile again")
	dgd.Spec.Annotations["example.com/note"] = "after"
	if _, err := reconciler.Reconcile(ctx, dgd, dgd); err != nil {
		t.Fatalf("second Reconcile returned an error: %v", err)
	}

	t.Log("Verify the edit converged onto the Service, which SyncResource alone would not do")
	service := &corev1.Service{}
	key := types.NamespacedName{
		Name:      dynamo.ElasticEPLeaderServiceName(dynamo.GetDCDResourceName(dgd, elasticEPComponentName, "")),
		Namespace: dgd.Namespace,
	}
	if err := kubeClient.Get(ctx, key, service); err != nil {
		t.Fatalf("expected the leader Service to exist: %v", err)
	}
	if got := service.Annotations["example.com/note"]; got != "after" {
		t.Errorf("annotation example.com/note = %q, want %q", got, "after")
	}
}

// newElasticEPTestDGD wraps a single component in the smallest DGD the stable-resources
// reconciler accepts: no modelRef, so no model service, and no frontend, so no ingress.
func newElasticEPTestDGD(component v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
			UID:       types.UID("dgd-uid"),
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components:       []v1beta1.DynamoComponentDeploymentSharedSpec{component},
		},
	}
}

func newElasticEPTestStableResourcesReconciler(
	t testing.TB,
	dgd *v1beta1.DynamoGraphDeployment,
	existing ...client.Object,
) (*groveStableResourcesReconciler, client.Client) {
	t.Helper()
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(append([]client.Object{dgd}, existing...)...).
		Build()
	reconciler := newGroveStableResourcesReconciler(
		kubeClient,
		events.NewFakeRecorder(100),
		&configv1alpha1.OperatorConfiguration{},
	)
	return reconciler, kubeClient
}
