/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/onsi/gomega"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func newDynamoGraphDeploymentControllerTestScheme(t testing.TB) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	for _, addToScheme := range []func(*runtime.Scheme) error{
		corev1.AddToScheme,
		autoscalingv1.AddToScheme,
		networkingv1.AddToScheme,
		v1alpha1.AddToScheme,
		v1beta1.AddToScheme,
		grovev1alpha1.AddToScheme,
	} {
		if err := addToScheme(s); err != nil {
			t.Fatalf("failed to add type to scheme: %v", err)
		}
	}
	return s
}

func TestDynamoGraphDeploymentReconciler_preserveExistingDCDBackendFramework(t *testing.T) {
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

	existing := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner-frontend",
			Namespace: "jsm",
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "",
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "Frontend",
				ComponentType: v1beta1.ComponentTypeFrontend,
			},
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			Build(),
	}

	desiredExisting := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      existing.Name,
			Namespace: existing.Namespace,
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
		},
	}
	gomega.NewWithT(t).Expect(reconciler.preserveExistingDCDBackendFramework(ctx, desiredExisting)).To(gomega.Succeed())
	gomega.NewWithT(t).Expect(desiredExisting.Spec.BackendFramework).To(gomega.Equal(""))

	desiredNew := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner-vllmdecodeworker-2dad72b9",
			Namespace: "jsm",
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
		},
	}
	gomega.NewWithT(t).Expect(reconciler.preserveExistingDCDBackendFramework(ctx, desiredNew)).To(gomega.Succeed())
	gomega.NewWithT(t).Expect(desiredNew.Spec.BackendFramework).To(gomega.Equal("vllm"))
}

func TestDynamoGraphDeploymentReconciler_reconcileScalingAdapters(t *testing.T) {
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)

	tests := []struct {
		name                 string
		dgd                  *v1beta1.DynamoGraphDeployment
		existingAdapters     []v1alpha1.DynamoGraphDeploymentScalingAdapter
		expectedAdapterCount int
		expectedAdapters     map[string]int32 // map of adapter name to expected replicas
		expectDeleted        []string         // adapter names that should be deleted
	}{
		{
			name: "creates adapters for services with scalingAdapter.enabled=true",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
						"decode": {
							Replicas: ptr.To(int32(3)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 2,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
				"test-dgd-decode":   3,
			},
		},
		{
			name: "uses default replicas when not specified",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-worker": 1, // default replicas
			},
		},
		{
			name: "skips adapter creation when not enabled",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
						"decode": {
							Replicas: ptr.To(int32(3)),
							// No ScalingAdapter or Enabled=false means no adapter created
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
			},
		},
		{
			name: "deletes adapter when service is removed",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					UID:       "test-uid",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 2,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "Frontend",
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-removed",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 1,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "removed",
						},
					},
				},
			},
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"test-dgd-frontend": 2,
			},
			expectDeleted: []string{"test-dgd-removed"},
		},
		{
			name: "deletes adapter when scalingAdapter.enabled is not set",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					UID:       "test-uid",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {
							Replicas: ptr.To(int32(2)),
							// No ScalingAdapter means adapter should be deleted
						},
					},
				},
			}),
			existingAdapters: []v1alpha1.DynamoGraphDeploymentScalingAdapter{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
						},
						OwnerReferences: []metav1.OwnerReference{
							{
								APIVersion: "nvidia.com/v1alpha1",
								Kind:       "DynamoGraphDeployment",
								Name:       "test-dgd",
								UID:        "test-uid",
							},
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentScalingAdapterSpec{
						Replicas: 2,
						DGDRef: v1alpha1.DynamoGraphDeploymentServiceRef{
							Name:        "test-dgd",
							ServiceName: "Frontend",
						},
					},
				},
			},
			expectedAdapterCount: 0,
			expectedAdapters:     map[string]int32{},
			expectDeleted:        []string{"test-dgd-frontend"},
		},
		{
			name: "adapter name uses lowercase service name",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my-dgd",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"MyService": {
							Replicas: ptr.To(int32(1)),
							ScalingAdapter: &v1alpha1.ScalingAdapter{
								Enabled: true,
							},
						},
					},
				},
			}),
			expectedAdapterCount: 1,
			expectedAdapters: map[string]int32{
				"my-dgd-myservice": 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build initial objects
			var initObjs []client.Object
			initObjs = append(initObjs, tt.dgd)
			for i := range tt.existingAdapters {
				initObjs = append(initObjs, &tt.existingAdapters[i])
			}

			// Create fake client
			fakeClient := fake.NewClientBuilder().
				WithScheme(testScheme).
				WithObjects(initObjs...).
				Build()

			// Create reconciler
			r := &DynamoGraphDeploymentReconciler{
				Client:   fakeClient,
				Recorder: record.NewFakeRecorder(10),
			}

			// Run reconcileScalingAdapters
			ctx := context.Background()
			err := r.reconcileScalingAdapters(ctx, tt.dgd)
			if err != nil {
				t.Fatalf("reconcileScalingAdapters() error = %v", err)
			}

			// Verify adapters
			adapterList := &v1alpha1.DynamoGraphDeploymentScalingAdapterList{}
			if err := fakeClient.List(ctx, adapterList, client.InNamespace("default")); err != nil {
				t.Fatalf("Failed to list adapters: %v", err)
			}

			if len(adapterList.Items) != tt.expectedAdapterCount {
				t.Errorf("Expected %d adapters, got %d", tt.expectedAdapterCount, len(adapterList.Items))
			}

			// Check expected adapters exist with correct replicas
			for name, expectedReplicas := range tt.expectedAdapters {
				adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{}
				err := fakeClient.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, adapter)
				if err != nil {
					t.Errorf("Expected adapter %s to exist, but got error: %v", name, err)
					continue
				}
				if adapter.Spec.Replicas != expectedReplicas {
					t.Errorf("Adapter %s has replicas=%d, expected %d", name, adapter.Spec.Replicas, expectedReplicas)
				}
			}

			// Check that deleted adapters don't exist
			for _, name := range tt.expectDeleted {
				adapter := &v1alpha1.DynamoGraphDeploymentScalingAdapter{}
				err := fakeClient.Get(ctx, types.NamespacedName{Name: name, Namespace: "default"}, adapter)
				if err == nil {
					t.Errorf("Expected adapter %s to be deleted, but it still exists", name)
				}
			}
		})
	}
}

func TestDynamoGraphDeploymentReconciler_reconcilePVCs(t *testing.T) {
	newScheme := func(t testing.TB) *runtime.Scheme {
		t.Helper()
		s := runtime.NewScheme()
		g := gomega.NewGomegaWithT(t)
		g.Expect(corev1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
		g.Expect(v1alpha1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
		g.Expect(v1beta1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
		return s
	}

	t.Run("native beta DGD is a no-op", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		dgd := &v1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "native", Namespace: "default"},
		}
		fakeClient := fake.NewClientBuilder().
			WithScheme(newScheme(t)).
			WithObjects(dgd).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		g.Expect(reconciler.reconcilePVCs(ctx, dgd)).NotTo(gomega.HaveOccurred())

		pvcs := &corev1.PersistentVolumeClaimList{}
		g.Expect(fakeClient.List(ctx, pvcs, client.InNamespace("default"))).NotTo(gomega.HaveOccurred())
		g.Expect(pvcs.Items).To(gomega.BeEmpty())
	})

	t.Run("converted alpha DGD creates preserved top-level PVC", func(t *testing.T) {
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		create := true
		pvcName := "model-cache"
		storage := resource.MustParse("5Gi")
		dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "converted", Namespace: "default"},
			Spec: v1alpha1.DynamoGraphDeploymentSpec{
				PVCs: []v1alpha1.PVC{{
					Create:           &create,
					Name:             &pvcName,
					StorageClass:     "standard",
					Size:             storage,
					VolumeAccessMode: corev1.ReadWriteOnce,
				}},
			},
		})
		fakeClient := fake.NewClientBuilder().
			WithScheme(newScheme(t)).
			WithObjects(dgd).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		g.Expect(reconciler.reconcilePVCs(ctx, dgd)).NotTo(gomega.HaveOccurred())

		pvc := &corev1.PersistentVolumeClaim{}
		g.Expect(fakeClient.Get(ctx, types.NamespacedName{Name: pvcName, Namespace: "default"}, pvc)).NotTo(gomega.HaveOccurred())
		g.Expect(pvc.Spec.AccessModes).To(gomega.Equal([]corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce}))
		g.Expect(pvc.Spec.StorageClassName).NotTo(gomega.BeNil())
		g.Expect(*pvc.Spec.StorageClassName).To(gomega.Equal("standard"))
		gotStorage := pvc.Spec.Resources.Requests[corev1.ResourceStorage]
		g.Expect(gotStorage.Cmp(storage)).To(gomega.Equal(0))
		g.Expect(metav1.IsControlledBy(pvc, dgd)).To(gomega.BeTrue())
	})
}

func TestDynamoGraphDeploymentReconciler_reconcileGMSResourceClaimTemplates_DRAValidation(t *testing.T) {
	tests := []struct {
		name    string
		spec    v1beta1.DynamoComponentDeploymentSharedSpec
		wantErr bool
	}{
		{
			name: "intra-pod failover does not require DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					Failover: &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeIntraPod},
				},
			},
		},
		{
			name: "inter-pod failover requires DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					Failover: &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeInterPod},
				},
			},
			wantErr: true,
		},
		{
			name: "gpu memory service requires DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			r := &DynamoGraphDeploymentReconciler{
				RuntimeConfig: &controller_common.RuntimeConfig{DRAEnabled: false},
			}
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{tt.spec},
				},
			}

			err := r.reconcileGMSResourceClaimTemplates(context.Background(), dgd)
			if tt.wantErr {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(err.Error()).To(gomega.ContainSubstring("requires DRA"))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())
		})
	}
}

func TestDynamoGraphDeploymentReconciler_createCheckpointCR_reusesExistingCapture(t *testing.T) {
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-worker-checkpoint",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: record.NewFakeRecorder(10),
	}

	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
	})
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: string(commonconsts.ComponentTypeWorker),
		Checkpoint: &v1alpha1.ServiceCheckpointConfig{
			Enabled: true,
			Mode:    v1alpha1.CheckpointModeAuto,
			Identity: &v1alpha1.DynamoCheckpointIdentity{
				Model:                identity.Model,
				BackendFramework:     identity.BackendFramework,
				TensorParallelSize:   1,
				PipelineParallelSize: 1,
				ExtraParameters:      map[string]string{},
			},
		},
		ExtraPodSpec: &v1alpha1.ExtraPodSpec{
			MainContainer: &corev1.Container{
				Name:  "main",
				Image: "new-writer:latest",
			},
		},
	}

	ckpt, err := reconciler.createCheckpointCR(ctx, dgd, "worker", betaComponent(t, component))
	if err != nil {
		t.Fatalf("createCheckpointCR() error = %v", err)
	}
	if ckpt.Name != "existing-worker-checkpoint" {
		t.Fatalf("createCheckpointCR() returned checkpoint %s, want existing-worker-checkpoint", ckpt.Name)
	}

	updated := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "existing-worker-checkpoint", Namespace: "default"}, updated); err != nil {
		t.Fatalf("Failed to get checkpoint: %v", err)
	}
	if len(updated.Spec.Job.PodTemplateSpec.Spec.Containers) != 1 {
		t.Fatalf("expected one job container, got %d", len(updated.Spec.Job.PodTemplateSpec.Spec.Containers))
	}
	if updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image != "keep-existing:latest" {
		t.Fatalf("existing job image was mutated to %s", updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image)
	}
}

func TestDynamoGraphDeploymentReconciler_reconcileCheckpoints_checkpointRefSkipsAutoCreateWhileReferencedCRIsNotReady(t *testing.T) {
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseCreating,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: record.NewFakeRecorder(10),
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	checkpointStatuses, checkpointInfos, err := reconciler.reconcileCheckpoints(ctx, dgd)
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if info.Ready {
		t.Fatalf("expected referenced checkpoint to remain not ready")
	}
	if !info.Exists {
		t.Fatalf("expected referenced checkpoint to exist")
	}
	if info.Hash != hash {
		t.Fatalf("checkpoint hash = %s, want %s", info.Hash, hash)
	}
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}

	checkpoints := &v1alpha1.DynamoCheckpointList{}
	if err := reconciler.List(ctx, checkpoints, client.InNamespace("default")); err != nil {
		t.Fatalf("failed to list checkpoints: %v", err)
	}
	if len(checkpoints.Items) != 1 {
		t.Fatalf("expected only the referenced checkpoint to exist, found %d", len(checkpoints.Items))
	}
	if checkpoints.Items[0].Name != friendlyCheckpointName {
		t.Fatalf("unexpected checkpoint %s", checkpoints.Items[0].Name)
	}
}

func TestDynamoGraphDeploymentReconciler_reconcileCheckpoints_checkpointRefUsesReadyReferencedCR(t *testing.T) {
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: record.NewFakeRecorder(10),
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	checkpointStatuses, checkpointInfos, err := reconciler.reconcileCheckpoints(ctx, dgd)
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if !info.Ready {
		t.Fatalf("expected referenced checkpoint to be ready")
	}
	if !info.Exists {
		t.Fatalf("expected referenced checkpoint to exist")
	}
	if info.Hash != hash {
		t.Fatalf("checkpoint hash = %s, want %s", info.Hash, hash)
	}
	if checkpointStatuses["worker"].CheckpointName != friendlyCheckpointName {
		t.Fatalf("checkpoint status name = %s, want friendly-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}
	if !checkpointStatuses["worker"].Ready {
		t.Fatalf("expected checkpoint status to be ready")
	}
}

func TestDynamoGraphDeploymentReconciler_reconcileCheckpoints_createsCheckpointStoragePVC(t *testing.T) {
	if err := v1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}

	ctx := context.Background()
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      friendlyCheckpointName,
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme.Scheme).
			WithObjects(referenced).
			WithStatusSubresource(referenced).
			Build(),
		Config: &configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{
				Storage: configv1alpha1.CheckpointStorageConfiguration{
					Type: configv1alpha1.CheckpointStorageTypePVC,
					PVC: configv1alpha1.CheckpointPVCConfig{
						PVCName:          "snapshot-pvc",
						BasePath:         "/checkpoints",
						Create:           true,
						Size:             "2Gi",
						StorageClassName: "efs-sc",
						AccessMode:       string(corev1.ReadWriteMany),
					},
				},
			},
		},
		Recorder: record.NewFakeRecorder(10),
	}

	ref := friendlyCheckpointName
	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled:       true,
						Mode:          v1alpha1.CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	})

	if _, _, err := reconciler.reconcileCheckpoints(ctx, dgd); err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	pvc := &corev1.PersistentVolumeClaim{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "snapshot-pvc", Namespace: "default"}, pvc); err != nil {
		t.Fatalf("expected checkpoint storage PVC to be created: %v", err)
	}
	storageRequest := pvc.Spec.Resources.Requests[corev1.ResourceStorage]
	if storageRequest.String() != "2Gi" {
		t.Fatalf("PVC storage request = %s, want 2Gi", storageRequest.String())
	}
	if pvc.Spec.StorageClassName == nil || *pvc.Spec.StorageClassName != "efs-sc" {
		t.Fatalf("PVC storageClassName = %v, want efs-sc", pvc.Spec.StorageClassName)
	}
	if len(pvc.Spec.AccessModes) != 1 || pvc.Spec.AccessModes[0] != corev1.ReadWriteMany {
		t.Fatalf("PVC accessModes = %v, want [ReadWriteMany]", pvc.Spec.AccessModes)
	}
}

func TestDynamoGraphDeploymentReconciler_reconcileCheckpoints_autoModeWaitsForExistingCreatingCheckpoint(t *testing.T) {
	ctx := context.Background()
	testScheme := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	if err != nil {
		t.Fatalf("Failed to compute checkpoint hash: %v", err)
	}

	existing := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "existing-worker-checkpoint",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "keep-existing:latest",
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseCreating,
			IdentityHash: hash,
		},
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(testScheme).
			WithObjects(existing).
			WithStatusSubresource(existing).
			Build(),
		Config:   &configv1alpha1.OperatorConfiguration{},
		Recorder: record.NewFakeRecorder(10),
	}

	dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: string(commonconsts.ComponentTypeWorker),
					Checkpoint: &v1alpha1.ServiceCheckpointConfig{
						Enabled: true,
						Mode:    v1alpha1.CheckpointModeAuto,
						Identity: &v1alpha1.DynamoCheckpointIdentity{
							Model:            identity.Model,
							BackendFramework: identity.BackendFramework,
						},
					},
					ExtraPodSpec: &v1alpha1.ExtraPodSpec{
						MainContainer: &corev1.Container{
							Name:  "main",
							Image: "new-writer:latest",
						},
					},
				},
			},
		},
	})

	checkpointStatuses, checkpointInfos, err := reconciler.reconcileCheckpoints(ctx, dgd)
	if err != nil {
		t.Fatalf("reconcileCheckpoints() error = %v", err)
	}

	info, ok := checkpointInfos["worker"]
	if !ok {
		t.Fatalf("expected checkpoint info for worker service")
	}
	if info.Ready {
		t.Fatalf("expected existing checkpoint to remain not ready")
	}
	if !info.Exists {
		t.Fatalf("expected existing checkpoint to be detected")
	}
	if info.Hash != hash {
		t.Fatalf("checkpoint hash = %s, want %s", info.Hash, hash)
	}
	if checkpointStatuses["worker"].CheckpointName != "existing-worker-checkpoint" {
		t.Fatalf("checkpoint status name = %s, want existing-worker-checkpoint", checkpointStatuses["worker"].CheckpointName)
	}

	updated := &v1alpha1.DynamoCheckpoint{}
	if err := reconciler.Get(ctx, types.NamespacedName{Name: "existing-worker-checkpoint", Namespace: "default"}, updated); err != nil {
		t.Fatalf("Failed to get checkpoint: %v", err)
	}
	if len(updated.Spec.Job.PodTemplateSpec.Spec.Containers) != 1 {
		t.Fatalf("expected one job container, got %d", len(updated.Spec.Job.PodTemplateSpec.Spec.Containers))
	}
	if updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image != "keep-existing:latest" {
		t.Fatalf("existing job image was mutated to %s", updated.Spec.Job.PodTemplateSpec.Spec.Containers[0].Image)
	}
}

// mockScaleClient implements scale.ScalesGetter for testing
type mockScaleClient struct{}

func (m *mockScaleClient) Scales(namespace string) scale.ScaleInterface {
	return &mockScaleInterface{}
}

// mockScaleInterface implements scale.ScaleInterface for testing
type mockScaleInterface struct{}

func (m *mockScaleInterface) Get(ctx context.Context, resource schema.GroupResource, name string, opts metav1.GetOptions) (*autoscalingv1.Scale, error) {
	// Return a dummy scale object - we don't actually need scaling in the test
	return &autoscalingv1.Scale{}, nil
}

func (m *mockScaleInterface) Update(ctx context.Context, resource schema.GroupResource, scale *autoscalingv1.Scale, opts metav1.UpdateOptions) (*autoscalingv1.Scale, error) {
	// Return success without actually doing anything
	return scale, nil
}

func (m *mockScaleInterface) Patch(ctx context.Context, gvr schema.GroupVersionResource, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions) (*autoscalingv1.Scale, error) {
	// Return a dummy scale object
	return &autoscalingv1.Scale{}, nil
}

func Test_reconcileGroveResources(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                   string
		dgdSpec                v1alpha1.DynamoGraphDeploymentSpec
		existingGroveResources []client.Object
		draEnabled             bool
		wantReconcileResult    ReconcileResult
		wantErrSubstring       string
	}{
		{
			name: "singular frontend service with 2 replicas - creates a PodClique with 2 replicas - ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           2,
						UpdatedReplicas:    2,
						ReadyReplicas:      2,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:   v1beta1.ComponentKindPodClique,
						ComponentNames:  []string{"test-dgd-0-frontend"},
						Replicas:        2,
						UpdatedReplicas: 2,
						ReadyReplicas:   ptr.To(int32(2)),
					},
				},
			},
		},
		{
			name: "frontend service with 1 replica, decode service with 2 replicas - 2 PodCliques - one unready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"decode": {
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           2,
						UpdatedReplicas:    1,
						ReadyReplicas:      1, // Only 1 ready, but 2 desired
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: Message("Resources not ready: test-dgd: podclique/test-dgd-0-decode: desired=2, ready=1"),
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:   v1beta1.ComponentKindPodClique,
						ComponentNames:  []string{"test-dgd-0-frontend"},
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:   v1beta1.ComponentKindPodClique,
						ComponentNames:  []string{"test-dgd-0-decode"},
						Replicas:        2,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
				},
			},
		},
		{
			name: "decode worker multinode (PCSG), prefill worker multinode (PCSG) - both ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"decode": {
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(1)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
					},
					"prefill": {
						ComponentType: string(commonconsts.ComponentTypeWorker),
						Replicas:      ptr.To(int32(1)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 4,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-decode",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-prefill",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"decode": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-decode"},
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-prefill"},
						Replicas:          1,
						UpdatedReplicas:   1,
						AvailableReplicas: ptr.To(int32(1)),
					},
				},
			},
		},
		{
			name: "frontend worker (PodClique), aggregated worker multinode (PCSG) - PCSG unready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"aggregated": {
						ComponentType: string(commonconsts.ComponentTypeWorker),
						Replicas:      ptr.To(int32(2)),
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 8,
						},
					},
				},
			},
			existingGroveResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-frontend",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
				&grovev1alpha1.PodCliqueScalingGroup{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-0-aggregated",
						Namespace: "default",
					},
					Spec: grovev1alpha1.PodCliqueScalingGroupSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueScalingGroupStatus{
						Replicas:           2,
						UpdatedReplicas:    2,
						AvailableReplicas:  1, // Only 1 available, but 2 desired
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: Message("Resources not ready: test-dgd: pcsg/test-dgd-0-aggregated: desired=2, available=1"),
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:   v1beta1.ComponentKindPodClique,
						ComponentNames:  []string{"test-dgd-0-frontend"},
						Replicas:        1,
						UpdatedReplicas: 1,
						ReadyReplicas:   ptr.To(int32(1)),
					},
					"aggregated": {
						ComponentKind:     v1beta1.ComponentKindPodCliqueScalingGroup,
						ComponentNames:    []string{"test-dgd-0-aggregated"},
						Replicas:          2,
						UpdatedReplicas:   2,
						AvailableReplicas: ptr.To(int32(1)),
					},
				},
			},
		},
		{
			name: "inter-pod GMS failover requires DRA - returns clear error when DRA is disabled",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"decode": {
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(1)),
						Failover: &v1alpha1.FailoverSpec{
							Enabled:    true,
							Mode:       v1alpha1.GMSModeInterPod,
							NumShadows: 1,
						},
					},
				},
			},
			draEnabled:       false,
			wantErrSubstring: "requires DRA",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := newDynamoGraphDeploymentControllerTestScheme(t)

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingGroveResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        fakeKubeClient,
				Recorder:      recorder,
				Config:        &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{DRAEnabled: tt.draEnabled},
				ScaleClient:   &mockScaleClient{},
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			result, err := reconciler.reconcileGroveResources(ctx, dgd, nil, nil)
			if tt.wantErrSubstring != "" {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(err.Error()).To(gomega.ContainSubstring(tt.wantErrSubstring))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())

			g.Expect(result).To(gomega.Equal(tt.wantReconcileResult))
		})
	}
}

func Test_reconcileGroveResources_UsesPreservedAlphaServiceIngress(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	className := "custom-nginx"
	alpha := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dgd",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Labels:           map[string]string{"graph-label": "kept"},
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: commonconsts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(1)),
					Labels:        map[string]string{"legacy-label": "kept"},
					Ingress: &v1alpha1.IngressSpec{
						Enabled:                    true,
						Host:                       "legacy-frontend",
						IngressControllerClassName: &className,
					},
				},
			},
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{}
	g.Expect(alpha.ConvertTo(dgd)).NotTo(gomega.HaveOccurred())

	s := scheme.Scheme
	g.Expect(v1alpha1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(v1beta1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(corev1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(networkingv1.AddToScheme(s)).NotTo(gomega.HaveOccurred())
	g.Expect(grovev1alpha1.AddToScheme(s)).NotTo(gomega.HaveOccurred())

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dgd).
		Build()

	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fakeKubeClient,
		Recorder:      record.NewFakeRecorder(100),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{},
		ScaleClient:   &mockScaleClient{},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
	}

	_, err := reconciler.reconcileGroveResources(ctx, dgd, nil, nil)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	ingress := &networkingv1.Ingress{}
	g.Expect(fakeKubeClient.Get(ctx, types.NamespacedName{Name: "test-dgd-frontend", Namespace: "default"}, ingress)).NotTo(gomega.HaveOccurred())
	g.Expect(ingress.Spec.IngressClassName).NotTo(gomega.BeNil())
	g.Expect(*ingress.Spec.IngressClassName).To(gomega.Equal(className))
	g.Expect(ingress.Spec.Rules).To(gomega.HaveLen(1))
	g.Expect(ingress.Spec.Rules[0].Host).To(gomega.Equal("legacy-frontend.local"))

	service := &corev1.Service{}
	g.Expect(fakeKubeClient.Get(ctx, types.NamespacedName{Name: "test-dgd-frontend", Namespace: "default"}, service)).NotTo(gomega.HaveOccurred())
	g.Expect(service.Labels["graph-label"]).To(gomega.Equal("kept"))
	g.Expect(service.Labels["legacy-label"]).To(gomega.Equal("kept"))
}

func TestDynamoGraphDeploymentReconciler_prepareGroveRenderDeployment_PreservesLegacyWorkerSelectors(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner",
			Namespace: "jsm",
			Annotations: map[string]string{
				commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1))},
				{ComponentName: "Planner", ComponentType: v1beta1.ComponentTypePlanner, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmDecodeWorker", ComponentType: v1beta1.ComponentTypeDecode, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmPrefillWorker", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner",
			Namespace: "jsm",
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "vllmprefillworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:        "VllmPrefillWorker",
							commonconsts.KubeLabelDynamoComponentType:    commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType: commonconsts.ComponentTypePrefill,
						},
						Annotations: map[string]string{
							commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
						},
					},
					{Name: "frontend"},
					{
						Name: "vllmdecodeworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:        "VllmDecodeWorker",
							commonconsts.KubeLabelDynamoComponentType:    commonconsts.ComponentTypeWorker,
							commonconsts.KubeLabelDynamoSubComponentType: commonconsts.ComponentTypeDecode,
						},
					},
				},
			},
		},
	}

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, existingPCS).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: fakeKubeClient}

	renderDGD, existing, err := reconciler.prepareGroveRenderDeployment(ctx, dgd)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(existing).NotTo(gomega.BeNil())
	g.Expect(dgd.GetComponentByName("VllmDecodeWorker").ComponentType).To(gomega.Equal(v1beta1.ComponentTypeDecode))

	prefill := renderDGD.GetComponentByName("VllmPrefillWorker")
	if prefill == nil {
		t.Fatal("expected rendered prefill component")
	}
	g.Expect(prefill.ComponentType).To(gomega.Equal(v1beta1.ComponentTypeWorker))
	g.Expect(prefill.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypePrefill))

	decode := renderDGD.GetComponentByName("VllmDecodeWorker")
	if decode == nil {
		t.Fatal("expected rendered decode component")
	}
	g.Expect(decode.ComponentType).To(gomega.Equal(v1beta1.ComponentTypeWorker))
	g.Expect(decode.PodTemplate.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypeDecode))

	generatedPCS, err := dynamo.GenerateGrovePodCliqueSet(ctx, renderDGD, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, fakeKubeClient, nil, nil, nil, nil)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	preserveGrovePodCliqueSetOrder(generatedPCS, existing)
	g.Expect(generatedPCS.Spec.Template.Cliques[0].Name).To(gomega.Equal("vllmprefillworker"))

	var prefillClique *grovev1alpha1.PodCliqueTemplateSpec
	for _, clique := range generatedPCS.Spec.Template.Cliques {
		if clique.Name == "vllmprefillworker" {
			prefillClique = clique
			break
		}
	}
	if prefillClique == nil {
		t.Fatal("expected rendered prefill clique")
	}
	g.Expect(prefillClique.Labels[commonconsts.KubeLabelDynamoComponentType]).To(gomega.Equal(commonconsts.ComponentTypeWorker))
	g.Expect(prefillClique.Labels[commonconsts.KubeLabelDynamoSubComponentType]).To(gomega.Equal(commonconsts.ComponentTypePrefill))
	g.Expect(prefillClique.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion]).To(gomega.Equal("1.1.0"))

	decodeService, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dynamo.GetDCDResourceName(renderDGD, "VllmDecodeWorker", ""),
		Namespace:       renderDGD.Namespace,
		ComponentType:   string(decode.ComponentType),
		DynamoNamespace: renderDGD.GetDynamoNamespaceForComponent(decode),
		ComponentName:   "VllmDecodeWorker",
		Labels:          dynamo.GetDGDComponentResourceLabels(renderDGD, "VllmDecodeWorker", decode),
		IsK8sDiscovery:  true,
	})
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(decodeService.Spec.Selector[commonconsts.KubeLabelDynamoComponentType]).To(gomega.Equal(commonconsts.ComponentTypeWorker))
}

func TestDynamoGraphDeploymentReconciler_prepareGroveRenderDeployment_KeepsNativeWorkerSelectors(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "prefill",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:     "prefill",
							commonconsts.KubeLabelDynamoComponentType: commonconsts.ComponentTypePrefill,
						},
					},
				},
			},
		},
	}

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, existingPCS).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: fakeKubeClient}

	renderDGD, _, err := reconciler.prepareGroveRenderDeployment(ctx, dgd)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	prefill := renderDGD.GetComponentByName("prefill")
	if prefill == nil {
		t.Fatal("expected rendered prefill component")
	}
	g.Expect(prefill.ComponentType).To(gomega.Equal(v1beta1.ComponentTypePrefill))
}

func Test_computeRestartStatus(t *testing.T) {
	ctx := context.Background()
	newID := "restart-1"
	oldID := "restart-0"

	tests := []struct {
		name              string
		dgdSpec           v1alpha1.DynamoGraphDeploymentSpec
		dgdStatus         v1alpha1.DynamoGraphDeploymentStatus
		existingResources []client.Object
		groveEnabled      bool
		wantRestartStatus *v1alpha1.RestartStatus
	}{
		{
			name: "no restart requested - returns nil",
		},
		{
			name: "no restart at time - returns nil",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{},
			},
		},
		{
			name: "no restart requested but has completed status - preserves status",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: oldID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "no restart requested but has restarting status - returns nil",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
				},
			},
		},
		{
			name: "restart already processed (completed) - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "restart already processed (failed) - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseFailed,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseFailed,
			},
		},
		{
			name: "parallel restart - all services complete (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "parallel restart - services still restarting (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:   "frontend",
						ComponentType: string(commonconsts.ComponentTypeFrontend),
						Replicas:      ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:   "decode",
						ComponentType: string(commonconsts.ComponentTypeDecode),
						Replicas:      ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend", "decode"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1, // Not yet caught up
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"decode"},
			},
		},
		{
			name: "sequential restart - first service starting",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "sequential restart - first service done, moving to second",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"decode"},
			},
		},
		{
			name: "sequential restart - all services complete",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeSequential,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "sequential restart - stale in-progress component resets to first service",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"removed"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-removed",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "default strategy (sequential) - no strategy specified",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "parallel restart with empty services - returns completed",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "sequential restart with empty services - returns completed",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeSequential,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "parallel restart - new request with ready resources should NOT complete immediately (race condition fix)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				// No existing restart status - brand new restart request
			},
			existingResources: []client.Object{
				// DCD is READY - simulating state BEFORE restart annotation is applied
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 1,
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting, // NOT Completed!
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "Grove pathway - parallel restart complete",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				&grovev1alpha1.PodCliqueSet{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd",
						Namespace:  "default",
						Generation: 1,
					},
					Status: grovev1alpha1.PodCliqueSetStatus{
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-0-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 1,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           1,
						UpdatedReplicas:    1,
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			groveEnabled: true,
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseCompleted,
			},
		},
		{
			name: "Grove pathway - sequential restart in progress",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(2)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
			},
			existingResources: []client.Object{
				&grovev1alpha1.PodClique{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-0-frontend",
						Namespace:  "default",
						Generation: 1,
					},
					Spec: grovev1alpha1.PodCliqueSpec{
						Replicas: 2,
					},
					Status: grovev1alpha1.PodCliqueStatus{
						Replicas:           2,
						UpdatedReplicas:    1, // Not fully updated
						ReadyReplicas:      1,
						ObservedGeneration: ptr.To(int64(1)),
					},
				},
			},
			groveEnabled: true,
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "parallel restart - new restart request during ongoing restart resets to all services (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID, // NEW timestamp
					Strategy: &v1alpha1.RestartStrategy{
						Type: v1alpha1.RestartStrategyTypeParallel,
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
					"completed": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend", "decode"}, // completed service already done
				},
			},
			existingResources: []client.Object{
				// All services are now ready (simulating state after new restart timestamp is applied)
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-completed",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionFalse},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"completed", "decode", "frontend"}, // ALL services, sorted
			},
		},
		{
			name: "sequential restart - new restart request during ongoing restart resets to first service (DCD pathway)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID, // NEW timestamp
					Strategy: &v1alpha1.RestartStrategy{
						Type:  v1alpha1.RestartStrategyTypeSequential,
						Order: []string{"frontend", "decode", "worker"},
					},
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
					"decode": {
						Replicas: ptr.To(int32(1)),
					},
					"worker": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"decode"},
				},
			},
			existingResources: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-frontend",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-decode",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "test-dgd-worker",
						Namespace:  "default",
						Generation: 2,
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						ObservedGeneration: 2,
						Conditions: []metav1.Condition{
							{Type: v1alpha1.DynamoGraphDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue},
						},
					},
				}),
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"}, // Reset to FIRST service
			},
		},
		{
			name: "rolling update in progress + new restart request - superseded",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhaseInProgress,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "rolling update pending + restart already in progress - superseded",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseRestarting,
					InProgress: []string{"frontend"},
				},
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhasePending,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "rolling update completed + restart request - normal processing",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				RollingUpdate: &v1alpha1.RollingUpdateStatus{
					Phase: v1alpha1.RollingUpdatePhaseCompleted,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "restart already processed as superseded - returns existing status",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				Restart: &v1alpha1.Restart{
					ID: newID,
				},
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						Replicas: ptr.To(int32(1)),
					},
				},
			},
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: newID,
					Phase:      v1alpha1.RestartPhaseSuperseded,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: newID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
		{
			name: "no restart requested but has superseded status - preserves status",
			dgdStatus: v1alpha1.DynamoGraphDeploymentStatus{
				Restart: &v1alpha1.RestartStatus{
					ObservedID: oldID,
					Phase:      v1alpha1.RestartPhaseSuperseded,
				},
			},
			wantRestartStatus: &v1alpha1.RestartStatus{
				ObservedID: oldID,
				Phase:      v1alpha1.RestartPhaseSuperseded,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = grovev1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec:   tt.dgdSpec,
				Status: tt.dgdStatus,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingResources...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   fakeKubeClient,
				Recorder: recorder,
				Config:   &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{
					GroveEnabled: tt.groveEnabled,
				},
			}

			result := reconciler.computeRestartStatus(ctx, dgd)

			if tt.wantRestartStatus == nil {
				g.Expect(result).To(gomega.BeNil())
				return
			}

			g.Expect(result).NotTo(gomega.BeNil())
			g.Expect(result).To(gomega.Equal(betaRestartStatus(tt.wantRestartStatus)))
		})
	}
}

func Test_reconcileDynamoComponentsDeployments(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                string
		dgdSpec             v1alpha1.DynamoGraphDeploymentSpec
		existingDCDs        []client.Object
		wantReconcileResult ReconcileResult
	}{
		{
			name: "single service - DCD ready (Available condition = True)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
				},
			},
		},
		{
			name: "single service - DCD not ready (Available condition = False)",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-frontend: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
				},
			},
		},
		{
			name: "multiple services - all DCDs ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(3)),
					},
				},
			},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-e1f2a6fe",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-e1f2a6fe-deployment"},
							Replicas:          2,
							UpdatedReplicas:   2,
							ReadyReplicas:     ptr.To(int32(2)),
							AvailableReplicas: ptr.To(int32(2)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill-e1f2a6fe",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-prefill-e1f2a6fe-deployment"},
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStateSuccessful,
				Reason:  "all_resources_are_ready",
				Message: "All resources are ready",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-e1f2a6fe-deployment"},
						Replicas:          2,
						UpdatedReplicas:   2,
						ReadyReplicas:     ptr.To(int32(2)),
						AvailableReplicas: ptr.To(int32(2)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-prefill-e1f2a6fe-deployment"},
						Replicas:          3,
						UpdatedReplicas:   3,
						ReadyReplicas:     ptr.To(int32(3)),
						AvailableReplicas: ptr.To(int32(3)),
					},
				},
			},
		},
		{
			name: "multiple services - some DCDs ready, some not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
					"prefill": {
						ServiceName:     "prefill",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypePrefill),
						Replicas:        ptr.To(int32(3)),
					},
				},
			},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(1)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-e1f2a6fe",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-e1f2a6fe-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-prefill-e1f2a6fe",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "prefill",
							Replicas:    ptr.To(int32(3)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionTrue,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-prefill-e1f2a6fe-deployment"},
							Replicas:          3,
							UpdatedReplicas:   3,
							ReadyReplicas:     ptr.To(int32(3)),
							AvailableReplicas: ptr.To(int32(3)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode-e1f2a6fe: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(1)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-e1f2a6fe-deployment"},
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"prefill": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-prefill-e1f2a6fe-deployment"},
						Replicas:          3,
						UpdatedReplicas:   3,
						ReadyReplicas:     ptr.To(int32(3)),
						AvailableReplicas: ptr.To(int32(3)),
					},
				},
			},
		},
		{
			name: "multiple services - all DCDs not ready",
			dgdSpec: v1alpha1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"frontend": {
						ServiceName:     "frontend",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeFrontend),
						Replicas:        ptr.To(int32(1)),
					},
					"decode": {
						ServiceName:     "decode",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(2)),
					},
				},
			},
			existingDCDs: []client.Object{
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-frontend",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "frontend",
							Replicas:    ptr.To(int32(1)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-frontend-deployment"},
							Replicas:          1,
							UpdatedReplicas:   0,
							ReadyReplicas:     ptr.To(int32(0)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
				betaDCD(t, &v1alpha1.DynamoComponentDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dgd-decode-5f3d46ba",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: "vllm",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "decode",
							Replicas:    ptr.To(int32(2)),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions: []metav1.Condition{
							{
								Type:   v1alpha1.DynamoGraphDeploymentConditionTypeAvailable,
								Status: metav1.ConditionFalse,
							},
						},
						Service: &v1alpha1.ServiceReplicaStatus{
							ComponentKind:     v1alpha1.ComponentKindDeployment,
							ComponentNames:    []string{"test-dgd-decode-5f3d46ba-deployment"},
							Replicas:          2,
							UpdatedReplicas:   1,
							ReadyReplicas:     ptr.To(int32(1)),
							AvailableReplicas: ptr.To(int32(0)),
						},
					},
				}),
			},
			wantReconcileResult: ReconcileResult{
				State:   v1beta1.DGDStatePending,
				Reason:  "some_resources_are_not_ready",
				Message: "Resources not ready: test-dgd-decode-5f3d46ba: Component deployment not ready - Available condition not true; test-dgd-frontend: Component deployment not ready - Available condition not true",
				ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
					"frontend": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-frontend-deployment"},
						Replicas:          1,
						UpdatedReplicas:   0,
						ReadyReplicas:     ptr.To(int32(0)),
						AvailableReplicas: ptr.To(int32(0)),
					},
					"decode": {
						ComponentKind:     v1beta1.ComponentKindDeployment,
						ComponentNames:    []string{"test-dgd-decode-5f3d46ba-deployment"},
						Replicas:          2,
						UpdatedReplicas:   1,
						ReadyReplicas:     ptr.To(int32(1)),
						AvailableReplicas: ptr.To(int32(0)),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
				Spec: tt.dgdSpec,
			})

			var objects []client.Object
			objects = append(objects, dgd)
			objects = append(objects, tt.existingDCDs...)

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        fakeKubeClient,
				Recorder:      recorder,
				Config:        &configv1alpha1.OperatorConfiguration{},
				RuntimeConfig: &controller_common.RuntimeConfig{},
			}

			result, err := reconciler.reconcileDynamoComponentsDeployments(ctx, dgd, nil)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			g.Expect(result).To(gomega.Equal(tt.wantReconcileResult))
		})
	}
}

func TestPropagateTopologyCondition(t *testing.T) {
	tests := []struct {
		name           string
		dgd            *v1beta1.DynamoGraphDeployment
		pcs            *grovev1alpha1.PodCliqueSet
		groveEnabled   bool
		wantCondition  bool
		wantStatus     metav1.ConditionStatus
		wantReason     string
		wantEventCount int
	}{
		{
			name: "no topology constraints - no condition added",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
				},
			}),
			groveEnabled:  true,
			wantCondition: false,
		},
		{
			name: "topology set but Grove not enabled - no condition added",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test", Namespace: "default",
					Annotations: map[string]string{commonconsts.KubeAnnotationEnableGrove: "false"},
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			groveEnabled:  false,
			wantCondition: false,
		},
		{
			name: "topology set, PCS has no topology condition - unknown",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status:     grovev1alpha1.PodCliqueSetStatus{},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionUnknown,
			wantReason:    v1alpha1.ConditionReasonTopologyConditionPending,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=True with ClusterTopologyLevelsUnavailable",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionTrue,
							Reason:  groveconstants.ConditionReasonTopologyLevelsUnavailable,
							Message: "Topology level 'rack' is no longer available",
						},
					},
				},
			},
			groveEnabled:   true,
			wantCondition:  true,
			wantStatus:     metav1.ConditionFalse,
			wantReason:     v1alpha1.ConditionReasonTopologyLevelsUnavailable,
			wantEventCount: 1,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=True with ClusterTopologyNotFound",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionTrue,
							Reason:  groveconstants.ConditionReasonClusterTopologyNotFound,
							Message: "ClusterTopology 'default' not found",
						},
					},
				},
			},
			groveEnabled:   true,
			wantCondition:  true,
			wantStatus:     metav1.ConditionFalse,
			wantReason:     v1alpha1.ConditionReasonTopologyDefinitionNotFound,
			wantEventCount: 1,
		},
		{
			name: "PCS reports TopologyLevelsUnavailable=False - all levels available",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status: grovev1alpha1.PodCliqueSetStatus{
					Conditions: []metav1.Condition{
						{
							Type:    groveconstants.ConditionTopologyLevelsUnavailable,
							Status:  metav1.ConditionFalse,
							Reason:  groveconstants.ConditionReasonAllTopologyLevelsAvailable,
							Message: "All topology levels available",
						},
					},
				},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionTrue,
			wantReason:    v1alpha1.ConditionReasonAllTopologyLevelsAvailable,
		},
		{
			name: "service-only topology constraint triggers condition propagation",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology"},
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							TopologyConstraint: &v1alpha1.TopologyConstraint{PackDomain: v1alpha1.TopologyDomain("rack")},
						},
					},
				},
			}),
			pcs: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Status:     grovev1alpha1.PodCliqueSetStatus{},
			},
			groveEnabled:  true,
			wantCondition: true,
			wantStatus:    metav1.ConditionUnknown,
			wantReason:    v1alpha1.ConditionReasonTopologyConditionPending,
		},
		{
			name: "PCS not found yet - no condition added",
			dgd: betaDGD(t, &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					TopologyConstraint: &v1alpha1.SpecTopologyConstraint{TopologyProfile: "test-topology", PackDomain: v1alpha1.TopologyDomain("rack")},
				},
			}),
			pcs:           nil,
			groveEnabled:  true,
			wantCondition: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = grovev1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			objs := []client.Object{}
			if tt.pcs != nil {
				objs = append(objs, tt.pcs)
			}

			fakeClient := fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).Build()
			recorder := record.NewFakeRecorder(10)

			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   fakeClient,
				Recorder: recorder,
				RuntimeConfig: &controller_common.RuntimeConfig{
					GroveEnabled: tt.groveEnabled,
				},
			}

			ctx := context.Background()
			reconciler.propagateTopologyCondition(ctx, tt.dgd)

			var topoCond *metav1.Condition
			for i := range tt.dgd.Status.Conditions {
				if tt.dgd.Status.Conditions[i].Type == v1alpha1.ConditionTypeTopologyLevelsAvailable {
					topoCond = &tt.dgd.Status.Conditions[i]
					break
				}
			}

			if !tt.wantCondition {
				g.Expect(topoCond).To(gomega.BeNil(), "expected no TopologyLevelsAvailable condition")
				return
			}

			g.Expect(topoCond).NotTo(gomega.BeNil(), "expected TopologyLevelsAvailable condition to be set")
			g.Expect(topoCond.Status).To(gomega.Equal(tt.wantStatus))
			g.Expect(topoCond.Reason).To(gomega.Equal(tt.wantReason))

			close(recorder.Events)
			eventCount := 0
			for range recorder.Events {
				eventCount++
			}
			g.Expect(eventCount).To(gomega.Equal(tt.wantEventCount))
		})
	}
}

func TestMapPodCliqueScalingGroupToRequests(t *testing.T) {
	// Register Grove types with the scheme so fake client can handle them
	if err := grovev1alpha1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("Failed to add grovev1alpha1 to scheme: %v", err)
	}

	tests := []struct {
		name         string
		obj          client.Object
		existingPCS  *grovev1alpha1.PodCliqueSet // PCS object that exists in the cluster
		wantRequests int
		wantName     string
		wantNs       string
	}{
		{
			name: "PCSG with PodCliqueSet controller ownerRef returns DGD request",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dynamo-recipe-0-worker",
					Namespace: "mwieczorek-dsv32-trtllm-agg",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "dynamo-recipe",
							Controller: ptr.To(true),
						},
					},
				},
			},
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dynamo-recipe",
					Namespace: "mwieczorek-dsv32-trtllm-agg",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: "dynamo-recipe",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: v1alpha1.GroupVersion.String(),
							Kind:       "DynamoGraphDeployment",
							Name:       "dynamo-recipe",
							Controller: ptr.To(true),
						},
					},
				},
			},
			wantRequests: 1,
			wantName:     "dynamo-recipe",
			wantNs:       "mwieczorek-dsv32-trtllm-agg",
		},
		{
			name: "PCSG with truncated PCS name resolves to original DGD name",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "truncated-pcs-0-worker",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "truncated-pcs",
							Controller: ptr.To(true),
						},
					},
				},
			},
			existingPCS: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "truncated-pcs",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: "my-very-long-original-dgd-name",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: v1alpha1.GroupVersion.String(),
							Kind:       "DynamoGraphDeployment",
							Name:       "my-very-long-original-dgd-name",
							Controller: ptr.To(true),
						},
					},
				},
			},
			wantRequests: 1,
			wantName:     "my-very-long-original-dgd-name",
			wantNs:       "default",
		},
		{
			name: "PCSG with no ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "orphan-pcsg",
					Namespace: "default",
				},
			},
			wantRequests: 0,
		},
		{
			name: "PCSG with non-controller PodCliqueSet ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pcsg-with-non-controller-ref",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: grovev1alpha1.SchemeGroupVersion.String(),
							Kind:       "PodCliqueSet",
							Name:       "some-pcs",
							// Controller flag omitted: metav1.GetControllerOf must ignore this ref.
						},
					},
				},
			},
			wantRequests: 0,
		},
		{
			name: "PCSG with non-PodCliqueSet ownerRef returns no requests",
			obj: &grovev1alpha1.PodCliqueScalingGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "weird-pcsg",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "not-a-pcs",
						},
					},
				},
			},
			wantRequests: 0,
		},
		{
			name:         "non-PCSG object returns no requests",
			obj:          &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"}},
			wantRequests: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Build fake client with existing PCS if provided
			builder := fake.NewClientBuilder().WithScheme(scheme.Scheme)
			if tt.existingPCS != nil {
				builder = builder.WithObjects(tt.existingPCS)
			}
			r := &DynamoGraphDeploymentReconciler{
				Client: builder.Build(),
			}
			reqs := r.mapPodCliqueScalingGroupToRequests(context.Background(), tt.obj)

			g.Expect(reqs).To(gomega.HaveLen(tt.wantRequests))
			if tt.wantRequests == 1 {
				g.Expect(reqs[0].Name).To(gomega.Equal(tt.wantName))
				g.Expect(reqs[0].Namespace).To(gomega.Equal(tt.wantNs))
			}
		})
	}
}
