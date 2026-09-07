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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestDGDPVCReconciler_Reconcile(t *testing.T) {
	t.Run("native beta DGD is a no-op", func(t *testing.T) {
		t.Log("Build a native beta DGD without preserved alpha PVCs")
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		dgd := &v1beta1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "native", Namespace: "default"},
		}
		fakeClient := fake.NewClientBuilder().
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithObjects(dgd).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		t.Log("Reconcile compatibility PVCs")
		g.Expect(newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)).NotTo(gomega.HaveOccurred())

		t.Log("Verify no PVC was created")
		pvcs := &corev1.PersistentVolumeClaimList{}
		g.Expect(fakeClient.List(ctx, pvcs, client.InNamespace("default"))).NotTo(gomega.HaveOccurred())
		g.Expect(pvcs.Items).To(gomega.BeEmpty())
	})

	t.Run("missing preserved PVC with creation disabled returns an error", func(t *testing.T) {
		t.Log("Build a converted alpha DGD that references a missing non-creatable PVC")
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		create := false
		pvcName := VolumeNameModelCache
		dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "converted", Namespace: "default"},
			Spec: v1alpha1.DynamoGraphDeploymentSpec{
				PVCs: []v1alpha1.PVC{{
					Create: &create,
					Name:   &pvcName,
				}},
			},
		})
		fakeClient := fake.NewClientBuilder().
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithObjects(dgd).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		t.Log("Reconcile compatibility PVCs")
		err := newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)

		t.Log("Verify the missing PVC is reported without creating one")
		g.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(
			"does not exist and create is not enabled",
		)))
		pvcs := &corev1.PersistentVolumeClaimList{}
		g.Expect(fakeClient.List(ctx, pvcs, client.InNamespace("default"))).NotTo(gomega.HaveOccurred())
		g.Expect(pvcs.Items).To(gomega.BeEmpty())
	})

	t.Run("converted alpha DGD creates preserved top-level PVC", func(t *testing.T) {
		t.Log("Build a converted alpha DGD with a preserved top-level PVC")
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		create := true
		pvcName := VolumeNameModelCache
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
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithObjects(dgd).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		t.Log("Reconcile compatibility PVCs")
		g.Expect(newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)).NotTo(gomega.HaveOccurred())

		t.Log("Verify the desired PVC and owner reference")
		pvc := &corev1.PersistentVolumeClaim{}
		g.Expect(fakeClient.Get(ctx, types.NamespacedName{Name: pvcName, Namespace: "default"}, pvc)).NotTo(gomega.HaveOccurred())
		g.Expect(pvc.Spec.AccessModes).To(gomega.Equal([]corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce}))
		g.Expect(pvc.Spec.StorageClassName).NotTo(gomega.BeNil())
		g.Expect(*pvc.Spec.StorageClassName).To(gomega.Equal("standard"))
		gotStorage := pvc.Spec.Resources.Requests[corev1.ResourceStorage]
		g.Expect(gotStorage.Cmp(storage)).To(gomega.Equal(0))
		g.Expect(metav1.IsControlledBy(pvc, dgd)).To(gomega.BeTrue())
	})

	t.Run("cached not found followed by already exists converges", func(t *testing.T) {
		t.Log("Build a converted alpha DGD and simulate a stale cached PVC read")
		g := gomega.NewGomegaWithT(t)
		ctx := context.Background()
		create := true
		pvcName := VolumeNameModelCache
		dgd := betaDGD(t, &v1alpha1.DynamoGraphDeployment{
			ObjectMeta: metav1.ObjectMeta{Name: "converted", Namespace: "default"},
			Spec: v1alpha1.DynamoGraphDeploymentSpec{
				PVCs: []v1alpha1.PVC{{
					Create: &create,
					Name:   &pvcName,
					Size:   resource.MustParse("5Gi"),
				}},
			},
		})
		fakeClient := fake.NewClientBuilder().
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithObjects(dgd).
			WithInterceptorFuncs(interceptor.Funcs{
				Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
					if _, ok := obj.(*corev1.PersistentVolumeClaim); ok {
						return apierrors.NewAlreadyExists(
							schema.GroupResource{Resource: "persistentvolumeclaims"},
							obj.GetName(),
						)
					}
					return c.Create(ctx, obj, opts...)
				},
			}).
			Build()
		reconciler := &DynamoGraphDeploymentReconciler{Client: fakeClient}

		t.Log("Reconcile compatibility PVCs")
		err := newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)

		t.Log("Verify the create race is treated as converged pending cache observation")
		g.Expect(err).NotTo(gomega.HaveOccurred())
	})
}
