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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDGDPVCReconciler_Reconcile(t *testing.T) {
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

		g.Expect(newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)).NotTo(gomega.HaveOccurred())

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

		g.Expect(newDGDPVCReconciler(newTestDGDResourceSyncer(reconciler)).Reconcile(ctx, dgd)).NotTo(gomega.HaveOccurred())

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
