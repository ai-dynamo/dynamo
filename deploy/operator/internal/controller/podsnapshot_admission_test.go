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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
)

// The PodReference.Containers cap (MinItems=1/MaxItems=1, Required) generates OpenAPI array
// validation that only a real apiserver enforces. The fake client stores whatever it is handed, so
// these length 0/2/missing rejection cases live in envtest and drive the suite's global client
// directly (no reconciler needed).
var _ = Describe("PodSnapshot containers admission", func() {
	// newPodSnapshot builds a minimal-valid PodSnapshot with the given containers slice.
	newPodSnapshot := func(name string, containers []string) *nvidiacomv1alpha1.PodSnapshot {
		return &nvidiacomv1alpha1.PodSnapshot{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: envtestNamespace},
			Spec: nvidiacomv1alpha1.PodSnapshotSpec{
				Source: nvidiacomv1alpha1.PodSnapshotSource{
					PodRef: nvidiacomv1alpha1.PodReference{Name: "worker-0", Containers: containers},
				},
			},
		}
	}

	It("rejects containers of length 0 (minItems)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshot("snap-len0", []string{}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects containers of length 2 (maxItems)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshot("snap-len2", []string{"main", "sidecar"}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("accepts containers of length 1", func() {
		snap := newPodSnapshot("snap-len1", []string{"main"})
		Expect(k8sClient.Create(context.Background(), snap)).To(Succeed())
		Expect(k8sClient.Delete(context.Background(), snap)).To(Succeed())
	})

	It("rejects a blank container name (items minLength)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshot("snap-blank", []string{""}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects a non-DNS-1123 container name (items pattern)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshot("snap-badname", []string{"Bad_Name"}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects missing containers (required)", func() {
		// Build the object with the containers key genuinely absent (a nil slice would
		// serialize as "containers": null, which exercises type validation, not the
		// required check). This isolates the +kubebuilder:validation:Required marker.
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(nvidiacomv1alpha1.GroupVersion.WithKind("PodSnapshot"))
		u.SetName("snap-missing")
		u.SetNamespace(envtestNamespace)
		Expect(unstructured.SetNestedField(u.Object, "worker-0", "spec", "source", "podRef", "name")).To(Succeed())

		err := k8sClient.Create(context.Background(), u)
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})
})

var _ = Describe("PodSnapshotContent containers admission", func() {
	// newPodSnapshotContent builds a minimal-valid cluster-scoped PodSnapshotContent with the given
	// containers slice.
	newPodSnapshotContent := func(name string, containers []string) *nvidiacomv1alpha1.PodSnapshotContent {
		return &nvidiacomv1alpha1.PodSnapshotContent{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: nvidiacomv1alpha1.PodSnapshotContentSpec{
				PodSnapshotRef: nvidiacomv1alpha1.PodSnapshotReference{
					Namespace: "default", Name: "snap-a", UID: types.UID("snap-uid"),
				},
				Source: nvidiacomv1alpha1.PodSnapshotContentSource{
					PodRef:   nvidiacomv1alpha1.PodReference{Name: "worker-0", Containers: containers},
					NodeName: "node-a",
				},
			},
		}
	}

	It("rejects containers of length 0 (minItems)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshotContent("content-len0", []string{}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects containers of length 2 (maxItems)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshotContent("content-len2", []string{"main", "sidecar"}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("accepts containers of length 1", func() {
		content := newPodSnapshotContent("content-len1", []string{"main"})
		Expect(k8sClient.Create(context.Background(), content)).To(Succeed())
		Expect(k8sClient.Delete(context.Background(), content)).To(Succeed())
	})

	It("rejects a blank container name (items minLength)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshotContent("content-blank", []string{""}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects a non-DNS-1123 container name (items pattern)", func() {
		err := k8sClient.Create(context.Background(), newPodSnapshotContent("content-badname", []string{"Bad_Name"}))
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})

	It("rejects missing containers (required)", func() {
		// Omit the containers key entirely (not a nil slice) so this isolates the required
		// marker rather than falling through to null/type validation.
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(nvidiacomv1alpha1.GroupVersion.WithKind("PodSnapshotContent"))
		u.SetName("content-missing")
		Expect(unstructured.SetNestedField(u.Object, "default", "spec", "snapshotRef", "namespace")).To(Succeed())
		Expect(unstructured.SetNestedField(u.Object, "snap-a", "spec", "snapshotRef", "name")).To(Succeed())
		Expect(unstructured.SetNestedField(u.Object, "worker-0", "spec", "source", "podRef", "name")).To(Succeed())
		Expect(unstructured.SetNestedField(u.Object, "node-a", "spec", "source", "nodeName")).To(Succeed())

		err := k8sClient.Create(context.Background(), u)
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsInvalid(err)).To(BeTrue(), "expected an Invalid admission error, got %v", err)
		Expect(err.Error()).To(ContainSubstring("containers"))
	})
})
