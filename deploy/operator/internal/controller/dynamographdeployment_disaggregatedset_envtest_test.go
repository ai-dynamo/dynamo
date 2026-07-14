/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("DisaggregatedSet", func() {
	It("creates a DisaggregatedSet with two roles", func() {
		ctx := context.Background()

		dgd := newDSHappyPathDGD()
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() {
			_ = k8sClient.Delete(ctx, dgd)
		})

		reconciler := &DynamoGraphDeploymentReconciler{
			Client:        k8sClient,
			Recorder:      record.NewFakeRecorder(10),
			Config:        &configv1alpha1.OperatorConfiguration{},
			RuntimeConfig: newTestRuntimeConfig(true),
		}

		res, err := reconciler.reconcileDisaggregatedSetResources(ctx, dgd, nil, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.State).To(Equal(nvidiacomv1beta1.DGDStatePending))

		ds := &unstructured.Unstructured{}
		ds.SetGroupVersionKind(disaggregatedSetGVK)
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)).To(Succeed())

		roles, found, err := unstructured.NestedSlice(ds.Object, "spec", "roles")
		Expect(err).NotTo(HaveOccurred())
		Expect(found).To(BeTrue())
		Expect(roles).To(HaveLen(2))
	})
})

func newDSHappyPathDGD() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-ds",
			Namespace: "default",
			UID:       "demo-ds-uid",
			Annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "prefill",
					ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:   dsTestPodTemplate(),
				},
				{
					ComponentName: "decode",
					ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:   dsTestPodTemplate(),
				},
			},
		},
	}
}

func dsTestPodTemplate() *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    "main",
				Image:   "busybox:1.36",
				Command: []string{"sh"},
				Args:    []string{"-c", "sleep 3600"},
			}},
		},
	}
}
