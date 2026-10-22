//go:build !clustertest

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
	"strconv"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("DynamoGraphDeploymentRequest v1beta2 controller", func() {
	var reconciler *DynamoGraphDeploymentRequestV1beta2Reconciler

	BeforeEach(func() {
		reconciler = &DynamoGraphDeploymentRequestV1beta2Reconciler{Client: k8sClient}
	})

	reconcile := func(ctx context.Context, key types.NamespacedName) {
		_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: key})
		Expect(err).NotTo(HaveOccurred())
	}

	It("Should create one Run per generation through the API server", func() {
		ctx := context.Background()
		dgdr := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
			ObjectMeta: metav1.ObjectMeta{Name: "v1beta2-run-per-generation", Namespace: envtestNamespace},
			Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
				NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
					Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-32B"},
				},
			},
		}

		By("Creating a native v1beta2 request; admission supplies the image and the hardware")
		Expect(k8sClient.Create(ctx, dgdr)).Should(Succeed())
		defer func() { _ = k8sClient.Delete(ctx, dgdr) }()
		key := client.ObjectKeyFromObject(dgdr)

		By("Reconciling creates the Run of generation 1")
		reconcile(ctx, key)
		live := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
		Expect(k8sClient.Get(ctx, key, live)).Should(Succeed())
		firstName := dgdr.Name + "-" + strconv.FormatInt(live.Generation, 10)
		first := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: envtestNamespace, Name: firstName}, first)).Should(Succeed())
		Expect(metav1.IsControlledBy(first, live)).To(BeTrue())
		Expect(first.Spec.Hardware).NotTo(BeNil())
		Expect(first.Spec.AISimulate.Image).To(Equal(controllerAISimulateImage))
		Expect(first.Spec.AISimulate.VersionOverride).To(Equal("1.2.0"))
		Expect(first.Spec.AISimulate.Parameters).NotTo(BeNil())

		By("Reading the status the controller wrote through the status subresource")
		Expect(live.Status.ObservedGeneration).To(Equal(live.Generation))
		Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		Expect(live.Status.ActiveRunRef.Name).To(Equal(firstName))
		condition := meta.FindStatusCondition(live.Status.Conditions, DGDRReconciledConditionV1beta2)
		Expect(condition).NotTo(BeNil())
		Expect(condition.Status).To(Equal(metav1.ConditionTrue))
		Expect(live.Status.V1Beta1).To(BeNil())

		By("Reconciling again creates nothing")
		reconcile(ctx, key)
		runs := &nvidiacomv1beta2.DynamoGraphDeploymentRunList{}
		Expect(k8sClient.List(ctx, runs, client.InNamespace(envtestNamespace))).Should(Succeed())
		Expect(runs.Items).To(HaveLen(1))

		By("Changing the spec while the Run is active starts nothing and says why")
		Expect(k8sClient.Get(ctx, key, live)).Should(Succeed())
		live.Spec.Model.Name = "Qwen/Qwen3-30B-A3B"
		Expect(k8sClient.Update(ctx, live)).Should(Succeed())
		reconcile(ctx, key)
		Expect(k8sClient.List(ctx, runs, client.InNamespace(envtestNamespace))).Should(Succeed())
		Expect(runs.Items).To(HaveLen(1))
		Expect(k8sClient.Get(ctx, key, live)).Should(Succeed())
		Expect(live.Generation).To(BeNumerically(">", 1))
		condition = meta.FindStatusCondition(live.Status.Conditions, DGDRReconciledConditionV1beta2)
		Expect(condition).NotTo(BeNil())
		Expect(condition.Reason).To(Equal(dgdrReasonWaitingForActiveRun))

		By("Finishing the first Run lets the next generation start")
		Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(first), first)).Should(Succeed())
		first.Status.Phase = nvidiacomv1beta2.DGDRRunPhase(dgdrRunPhaseSucceeded)
		Expect(k8sClient.Status().Update(ctx, first)).Should(Succeed())
		reconcile(ctx, key)

		Expect(k8sClient.Get(ctx, key, live)).Should(Succeed())
		secondName := dgdr.Name + "-" + strconv.FormatInt(live.Generation, 10)
		Expect(live.Status.ObservedGeneration).To(Equal(live.Generation))
		Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		Expect(live.Status.ActiveRunRef.Name).To(Equal(secondName))
		Expect(live.Status.RecentRunRefs).To(HaveLen(1))
		Expect(live.Status.RecentRunRefs[0].Name).To(Equal(firstName))
		second := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Namespace: envtestNamespace, Name: secondName}, second)).Should(Succeed())
		Expect(second.Spec.Model.Name).To(Equal("Qwen/Qwen3-30B-A3B"))
	})
})
