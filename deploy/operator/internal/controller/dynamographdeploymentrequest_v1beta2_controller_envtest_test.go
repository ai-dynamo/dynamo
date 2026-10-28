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
	"testing"
	"time"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// This test runs the controller under a real manager, so only the watch events (a Request edit, a Run
// status update, a deleted Run) can make it reconcile. It fails if SetupWithManager loses the Owns
// watch, an event predicate drops Run status updates, or the owner mapping breaks. It also covers the
// API server and admission side: the Request is created model-only and admission supplies the image
// and the hardware.
func TestDGDRV1beta2ControllerReactsToRequestAndRunEvents(t *testing.T) {
	g := NewWithT(t)
	env := sharedEnv.ForTest(t)
	c := env.Client()
	ctx := context.Background()

	env.StartManager(func(mgr ctrl.Manager) error {
		return (&DynamoGraphDeploymentRequestV1beta2Reconciler{
			Client:        mgr.GetClient(),
			APIReader:     mgr.GetAPIReader(),
			Config:        env.OperatorConfig(),
			RuntimeConfig: env.RuntimeConfig(),
		}).SetupWithManager(mgr)
	})

	const timeout = 30 * time.Second
	const poll = 200 * time.Millisecond

	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		Name: "watched", Namespace: env.Namespace(),
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
			NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
				Model: &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-32B"},
			},
		},
	}
	key := client.ObjectKeyFromObject(request)
	getRequest := func(g Gomega) *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
		live := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}
		g.Expect(c.Get(ctx, key, live)).To(Succeed())
		return live
	}
	accepted := func(live *nvidiacomv1beta2.DynamoGraphDeploymentRequest) *metav1.Condition {
		return meta.FindStatusCondition(live.Status.Conditions, nvidiacomv1beta2.DGDRRequestConditionAccepted)
	}
	getRun := func(g Gomega, name string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
		run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
		g.Expect(c.Get(ctx, types.NamespacedName{Namespace: env.Namespace(), Name: name}, run)).To(Succeed())
		return run
	}

	t.Log("Creating the Request produces its first Run without a direct Reconcile call")
	g.Expect(c.Create(ctx, request)).To(Succeed())
	var firstName string
	g.Eventually(func(g Gomega) {
		live := getRequest(g)
		g.Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		g.Expect(accepted(live)).NotTo(BeNil())
		g.Expect(accepted(live).Status).To(Equal(metav1.ConditionTrue))
		g.Expect(live.Status.ObservedGeneration).To(Equal(live.Generation))
		firstName = live.Status.ActiveRunRef.Name
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())

	first := getRun(g, firstName)
	g.Expect(metav1.IsControlledBy(first, getRequest(g))).To(BeTrue())
	g.Expect(first.Spec.Hardware).NotTo(BeNil())
	g.Expect(first.Spec.AISimulate.Image).To(Equal(controllerAISimulateImage))
	g.Expect(first.Spec.AISimulate.VersionOverride).To(Equal("1.2.0"))
	input, err := frozenInputForTest(first)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(input).To(HaveKey("search_space"))

	t.Log("Editing the Request foreground-cancels the active Run")
	g.Eventually(func(g Gomega) {
		live := getRequest(g)
		live.Spec.Model.Name = "Qwen/Qwen3-30B-A3B"
		g.Expect(c.Update(ctx, live)).To(Succeed())
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())
	g.Eventually(func(g Gomega) {
		terminating := getRun(g, firstName)
		g.Expect(terminating.DeletionTimestamp).NotTo(BeNil())
		live := getRequest(g)
		condition := accepted(live)
		g.Expect(condition).NotTo(BeNil())
		g.Expect(condition.Status).To(Equal(metav1.ConditionFalse))
		g.Expect(condition.Reason).To(Equal(dgdrReasonCancelingStaleRun))
		g.Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		g.Expect(live.Status.ActiveRunRef.Name).To(Equal(firstName))
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())

	t.Log("Complete foreground deletion; envtest has no garbage-collector controller")
	g.Eventually(func(g Gomega) {
		terminating := getRun(g, firstName)
		terminating.Finalizers = nil
		g.Expect(c.Update(ctx, terminating)).To(Succeed())
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())
	g.Eventually(func(g Gomega) {
		err := c.Get(ctx, types.NamespacedName{Namespace: env.Namespace(), Name: firstName}, &nvidiacomv1beta2.DynamoGraphDeploymentRun{})
		g.Expect(apierrors.IsNotFound(err)).To(BeTrue(), "superseded Run lookup returned %v", err)
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())

	t.Log("The completed deletion event wakes the Request and starts only the newest generation")
	var secondName string
	var secondUID types.UID
	g.Eventually(func(g Gomega) {
		live := getRequest(g)
		g.Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		g.Expect(live.Status.ActiveRunRef.Name).NotTo(Equal(firstName))
		g.Expect(live.Status.ObservedGeneration).To(Equal(live.Generation))
		g.Expect(accepted(live).Status).To(Equal(metav1.ConditionTrue))
		g.Expect(live.Status.RecentRunRefs).To(BeEmpty())
		second := getRun(g, live.Status.ActiveRunRef.Name)
		g.Expect(second.Spec.Model.Name).To(Equal("Qwen/Qwen3-30B-A3B"))
		secondName, secondUID = second.Name, second.UID
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())

	t.Log("Deleting the active Run wakes the Request through the owner mapping and it is recreated")
	g.Expect(c.Delete(ctx, getRun(g, secondName))).To(Succeed())
	g.Eventually(func(g Gomega) {
		recreated := getRun(g, secondName)
		g.Expect(recreated.UID).NotTo(Equal(secondUID))
		live := getRequest(g)
		g.Expect(live.Status.ActiveRunRef).NotTo(BeNil())
		g.Expect(live.Status.ActiveRunRef.Name).To(Equal(secondName))
	}).WithTimeout(timeout).WithPolling(poll).Should(Succeed())
}
