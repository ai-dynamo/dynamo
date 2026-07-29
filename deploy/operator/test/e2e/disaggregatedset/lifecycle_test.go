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

package disaggregatedset

import (
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

var _ = Describe(
	"DisaggregatedSet live lifecycle",
	Ordered,
	Label("e2e", "integration", "k8s", "gpu_0"),
	func() {
		var (
			dgd             *nvidiacomv1beta1.DynamoGraphDeployment
			dsName          string
			initialRevision string
		)

		BeforeAll(func() {
			By("creating a DGD that opts two multinode roles into DisaggregatedSet")
			dgd = newTestDGD(uniqueName())
			Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		})

		AfterAll(func() {
			if dgd == nil {
				return
			}
			By("deleting the test DGD")
			err := k8sClient.Delete(ctx, dgd)
			Expect(err == nil || apierrors.IsNotFound(err)).To(BeTrue())
		})

		It("creates a real DisaggregatedSet and ready LWS children", func() {
			By("waiting for the Dynamo operator to create the DisaggregatedSet")
			ds := waitForDisaggregatedSet(dgd)
			dsName = ds.Name
			initialRevision = disaggregatedsetutils.ComputeRevision(ds.Spec.Roles)

			By("waiting for the LWS v0.9.0 controller to create and ready both role workloads")
			waitForCurrentLeaderWorkerSets(ds)
			waitForDGDSuccessful(dgd)

			By("verifying DisaggregatedSet-owned serving Services are controlled by the DGD")
			verifyServiceOwnership(dgd, "DynamoGraphDeployment")
		})

		It("reconciles graph metadata into both role pod templates", func() {
			By("updating graph-level pod metadata")
			current := &nvidiacomv1beta1.DynamoGraphDeployment{}
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
			current.Spec.Labels[testMetadataLabel] = "updated"
			current.Spec.Annotations[testMetadataAnnotation] = "updated"
			Expect(k8sClient.Update(ctx, current)).To(Succeed())

			By("waiting for both DisaggregatedSet roles to observe the metadata update")
			ds := waitForRoleMetadata(dgd, "updated")
			initialRevision = disaggregatedsetutils.ComputeRevision(ds.Spec.Roles)
			waitForCurrentLeaderWorkerSets(ds)
			waitForDGDSuccessful(dgd)
		})

		It("rolls both roles through one DisaggregatedSet revision", func() {
			By("requesting a sequential graph restart")
			current := &nvidiacomv1beta1.DynamoGraphDeployment{}
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
			current.Spec.Restart = sequentialRestart("ds-e2e-restart")
			Expect(k8sClient.Update(ctx, current)).To(Succeed())

			By("verifying both roles share one new restart annotation and revision")
			ds := waitForRestartRevision(dgd, initialRevision)
			waitForCurrentLeaderWorkerSets(ds)

			By("waiting for the graph restart to complete")
			Eventually(func(g Gomega) {
				latest := &nvidiacomv1beta1.DynamoGraphDeployment{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), latest)).To(Succeed())
				g.Expect(latest.Status.Restart).NotTo(BeNil())
				g.Expect(latest.Status.Restart.ObservedID).To(Equal("ds-e2e-restart"))
				g.Expect(latest.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseCompleted))
			}, flagReadyTimeout, time.Second).Should(Succeed())
		})

		It("falls back to DCDs and restores Service ownership after a Service deletion", func() {
			By("removing the opt-in annotation to exercise the metadata-only DGD watch")
			current := &nvidiacomv1beta1.DynamoGraphDeployment{}
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
			delete(current.Annotations, consts.KubeAnnotationEnableDisaggregatedSet)
			Expect(k8sClient.Update(ctx, current)).To(Succeed())

			By("waiting for replacement DCDs while the DisaggregatedSet still serves")
			waitForReplacementDCDs(dgd)
			ds := &disaggregatedsetv1.DisaggregatedSet{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Namespace: dgd.Namespace,
				Name:      dsName,
			}, ds)).To(Succeed())

			By("deleting the DGD-owned shared model Service before ownership handoff")
			deleteModelService(dgd)

			By("waiting for replacement DCD workloads and DisaggregatedSet cleanup")
			waitForFallbackComplete(dgd)

			By("verifying component and shared model Services are now DCD-owned")
			verifyServiceOwnership(dgd, "DynamoComponentDeployment")
		})
	},
)
