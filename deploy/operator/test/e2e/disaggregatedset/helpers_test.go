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
	"fmt"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

const (
	testMetadataLabel      = "e2e.dynamo/metadata"
	testMetadataAnnotation = "e2e.dynamo/metadata"
)

func newTestDGD(name string) *nvidiacomv1beta1.DynamoGraphDeployment {
	replicas := int32(1)
	modelName := name + "-model"
	component := func(
		name string,
		componentType nvidiacomv1beta1.ComponentType,
	) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
		return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: name,
			ComponentType: componentType,
			Replicas:      &replicas,
			Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
			ModelRef:      &nvidiacomv1beta1.ModelReference{Name: modelName},
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:    consts.MainContainerName,
						Image:   flagWorkloadImage,
						Command: []string{"sh", "-c"},
						Args:    []string{"sleep 3600"},
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}},
						},
						StartupProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}},
						},
					}},
				},
			},
		}
	}

	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: flagNamespace,
			Labels:    map[string]string{"test.dynamo/managed": "true"},
			Annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Labels:           map[string]string{testMetadataLabel: "initial"},
			Annotations:      map[string]string{testMetadataAnnotation: "initial"},
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				component("prefill", nvidiacomv1beta1.ComponentTypePrefill),
				component("decode", nvidiacomv1beta1.ComponentTypeDecode),
			},
		},
	}
}

func waitForDisaggregatedSet(dgd *nvidiacomv1beta1.DynamoGraphDeployment) *disaggregatedsetv1.DisaggregatedSet {
	var result *disaggregatedsetv1.DisaggregatedSet
	Eventually(func(g Gomega) {
		current := &disaggregatedsetv1.DisaggregatedSet{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
		g.Expect(metav1.IsControlledBy(current, dgd)).To(BeTrue())
		g.Expect(current.Spec.Roles).To(HaveLen(2))
		result = current
	}, flagReadyTimeout, time.Second).Should(Succeed())
	return result
}

func waitForCurrentLeaderWorkerSets(ds *disaggregatedsetv1.DisaggregatedSet) {
	revision := disaggregatedsetutils.ComputeRevision(ds.Spec.Roles)
	Eventually(func(g Gomega) {
		list := &leaderworkersetv1.LeaderWorkerSetList{}
		g.Expect(k8sClient.List(
			ctx,
			list,
			client.InNamespace(ds.Namespace),
			client.MatchingLabels{
				disaggregatedsetv1.SetNameLabelKey:  ds.Name,
				disaggregatedsetv1.RevisionLabelKey: revision,
			},
		)).To(Succeed())
		g.Expect(list.Items).To(HaveLen(2))
		roles := map[string]bool{}
		for i := range list.Items {
			lws := &list.Items[i]
			roles[lws.Labels[disaggregatedsetv1.RoleLabelKey]] = true
			g.Expect(lws.Status.ObservedGeneration).To(BeNumerically(">=", lws.Generation))
			g.Expect(lws.Status.Replicas).To(Equal(int32(1)))
			g.Expect(lws.Status.UpdatedReplicas).To(Equal(int32(1)))
			g.Expect(lws.Status.ReadyReplicas).To(Equal(int32(1)))
		}
		g.Expect(roles).To(Equal(map[string]bool{"prefill": true, "decode": true}))
	}, flagReadyTimeout, time.Second).Should(Succeed())
}

func waitForDGDSuccessful(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	Eventually(func(g Gomega) {
		current := &nvidiacomv1beta1.DynamoGraphDeployment{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
		g.Expect(current.Status.ObservedGeneration).To(BeNumerically(">=", current.Generation))
		g.Expect(current.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
	}, flagReadyTimeout, time.Second).Should(Succeed())
}

func waitForRoleMetadata(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	value string,
) *disaggregatedsetv1.DisaggregatedSet {
	var result *disaggregatedsetv1.DisaggregatedSet
	Eventually(func(g Gomega) {
		current := &disaggregatedsetv1.DisaggregatedSet{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
		g.Expect(current.Spec.Roles).To(HaveLen(2))
		for i := range current.Spec.Roles {
			role := &current.Spec.Roles[i]
			g.Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate).NotTo(BeNil())
			g.Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate.Labels).To(HaveKeyWithValue(testMetadataLabel, value))
			g.Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations).To(
				HaveKeyWithValue(testMetadataAnnotation, value),
			)
			g.Expect(role.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels).To(HaveKeyWithValue(testMetadataLabel, value))
			g.Expect(role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations).To(
				HaveKeyWithValue(testMetadataAnnotation, value),
			)
		}
		result = current
	}, flagReadyTimeout, time.Second).Should(Succeed())
	return result
}

func waitForRestartRevision(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	oldRevision string,
) *disaggregatedsetv1.DisaggregatedSet {
	var result *disaggregatedsetv1.DisaggregatedSet
	Eventually(func(g Gomega) {
		current := &disaggregatedsetv1.DisaggregatedSet{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
		revision := disaggregatedsetutils.ComputeRevision(current.Spec.Roles)
		g.Expect(revision).NotTo(Equal(oldRevision))

		restartValues := map[string]bool{}
		for i := range current.Spec.Roles {
			role := &current.Spec.Roles[i]
			g.Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate).NotTo(BeNil())
			leaderRestart := role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations[consts.RestartAnnotation]
			workerRestart := role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations[consts.RestartAnnotation]
			g.Expect(leaderRestart).NotTo(BeEmpty())
			g.Expect(workerRestart).To(Equal(leaderRestart))
			restartValues[leaderRestart] = true
		}
		g.Expect(restartValues).To(HaveLen(1), "all DisaggregatedSet roles must share one restart revision")
		result = current
	}, flagReadyTimeout, time.Second).Should(Succeed())
	return result
}

func waitForReplacementDCDs(dgd *nvidiacomv1beta1.DynamoGraphDeployment) []nvidiacomv1beta1.DynamoComponentDeployment {
	var replacements []nvidiacomv1beta1.DynamoComponentDeployment
	Eventually(func(g Gomega) {
		list := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
		g.Expect(k8sClient.List(
			ctx,
			list,
			client.InNamespace(dgd.Namespace),
			client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
		)).To(Succeed())
		owned := make([]nvidiacomv1beta1.DynamoComponentDeployment, 0, 2)
		for i := range list.Items {
			if metav1.IsControlledBy(&list.Items[i], dgd) {
				owned = append(owned, list.Items[i])
			}
		}
		g.Expect(owned).To(HaveLen(2))
		replacements = owned
	}, flagReadyTimeout, time.Second).Should(Succeed())
	return replacements
}

func waitForFallbackComplete(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	Eventually(func(g Gomega) {
		ds := &disaggregatedsetv1.DisaggregatedSet{}
		err := k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), ds)
		g.Expect(apierrors.IsNotFound(err)).To(BeTrue())

		replacements := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
		g.Expect(k8sClient.List(
			ctx,
			replacements,
			client.InNamespace(dgd.Namespace),
			client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
		)).To(Succeed())
		ready := 0
		for i := range replacements.Items {
			replacement := &replacements.Items[i]
			if !metav1.IsControlledBy(replacement, dgd) {
				continue
			}
			isReady, reason := replacement.IsReady()
			g.Expect(isReady).To(BeTrue(), "replacement DCD %s is not ready: %s", replacement.Name, reason)
			ready++
		}
		g.Expect(ready).To(Equal(2))

		current := &nvidiacomv1beta1.DynamoGraphDeployment{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
		g.Expect(current.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
	}, flagReadyTimeout, time.Second).Should(Succeed())
}

func verifyServiceOwnership(dgd *nvidiacomv1beta1.DynamoGraphDeployment, ownerKind string) {
	Expect(dgd.Spec.Components).NotTo(BeEmpty())
	Expect(dgd.Spec.Components[0].ModelRef).NotTo(BeNil())
	modelService := &corev1.Service{}
	Expect(k8sClient.Get(ctx, client.ObjectKey{
		Namespace: dgd.Namespace,
		Name:      dynamo.GenerateServiceName(dgd.Spec.Components[0].ModelRef.Name),
	}, modelService)).To(Succeed())
	owner := metav1.GetControllerOf(modelService)
	Expect(owner).NotTo(BeNil())
	Expect(owner.Kind).To(Equal(ownerKind))

	services := &corev1.ServiceList{}
	Expect(k8sClient.List(
		ctx,
		services,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	)).To(Succeed())
	componentOwners := map[string]string{}
	for i := range services.Items {
		component := services.Items[i].Labels[consts.KubeLabelDynamoComponent]
		if component == "" {
			continue
		}
		controller := metav1.GetControllerOf(&services.Items[i])
		Expect(controller).NotTo(BeNil(), "component Service %s has no controller owner", services.Items[i].Name)
		Expect(controller.Kind).To(Equal(ownerKind))
		componentOwners[component] = controller.Name
	}
	Expect(componentOwners).To(HaveKey("prefill"))
	Expect(componentOwners).To(HaveKey("decode"))
}

func deleteModelService(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	Expect(dgd.Spec.Components).NotTo(BeEmpty())
	Expect(dgd.Spec.Components[0].ModelRef).NotTo(BeNil())
	service := &corev1.Service{}
	Expect(k8sClient.Get(ctx, client.ObjectKey{
		Namespace: dgd.Namespace,
		Name:      dynamo.GenerateServiceName(dgd.Spec.Components[0].ModelRef.Name),
	}, service)).To(Succeed())
	Expect(k8sClient.Delete(ctx, service)).To(Succeed())
	Eventually(func() bool {
		err := k8sClient.Get(ctx, client.ObjectKeyFromObject(service), &corev1.Service{})
		return apierrors.IsNotFound(err)
	}, time.Minute, time.Second).Should(BeTrue())
}

func uniqueName() string {
	return fmt.Sprintf("ds-e2e-%d", time.Now().UnixNano())
}

func sequentialRestart(id string) *nvidiacomv1beta1.Restart {
	return &nvidiacomv1beta1.Restart{
		ID: id,
		Strategy: &nvidiacomv1beta1.RestartStrategy{
			Type:  nvidiacomv1beta1.RestartStrategyTypeSequential,
			Order: []string{"prefill", "decode"},
		},
	}
}
