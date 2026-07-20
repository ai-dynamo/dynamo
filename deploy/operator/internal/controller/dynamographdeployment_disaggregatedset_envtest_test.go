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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"

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

	It("keeps the serving resources during DCD to DS cutover", func() {
		ctx := context.Background()
		dgd := newDSHappyPathDGD()
		dgd.Name = "demo-ds-cutover"
		dgd.UID = ""
		dgd.Annotations = nil
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() {
			_ = k8sClient.Delete(ctx, dgd)
		})
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, dgd)).To(Succeed())

		runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{LWS: true, DisaggregatedSet: true}}
		operatorConfig := &configv1alpha1.OperatorConfiguration{
			Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		}
		reconciler := &DynamoGraphDeploymentReconciler{
			Client:        k8sClient,
			Recorder:      record.NewFakeRecorder(100),
			Config:        operatorConfig,
			RuntimeConfig: runtimeConfig,
		}
		dcdReconciler := &DynamoComponentDeploymentReconciler{
			Client:        k8sClient,
			Recorder:      record.NewFakeRecorder(100),
			Config:        reconciler.Config,
			RuntimeConfig: runtimeConfig,
		}

		result, err := reconciler.reconcileWorkloadResources(ctx, dgd, true, false, "", nil, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.State).To(Equal(nvidiacomv1beta1.DGDStatePending))
		legacyDCDs := ownedCutoverDCDs(ctx, dgd)
		Expect(legacyDCDs).To(HaveLen(2))
		serviceUIDs := map[string]types.UID{}
		for i := range legacyDCDs {
			modified, err := dcdReconciler.createOrUpdateOrDeleteServices(ctx, generateResourceOption{dynamoComponentDeployment: &legacyDCDs[i]})
			Expect(err).NotTo(HaveOccurred())
			Expect(modified).To(BeTrue())
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: legacyDCDs[i].Name, Namespace: dgd.Namespace}, service)).To(Succeed())
			serviceUIDs[service.Name] = service.UID
		}
		Expect(serviceUIDs).To(HaveLen(2))

		markCutoverDCDsReady(ctx, dgd)
		result, err = reconciler.reconcileWorkloadResources(ctx, dgd, true, false, "", nil, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))

		dgd.Annotations = map[string]string{consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue}
		Expect(k8sClient.Update(ctx, dgd)).To(Succeed())
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, dgd)).To(Succeed())
		useDS, reason := reconciler.shouldUseDisaggregatedSet(dgd)
		Expect(useDS).To(BeTrue(), reason)

		result, err = reconciler.reconcileWorkloadResources(ctx, dgd, true, useDS, reason, nil, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.State).To(Equal(nvidiacomv1beta1.DGDStatePending))
		Expect(ownedCutoverDCDs(ctx, dgd)).To(HaveLen(2), "legacy DCDs must remain until DS is ready")
		Expect(cutoverServiceUIDs(ctx, dgd.Namespace, serviceUIDs)).To(Equal(serviceUIDs))

		ds := newDisaggregatedSetObject()
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)).To(Succeed())
		ds.Object["status"] = map[string]any{
			"observedGeneration": ds.GetGeneration(),
			"roleStatuses": []any{
				map[string]any{"name": "prefill", "replicas": int64(1), "updatedReplicas": int64(1), "readyReplicas": int64(1)},
				map[string]any{"name": "decode", "replicas": int64(1), "updatedReplicas": int64(1), "readyReplicas": int64(1)},
			},
		}
		Expect(k8sClient.Status().Update(ctx, ds)).To(Succeed())

		result, err = reconciler.reconcileWorkloadResources(ctx, dgd, true, useDS, reason, nil, nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
		Expect(ownedCutoverDCDs(ctx, dgd)).To(BeEmpty())
		Expect(cutoverServiceUIDs(ctx, dgd.Namespace, serviceUIDs)).To(Equal(serviceUIDs), "DS cutover must preserve component Services")
	})
})

func ownedCutoverDCDs(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) []nvidiacomv1beta1.DynamoComponentDeployment {
	list := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	Expect(k8sClient.List(ctx, list)).To(Succeed())
	owned := make([]nvidiacomv1beta1.DynamoComponentDeployment, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], dgd) {
			owned = append(owned, list.Items[i])
		}
	}
	return owned
}

func markCutoverDCDsReady(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	for _, dcd := range ownedCutoverDCDs(ctx, dgd) {
		replicas := ptr.Deref(dcd.Spec.Replicas, int32(1))
		dcd.Status.ObservedGeneration = dcd.Generation
		dcd.Status.Conditions = []metav1.Condition{{
			Type:               nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable,
			Status:             metav1.ConditionTrue,
			Reason:             "CutoverTestReady",
			LastTransitionTime: metav1.Now(),
		}}
		dcd.Status.Component = &nvidiacomv1beta1.ComponentReplicaStatus{
			ComponentKind:     nvidiacomv1beta1.ComponentKindLeaderWorkerSet,
			ComponentNames:    []string{dcd.Name + "-0"},
			Replicas:          replicas,
			UpdatedReplicas:   replicas,
			ReadyReplicas:     ptr.To(replicas),
			AvailableReplicas: ptr.To(replicas),
		}
		Expect(k8sClient.Status().Update(ctx, &dcd)).To(Succeed())
	}
}

func cutoverServiceUIDs(ctx context.Context, namespace string, expected map[string]types.UID) map[string]types.UID {
	uids := make(map[string]types.UID, len(expected))
	for name := range expected {
		service := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: namespace}, service)).To(Succeed())
		uids[name] = service.UID
	}
	return uids
}

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
