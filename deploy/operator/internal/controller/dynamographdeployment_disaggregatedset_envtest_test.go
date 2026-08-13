//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	disaggregatedsetutils "sigs.k8s.io/lws/pkg/utils/disaggregatedset"
)

var _ = Describe("DisaggregatedSet envtest semantics", func() {
	It("propagates graph metadata into both DS roles and rotates the revision", func() {
		ctx := context.Background()
		dgd := newEnvtestDSHappyPathDGD("demo-ds-metadata")
		dgd.Spec.Labels = map[string]string{"e2e.dynamo/metadata": "initial"}
		dgd.Spec.Annotations = map[string]string{"e2e.dynamo/metadata": "initial"}
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })

		reconciler, _ := newEnvtestDSReconcilers()

		By("creating the initial DisaggregatedSet")
		_, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		initialDS := fetchTypedDisaggregatedSet(ctx, current)
		initialRevision := disaggregatedsetutils.ComputeRevision(initialDS.Spec.Roles)
		Expect(initialDS.Spec.Roles).To(HaveLen(2))

		By("updating graph-level metadata and reconciling again")
		current.Spec.Labels["e2e.dynamo/metadata"] = "updated"
		current.Spec.Annotations["e2e.dynamo/metadata"] = "updated"
		Expect(k8sClient.Update(ctx, current)).To(Succeed())

		_, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		updatedDS := fetchTypedDisaggregatedSet(ctx, current)
		updatedRevision := disaggregatedsetutils.ComputeRevision(updatedDS.Spec.Roles)
		Expect(updatedRevision).NotTo(Equal(initialRevision))
		for i := range updatedDS.Spec.Roles {
			role := &updatedDS.Spec.Roles[i]
			Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate).NotTo(BeNil())
			Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate.Labels).To(HaveKeyWithValue("e2e.dynamo/metadata", "updated"))
			Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations).To(HaveKeyWithValue("e2e.dynamo/metadata", "updated"))
			Expect(role.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels).To(HaveKeyWithValue("e2e.dynamo/metadata", "updated"))
			Expect(role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations).To(HaveKeyWithValue("e2e.dynamo/metadata", "updated"))
		}
	})

	It("coalesces sequential restart across both DS roles and completes", func() {
		ctx := context.Background()
		dgd := newEnvtestDSHappyPathDGD("demo-ds-restart")
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })

		reconciler, _ := newEnvtestDSReconcilers()

		By("creating a ready baseline DisaggregatedSet revision")
		_, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		baselineDS := fetchTypedDisaggregatedSet(ctx, current)
		baselineRevision := disaggregatedsetutils.ComputeRevision(baselineDS.Spec.Roles)
		markDisaggregatedSetReady(ctx, current)
		baselineResult, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(baselineResult.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))

		By("requesting a sequential restart through the DGD")
		current.Spec.Restart = &nvidiacomv1beta1.Restart{
			ID: "ds-envtest-restart",
			Strategy: &nvidiacomv1beta1.RestartStrategy{
				Type:  nvidiacomv1beta1.RestartStrategyTypeSequential,
				Order: []string{"prefill", "decode"},
			},
		}
		Expect(k8sClient.Update(ctx, current)).To(Succeed())

		restartResult, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(restartResult.Status.Restart).NotTo(BeNil())
		Expect(restartResult.Status.Restart.ObservedID).To(Equal("ds-envtest-restart"))
		Expect(restartResult.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseRestarting))
		Expect(restartResult.Status.Restart.InProgress).To(Equal([]string{"prefill"}))

		restartedDS := fetchTypedDisaggregatedSet(ctx, current)
		restartedRevision := disaggregatedsetutils.ComputeRevision(restartedDS.Spec.Roles)
		Expect(restartedRevision).NotTo(Equal(baselineRevision))
		restartValues := map[string]bool{}
		for i := range restartedDS.Spec.Roles {
			role := &restartedDS.Spec.Roles[i]
			Expect(role.Spec.LeaderWorkerTemplate.LeaderTemplate).NotTo(BeNil())
			leaderRestart := role.Spec.LeaderWorkerTemplate.LeaderTemplate.Annotations[consts.RestartAnnotation]
			workerRestart := role.Spec.LeaderWorkerTemplate.WorkerTemplate.Annotations[consts.RestartAnnotation]
			Expect(leaderRestart).NotTo(BeEmpty())
			Expect(workerRestart).To(Equal(leaderRestart))
			restartValues[leaderRestart] = true
		}
		Expect(restartValues).To(HaveLen(1), "both DS roles must share one restart revision")

		By("persisting restart status and reconciling after the DS becomes ready")
		current = persistWorkloadProgramStatus(ctx, current, restartResult.Status)
		markDisaggregatedSetReady(ctx, current)
		completedResult, _ := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(completedResult.Status.Restart).NotTo(BeNil())
		Expect(completedResult.Status.Restart.ObservedID).To(Equal("ds-envtest-restart"))
		Expect(completedResult.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseCompleted))
	})

	It("preserves component services across DCD to DS cutover and back to DCD", func() {
		ctx := context.Background()
		dgd := newEnvtestDSHappyPathDGD("demo-ds-cutover")
		dgd.Annotations = nil
		for i := range dgd.Spec.Components {
			dgd.Spec.Components[i].ModelRef = &nvidiacomv1beta1.ModelReference{Name: "shared-smoke-model"}
		}
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })

		reconciler, dcdReconciler := newEnvtestDSReconcilers()

		By("creating the initial DCD pathway resources and component services")
		result, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStatePending))
		legacyDCDs := ownedEnvtestCutoverDCDs(ctx, current)
		Expect(legacyDCDs).To(HaveLen(2))
		serviceUIDs := createComponentServices(ctx, dcdReconciler, legacyDCDs, false)

		By("enabling DisaggregatedSet and keeping DCD services while DS is pending")
		markEnvtestCutoverDCDsReady(ctx, current)
		_, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		if current.Annotations == nil {
			current.Annotations = map[string]string{}
		}
		current.Annotations[consts.KubeAnnotationEnableDisaggregatedSet] = consts.KubeLabelValueTrue
		Expect(k8sClient.Update(ctx, current)).To(Succeed())

		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStatePending))
		Expect(ownedEnvtestCutoverDCDs(ctx, current)).To(HaveLen(2))
		Expect(envtestCutoverServiceUIDs(ctx, current.Namespace, serviceUIDs)).To(Equal(serviceUIDs))

		By("marking the DisaggregatedSet ready and handing service ownership to the DGD")
		markDisaggregatedSetReady(ctx, current)

		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
		Expect(ownedEnvtestCutoverDCDs(ctx, current)).To(BeEmpty())
		Expect(envtestCutoverServiceUIDs(ctx, current.Namespace, serviceUIDs)).To(Equal(serviceUIDs))
		for name := range serviceUIDs {
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: current.Namespace}, service)).To(Succeed())
			owner := metav1.GetControllerOf(service)
			Expect(owner).NotTo(BeNil())
			Expect(owner.Kind).To(Equal(dynamoGraphDeploymentKind))
			Expect(owner.UID).To(Equal(current.UID))
		}
		typedDS := fetchTypedDisaggregatedSet(ctx, current)
		activeRevision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)
		for name := range serviceUIDs {
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: current.Namespace}, service)).To(Succeed())
			Expect(service.Spec.Selector).To(HaveKeyWithValue(disaggregatedsetv1.SetNameLabelKey, disaggregatedSetName(current)))
			Expect(service.Spec.Selector).To(HaveKeyWithValue(disaggregatedsetv1.RevisionLabelKey, activeRevision))
			Expect([]string{"prefill", "decode"}).To(ContainElement(service.Spec.Selector[disaggregatedsetv1.RoleLabelKey]))
		}

		modelServiceName := dynamo.GenerateServiceName("shared-smoke-model")
		modelService := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: modelServiceName, Namespace: current.Namespace}, modelService)).To(Succeed())
		Expect(metav1.GetControllerOf(modelService)).NotTo(BeNil())
		Expect(metav1.GetControllerOf(modelService).Kind).To(Equal("DynamoGraphDeployment"))
		modelServiceUID := modelService.UID

		By("falling back to DCDs and restoring service ownership before deleting the DS")
		delete(current.Annotations, consts.KubeAnnotationEnableDisaggregatedSet)
		Expect(k8sClient.Update(ctx, current)).To(Succeed())

		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStatePending))
		replacementDCDs := ownedEnvtestCutoverDCDs(ctx, current)
		Expect(replacementDCDs).To(HaveLen(2))
		for name := range serviceUIDs {
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: current.Namespace}, service)).To(Succeed())
			owner := metav1.GetControllerOf(service)
			Expect(owner).NotTo(BeNil())
			Expect(owner.Kind).To(Equal(dynamoGraphDeploymentKind))
		}
		_ = createComponentServices(ctx, dcdReconciler, replacementDCDs, false)
		for name := range serviceUIDs {
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: current.Namespace}, service)).To(Succeed())
			Expect(isDisaggregatedSetServiceSelector(service)).To(BeTrue(), "pending DCDs must not take traffic from the ready DisaggregatedSet")
		}
		markEnvtestCutoverDCDsReady(ctx, current)
		_ = createComponentServices(ctx, dcdReconciler, replacementDCDs, true)

		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
		Expect(apierrors.IsNotFound(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(current), Namespace: current.Namespace}, newDisaggregatedSetObject()))).To(BeTrue())

		fallbackDCDs := ownedEnvtestCutoverDCDs(ctx, current)
		fallbackDCDNames := map[string]struct{}{}
		for _, dcd := range fallbackDCDs {
			fallbackDCDNames[dcd.Name] = struct{}{}
			service := &corev1.Service{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: dynamo.NormalizeKubeResourceName(dcd.Name), Namespace: current.Namespace}, service)).To(Succeed())
			Expect(isDisaggregatedSetServiceSelector(service)).To(BeFalse(), "ready DCDs must take over the Service before the DisaggregatedSet is deleted")
			owner := metav1.GetControllerOf(service)
			Expect(owner).NotTo(BeNil())
			Expect(owner.Kind).To(Equal("DynamoComponentDeployment"))
			Expect(owner.Name).To(Equal(dcd.Name))
		}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: modelServiceName, Namespace: current.Namespace}, modelService)).To(Succeed())
		Expect(modelService.UID).To(Equal(modelServiceUID))
		modelOwner := metav1.GetControllerOf(modelService)
		Expect(modelOwner).NotTo(BeNil())
		Expect(modelOwner.Kind).To(Equal("DynamoComponentDeployment"))
		Expect(fallbackDCDNames).To(HaveKey(modelOwner.Name))
	})

	It("recreates a missing shared model service before DS fallback completes", func() {
		ctx := context.Background()
		dgd := newEnvtestDSHappyPathDGD("demo-ds-missing-service")
		for i := range dgd.Spec.Components {
			dgd.Spec.Components[i].ModelRef = &nvidiacomv1beta1.ModelReference{Name: "shared-missing-model"}
		}
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })

		reconciler, dcdReconciler := newEnvtestDSReconcilers()

		By("creating a DS-owned shared model service")
		_, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		markDisaggregatedSetReady(ctx, current)
		result, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))

		By("creating replacement DCDs and deleting the shared model service before ownership handoff")
		delete(current.Annotations, consts.KubeAnnotationEnableDisaggregatedSet)
		Expect(k8sClient.Update(ctx, current)).To(Succeed())
		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStatePending))

		replacementDCDs := ownedEnvtestCutoverDCDs(ctx, current)
		Expect(replacementDCDs).To(HaveLen(2))
		_ = createComponentServices(ctx, dcdReconciler, replacementDCDs, false)

		modelServiceName := dynamo.GenerateServiceName("shared-missing-model")
		modelService := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: modelServiceName, Namespace: current.Namespace}, modelService)).To(Succeed())
		Expect(k8sClient.Delete(ctx, modelService)).To(Succeed())
		Eventually(func() bool {
			err := k8sClient.Get(ctx, types.NamespacedName{Name: modelServiceName, Namespace: current.Namespace}, &corev1.Service{})
			return apierrors.IsNotFound(err)
		}, time.Minute, time.Second).Should(BeTrue())
		markEnvtestCutoverDCDsReady(ctx, current)

		By("recreating the missing model service under a replacement DCD owner")
		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))
		Expect(apierrors.IsNotFound(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(current), Namespace: current.Namespace}, newDisaggregatedSetObject()))).To(BeTrue())

		replacementDCDNames := map[string]struct{}{}
		for _, dcd := range ownedEnvtestCutoverDCDs(ctx, current) {
			replacementDCDNames[dcd.Name] = struct{}{}
		}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: modelServiceName, Namespace: current.Namespace}, modelService)).To(Succeed())
		modelOwner := metav1.GetControllerOf(modelService)
		Expect(modelOwner).NotTo(BeNil())
		Expect(modelOwner.Kind).To(Equal("DynamoComponentDeployment"))
		Expect(replacementDCDNames).To(HaveKey(modelOwner.Name))
	})

	It("keeps the selected DisaggregatedSet when later intent is unsupported", func() {
		ctx := context.Background()
		dgd := newEnvtestDSHappyPathDGD("demo-ds-fallback-gating")
		Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })

		reconciler, _ := newEnvtestDSReconcilers()

		By("creating a ready DisaggregatedSet")
		_, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		markDisaggregatedSetReady(ctx, current)
		result, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateSuccessful))

		By("making the deployment ineligible for DisaggregatedSet while keeping the annotation")
		current.Spec.Components[0].ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
		Expect(k8sClient.Update(ctx, current)).To(Succeed())

		By("reporting unsupported intent without switching workload pathways")
		result, current = reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
		Expect(result.Status.State).To(Equal(nvidiacomv1beta1.DGDStateFailed))
		eligibility := apiMeta.FindStatusCondition(result.Status.Conditions, "DisaggregatedSetEligible")
		Expect(eligibility).NotTo(BeNil())
		Expect(eligibility.Status).To(Equal(metav1.ConditionFalse))
		Expect(eligibility.Reason).To(Equal("UnsupportedIntent"))
		Expect(eligibility.Message).To(ContainSubstring("scalingAdapter"))
		Expect(ownedEnvtestCutoverDCDs(ctx, current)).To(BeEmpty())
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(current), Namespace: current.Namespace}, newDisaggregatedSetObject())).To(Succeed())
	})
})

func newEnvtestDSReconcilers() (*DynamoGraphDeploymentReconciler, *DynamoComponentDeploymentReconciler) {
	runtimeConfig := &commoncontroller.RuntimeConfig{
		Gate:         features.Gates{LWS: true, DisaggregatedSet: true},
		Capabilities: features.Capabilities{DisaggregatedSetAPI: true},
	}
	operatorConfig := &configv1alpha1.OperatorConfiguration{
		Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		Namespace: configv1alpha1.NamespaceConfiguration{Restricted: envtestNamespace},
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
		Config:        operatorConfig,
		RuntimeConfig: runtimeConfig,
	}
	return reconciler, dcdReconciler
}

func reconcileCurrentDGDProgram(
	ctx context.Context,
	reconciler *DynamoGraphDeploymentReconciler,
	name string,
	namespace string,
) (workloadProgramResult, *nvidiacomv1beta1.DynamoGraphDeployment) {
	current := &nvidiacomv1beta1.DynamoGraphDeployment{}
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: namespace}, current)).To(Succeed())
	result, err := reconciler.selectWorkloadProgram(current).Reconcile(ctx, workloadProgramRequest{DGD: current})
	Expect(err).NotTo(HaveOccurred())
	return result, current
}

func persistWorkloadProgramStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) *nvidiacomv1beta1.DynamoGraphDeployment {
	current := &nvidiacomv1beta1.DynamoGraphDeployment{}
	Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
	current.Status = status
	Expect(k8sClient.Status().Update(ctx, current)).To(Succeed())
	Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(dgd), current)).To(Succeed())
	return current
}

func fetchTypedDisaggregatedSet(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *disaggregatedsetv1.DisaggregatedSet {
	raw := newDisaggregatedSetObject()
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, raw)).To(Succeed())
	typed := &disaggregatedsetv1.DisaggregatedSet{}
	Expect(runtime.DefaultUnstructuredConverter.FromUnstructured(raw.Object, typed)).To(Succeed())
	return typed
}

func markDisaggregatedSetReady(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	ds := newDisaggregatedSetObject()
	Expect(k8sClient.Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds)).To(Succeed())
	typedDS := &disaggregatedsetv1.DisaggregatedSet{}
	Expect(runtime.DefaultUnstructuredConverter.FromUnstructured(ds.Object, typedDS)).To(Succeed())
	revision := disaggregatedsetutils.ComputeRevision(typedDS.Spec.Roles)
	sliceCount := int(disaggregatedsetutils.GetSlices(typedDS))

	// Simulate LWS v0.10 convergence: each (slice, role, revision) has one ready
	// child, and children from the previous revision have been removed.
	existing := &leaderworkersetv1.LeaderWorkerSetList{}
	Expect(k8sClient.List(ctx, existing, client.InNamespace(ds.GetNamespace()), client.MatchingLabels{
		disaggregatedsetv1.SetNameLabelKey: ds.GetName(),
	})).To(Succeed())
	for i := range existing.Items {
		Expect(k8sClient.Delete(ctx, &existing.Items[i])).To(Succeed())
	}

	roleStatuses := make([]any, 0, len(typedDS.Spec.Roles))
	for i := range typedDS.Spec.Roles {
		role := &typedDS.Spec.Roles[i]
		desiredReplicas := ptr.Deref(role.Spec.Replicas, int32(1))
		for slice := range sliceCount {
			child := &leaderworkersetv1.LeaderWorkerSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      disaggregatedsetutils.GenerateName(ds.GetName(), slice, revision, role.Name),
					Namespace: ds.GetNamespace(),
					Labels:    disaggregatedsetutils.GenerateLabels(ds.GetName(), slice, revision, role.Name),
					OwnerReferences: []metav1.OwnerReference{{
						APIVersion: disaggregatedsetv1.GroupVersion.String(),
						Kind:       "DisaggregatedSet",
						Name:       ds.GetName(),
						UID:        ds.GetUID(),
						Controller: ptr.To(true),
					}},
				},
				Spec: role.Spec,
			}
			Expect(k8sClient.Create(ctx, child)).To(Succeed())
			child.Status = leaderworkersetv1.LeaderWorkerSetStatus{
				ObservedGeneration: child.Generation,
				Replicas:           desiredReplicas,
				UpdatedReplicas:    desiredReplicas,
				ReadyReplicas:      desiredReplicas,
			}
			Expect(k8sClient.Status().Update(ctx, child)).To(Succeed())
		}
		totalReplicas := int64(desiredReplicas) * int64(sliceCount)
		roleStatuses = append(roleStatuses, map[string]any{
			"name": role.Name, "replicas": totalReplicas, "updatedReplicas": totalReplicas, "readyReplicas": totalReplicas,
		})
	}
	ds.Object["status"] = map[string]any{
		"observedGeneration": ds.GetGeneration(),
		"roleStatuses":       roleStatuses,
	}
	Expect(k8sClient.Status().Update(ctx, ds)).To(Succeed())
}

func createComponentServices(
	ctx context.Context,
	dcdReconciler *DynamoComponentDeploymentReconciler,
	dcds []nvidiacomv1beta1.DynamoComponentDeployment,
	targetReady bool,
) map[string]types.UID {
	serviceUIDs := map[string]types.UID{}
	for i := range dcds {
		_, err := dcdReconciler.createOrUpdateOrDeleteServices(ctx, generateResourceOption{
			dynamoComponentDeployment: &dcds[i],
			serviceTargetReady:        targetReady,
		})
		Expect(err).NotTo(HaveOccurred())
		service := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{
			Name:      dynamo.NormalizeKubeResourceName(dcds[i].Name),
			Namespace: dcds[i].Namespace,
		}, service)).To(Succeed())
		serviceUIDs[service.Name] = service.UID
	}
	return serviceUIDs
}

func ownedEnvtestCutoverDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) []nvidiacomv1beta1.DynamoComponentDeployment {
	list := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	Expect(k8sClient.List(ctx, list, client.InNamespace(dgd.Namespace))).To(Succeed())
	owned := make([]nvidiacomv1beta1.DynamoComponentDeployment, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], dgd) {
			owned = append(owned, list.Items[i])
		}
	}
	return owned
}

func markEnvtestCutoverDCDsReady(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	for _, dcd := range ownedEnvtestCutoverDCDs(ctx, dgd) {
		replicas := ptr.Deref(dcd.Spec.Replicas, int32(1))
		dcd.Status.ObservedGeneration = dcd.Generation
		dcd.Status.Conditions = []metav1.Condition{{
			Type:               nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: dcd.Generation,
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

func envtestCutoverServiceUIDs(ctx context.Context, namespace string, expected map[string]types.UID) map[string]types.UID {
	uids := make(map[string]types.UID, len(expected))
	for name := range expected {
		service := &corev1.Service{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: namespace}, service)).To(Succeed())
		uids[name] = service.UID
	}
	return uids
}

func newEnvtestDSHappyPathDGD(name string) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: envtestNamespace,
			UID:       types.UID(fmt.Sprintf("%s-uid", name)),
			Annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:          "prefill",
					ComponentType:          nvidiacomv1beta1.ComponentTypePrefill,
					RuntimeVersionOverride: "1.0.0",
					Multinode:              &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:            envtestDSTestPodTemplate(),
				},
				{
					ComponentName:          "decode",
					ComponentType:          nvidiacomv1beta1.ComponentTypeDecode,
					RuntimeVersionOverride: "1.0.0",
					Multinode:              &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
					PodTemplate:            envtestDSTestPodTemplate(),
				},
			},
		},
	}
}

func envtestDSTestPodTemplate() *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:    consts.MainContainerName,
				Image:   "busybox:1.36",
				Command: []string{"sh"},
				Args:    []string{"-c", "sleep 3600"},
			}},
		},
	}
}
