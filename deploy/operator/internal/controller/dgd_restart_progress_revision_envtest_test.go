//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
)

var _ = Describe("Component restart progress after a failed first DCD write", func() {
	DescribeTable("keeps the restart pending until the requested DCD revision is observed and ready",
		func(strategy nvidiacomv1beta1.RestartStrategyType) {
			ctx := context.Background()
			order := []string{"frontend", "decode"}
			dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
				Name:      "component-restart-retry-" + strings.ToLower(string(strategy)),
				Namespace: envtestNamespace,
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{
							ComponentName: "frontend", ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
							Replicas: new(int32(1)), RuntimeVersionOverride: "1.0.0", PodTemplate: envtestDSTestPodTemplate(),
						},
						{
							ComponentName: "decode", ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
							Replicas: new(int32(1)), RuntimeVersionOverride: "1.0.0", PodTemplate: envtestDSTestPodTemplate(),
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, dgd)).To(Succeed())
			DeferCleanup(func() { _ = k8sClient.Delete(ctx, dgd) })
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:   k8sClient,
				Recorder: events.NewFakeRecorder(100),
				Config: &configv1alpha1.OperatorConfiguration{
					Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
					Namespace: configv1alpha1.NamespaceConfiguration{Restricted: envtestNamespace},
				},
				RuntimeConfig: &commoncontroller.RuntimeConfig{},
			}

			By("observing the ready baseline before requesting a new restart")
			_, current := reconcileCurrentDGDProgram(ctx, reconciler, dgd.Name, dgd.Namespace)
			Expect(current.Annotations[consts.KubeAnnotationWorkloadProvider]).To(Equal(consts.WorkloadProviderComponent))
			markRestartDCDsReady(ctx, current)
			current.Spec.Restart = &nvidiacomv1beta1.Restart{
				ID:       "restart-retry",
				Strategy: &nvidiacomv1beta1.RestartStrategy{Type: strategy},
			}
			if strategy == nvidiacomv1beta1.RestartStrategyTypeSequential {
				current.Spec.Restart.Strategy.Order = order
			}
			Expect(k8sClient.Update(ctx, current)).To(Succeed())

			By("failing the DCD write and persisting the returned restart status")
			conflictingClient := &restartWriteConflictClient{
				Client:    k8sClient,
				gvk:       nvidiacomv1beta1.GroupVersion.WithKind("DynamoComponentDeployment"),
				failWrite: true,
			}
			reconciler.Client = conflictingClient
			program := reconciler.newComponentProgram()
			result, err := program.Reconcile(ctx, workloadProgramRequest{DGD: current})
			Expect(err).To(HaveOccurred())
			Expect(result.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseRestarting))
			current = persistWorkloadProgramStatus(ctx, current, result.Status)

			By("retrying against the ready old revision without advancing or completing")
			pending := []string{"frontend"}
			if strategy == nvidiacomv1beta1.RestartStrategyTypeParallel {
				pending = order
			}
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: current})
			Expect(err).To(HaveOccurred())
			Expect(result.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseRestarting))
			Expect(result.Status.Restart.InProgress).To(ConsistOf(pending))
			for _, dcd := range ownedEnvtestCutoverDCDs(ctx, current) {
				Expect(dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation]).To(BeEmpty())
			}

			By("accepting the write but waiting until the requested revision becomes ready")
			conflictingClient.failWrite = false
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: current})
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Status.Restart.InProgress).To(ConsistOf(pending))
			current = persistWorkloadProgramStatus(ctx, current, result.Status)
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: current})
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Status.Restart.InProgress).To(ConsistOf(pending))

			By("advancing or completing only after the requested revision is ready")
			markRestartDCDsReady(ctx, current)
			result, err = program.Reconcile(ctx, workloadProgramRequest{DGD: current})
			Expect(err).NotTo(HaveOccurred())
			if strategy == nvidiacomv1beta1.RestartStrategyTypeParallel {
				Expect(result.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseCompleted))
			} else {
				Expect(result.Status.Restart.Phase).To(Equal(nvidiacomv1beta1.RestartPhaseRestarting))
				Expect(result.Status.Restart.InProgress).To(Equal([]string{"decode"}))
			}
		},
		Entry("sequential", nvidiacomv1beta1.RestartStrategyTypeSequential),
		Entry("parallel", nvidiacomv1beta1.RestartStrategyTypeParallel),
	)
})

// markRestartDCDsReady reports every DCD owned by dgd as available at its
// current generation, standing in for the DCD controller.
func markRestartDCDsReady(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	dcds := ownedEnvtestCutoverDCDs(ctx, dgd)
	Expect(dcds).NotTo(BeEmpty())
	for i := range dcds {
		dcd := &dcds[i]
		dcd.Status.ObservedGeneration = dcd.Generation
		dcd.Status.Conditions = []metav1.Condition{{
			Type:               nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable,
			Status:             metav1.ConditionTrue,
			Reason:             "Ready",
			LastTransitionTime: metav1.Now(),
		}}
		Expect(k8sClient.Status().Update(ctx, dcd)).To(Succeed())
	}
}
