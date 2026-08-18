/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"reflect"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPowerGateRendererAppliesAfterFinalPodRender(t *testing.T) {
	t.Log("Build one DGD-owned worker with an opaque custom command")
	dgdUID := types.UID("dgd-uid")
	controller := true
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph-decode",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoGraphDeploymentName: "graph",
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.Group + "/v1alpha1",
				Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
				Name:       "graph",
				UID:        dgdUID,
				Controller: &controller,
			}},
		},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:    commonconsts.MainContainerName,
						Image:   "dynamo:test",
						Command: []string{"/bin/sh", "-c"},
						Args:    []string{"setup && exec python3 -m dynamo.vllm | tee /tmp/backend.log"},
					}},
				}},
			},
		},
	}
	staticDGD := &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: dgdUID},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{
			ComponentName: "decode",
			ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
		}, {
			ComponentName: "frontend",
			ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
		}}},
	}

	t.Log("Render the complete static baseline without a DGPB")
	staticRenderer := testPowerGateRenderer(t, staticDGD)
	staticPod, err := staticRenderer.generatePodTemplateSpec(context.Background(), dcd, dynamo.RoleMain, noContainerGPUs())
	if err != nil {
		t.Fatalf("generatePodTemplateSpec() static error = %v", err)
	}
	staticMain := staticPod.Spec.Containers[0]

	t.Log("Publish the exact DGD-owned component bounds and render transactionally")
	transactionalDGD := staticDGD.DeepCopy()
	transactionalDGD.Annotations = map[string]string{
		nvidiacomv1beta1.DynamoGraphPowerControlModeAnnotation: nvidiacomv1beta1.DynamoGraphPowerControlModeTransactionalReplicaFence,
	}
	missingBudgetRenderer := testPowerGateRenderer(t, transactionalDGD)
	if _, err := missingBudgetRenderer.generatePodTemplateSpec(context.Background(), dcd, dynamo.RoleMain, noContainerGPUs()); err == nil {
		t.Fatal("generatePodTemplateSpec() missing DGPB error = nil, want fail closed")
	}
	dgpb := &nvidiacomv1beta1.DynamoGraphPowerBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1beta1.GroupVersion.String(),
				Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
				Name:       "graph",
				UID:        dgdUID,
				Controller: &controller,
			}},
		},
		Status: nvidiacomv1beta1.DynamoGraphPowerBudgetStatus{
			DGDUID: string(dgdUID),
			Components: []nvidiacomv1beta1.DynamoGraphPowerBudgetComponentStatus{{
				Name:                       "decode",
				PhysicalGPUsPerReplica:     2,
				InGateBoundWattsPerGPU:     350,
				UnenforcedBoundWattsPerGPU: 700,
			}},
		},
	}
	transactionalRenderer := testPowerGateRenderer(t, transactionalDGD, dgpb)
	transactionalPod, err := transactionalRenderer.generatePodTemplateSpec(context.Background(), dcd, dynamo.RoleMain, noContainerGPUs())
	if err != nil {
		t.Fatalf("generatePodTemplateSpec() transactional error = %v", err)
	}

	t.Log("Verify only the structural prefix was added to final command and args")
	transactionalMain := transactionalPod.Spec.Containers[0]
	wantCommand := append([]string{dynamo.PowerGateExecutable, dynamo.PowerGateSeparator}, staticMain.Command...)
	if !reflect.DeepEqual(transactionalMain.Command, wantCommand) {
		t.Fatalf("command = %#v, want %#v", transactionalMain.Command, wantCommand)
	}
	if !reflect.DeepEqual(transactionalMain.Args, staticMain.Args) {
		t.Fatalf("args = %#v, want unchanged %#v", transactionalMain.Args, staticMain.Args)
	}
	assertRenderedPowerGateEnv(t, transactionalMain.Env)

	t.Log("Reject a mutable child type that attempts to suppress the transactional gate")
	mutatedDCD := dcd.DeepCopy()
	mutatedDCD.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
	if _, err := transactionalRenderer.generatePodTemplateSpec(context.Background(), mutatedDCD, dynamo.RoleMain, noContainerGPUs()); err == nil {
		t.Fatal("generatePodTemplateSpec() mutable child classification error = nil")
	}

	t.Log("Keep an authoritative nonworker parent component on the ungated path")
	frontendDCD := dcd.DeepCopy()
	frontendDCD.Name = "graph-frontend"
	frontendDCD.Spec.BackendFramework = ""
	frontendDCD.Spec.ComponentName = "frontend"
	frontendDCD.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
	frontendPod, err := transactionalRenderer.generatePodTemplateSpec(context.Background(), frontendDCD, dynamo.RoleMain, noContainerGPUs())
	if err != nil {
		t.Fatalf("generatePodTemplateSpec() transactional frontend error = %v", err)
	}
	if frontendPod.Spec.Containers[0].Command[0] == dynamo.PowerGateExecutable {
		t.Fatal("transactional frontend unexpectedly received the power gate")
	}

	t.Log("Reject a nonworker child mutation that attempts to create an ungated worker")
	mutatedFrontendDCD := frontendDCD.DeepCopy()
	mutatedFrontendDCD.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeDecode
	mutatedFrontendDCD.Spec.BackendFramework = string(dynamo.BackendFrameworkVLLM)
	if _, err := transactionalRenderer.generatePodTemplateSpec(context.Background(), mutatedFrontendDCD, dynamo.RoleMain, noContainerGPUs()); err == nil {
		t.Fatal("generatePodTemplateSpec() nonworker-to-worker mutation error = nil")
	}
}

func testPowerGateRenderer(t *testing.T, objects ...client.Object) *dcdWorkloadRenderer {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(v1beta1) error = %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(corev1) error = %v", err)
	}
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme(appsv1) error = %v", err)
	}
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()
	return newDCDWorkloadRenderer(
		reader,
		&configv1alpha1.OperatorConfiguration{
			Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		},
		&commoncontroller.RuntimeConfig{},
		&mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
	)
}

func assertRenderedPowerGateEnv(t *testing.T, env []corev1.EnvVar) {
	t.Helper()

	want := map[string]string{
		dynamo.PowerGateDGDUIDEnv:                 "dgd-uid",
		dynamo.PowerGateComponentEnv:              "decode",
		dynamo.PowerGateExpectedGPUCountEnv:       "2",
		dynamo.PowerGateInGateBoundWattsPerGPUEnv: "350",
	}
	for _, item := range env {
		if expected, exists := want[item.Name]; exists {
			if item.Value != expected {
				t.Fatalf("environment %q = %q, want %q", item.Name, item.Value, expected)
			}
			delete(want, item.Name)
		}
	}
	if len(want) != 0 {
		t.Fatalf("missing power gate environment: %#v", want)
	}
}
