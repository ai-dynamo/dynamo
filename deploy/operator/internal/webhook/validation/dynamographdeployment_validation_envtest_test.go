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

package validation_test

import (
	"fmt"
	"maps"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	k8sptr "k8s.io/utils/ptr"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

const (
	dgdAdmissionWorkerName      = "worker"
	dgdAdmissionUpperWorkerName = "WORKER"
	dgdAdmissionPriorityClass   = "high-priority"
)

const sglangBackendFramework = "sglang"

// dgdAdmissionTestCase keeps the native admission and post-admission expectations together.
type dgdAdmissionTestCase struct {
	name               string
	deployment         runtime.Object
	oldDeployment      runtime.Object
	oldBeforeUpdate    runtime.Object
	mutateRequest      func(*testing.T, map[string]any) // mutates the source-version request map
	groveDisabled      bool                             // disables the configured Grove pathway
	lpxDisabled        bool                             // disables the LPX integration
	checkpointOff      bool                             // disables checkpoint creation and restore
	seedWithoutWebhook bool                             // seeds oldDeployment without validating it
	terminating        bool                             // deletes the seeded object so the update runs while it terminates
	username           string                           // supplies the admission request identity

	wantSchemaErr      string
	wantCELErr         string
	wantAdmissionErrs  []string
	wantWebhookErrs    []string
	wantWarnings       []string
	notWantErr         string
	wantPodAnnotations map[string]string
	wantOriginVersion  string
	wantProvider       string
	wantRoleReplicas   map[string]int32
	wantReplicas       map[string]*int32
}

func TestDynamoGraphDeploymentValidator_Validate(t *testing.T) {
	const updatedSidecarImage = "runtime:1.6.0"
	const initialSidecarImage = "runtime:1.5.0"
	const unrelatedInitContainerName = "setup"
	const customSidecarImage = "runtime:custom"
	const trtllmBackendFramework = "trtllm"

	longDGDName := "test-graph-" + strings.Repeat("x", 50)
	boundaryComponentName := "w" + strings.Repeat("x", 36)
	tooLongComponentName := boundaryComponentName + "x"

	tests := []dgdAdmissionTestCase{
		{name: "vLLM multi-GPU snapshot failover is admitted", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), "vllm")
		})},
		{name: "SGLang multi-GPU snapshot failover is admitted", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			dgd.Spec.BackendFramework = sglangBackendFramework
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), sglangBackendFramework)
		})},
		{name: "TRT-LLM snapshot failover is rejected at admission", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			dgd.Spec.BackendFramework = trtllmBackendFramework
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), trtllmBackendFramework)
		}), wantWebhookErrs: []string{"spec.components[1].experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "inferred TRT-LLM snapshot failover is rejected at admission", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			dgd.Spec.BackendFramework = ""
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), trtllmBackendFramework)
		}), wantWebhookErrs: []string{"spec.components[1].experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "TRT-LLM snapshot without failover remains admitted", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			dgd.Spec.BackendFramework = trtllmBackendFramework
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, trtllmBackendFramework)
			worker.Experimental.Failover = nil
		})},
		{name: "TRT-LLM snapshot failover cannot be enabled on update", oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			dgd.Spec.BackendFramework = trtllmBackendFramework
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, trtllmBackendFramework)
			worker.Experimental.Failover = nil
		}), deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			dgd.Spec.BackendFramework = trtllmBackendFramework
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), trtllmBackendFramework)
		}), wantWebhookErrs: []string{"spec.components[1].experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "snapshot failover rejects conflicting backend selection", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), sglangBackendFramework)
		}), wantWebhookErrs: []string{"spec.components[1].experimental.failover: Forbidden: Snapshot-backed intra-pod failover requires a consistent vLLM or SGLang backend"}},
		{name: "automatic snapshot failover cannot use Immediate startup", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, "vllm")
			worker.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicyImmediate
		}), wantWebhookErrs: []string{"spec.components[1].experimental.checkpoint.startupPolicy: Forbidden: Snapshot-backed intra-pod failover requires WaitForCheckpoint for automatic capture"}},
		{name: "blank checkpointRef cannot bypass automatic snapshot failover startup policy", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, "vllm")
			worker.Experimental.Checkpoint.CheckpointRef = k8sptr.To(" \t ")
			worker.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicyImmediate
		}), wantWebhookErrs: []string{"spec.components[1].experimental.checkpoint.startupPolicy: Forbidden: Snapshot-backed intra-pod failover requires WaitForCheckpoint for automatic capture"}},
		{name: "snapshot failover rejects a helper capture target", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, "vllm")
			worker.Experimental.Checkpoint.TargetContainerName = "helper"
		}), wantWebhookErrs: []string{`spec.components[1].experimental.checkpoint.targetContainerName: Invalid value: "helper": must be main for intra-pod failover`}},
		{name: "snapshot failover rejects multiple nodes", deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, "vllm")
			worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
		}), wantWebhookErrs: []string{"spec.components[1].multinode: Forbidden: Snapshot-backed intra-pod failover requires a single node setup"}},
		{name: "snapshot failover update cannot enable Immediate startup", oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			enableBetaSnapshotFailover(dgd.GetComponentByName("worker"), sglangBackendFramework)
			dgd.Spec.BackendFramework = sglangBackendFramework
		}), deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
			enableBetaContainerDiscovery(dgd)
			worker := dgd.GetComponentByName("worker")
			enableBetaSnapshotFailover(worker, sglangBackendFramework)
			dgd.Spec.BackendFramework = sglangBackendFramework
			worker.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicyImmediate
		}), wantWebhookErrs: []string{"spec.components[1].experimental.checkpoint.startupPolicy: Forbidden: Snapshot-backed intra-pod failover requires WaitForCheckpoint for automatic capture"}},
		// Sidecar mode is derived from live init-container names in both API versions.
		{name: "beta unrelated init container retains standard mode", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "beta regular runtime container retains standard mode", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "runtime", Image: "helper:latest"})
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "beta adding runtime activates sidecar validation", oldDeployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		}), deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
		}), wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`}},
		{name: "beta removing runtime returns to main", oldDeployment: nativeDGDForAdmission(t, false, nil), deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "beta renaming runtime returns to main", oldDeployment: nativeDGDForAdmission(t, false, nil), deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "beta removing runtime revalidates engine version", oldDeployment: nativeDGDForAdmission(t, false, nil), deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
		}), wantWebhookErrs: []string{`spec.components[1].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`}},
		{name: "alpha unrelated init container retains standard mode", deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "alpha regular runtime container retains standard mode", deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "runtime", Image: "helper:latest"})
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "alpha adding runtime activates sidecar validation", oldDeployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		}), deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
		}), wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`}},
		{name: "alpha removing runtime returns to main", oldDeployment: nativeDGDForAdmission(t, true, nil), deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "alpha renaming runtime returns to main", oldDeployment: nativeDGDForAdmission(t, true, nil), deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "alpha removing runtime revalidates engine version", oldDeployment: nativeDGDForAdmission(t, true, nil), deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
		}), wantWebhookErrs: []string{`spec.services[worker].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`}},
		{name: "native sidecar beta rejects gms create",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects failover create",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.components[1].experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects gms update", oldDeployment: nativeDGDForAdmission(t, false, nil),
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects failover update", oldDeployment: nativeDGDForAdmission(t, false, nil),
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.components[1].experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects multinode", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
		}), wantWebhookErrs: []string{`spec.components[1].multinode: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar alpha rejects gms create",
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar alpha rejects failover create",
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.components[1].experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar alpha rejects gms update", oldDeployment: nativeDGDForAdmission(t, true, nil),
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar alpha rejects failover update", oldDeployment: nativeDGDForAdmission(t, true, nil),
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.components[1].experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar alpha rejects multinode", deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
		}), wantWebhookErrs: []string{`spec.components[1].multinode: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar requires engine main", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Name = "engine"
		}), wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.containers: Required value: main engine container is required for component "worker" with a runtime init container`}},
		{name: "native sidecar rejects non-worker component type",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
			}), wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].name: Forbidden: is supported only for worker, prefill, and decode components`}},

		{name: "native sidecar beta custom runtime image create needs override",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].Image = customSidecarImage
			}),
			wantWebhookErrs: []string{`spec.components[1].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`},
		},
		{name: "native sidecar beta custom runtime image update needs override", oldDeployment: nativeDGDForAdmission(t, false, nil),
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].Image = customSidecarImage
			}),
			wantWebhookErrs: []string{`spec.components[1].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`},
		},
		{name: "native sidecar alpha custom runtime image create needs override",
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].Image = customSidecarImage
			}),
			wantWebhookErrs: []string{`spec.services[worker].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`},
		},
		{name: "native sidecar alpha custom runtime image update needs override", oldDeployment: nativeDGDForAdmission(t, true, nil),
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].Image = customSidecarImage
			}),
			wantWebhookErrs: []string{`spec.services[worker].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag`},
		},
		// Native-sidecar admission, including alpha conversion and unchanged update invariants.
		{name: "native sidecar beta create", deployment: nativeDGDForAdmission(t, false, nil)},
		{name: "native sidecar beta update", oldDeployment: nativeDGDForAdmission(t, false, nil), deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Image = updatedSidecarImage
		})},
		{name: "native sidecar beta disabled checkpoint", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: false}}
		})},
		{name: "native sidecar beta rejects nonrestartable init create",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar beta rejects checkpoint create",
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar beta rejects nonrestartable init update", oldDeployment: nativeDGDForAdmission(t, false, nil),
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar beta rejects checkpoint update", oldDeployment: nativeDGDForAdmission(t, false, nil),
			deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar alpha create", deployment: nativeDGDForAdmission(t, true, nil)},
		{name: "native sidecar alpha update", oldDeployment: nativeDGDForAdmission(t, true, nil), deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Image = updatedSidecarImage
		})},
		{name: "native sidecar alpha disabled checkpoint", deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: false}}
		})},
		{name: "native sidecar alpha rejects nonrestartable init create",
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar alpha rejects checkpoint create",
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar alpha rejects nonrestartable init update", oldDeployment: nativeDGDForAdmission(t, true, nil),
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.components[1].podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar alpha rejects checkpoint update", oldDeployment: nativeDGDForAdmission(t, true, nil),
			deployment: nativeDGDForAdmission(t, true, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar with regular frontend", deployment: nativeDGDForAdmission(t, false, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.FrontendSidecar = k8sptr.To("http-frontend")
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "http-frontend", Image: "frontend:1.5.0"})
		})},
		// Baseline create-path rules.
		{
			name:        "valid deployment with components",
			lpxDisabled: true,
			deployment:  betaDGDForAdmission(nil),
		},
		{
			name: "beta component main image is required when pod template is absent on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = nil
			}),
			wantWebhookErrs: []string{"spec.components[1].podTemplate.spec.containers: Required value: is required"},
		},
		{
			name: "alpha service main image is required when extra pod spec is absent on create",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ExtraPodSpec = nil
			}),
			wantWebhookErrs: []string{"spec.services[worker].extraPodSpec.mainContainer.image: Required value: is required"},
		},
		{
			name: "component custom image requires runtime version override",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: customRuntimeImage}},
				}}
			}),
			wantWebhookErrs: []string{"spec.components[1].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag"},
		},
		{
			name: "alpha component custom image uses source-version path",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = customRuntimeImage
			}),
			wantWebhookErrs: []string{"spec.services[worker].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag"},
		},
		{
			name:               "unchanged legacy beta component runtime version is ratcheted on update",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
				dgd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name:               "unchanged legacy alpha service runtime version is ratcheted on update",
			seedWithoutWebhook: true,
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = customRuntimeImage
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = customRuntimeImage
				dgd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name: "existing v1beta1 DGD with a 1.4 Go EPP remains editable after operator upgrade",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Labels = map[string]string{"updated": "true"}
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
		},
		{
			name: "existing v1alpha1 DGD with a 1.4 Go EPP remains editable after operator upgrade",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Labels = map[string]string{"updated": "true"}
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
		},
		{
			name:               "unrelated DGD update ratchets an identical pre-existing native image and eppConfig mismatch",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = frontendImage150
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Labels = map[string]string{"updated": "true"}
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = frontendImage150
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
		},
		{
			name:               "unrelated alpha DGD update ratchets an identical pre-existing native image and eppConfig mismatch",
			seedWithoutWebhook: true,
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = frontendImage150
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Labels = map[string]string{"updated": "true"}
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = frontendImage150
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
		},
		{
			// type is immutable once set, so the only transition into EPP starts
			// from an unset type; it must still validate the complete contract.
			name: "setting a previously unset DGD component type to EPP validates the complete runtime contract",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = ""
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			wantWebhookErrs: []string{"spec.components[1].eppConfig: Required value: is required for legacy Go EPP images with runtime version earlier than 1.5.0"},
		},
		{
			name: "DGD atomically migrates from a 1.4 Go EPP to a 1.5 Rust EPP",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
		},
		{
			name: "DGD atomically rolls back from a 1.5 Rust EPP to a 1.4 Go EPP",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
		},
		{
			name: "beta component image change to custom requires runtime version override",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.RuntimeVersionOverride = ""
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.RuntimeVersionOverride = ""
				worker.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
			}),
			wantWebhookErrs: []string{"spec.components[1].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag"},
		},
		{
			name:               "changing a legacy alpha custom image requires runtime version override",
			seedWithoutWebhook: true,
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = customRuntimeImage
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = "registry.example/runtime:other-custom"
			}),
			wantWebhookErrs: []string{"spec.services[worker].runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag"},
		},
		{
			name: "no components",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components = nil
			}),
			wantWebhookErrs: []string{"spec.components: Required value: must have at least one component"},
		},
		{
			name: "component replicas must be non-negative",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Replicas = k8sptr.To(int32(-1))
			}),
			wantAdmissionErrs: []string{
				"spec.components[1].replicas: Invalid value: -1: spec.components[1].replicas in body should be greater than or equal to 0",
			},
		},
		{
			name:          "component minAvailable requires Grove",
			groveDisabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			wantWebhookErrs: []string{"spec.components[1].minAvailable: Forbidden: is currently supported only for Grove-backed DynamoGraphDeployment components"},
		},
		{
			name: "restart on create is rejected by CEL before webhook validation",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = &nvidiacomv1beta1.Restart{
					ID: "roll",
					Strategy: &nvidiacomv1beta1.RestartStrategy{
						Type:  nvidiacomv1beta1.RestartStrategyTypeParallel,
						Order: []string{"frontend", "worker"},
					},
				}
			}),
			wantCELErr: "spec: Invalid value: spec.restart must be unset on create; set spec.restart.id after creation to request a restart",
		},
		{
			name: "component topology constraint requires deployment topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantWebhookErrs: []string{"spec.topologyConstraint: Required value: is required when any component topology constraint is set"},
		},
		{
			name:          "inter-pod GMS requires Grove",
			groveDisabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService.mode: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "inter-pod GMS requires vLLM backend",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.BackendFramework = "sglang"
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.gpuMemoryService.mode: Invalid value: \"InterPod\": the inter-pod GMS layout is currently supported only for vLLM (detected backend: sglang)"},
		},
		{
			name: "KV transfer policy selector is rejected by CEL before webhook validation",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						Domain: "rack",
					},
				}
			}),
			wantCELErr: "spec.experimental.kvTransferPolicy: Invalid value: exactly one of labelKey or clusterTopologyName is required",
		},
		{
			name: "intra-pod failover requires container discovery",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode: nvidiacomv1beta1.GMSModeIntraPod,
				}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/dynamo-kube-discovery-mode]: Invalid value: "": must be "container" when intra-pod failover is configured`},
		},
		{
			name: "checkpoint job with checkpointRef is rejected by CEL before webhook validation",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: k8sptr.To("existing-checkpoint"),
						Job:           &nvidiacomv1beta1.ComponentCheckpointJobConfig{},
					},
				}
			}),
			wantCELErr: "spec.components[1].experimental.checkpoint: Invalid value: checkpoint.job cannot be set when checkpointRef is specified",
		},
		{
			name: "GMS requires GPU resources on the main container",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode: nvidiacomv1beta1.GMSModeIntraPod,
					},
				}
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.gpuMemoryService: Forbidden: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1"},
		},

		// Cross-version schema and conversion boundaries.
		{
			name: "v1beta1 component name is required by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].ComponentName = ""
			}),
			wantSchemaErr: `spec.components[1].name: Invalid value: "": spec.components[1].name in body should be at least 1 chars long`,
		},
		{
			name:       "v1alpha1 converted empty service map key is accepted",
			deployment: alphaDGDForAdmissionWithServiceNames(""),
		},
		{
			name: "v1beta1 component names are unique case-insensitively in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
			}),
			wantCELErr: "spec.components: Invalid value: component names must be unique case-insensitively",
		},
		{
			name:       "v1alpha1 converted service names may collide case-insensitively",
			deployment: alphaDGDForAdmissionWithServiceNames(dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName),
		},
		{
			name:          "v1beta1 introducing case-insensitive component names is rejected by CEL on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: dgdAdmissionWithLabel(t, betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionUpperWorkerName
			})),
			wantCELErr: "spec.components: Invalid value: component names must be unique case-insensitively",
		},
		{
			name:          "v1alpha1 case-insensitive service names remain updateable",
			oldDeployment: alphaDGDForAdmissionWithServiceNames(dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName),
			deployment:    dgdAdmissionWithLabel(t, alphaDGDForAdmissionWithServiceNames(dgdAdmissionWorkerName, dgdAdmissionUpperWorkerName)),
		},
		{
			name:          "v1alpha1 empty service name remains updateable",
			oldDeployment: alphaDGDForAdmissionWithServiceNames(""),
			deployment:    dgdAdmissionWithLabel(t, alphaDGDForAdmissionWithServiceNames("")),
		},
		{
			name: "v1beta1 compilation cache mount requires a PVC name in the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).CompilationCache = &nvidiacomv1beta1.CompilationCacheConfig{}
			}),
			wantSchemaErr: `spec.components[1].compilationCache.pvcName: Invalid value: "": spec.components[1].compilationCache.pvcName in body should be at least 1 chars long`,
		},
		{
			name: "v1alpha1 converted compilation cache mount with an empty PVC name is rejected",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].VolumeMounts = []nvidiacomv1alpha1.VolumeMount{{
					UseAsCompilationCache: true,
				}}
			}),
			mutateRequest: setAlphaCompilationCacheVolumeNameEmpty,
			wantSchemaErr: `spec.services.worker.volumeMounts[0].name: Required value`,
		},
		{
			name: "v1beta1 sidecars must provide an image in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}, {Name: "metrics"}},
				}}
			}),
			wantCELErr: "spec.components[1].podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 converted sidecar without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"}, PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name: "v1alpha1 frontend sidecar without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].FrontendSidecar = &nvidiacomv1alpha1.FrontendSidecarSpec{}
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"}, PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name: "v1beta1 init containers must provide an image in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantCELErr: "spec.components[1].podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 converted init container without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"}, PodSpec: &corev1.PodSpec{
					InitContainers: []corev1.Container{{Name: "prep"}},
				}}
			}),
		},
		{
			name: "v1beta1 pod template backend annotation is validated by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}}},
				}
			}),
			wantCELErr: "spec.components[1].podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			// Component pod metadata must survive structural pruning.
			name: "v1beta1 discovery annotation survives the component pod template API server round trip",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
						consts.KubeAnnotationDynamoKubeDiscoveryMode:        "container",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}}},
				}
			}),
			wantPodAnnotations: map[string]string{
				consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
				consts.KubeAnnotationDynamoKubeDiscoveryMode:        "container",
			},
		},
		{
			name: "v1alpha1 converted extraPodMetadata annotation does not receive v1beta1 CEL validation",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{
					Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "typo"},
				}
			}),
		},
		{
			name: "v1alpha1 invalid service annotation remains rejected by the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
				}
			}),
			wantWebhookErrs: []string{`spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend]: Invalid value: "invalid": must be "mp" or "ray"`},
		},
		{
			name: "v1beta1 frontend sidecar must reference an existing container",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.FrontendSidecar = k8sptr.To("missing")
				worker.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
				}}
			}),
			wantWebhookErrs: []string{`spec.components[1].frontendSidecar: Invalid value: "missing": must match a podTemplate.spec.containers name`},
		},
		{
			name:       "valid v1alpha1 deployment reaches the webhook",
			deployment: alphaDGDForAdmission(nil),
		},
		{
			name:          "valid v1beta1 update reaches the webhook",
			oldDeployment: betaDGDForAdmission(nil),
			deployment:    dgdAdmissionWithLabel(t, betaDGDForAdmission(nil)),
		},
		{
			name: "v1beta1 pod template container counts are not artificially bounded",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				podTemplate := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
				}}
				for i := range 32 {
					podTemplate.Spec.Containers = append(podTemplate.Spec.Containers, corev1.Container{
						Name:  fmt.Sprintf("sidecar-%d", i),
						Image: "sidecar:latest",
					})
					podTemplate.Spec.InitContainers = append(podTemplate.Spec.InitContainers, corev1.Container{
						Name:  fmt.Sprintf("init-%d", i),
						Image: "init:latest",
					})
				}
				betaWorkerComponent(dgd).PodTemplate = podTemplate
			}),
		},

		// Deprecated minima warn even when RollingRecreate or OnDelete is selected.
		{
			name: "deprecated standalone minimum warns with implicit RollingRecreate",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			wantWarnings: []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`},
		},
		{
			name: "deprecated scaling group minimum warns with OnDelete",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "OnDelete"}
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				worker.MinAvailable = k8sptr.To(int32(1))
			}),
			wantWarnings: []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.minAvailable and migrate all component minima together without changing their effective values`},
		},

		{
			name:          "legacy minima do not repeat warnings on an unrelated label edit",
			oldDeployment: legacyMinimumDGDForAdmission(1),
			deployment:    dgdAdmissionWithLabel(t, legacyMinimumDGDForAdmission(1)),
		},
		{
			name:          "legacy minima do not repeat warnings on a replica-only update",
			oldDeployment: legacyMinimumDGDForAdmission(1),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.MinAvailable = k8sptr.To(int32(1))
				worker.Replicas = k8sptr.To(int32(3))
			}),
		},
		{
			name: "legacy alpha minima do not repeat warnings on a replica-only update",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.MinAvailable = k8sptr.To(int32(1))
				worker.Replicas = k8sptr.To(int32(3))
			}),
		},
		{
			name:          "migrating back to legacy minima warns about their introduction",
			oldDeployment: nativeMinimumDGDForAdmission(2),
			deployment:    legacyMinimumDGDForAdmission(2),
			wantWarnings:  []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`},
		},

		// Native availability and migration use the same production schema/conversion/webhook chain.
		{
			name: "native standalone minimum is accepted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":1}}`)
			}),
		},
		{
			name: "native multinode minimum is accepted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				worker.ProviderOverride = groveProviderOverride("", `{"minAvailable":1}`)
			}),
		},
		{
			name: "native alpha minimum is accepted through conversion",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ProviderOverride = alphaGroveProviderOverride(provideroverride.TargetPodCliqueTemplateSpec, `{"spec":{"minAvailable":1}}`)
			}),
		},
		{
			name: "native minimum must fit positive replicas",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(1))
				worker.ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":2}}`)
			}),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: 2: minAvailable must be less than or equal to replicas unless replicas is 0`},
		},
		{
			name: "native minimum allows scale to zero",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(0))
				worker.ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":2}}`)
			}),
		},
		{
			name: "native zero minimum is not hidden by defaulting",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":0}}`)
			}),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: null: must be a positive 32-bit integer`},
		},
		{
			name: "native null minimum is not hidden by defaulting",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":null}}`)
			}),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: null: is required`},
		},
		{
			name: "P and D cannot mix legacy and native minima",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].MinAvailable = k8sptr.To(int32(1))
				betaWorkerComponent(dgd).ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":1}}`)
			}),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value: Forbidden: cannot mix provider-native minAvailable with deprecated component minAvailable; migrate all components together`},
			wantWarnings:    []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`},
		},
		{
			name:          "all legacy minima migrate without changing their immutable values",
			oldDeployment: legacyMinimumDGDForAdmission(2),
			deployment:    nativeMinimumDGDForAdmission(2),
		},
		{
			name:            "migration cannot change effective minimum",
			oldDeployment:   legacyMinimumDGDForAdmission(2),
			deployment:      nativeMinimumDGDForAdmission(3),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: 3: minAvailable is immutable after creation`},
		},
		{
			name:            "native minimum cannot change after creation",
			oldDeployment:   nativeMinimumDGDForAdmission(2),
			deployment:      nativeMinimumDGDForAdmission(3),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: 3: minAvailable is immutable after creation`},
		},
		{
			name:          "removing native minimum cannot bypass immutability",
			oldDeployment: nativeMinimumDGDForAdmission(2),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				for i := range dgd.Spec.Components {
					dgd.Spec.Components[i].ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":1}}`)
				}
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(4))
				worker.ProviderOverride = nil
			}),
			wantWebhookErrs: []string{`spec.components[1].providerOverride.value.spec.minAvailable: Invalid value: 1: minAvailable is immutable after creation`},
		},
		{
			name: "member role cannot set an independent minimum",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLeader, ProviderOverride: groveProviderOverride("", `{"spec":{"minAvailable":1}}`)}, {Name: nvidiacomv1beta1.ComponentRoleWorker}}
			}),
			wantWebhookErrs: []string{`spec.components[1].roles[0].providerOverride.value.spec.minAvailable: Forbidden: minAvailable (1) belongs to the owning component`},
		},

		// Only explicit Coherent intent emits warnings about full component disruption.
		{
			name:         "explicit Coherent native minimum can take the entire worker down",
			deployment:   minimumAvailabilityDGDForAdmission(2, 2, "Coherent"),
			wantWarnings: []string{`spec.components[1] ("worker"): Coherent updates may temporarily make this entire component unavailable (replicas=2, minAvailable=2, maxUnavailable=2); provision spare serving capacity before updating`},
		},
		{
			name:         "explicit Coherent default minimum warns for a single worker",
			deployment:   minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			wantWarnings: []string{`spec.components[1] ("worker"): Coherent updates may temporarily make this entire component unavailable (replicas=1, minAvailable=1, maxUnavailable=1); provision spare serving capacity before updating`},
		},
		{
			name:       "native minimum alone does not warn or enable Coherent",
			deployment: minimumAvailabilityDGDForAdmission(2, 2, ""),
		},
		{
			name:       "Coherent partial minimum leaves serving capacity",
			deployment: minimumAvailabilityDGDForAdmission(1, 4, "Coherent"),
		},
		{
			name:       "Coherent scaled-to-zero component does not warn",
			deployment: minimumAvailabilityDGDForAdmission(2, 0, "Coherent"),
		},
		{
			name:          "Coherent warning is not repeated on an unrelated label edit",
			oldDeployment: minimumAvailabilityDGDForAdmission(2, 2, "Coherent"),
			deployment:    dgdAdmissionWithLabel(t, minimumAvailabilityDGDForAdmission(2, 2, "Coherent")),
		},
		{
			name:          "Coherent replica reduction introduces full component disruption risk",
			oldDeployment: minimumAvailabilityDGDForAdmission(1, 2, "Coherent"),
			deployment:    minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			wantWarnings:  []string{`spec.components[1] ("worker"): Coherent updates may temporarily make this entire component unavailable (replicas=1, minAvailable=1, maxUnavailable=1); provision spare serving capacity before updating`},
		},
		{
			name:          "Coherent scale from zero introduces full component disruption risk",
			oldDeployment: minimumAvailabilityDGDForAdmission(1, 0, "Coherent"),
			deployment:    minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			wantWarnings:  []string{`spec.components[1] ("worker"): Coherent updates may temporarily make this entire component unavailable (replicas=1, minAvailable=1, maxUnavailable=1); provision spare serving capacity before updating`},
		},
		{
			name:          "Coherent scale up leaves serving capacity without warning",
			oldDeployment: minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			deployment:    minimumAvailabilityDGDForAdmission(1, 2, "Coherent"),
		},
		{
			name: "Coherent explicit default replica count does not repeat the warning",
			oldDeployment: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				dgd := minimumAvailabilityDGDForAdmission(1, 1, "Coherent")
				betaWorkerComponent(dgd).Replicas = nil
				return dgd
			}(),
			deployment: minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
		},
		{
			name:          "Coherent omitted default replica count does not repeat the warning",
			oldDeployment: minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			deployment: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				dgd := minimumAvailabilityDGDForAdmission(1, 1, "Coherent")
				betaWorkerComponent(dgd).Replicas = nil
				return dgd
			}(),
		},
		{
			name:          "Coherent pod template edit still warns about existing full component disruption risk",
			oldDeployment: minimumAvailabilityDGDForAdmission(1, 1, "Coherent"),
			deployment: func() *nvidiacomv1beta1.DynamoGraphDeployment {
				dgd := minimumAvailabilityDGDForAdmission(1, 1, "Coherent")
				betaWorkerComponent(dgd).PodTemplate.Spec.Containers[0].Image += "-coherent-update"
				return dgd
			}(),
			wantWarnings: []string{`spec.components[1] ("worker"): Coherent updates may temporarily make this entire component unavailable (replicas=1, minAvailable=1, maxUnavailable=1); provision spare serving capacity before updating`},
		},
		{
			name:          "removing the Coherent annotation restores RollingRecreate without warning",
			oldDeployment: minimumAvailabilityDGDForAdmission(2, 2, "Coherent"),
			deployment:    minimumAvailabilityDGDForAdmission(2, 2, ""),
		},

		// Replica availability rules.
		{
			name: "v1beta1 replicas below minAvailable are rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(1))
				worker.MinAvailable = k8sptr.To(int32(2))
			}),
			wantCELErr: "spec.components[1]: Invalid value: minAvailable must be less than or equal to replicas unless replicas is 0",
		},
		{
			name: "v1beta1 valid replicas and minAvailable reach the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(2))
				worker.MinAvailable = k8sptr.To(int32(1))
			}),
			wantWarnings: []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`},
		},
		{
			name: "v1beta1 unchanged minAvailable update reaches the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
		},
		{
			name: "v1beta1 changed effective minAvailable is rejected by admission",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(2))
			}),
			wantWebhookErrs: []string{"spec.components[1].minAvailable: Invalid value: 2: minAvailable is immutable after creation"},
		},
		{
			name:          "v1alpha1 componentType change is rejected by CEL",
			oldDeployment: alphaDGDForAdmission(nil),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services[dgdAdmissionWorkerName].ComponentType = consts.ComponentTypeEPP
			}),
			wantCELErr: "spec.services[worker]: Invalid value: componentType is immutable after it is set",
		},
		{
			name:          "v1beta1 type change is rejected by CEL",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).ComponentType = nvidiacomv1beta1.ComponentTypeEPP
			}),
			wantCELErr: "spec.components[1]: Invalid value: type is immutable after it is set",
		},
		{
			name: "v1beta1 removed minAvailable update is restored by defaulting",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: betaDGDForAdmission(nil),
		},
		{
			name: "v1beta1 power tuple is accepted on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
		},
		{
			name: "v1beta1 power annotation with intra-pod GMS DRA is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaIntraPodGMS(betaWorkerComponent(dgd))
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].experimental.gpuMemoryService: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 power annotation with inter-pod GMS DRA is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].experimental.gpuMemoryService: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 power annotation with DRA claim template is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				setBetaWorkerResourceClaim(dgd, corev1.PodResourceClaim{
					Name:                      "gpu",
					ResourceClaimTemplateName: k8sptr.To("gpu-template"),
				})
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec.containers[0].resources.claims: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 power annotation with direct DRA claim is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				setBetaWorkerResourceClaim(dgd, corev1.PodResourceClaim{
					Name:              "gpu",
					ResourceClaimName: k8sptr.To("gpu-claim"),
				})
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec.containers[0].resources.claims: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 DRA claim without power annotation remains accepted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerResourceClaim(dgd, corev1.PodResourceClaim{
					Name:                      "device",
					ResourceClaimTemplateName: k8sptr.To("device-template"),
				})
			}),
		},
		{
			name: "v1beta1 power annotation with zero value is rejected on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "0", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "0": must be greater than zero`,
			},
		},
		{
			name: "v1beta1 power annotation with negative value is rejected on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "-300", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "-300": must be greater than zero`,
			},
		},
		{
			name: "v1beta1 power annotation with non-integer value is rejected on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300W", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "300W": must be a decimal integer`,
			},
		},
		{
			name: "v1beta1 power annotation with init-container DRA claim is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				worker := betaWorkerComponent(dgd)
				worker.PodTemplate.Spec.InitContainers = []corev1.Container{
					{
						Name:  "init-gpu",
						Image: "nvcr.io/nvidia/cuda:12.0-base",
						Resources: corev1.ResourceRequirements{
							Claims: []corev1.ResourceClaim{{Name: "gpu"}},
						},
					},
				}
				worker.PodTemplate.Spec.ResourceClaims = []corev1.PodResourceClaim{
					{Name: "gpu", ResourceClaimTemplateName: k8sptr.To("gpu-template")},
				}
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec.initContainers[0].resources.claims: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 power annotation cannot be added to a DRA component",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerResourceClaim(dgd, corev1.PodResourceClaim{
					Name:                      "device",
					ResourceClaimTemplateName: k8sptr.To("device-template"),
				})
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				setBetaWorkerResourceClaim(dgd, corev1.PodResourceClaim{
					Name:                      "device",
					ResourceClaimTemplateName: k8sptr.To("device-template"),
				})
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec.containers[0].resources.claims: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1beta1 power annotation change is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "350", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "350": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 power annotation addition is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				delete(betaWorkerComponent(dgd).PodTemplate.Annotations, consts.KubeAnnotationGPUPowerLimit)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "300": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 power annotation removal is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				delete(betaWorkerComponent(dgd).PodTemplate.Annotations, consts.KubeAnnotationGPUPowerLimit)
			}),
			wantWebhookErrs: []string{
				"spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: null: " + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 effective GPU change is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "2", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec.containers[0].resources.limits[nvidia.com/gpu]: Invalid value: "2": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 shadow request change is allowed when GPU limit is unchanged",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				betaWorkerComponent(dgd).PodTemplate.Spec.Containers[0].Resources.Requests = corev1.ResourceList{
					corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("8"),
				}
			}),
		},
		{
			name: "v1beta1 GPU sidecar addition is rejected for a power component",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				addBetaWorkerGPUSidecar(dgd)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec: Invalid value: "2": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 GPU sidecar removal is rejected for a power component",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				addBetaWorkerGPUSidecar(dgd)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].podTemplate.spec: Invalid value: "1": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 power node count change is rejected by the webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 3)
			}),
			wantWebhookErrs: []string{
				"spec.components[1].multinode.nodeCount: Invalid value: 3: " + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1beta1 unrelated power component update reaches webhook",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaWorkerPowerInputs(dgd, "300", "1", 2)
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(2))
				worker.PodTemplate.Spec.Containers[0].Image = "registry.example/runtime:1.2.0"
			}),
		},
		{
			name:          "v1beta1 unannotated component GPU change remains allowed",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
					corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("2"),
				}
			}),
		},
		{
			name: "v1alpha1 power annotation change is rejected after conversion by the webhook",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "350", "1", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[0].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "350": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1alpha1 power annotation with GMS DRA is rejected after conversion",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
				dgd.Spec.Services[dgdAdmissionWorkerName].GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
			wantWebhookErrs: []string{
				`spec.components[0].experimental.gpuMemoryService: Forbidden: cannot be combined with annotation "dynamo.nvidia.com/gpu-power-limit": power-aware planning does not support DRA-backed device allocation`,
			},
		},
		{
			name: "v1alpha1 effective GPU change is rejected after conversion by the webhook",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "2", 2)
			}),
			wantWebhookErrs: []string{
				`spec.components[0].podTemplate.spec.containers[0].resources.limits[nvidia.com/gpu]: Invalid value: "2": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1alpha1 power node count change is rejected after conversion by the webhook",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 3)
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode.nodeCount: Invalid value: 3: " + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name: "v1alpha1 unrelated power service update reaches webhook",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaWorkerPowerInputs(dgd, "300", "1", 2)
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.Replicas = k8sptr.To(int32(2))
				worker.ExtraPodSpec.MainContainer.Image = "registry.example/runtime:1.2.0"
			}),
		},

		// Grove forceScalingGroup rules.
		{
			name:          "component grove.forceScalingGroup requires Grove",
			groveDisabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.grove.forceScalingGroup: Forbidden: is currently supported only for Grove-backed DynamoGraphDeployment components"},
		},
		{
			name: "v1beta1 grove.forceScalingGroup on a single-node component reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
		},
		{
			name: "v1beta1 redundant grove.forceScalingGroup on a multinode component reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
		},

		// Compound component role rules.
		{
			name: "multinode component without explicit roles preserves implicit behavior",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
			}),
		},
		{
			name: "frontend component cannot be multinode",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name: "planner component cannot be multinode",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentType = nvidiacomv1beta1.ComponentTypePlanner
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name: "v1alpha1 frontend component cannot be multinode after conversion",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeFrontend
				worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name: "complete explicit multinode roles are admitted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoles(betaWorkerComponent(dgd), 4)
			}),
		},
		{
			name: "complete role pod templates are admitted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
			}),
		},
		{
			name: "role pod templates reject native sidecars in unsupported multinode layouts",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				setBetaExplicitMultinodeRoleTemplates(worker, 4)
				worker.Roles[0].PodTemplate.Spec.InitContainers = []corev1.Container{{
					Name: consts.RuntimeContainerName, Image: initialSidecarImage,
					RestartPolicy: k8sptr.To(corev1.ContainerRestartPolicyAlways),
				}}
			}),
			wantWebhookErrs: []string{`spec.components[1].roles[0].podTemplate.spec.initContainers[0].name: Forbidden: Dynamo sidecar mode is not supported in multinode or LPX role pod templates`},
		},
		{
			name: "role pod templates require a 1.6 or later Planner",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
				addBetaPlanner(dgd, "registry.example/planner:1.5.0", "")
			}),
			wantWebhookErrs: []string{`spec.components[1].roles: Forbidden: role-specific PodTemplates require every planner component to use Dynamo runtime 1.6.0 or later; planner component "planner" resolves to 1.5.0`},
		},
		{
			name: "role pod templates gate the Planner override and not engine runtimes",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
				addBetaPlanner(dgd, "registry.example/planner:1.5.0", "1.6.0")
			}),
		},
		{
			name: "role pod templates reject a Planner override below 1.6 despite its image tag",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
				addBetaPlanner(dgd, "registry.example/planner:1.6.0", "1.5.0")
			}),
			wantWebhookErrs: []string{`spec.components[1].roles: Forbidden: role-specific PodTemplates require every planner component to use Dynamo runtime 1.6.0 or later; planner component "planner" resolves to 1.5.0`},
		},
		{
			name: "switching to role pod templates rejects an existing pre-1.6 Planner",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				addBetaPlanner(dgd, "registry.example/planner:1.5.0", "")
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
				addBetaPlanner(dgd, "registry.example/planner:1.5.0", "")
			}),
			wantWebhookErrs: []string{`spec.components[1].roles: Forbidden: role-specific PodTemplates require every planner component to use Dynamo runtime 1.6.0 or later; planner component "planner" resolves to 1.5.0`},
		},
		{
			name: "switching to role pod templates can atomically upgrade the Planner",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				addBetaPlanner(dgd, "registry.example/planner:1.5.0", "")
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoleTemplates(betaWorkerComponent(dgd), 4)
				addBetaPlanner(dgd, "registry.example/planner:1.6.0", "")
			}),
		},
		{
			name: "alpha role pod templates reject a pre-1.6 Planner at the source path",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				setAlphaExplicitMultinodeRoleTemplates(dgd.Spec.Services[dgdAdmissionWorkerName], 4)
				addAlphaPlanner(dgd, "registry.example/planner:1.5.0", "")
			}),
			wantWebhookErrs: []string{`spec.services[worker].roles: Forbidden: role-specific PodTemplates require every planner component to use Dynamo runtime 1.6.0 or later; planner component "planner" resolves to 1.5.0`},
		},
		{
			name: "role pod template sidecars require images in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				setBetaExplicitMultinodeRoleTemplates(worker, 4)
				worker.Roles[0].PodTemplate.Spec.Containers = append(
					worker.Roles[0].PodTemplate.Spec.Containers,
					corev1.Container{Name: "metrics"},
				)
			}),
			wantCELErr: "spec.components[1].roles[0].podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name: "role pod template backend annotations are validated by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				setBetaExplicitMultinodeRoleTemplates(worker, 4)
				worker.Roles[0].PodTemplate.Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
				}
			}),
			wantCELErr: "spec.components[1].roles[0].podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name: "v1alpha1 explicit multinode roles convert and are admitted",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 4}
				worker.Roles = []nvidiacomv1alpha1.ComponentRoleSpec{
					{Name: nvidiacomv1alpha1.ComponentRoleLeader},
					{Name: nvidiacomv1alpha1.ComponentRoleWorker},
				}
			}),
		},
		{
			name: "roles require a component role schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
				}
			}),
			wantWebhookErrs: []string{"spec.components[1].roles: Forbidden: roles are supported only for component shapes that define a role schema; this release supports multinode and LPX components"},
		},
		{
			name: "explicit multinode roles require the complete role set",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
				}
			}),
			wantWebhookErrs: []string{`spec.components[1].roles: Required value: must contain the "worker" role`},
		},
		{
			name: "explicit multinode roles reject unknown role names",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: "coordinator"},
				}
			}),
			wantWebhookErrs: []string{
				`spec.components[1].roles[0].name: Unsupported value: "coordinator": supported values: "leader", "worker"`,
				`spec.components[1].roles: Required value: must contain the "leader" role`,
				`spec.components[1].roles: Required value: must contain the "worker" role`,
			},
		},
		{
			name: "v1beta1 explicit multinode roles default omitted replicas",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader},
					{Name: nvidiacomv1beta1.ComponentRoleWorker},
				}
			}),
			wantRoleReplicas: map[string]int32{
				nvidiacomv1beta1.ComponentRoleLeader: 1,
				nvidiacomv1beta1.ComponentRoleWorker: 3,
			},
		},
		{
			name: "v1alpha1 explicit multinode roles receive hub defaults",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 4}
				worker.Roles = []nvidiacomv1alpha1.ComponentRoleSpec{
					{Name: nvidiacomv1alpha1.ComponentRoleLeader},
					{Name: nvidiacomv1alpha1.ComponentRoleWorker},
				}
			}),
			wantRoleReplicas: map[string]int32{
				nvidiacomv1beta1.ComponentRoleLeader: 1,
				nvidiacomv1beta1.ComponentRoleWorker: 3,
			},
		},
		{
			name: "explicit multinode role replicas must match node count",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				setBetaExplicitMultinodeRoles(betaWorkerComponent(dgd), 4)
				betaWorkerComponent(dgd).Roles[1].Replicas = k8sptr.To(int32(2))
			}),
			wantWebhookErrs: []string{
				`spec.components[1].roles[1].replicas: Invalid value: 2: must equal 3 for multinode role "worker"`,
			},
		},

		// Checkpoint rules.
		{
			name: "v1beta1 valid checkpoint configuration reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
		},
		{
			name: "non-worker checkpoint is rejected at admission",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				frontend := dgd.GetComponentByName("frontend")
				frontend.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].experimental.checkpoint: Forbidden: checkpoint functionality is supported only for worker, prefill, and decode components",
			},
		},
		{
			name:          "checkpoint configuration requires operator feature gate",
			checkpointOff: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled: true,
						Job: &nvidiacomv1beta1.ComponentCheckpointJobConfig{
							GMSClientContainers: []string{"main"},
						},
					},
				}
			}),
			wantWebhookErrs: []string{
				"spec.components[1].experimental.checkpoint: Forbidden: checkpoint functionality is disabled in the operator configuration",
				"spec.components[1].experimental.checkpoint.job.gmsClientContainers: Forbidden: requires gpuMemoryService to be set",
			},
		},

		// KV-transfer CEL rules.
		{
			name: "v1beta1 conflicting KV-transfer selectors are rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey:            "topology.kubernetes.io/zone",
						ClusterTopologyName: "grove-topology",
						Domain:              "zone",
					},
				}
			}),
			wantCELErr: "spec.experimental.kvTransferPolicy: Invalid value: exactly one of labelKey or clusterTopologyName is required",
		},
		{
			name: "v1beta1 label-key KV-transfer policy reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey: "topology.kubernetes.io/zone",
						Domain:   "zone",
					},
				}
			}),
		},
		{
			name: "v1beta1 preferred KV-transfer enforcement without weight is rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey:    "topology.kubernetes.io/zone",
						Domain:      "zone",
						Enforcement: nvidiacomv1beta1.KvTransferEnforcementPreferred,
					},
				}
			}),
			wantCELErr: "spec.experimental.kvTransferPolicy: Invalid value: preferredWeight is required when enforcement is preferred",
		},
		{
			name: "v1beta1 required KV-transfer enforcement with weight is rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey:        "topology.kubernetes.io/zone",
						Domain:          "zone",
						Enforcement:     nvidiacomv1beta1.KvTransferEnforcementRequired,
						PreferredWeight: k8sptr.To(float32(1)),
					},
				}
			}),
			wantCELErr: "spec.experimental.kvTransferPolicy: Invalid value: preferredWeight may only be set when enforcement is preferred",
		},
		{
			name: "v1beta1 valid preferred KV-transfer enforcement reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey:        "topology.kubernetes.io/zone",
						Domain:          "zone",
						Enforcement:     nvidiacomv1beta1.KvTransferEnforcementPreferred,
						PreferredWeight: k8sptr.To(float32(1)),
					},
				}
			}),
		},
		{
			name: "KV-transfer label key format is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{LabelKey: "bad prefix/zone", Domain: "zone"},
				}
			}),
			wantSchemaErr: `spec.experimental.kvTransferPolicy.labelKey: Invalid value: "bad prefix/zone": spec.experimental.kvTransferPolicy.labelKey in body should match '^(([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)(\.[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)*/)?([A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?)$'`,
		},
		{
			name: "KV-transfer label key name segment is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{LabelKey: "topology.kubernetes.io/-zone", Domain: "zone"},
				}
			}),
			wantSchemaErr: `spec.experimental.kvTransferPolicy.labelKey: Invalid value: "topology.kubernetes.io/-zone": spec.experimental.kvTransferPolicy.labelKey in body should match '^(([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)(\.[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)*/)?([A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?)$'`,
		},
		{
			name: "KV-transfer cluster topology name must be a DNS-1123 subdomain",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: "Bad_Name", Domain: "zone"},
				}
			}),
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy.clusterTopologyName: Invalid value: "Bad_Name": a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')`},
		},
		{
			name: "KV-transfer cluster topology name requires Grove pathway",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse}
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: "grove-topology", Domain: "zone"},
				}
			}),
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy.clusterTopologyName: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "KV-transfer domain is required by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{LabelKey: "topology.kubernetes.io/zone"},
				}
			}),
			wantSchemaErr: `spec.experimental.kvTransferPolicy.domain: Invalid value: "": spec.experimental.kvTransferPolicy.domain in body should match '^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'`,
		},
		{
			name: "KV-transfer domain format is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{LabelKey: "topology.kubernetes.io/zone", Domain: "Zone"},
				}
			}),
			wantSchemaErr: `spec.experimental.kvTransferPolicy.domain: Invalid value: "Zone": spec.experimental.kvTransferPolicy.domain in body should match '^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'`,
		},
		{
			name: "KV-transfer enforcement enum is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey: "topology.kubernetes.io/zone", Domain: "zone", Enforcement: "sometimes",
					},
				}
			}),
			wantSchemaErr: `spec.experimental.kvTransferPolicy.enforcement: Unsupported value: "sometimes": supported values: "required", "preferred"`,
		},
		{
			name: "KV-transfer preferred weight range is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						LabelKey:        "topology.kubernetes.io/zone",
						Domain:          "zone",
						Enforcement:     nvidiacomv1beta1.KvTransferEnforcementPreferred,
						PreferredWeight: k8sptr.To(float32(1.2)),
					},
				}
			}),
			wantSchemaErr: "spec.experimental.kvTransferPolicy.preferredWeight: Invalid value: 1.2: spec.experimental.kvTransferPolicy.preferredWeight in body should be less than or equal to 1",
		},
		{
			name: "KV-transfer cluster topology policy is valid",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: "grove-topology", Domain: "rack"},
				}
			}),
		},
		{
			name: "KV-transfer cluster topology policy rejects missing topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: "missing-topology", Domain: "rack"},
				}
			}),
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy.clusterTopologyName: Invalid value: "missing-topology": references a ClusterTopologyBinding resource that was not found`},
		},
		{
			name: "KV-transfer cluster topology policy rejects missing domain",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{ClusterTopologyName: "grove-topology", Domain: "host"},
				}
			}),
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy.domain: Invalid value: "host": does not exist in ClusterTopologyBinding "grove-topology"; available domains: [rack zone]`},
		},

		// GMS and failover rules.
		{
			name: "v1beta1 inter-pod GMS client containers are rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"metrics"}
			}),
			wantCELErr: "spec.components[1].experimental.gpuMemoryService: Invalid value: extraClientContainers is only supported with mode=IntraPod",
		},
		{
			name: "v1beta1 non-empty GMS extra client pods are rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientPods = []nvidiacomv1beta1.GMSClientPodSpec{{Name: "client"}}
			}),
			wantCELErr: "spec.components[1].experimental.gpuMemoryService: Invalid value: extraClientPods is reserved for inter-pod GMS and is not implemented yet",
		},
		{
			name: "v1beta1 valid GMS configuration reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaIntraPodGMS(betaWorkerComponent(dgd))
			}),
		},
		{
			name: "GMS rejects frontend component",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaIntraPodGMS(&dgd.Spec.Components[0])
			}),
			wantWebhookErrs: []string{"spec.components[0].experimental.gpuMemoryService: Forbidden: GPU memory service is only supported for worker, prefill, or decode components"},
		},
		{
			name: "GMS client container names are validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"Bad_Name"}
			}),
			wantSchemaErr: `spec.components[1].experimental.gpuMemoryService.extraClientContainers[0]: Invalid value: "Bad_Name": spec.components[1].experimental.gpuMemoryService.extraClientContainers[0] in body should match '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'`,
		},
		{
			name: "GMS client container references must resolve",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"missing-b", "missing-a"}
			}),
			wantWebhookErrs: []string{
				`spec.components[1].experimental.gpuMemoryService.extraClientContainers[0]: Invalid value: "missing-b": does not name a container in podTemplate.spec.containers`,
				`spec.components[1].experimental.gpuMemoryService.extraClientContainers[1]: Invalid value: "missing-a": does not name a container in podTemplate.spec.containers`,
			},
		},
		{
			name: "GMS client container references must be unique",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.PodTemplate.Spec.Containers = append(worker.PodTemplate.Spec.Containers,
					corev1.Container{Name: "loader", Image: "loader:latest"},
				)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"loader", "loader"}
			}),
			wantSchemaErr: `spec.components[1].experimental.gpuMemoryService.extraClientContainers[1]: Duplicate value: "loader"`,
		},
		{
			name: "GMS accepts multiple declared client containers",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.PodTemplate.Spec.Containers = append(worker.PodTemplate.Spec.Containers,
					corev1.Container{Name: "loader", Image: "loader:latest"},
					corev1.Container{Name: "metrics-client", Image: "metrics:latest"},
				)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"loader", "metrics-client"}
			}),
		},
		{
			name: "intra-pod failover requires GMS",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaContainerDiscovery(dgd)
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
				}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.failover: Forbidden: gpuMemoryService is required when failover mode is "IntraPod"`},
		},
		{
			name: "failover mode must match GMS mode",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode:       nvidiacomv1beta1.GMSModeInterPod,
					NumShadows: 1,
				}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.failover.mode: Invalid value: "InterPod": must match gpuMemoryService.mode "IntraPod"`},
		},
		{
			name: "intra-pod failover shadow maximum is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaContainerDiscovery(dgd)
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode:       nvidiacomv1beta1.GMSModeIntraPod,
					NumShadows: 2,
				}
			}),
			wantSchemaErr: "spec.components[1].experimental.failover.numShadows: Invalid value: 2: spec.components[1].experimental.failover.numShadows in body should be less than or equal to 1",
		},
		{
			name: "inter-pod failover requires GMS",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeInterPod, NumShadows: 1},
				}
			}),
			wantWebhookErrs: []string{
				`spec.components[1].experimental.failover: Forbidden: gpuMemoryService is required when failover mode is "InterPod"`,
				"spec.components[1].experimental.failover: Forbidden: GMS failover requires at least 1 GPU in podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu",
			},
		},
		{
			name: "inter-pod failover shadow-count minimum is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeInterPod, NumShadows: -1}
			}),
			wantSchemaErr: "spec.components[1].experimental.failover.numShadows: Invalid value: -1: spec.components[1].experimental.failover.numShadows in body should be greater than or equal to 1",
		},
		{
			name: "inter-pod failover rejects frontend component",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeInterPod, NumShadows: 1},
				}
			}),
			wantWebhookErrs: []string{
				`spec.components[0].experimental.failover: Forbidden: gpuMemoryService is required when failover mode is "InterPod"`,
				"spec.components[0].experimental.failover: Forbidden: GMS failover requires at least 1 GPU in podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu",
				`spec.components[0].experimental.failover: Forbidden: GMS failover is not supported for component type "frontend"`,
			},
		},
		{
			name: "ordinary GMS snapshot combination is accepted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}
			}),
		},
		{
			name: "checkpoint compatibility is revalidated on update",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}
			}),
			wantWebhookErrs: []string{
				"spec.components[1].experimental.checkpoint: Forbidden: Snapshot with gpuMemoryService.mode=InterPod is unsupported",
			},
		},

		// Source-version compatibility rules.
		{
			name: "alpha PVC empty name requirement is preserved structurally",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{{
					Name:   k8sptr.To(""),
					Create: k8sptr.To(false),
				}}
			}),
			wantWebhookErrs: []string{"spec.pvcs[0].name: Required value: is required"},
		},
		{
			name: "alpha ingress requires host",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				className := "nginx"
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType:          consts.ComponentTypeFrontend,
					RuntimeVersionOverride: "1.1.0",
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
						MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"},
					},
					Ingress: &nvidiacomv1alpha1.IngressSpec{
						Enabled:                    true,
						IngressControllerClassName: &className,
					},
				}
			}),
			wantWebhookErrs: []string{"spec.services[frontend].ingress.host: Required value: is required when ingress is enabled"},
		},
		{
			name: "alpha volume mounts require mount point unless used as compilation cache",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].VolumeMounts = []nvidiacomv1alpha1.VolumeMount{{Name: "cache"}}
			}),
			wantWebhookErrs: []string{"spec.services[worker].volumeMounts[0].mountPoint: Required value: is required when useAsCompilationCache is false"},
		},
		{
			// eppConfig stays in the API while Go EPP is deprecated, so its
			// v1alpha1 source-exclusivity rule must keep its coverage. A
			// pre-1.5.0 runtime is a legacy Go EPP, so eppConfig is expected
			// here and only its shape is at fault.
			name: "alpha EPP config sources are mutually exclusive",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp"}},
					Config: &apixv1alpha1.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Forbidden: exactly one of configMapRef or config is required"},
		},
		{
			name: "alpha DGD EPP config map requires a name at the submitted source path",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{},
				}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig.configMapRef.name: Required value: is required"},
		},
		{
			name: "alpha DGD legacy EPP contract error uses the submitted source path",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = legacyEPPImage140
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Required value: is required for legacy Go EPP images with runtime version earlier than 1.5.0"},
		},
		{
			name: "alpha DGD native EPP contract error uses the submitted source path",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.RuntimeVersionOverride = ""
				worker.ExtraPodSpec.MainContainer.Image = frontendImage150
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "legacy-epp-config"}},
				}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Forbidden: must be omitted for native Rust EPP images with runtime version 1.5.0 or later"},
		},
		{
			name: "alpha intra-pod failover shadow maximum is preserved structurally",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].Failover = &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeIntraPod,
					NumShadows: 2,
				}
			}),
			wantWebhookErrs: []string{
				`metadata.annotations[nvidia.com/dynamo-kube-discovery-mode]: Invalid value: "": must be "container" when intra-pod failover is configured`,
				`spec.components[0].experimental.failover: Forbidden: gpuMemoryService is required when failover mode is "IntraPod"`,
				`spec.services[worker].failover.numShadows: Invalid value: 2: is invalid for mode="intraPod": intraPod uses a fixed 1 primary + 1 shadow sidecar; use failover.mode="interPod" to configure numShadows`,
			},
		},
		{
			name: "alpha frontend sidecar rejects generated container name conflict",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType:          consts.ComponentTypeFrontend,
					RuntimeVersionOverride: "1.1.0",
					FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
						Image: "custom/frontend:latest",
					},
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"}, PodSpec: &corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  consts.FrontendSidecarContainerName,
							Image: "custom/frontend:latest",
						}},
					}},
				}
			}),
			wantWebhookErrs: []string{`spec.services[frontend].frontendSidecar: Forbidden: cannot inject frontend sidecar: a container named "sidecar-frontend" already exists in extraPodSpec.containers`},
		},
		{
			name: "alpha GMS client container names are validated by the source schema",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               false,
					ExtraClientContainers: []string{"Bad_Name"},
				}
			}),
			wantSchemaErr: `spec.services.worker.gpuMemoryService.extraClientContainers[0]: Invalid value: "Bad_Name": spec.services.worker.gpuMemoryService.extraClientContainers[0] in body should match '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'`,
		},
		{
			name: "nil alpha service entry is pruned",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["ghost"] = nil
			}),
		},
		{
			name: "valid preserved alpha-only fields are accepted",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				host := "worker.example.com"
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{{Name: k8sptr.To("cache"), Create: k8sptr.To(false)}}
				service := dgd.Spec.Services["worker"]
				service.Ingress = &nvidiacomv1alpha1.IngressSpec{Enabled: true, Host: host}
				service.Annotations = map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "ray"}
				service.VolumeMounts = []nvidiacomv1alpha1.VolumeMount{{Name: "cache", UseAsCompilationCache: true}}
				service.SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{Disabled: true}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               false,
					Mode:                  nvidiacomv1alpha1.GMSModeIntraPod,
					ExtraClientContainers: []string{"metrics"},
				}
			}),
		},
		{
			name: "alpha PVC name requirement is preserved structurally",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{{}}
			}),
			wantSchemaErr: "spec.pvcs[0].name: Required value",
		},
		{
			name: "alpha PVC create value constraints are preserved structurally",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{{Name: k8sptr.To("cache"), Create: k8sptr.To(true)}}
			}),
			wantCELErr: "spec.pvcs[0]: Invalid value: When create is true, size, storageClass, and volumeAccessMode are required",
		},
		{
			name: "alpha compatibility warnings are preserved",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				legacyNamespace := "legacy-namespace"
				service := dgd.Spec.Services["worker"]
				service.DynamoNamespace = &legacyNamespace
				//nolint:staticcheck // SA1019: Intentionally testing deprecated field warnings.
				service.Autoscaling = &nvidiacomv1alpha1.Autoscaling{Enabled: true}
			}),
			wantWarnings: []string{
				`spec.services[worker].dynamoNamespace is deprecated and ignored. Value "legacy-namespace" will be replaced with "default-test-graph". Remove this field from your configuration`,
				"spec.services[worker].autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md",
			},
		},
		{
			name: "GMS accepts GPU from alpha dedicated resources",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.Resources = &nvidiacomv1alpha1.Resources{Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"}}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: nvidiacomv1alpha1.GMSModeIntraPod}
			}),
		},
		{
			name: "GMS accepts GPU from alpha extraPodSpec main container resources",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{MainContainer: &corev1.Container{
					Image: "registry.example/runtime:1.1.0",
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
						corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
					}},
				}}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: nvidiacomv1alpha1.GMSModeIntraPod}
			}),
		},
		{
			name: "GMS accepts alpha GPUType resource after conversion",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.Resources = &nvidiacomv1alpha1.Resources{
					Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1", GPUType: "example.com/gpu"},
				}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: nvidiacomv1alpha1.GMSModeIntraPod}
			}),
		},
		{
			name: "v1alpha1 enabled shared memory without a positive size is rejected by source CEL",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{}
			}),
			wantCELErr: "spec.services[worker].sharedMemory: Invalid value: size is required when disabled is false",
		},
		{
			name: "v1alpha1 valid shared memory reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{
					Size: resource.MustParse("1Gi"),
				}
			}),
		},
		{
			name: "v1alpha1 EPP config without a source reaches the webhook without v1beta1 CEL",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Forbidden: exactly one of configMapRef or config is required"},
		},
		{
			name: "v1beta1 EPP config without a source is rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{}
			}),
			wantCELErr: "spec.components[1].eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			name: "v1beta1 EPP config with both sources is rejected by CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp"}},
					Config: &apixv1alpha1.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantCELErr: "spec.components[1].eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			// A well-formed eppConfig on a pre-1.5.0 runtime is still the
			// supported legacy Go EPP shape and must keep being accepted
			// while the field is only deprecated.
			name: "v1beta1 valid EPP config reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				worker.Replicas = k8sptr.To(int32(1))
				worker.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp"}},
				}
			}),
		},

		// Structural root and scheduling rules.
		{
			name:          "priority class requires Grove",
			groveDisabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
			}),
			wantWebhookErrs: []string{`spec.priorityClassName: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "priority class is allowed with Grove",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
			}),
		},
		{
			name: "minAvailable must be positive",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(0))
			}),
			wantSchemaErr: "spec.components[1].minAvailable: Invalid value: 0: spec.components[1].minAvailable in body should be greater than or equal to 1",
		},
		{
			name: "replicas zero can keep minAvailable for scale-up intent",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(0))
				worker.MinAvailable = k8sptr.To(int32(2))
			}),
			wantWarnings: []string{`spec.components[0].minAvailable ("frontend") is deprecated; use spec.components[0].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`, `spec.components[1].minAvailable ("worker") is deprecated; use spec.components[1].providerOverride.value.spec.minAvailable and migrate all component minima together without changing their effective values`},
		},
		{
			name: "rendered Grove resource name length accepts boundary",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = boundaryComponentName
			}),
		},
		{
			name: "rendered Grove resource name length rejects overflow",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = tooLongComponentName
			}),
			wantWebhookErrs: []string{fmt.Sprintf(
				"spec.components[1].name: Invalid value: %q: combined Grove resource name length 46 exceeds the 45-character limit; shorten the deployment or component name",
				tooLongComponentName,
			)},
		},
		{
			name:          "rendered Grove resource name length is skipped outside Grove",
			groveDisabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = tooLongComponentName
			}),
		},

		// Topology rules.
		{
			name: "spec pack domain format is validated by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Generation = 2
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "Bad_Domain",
				}
			}),
			wantSchemaErr: `spec.topologyConstraint.packDomain: Invalid value: "Bad_Domain": spec.topologyConstraint.packDomain in body should match '^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'`,
		},
		{
			name: "component topology pack domain is required by the schema",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Generation = 2
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				dgd.Spec.Components[0].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantSchemaErr: `spec.components[0].topologyConstraint.packDomain: Invalid value: "": spec.components[0].topologyConstraint.packDomain in body should match '^[a-z0-9]([a-z0-9-]*[a-z0-9])?$'`,
		},
		{
			name: "deployment topology without pack domain allows unconstrained components",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Generation = 2
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
		},
		{
			name: "deployment topology without pack domain requires a component topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Generation = 2
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
			}),
			wantWebhookErrs: []string{"spec.topologyConstraint.packDomain: Required value: is required when no component topologyConstraint is set"},
		},
		{
			name: "deployment pack domain can be inherited",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
		},
		{
			name: "deployment pack domain can be mixed with narrower component topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "zone",
				}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
		},
		{
			name: "component topology with deployment topology is valid",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				dgd.Spec.Components[0].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
		},
		{
			name: "missing cluster topology is rejected",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "missing-topology",
					PackDomain:          "rack",
				}
			}),
			wantWebhookErrs: []string{`spec.topologyConstraint.clusterTopologyName: Invalid value: "missing-topology": references a ClusterTopologyBinding resource that was not found`},
		},
		{
			name: "independent topology errors aggregate",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "missing-topology"}
			}),
			wantWebhookErrs: []string{
				`spec.topologyConstraint.packDomain: Required value: is required when no component topologyConstraint is set`,
				`spec.topologyConstraint.clusterTopologyName: Invalid value: "missing-topology": references a ClusterTopologyBinding resource that was not found`,
			},
		},
		{
			name: "pack domain must exist in cluster topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "host",
				}
			}),
			wantWebhookErrs: []string{`spec.topologyConstraint.packDomain: Invalid value: "host": does not exist in ClusterTopologyBinding "grove-topology"; available domains: [rack zone]`},
		},
		{
			name: "component topology cannot be broader than spec topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
			}),
			wantWebhookErrs: []string{`spec.components[1].topologyConstraint.packDomain: Invalid value: "zone": must be equal to or narrower than the deployment-level domain "rack"`},
		},

		// Provider-native override rules.
		{
			name: "valid Grove root provider override is admitted and defaulted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					"",
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
		},
		{
			name: "v1alpha1 Grove root provider override converts and defaults",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = alphaGroveProviderOverride(
					"",
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
		},
		{
			name:          "first provider override can be added without lifecycle history",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
		},
		{
			name: "provider override can be removed without lifecycle status",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			deployment: betaDGDForAdmission(nil),
		},
		{
			name: "valid Grove component and multinode role overrides are admitted and defaulted",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				frontend := dgd.GetComponentByName("frontend")
				frontend.ProviderOverride = groveProviderOverride(
					"",
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"host"}}}`,
				)
				worker := betaWorkerComponent(dgd)
				worker.ProviderOverride = groveProviderOverride(
					"",
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}`,
				)
				setBetaExplicitMultinodeRoles(worker, 2)
				worker.Roles[0].ProviderOverride = groveProviderOverride(
					"",
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"host"}}}`,
				)
				worker.Roles[1].ProviderOverride = groveProviderOverride(
					"",
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"host"}}}`,
				)
			}),
		},
		{
			name:               "provider override requires a materialized workload provider on legacy update",
			seedWithoutWebhook: true,
			oldBeforeUpdate: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
			}),
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			wantWebhookErrs: []string{`spec.providerOverride: Forbidden: requires controller-owned annotation "nvidia.com/workload-provider" to be materialized; wait for controller adoption and retry`},
		},
		{
			name: "provider override target must match its DGD context",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueTemplateSpec,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			wantWebhookErrs: []string{`spec.providerOverride.target: Invalid value: "PodCliqueTemplateSpec": must match the provider-context target "PodCliqueSet"`},
		},
		{
			name: "provider override rejects Dynamo-owned fields",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"replicas":3,"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			wantWebhookErrs: []string{`spec.providerOverride.value.spec.replicas: Forbidden: is Dynamo-owned or not enabled for provider override`},
		},
		{
			name: "malformed provider override field is invalid without echoing its value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"pack":"rack"}}}}`,
				)
			}),
			wantWebhookErrs: []string{`spec.providerOverride.value: Invalid value: null: does not match the registered PodCliqueSet schema: json: cannot unmarshal string into Go struct field TopologyConstraint.spec.template.topologyConstraint.pack of type v1alpha1.TopologyPackConstraint`},
		},
		{
			name: "typed and provider-native Grove topology cannot be combined",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			wantWebhookErrs: []string{`spec.providerOverride.value: Forbidden: cannot be combined with spec.topologyConstraint or components[].topologyConstraint; use either the typed topology API or provider-native Grove topology overrides`},
		},
		{
			name: "component workload provider rejects provider overrides",
			deployment: betaComponentDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			wantWebhookErrs: []string{`spec.providerOverride: Forbidden: requires workload provider "grove", but "component" is selected`},
		},
		{
			name:               "legacy provider override schema identity is immutable during repair",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueTemplateSpec,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
				dgd.Spec.ProviderOverride.APIVersion = "grove.io/v2"
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"spec":{"template":{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"rack"}}}}}`,
				)
			}),
			wantWebhookErrs: []string{
				`spec.providerOverride.apiVersion: Invalid value: "grove.io/v1alpha1": ` + apivalidation.FieldImmutableErrorMsg,
				`spec.providerOverride.target: Invalid value: "PodCliqueSet": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},
		{
			name:               "legacy component role provider identity is immutable during repair",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				setBetaExplicitMultinodeRoles(worker, 2)
				worker.Roles[0].ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueSet,
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"host"}}}`,
				)
				worker.Roles[0].ProviderOverride.APIVersion = "grove.io/v2"
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				setBetaExplicitMultinodeRoles(worker, 2)
				worker.Roles[0].ProviderOverride = groveProviderOverride(
					provideroverride.TargetPodCliqueTemplateSpec,
					`{"topologyConstraint":{"topologyName":"grove-topology","pack":{"required":"host"}}}`,
				)
			}),
			wantWebhookErrs: []string{
				`spec.components[1].roles[0].providerOverride.apiVersion: Invalid value: "grove.io/v1alpha1": ` + apivalidation.FieldImmutableErrorMsg,
				`spec.components[1].roles[0].providerOverride.target: Invalid value: "PodCliqueTemplateSpec": ` + apivalidation.FieldImmutableErrorMsg,
			},
		},

		// Metadata annotation rules.
		{
			name: "workload provider input is normalized from routing intent on create",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
				}
			}),
			wantProvider: consts.WorkloadProviderGrove,
		},
		{
			name:               "unsupported stored workload provider is rejected",
			seedWithoutWebhook: true,
			oldBeforeUpdate:    betaComponentDGDForAdmission(nil),
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: "unknown",
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: "unknown",
				}
				dgd.Labels = map[string]string{"updated": "true"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/workload-provider]: Unsupported value: "unknown": supported values: "component", "grove"`},
		},
		{
			name:               "user cannot materialize a legacy workload provider",
			seedWithoutWebhook: true,
			oldBeforeUpdate:    betaComponentDGDForAdmission(nil),
			oldDeployment:      betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
				}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/workload-provider]: Forbidden: may only be materialized by the Dynamo operator`},
		},
		{
			name:               "operator can materialize a legacy workload provider",
			seedWithoutWebhook: true,
			username:           admissionOperatorPrincipal,
			oldBeforeUpdate:    betaComponentDGDForAdmission(nil),
			oldDeployment:      betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
				}
			}),
		},
		{
			name:               "workload provider cannot change",
			seedWithoutWebhook: true,
			oldDeployment:      betaComponentDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/workload-provider]: Invalid value: "grove": ` + apivalidation.FieldImmutableErrorMsg},
		},
		{
			name:               "Grove provider retains Grove admission semantics when the gate is disabled",
			seedWithoutWebhook: true,
			groveDisabled:      true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
				dgd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name:               "component provider retains component admission semantics when Grove is enabled",
			seedWithoutWebhook: true,
			oldDeployment: betaComponentDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
				}
				dgd.Spec.PriorityClassName = dgdAdmissionPriorityClass
				dgd.Labels = map[string]string{"updated": "true"}
			}),
			wantWebhookErrs: []string{`spec.priorityClassName: Forbidden: requires the Grove pathway, but workload provider "component" is selected`},
		},
		{
			name: "create overwrites a user-supplied origin version",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
			}),
			wantOriginVersion: "1.6.0",
		},
		{
			name:               "origin version cannot be materialized on update",
			seedWithoutWebhook: true,
			oldBeforeUpdate:    betaDGDForAdmission(nil),
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationDynamoOperatorOriginVersion: "1.2.3",
					consts.KubeAnnotationWorkloadProvider:            consts.WorkloadProviderGrove,
				}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: "1.2.3": field is immutable`},
		},
		{
			name: "origin version cannot change on update",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
					consts.KubeAnnotationWorkloadProvider:            consts.WorkloadProviderGrove,
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationDynamoOperatorOriginVersion: "1.2.3",
					consts.KubeAnnotationWorkloadProvider:            consts.WorkloadProviderGrove,
				}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: "1.2.3": field is immutable`},
		},
		{
			name: "origin version cannot be removed on update",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
					consts.KubeAnnotationWorkloadProvider:            consts.WorkloadProviderGrove,
				}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
				}
			}),
			wantWebhookErrs: []string{
				"metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: null: field is immutable",
			},
		},
		{
			name: "vLLM backend annotation accepts mp",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "mp"}
			}),
		},
		{
			name: "vLLM backend annotation accepts ray case-insensitively",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "RAY"}
			}),
		},
		{
			name: "vLLM backend annotation rejects unknown value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "typo"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/vllm-distributed-executor-backend]: Invalid value: "typo": must be "mp" or "ray"`},
		},
		{
			name:         "Grove update strategy annotation accepts Coherent",
			wantWarnings: []string{`spec.components[0] ("frontend"): Coherent updates may temporarily make this entire component unavailable (replicas=1, minAvailable=1, maxUnavailable=1); provision spare serving capacity before updating`},
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "Coherent"}
			}),
		},
		{
			name: "Grove update strategy annotation accepts RollingRecreate",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "RollingRecreate"}
			}),
		},
		{
			name: "Grove update strategy annotation accepts OnDelete",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "OnDelete"}
			}),
		},
		{
			name: "Grove update strategy annotation rejects lowercase value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "ondelete"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: "ondelete": supported values: "Coherent", "RollingRecreate", "OnDelete"`},
		},
		{
			name: "Grove update strategy annotation rejects whitespace",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: " OnDelete "}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: " OnDelete ": supported values: "Coherent", "RollingRecreate", "OnDelete"`},
		},
		{
			name: "Grove update strategy annotation rejects unknown value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "BlueGreen"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: "BlueGreen": supported values: "Coherent", "RollingRecreate", "OnDelete"`},
		},
		{
			name: "discovery mode accepts pod",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "pod"}
			}),
		},
		{
			name: "discovery mode accepts container",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}
			}),
		},
		{
			name: "discovery mode rejects unknown value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "endpoint"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/dynamo-kube-discovery-mode]: Unsupported value: "endpoint": supported values: "pod", "container"`},
		},
		{
			name: "independent root errors aggregate with exact field paths",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components = nil
				dgd.Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					consts.KubeAnnotationDynamoKubeDiscoveryMode:        "invalid",
				}
			}),
			wantWebhookErrs: []string{
				`metadata.annotations[nvidia.com/vllm-distributed-executor-backend]: Invalid value: "invalid": must be "mp" or "ray"`,
				`metadata.annotations[nvidia.com/dynamo-kube-discovery-mode]: Unsupported value: "invalid": supported values: "pod", "container"`,
				"spec.components: Required value: must have at least one component",
			},
		},

		// Component-set updates.
		{
			name:          "component topology is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = append(spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName:          "extra",
					Replicas:               k8sptr.To(int32(1)),
					RuntimeVersionOverride: "1.1.0",
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name:  consts.MainContainerName,
						Image: "registry.example/runtime:1.1.0",
						Env:   []corev1.EnvVar{{Name: "TOKEN", Value: "do-not-leak-this-value"}},
					}}}},
				})
			}),
			wantWebhookErrs: []string{"spec.components: Forbidden: component topology is immutable and cannot be modified after creation: components added: [extra]"},
			notWantErr:      "do-not-leak-this-value",
		},
		{
			name:          "component removal is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = spec.Components[:1]
			}),
			wantWebhookErrs: []string{"spec.components: Forbidden: component topology is immutable and cannot be modified after creation: components removed: [worker]"},
		},
		{
			name:          "component add and remove reports both sides",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					spec.Components[1],
					{
						ComponentName:          "extra",
						Replicas:               k8sptr.To(int32(1)),
						RuntimeVersionOverride: "1.1.0",
						PodTemplate:            &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}}}},
					},
				}
			}),
			wantWebhookErrs: []string{"spec.components: Forbidden: component topology is immutable and cannot be modified after creation: components added: [extra], components removed: [frontend]"},
		},
		{
			name:          "component reorder is allowed",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components[0], spec.Components[1] = spec.Components[1], spec.Components[0]
			}),
		},

		// Multinode updates.
		{
			name:               "unchanged legacy frontend multinode survives an unrelated update",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				dgd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name:               "v1alpha1 unchanged legacy frontend multinode survives conversion and update",
			seedWithoutWebhook: true,
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeFrontend
				worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services[dgdAdmissionWorkerName]
				worker.ComponentType = consts.ComponentTypeFrontend
				worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2}
				dgd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name:               "legacy multinode cannot set an unsupported component type",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentType = ""
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name:               "legacy frontend multinode can be removed",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDForAdmission(nil),
		},
		{
			name:               "legacy frontend multinode cannot change node count",
			seedWithoutWebhook: true,
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name:          "single-node to multinode transition is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{`spec.components[1].multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`},
		},
		{
			name: "node count-only update is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
			}),
			wantWebhookErrs: []string{"spec.components[1].multinode.nodeCount: Invalid value: 3: " + apivalidation.FieldImmutableErrorMsg},
		},
		{
			name: "implicit to semantically equivalent explicit roles is allowed",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				setBetaExplicitMultinodeRoles(worker, 2)
			}),
		},
		{
			name: "explicit to semantically equivalent implicit roles is allowed",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				setBetaExplicitMultinodeRoles(worker, 2)
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
		},
		{
			name: "explicit role list reorder is allowed",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				setBetaExplicitMultinodeRoles(worker, 2)
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				setBetaExplicitMultinodeRoles(worker, 2)
				worker.Roles[0], worker.Roles[1] = worker.Roles[1], worker.Roles[0]
			}),
		},

		// Topology updates.
		{
			name:          "spec topology constraint is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			wantWebhookErrs: []string{`spec.topologyConstraint: Invalid value: {"clusterTopologyName":"grove-topology","packDomain":"rack"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints`},
		},
		{
			name: "spec topology constraint change is immutable",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "zone",
				}
			}),
			wantWebhookErrs: []string{`spec.topologyConstraint: Invalid value: {"clusterTopologyName":"grove-topology","packDomain":"zone"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints`},
		},
		{
			name: "spec topology constraint removal is immutable",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			deployment:      newBetaDGDForValidation(),
			wantWebhookErrs: []string{"spec.topologyConstraint: Invalid value: null: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints"},
		},
		{
			name: "unchanged topology constraints are allowed",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
		},
		{
			name: "component topology constraint is immutable",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
			}),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantWebhookErrs: []string{`spec.components[1].topologyConstraint: Invalid value: {"packDomain":"rack"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints`},
		},
		{
			name: "component topology constraint change is immutable",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
			}),
			wantWebhookErrs: []string{`spec.components[1].topologyConstraint: Invalid value: {"packDomain":"zone"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints`},
		},
		{
			name: "component topology constraint removal is immutable",
			oldDeployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology", PackDomain: "zone"}
			}),
			wantWebhookErrs: []string{"spec.components[1].topologyConstraint: Invalid value: null: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change topology constraints"},
		},

		// KV-transfer updates.
		{
			name:          "kv transfer policy is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			}),
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy: Invalid value: {"labelKey":"topology.kubernetes.io/zone","domain":"zone","enforcement":"required"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy`},
		},
		{
			name: "unchanged kv transfer policy is allowed",
			oldDeployment: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			}),
			deployment: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:    "topology.kubernetes.io/zone",
				Domain:      "zone",
				Enforcement: nvidiacomv1beta1.KvTransferEnforcementRequired,
			}),
		},
		{
			name: "kv transfer policy removal is immutable",
			oldDeployment: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			}),
			deployment:      newBetaDGDForValidation(),
			wantWebhookErrs: []string{"spec.experimental.kvTransferPolicy: Invalid value: null: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy"},
		},

		// GMS and failover updates.
		{
			name: "GMS extra client container update must resolve",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaIntraPodGMS(worker)
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaIntraPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"missing-client"}
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService.extraClientContainers[0]: Invalid value: "missing-client": does not name a container in podTemplate.spec.containers`},
		},
		{
			name:          "inter-pod GMS layout is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.gpuMemoryService.mode: Invalid value: "InterPod": the inter-pod GMS layout cannot be toggled after creation; delete and recreate the DynamoGraphDeployment`},
		},
		{
			name: "inter-pod GMS layout removal is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			deployment:      newBetaDGDForValidation(),
			wantWebhookErrs: []string{"spec.components[1].experimental.gpuMemoryService.mode: Invalid value: null: the inter-pod GMS layout cannot be toggled after creation; delete and recreate the DynamoGraphDeployment"},
		},
		{
			name: "inter-pod failover toggle is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
				enableBetaInterPodFailover(worker, 1)
			}),
			wantWebhookErrs: []string{`spec.components[1].experimental.failover: Invalid value: {"mode":"InterPod","numShadows":1}: inter-pod GMS failover cannot be toggled after creation; delete and recreate the DynamoGraphDeployment`},
		},
		{
			name: "inter-pod failover removal is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
				enableBetaInterPodFailover(worker, 1)
			}),
			deployment: newBetaDGDForValidation(),
			wantWebhookErrs: []string{
				"spec.components[1].experimental.gpuMemoryService.mode: Invalid value: null: the inter-pod GMS layout cannot be toggled after creation; delete and recreate the DynamoGraphDeployment",
				"spec.components[1].experimental.failover: Invalid value: null: inter-pod GMS failover cannot be toggled after creation; delete and recreate the DynamoGraphDeployment",
			},
		},
		// Grove forceScalingGroup updates.
		{
			name:          "grove.forceScalingGroup addition is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.grove.forceScalingGroup: Invalid value: true: cannot be toggled after creation; delete and recreate the DynamoGraphDeployment to change it"},
		},
		{
			name: "grove.forceScalingGroup removal is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
			deployment:      newBetaDGDForValidation(),
			wantWebhookErrs: []string{"spec.components[1].experimental.grove.forceScalingGroup: Invalid value: null: cannot be toggled after creation; delete and recreate the DynamoGraphDeployment to change it"},
		},
		{
			name: "unchanged grove.forceScalingGroup update reaches the webhook",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
		},
		{
			name: "explicit false grove.forceScalingGroup addition reaches the webhook",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(false)},
				}
			}),
		},
		{
			name: "grove block removal with retained experimental is immutable",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: k8sptr.To(true)},
				}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.grove.forceScalingGroup: Invalid value: null: cannot be toggled after creation; delete and recreate the DynamoGraphDeployment to change it"},
		},
		{
			name: "inter-pod failover shadow count is immutable",
			oldDeployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				enableAlphaInterPodGMSFailover(dgd.Spec.Services["worker"], 1)
			}),
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				enableAlphaInterPodGMSFailover(dgd.Spec.Services["worker"], 2)
			}),
			wantWebhookErrs: []string{"spec.components[0].experimental.failover.numShadows: Invalid value: 2: is immutable for inter-pod GMS failover; delete and recreate the DynamoGraphDeployment to change it"},
		},

		// Scaling adapter updates.
		{
			name: "scaling adapter blocks direct replica changes",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			username:        "system:serviceaccount:default:regular-user",
			wantWebhookErrs: []string{"spec.components[1].replicas: Forbidden: cannot be modified directly when scaling adapter is enabled; scale or update the related DynamoGraphDeploymentScalingAdapter instead"},
		},
		{
			name: "scaling adapter fails closed without user info",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			wantWebhookErrs: []string{"spec.components[1].replicas: Forbidden: cannot be modified directly when scaling adapter is enabled; scale or update the related DynamoGraphDeploymentScalingAdapter instead"},
		},
		{
			name: "scaling adapter removal cannot bypass replica ownership",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = nil
				worker.Replicas = k8sptr.To(int32(3))
			}),
			username:        "system:serviceaccount:default:regular-user",
			wantWebhookErrs: []string{"spec.components[1].replicas: Forbidden: cannot be modified directly when scaling adapter is enabled; scale or update the related DynamoGraphDeploymentScalingAdapter instead"},
		},
		{
			name: "operator can change scaling-adapter-owned replicas",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			username: admissionOperatorPrincipal,
		},

		// Backend and restart updates.
		{
			name: "restart id cannot change during active rolling update",
			oldDeployment: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "old"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
					}
				},
			),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "new"}
			}),
			wantWebhookErrs: []string{"spec.restart.id: Invalid value: \"new\": cannot be changed while a rolling update is InProgress"},
		},
		{
			name: "restart id can stay unchanged during active rolling update",
			oldDeployment: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "same"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
					}
				},
			),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "same"}
			}),
		},
		{
			name: "restart id can change after completed rolling update",
			oldDeployment: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "old"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
					}
				},
			),
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "new"}
			}),
		},
		{
			name:          "restart id is required on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = &nvidiacomv1beta1.Restart{}
			}),
			wantSchemaErr: `spec.restart.id: Invalid value: "": spec.restart.id in body should be at least 1 chars long`,
		},
		{
			name:          "duplicate restart order is rejected on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "worker", "worker")
			}),
			wantWebhookErrs: []string{`spec.restart.strategy.order: Invalid value: ["frontend","worker","worker"]: must be unique`},
		},
		{
			name:          "unknown restart order component is rejected on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "ghost")
			}),
			wantWebhookErrs: []string{`spec.restart.strategy.order[1]: Unsupported value: "ghost": supported values: "frontend", "worker"`},
		},
		{
			name:          "incomplete restart order is rejected on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "worker")
			}),
			wantWebhookErrs: []string{`spec.restart.strategy.order: Invalid value: ["worker"]: must have the same number of unique components as the deployment`},
		},
		{
			name:          "empty sequential restart order is valid on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential)
			}),
		},
		{
			name:          "complete sequential restart order is valid on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "worker")
			}),
		},
		{
			name:          "parallel restart without order is valid on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeParallel)
			}),
		},
		{
			name:          "parallel restart rejects order on update",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeParallel, "frontend", "worker")
			}),
			wantWebhookErrs: []string{"spec.restart.strategy.order: Forbidden: cannot be specified when strategy is parallel"},
		},
		{
			name:          "v1beta1 backend framework update reaches the webhook and warns",
			oldDeployment: betaDGDForAdmission(nil),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.BackendFramework = sglangBackendFramework
			}),
			wantWebhookErrs: []string{"spec.backendFramework: Invalid value: \"sglang\": is immutable and cannot be changed after creation"},
			wantWarnings:    []string{"Changing spec.backendFramework may cause unexpected behavior"},
		},
		// Terminating updates. A finalizer can hold a DGD terminating for an
		// arbitrary period, so durable controller-owned metadata still has to be
		// protected during that window while anything cleanup needs stays allowed.
		// Each row deletes a finalizer-held object first, so the deletionTimestamp
		// is the API server's and is present on both the old and the new object.
		{
			name:               "terminating rejects replacing the workload provider",
			terminating:        true,
			seedWithoutWebhook: true,
			oldDeployment:      betaTerminatingDGDForAdmission(nil),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = consts.WorkloadProviderComponent
			}),
			wantWebhookErrs: []string{
				`metadata.annotations[nvidia.com/workload-provider]: Invalid value: "component": field is immutable`,
			},
		},
		{
			name:               "terminating rejects removing the workload provider",
			terminating:        true,
			seedWithoutWebhook: true,
			oldDeployment:      betaTerminatingDGDForAdmission(nil),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				delete(dgd.Annotations, consts.KubeAnnotationWorkloadProvider)
				dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.6.0"
			}),
			// Removal reports a null bad value, since there is no new value to name.
			wantWebhookErrs: []string{
				"metadata.annotations[nvidia.com/workload-provider]: Invalid value: null: field is immutable",
			},
		},
		{
			// Reachable without broadening the seeder bypass: the seed creates a
			// provider-bearing object, then the bypassed seeder UPDATE removes it,
			// which the mutating webhook does not undo on UPDATE. That leaves the
			// stored no-provider state a legacy object reaches the update path in.
			name:               "terminating rejects an unsupported workload provider",
			terminating:        true,
			seedWithoutWebhook: true,
			username:           admissionOperatorPrincipal,
			oldBeforeUpdate:    betaTerminatingDGDForAdmission(nil),
			oldDeployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				delete(dgd.Annotations, consts.KubeAnnotationWorkloadProvider)
			}),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = "bogus"
			}),
			wantWebhookErrs: []string{
				`metadata.annotations[nvidia.com/workload-provider]: Unsupported value: "bogus": supported values: "component", "grove"`,
			},
		},
		{
			name:               "terminating accepts a finalizer-only update",
			terminating:        true,
			seedWithoutWebhook: true,
			oldDeployment:      betaTerminatingDGDForAdmission(nil),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Finalizers = nil
			}),
		},
		{
			// A legacy object whose existing settings no longer satisfy today's
			// rules has to stay deletable, so no new-state rule may run here.
			name:               "terminating accepts finalizer removal despite an invalid gpu memory service",
			terminating:        true,
			seedWithoutWebhook: true,
			oldDeployment:      betaTerminatingDGDForAdmission(withInvalidGPUMemoryService),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				withInvalidGPUMemoryService(dgd)
				dgd.Finalizers = nil
			}),
		},
		{
			// Pins the boundary this draws: only the metadata update rules run
			// while terminating, so a spec change the create-side rules would
			// reject is accepted. Same edit on a live object is rejected by the
			// "beta component main image is required" row above.
			name:               "terminating accepts a spec update the create-side rules would reject",
			terminating:        true,
			seedWithoutWebhook: true,
			oldDeployment:      betaTerminatingDGDForAdmission(nil),
			deployment: betaTerminatingDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = nil
			}),
		},
	}

	t.Log("Add the LPX scenarios to the same DGD admission table")
	tests = append(tests, lpxDGDAdmissionCases()...)

	t.Log("Own one topology fixture for all DGD admission scenarios")
	createTestClusterTopology(t, admissionEnv.ForTest(t))

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Assemble the admission scenario from the table entry")
			gates := features.Gates{Checkpoint: !tt.checkpointOff, Grove: !tt.groveDisabled, LPX: !tt.lpxDisabled}
			test := admissionTestCase{
				object:             tt.deployment,
				oldObject:          tt.oldDeployment,
				oldBeforeUpdate:    tt.oldBeforeUpdate,
				mutateObject:       tt.mutateRequest,
				gates:              gates,
				seedWithoutWebhook: tt.seedWithoutWebhook,
				terminating:        tt.terminating,
				username:           tt.username,
				wantSchemaError:    tt.wantSchemaErr,
				wantCELError:       tt.wantCELErr,
				wantAdmissionErrs:  tt.wantAdmissionErrs,
				wantWebhookErrors:  tt.wantWebhookErrs,
				wantWarnings:       tt.wantWarnings,
				notWantError:       tt.notWantErr,
			}
			if tt.oldDeployment != nil {
				if test.oldBeforeUpdate == nil {
					test.oldBeforeUpdate = dgdBeforeRestart(t, tt.oldDeployment)
				}
				if tt.checkpointOff || tt.groveDisabled || tt.lpxDisabled {
					seedGates := gates
					seedGates.Checkpoint = true
					seedGates.Grove = true
					seedGates.LPX = true
					test.seedGates = &seedGates
				}
			}
			actual := runAdmissionTest(t, test)
			if tt.wantPodAnnotations != nil || tt.wantOriginVersion != "" || tt.wantProvider != "" || tt.wantRoleReplicas != nil || tt.wantReplicas != nil {
				t.Log("Convert the admitted DGD for result assertions")
				actualDGD := admittedBetaDGD(t, actual)

				t.Log("Verify admission preserved component replica intent")
				for name, want := range tt.wantReplicas {
					component := actualDGD.GetComponentByName(name)
					if component == nil {
						t.Fatalf("admitted DGD has no component %q", name)
					}
					if !k8sptr.Equal(component.Replicas, want) {
						t.Fatalf("component %q: replicas = %v, want %v", name, component.Replicas, want)
					}
				}

				if tt.wantOriginVersion != "" {
					t.Log("Verify creation stamped the authoritative operator origin version")
					if got := actualDGD.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]; got != tt.wantOriginVersion {
						t.Fatalf("operator origin version = %q, want %q", got, tt.wantOriginVersion)
					}
				}
				if tt.wantProvider != "" {
					t.Log("Verify creation-time routing intent determined the admitted workload provider")
					if got := actualDGD.Annotations[consts.KubeAnnotationWorkloadProvider]; got != tt.wantProvider {
						t.Fatalf("workload provider = %q, want %q", got, tt.wantProvider)
					}
				}
				if tt.wantRoleReplicas != nil {
					t.Log("Verify admission persisted the defaulted multinode role replicas")
					component := actualDGD.GetComponentByName(dgdAdmissionWorkerName)
					if component == nil {
						t.Fatalf("admitted DGD has no component %q", dgdAdmissionWorkerName)
					}
					actualRoleReplicas := make(map[string]int32, len(component.Roles))
					for i := range component.Roles {
						role := &component.Roles[i]
						actualRoleReplicas[role.Name] = k8sptr.Deref(role.Replicas, 0)
					}
					if !maps.Equal(actualRoleReplicas, tt.wantRoleReplicas) {
						t.Fatalf("role replicas = %v, want %v", actualRoleReplicas, tt.wantRoleReplicas)
					}
				}
				if tt.wantPodAnnotations == nil {
					return
				}

				t.Log("Verify the API server preserved embedded component pod-template annotations")
				component := actualDGD.GetComponentByName(dgdAdmissionWorkerName)
				if component == nil || component.PodTemplate == nil {
					t.Fatalf("admitted DGD has no pod template for component %q", dgdAdmissionWorkerName)
				}
				if got := component.PodTemplate.Annotations; !maps.Equal(got, tt.wantPodAnnotations) {
					t.Fatalf(
						"spec.components[%q].podTemplate.metadata.annotations = %v, want %v",
						dgdAdmissionWorkerName,
						got,
						tt.wantPodAnnotations,
					)
				}
			}
		})
	}
}

func admittedBetaDGD(t *testing.T, actual *unstructured.Unstructured) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()
	beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if actual.GetAPIVersion() == nvidiacomv1beta1.GroupVersion.String() {
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(actual.Object, beta); err != nil {
			t.Fatalf("convert admitted v1beta1 DGD: %v", err)
		}
		return beta
	}

	alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(actual.Object, alpha); err != nil {
		t.Fatalf("convert admitted v1alpha1 DGD: %v", err)
	}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("convert admitted DGD to v1beta1: %v", err)
	}
	return beta
}

func setAlphaCompilationCacheVolumeNameEmpty(t *testing.T, request map[string]any) {
	t.Helper()
	spec, ok := request["spec"].(map[string]any)
	if !ok {
		t.Fatal("request spec is missing or not an object")
	}
	services, ok := spec["services"].(map[string]any)
	if !ok {
		t.Fatal("request spec.services is missing or not an object")
	}
	worker, ok := services["worker"].(map[string]any)
	if !ok {
		t.Fatal("request spec.services.worker is missing or not an object")
	}
	volumeMounts, ok := worker["volumeMounts"].([]any)
	if !ok || len(volumeMounts) == 0 {
		t.Fatal("request spec.services.worker.volumeMounts is missing or empty")
	}
	volumeMount, ok := volumeMounts[0].(map[string]any)
	if !ok {
		t.Fatal("request spec.services.worker.volumeMounts[0] is not an object")
	}
	volumeMount["name"] = ""
}

func betaDGDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := newBetaDGDForValidation()
	dgd.TypeMeta = metav1.TypeMeta{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       "DynamoGraphDeployment",
	}
	if mutate != nil {
		mutate(dgd)
	}
	return dgd
}

// dgdTerminatingFinalizer holds a DGD in the terminating state so an update can
// be submitted while deletion is pending.
const dgdTerminatingFinalizer = "nvidia.com/dynamo-graph-deployment-finalizer"

// betaTerminatingDGDForAdmission builds a DGD that is seeded with a finalizer and
// an already-materialized workload provider, which is the state a terminating
// update starts from. Callers drop either one to express the case under test.
func betaTerminatingDGDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		dgd.Finalizers = []string{dgdTerminatingFinalizer}
		dgd.Annotations = map[string]string{
			consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove,
		}
		if mutate != nil {
			mutate(dgd)
		}
	})
}

// withInvalidGPUMemoryService gives the worker a GPU memory service block that
// today's create-side rules reject, standing in for a legacy object that has to
// stay deletable.
func withInvalidGPUMemoryService(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			ExtraClientContainers: []string{"no-such-container"},
		},
	}
}

func betaComponentDGDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		dgd.Annotations = map[string]string{
			consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent,
			consts.KubeAnnotationEnableGrove:      consts.KubeLabelValueFalse,
		}
		if mutate != nil {
			mutate(dgd)
		}
	})
}

func alphaDGDForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeployment),
) *nvidiacomv1alpha1.DynamoGraphDeployment {
	dgd := newAlphaDGDForCompatibilityValidation()
	dgd.TypeMeta = metav1.TypeMeta{
		APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
		Kind:       "DynamoGraphDeployment",
	}
	if mutate != nil {
		mutate(dgd)
	}
	return dgd
}

func alphaDGDForAdmissionWithServiceNames(names ...string) *nvidiacomv1alpha1.DynamoGraphDeployment {
	return alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
		service := dgd.Spec.Services["worker"]
		dgd.Spec.Services = make(map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec, len(names))
		for _, name := range names {
			dgd.Spec.Services[name] = service.DeepCopy()
		}
	})
}

func dgdAdmissionWithLabel(t *testing.T, deployment runtime.Object) runtime.Object {
	t.Helper()
	switch deployment := deployment.(type) {
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		deployment = deployment.DeepCopy()
		deployment.Labels = map[string]string{"updated": "true"}
		return deployment
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		deployment = deployment.DeepCopy()
		deployment.Labels = map[string]string{"updated": "true"}
		return deployment
	default:
		t.Fatalf("unsupported DGD type %T", deployment)
		return nil
	}
}

func dgdBeforeRestart(t *testing.T, deployment runtime.Object) runtime.Object {
	t.Helper()
	switch deployment := deployment.(type) {
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		if deployment.Spec.Restart == nil {
			return nil
		}
		before := deployment.DeepCopy()
		before.Spec.Restart = nil
		before.Status = nvidiacomv1beta1.DynamoGraphDeploymentStatus{}
		return before
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		if deployment.Spec.Restart == nil {
			return nil
		}
		before := deployment.DeepCopy()
		before.Spec.Restart = nil
		before.Status = nvidiacomv1alpha1.DynamoGraphDeploymentStatus{}
		return before
	default:
		t.Fatalf("unsupported DGD type %T", deployment)
		return nil
	}
}

func newBetaDGDForValidation() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName:          "frontend",
					ComponentType:          nvidiacomv1beta1.ComponentTypeFrontend,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(1)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
				{
					ComponentName:          "worker",
					ComponentType:          nvidiacomv1beta1.ComponentTypeWorker,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(2)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
					}},
				},
			},
		},
	}
}

func newAlphaDGDForCompatibilityValidation() *nvidiacomv1alpha1.DynamoGraphDeployment {
	return &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:          consts.ComponentTypeWorker,
					RuntimeVersionOverride: "1.1.0",
					Replicas:               k8sptr.To(int32(1)),
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
						MainContainer: &corev1.Container{Image: "registry.example/runtime:1.1.0"},
					},
				},
			},
		},
	}
}

func betaDGDWithSpec(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := newBetaDGDForValidation()
	mutate(&dgd.Spec)
	return dgd
}

func betaDGDWithWorker(
	mutate func(*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
		for i := range spec.Components {
			if spec.Components[i].ComponentName == "worker" {
				mutate(&spec.Components[i])
				return
			}
		}
	})
}

func betaDGDWithStatus(
	mutateSpec func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec),
	mutateStatus func(*nvidiacomv1beta1.DynamoGraphDeploymentStatus),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := betaDGDWithSpec(mutateSpec)
	mutateStatus(&dgd.Status)
	return dgd
}

func betaDGDWithKvTransferPolicy(
	policy *nvidiacomv1beta1.KvTransferPolicy,
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
		spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
			KvTransferPolicy: policy,
		}
	})
}

func betaRestart(
	strategyType nvidiacomv1beta1.RestartStrategyType,
	order ...string,
) *nvidiacomv1beta1.Restart {
	return &nvidiacomv1beta1.Restart{
		ID: "roll",
		Strategy: &nvidiacomv1beta1.RestartStrategy{
			Type:  strategyType,
			Order: order,
		},
	}
}

func betaWorkerComponent(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return dgd.GetComponentByName("worker")
}

func setBetaExplicitMultinodeRoles(
	worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	nodeCount int32,
) {
	worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: nodeCount}
	worker.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
		{Name: nvidiacomv1beta1.ComponentRoleLeader},
		{Name: nvidiacomv1beta1.ComponentRoleWorker},
	}
}

func setBetaExplicitMultinodeRoleTemplates(
	worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	nodeCount int32,
) {
	setBetaExplicitMultinodeRoles(worker, nodeCount)
	for i := range worker.Roles {
		worker.Roles[i].PodTemplate = worker.PodTemplate.DeepCopy()
	}
	worker.PodTemplate = nil
}

func addBetaPlanner(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	image string,
	runtimeVersionOverride string,
) {
	dgd.Spec.Components = append(dgd.Spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:          "planner",
		ComponentType:          nvidiacomv1beta1.ComponentTypePlanner,
		RuntimeVersionOverride: runtimeVersionOverride,
		Replicas:               k8sptr.To(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: consts.MainContainerName, Image: image}},
		}},
	})
}

func setAlphaExplicitMultinodeRoleTemplates(
	worker *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
	nodeCount int32,
) {
	template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:  consts.MainContainerName,
			Image: worker.ExtraPodSpec.MainContainer.Image,
		}},
	}}
	worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: nodeCount}
	worker.Roles = []nvidiacomv1alpha1.ComponentRoleSpec{
		{Name: nvidiacomv1alpha1.ComponentRoleLeader, PodTemplate: template.DeepCopy()},
		{Name: nvidiacomv1alpha1.ComponentRoleWorker, PodTemplate: template.DeepCopy()},
	}
	worker.ExtraPodSpec = nil
}

func addAlphaPlanner(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	image string,
	runtimeVersionOverride string,
) {
	dgd.Spec.Services["planner"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType:          consts.ComponentTypePlanner,
		RuntimeVersionOverride: runtimeVersionOverride,
		Replicas:               k8sptr.To(int32(1)),
		ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
			MainContainer: &corev1.Container{Image: image},
		},
	}
}

func setBetaWorkerPowerInputs(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	watts string,
	gpus string,
	nodeCount int32,
) {
	worker := betaWorkerComponent(dgd)
	worker.PodTemplate.Annotations = map[string]string{
		consts.KubeAnnotationGPUPowerLimit: watts,
	}
	worker.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse(gpus),
	}
	worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: nodeCount}
}

func addBetaWorkerGPUSidecar(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	worker := betaWorkerComponent(dgd)
	worker.PodTemplate.Spec.Containers = append(worker.PodTemplate.Spec.Containers, corev1.Container{
		Name:  "gpu-sidecar",
		Image: "registry.example/gpu-sidecar:1.0.0",
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
			corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
		}},
	})
}

func setBetaWorkerResourceClaim(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	podClaim corev1.PodResourceClaim,
) {
	worker := betaWorkerComponent(dgd)
	worker.PodTemplate.Spec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: podClaim.Name}}
	worker.PodTemplate.Spec.ResourceClaims = []corev1.PodResourceClaim{podClaim}
}

func setAlphaWorkerPowerInputs(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	watts string,
	gpus string,
	nodeCount int32,
) {
	worker := dgd.Spec.Services[dgdAdmissionWorkerName]
	worker.ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{
		Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: watts},
	}
	worker.Resources = &nvidiacomv1alpha1.Resources{
		Limits: &nvidiacomv1alpha1.ResourceItem{GPU: gpus},
	}
	worker.Multinode = &nvidiacomv1alpha1.MultinodeSpec{NodeCount: nodeCount}
}

func enableBetaContainerDiscovery(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}
}

func enableAlphaInterPodGMSFailover(
	component *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
	numShadows int32,
) {
	component.Resources = &nvidiacomv1alpha1.Resources{
		Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
	}
	component.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
		Enabled: true,
		Mode:    nvidiacomv1alpha1.GMSModeInterPod,
	}
	component.Failover = &nvidiacomv1alpha1.FailoverSpec{
		Enabled:    true,
		Mode:       nvidiacomv1alpha1.GMSModeInterPod,
		NumShadows: numShadows,
	}
}

func enableBetaInterPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeInterPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  consts.MainContainerName,
				Image: "registry.example/runtime:1.1.0",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
					},
				},
			}},
		},
	}
}

// enableBetaSnapshotFailover constructs a fresh multi-GPU capture/election
// contract in each admission fixture; no fixture objects are shared.
func enableBetaSnapshotFailover(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, backend string) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
		Failover:         &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
		Checkpoint:       &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true, StartupPolicy: nvidiacomv1beta1.CheckpointStartupPolicyWaitForCheckpoint},
	}
	tpFlag := "--tensor-parallel-size"
	if backend == sglangBackendFramework {
		tpFlag = "--tp"
	}
	component.PodTemplate.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo." + backend}
	component.PodTemplate.Spec.Containers[0].Args = []string{"--model", "Qwen/Qwen3-0.6B", tpFlag, "2"}
	component.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("2")}
}

func enableBetaIntraPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeIntraPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  consts.MainContainerName,
				Image: "registry.example/runtime:1.1.0",
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
					},
				},
			}},
		},
	}
}

func enableBetaInterPodFailover(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	numShadows int32,
) {
	if component.Experimental == nil {
		component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
	}
	component.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
		Mode:       nvidiacomv1beta1.GMSModeInterPod,
		NumShadows: numShadows,
	}
}

// nativeDGDForAdmission builds a user-authored native-sidecar topology in either API version.
func nativeDGDForAdmission(t *testing.T, alpha bool, mutate func(*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec)) runtime.Object {
	t.Helper()
	object := betaDGDForAdmission(nil)
	if object.Annotations == nil {
		object.Annotations = map[string]string{}
	}
	object.Annotations[consts.KubeAnnotationDynamoKubeDiscoveryMode] = "container"
	component := betaWorkerComponent(object)
	component.RuntimeVersionOverride = ""
	component.PodTemplate.Spec.Containers[0].Image = "engine:latest"
	component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.5.0", RestartPolicy: k8sptr.To(corev1.ContainerRestartPolicyAlways)}}
	if mutate != nil {
		mutate(component)
	}
	if !alpha {
		return object
	}
	spoke := &nvidiacomv1alpha1.DynamoGraphDeployment{}
	if err := spoke.ConvertFrom(object); err != nil {
		t.Fatal(err)
	}
	return spoke
}

// These complete fixtures make every persisted legacy default explicit for removal on migration.
func legacyMinimumDGDForAdmission(minimum int32) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		for i := range dgd.Spec.Components {
			dgd.Spec.Components[i].MinAvailable = k8sptr.To(int32(1))
		}
		worker := betaWorkerComponent(dgd)
		worker.Replicas = k8sptr.To(int32(4))
		worker.MinAvailable = k8sptr.To(minimum)
	})
}

func nativeMinimumDGDForAdmission(minimum int32) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		for i := range dgd.Spec.Components {
			dgd.Spec.Components[i].ProviderOverride = groveProviderOverride("", `{"spec":{"minAvailable":1}}`)
		}
		worker := betaWorkerComponent(dgd)
		worker.Replicas = k8sptr.To(int32(4))
		worker.ProviderOverride = groveProviderOverride("", fmt.Sprintf(`{"spec":{"minAvailable":%d}}`, minimum))
	})
}

func minimumAvailabilityDGDForAdmission(minimum, replicas int32, strategy string) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := nativeMinimumDGDForAdmission(minimum)
	dgd.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderGrove}
	dgd.Spec.Components[0].Replicas = k8sptr.To(int32(2))
	betaWorkerComponent(dgd).Replicas = k8sptr.To(replicas)
	if strategy != "" {
		dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy] = strategy
	}
	return dgd
}
