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

package validation_test

import (
	"fmt"
	"maps"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	k8sptr "k8s.io/utils/ptr"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// Backend values shared by the DGD and DCD admission tables.
const (
	admissionSGLangBackend = "sglang"
	admissionVLLMBackend   = "vllm"
	admissionTRTLLMBackend = "trtllm"
)

func TestDynamoComponentDeploymentValidator_Validate(t *testing.T) {
	const updatedSidecarImage = "runtime:1.6.0"
	const initialSidecarImage = "runtime:1.5.0"
	const unrelatedInitContainerName = "setup"

	var (
		oneReplica      = int32(1)
		validReplicas   = int32(3)
		validMinAvail   = int32(2)
		negativeSHMSize = resource.MustParse("-1Gi")
	)

	type dcdAdmissionTestCase struct {
		name               string
		deployment         runtime.Object
		oldDeployment      runtime.Object
		oldBeforeUpdate    runtime.Object
		checkpointOff      bool
		seedWithoutWebhook bool
		wantSchemaErr      string
		wantCELErr         string
		wantWebhookErrs    []string
		wantWarnings       []string
		wantOriginVersion  *string
		wantPodAnnotations map[string]string
		wantRoleReplicas   map[string]int32
	}
	tests := []dcdAdmissionTestCase{
		{name: "DGD-owned vLLM multi-GPU automatic candidate is admitted without checkpointRef", deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, "vllm")
			dcd.OwnerReferences = []metav1.OwnerReference{{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment", Name: "graph", UID: "graph-uid", Controller: new(true)}}
			dcd.Spec.PodTemplate.Annotations = map[string]string{
				consts.RestoreCandidateSourceKindAnnotation: consts.RestoreCandidateSourceSnapshotJob,
				consts.CheckpointNameAnnotation:             "capture-job",
				consts.SnapshotJobCandidateUIDAnnotation:    "capture-job-uid",
			}
		})},
		{name: "DGD-owned SGLang multi-GPU checkpointRef restore is admitted", deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Spec.BackendFramework = admissionSGLangBackend
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, admissionSGLangBackend)
			dcd.OwnerReferences = []metav1.OwnerReference{{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment", Name: "graph", UID: "graph-uid", Controller: new(true)}}
			dcd.Spec.Experimental.Checkpoint.CheckpointRef = new("existing-snapshot")
		})},
		{name: "TRT-LLM snapshot failover is rejected at admission", deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Spec.BackendFramework = admissionTRTLLMBackend
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, admissionTRTLLMBackend)
		}), wantWebhookErrs: []string{"spec.experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "inferred TRT-LLM snapshot failover is rejected at admission", deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Spec.BackendFramework = ""
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, admissionTRTLLMBackend)
		}), wantWebhookErrs: []string{"spec.experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "TRT-LLM snapshot failover cannot be enabled on update", oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Spec.BackendFramework = admissionTRTLLMBackend
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, admissionTRTLLMBackend)
			dcd.Spec.Experimental.Failover = nil
		}), deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Spec.BackendFramework = admissionTRTLLMBackend
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, admissionTRTLLMBackend)
		}), wantWebhookErrs: []string{"spec.experimental.failover: Forbidden: Snapshot-backed intra-pod failover supports only vLLM and SGLang (detected: trtllm)"}},
		{name: "automatic failover default startup policy cannot bypass election", deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			enableBetaSnapshotFailover(&dcd.Spec.DynamoComponentDeploymentSharedSpec, "vllm")
			dcd.Spec.Experimental.Checkpoint.StartupPolicy = ""
		}), wantWebhookErrs: []string{"spec.experimental.checkpoint.startupPolicy: Forbidden: Snapshot-backed intra-pod failover requires WaitForCheckpoint for automatic capture"}},
		// Sidecar mode is derived from live init-container names.
		{name: "beta unrelated init container retains standard mode", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "beta regular runtime container retains standard mode", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "runtime", Image: "helper:latest"})
			c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
		})},
		{name: "beta adding runtime activates sidecar validation", oldDeployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		}), deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
			c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
		}), wantWebhookErrs: []string{`spec.podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`}},
		{name: "beta removing runtime returns to main", oldDeployment: nativeDCDForAdmission(t, nil), deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers = nil
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "beta renaming runtime returns to main", oldDeployment: nativeDCDForAdmission(t, nil), deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Name = unrelatedInitContainerName
			c.PodTemplate.Spec.Containers[0].Image = initialSidecarImage
		})},
		{name: "native sidecar beta rejects gms create",
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects failover create",
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects gms update", oldDeployment: nativeDCDForAdmission(t, nil),
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}}
			}), wantWebhookErrs: []string{`spec.experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta rejects failover update", oldDeployment: nativeDCDForAdmission(t, nil),
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{}, Failover: &nvidiacomv1beta1.FailoverSpec{NumShadows: 1}}
			}), wantWebhookErrs: []string{`spec.experimental.gpuMemoryService: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`, `spec.experimental.failover: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`}},
		{name: "native sidecar beta multinode requires role templates", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
		}), wantWebhookErrs: []string{`spec.roles: Required value: multinode sidecars require complete leader and worker podTemplates`}},
		{name: "native sidecar requires engine main", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.Containers[0].Name = "engine"
		}), wantWebhookErrs: []string{`spec.podTemplate.spec.containers: Required value: main engine container is required for component "worker" with a runtime init container`}},
		{name: "native sidecar rejects non-worker component type",
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
			}), wantWebhookErrs: []string{`spec.podTemplate.spec.initContainers[0].name: Forbidden: is supported only for worker, prefill, decode, and LPX components`}},

		// Native-sidecar admission, including alpha conversion and unchanged update invariants.
		{name: "native sidecar beta create", deployment: nativeDCDForAdmission(t, nil)},
		{name: "native sidecar beta update", oldDeployment: nativeDCDForAdmission(t, nil), deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.PodTemplate.Spec.InitContainers[0].Image = updatedSidecarImage
		})},
		{name: "native sidecar beta disabled checkpoint", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: false}}
		})},
		{name: "native sidecar beta rejects nonrestartable init create",
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar beta rejects checkpoint create",
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar beta rejects nonrestartable init update", oldDeployment: nativeDCDForAdmission(t, nil),
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.PodTemplate.Spec.InitContainers[0].RestartPolicy = nil
			}),
			wantWebhookErrs: []string{`spec.podTemplate.spec.initContainers[0].restartPolicy: Invalid value: "": must be Always for component "worker" with a runtime init container`},
		},
		{name: "native sidecar beta rejects checkpoint update", oldDeployment: nativeDCDForAdmission(t, nil),
			deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				c.Experimental = &nvidiacomv1beta1.ExperimentalSpec{Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}}
			}),
			wantWebhookErrs: []string{`spec.experimental.checkpoint.enabled: Forbidden: is not currently supported for component "worker" with a runtime init container; support is planned for a future release`},
		},
		{name: "native sidecar with regular frontend", deployment: nativeDCDForAdmission(t, func(c *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
			c.FrontendSidecar = new("http-frontend")
			c.PodTemplate.Spec.Containers = append(c.PodTemplate.Spec.Containers, corev1.Container{Name: "http-frontend", Image: "frontend:1.5.0"})
		})},
		{
			name: "v1beta1 explicit multinode roles are shared with standalone components",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				dcd.Spec.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
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
			name: "v1beta1 explicit multinode role replicas must match node count",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 4}
				dcd.Spec.Roles = []nvidiacomv1beta1.ComponentRoleSpec{
					{Name: nvidiacomv1beta1.ComponentRoleLeader, Replicas: new(int32(1))},
					{Name: nvidiacomv1beta1.ComponentRoleWorker, Replicas: new(int32(2))},
				}
			}),
			wantWebhookErrs: []string{
				`spec.roles[1].replicas: Invalid value: 2: must equal 3 for multinode role "worker"`,
			},
		},
		{
			name: "v1beta1 frontend component cannot be multinode",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name: "v1beta1 planner component cannot be multinode",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypePlanner
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{
				"spec.multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name: "v1beta1 complete role PodTemplates are admitted for standalone components",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				setBetaExplicitMultinodeRoleTemplates(&dcd.Spec.DynamoComponentDeploymentSharedSpec, 2)
			}),
		},
		{
			name: "standalone canonical lpx component is rejected",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeLPX
				dcd.Spec.PodTemplate = nil
				dcd.Spec.LPX = &nvidiacomv1beta1.LPXConfig{BuildID: "test/build"}
				dcd.Spec.Roles = []nvidiacomv1beta1.ComponentRoleSpec{{Name: nvidiacomv1beta1.ComponentRoleLeader}}
			}),
			wantCELErr:   "spec: Invalid value: standalone LPX DynamoComponentDeployments are not supported; use DynamoGraphDeployment",
			wantWarnings: []string{`unknown field "spec.lpx"`},
		},
		{
			name: "v1beta1 main image is required when pod template is absent on create",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = nil
			}),
			wantWebhookErrs: []string{"spec.podTemplate.spec.containers: Required value: is required"},
		},
		{
			name:          "v1beta1 main image cannot be removed by removing pod template",
			oldDeployment: betaDCDForAdmission(nil),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = nil
			}),
			wantWebhookErrs: []string{"spec.podTemplate.spec.containers: Required value: is required"},
		},
		{
			name: "v1beta1 main image is required",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}},
				}}
			}),
			wantWebhookErrs: []string{"spec.podTemplate.spec.containers[0].image: Required value: is required"},
		},
		{
			name: "v1beta1 custom image does not require runtime version override",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
			}),
		},
		{
			name: "v1beta1 metadata update with custom image does not require runtime version override",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
				dcd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name: "v1beta1 image change to custom does not require runtime version override",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = customRuntimeImage
			}),
		},
		{
			name: "v1beta1 derives runtime version from a semver image tag",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:v1.2.3-cuda12"}},
				}}
			}),
		},
		{
			name: "runtime version override takes precedence over a semver image tag",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = "1.1.0"
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/vllm-opus:4.8.2"}},
				}}
			}),
		},
		{
			name: "v1beta1 accepts four-digit runtime version override segments",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = "9999.9999.9999"
			}),
		},
		{
			name: "v1beta1 rejects runtime version override segments longer than four digits",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.RuntimeVersionOverride = "10000.0.0"
			}),
			wantSchemaErr: `spec.runtimeVersionOverride: Invalid value: "10000.0.0": spec.runtimeVersionOverride in body should match '^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$'`,
		},
		{
			checkpointOff: true,
			name:          "checkpoint configuration requires operator feature gate",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
			wantWebhookErrs: []string{"spec.experimental.checkpoint: Forbidden: checkpoint functionality is disabled in the operator configuration"},
		},
		{
			name:          "checkpoint update requires operator feature gate",
			checkpointOff: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}),
			wantWebhookErrs: []string{"spec.experimental.checkpoint: Forbidden: checkpoint functionality is disabled in the operator configuration"},
		},
		{
			name: "standalone non-worker checkpoint is rejected at admission",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: new("frontend-snapshot"),
					},
				}
			}),
			wantWebhookErrs: []string{"spec.experimental.checkpoint: Forbidden: checkpoint functionality is supported only for worker, prefill, and decode components"},
		},
		{
			name: "v1beta1 standalone worker checkpointRef is rejected",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: new("worker-snapshot"),
					},
				}
			}),
			wantWebhookErrs: []string{"spec.experimental.checkpoint.checkpointRef: Forbidden: worker-class checkpointRef is supported only on DynamoGraphDeployment-managed components"},
		},
		{
			name: "DGD-managed worker checkpointRef is accepted",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.OwnerReferences = []metav1.OwnerReference{{
					APIVersion: nvidiacomv1beta1.GroupVersion.String(),
					Kind:       nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind,
					Name:       "graph",
					UID:        "graph-uid",
					Controller: new(true),
				}}
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: new("worker-snapshot"),
					},
				}
			}),
		},
		{
			name: "v1beta1 minAvailable reaches the standalone DCD webhook",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantWebhookErrs: []string{"spec.minAvailable: Forbidden: is currently supported only for Grove-backed DynamoGraphDeployment components"},
		},
		{
			name: "v1beta1 structural validation aggregates independent shared-spec errors",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.SharedMemorySize = &negativeSHMSize
				dcd.Spec.FrontendSidecar = new("frontend")
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{},
				}
			}),
			wantWebhookErrs: []string{
				`spec.sharedMemorySize: Invalid value: "-1Gi": must be non-negative`,
				`spec.frontendSidecar: Invalid value: "frontend": must match a podTemplate.spec.containers name`,
				"spec.experimental.gpuMemoryService: Forbidden: GPU memory service is only supported for worker, prefill, or decode components",
				"spec.experimental.gpuMemoryService: Forbidden: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
			},
		},

		{
			name: "v1beta1 replicas below minAvailable are rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantCELErr: "spec: Invalid value: minAvailable must be less than or equal to replicas unless replicas is 0",
		},
		{
			name:               "v1beta1 minAvailable change is rejected by CEL",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &oneReplica
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantCELErr: "spec: Invalid value: minAvailable is immutable after creation",
		},
		{
			name:          "v1beta1 type change is rejected by CEL",
			oldDeployment: betaDCDForAdmission(nil),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
			}),
			wantCELErr: "spec: Invalid value: type is immutable after it is set",
		},
		{
			name: "v1beta1 inter-pod GMS client containers are rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode:                  nvidiacomv1beta1.GMSModeInterPod,
						ExtraClientContainers: []string{"metrics"},
					},
				}
			}),
			wantCELErr: "spec.experimental.gpuMemoryService: Invalid value: extraClientContainers is only supported with mode=IntraPod",
		},
		{
			name: "v1beta1 non-empty GMS extra client pods are rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode:            nvidiacomv1beta1.GMSModeInterPod,
						ExtraClientPods: []nvidiacomv1beta1.GMSClientPodSpec{{Name: "client"}},
					},
				}
			}),
			wantCELErr: "spec.experimental.gpuMemoryService: Invalid value: extraClientPods is reserved for inter-pod GMS and is not implemented yet",
		},
		{
			name: "v1beta1 checkpoint job with checkpointRef is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: new("existing-checkpoint"),
						Job:           &nvidiacomv1beta1.ComponentCheckpointJobConfig{},
					},
				}
			}),
			wantCELErr: "spec.experimental.checkpoint: Invalid value: checkpoint.job cannot be set when checkpointRef is specified",
		},

		{
			name: "v1beta1 checkpoint with inter-pod GMS and failover remains rejected",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				enableBetaInterPodGMS(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
				dcd.Spec.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}
				dcd.Spec.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode:       nvidiacomv1beta1.GMSModeInterPod,
					NumShadows: 1,
				}
			}),
			wantWebhookErrs: []string{
				"spec.experimental.checkpoint: Forbidden: Snapshot with gpuMemoryService.mode=InterPod is unsupported",
			},
		},
		{
			name: "v1beta1 checkpoint identity rejects lpu backend",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Identity: &nvidiacomv1beta1.DynamoCheckpointIdentity{
							Model:            "model",
							BackendFramework: "lpu",
						},
					},
				}
			}),
			wantSchemaErr: `spec.experimental.checkpoint.identity.backendFramework: Unsupported value: "lpu": supported values: "vllm", "sglang", "trtllm"`,
		},
		{
			name: "v1beta1 deprecated checkpoint mode with checkpointRef is accepted",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.OwnerReferences = []metav1.OwnerReference{{
					APIVersion: nvidiacomv1beta1.GroupVersion.String(),
					Kind:       "DynamoGraphDeployment",
					Name:       "graph",
					UID:        "graph-uid",
					Controller: new(true),
				}}
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						Mode:          nvidiacomv1beta1.CheckpointModeManual,
						CheckpointRef: new("existing-checkpoint"),
					},
				}
			}),
		},

		{
			name: "GMS extra client container reference must resolve",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				enableBetaIntraPodGMS(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
				dcd.Spec.Experimental.GPUMemoryService.ExtraClientContainers = []string{"missing-client"}
			}),
			wantWebhookErrs: []string{`spec.experimental.gpuMemoryService.extraClientContainers[0]: Invalid value: "missing-client": does not name a container in podTemplate.spec.containers`},
		},
		{
			name: "GMS extra client container update reports one indexed cause",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				enableBetaIntraPodGMS(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				enableBetaIntraPodGMS(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
				dcd.Spec.Experimental.GPUMemoryService.ExtraClientContainers = []string{"missing-client"}
			}),
			wantWebhookErrs: []string{`spec.experimental.gpuMemoryService.extraClientContainers[0]: Invalid value: "missing-client": does not name a container in podTemplate.spec.containers`},
		},
		{
			name: "v1beta1 checkpoint GMS clients reject inter-pod GMS",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				enableBetaInterPodGMS(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
				dcd.Spec.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{
					Enabled: true,
					Job: &nvidiacomv1beta1.ComponentCheckpointJobConfig{
						GMSClientContainers: []string{"gms-saver"},
					},
				}
			}),
			wantWebhookErrs: []string{
				"spec.experimental.checkpoint.job.gmsClientContainers: Forbidden: is only supported with gpuMemoryService.mode=IntraPod",
				"spec.experimental.checkpoint: Forbidden: Snapshot with gpuMemoryService.mode=InterPod is unsupported",
			},
		},

		{
			name: "v1beta1 native Rust EPP accepts more than one replica",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &validMinAvail
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
		},
		{
			name: "v1beta1 legacy Go EPP still requires one replica",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &validMinAvail
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			wantWebhookErrs: []string{
				"spec.replicas: Invalid value: 2: EPP component must have exactly 1 replica",
			},
		},
		{
			name: "v1beta1 native Rust EPP accepts a 1.5 image without eppConfig",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
		},
		{
			name: "v1beta1 legacy Go EPP accepts a 1.4 image with eppConfig",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
		},
		{
			name: "v1beta1 rejects eppConfig with a 1.5 image",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Forbidden: must be omitted for native Rust EPP images with runtime version 1.5.0 or later"},
		},
		{
			name: "v1beta1 rejects a 1.4 image without eppConfig",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Required value: is required for legacy Go EPP images with runtime version earlier than 1.5.0"},
		},
		// EPP is never exempt from the runtime-version requirement: the eppConfig
		// contract is decided entirely by the resolved version, so an unresolvable
		// one is reported rather than admitted with no contract checked.
		{
			name: "v1beta1 EPP with an unresolvable image version requires runtimeVersionOverride",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = "registry.example/dynamo-frontend:latest"
			}),
			wantWebhookErrs: []string{"spec.runtimeVersionOverride: Required value: is required when the specified Dynamo runtime container image has no parseable semantic-version tag"},
		},
		// eppConfig is deprecated but still served, so its shape rules keep
		// their coverage. The default 1.1.0 fixture image is a pre-1.5.0
		// legacy Go EPP runtime, where an eppConfig is expected and only its
		{
			name: "v1beta1 empty EPP config is rejected by source CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{}
			}),
			wantCELErr: "spec.eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			name: "v1beta1 conflicting EPP config is rejected by source CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{Name: "epp-config"},
					Config: &apixv1alpha1.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantCELErr: "spec.eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			// eppConfig is only meaningful for an EPP component; the controller
			// silently ignores it on any other type, so CEL rejects it at
			// admission instead of letting it land as dead configuration.
			name: "v1beta1 rejects eppConfig on a non-epp component",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			wantCELErr: "spec: Invalid value: eppConfig may only be set when type is epp",
		},

		// Pair shared pod-template validation across the v1beta1 schema.
		{
			name: "v1beta1 sidecar without image is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "metrics"}},
				}}
			}),
			wantCELErr: "spec.podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name: "v1beta1 init container without image is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantCELErr: "spec.podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name: "v1beta1 invalid pod template backend annotation is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantCELErr: "spec.podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			// Generated DCD pod metadata must survive structural pruning.
			name: "v1beta1 discovery annotation survives the generated DCD API server round trip",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
						consts.KubeAnnotationDynamoKubeDiscoveryMode:        "container",
					},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "main:1.1.0"}}},
				}
			}),
			wantPodAnnotations: map[string]string{
				consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
				consts.KubeAnnotationDynamoKubeDiscoveryMode:        "container",
			},
		},
		{
			name:       "valid v1beta1 deployment reaches the v1beta1 webhook",
			deployment: betaDCDForAdmission(nil),
		},
		{
			name:          "v1beta1 update without changes reaches the v1beta1 webhook",
			oldDeployment: betaDCDForAdmission(nil),
			deployment:    betaDCDForAdmission(nil),
		},
		{
			name: "v1beta1 backend framework update reaches and is rejected by the v1beta1 webhook",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = admissionSGLangBackend
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = admissionVLLMBackend
			}),
			wantWebhookErrs: []string{`spec.backendFramework: Invalid value: "vllm": is immutable and cannot be changed after creation`},
			wantWarnings:    []string{"Changing spec.backendFramework may cause unexpected behavior"},
		},
		{
			name: "v1beta1 replicas update reaches the v1beta1 webhook",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &oneReplica
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
			}),
		},
		{
			// Unchanged compliant legacy pair: an unrelated field change
			// (replicas stays 1 here; only exercising the update path) must
			// not re-trigger the image/eppConfig compatibility check.
			name: "v1beta1 unchanged legacy image and eppConfig pair remains allowed on update",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
		},
		{
			name:               "v1beta1 unrelated update ratchets an identical pre-existing native image and eppConfig mismatch",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{Name: "legacy-epp-config"},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Labels = map[string]string{"updated": "true"}
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{Name: "legacy-epp-config"},
				}
			}),
		},
		{
			// type is immutable once set, so the only transition into EPP starts
			// from an unset type; it must still validate the complete contract.
			name: "v1beta1 setting a previously unset component type to EPP validates the complete runtime contract",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = ""
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Required value: is required for legacy Go EPP images with runtime version earlier than 1.5.0"},
		},
		{
			name:               "v1beta1 changing eppConfig on a pre-existing native mismatch is rejected",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{Name: "legacy-epp-config"},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{Name: "changed-epp-config"},
				}
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Forbidden: must be omitted for native Rust EPP images with runtime version 1.5.0 or later"},
		},
		{
			// Split update: eppConfig cleared but the image stays at the
			// legacy 1.4 tag -- the Go EPP binary would start with no
			// CLI flags/config mount. Must be rejected even though the
			// image field itself did not change.
			name: "v1beta1 clearing eppConfig while image stays legacy is rejected on update",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Required value: is required for legacy Go EPP images with runtime version earlier than 1.5.0"},
		},
		{
			// Split update: eppConfig added while the image stays at a
			// native Rust EPP 1.5 tag -- the Rust binary would get handed a
			// legacy config mount it never reads. Must be rejected even
			// though the image field itself did not change.
			name: "v1beta1 adding eppConfig while image stays native Rust is rejected on update",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Forbidden: must be omitted for native Rust EPP images with runtime version 1.5.0 or later"},
		},
		{
			// Atomic migration: image and eppConfig change together, in the
			// same update, from a compliant legacy pair to a compliant
			// native pair. Must be accepted -- only the new tuple matters.
			name: "v1beta1 atomic migration from legacy image with eppConfig to native image without it is accepted",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
		},
		{
			// Atomic rollback: the reverse migration, in one update. Must
			// also be accepted.
			name: "v1beta1 atomic rollback from native image to legacy image with eppConfig is accepted",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = frontendImage150
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.RuntimeVersionOverride = ""
				dcd.Spec.PodTemplate.Spec.Containers[0].Image = legacyEPPImage140
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{
						Name: "legacy-epp-config",
					},
				}
			}),
		},
		{
			name:               "v1beta1 unchanged legacy frontend multinode survives an unrelated update",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
				dcd.Labels = map[string]string{"updated": "true"}
			}),
		},
		{
			name:               "v1beta1 legacy frontend multinode can be removed",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
			}),
		},
		{
			name:               "v1beta1 legacy frontend multinode cannot change node count",
			seedWithoutWebhook: true,
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeFrontend
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
			}),
			wantWebhookErrs: []string{
				"spec.multinode: Forbidden: multinode is supported only for worker, prefill, or decode components",
			},
		},
		{
			name:          "v1beta1 multinode layout change is rejected by the shared update validator",
			oldDeployment: betaDCDForAdmission(nil),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{`spec.multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`},
		},
		{
			name: "v1beta1 multinode node count is immutable",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
			}),
			wantWebhookErrs: []string{"spec.multinode.nodeCount: Invalid value: 3: " + apivalidation.FieldImmutableErrorMsg},
		},
		{
			name: "v1beta1 implicit to semantically equivalent explicit roles is allowed",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				setBetaExplicitMultinodeRoles(&dcd.Spec.DynamoComponentDeploymentSharedSpec, 2)
			}),
		},
		{
			name: "v1beta1 explicit to semantically equivalent implicit roles is allowed",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				setBetaExplicitMultinodeRoles(&dcd.Spec.DynamoComponentDeploymentSharedSpec, 2)
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
		},
	}

	// DCD admission forwards the backend and enforces complete role templates for each supported backend.
	for _, backend := range []string{admissionSGLangBackend, admissionVLLMBackend, admissionTRTLLMBackend} {
		tests = append(tests,
			dcdAdmissionTestCase{name: backend + " sidecar multinode DCD rejects a component template", deployment: multinodeSidecarDCDForAdmission(t, backend, false, false), wantWebhookErrs: []string{`spec.roles: Required value: multinode sidecars require complete leader and worker podTemplates`}},
			dcdAdmissionTestCase{name: backend + " sidecar multinode DCD accepts complete roles", deployment: multinodeSidecarDCDForAdmission(t, backend, true, false)},
		)
	}

	// Origin remains immutable, including legacy absence.
	{
		created := betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
			dcd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
		})
		tests = append(tests, dcdAdmissionTestCase{name: "origin CREATE overwrites authored origin", deployment: created, wantOriginVersion: new("1.6.0")})
		for _, origin := range []struct{ name, old, new string }{
			{"legacy unrelated update", "", ""},
			{"legacy invalid origin unrelated update", "custom-build", "custom-build"},
			{"cannot repair legacy invalid origin", "custom-build", "1.7.0"},
			{"cannot remove legacy invalid origin", "custom-build", ""},
			{"cannot backfill origin", "", "1.7.0"},
			{"cannot change origin", "1.6.0", "1.7.0"},
			{"cannot remove origin", "1.6.0", ""},
		} {
			old := betaDCDForAdmission(nil)
			if origin.old != "" {
				old.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin.old}
			}
			updated := old.DeepCopy()
			updated.Annotations = map[string]string{"example.com/unrelated": "edit"}
			if origin.new != "" {
				updated.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = origin.new
			}
			seed := old.DeepCopy()
			seed.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}
			tc := dcdAdmissionTestCase{name: "origin " + origin.name, oldDeployment: old, oldBeforeUpdate: seed, deployment: updated, seedWithoutWebhook: true}
			if origin.old != origin.new {
				badValue := fmt.Sprintf("%q", origin.new)
				if origin.new == "" {
					badValue = "null"
				}
				tc.wantWebhookErrs = []string{"metadata.annotations[" + consts.KubeAnnotationDynamoOperatorOriginVersion + "]: Invalid value: " + badValue + ": " + apivalidation.FieldImmutableErrorMsg}
			}
			tests = append(tests, tc)
		}
	}

	t.Log("Add complete-probe cases to the admission table for the served API version")
	for _, placement := range []string{"main", "frontend sidecar", "role"} {
		beta := betaDCDForAdmission(nil)
		betaSpec := &beta.Spec.DynamoComponentDeploymentSharedSpec
		betaPath := field.NewPath("spec")
		betaContainer := &betaSpec.PodTemplate.Spec.Containers[0]

		// Select a representative container in the template.
		switch placement {
		case "main":
			betaPath = betaPath.Child("podTemplate", "spec", "containers").Index(0)
		case "frontend sidecar":
			betaSpec.FrontendSidecar = new("frontend")
			betaSpec.PodTemplate.Spec.Containers = append(betaSpec.PodTemplate.Spec.Containers, corev1.Container{Name: "frontend", Image: frontendImage150})
			betaContainer = &betaSpec.PodTemplate.Spec.Containers[1]
			betaPath = betaPath.Child("podTemplate", "spec", "containers").Index(1)
		case "role":
			setBetaExplicitMultinodeRoleTemplates(betaSpec, 2)
			betaContainer = &betaSpec.Roles[1].PodTemplate.Spec.Containers[0]
			betaPath = betaPath.Child("roles").Index(1).Child("podTemplate", "spec", "containers").Index(0)
		}

		// Keep the shared probe cases on the resource's production admission path.
		probeCases := containerProbeAdmissionCases(beta, betaContainer, betaPath)
		for _, probeCase := range probeCases {
			tests = append(tests, dcdAdmissionTestCase{
				name: probeCase.name, deployment: probeCase.deployment,
				oldDeployment: probeCase.oldDeployment, wantCELErr: probeCase.wantCELErr,
			})
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gates := features.Gates{Checkpoint: !tt.checkpointOff}
			test := admissionTestCase{
				object:             tt.deployment,
				oldObject:          tt.oldDeployment,
				oldBeforeUpdate:    tt.oldBeforeUpdate,
				gates:              gates,
				seedWithoutWebhook: tt.seedWithoutWebhook,
				wantSchemaError:    tt.wantSchemaErr,
				wantCELError:       tt.wantCELErr,
				wantWebhookErrors:  tt.wantWebhookErrs,
				wantWarnings:       tt.wantWarnings,
			}
			if tt.checkpointOff && tt.oldDeployment != nil {
				seedGates := gates
				seedGates.Checkpoint = true
				test.seedGates = &seedGates
			}
			actual := runAdmissionTest(t, test)
			if tt.wantOriginVersion != nil {
				require.Equal(t, *tt.wantOriginVersion, actual.GetAnnotations()[consts.KubeAnnotationDynamoOperatorOriginVersion])
			}
			if tt.wantPodAnnotations != nil || tt.wantRoleReplicas != nil {
				actualDCD := admittedBetaDCD(t, actual)
				if tt.wantRoleReplicas != nil {
					t.Log("Verify admission persisted the defaulted multinode role replicas")
					actualRoleReplicas := make(map[string]int32, len(actualDCD.Spec.Roles))
					for i := range actualDCD.Spec.Roles {
						role := &actualDCD.Spec.Roles[i]
						actualRoleReplicas[role.Name] = k8sptr.Deref(role.Replicas, 0)
					}
					if !maps.Equal(actualRoleReplicas, tt.wantRoleReplicas) {
						t.Fatalf("role replicas = %v, want %v", actualRoleReplicas, tt.wantRoleReplicas)
					}
				}
				if tt.wantPodAnnotations == nil {
					return
				}

				t.Log("Verify the API server preserved embedded pod-template annotations")
				if actualDCD.Spec.PodTemplate == nil {
					t.Fatal("admitted DCD has no spec.podTemplate")
				}
				if got := actualDCD.Spec.PodTemplate.Annotations; !maps.Equal(got, tt.wantPodAnnotations) {
					t.Fatalf("spec.podTemplate.metadata.annotations = %v, want %v", got, tt.wantPodAnnotations)
				}
			}
		})
	}
}

func admittedBetaDCD(t *testing.T, actual *unstructured.Unstructured) *nvidiacomv1beta1.DynamoComponentDeployment {
	t.Helper()
	beta := &nvidiacomv1beta1.DynamoComponentDeployment{}
	if actual.GetAPIVersion() == nvidiacomv1beta1.GroupVersion.String() {
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(actual.Object, beta); err != nil {
			t.Fatalf("convert admitted v1beta1 DCD: %v", err)
		}
		return beta
	}

	alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(actual.Object, alpha); err != nil {
		t.Fatalf("convert admitted v1alpha1 DCD: %v", err)
	}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("convert admitted DCD to v1beta1: %v", err)
	}
	return beta
}

func betaDCDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoComponentDeployment),
) *nvidiacomv1beta1.DynamoComponentDeployment {
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		APIVersion: nvidiacomv1beta1.GroupVersion.String(),
		Kind:       "DynamoComponentDeployment",
		Name:       "worker", Namespace: "default",
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: admissionVLLMBackend,
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName:          "worker",
				RuntimeVersionOverride: "1.1.0",
				ComponentType:          nvidiacomv1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "registry.example/runtime:1.1.0"}},
				}},
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dcd
}

// nativeDCDForAdmission builds a user-authored native-sidecar topology.
func nativeDCDForAdmission(t *testing.T, mutate func(*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec)) runtime.Object {
	t.Helper()
	object := betaDCDForAdmission(nil)
	component := &object.Spec.DynamoComponentDeploymentSharedSpec
	component.RuntimeVersionOverride = ""
	component.PodTemplate.Spec.Containers[0].Image = "engine:latest"
	component.PodTemplate.Spec.InitContainers = []corev1.Container{{Name: "runtime", Image: "runtime:1.5.0", RestartPolicy: k8sptr.To(corev1.ContainerRestartPolicyAlways)}}
	if mutate != nil {
		mutate(component)
	}
	return object
}
