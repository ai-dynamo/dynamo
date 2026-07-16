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

package validation

import (
	"fmt"
	"slices"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8sptr "k8s.io/utils/ptr"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

const (
	dcdAdmissionSGLangBackend = "sglang"
	dcdAdmissionVLLMBackend   = "vllm"
)

func TestDynamoComponentDeploymentValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamocomponentdeployments.yaml")

	var (
		oneReplica       = int32(1)
		validReplicas    = int32(3)
		negativeReplicas = int32(-1)
		validMinAvail    = int32(2)
		negativeSHMSize  = resource.MustParse("-1Gi")
		workerGPU        = &nvidiacomv1alpha1.Resources{
			Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
		}
	)

	tests := []struct {
		name            string
		deployment      runtime.Object
		oldDeployment   runtime.Object
		wantSchemaErr   string
		wantCELErr      string
		wantWebhookErrs []string
		wantWarnings    []string
	}{
		// Baseline schema and webhook behavior.
		{
			name: "valid deployment",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
		},
		{
			name: "invalid replicas",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &negativeReplicas
			}),
			wantSchemaErr: "spec.replicas: Invalid value: -1: spec.replicas in body should be greater than or equal to 0",
		},
		{
			name: "minAvailable is unsupported for standalone DCD",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantWebhookErrs: []string{"spec.minAvailable: Forbidden: is currently supported only for Grove-backed DynamoGraphDeployment components"},
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
				dcd.Spec.FrontendSidecar = k8sptr.To("frontend")
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{},
				}
			}),
			wantWebhookErrs: []string{
				`spec.sharedMemorySize: Invalid value: "-1Gi": must be non-negative`,
				"spec.podTemplate.spec.containers: Required value: is required when frontendSidecar is set",
				"spec.experimental.gpuMemoryService: Forbidden: GPU memory service is only supported for worker, prefill, or decode components",
				"spec.experimental.gpuMemoryService: Forbidden: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
			},
		},
		{
			name: "invalid ingress",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Ingress = &nvidiacomv1alpha1.IngressSpec{Enabled: true}
			}),
			wantWebhookErrs: []string{"spec.ingress.host: Required value: is required when ingress is enabled"},
		},
		{
			name: "invalid volume mount",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.VolumeMounts = []nvidiacomv1alpha1.VolumeMount{{Name: "data"}}
			}),
			wantWebhookErrs: []string{"spec.volumeMounts[0].mountPoint: Required value: is required when useAsCompilationCache is false"},
		},
		{
			name: "invalid shared memory",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{}
			}),
			wantCELErr: "spec.sharedMemory: Invalid value: size is required when disabled is false",
		},

		// CEL rules generated into the standalone DCD CRD.
		{
			name: "v1alpha1 replicas below minAvailable are rejected by CEL",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantCELErr: "spec: Invalid value: minAvailable must be less than or equal to replicas unless replicas is 0",
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
			name: "v1alpha1 minAvailable change is rejected by CEL",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &oneReplica
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Replicas = &validReplicas
				dcd.Spec.MinAvailable = &validMinAvail
			}),
			wantCELErr: "spec: Invalid value: minAvailable is immutable after creation",
		},
		{
			name: "v1beta1 minAvailable change is rejected by CEL",
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
			name: "v1alpha1 inter-pod GMS client containers are rejected by CEL",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               true,
					Mode:                  nvidiacomv1alpha1.GMSModeInterPod,
					ExtraClientContainers: []string{"metrics"},
				}
			}),
			wantCELErr: "spec.gpuMemoryService: Invalid value: extraClientContainers is only supported with mode=intraPod",
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
			name: "v1alpha1 non-empty GMS extra client pods are rejected by CEL",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:         true,
					Mode:            nvidiacomv1alpha1.GMSModeInterPod,
					ExtraClientPods: []nvidiacomv1alpha1.GMSClientPodSpec{{Name: "client"}},
				}
			}),
			wantCELErr: "spec.gpuMemoryService: Invalid value: extraClientPods is reserved for inter-pod GMS and is not implemented yet",
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
			name: "v1beta1 EPP config on a worker is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp-config"}},
				}
			}),
			wantCELErr: "spec: Invalid value: eppConfig may only be set when type is epp",
		},
		{
			name: "v1beta1 checkpoint job with checkpointRef is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: k8sptr.To("existing-checkpoint"),
						Job:           &nvidiacomv1beta1.ComponentCheckpointJobConfig{},
					},
				}
			}),
			wantCELErr: "spec.experimental.checkpoint: Invalid value: checkpoint.job cannot be set when checkpointRef is specified",
		},

		// Shared v1alpha1 validation reached through standalone DCD admission.
		{
			name: "valid shared spec with all fields",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: true,
					Host:    "example.com",
				},
				VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
					{Name: "cache", MountPoint: "/cache"},
					{Name: "compilation", UseAsCompilationCache: true},
				},
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Size: resource.MustParse("1Gi"),
				},
			}),
		},
		{
			name: "empty dynamo namespace is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				DynamoNamespace: k8sptr.To(""),
			}),
		},
		{
			name: "disabled ingress does not require a host",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Ingress: &nvidiacomv1alpha1.IngressSpec{},
			}),
		},
		{
			name: "disabled shared memory does not require a size",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{Disabled: true},
			}),
		},
		{
			name: "vLLM ray service annotation reaches the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "ray"},
			}),
		},
		{
			name: "vLLM mp service annotation reaches the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "mp"},
			}),
		},
		{
			name: "invalid vLLM service annotation is rejected by the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid"},
			}),
			wantWebhookErrs: []string{`spec.annotations[nvidia.com/vllm-distributed-executor-backend]: Invalid value: "invalid": must be "mp" or "ray"`},
		},
		{
			name: "checkpoint without GMS is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: true,
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
				},
			}),
		},
		{
			name: "disabled checkpoint with GMS is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				Checkpoint:    &nvidiacomv1alpha1.ServiceCheckpointConfig{},
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
			}),
		},
		{
			name: "GMS extra client containers are accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               true,
					Mode:                  nvidiacomv1alpha1.GMSModeIntraPod,
					ExtraClientContainers: []string{"gms-loader"},
				},
			}),
		},
		{
			name: "checkpoint target container name is validated by the source schema",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled:             true,
					TargetContainerName: "Bad_Name",
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
				},
			}),
			wantSchemaErr: `spec.checkpoint.targetContainerName: Invalid value: "Bad_Name": spec.checkpoint.targetContainerName in body should match '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'`,
		},
		{
			name: "GMS extra client container names are validated by the source schema",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               true,
					Mode:                  nvidiacomv1alpha1.GMSModeIntraPod,
					ExtraClientContainers: []string{"Bad_Name"},
				},
			}),
			wantSchemaErr: `spec.gpuMemoryService.extraClientContainers[0]: Invalid value: "Bad_Name": spec.gpuMemoryService.extraClientContainers[0] in body should match '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'`,
		},
		{
			name: "checkpoint job with checkpointRef is rejected by source CEL",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled:       true,
					CheckpointRef: k8sptr.To("existing-checkpoint"),
					Job:           &nvidiacomv1alpha1.ServiceCheckpointJobConfig{},
				},
			}),
			wantCELErr: "spec.checkpoint: Invalid value: checkpoint.job cannot be set when checkpointRef is specified",
		},
		{
			name: "deprecated checkpoint mode with checkpointRef is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled:       true,
					Mode:          nvidiacomv1alpha1.CheckpointModeManual,
					CheckpointRef: k8sptr.To("existing-checkpoint"),
				},
			}),
		},
		{
			name: "checkpoint GMS clients require GMS",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: true,
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
					Job: &nvidiacomv1alpha1.ServiceCheckpointJobConfig{
						GMSClientContainers: []string{"gms-saver"},
					},
				},
			}),
			wantWebhookErrs: []string{"spec.experimental.checkpoint.job.gmsClientContainers: Forbidden: requires gpuMemoryService to be set"},
		},
		{
			name: "checkpoint GMS client names are validated by the source schema",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
					Job: &nvidiacomv1alpha1.ServiceCheckpointJobConfig{
						GMSClientContainers: []string{"Bad_Name"},
					},
				},
			}),
			wantSchemaErr: `spec.checkpoint.job.gmsClientContainers[0]: Invalid value: "Bad_Name": spec.checkpoint.job.gmsClientContainers[0] in body should match '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'`,
		},
		{
			name: "frontend sidecar without extra containers is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{Image: "frontend:latest"},
			}),
		},
		{
			name: "frontend sidecar container-name collision is rejected",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{Image: "frontend:latest"},
				ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.FrontendSidecarContainerName, Image: "conflict:latest"}},
				}},
			}),
			wantWebhookErrs: []string{`spec.frontendSidecar: Forbidden: cannot inject frontend sidecar: a container named "sidecar-frontend" already exists in extraPodSpec.containers`},
		},
		{
			name: "frontend sidecar with non-conflicting containers is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{Image: "frontend:latest"},
				ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "other-sidecar", Image: "other:latest"}},
				}},
			}),
		},

		// GMS and failover compatibility rules.
		{
			name: "GMS rejects non-worker components",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeFrontend,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
				},
			}),
			wantWebhookErrs: []string{"spec.experimental.gpuMemoryService: Forbidden: GPU memory service is only supported for worker, prefill, or decode components"},
		},
		{
			name: "GMS requires a GPU",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType:    consts.ComponentTypeWorker,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			}),
			wantWebhookErrs: []string{"spec.experimental.gpuMemoryService: Forbidden: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1"},
		},
		{
			name: "GMS accepts GPU requests when limits are unset",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources: &nvidiacomv1alpha1.Resources{
					Requests: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
				},
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			}),
		},
		{
			name: "GMS rejects non-numeric GPU limits",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources: &nvidiacomv1alpha1.Resources{
					Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "not-a-number"},
				},
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			}),
			wantWebhookErrs: []string{"spec.experimental.gpuMemoryService: Forbidden: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1"},
		},
		{
			name: "checkpoint GMS clients reject inter-pod GMS",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: true,
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
					Job: &nvidiacomv1alpha1.ServiceCheckpointJobConfig{
						GMSClientContainers: []string{"gms-saver"},
					},
				},
			}),
			wantWebhookErrs: []string{
				"spec.experimental.checkpoint.job.gmsClientContainers: Forbidden: is only supported with gpuMemoryService.mode=IntraPod",
				"spec.experimental.checkpoint: Forbidden: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
			},
		},
		{
			name: "checkpoint GMS clients accept intra-pod GMS",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: dcdAdmissionVLLMBackend,
					},
					Job: &nvidiacomv1alpha1.ServiceCheckpointJobConfig{
						GMSClientContainers: []string{"gms-saver"},
					},
				},
			}),
		},
		{
			name: "standalone inter-pod GMS is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
			}),
		},
		{
			name: "intra-pod GMS is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
			}),
		},
		{
			name: "unset GMS mode is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType:    consts.ComponentTypeWorker,
				Resources:        workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			}),
		},
		{
			name: "inter-pod failover requires GMS",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			}),
			wantWebhookErrs: []string{`spec.experimental.failover: Forbidden: gpuMemoryService is required when failover mode is "InterPod"`},
		},
		{
			name: "inter-pod failover requires matching GMS mode",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			}),
			wantWebhookErrs: []string{`spec.experimental.failover.mode: Invalid value: "InterPod": must match gpuMemoryService.mode "IntraPod"`},
		},
		{
			name: "matching inter-pod GMS failover is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			}),
		},
		{
			name: "disabled failover permits dormant shadow configuration",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{NumShadows: 2},
			}),
		},
		{
			name: "intra-pod failover rejects multiple shadows",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeIntraPod,
					NumShadows: 2,
				},
			}),
			wantWebhookErrs: []string{`spec.failover.numShadows: Invalid value: 2: is invalid for mode="intraPod": intraPod uses a fixed 1 primary + 1 shadow sidecar; use failover.mode="interPod" to configure numShadows`},
		},
		{
			name: "single-shadow intra-pod failover is accepted",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeIntraPod,
					NumShadows: 1,
				},
			}),
		},

		// Compatibility warnings.
		{
			name: "deprecated autoscaling emits a warning",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
				//nolint:staticcheck // SA1019: Intentionally testing the deprecated compatibility warning.
				Autoscaling: &nvidiacomv1alpha1.Autoscaling{
					Enabled:     true,
					MinReplicas: 1,
					MaxReplicas: 10,
				},
			}),
			wantWarnings: []string{"spec.autoscaling is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md"},
		},
		{
			name: "deprecated dynamo namespace warning shows calculated namespace",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Namespace = "hannahz"
				dcd.OwnerReferences = []metav1.OwnerReference{{Kind: "DynamoGraphDeployment", Name: "trtllm-disagg"}}
				dcd.Spec.DynamoNamespace = k8sptr.To("my-custom-namespace")
			}),
			wantWarnings: []string{`spec.dynamoNamespace is deprecated and ignored. Value "my-custom-namespace" will be replaced with "hannahz-trtllm-disagg". Remove this field from your configuration`},
		},

		// EPP rules and their source-version ownership.
		{
			name: "v1alpha1 EPP cannot be multinode",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
			}),
			wantWebhookErrs: []string{
				"spec.multinode: Forbidden: EPP component cannot be multinode",
				"spec.eppConfig: Required value: is required for EPP components",
			},
		},
		{
			name: "v1alpha1 EPP requires one replica",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				Replicas:      &validMinAvail,
			}),
			wantWebhookErrs: []string{
				"spec.replicas: Invalid value: 2: EPP component must have exactly 1 replica",
				"spec.eppConfig: Required value: is required for EPP components",
			},
		},
		{
			name: "v1alpha1 EPP requires configuration",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Required value: is required for EPP components"},
		},
		{
			name: "v1beta1 EPP without configuration reaches and is rejected by the v1beta1 webhook",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Required value: is required for EPP components"},
		},
		{
			name: "v1alpha1 empty EPP config reaches and is rejected by the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				EPPConfig:     &nvidiacomv1alpha1.EPPConfig{},
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Forbidden: exactly one of configMapRef or config is required"},
		},
		{
			name: "v1beta1 empty EPP config is rejected by source CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{}
			}),
			wantCELErr: "spec.eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			name: "v1alpha1 conflicting EPP config reaches and is rejected by the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				EPPConfig: &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp-config"}},
					Config: &apixv1alpha1.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				},
			}),
			wantWebhookErrs: []string{"spec.eppConfig: Forbidden: exactly one of configMapRef or config is required"},
		},
		{
			name: "v1beta1 conflicting EPP config is rejected by source CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp-config"}},
					Config: &apixv1alpha1.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantCELErr: "spec.eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			name: "v1alpha1 EPP config map without a name reaches and is rejected by the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				EPPConfig: &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{},
				},
			}),
			wantWebhookErrs: []string{"spec.eppConfig.configMapRef.name: Required value: is required"},
		},
		{
			name: "valid v1alpha1 EPP config reaches the webhook",
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeEPP,
				Replicas:      &oneReplica,
				EPPConfig: &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp-config"}},
				},
			}),
		},
		{
			name: "valid v1beta1 EPP config reaches the v1beta1 webhook",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.ComponentType = nvidiacomv1beta1.ComponentTypeEPP
				dcd.Spec.Replicas = &oneReplica
				dcd.Spec.EPPConfig = &nvidiacomv1beta1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp-config"}},
				}
			}),
		},

		// Pair shared pod-template validation across both served source versions.
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
			name: "v1alpha1 sidecar without image reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
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
			name: "v1alpha1 init container without image reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
		},
		{
			name: "v1beta1 invalid pod template backend annotation is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantCELErr: "spec.podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name: "v1beta1 valid pod template backend annotation reaches the webhook",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
		},
		{
			name: "v1alpha1 invalid extra pod metadata annotation reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{
					Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid"},
				}
			}),
		},
		{
			name:       "valid v1beta1 deployment reaches the v1beta1 webhook",
			deployment: betaDCDForAdmission(nil),
		},
		{
			name: "v1alpha1 update without changes reaches the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
		},
		{
			name:          "v1beta1 update without changes reaches the v1beta1 webhook",
			oldDeployment: betaDCDForAdmission(nil),
			deployment:    betaDCDForAdmission(nil),
		},
		{
			name: "v1alpha1 backend framework update reaches and is rejected by the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionVLLMBackend
			}),
			wantWebhookErrs: []string{`spec.backendFramework: Invalid value: "vllm": is immutable and cannot be changed after creation`},
			wantWarnings:    []string{"Changing spec.backendFramework may cause unexpected behavior"},
		},
		{
			name: "v1beta1 backend framework update reaches and is rejected by the v1beta1 webhook",
			oldDeployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionVLLMBackend
			}),
			wantWebhookErrs: []string{`spec.backendFramework: Invalid value: "vllm": is immutable and cannot be changed after creation`},
			wantWarnings:    []string{"Changing spec.backendFramework may cause unexpected behavior"},
		},
		{
			name: "v1alpha1 replicas update reaches the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
				dcd.Spec.Replicas = &oneReplica
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
				dcd.Spec.Replicas = &validReplicas
			}),
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
			name:          "v1beta1 multinode layout change is rejected by the shared update validator",
			oldDeployment: betaDCDForAdmission(nil),
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{`spec.multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`},
		},
		{
			name: "v1alpha1 update aggregates create and DCD-specific update errors",
			oldDeployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      &oneReplica,
				MinAvailable:  &oneReplica,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			}),
			deployment: alphaDCDWithSharedSpec(nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Replicas:      &oneReplica,
				MinAvailable:  &oneReplica,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 2,
				},
			}),
			wantWebhookErrs: []string{
				"spec.minAvailable: Forbidden: is currently supported only for Grove-backed DynamoGraphDeployment components",
				"spec.experimental.failover.numShadows: Invalid value: 2: is immutable for inter-pod GMS failover; delete and recreate the DynamoComponentDeployment to change it",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := admissionUnstructured(t, tt.deployment)
			var oldRequest map[string]any
			if tt.oldDeployment != nil {
				oldRequest = admissionUnstructured(t, tt.oldDeployment)
			}

			version := admissionSourceVersion(t, tt.deployment)
			if tt.oldDeployment != nil && admissionSourceVersion(t, tt.oldDeployment) != version {
				t.Fatalf("old and current source versions differ")
			}
			requestValidator, ok := requestValidators[version]
			if !ok {
				t.Fatalf("no request validator for source version %q", version)
			}

			schemaErrs := requestValidator.validateSchema(request, oldRequest)
			if tt.wantSchemaErr != "" {
				if tt.wantCELErr != "" || len(tt.wantWebhookErrs) != 0 || len(tt.wantWarnings) != 0 {
					t.Fatal("schema rejection cannot have downstream expectations")
				}
				assertRequestValidationError(t, schemaErrs, tt.wantSchemaErr)
				return
			}
			if len(schemaErrs) != 0 {
				t.Fatalf("schema errors = %v, want none", schemaErrs)
			}

			celErrs := requestValidator.celValidator(request, oldRequest)
			if tt.wantCELErr != "" {
				if len(tt.wantWebhookErrs) != 0 || len(tt.wantWarnings) != 0 {
					t.Fatal("CEL rejection cannot have webhook expectations")
				}
				assertRequestValidationError(t, celErrs, tt.wantCELErr)
				return
			}
			if len(celErrs) != 0 {
				t.Fatalf("CEL errors = %v, want none", celErrs)
			}

			handler := NewDynamoComponentDeploymentHandler()
			ctx := dgdAdmissionContext(dgdAdmissionOperation(tt.oldDeployment), nvidiacomv1beta1.DynamoComponentDeploymentGVK)
			ctx = features.WithGate(ctx, features.Gates{Checkpoint: true})
			var warnings []string
			var err error
			if tt.oldDeployment == nil {
				warnings, err = handler.ValidateCreate(ctx, dcdAdmissionBeta(t, tt.deployment))
			} else {
				warnings, err = handler.ValidateUpdate(
					ctx,
					dcdAdmissionBeta(t, tt.oldDeployment),
					dcdAdmissionBeta(t, tt.deployment),
				)
			}
			assertDCDWebhookErrors(t, err, tt.wantWebhookErrs)
			if !slices.Equal(warnings, tt.wantWarnings) {
				t.Fatalf("webhook warnings = %v, want %v", warnings, tt.wantWarnings)
			}
		})
	}
}

func assertDCDWebhookErrors(t *testing.T, err error, want []string) {
	t.Helper()
	if len(want) == 0 {
		if err != nil {
			t.Fatalf("webhook error = %v, want none", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("webhook errors = nil, want %v", want)
	}
	statusErr, ok := err.(*k8serrors.StatusError)
	if !ok || !k8serrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	got := make([]string, len(causes))
	for i, cause := range causes {
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		got[i] = fmt.Sprintf("%s: %s", cause.Field, cause.Message)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("webhook errors = %v, want %v", got, want)
	}
}

func alphaDCDForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoComponentDeployment),
) *nvidiacomv1alpha1.DynamoComponentDeployment {
	dcd := &nvidiacomv1alpha1.DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoComponentDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: dcdAdmissionVLLMBackend,
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:   "worker",
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dcd
}

func alphaDCDWithSharedSpec(
	spec nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1alpha1.DynamoComponentDeployment {
	return alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
		dcd.Spec.DynamoComponentDeploymentSharedSpec = spec
	})
}

func betaDCDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoComponentDeployment),
) *nvidiacomv1beta1.DynamoComponentDeployment {
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       "DynamoComponentDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: dcdAdmissionVLLMBackend,
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dcd
}

func dcdAdmissionBeta(t *testing.T, deployment runtime.Object) *nvidiacomv1beta1.DynamoComponentDeployment {
	t.Helper()
	switch deployment := deployment.(type) {
	case *nvidiacomv1alpha1.DynamoComponentDeployment:
		beta := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := deployment.ConvertTo(beta); err != nil {
			t.Fatalf("convert v1alpha1 DCD to v1beta1: %v", err)
		}
		return beta
	case *nvidiacomv1beta1.DynamoComponentDeployment:
		return deployment.DeepCopy()
	default:
		t.Fatalf("unsupported DCD type %T", deployment)
		return nil
	}
}
