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

package validation

import (
	"context"
	"fmt"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	dgdAdmissionWorkerName      = "worker"
	dgdAdmissionUpperWorkerName = "WORKER"
)

func TestDynamoGraphDeploymentValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamographdeployments.yaml")

	tests := []struct {
		name           string
		deployment     runtime.Object
		oldDeployment  runtime.Object
		mutateRequest  func(*testing.T, map[string]any)
		groveEnabled   bool
		wantSchemaErr  string
		wantCELErr     string
		wantWebhookErr string
	}{
		{
			name:       "valid deployment with components",
			deployment: betaDGDForAdmission(nil),
		},
		{
			name: "no components",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components = nil
			}),
			wantWebhookErr: "spec.components must have at least one component",
		},
		{
			name: "component replicas must be non-negative",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Replicas = k8sptr.To(int32(-1))
			}),
			wantSchemaErr: "spec.components[1].replicas: Invalid value: -1: spec.components[1].replicas in body should be greater than or equal to 0",
		},
		{
			name: "component minAvailable requires Grove",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			wantWebhookErr: "spec.components[worker].minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components",
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
			wantWebhookErr: "spec.topologyConstraint with clusterTopologyName is required when any topology constraint is set",
		},
		{
			name:         "inter-pod GMS requires Grove",
			groveEnabled: false,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
			}),
			wantWebhookErr: `spec.components[worker]: experimental.gpuMemoryService.mode="InterPod" requires the Grove pathway`,
		},
		{
			name:         "inter-pod GMS requires vLLM backend",
			groveEnabled: true,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.BackendFramework = "sglang"
				enableBetaInterPodGMS(betaWorkerComponent(dgd))
			}),
			wantWebhookErr: `spec.components[worker]: the inter-pod GMS layout (experimental.gpuMemoryService.mode="InterPod") is currently supported only for vLLM`,
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
			wantWebhookErr: `failover requires per-container K8s discovery; set annotation "nvidia.com/dynamo-kube-discovery-mode" to "container"`,
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
			wantWebhookErr: "spec.components[worker].experimental.gpuMemoryService: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
		},

		// Pair every validation rule changed by this PR across both served source versions.
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
			name: "v1beta1 case-insensitive component names are rejected by CEL on update",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
			}),
			deployment: dgdAdmissionWithLabel(t, betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[0].ComponentName = dgdAdmissionWorkerName
				dgd.Spec.Components[1].ComponentName = dgdAdmissionUpperWorkerName
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
			name: "v1alpha1 converted compilation cache mount with an empty PVC name reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].VolumeMounts = []nvidiacomv1alpha1.VolumeMount{{
					UseAsCompilationCache: true,
				}}
			}),
			mutateRequest: setAlphaCompilationCacheVolumeNameEmpty,
		},
		{
			name: "v1beta1 sidecars must provide an image in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "metrics"}},
				}}
			}),
			wantCELErr: "spec.components[1].podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 converted sidecar without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name: "v1alpha1 frontend sidecar without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].FrontendSidecar = &nvidiacomv1alpha1.FrontendSidecarSpec{}
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name: "v1beta1 init containers must provide an image in CEL",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantCELErr: "spec.components[1].podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 converted init container without image reaches the webhook",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
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
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantCELErr: "spec.components[1].podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name: "v1beta1 valid pod template backend annotation reaches the webhook",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
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
			wantWebhookErr: `spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend] has invalid value "invalid": must be "mp" or "ray"`,
		},
		{
			name: "v1beta1 frontend sidecar must reference an existing container",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.FrontendSidecar = k8sptr.To("missing")
				worker.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}},
				}}
			}),
			wantWebhookErr: `spec.components[worker].frontendSidecar "missing" does not match any podTemplate.spec.containers name`,
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
					Containers: []corev1.Container{{Name: consts.MainContainerName}},
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			current := admissionUnstructured(t, tt.deployment)
			if tt.mutateRequest != nil {
				tt.mutateRequest(t, current)
			}
			var old map[string]any
			if tt.oldDeployment != nil {
				old = admissionUnstructured(t, tt.oldDeployment)
			}

			version := admissionSourceVersion(t, tt.deployment)
			requestValidator, ok := requestValidators[version]
			if !ok {
				t.Fatalf("no request validator for source version %q", version)
			}
			schemaErrs := requestValidator.validateSchema(current, old)
			if tt.wantSchemaErr != "" {
				assertRequestValidationError(t, schemaErrs, tt.wantSchemaErr)
				return
			}
			if len(schemaErrs) != 0 {
				t.Fatalf("schema errors = %v, want none", schemaErrs)
			}

			celErrs := requestValidator.celValidator(current, old)
			if tt.wantCELErr != "" {
				assertRequestValidationError(t, celErrs, tt.wantCELErr)
				return
			}
			if len(celErrs) != 0 {
				t.Fatalf("CEL errors = %v, want none", celErrs)
			}

			oldBeta := dgdAdmissionBeta(t, tt.oldDeployment)
			currentBeta := dgdAdmissionBeta(t, tt.deployment)
			handler := NewDynamoGraphDeploymentHandler(nil, "", tt.groveEnabled)
			ctx := dgdAdmissionContext(dgdAdmissionOperation(tt.oldDeployment), nvidiacomv1beta1.DynamoGraphDeploymentGVK)

			var err error
			if tt.oldDeployment == nil {
				_, err = handler.ValidateCreate(ctx, currentBeta)
			} else {
				_, err = handler.ValidateUpdate(ctx, oldBeta, currentBeta)
			}
			assertBetaValidationError(t, err, tt.wantWebhookErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateCheckpointFallback(t *testing.T) {
	deployment := betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
		worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
			Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
				Enabled:       true,
				CheckpointRef: k8sptr.To("existing-checkpoint"),
				Job:           &nvidiacomv1beta1.ComponentCheckpointJobConfig{},
			},
		}
	})

	validator := NewDynamoGraphDeploymentValidator(nil, false)
	_, err := validator.Validate(context.Background(), deployment)
	assertBetaValidationError(
		t,
		err,
		"spec.components[worker].experimental.checkpoint.job cannot be set when checkpointRef is specified",
	)
}

func TestDynamoGraphDeploymentValidator_GroveSchedulingMatrix(t *testing.T) {
	longDGDName := "test-graph-" + strings.Repeat("x", 50)
	boundaryComponentName := "w" + strings.Repeat("x", 36)
	tooLongComponentName := boundaryComponentName + "x"

	tests := []struct {
		name         string
		groveEnabled bool
		mutate       func(*nvidiacomv1beta1.DynamoGraphDeployment)
		wantErr      string
	}{
		{
			name:         "priority class requires Grove",
			groveEnabled: false,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = "high-priority"
			},
			wantErr: "spec.priorityClassName requires the Grove pathway",
		},
		{
			name:         "priority class is allowed with Grove",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = "high-priority"
			},
		},
		{
			name:         "minAvailable must be positive",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(0))
			},
			wantErr: "spec.components[worker].minAvailable must be greater than 0",
		},
		{
			name:         "replicas must cover minAvailable unless scaled to zero",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(1))
				worker.MinAvailable = k8sptr.To(int32(2))
			},
			wantErr: "spec.components[worker].replicas must be 0 or greater than or equal to minAvailable",
		},
		{
			name:         "replicas zero can keep minAvailable for scale-up intent",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Replicas = k8sptr.To(int32(0))
				worker.MinAvailable = k8sptr.To(int32(2))
			},
		},
		{
			name:         "rendered Grove resource name length accepts boundary",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = boundaryComponentName
			},
		},
		{
			name:         "rendered Grove resource name length rejects overflow",
			groveEnabled: true,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = tooLongComponentName
			},
			wantErr: "combined resource name length",
		},
		{
			name:         "rendered Grove resource name length is skipped outside Grove",
			groveEnabled: false,
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Name = longDGDName
				betaWorkerComponent(dgd).ComponentName = tooLongComponentName
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := newBetaDGDForValidation()
			tt.mutate(deployment)

			validator := NewDynamoGraphDeploymentValidator(nil, tt.groveEnabled)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateAggregatesErrors(t *testing.T) {
	deployment := betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
		spec.Restart = &nvidiacomv1beta1.Restart{}
		spec.Components[0].Replicas = k8sptr.To(int32(-1))
		spec.Components[1].Replicas = k8sptr.To(int32(-2))
	})
	deployment.Annotations = map[string]string{
		consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver",
		consts.KubeAnnotationDynamoKubeDiscoveryMode:     "bad-mode",
	}

	validator := NewDynamoGraphDeploymentValidator(nil, true)
	_, err := validator.Validate(context.Background(), deployment)
	for _, wantErr := range []string{
		"annotation nvidia.com/dynamo-operator-origin-version has invalid value",
		"annotation nvidia.com/dynamo-kube-discovery-mode has invalid value",
		"spec.restart.id is required",
		"spec.components[frontend].replicas must be non-negative",
		"spec.components[worker].replicas must be non-negative",
	} {
		assertBetaValidationError(t, err, wantErr)
	}
}

func TestDynamoGraphDeploymentValidator_AnnotationMatrix(t *testing.T) {
	tests := []struct {
		name       string
		annotation string
		value      string
		wantErr    string
	}{
		{
			name:       "origin version accepts semver",
			annotation: consts.KubeAnnotationDynamoOperatorOriginVersion,
			value:      "1.2.3",
		},
		{
			name:       "origin version rejects non-semver",
			annotation: consts.KubeAnnotationDynamoOperatorOriginVersion,
			value:      "not-semver",
			wantErr:    "annotation nvidia.com/dynamo-operator-origin-version has invalid value",
		},
		{
			name:       "vLLM backend accepts mp",
			annotation: consts.KubeAnnotationVLLMDistributedExecutorBackend,
			value:      "mp",
		},
		{
			name:       "vLLM backend accepts ray case-insensitively",
			annotation: consts.KubeAnnotationVLLMDistributedExecutorBackend,
			value:      "RAY",
		},
		{
			name:       "vLLM backend rejects unknown value",
			annotation: consts.KubeAnnotationVLLMDistributedExecutorBackend,
			value:      "typo",
			wantErr:    "annotation nvidia.com/vllm-distributed-executor-backend has invalid value",
		},
		{
			name:       "discovery mode accepts pod",
			annotation: consts.KubeAnnotationDynamoKubeDiscoveryMode,
			value:      "pod",
		},
		{
			name:       "discovery mode accepts container",
			annotation: consts.KubeAnnotationDynamoKubeDiscoveryMode,
			value:      "container",
		},
		{
			name:       "discovery mode rejects unknown value",
			annotation: consts.KubeAnnotationDynamoKubeDiscoveryMode,
			value:      "endpoint",
			wantErr:    "annotation nvidia.com/dynamo-kube-discovery-mode has invalid value",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := newBetaDGDForValidation()
			deployment.Annotations = map[string]string{tt.annotation: tt.value}

			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateAlphaCompatibility(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*nvidiacomv1alpha1.DynamoGraphDeployment)
		wantErr string
	}{
		{
			name: "alpha PVC create requires storage fields",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{
					{
						Name:   k8sptr.To("cache"),
						Create: k8sptr.To(true),
					},
				}
			},
			wantErr: "spec.pvcs[0].storageClass is required when create is true",
		},
		{
			name: "alpha ingress requires host",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				className := "nginx"
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
					Ingress: &nvidiacomv1alpha1.IngressSpec{
						Enabled:                    true,
						IngressControllerClassName: &className,
					},
				}
			},
			wantErr: "spec.services[frontend].ingress.host is required when ingress is enabled",
		},
		{
			name: "alpha service annotations are validated",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "typo",
				}
			},
			wantErr: `spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend] has invalid value "typo"`,
		},
		{
			name: "alpha volume mounts require mount point unless used as compilation cache",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].VolumeMounts = []nvidiacomv1alpha1.VolumeMount{
					{
						Name: "cache",
					},
				}
			},
			wantErr: "spec.services[worker].volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
		},
		{
			name: "alpha sharedMemory requires size when enabled",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
				}
			},
			wantErr: "spec.services[worker].sharedMemory.size is required when disabled is false",
		},
		{
			name: "alpha frontend sidecar rejects generated container name conflict",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
					FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
						Image: "custom/frontend:latest",
					},
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
						PodSpec: &corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  consts.FrontendSidecarContainerName,
									Image: "custom/frontend:latest",
								},
							},
						},
					},
				}
			},
			wantErr: `spec.services[frontend]: cannot inject frontend sidecar: a container named "sidecar-frontend" already exists in extraPodSpec.containers`,
		},
		{
			name: "disabled alpha GMS still validates extra client container names",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               false,
					ExtraClientContainers: []string{"Bad_Name"},
				}
			},
			wantErr: `spec.services[worker].gpuMemoryService.extraClientContainers[0] "Bad_Name" is not a valid Kubernetes container name`,
		},
		{
			name: "nil alpha service entry is rejected",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["ghost"] = nil
			},
			wantErr: "spec.services[ghost] must not be null",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDFromAlpha(t, tt.mutate)
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateAlphaCompatibilityAdditionalEdges(t *testing.T) {
	t.Run("valid preserved alpha-only fields are accepted", func(t *testing.T) {
		deployment := betaDGDFromAlpha(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
			host := "worker.example.com"
			dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{
				{
					Name:   k8sptr.To("cache"),
					Create: k8sptr.To(false),
				},
			}
			service := dgd.Spec.Services["worker"]
			service.Ingress = &nvidiacomv1alpha1.IngressSpec{
				Enabled: true,
				Host:    host,
			}
			service.Annotations = map[string]string{
				consts.KubeAnnotationVLLMDistributedExecutorBackend: "ray",
			}
			service.VolumeMounts = []nvidiacomv1alpha1.VolumeMount{
				{
					Name:                  "cache",
					UseAsCompilationCache: true,
				},
			}
			service.SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{Disabled: true}
			service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled:               false,
				ExtraClientContainers: []string{"metrics"},
			}
		})

		validator := NewDynamoGraphDeploymentValidator(nil, true)
		_, err := validator.Validate(context.Background(), deployment)
		assertBetaValidationError(t, err, "")
	})

	t.Run("missing alpha PVC name is rejected", func(t *testing.T) {
		deployment := betaDGDFromAlpha(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
			dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{{}}
		})

		validator := NewDynamoGraphDeploymentValidator(nil, true)
		_, err := validator.Validate(context.Background(), deployment)
		assertBetaValidationError(t, err, "spec.pvcs[0].name is required")
	})

	t.Run("multiple alpha PVC errors are returned together", func(t *testing.T) {
		deployment := betaDGDFromAlpha(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
			dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{
				{
					Create: k8sptr.To(true),
				},
			}
		})

		validator := NewDynamoGraphDeploymentValidator(nil, true)
		_, err := validator.Validate(context.Background(), deployment)
		for _, wantErr := range []string{
			"spec.pvcs[0].name is required",
			"spec.pvcs[0].storageClass is required when create is true",
			"spec.pvcs[0].size is required when create is true",
			"spec.pvcs[0].volumeAccessMode is required when create is true",
		} {
			assertBetaValidationError(t, err, wantErr)
		}
	})
}

func TestDynamoGraphDeploymentValidator_ValidateAlphaCompatibilityWarnings(t *testing.T) {
	legacyNamespace := "legacy-namespace"
	deployment := betaDGDFromAlpha(t, func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
		service := dgd.Spec.Services["worker"]
		service.DynamoNamespace = &legacyNamespace
		//nolint:staticcheck // SA1019: Intentionally testing deprecated field warnings.
		service.Autoscaling = &nvidiacomv1alpha1.Autoscaling{Enabled: true}
	})

	validator := NewDynamoGraphDeploymentValidator(nil, true)
	warnings, err := validator.Validate(context.Background(), deployment)
	if err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	assertWarningsContain(t, warnings, "spec.services[worker].dynamoNamespace is deprecated and ignored")
	assertWarningsContain(t, warnings, "spec.services[worker].autoscaling is deprecated and ignored")
}

func TestDynamoGraphDeploymentValidator_ValidateConvertedAlphaResourceSemantics(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*nvidiacomv1alpha1.DynamoGraphDeployment)
		wantErr string
	}{
		{
			name: "GMS accepts GPU from alpha dedicated resources",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.Resources = &nvidiacomv1alpha1.Resources{
					Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
				}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			},
		},
		{
			name: "GMS accepts GPU from alpha extraPodSpec main container resources",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{
								corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
							},
						},
					},
				}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			},
		},
		{
			name: "GMS accepts alpha GPUType resource after conversion",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				service := dgd.Spec.Services["worker"]
				service.Resources = &nvidiacomv1alpha1.Resources{
					Limits: &nvidiacomv1alpha1.ResourceItem{
						GPU:     "1",
						GPUType: "example.com/gpu",
					},
				}
				service.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDFromAlpha(t, tt.mutate)
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_RestartMatrix(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec)
		wantErr string
	}{
		{
			name: "missing restart id",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{}
			},
			wantErr: "spec.restart.id is required",
		},
		{
			name: "duplicate restart order",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "worker", "worker")
			},
			wantErr: "spec.restart.strategy.order must be unique",
		},
		{
			name: "unknown restart order component",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "ghost")
			},
			wantErr: "spec.restart.strategy.order contains unknown component: ghost",
		},
		{
			name: "restart order missing component",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "worker")
			},
			wantErr: "spec.restart.strategy.order must have the same number of unique components as the deployment",
		},
		{
			name: "empty sequential restart order is valid",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential)
			},
		},
		{
			name: "complete sequential restart order is valid",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeSequential, "frontend", "worker")
			},
		},
		{
			name: "parallel restart without order is valid",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeParallel)
			},
		},
		{
			name: "parallel restart rejects order",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = betaRestart(nvidiacomv1beta1.RestartStrategyTypeParallel, "frontend", "worker")
			},
			wantErr: "spec.restart.strategy.order cannot be specified when strategy is parallel",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDWithSpec(tt.mutate)
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_TopologyMatrix(t *testing.T) {
	topologyManager := newGroveTopologyTestManager(t, newTestClusterTopology())
	missingTopologyManager := newGroveTopologyTestManager(t)

	tests := []struct {
		name    string
		mgr     ctrl.Manager
		mutate  func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec)
		wantErr string
	}{
		{
			name: "spec pack domain format is validated",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "Bad_Domain",
				}
			},
			wantErr: `spec.topologyConstraint.packDomain "Bad_Domain" is not a valid topology domain`,
		},
		{
			name: "component topology requires deployment topology",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			},
			wantErr: "spec.topologyConstraint with clusterTopologyName is required when any topology constraint is set",
		},
		{
			name: "component topology requires pack domain",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				spec.Components[0].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			},
			wantErr: "spec.components[frontend].topologyConstraint.packDomain is required",
		},
		{
			name: "deployment topology without pack domain requires every component topology",
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			},
			wantErr: "spec.components[frontend].topologyConstraint is required because spec.topologyConstraint.packDomain is not set",
		},
		{
			name: "deployment pack domain can be inherited",
			mgr:  topologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			},
		},
		{
			name: "deployment pack domain can be mixed with narrower component topology",
			mgr:  topologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "zone",
				}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			},
		},
		{
			name: "component topology with deployment topology is valid",
			mgr:  topologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				spec.Components[0].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			},
		},
		{
			name: "missing cluster topology is rejected",
			mgr:  missingTopologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "missing-topology",
					PackDomain:          "rack",
				}
			},
			wantErr: `topology-aware scheduling requires a ClusterTopology resource "missing-topology" but it was not found`,
		},
		{
			name: "pack domain must exist in cluster topology",
			mgr:  topologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "host",
				}
			},
			wantErr: `spec.topologyConstraint.packDomain: domain "host" does not exist in ClusterTopology "grove-topology"`,
		},
		{
			name: "component topology cannot be broader than spec topology",
			mgr:  topologyManager,
			mutate: func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
			},
			wantErr: `spec.components[worker]: topologyConstraint.packDomain "zone" is broader than spec-level "rack"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDWithSpec(tt.mutate)
			validator := NewDynamoGraphDeploymentValidator(tt.mgr, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_KvTransferPolicyMatrix(t *testing.T) {
	topologyManager := newGroveTopologyTestManager(t, newTestClusterTopology())
	missingTopologyManager := newGroveTopologyTestManager(t)

	tests := []struct {
		name      string
		mgr       ctrl.Manager
		mutateDGD func(*nvidiacomv1beta1.DynamoGraphDeployment)
		policy    *nvidiacomv1beta1.KvTransferPolicy
		wantErr   string
	}{
		{
			name:    "missing topology selector",
			policy:  &nvidiacomv1beta1.KvTransferPolicy{Domain: "zone"},
			wantErr: "spec.experimental.kvTransferPolicy: exactly one of labelKey or clusterTopologyName is required",
		},
		{
			name: "both topology selectors",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:            "topology.kubernetes.io/zone",
				ClusterTopologyName: "grove-topology",
				Domain:              "zone",
			},
			wantErr: "spec.experimental.kvTransferPolicy: exactly one of labelKey or clusterTopologyName is required",
		},
		{
			name: "invalid label key",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "bad prefix/zone",
				Domain:   "zone",
			},
			wantErr: `spec.experimental.kvTransferPolicy.labelKey "bad prefix/zone" is not a valid Kubernetes label key`,
		},
		{
			name: "invalid label key name segment",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/-zone",
				Domain:   "zone",
			},
			wantErr: `spec.experimental.kvTransferPolicy.labelKey "topology.kubernetes.io/-zone" is not a valid Kubernetes label key`,
		},
		{
			name: "label key policy is valid",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			},
		},
		{
			name: "invalid cluster topology name",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				ClusterTopologyName: "Bad_Name",
				Domain:              "zone",
			},
			wantErr: `spec.experimental.kvTransferPolicy.clusterTopologyName "Bad_Name" is not a valid Kubernetes resource name`,
		},
		{
			name: "cluster topology name requires Grove pathway",
			mutateDGD: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationEnableGrove: consts.KubeLabelValueFalse}
			},
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				ClusterTopologyName: "grove-topology",
				Domain:              "zone",
			},
			wantErr: "spec.experimental.kvTransferPolicy.clusterTopologyName requires the Grove pathway",
		},
		{
			name: "domain is required",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
			},
			wantErr: "spec.experimental.kvTransferPolicy.domain is required",
		},
		{
			name: "domain format is validated",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "Zone",
			},
			wantErr: `spec.experimental.kvTransferPolicy.domain "Zone" is not a valid topology domain`,
		},
		{
			name: "enforcement value is validated",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:    "topology.kubernetes.io/zone",
				Domain:      "zone",
				Enforcement: "sometimes",
			},
			wantErr: `spec.experimental.kvTransferPolicy.enforcement "sometimes" is invalid`,
		},
		{
			name: "preferred enforcement requires weight",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:    "topology.kubernetes.io/zone",
				Domain:      "zone",
				Enforcement: nvidiacomv1beta1.KvTransferEnforcementPreferred,
			},
			wantErr: `spec.experimental.kvTransferPolicy.preferredWeight is required when enforcement is "preferred"`,
		},
		{
			name: "preferred weight must be in range",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:        "topology.kubernetes.io/zone",
				Domain:          "zone",
				Enforcement:     nvidiacomv1beta1.KvTransferEnforcementPreferred,
				PreferredWeight: k8sptr.To(float32(1.2)),
			},
			wantErr: "spec.experimental.kvTransferPolicy.preferredWeight 1.2 is invalid",
		},
		{
			name: "required enforcement cannot set preferred weight",
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:        "topology.kubernetes.io/zone",
				Domain:          "zone",
				Enforcement:     nvidiacomv1beta1.KvTransferEnforcementRequired,
				PreferredWeight: k8sptr.To(float32(0.5)),
			},
			wantErr: "spec.experimental.kvTransferPolicy.preferredWeight must not be set when enforcement is \"required\"",
		},
		{
			name: "cluster topology policy is valid",
			mgr:  topologyManager,
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				ClusterTopologyName: "grove-topology",
				Domain:              "rack",
			},
		},
		{
			name: "cluster topology policy rejects missing topology",
			mgr:  missingTopologyManager,
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				ClusterTopologyName: "missing-topology",
				Domain:              "rack",
			},
			wantErr: `spec.experimental.kvTransferPolicy.clusterTopologyName "missing-topology" references a ClusterTopology resource that was not found`,
		},
		{
			name: "cluster topology policy rejects missing domain",
			mgr:  topologyManager,
			policy: &nvidiacomv1beta1.KvTransferPolicy{
				ClusterTopologyName: "grove-topology",
				Domain:              "host",
			},
			wantErr: `spec.experimental.kvTransferPolicy.domain "host" does not exist in ClusterTopology "grove-topology"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDWithKvTransferPolicy(tt.policy)
			if tt.mutateDGD != nil {
				tt.mutateDGD(deployment)
			}
			validator := NewDynamoGraphDeploymentValidator(tt.mgr, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_GMSFailoverMatrix(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*nvidiacomv1beta1.DynamoGraphDeployment)
		wantErr string
	}{
		{
			name: "GMS rejects frontend component",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaIntraPodGMS(&dgd.Spec.Components[0])
			},
			wantErr: "spec.components[frontend].experimental.gpuMemoryService: GPU memory service is only supported for worker components",
		},
		{
			name: "GMS requires main container GPU",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
				}
			},
			wantErr: "spec.components[worker].experimental.gpuMemoryService: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
		},
		{
			name: "GMS validates extra client container names",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"Bad_Name"}
			},
			wantErr: `spec.components[worker].experimental.gpuMemoryService.extraClientContainers[0] "Bad_Name" is not a valid Kubernetes container name`,
		},
		{
			name: "inter-pod GMS rejects extra client containers",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientContainers = []string{"metrics"}
			},
			wantErr: "spec.components[worker].experimental.gpuMemoryService.extraClientContainers is only supported with mode=IntraPod",
		},
		{
			name: "GMS extra client pods are still reserved",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.GPUMemoryService.ExtraClientPods = []nvidiacomv1beta1.GMSClientPodSpec{
					{Name: "client"},
				}
			},
			wantErr: "spec.components[worker].experimental.gpuMemoryService.extraClientPods is reserved for inter-pod GMS and is not implemented yet",
		},
		{
			name: "intra-pod failover requires GMS",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaContainerDiscovery(dgd)
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{Mode: nvidiacomv1beta1.GMSModeIntraPod},
				}
			},
			wantErr: "spec.components[worker].experimental.failover: intraPod failover requires gpuMemoryService to be set",
		},
		{
			name: "failover mode must match GMS mode",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode:       nvidiacomv1beta1.GMSModeInterPod,
					NumShadows: 1,
				}
			},
			wantErr: `spec.components[worker].experimental.failover: interPod failover requires gpuMemoryService.mode="InterPod"`,
		},
		{
			name: "intra-pod failover rejects custom shadow count",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				enableBetaContainerDiscovery(dgd)
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode:       nvidiacomv1beta1.GMSModeIntraPod,
					NumShadows: 2,
				}
			},
			wantErr: `spec.components[worker].experimental.failover.numShadows=2 is invalid for mode="IntraPod"`,
		},
		{
			name: "inter-pod failover requires GMS",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{
						Mode:       nvidiacomv1beta1.GMSModeInterPod,
						NumShadows: 1,
					},
				}
			},
			wantErr: `spec.components[worker].experimental.failover: interPod failover requires gpuMemoryService.mode="InterPod"`,
		},
		{
			name: "inter-pod failover requires positive shadow count",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaInterPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode: nvidiacomv1beta1.GMSModeInterPod,
				}
			},
			wantErr: "spec.components[worker].experimental.failover.numShadows must be >= 1",
		},
		{
			name: "inter-pod failover rejects frontend component",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				frontend := &dgd.Spec.Components[0]
				frontend.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Failover: &nvidiacomv1beta1.FailoverSpec{
						Mode:       nvidiacomv1beta1.GMSModeInterPod,
						NumShadows: 1,
					},
				}
			},
			wantErr: `spec.components[frontend]: GMS failover is not supported for type "frontend"`,
		},
		{
			name: "GMS snapshot combination requires env gate",
			mutate: func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				t.Setenv(consts.DynamoOperatorAllowGMSSnapshotEnvVar, "")
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}
			},
			wantErr: "spec.components[worker].experimental.checkpoint: GMS + Snapshot is temporarily disabled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := newBetaDGDForValidation()
			tt.mutate(deployment)
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateUpdate(t *testing.T) {
	const operatorPrincipal = "system:serviceaccount:dynamo-system:dynamo-operator"

	tests := []struct {
		name      string
		oldDGD    *nvidiacomv1beta1.DynamoGraphDeployment
		newDGD    *nvidiacomv1beta1.DynamoGraphDeployment
		userInfo  *authenticationv1.UserInfo
		principal string
		wantErr   string
		wantWarns bool
	}{
		{
			name:   "component topology is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = append(spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "extra",
					Replicas:      k8sptr.To(int32(1)),
				})
			}),
			wantErr: "component topology is immutable and cannot be modified after creation: components added: [extra]",
		},
		{
			name:   "component removal is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = spec.Components[:1]
			}),
			wantErr: "component topology is immutable and cannot be modified after creation: components removed: [worker]",
		},
		{
			name:   "component add and remove reports both sides",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					spec.Components[1],
					{
						ComponentName: "extra",
						Replicas:      k8sptr.To(int32(1)),
					},
				}
			}),
			wantErr: "component topology is immutable and cannot be modified after creation: components added: [extra], components removed: [frontend]",
		},
		{
			name:   "component reorder is allowed",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components[0], spec.Components[1] = spec.Components[1], spec.Components[0]
			}),
		},
		{
			name:   "single-node to multinode transition is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantErr: "spec.components[worker] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "node count-only update remains allowed",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
			}),
		},
		{
			name:   "spec topology constraint is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			wantErr: "spec.topologyConstraint is immutable and cannot be added, removed, or changed after creation",
		},
		{
			name: "spec topology constraint change is immutable",
			oldDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "rack",
				}
			}),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "grove-topology",
					PackDomain:          "zone",
				}
			}),
			wantErr: "spec.topologyConstraint is immutable and cannot be added, removed, or changed after creation",
		},
		{
			name: "unchanged topology constraints are allowed",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
		},
		{
			name:   "component topology constraint is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantErr: "spec.components[worker].topologyConstraint is immutable and cannot be added, removed, or changed after creation",
		},
		{
			name: "component topology constraint change is immutable",
			oldDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "zone"}
			}),
			wantErr: "spec.components[worker].topologyConstraint is immutable and cannot be added, removed, or changed after creation",
		},
		{
			name:   "kv transfer policy is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			}),
			wantErr: "spec.experimental.kvTransferPolicy is immutable and cannot be added, removed, or changed after creation",
		},
		{
			name: "unchanged kv transfer policy is allowed",
			oldDGD: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey: "topology.kubernetes.io/zone",
				Domain:   "zone",
			}),
			newDGD: betaDGDWithKvTransferPolicy(&nvidiacomv1beta1.KvTransferPolicy{
				LabelKey:    "topology.kubernetes.io/zone",
				Domain:      "zone",
				Enforcement: nvidiacomv1beta1.KvTransferEnforcementRequired,
			}),
		},
		{
			name:   "inter-pod GMS layout is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			wantErr: "spec.components[worker].experimental.gpuMemoryService.mode: the inter-pod GMS layout cannot be toggled after creation",
		},
		{
			name: "inter-pod failover toggle is immutable",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
				enableBetaInterPodFailover(worker, 1)
			}),
			wantErr: "spec.components[worker].experimental.failover: inter-pod GMS failover cannot be toggled after creation",
		},
		{
			name: "inter-pod failover shadow count is immutable",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
				enableBetaInterPodFailover(worker, 1)
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
				enableBetaInterPodFailover(worker, 2)
			}),
			wantErr: "spec.components[worker].experimental.failover.numShadows is immutable for inter-pod GMS failover",
		},
		{
			name: "scaling adapter blocks direct replica changes",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			userInfo: &authenticationv1.UserInfo{
				Username: "system:serviceaccount:default:regular-user",
			},
			principal: operatorPrincipal,
			wantErr:   "spec.components[worker].replicas cannot be modified directly when scaling adapter is enabled",
		},
		{
			name: "scaling adapter fails closed without user info",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			principal: operatorPrincipal,
			wantErr:   "spec.components[worker].replicas cannot be modified directly when scaling adapter is enabled",
		},
		{
			name: "operator can change scaling-adapter-owned replicas",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			userInfo: &authenticationv1.UserInfo{
				Username: operatorPrincipal,
			},
			principal: operatorPrincipal,
		},
		{
			name: "minAvailable is immutable once set",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.MinAvailable = k8sptr.To(int32(1))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.MinAvailable = k8sptr.To(int32(2))
			}),
			wantErr: "spec.components[worker].minAvailable is immutable after creation",
		},
		{
			name:   "backend framework changes warn and fail",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.BackendFramework = "sglang"
			}),
			wantErr:   "spec.backendFramework is immutable and cannot be changed after creation",
			wantWarns: true,
		},
		{
			name: "restart id cannot change during active rolling update",
			oldDGD: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "old"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
					}
				},
			),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "new"}
			}),
			wantErr: "spec.restart.id cannot be changed while a rolling update is InProgress",
		},
		{
			name: "restart id can stay unchanged during active rolling update",
			oldDGD: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "same"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
					}
				},
			),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "same"}
			}),
		},
		{
			name: "restart id can change after completed rolling update",
			oldDGD: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "old"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseCompleted,
					}
				},
			),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "new"}
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			warnings, err := validator.ValidateUpdate(tt.oldDGD, tt.newDGD, tt.userInfo, tt.principal)
			assertBetaValidationError(t, err, tt.wantErr)
			if tt.wantWarns && len(warnings) == 0 {
				t.Fatal("ValidateUpdate() expected warnings but got none")
			}
			if !tt.wantWarns && len(warnings) != 0 {
				t.Fatalf("ValidateUpdate() unexpected warnings: %v", warnings)
			}
		})
	}
}

func dgdAdmissionBeta(t *testing.T, deployment runtime.Object) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()
	if deployment == nil {
		return nil
	}
	switch deployment := deployment.(type) {
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		return deployment.DeepCopy()
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := deployment.ConvertTo(beta); err != nil {
			t.Fatalf("convert v1alpha1 DGD to v1beta1: %v", err)
		}
		return beta
	default:
		t.Fatalf("unsupported DGD type %T", deployment)
		return nil
	}
}

func dgdAdmissionOperation(oldDeployment runtime.Object) admissionv1.Operation {
	if oldDeployment == nil {
		return admissionv1.Create
	}
	return admissionv1.Update
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
					ComponentName: "frontend",
					ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					Replicas:      k8sptr.To(int32(1)),
				},
				{
					ComponentName: "worker",
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(2)),
				},
			},
		},
	}
}

func betaDGDFromAlpha(
	t *testing.T,
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()

	alpha := newAlphaDGDForCompatibilityValidation()
	mutate(alpha)

	beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	return beta
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
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(1)),
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

func enableBetaContainerDiscovery(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
	dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoKubeDiscoveryMode: "container"}
}

func enableBetaInterPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeInterPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: consts.MainContainerName,
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
						},
					},
				},
			},
		},
	}
}

func enableBetaIntraPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeIntraPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: consts.MainContainerName,
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
						},
					},
				},
			},
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

type fakeManager struct {
	ctrl.Manager // satisfies the rest of the interface; panics if unexpected methods are used
	client       client.Client
	config       *rest.Config
}

func (m *fakeManager) GetClient() client.Client { return m.client }
func (m *fakeManager) GetConfig() *rest.Config  { return m.config }

func newGroveTopologyTestManager(t *testing.T, objects ...runtime.Object) ctrl.Manager {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := grovev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add Grove scheme: %v", err)
	}
	return &fakeManager{
		client: fake.NewClientBuilder().WithScheme(scheme).WithRuntimeObjects(objects...).Build(),
		config: &rest.Config{},
	}
}

func newTestClusterTopology() *grovev1alpha1.ClusterTopology {
	return &grovev1alpha1.ClusterTopology{
		ObjectMeta: metav1.ObjectMeta{Name: "grove-topology"},
		Spec: grovev1alpha1.ClusterTopologySpec{
			Levels: []grovev1alpha1.TopologyLevel{
				{Domain: grovev1alpha1.TopologyDomainZone, Key: "topology.kubernetes.io/zone"},
				{Domain: grovev1alpha1.TopologyDomainRack, Key: "nvidia.com/rack"},
			},
		},
	}
}

func assertBetaValidationError(t *testing.T, err error, wantErr string) {
	t.Helper()
	if wantErr == "" {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected error containing %q but got nil", wantErr)
	}
	if !strings.Contains(err.Error(), wantErr) {
		t.Fatalf("error = %q, want to contain %q", err.Error(), wantErr)
	}
}

func assertWarningsContain(t *testing.T, warnings []string, want string) {
	t.Helper()
	for _, warning := range warnings {
		if strings.Contains(warning, want) {
			return
		}
	}
	t.Fatalf("warnings = %v, want one containing %q", warnings, want)
}
