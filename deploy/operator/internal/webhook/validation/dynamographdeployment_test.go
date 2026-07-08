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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/eppconfig"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/client-go/rest"
	k8sptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

const (
	dgdAdmissionWorkerName      = "worker"
	dgdAdmissionUpperWorkerName = "WORKER"
	dgdAdmissionOperator        = "system:serviceaccount:dynamo-system:dynamo-operator"
)

const sglangBackendFramework = "sglang"

func TestDynamoGraphDeploymentValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamographdeployments.yaml")
	defaultManager := newGroveTopologyTestManager(t, newTestClusterTopology())
	missingTopologyManager := newGroveTopologyTestManager(t)
	inferencePoolManager := newInferencePoolTestManager(t)
	longDGDName := "test-graph-" + strings.Repeat("x", 50)
	boundaryComponentName := "w" + strings.Repeat("x", 36)
	tooLongComponentName := boundaryComponentName + "x"

	tests := []struct {
		name          string
		deployment    runtime.Object
		oldDeployment runtime.Object
		mutateRequest func(*testing.T, map[string]any) // mutates the source-version request map
		manager       ctrl.Manager                     // supplies webhook dependencies
		groveDisabled bool                             // disables the configured Grove pathway
		environment   map[string]string                // sets process environment for the case
		userInfo      *authenticationv1.UserInfo       // supplies the admission request identity
		operator      string                           // sets the configured operator principal

		wantSchemaErr   string
		wantCELErr      string
		wantWebhookErrs []string
		wantWarnings    []string
		notWantErr      string
	}{
		// Baseline create-path rules.
		{
			name:       "valid deployment with components",
			deployment: betaDGDForAdmission(nil),
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
			wantSchemaErr: "spec.components[1].replicas: Invalid value: -1: spec.components[1].replicas in body should be greater than or equal to 0",
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
			wantWebhookErrs: []string{"spec.components[1].experimental.gpuMemoryService.mode: Forbidden: requires the Grove pathway, but Grove is disabled in the operator configuration"},
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
			wantWebhookErrs: []string{`spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend]: Invalid value: "invalid": must be "mp" or "ray"`},
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
			name: "v1beta1 changed minAvailable update is rejected by CEL",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(2))
			}),
			wantCELErr: "spec.components[1]: Invalid value: minAvailable is immutable after creation",
		},
		{
			name: "v1beta1 removed minAvailable update is rejected by CEL",
			oldDeployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				betaWorkerComponent(dgd).MinAvailable = k8sptr.To(int32(1))
			}),
			deployment: betaDGDForAdmission(nil),
			wantCELErr: "spec.components[1]: Invalid value: minAvailable is immutable after creation",
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
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy.clusterTopologyName: Forbidden: requires the Grove pathway; remove or unset annotation "nvidia.com/enable-grove" (currently "false")`},
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
			name:    "KV-transfer cluster topology policy rejects missing topology",
			manager: missingTopologyManager,
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
			name:        "GMS snapshot combination requires env gate",
			environment: map[string]string{consts.DynamoOperatorAllowGMSSnapshotEnvVar: ""},
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				worker := betaWorkerComponent(dgd)
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true}
			}),
			wantWebhookErrs: []string{"spec.components[1].experimental.checkpoint: Forbidden: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate"},
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
					ComponentType: consts.ComponentTypeFrontend,
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
			name:    "alpha EPP config sources are mutually exclusive",
			manager: inferencePoolManager,
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{
					ConfigMapRef: &corev1.ConfigMapKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "epp"}},
					Config: &eppconfig.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Invalid value: null: exactly one of configMapRef or config is required"},
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
					ComponentType: consts.ComponentTypeFrontend,
					FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
						Image: "custom/frontend:latest",
					},
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
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
			name: "nil alpha service entry is rejected",
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["ghost"] = nil
			}),
			wantSchemaErr: `spec.services.ghost: Invalid value: "null": spec.services.ghost in body must be of type object: "null"`,
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
			name:    "v1alpha1 EPP config without a source reaches the webhook without v1beta1 CEL",
			manager: inferencePoolManager,
			deployment: alphaDGDForAdmission(func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				worker := dgd.Spec.Services["worker"]
				worker.ComponentType = consts.ComponentTypeEPP
				worker.EPPConfig = &nvidiacomv1alpha1.EPPConfig{}
			}),
			wantWebhookErrs: []string{"spec.services[worker].eppConfig: Invalid value: null: exactly one of configMapRef or config is required"},
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
					Config: &eppconfig.EndpointPickerConfig{
						Plugins:            []apixv1alpha1.PluginSpec{},
						SchedulingProfiles: []apixv1alpha1.SchedulingProfile{},
					},
				}
			}),
			wantCELErr: "spec.components[1].eppConfig: Invalid value: exactly one of configMapRef or config must be specified",
		},
		{
			name:    "v1beta1 valid EPP config reaches the webhook",
			manager: inferencePoolManager,
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
				dgd.Spec.PriorityClassName = "high-priority"
			}),
			wantWebhookErrs: []string{"spec.priorityClassName: Forbidden: requires the Grove pathway, but Grove is disabled in the operator configuration"},
		},
		{
			name: "priority class is allowed with Grove",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.PriorityClassName = "high-priority"
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
				"spec.components[1].name: Invalid value: %q: combined resource name length 46 exceeds the 45-character pod-name limit (PCS name + component name); shorten DynamoGraphDeployment name %q or component name %q",
				tooLongComponentName,
				longDGDName,
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
			name: "deployment topology without pack domain requires every component topology",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Generation = 2
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "grove-topology"}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantWebhookErrs: []string{"spec.components[0].topologyConstraint: Required value: is required because spec.topologyConstraint.packDomain is not set"},
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
			name:    "missing cluster topology is rejected",
			manager: missingTopologyManager,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{
					ClusterTopologyName: "missing-topology",
					PackDomain:          "rack",
				}
			}),
			wantWebhookErrs: []string{`spec.topologyConstraint.clusterTopologyName: Invalid value: "missing-topology": references a ClusterTopologyBinding resource that was not found`},
		},
		{
			name:    "independent topology errors aggregate",
			manager: missingTopologyManager,
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Spec.TopologyConstraint = &nvidiacomv1beta1.SpecTopologyConstraint{ClusterTopologyName: "missing-topology"}
				dgd.Spec.Components[1].TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantWebhookErrs: []string{
				"spec.components[0].topologyConstraint: Required value: is required because spec.topologyConstraint.packDomain is not set",
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

		// Metadata annotation rules.
		{
			name: "origin version accepts semver",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.2.3"}
			}),
		},
		{
			name: "origin version rejects non-semver",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: "not-semver": must be valid semver`},
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
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: "ondelete": supported values: "RollingRecreate", "OnDelete"`},
		},
		{
			name: "Grove update strategy annotation rejects whitespace",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: " OnDelete "}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: " OnDelete ": supported values: "RollingRecreate", "OnDelete"`},
		},
		{
			name: "Grove update strategy annotation rejects unknown value",
			deployment: betaDGDForAdmission(func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
				dgd.Annotations = map[string]string{consts.KubeAnnotationGroveUpdateStrategy: "BlueGreen"}
			}),
			wantWebhookErrs: []string{`metadata.annotations[nvidia.com/grove-update-strategy]: Unsupported value: "BlueGreen": supported values: "RollingRecreate", "OnDelete"`},
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
					consts.KubeAnnotationDynamoOperatorOriginVersion:    "not-semver",
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					consts.KubeAnnotationDynamoKubeDiscoveryMode:        "invalid",
				}
			}),
			wantWebhookErrs: []string{
				`metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: "not-semver": must be valid semver`,
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
					ComponentName: "extra",
					Replicas:      k8sptr.To(int32(1)),
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: consts.MainContainerName,
						Env:  []corev1.EnvVar{{Name: "TOKEN", Value: "do-not-leak-this-value"}},
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
						ComponentName: "extra",
						Replicas:      k8sptr.To(int32(1)),
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
			name:          "single-node to multinode transition is immutable",
			oldDeployment: newBetaDGDForValidation(),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			wantWebhookErrs: []string{`spec.components[1].multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`},
		},
		{
			name: "node count-only update remains allowed",
			oldDeployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2}
			}),
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Multinode = &nvidiacomv1beta1.MultinodeSpec{NodeCount: 3}
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
			wantWebhookErrs: []string{`spec.experimental.kvTransferPolicy: Invalid value: {"labelKey":"topology.kubernetes.io/zone","domain":"zone"}: is immutable and cannot be added, removed, or changed after creation; delete and recreate the DynamoGraphDeployment to change the KV transfer policy`},
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
			userInfo: &authenticationv1.UserInfo{
				Username: "system:serviceaccount:default:regular-user",
			},
			operator:        dgdAdmissionOperator,
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
			operator:        dgdAdmissionOperator,
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
			userInfo: &authenticationv1.UserInfo{
				Username: "system:serviceaccount:default:regular-user",
			},
			operator:        dgdAdmissionOperator,
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
			userInfo: &authenticationv1.UserInfo{
				Username: dgdAdmissionOperator,
			},
			operator: dgdAdmissionOperator,
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for name, value := range tt.environment {
				t.Setenv(name, value)
			}
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
			manager := tt.manager
			if manager == nil {
				manager = defaultManager
			}
			handler := NewDynamoGraphDeploymentHandler(manager, tt.operator, !tt.groveDisabled)
			ctx := dgdAdmissionContextWithUserInfo(
				dgdAdmissionOperation(tt.oldDeployment),
				nvidiacomv1beta1.DynamoGraphDeploymentGVK,
				tt.userInfo,
			)

			var (
				warnings []string
				err      error
			)
			if tt.oldDeployment == nil {
				warnings, err = handler.ValidateCreate(ctx, currentBeta)
			} else {
				warnings, err = handler.ValidateUpdate(ctx, oldBeta, currentBeta)
			}
			assertBetaValidationErrors(t, err, tt.wantWebhookErrs)
			if tt.notWantErr != "" && err != nil && strings.Contains(err.Error(), tt.notWantErr) {
				t.Fatalf("webhook error = %q, must not contain %q", err.Error(), tt.notWantErr)
			}
			if !slices.Equal(warnings, tt.wantWarnings) {
				t.Fatalf("warnings = %v, want %v", warnings, tt.wantWarnings)
			}
		})
	}
}

func TestDynamoGraphDeploymentConversionFailureIsFatal(t *testing.T) {
	dgd := newBetaDGDForValidation()
	dgd.Spec.Components = append(dgd.Spec.Components, dgd.Spec.Components[0])

	validator := newDynamoGraphDeploymentTestValidator(t, true)
	_, err := validator.Validate(context.Background(), dgd)
	if err == nil || !strings.Contains(err.Error(), "failed to reconstruct compatibility view") {
		t.Fatalf("Validate() error = %v, want fatal conversion error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("Validate() error = %v, want fatal conversion error rather than field validation error", err)
	}
}

func assertFieldPaths(t *testing.T, errs field.ErrorList, want []string) {
	t.Helper()
	got := make([]string, len(errs))
	for i := range errs {
		got[i] = errs[i].Field
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("field paths = %v, want %v", got, want)
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
	ctrl.Manager  // satisfies the rest of the interface; panics if unexpected methods are used
	client        client.Client
	config        *rest.Config
	scheme        *runtime.Scheme
	webhookServer ctrlwebhook.Server
}

func (m *fakeManager) GetClient() client.Client             { return m.client }
func (m *fakeManager) GetConfig() *rest.Config              { return m.config }
func (m *fakeManager) GetScheme() *runtime.Scheme           { return m.scheme }
func (m *fakeManager) GetWebhookServer() ctrlwebhook.Server { return m.webhookServer }

func newDynamoGraphDeploymentTestValidator(t *testing.T, groveEnabled bool) *DynamoGraphDeploymentValidator {
	t.Helper()
	return NewDynamoGraphDeploymentValidator(newGroveTopologyTestManager(t), groveEnabled)
}

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

func newInferencePoolTestManager(t *testing.T) ctrl.Manager {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		var response any
		switch request.URL.Path {
		case "/api":
			response = &metav1.APIVersions{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "APIVersions"},
				Versions: []string{"v1"},
			}
		case "/apis":
			groupVersion := metav1.GroupVersionForDiscovery{
				GroupVersion: "inference.networking.k8s.io/v1alpha2",
				Version:      "v1alpha2",
			}
			response = &metav1.APIGroupList{
				TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "APIGroupList"},
				Groups: []metav1.APIGroup{{
					Name:             "inference.networking.k8s.io",
					Versions:         []metav1.GroupVersionForDiscovery{groupVersion},
					PreferredVersion: groupVersion,
				}},
			}
		default:
			http.NotFound(w, request)
			return
		}

		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}))
	t.Cleanup(server.Close)

	manager := newGroveTopologyTestManager(t).(*fakeManager)
	manager.config = &rest.Config{Host: server.URL}
	return manager
}

func newTestClusterTopology() *grovev1alpha1.ClusterTopologyBinding {
	return &grovev1alpha1.ClusterTopologyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: "grove-topology"},
		Spec: grovev1alpha1.ClusterTopologyBindingSpec{
			Levels: []grovev1alpha1.TopologyLevel{
				{Domain: grovev1alpha1.TopologyDomainZone, Key: "topology.kubernetes.io/zone"},
				{Domain: grovev1alpha1.TopologyDomainRack, Key: "nvidia.com/rack"},
			},
		},
	}
}

func assertBetaValidationErrors(t *testing.T, err error, wantErrs []string) {
	t.Helper()
	if len(wantErrs) == 0 {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected errors %v but got nil", wantErrs)
	}
	statusErr, ok := err.(*k8serrors.StatusError)
	if !ok || !k8serrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	gotErrs := make([]string, len(causes))
	for i, cause := range causes {
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		gotErrs[i] = fmt.Sprintf("%s: %s", cause.Field, cause.Message)
	}
	if !slices.Equal(gotErrs, wantErrs) {
		t.Fatalf("webhook errors = %v, want %v", gotErrs, wantErrs)
	}
}
