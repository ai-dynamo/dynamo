/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"slices"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoCheckpointValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamocheckpoints.yaml")
	const (
		operatorPrincipal = "system:serviceaccount:dynamo-system:dynamo-operator"
		plannerPrincipal  = "system:serviceaccount:dynamo-system:dynamo-planner"
		ordinaryUser      = "checkpoint-user"
	)

	tests := []struct {
		name               string
		checkpoint         *nvidiacomv1alpha1.DynamoCheckpoint
		oldCheckpoint      *nvidiacomv1alpha1.DynamoCheckpoint
		mutateRequest      func(*testing.T, map[string]any)
		username           string
		operatorPrincipal  string
		omitAdmission      bool
		checkpointDisabled bool
		wantSchemaErr      string
		wantCELErr         string
		wantWebhook        []string
		wantWarnings       []string
	}{
		// Source-version schema and CEL boundaries.
		{
			name:       "valid checkpoint without GMS",
			checkpoint: dynamoCheckpointForAdmission(nil),
		},
		{
			name: "standalone checkpoint requires identity",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.Identity = nil
			}),
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name:              "operator may create identity-free automatic checkpoint",
			checkpoint:        automaticDynamoCheckpointForAdmission(nil),
			username:          operatorPrincipal,
			operatorPrincipal: operatorPrincipal,
		},
		{
			name:              "ordinary user cannot forge automatic checkpoint marker",
			checkpoint:        automaticDynamoCheckpointForAdmission(nil),
			username:          ordinaryUser,
			operatorPrincipal: operatorPrincipal,
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name:              "planner cannot create identity-free automatic checkpoint",
			checkpoint:        automaticDynamoCheckpointForAdmission(nil),
			username:          plannerPrincipal,
			operatorPrincipal: operatorPrincipal,
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name:       "empty operator principal rejects identity omission",
			checkpoint: automaticDynamoCheckpointForAdmission(nil),
			username:   operatorPrincipal,
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name:              "missing admission request rejects identity omission",
			checkpoint:        automaticDynamoCheckpointForAdmission(nil),
			operatorPrincipal: operatorPrincipal,
			omitAdmission:     true,
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name: "operator cannot omit identity without automatic marker",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.Identity = nil
			}),
			username:          operatorPrincipal,
			operatorPrincipal: operatorPrincipal,
			wantWebhook: []string{
				"spec.identity: Required value: is required for a standalone checkpoint",
			},
		},
		{
			name: "unsupported GMS mode is rejected by source schema",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GPUMemoryServiceMode("unknown"),
				}
			}),
			wantSchemaErr: `spec.gpuMemoryService.mode: Unsupported value: "unknown": supported values: "intraPod", "interPod"`,
		},
		{
			name: "inter-pod extra client containers are rejected by source CEL",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               true,
					Mode:                  nvidiacomv1alpha1.GMSModeInterPod,
					ExtraClientContainers: []string{"saver"},
				}
			}),
			wantCELErr: "spec.gpuMemoryService: Invalid value: extraClientContainers is only supported with mode=intraPod",
		},

		// Structural create rules.
		{
			name:       "prepared intra-pod GMS checkpoint is accepted",
			checkpoint: preparedDynamoCheckpointForAdmission(nil),
		},
		{
			name: "DGD-only metadata annotations are ignored",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
			}),
		},
		{
			name:               "checkpoint feature gate is enforced",
			checkpoint:         dynamoCheckpointForAdmission(nil),
			checkpointDisabled: true,
			wantWebhook: []string{
				"spec: Forbidden: checkpoint functionality is disabled in the operator configuration",
			},
		},
		{
			name: "disabled GMS does not require prepared pod wiring",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: false,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
		},
		{
			name: "inter-pod GMS checkpoints are not implemented",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				}
			}),
			wantWebhook: []string{
				`spec.gpuMemoryService.mode: Unsupported value: "interPod": supported values: "intraPod"`,
			},
		},
		{
			name: "unprepared pod wiring aggregates in Kubernetes PodSpec declaration order",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
			wantWebhook: []string{
				`spec.job.podTemplateSpec.spec.volumes: Required value: must contain the GMS shared volume "gms-intrapod-control"`,
				`spec.job.podTemplateSpec.spec.initContainers: Required value: must contain the GMS init sidecar "gms-server"`,
				`spec.job.podTemplateSpec.spec.containers[0].env: Required value: must contain GMS_SOCKET_DIR=/gms-intrapod-control for GMS`,
				`spec.job.podTemplateSpec.spec.containers[0].resources.claims: Required value: must contain the GMS resource claim "intrapod-shared-gpu"`,
				`spec.job.podTemplateSpec.spec.containers[0].volumeMounts: Required value: must mount volume "gms-intrapod-control" at "/gms-intrapod-control" for GMS`,
				`spec.job.podTemplateSpec.spec.resourceClaims: Required value: must contain the GMS pod resource claim "intrapod-shared-gpu"`,
			},
		},
		{
			name: "missing extra client container has its source-list path",
			checkpoint: preparedDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService.ExtraClientContainers = []string{"saver"}
			}),
			wantWebhook: []string{
				`spec.gpuMemoryService.extraClientContainers[0]: Invalid value: "saver": does not name a container in spec.job.podTemplateSpec.spec.containers`,
			},
		},
		{
			name: "extra client wiring failures use the exact container index",
			checkpoint: preparedDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService.ExtraClientContainers = []string{"saver"}
				checkpoint.Spec.Job.PodTemplateSpec.Spec.Containers = append(
					checkpoint.Spec.Job.PodTemplateSpec.Spec.Containers,
					corev1.Container{Name: "saver"},
				)
			}),
			wantWebhook: []string{
				`spec.job.podTemplateSpec.spec.containers[1].env: Required value: must contain GMS_SOCKET_DIR=/gms-intrapod-control for GMS`,
				`spec.job.podTemplateSpec.spec.containers[1].resources.claims: Required value: must contain the GMS resource claim "intrapod-shared-gpu"`,
				`spec.job.podTemplateSpec.spec.containers[1].volumeMounts: Required value: must mount volume "gms-intrapod-control" at "/gms-intrapod-control" for GMS`,
			},
		},
		{
			name: "explicit target container reference must resolve",
			checkpoint: preparedDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.Job.TargetContainerName = "missing"
			}),
			wantWebhook: []string{
				`spec.job.targetContainerName: Invalid value: "missing": does not name a container in podTemplateSpec.spec.containers`,
			},
		},
		// Update, CEL immutability, and deletion behavior.
		{
			name:          "unchanged checkpoint update is accepted",
			oldCheckpoint: preparedDynamoCheckpointForAdmission(nil),
			checkpoint:    preparedDynamoCheckpointForAdmission(nil),
		},
		{
			name:              "operator may update identity-free automatic checkpoint",
			oldCheckpoint:     automaticDynamoCheckpointForAdmission(nil),
			checkpoint:        automaticDynamoCheckpointForAdmission(nil),
			username:          operatorPrincipal,
			operatorPrincipal: operatorPrincipal,
		},
		{
			name: "legacy automatic identity is preserved on update",
			oldCheckpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				checkpoint.Spec.Identity = &identity
			}),
			checkpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				checkpoint.Spec.Identity = &identity
			}),
		},
		{
			name: "legacy automatic identity cannot be changed",
			oldCheckpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				checkpoint.Spec.Identity = &identity
			}),
			checkpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				identity.Model = alternateAdmissionModel
				checkpoint.Spec.Identity = &identity
			}),
			wantCELErr: "<nil>: Invalid value: spec.identity is immutable after creation",
		},
		{
			name: "legacy automatic identity cannot be removed",
			oldCheckpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				checkpoint.Spec.Identity = &identity
			}),
			checkpoint: automaticDynamoCheckpointForAdmission(nil),
			wantCELErr: "<nil>: Invalid value: spec.identity is immutable after creation",
		},
		{
			name:          "identity cannot be added to identity-free checkpoint",
			oldCheckpoint: automaticDynamoCheckpointForAdmission(nil),
			checkpoint: automaticDynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				identity := admissionCheckpointIdentity()
				checkpoint.Spec.Identity = &identity
			}),
			wantCELErr: "<nil>: Invalid value: spec.identity is immutable after creation",
		},
		{
			name:               "checkpoint feature gate applies on update",
			oldCheckpoint:      dynamoCheckpointForAdmission(nil),
			checkpoint:         dynamoCheckpointForAdmission(nil),
			checkpointDisabled: true,
			wantWebhook: []string{
				"spec: Forbidden: checkpoint functionality is disabled in the operator configuration",
			},
		},
		{
			name:          "identity immutability is enforced by source CEL",
			oldCheckpoint: dynamoCheckpointForAdmission(nil),
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.Identity.Model = alternateAdmissionModel
			}),
			wantCELErr: "<nil>: Invalid value: spec.identity is immutable after creation",
		},
		{
			name:          "identity removal is rejected by source CEL",
			oldCheckpoint: dynamoCheckpointForAdmission(nil),
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.Identity = nil
			}),
			wantCELErr: "<nil>: Invalid value: spec.identity is immutable after creation",
		},
		{
			name:          "create semantics also apply on update",
			oldCheckpoint: dynamoCheckpointForAdmission(nil),
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
			wantWebhook: []string{
				`spec.job.podTemplateSpec.spec.volumes: Required value: must contain the GMS shared volume "gms-intrapod-control"`,
				`spec.job.podTemplateSpec.spec.initContainers: Required value: must contain the GMS init sidecar "gms-server"`,
				`spec.job.podTemplateSpec.spec.containers[0].env: Required value: must contain GMS_SOCKET_DIR=/gms-intrapod-control for GMS`,
				`spec.job.podTemplateSpec.spec.containers[0].resources.claims: Required value: must contain the GMS resource claim "intrapod-shared-gpu"`,
				`spec.job.podTemplateSpec.spec.containers[0].volumeMounts: Required value: must mount volume "gms-intrapod-control" at "/gms-intrapod-control" for GMS`,
				`spec.job.podTemplateSpec.spec.resourceClaims: Required value: must contain the GMS pod resource claim "intrapod-shared-gpu"`,
			},
		},
		{
			name:               "deleting checkpoint skips update validation",
			oldCheckpoint:      dynamoCheckpointForAdmission(nil),
			checkpointDisabled: true,
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
				checkpoint.DeletionTimestamp = &metav1.Time{Time: time.Unix(1, 0)}
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			current := admissionUnstructured(t, tt.checkpoint)
			if tt.mutateRequest != nil {
				tt.mutateRequest(t, current)
			}
			var old map[string]any
			if tt.oldCheckpoint != nil {
				old = admissionUnstructured(t, tt.oldCheckpoint)
			}

			version := admissionSourceVersion(t, tt.checkpoint)
			requestValidator, ok := requestValidators[version]
			if !ok {
				t.Fatalf("no request validator for source version %q", version)
			}
			schemaErrs := requestValidator.validateSchema(current, old)
			if tt.wantSchemaErr != "" {
				if tt.wantCELErr != "" || len(tt.wantWebhook) != 0 || len(tt.wantWarnings) != 0 {
					t.Fatal("schema rejection cannot have downstream expectations")
				}
				assertRequestValidationError(t, schemaErrs, tt.wantSchemaErr)
				return
			}
			if len(schemaErrs) != 0 {
				t.Fatalf("schema errors = %v, want none", schemaErrs)
			}

			celErrs := requestValidator.celValidator(current, old)
			if tt.wantCELErr != "" {
				if len(tt.wantWebhook) != 0 || len(tt.wantWarnings) != 0 {
					t.Fatal("CEL rejection cannot have webhook expectations")
				}
				assertRequestValidationError(t, celErrs, tt.wantCELErr)
				return
			}
			if len(celErrs) != 0 {
				t.Fatalf("CEL errors = %v, want none", celErrs)
			}

			handler := NewDynamoCheckpointHandler(tt.operatorPrincipal)
			ctx := t.Context()
			if !tt.omitAdmission {
				ctx = dgdAdmissionContextWithUserInfo(
					dynamoCheckpointAdmissionOperation(tt.oldCheckpoint),
					nvidiacomv1alpha1.GroupVersion.WithKind("DynamoCheckpoint"),
					&authenticationv1.UserInfo{Username: tt.username},
				)
			}
			ctx = features.WithGate(ctx, features.Gates{
				Checkpoint: !tt.checkpointDisabled,
			})
			var warnings []string
			var err error
			if tt.oldCheckpoint == nil {
				warnings, err = handler.ValidateCreate(ctx, tt.checkpoint.DeepCopy())
			} else {
				warnings, err = handler.ValidateUpdate(ctx, tt.oldCheckpoint.DeepCopy(), tt.checkpoint.DeepCopy())
			}
			assertWebhookErrors(t, err, tt.wantWebhook)
			if !slices.Equal(warnings, tt.wantWarnings) {
				t.Fatalf("webhook warnings = %v, want %v", warnings, tt.wantWarnings)
			}
		})
	}
}

func dynamoCheckpointForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoCheckpoint),
) *nvidiacomv1alpha1.DynamoCheckpoint {
	identity := admissionCheckpointIdentity()
	checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoCheckpoint",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-checkpoint", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: &identity,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName}},
					},
				},
				TargetContainerName: consts.MainContainerName,
			},
		},
	}
	if mutate != nil {
		mutate(checkpoint)
	}
	return checkpoint
}

func admissionCheckpointIdentity() nvidiacomv1alpha1.DynamoCheckpointIdentity {
	return nvidiacomv1alpha1.DynamoCheckpointIdentity{
		Model:            "Qwen/Qwen3-0.6B",
		BackendFramework: "vllm",
	}
}

func automaticDynamoCheckpointForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoCheckpoint),
) *nvidiacomv1alpha1.DynamoCheckpoint {
	checkpoint := dynamoCheckpointForAdmission(nil)
	checkpoint.Annotations = map[string]string{
		consts.CheckpointAutoAnnotation: consts.KubeLabelValueTrue,
	}
	checkpoint.Labels = map[string]string{
		snapshotprotocol.CheckpointIDLabel: "checkpoint-id",
	}
	checkpoint.Spec.Identity = nil
	if mutate != nil {
		mutate(checkpoint)
	}
	return checkpoint
}

func preparedDynamoCheckpointForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoCheckpoint),
) *nvidiacomv1alpha1.DynamoCheckpoint {
	claimTemplateName := "checkpoint-test-worker-gpu"
	clientContainer := func(name string) corev1.Container {
		return corev1.Container{
			Name: name,
			Env: []corev1.EnvVar{
				{Name: gms.EnvSocketDir, Value: gms.SharedMountPath},
			},
			VolumeMounts: []corev1.VolumeMount{
				{Name: gms.SharedVolumeName, MountPath: gms.SharedMountPath},
			},
			Resources: corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
			},
		}
	}

	return dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
		checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
			Enabled: true,
			Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
		}
		checkpoint.Spec.Job.PodTemplateSpec.Spec = corev1.PodSpec{
			ResourceClaims: []corev1.PodResourceClaim{{
				Name:                      dra.ClaimName,
				ResourceClaimTemplateName: &claimTemplateName,
			}},
			Volumes: []corev1.Volume{{
				Name: gms.SharedVolumeName,
				VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{},
				},
			}},
			InitContainers: []corev1.Container{clientContainer(gms.ServerContainerName)},
			Containers:     []corev1.Container{clientContainer(consts.MainContainerName)},
		}
		if mutate != nil {
			mutate(checkpoint)
		}
	})
}

func dynamoCheckpointAdmissionOperation(
	oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint,
) admissionv1.Operation {
	if oldCheckpoint == nil {
		return admissionv1.Create
	}
	return admissionv1.Update
}
