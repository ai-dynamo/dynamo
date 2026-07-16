/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"slices"
	"strings"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoCheckpointValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamocheckpoints.yaml")

	tests := []struct {
		name          string
		checkpoint    *nvidiacomv1alpha1.DynamoCheckpoint
		oldCheckpoint *nvidiacomv1alpha1.DynamoCheckpoint
		mutateRequest func(*testing.T, map[string]any)
		gmsSnapshot   bool
		wantSchemaErr string
		wantCELErr    string
		wantWebhook   []string
		wantWarnings  []string
	}{
		// Source-version schema and CEL boundaries.
		{
			name:       "valid checkpoint without GMS",
			checkpoint: dynamoCheckpointForAdmission(nil),
		},
		{
			name:       "missing identity is rejected by source schema",
			checkpoint: dynamoCheckpointForAdmission(nil),
			mutateRequest: func(t *testing.T, request map[string]any) {
				t.Helper()
				delete(request["spec"].(map[string]any), "identity")
			},
			wantSchemaErr: "spec.identity: Required value",
		},
		{
			name: "unsupported GMS mode is rejected by source schema",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GPUMemoryServiceMode("unknown"),
				}
			}),
			gmsSnapshot:   true,
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
			gmsSnapshot: true,
			wantCELErr:  "spec.gpuMemoryService: Invalid value: extraClientContainers is only supported with mode=intraPod",
		},

		// Structural create rules.
		{
			name:        "prepared intra-pod GMS checkpoint is accepted",
			checkpoint:  preparedDynamoCheckpointForAdmission(nil),
			gmsSnapshot: true,
		},
		{
			name: "invalid shared metadata annotation uses its exact field path",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "not-semver"}
			}),
			wantWebhook: []string{`metadata.annotations[nvidia.com/dynamo-operator-origin-version]: Invalid value: "not-semver": must be valid semver`},
		},
		{
			name:       "GMS checkpoint feature gate is enforced",
			checkpoint: preparedDynamoCheckpointForAdmission(nil),
			wantWebhook: []string{
				"spec.gpuMemoryService: Forbidden: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
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
			gmsSnapshot: true,
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
			gmsSnapshot: true,
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
			gmsSnapshot: true,
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
			gmsSnapshot: true,
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
			gmsSnapshot: true,
			wantWebhook: []string{
				`spec.job.targetContainerName: Invalid value: "missing": does not name a container in podTemplateSpec.spec.containers`,
			},
		},
		{
			name: "gate and preparation failures aggregate",
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
			wantWebhook: []string{
				"spec.gpuMemoryService: Forbidden: GMS + Snapshot is temporarily disabled; disable gpuMemoryService or enable the internal GMS + Snapshot gate",
				`spec.job.podTemplateSpec.spec.volumes: Required value: must contain the GMS shared volume "gms-intrapod-control"`,
				`spec.job.podTemplateSpec.spec.initContainers: Required value: must contain the GMS init sidecar "gms-server"`,
				`spec.job.podTemplateSpec.spec.containers[0].env: Required value: must contain GMS_SOCKET_DIR=/gms-intrapod-control for GMS`,
				`spec.job.podTemplateSpec.spec.containers[0].resources.claims: Required value: must contain the GMS resource claim "intrapod-shared-gpu"`,
				`spec.job.podTemplateSpec.spec.containers[0].volumeMounts: Required value: must mount volume "gms-intrapod-control" at "/gms-intrapod-control" for GMS`,
				`spec.job.podTemplateSpec.spec.resourceClaims: Required value: must contain the GMS pod resource claim "intrapod-shared-gpu"`,
			},
		},

		// Update, CEL immutability, and deletion behavior.
		{
			name:          "unchanged checkpoint update is accepted",
			oldCheckpoint: preparedDynamoCheckpointForAdmission(nil),
			checkpoint:    preparedDynamoCheckpointForAdmission(nil),
			gmsSnapshot:   true,
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
			name:          "create semantics also apply on update",
			oldCheckpoint: dynamoCheckpointForAdmission(nil),
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
			}),
			gmsSnapshot: true,
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
			name:          "deleting checkpoint skips update validation",
			oldCheckpoint: dynamoCheckpointForAdmission(nil),
			checkpoint: dynamoCheckpointForAdmission(func(checkpoint *nvidiacomv1alpha1.DynamoCheckpoint) {
				checkpoint.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
				checkpoint.DeletionTimestamp = &metav1.Time{Time: time.Unix(1, 0)}
			}),
			gmsSnapshot: true,
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

			handler := NewDynamoCheckpointHandler()
			ctx := dgdAdmissionContext(dynamoCheckpointAdmissionOperation(tt.oldCheckpoint), nvidiacomv1alpha1.GroupVersion.WithKind("DynamoCheckpoint"))
			ctx = features.WithGate(ctx, features.Gates{GMSSnapshot: tt.gmsSnapshot})
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

func TestDynamoCheckpointHandlerBoundaryErrorsRemainRegular(t *testing.T) {
	handler := NewDynamoCheckpointHandler()
	_, err := handler.ValidateCreate(t.Context(), &runtime.Unknown{})
	if err == nil || !strings.Contains(err.Error(), "expected DynamoCheckpoint") {
		t.Fatalf("ValidateCreate() error = %v, want cast error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("ValidateCreate() error = %v, want regular boundary error", err)
	}

	checkpoint := dynamoCheckpointForAdmission(nil)
	_, err = handler.ValidateUpdate(t.Context(), &runtime.Unknown{}, checkpoint)
	if err == nil || !strings.Contains(err.Error(), "expected DynamoCheckpoint") {
		t.Fatalf("ValidateUpdate() error = %v, want old-object cast error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("ValidateUpdate() error = %v, want regular boundary error", err)
	}
}

func dynamoCheckpointForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoCheckpoint),
) *nvidiacomv1alpha1.DynamoCheckpoint {
	checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoCheckpoint",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-checkpoint", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model:            "Qwen/Qwen3-0.6B",
				BackendFramework: "vllm",
			},
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
