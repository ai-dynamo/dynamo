/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoGraphDeploymentRunValidator_Validate(t *testing.T) {
	tests := []struct {
		name  string
		test  admissionTestCase
		check func(*testing.T, *unstructured.Unstructured)
	}{
		// Runs carry resolved snapshots; admission must not default their typed fields.
		{
			name: "resolved snapshot is accepted without Request admission defaulting",
			test: admissionTestCase{object: v1beta2RunForAdmission(v1beta2AdmissionModel, nil)},
		},
		{
			name: "model-only Run is rejected before controller reconciliation",
			test: admissionTestCase{
				object:       v1beta2ModelOnlyRunForAdmission(v1beta2AdmissionModel),
				wantCELError: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
			},
		},
		{
			name: "empty hardware is an accepted resolved opt-out when input overrides supply hardware",
			test: admissionTestCase{object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := schemaRunForAdmission(`{"engine":{"hardware":"h200_sxm"}}`, nil)
				run.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
				return run
			}()},
			check: func(t *testing.T, result *unstructured.Unstructured) {
				hardware, found, err := unstructured.NestedMap(result.Object, "spec", "hardware")
				if err != nil || !found || len(hardware) != 0 {
					t.Fatalf("hardware = %v, found=%v, err=%v; admission must preserve the empty opt-out", hardware, found, err)
				}
			},
		},
		{
			name: "empty static workload remains an opt-out when overrides supply traffic",
			test: admissionTestCase{object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
				run := schemaRunForAdmission(`{"traffic":{"source":{"type":"synthetic","input_tokens":2,"output_tokens":1},"load":{"type":"concurrency","concurrency":1},"stop":{"requests":8}}}`, nil)
				run.Spec.Workload.Static = &nvidiacomv1beta2.StaticWorkloadSpec{}
				return run
			}()},
			check: func(t *testing.T, result *unstructured.Unstructured) {
				static, found, err := unstructured.NestedMap(result.Object, "spec", "workload", "static")
				if err != nil || !found || len(static) != 0 {
					t.Fatalf("static = %v, found=%v, err=%v; admission must preserve the empty opt-out", static, found, err)
				}
			},
		},
		{
			name: "partial hardware is rejected",
			test: admissionTestCase{
				object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
					run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
					run.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{
						SKU: new("h200_sxm"), Budget: new(int32(8)),
					}}
					return run
				}(),
				wantCELError: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
			},
		},
		{
			name: "partial static workload is rejected",
			test: admissionTestCase{
				object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
					run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
					run.Spec.Workload.Static = &nvidiacomv1beta2.StaticWorkloadSpec{InputTokens: new(int32(1024))}
					return run
				}(),
				wantCELError: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
			},
		},
		{
			name: "missing SLA snapshot is rejected",
			test: admissionTestCase{
				object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
					run := v1beta2RunForAdmission(v1beta2AdmissionModel, nil)
					run.Spec.Objective.SLA = nil
					return run
				}(),
				wantCELError: `spec: Invalid value: spec must be a fully resolved Request admission snapshot`,
			},
		},

		// The same lowered-input schema and execution guards apply to Requests and Runs.
		{
			name: "merged input rejects a nonpositive trial budget",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, nil),
				wantWebhookErrors: []string{
					`spec: Invalid value: "<redacted>": generated AISimulate input at /optimizer/max_trials violates JSON Schema keyword exclusiveMinimum`,
				},
			},
		},
		{
			name: "merged input rejects unknown root fields without exposing values",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"unknown_field":"private-value"}`, nil),
				wantWebhookErrors: []string{
					`spec: Invalid value: "<redacted>": generated AISimulate input at / violates JSON Schema keyword additionalProperties: unknown unknown_field`,
				},
				notWantError: "private-value",
			},
		},
		{
			name: "merged input rejects deletion of the required model",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"engine":{"model":null}}`, nil),
				wantWebhookErrors: []string{
					`spec: Invalid value: "<redacted>": generated AISimulate input at /engine violates JSON Schema keyword required: missing model`,
				},
			},
		},
		{
			name: "merged input rejects an invalid nested load union",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"traffic":{"load":{"concurrency":false}}}`, nil),
				wantWebhookErrors: []string{
					`spec: Invalid value: "<redacted>": generated AISimulate input at /traffic violates JSON Schema keyword anyOf; alternatives reject /traffic/load, /traffic/load/concurrency`,
				},
			},
		},
		{
			name: "merged input aggregates independent schema errors",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"engine":{"model":false},"optimizer":{"max_trials":-1}}`, nil),
				wantWebhookErrors: []string{
					`spec: Invalid value: "<redacted>": generated AISimulate input at /engine/model violates JSON Schema keyword type`,
					`spec: Invalid value: "<redacted>": generated AISimulate input at /optimizer/max_trials violates JSON Schema keyword exclusiveMinimum`,
				},
			},
		},
		{
			name: "structural and generated input errors are aggregated",
			test: admissionTestCase{
				object: schemaRunForAdmission(`{"dgdr_run":{"namespace":"other"},"optimizer":{"max_trials":-1}}`, nil),
				wantWebhookErrors: []string{
					"spec.aisimulate.inputOverrides.dgdr_run.namespace: Forbidden: is managed by the operator and cannot be overridden",
					`spec: Invalid value: "<redacted>": generated AISimulate input at /optimizer/max_trials violates JSON Schema keyword exclusiveMinimum`,
				},
			},
		},
		{
			name: "valid merged input may override typed model and load",
			test: admissionTestCase{object: schemaRunForAdmission(`{"engine":{"model":"org/override-model"},"traffic":{"load":{"concurrency":3}}}`, nil)},
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "spec", "model", "name"); got != v1beta2AdmissionModel {
					t.Fatalf("typed model = %q, want %q; overrides must not rewrite the snapshot", got, v1beta2AdmissionModel)
				}
			},
		},

		// A schema upgrade must not block existing Runs' metadata or status writes.
		{
			name: "metadata-only update keeps the immutable spec",
			test: admissionTestCase{
				oldObject: v1beta2RunForAdmission(v1beta2AdmissionModel, nil),
				object:    v1beta2RunForAdmission(v1beta2AdmissionModel, map[string]string{"updated": "true"}),
			},
		},
		{
			name: "unchanged invalid input remains metadata-updatable after a schema upgrade",
			test: admissionTestCase{
				seedWithoutWebhook: true,
				oldObject:          schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, nil),
				object:             schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, map[string]string{"updated": "true"}),
			},
		},
		{
			name: "unchanged invalid input remains status-updatable after a schema upgrade",
			test: admissionTestCase{
				seedWithoutWebhook: true,
				statusOnly:         true,
				oldObject:          schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, nil),
				object: func() *nvidiacomv1beta2.DynamoGraphDeploymentRun {
					run := schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, nil)
					run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseRunning
					return run
				}(),
			},
			check: func(t *testing.T, result *unstructured.Unstructured) {
				if got := v1beta2String(t, result, "status", "phase"); got != string(nvidiacomv1beta2.DGDRRunPhaseRunning) {
					t.Fatalf("status.phase = %q, want Running", got)
				}
			},
		},
		{
			name: "unchanged spec ratchets only upstream schema errors",
			test: admissionTestCase{
				seedWithoutWebhook: true,
				oldObject:          schemaRunForAdmission(`{"dgdr_run":{"namespace":"other"},"optimizer":{"max_trials":-1}}`, nil),
				object:             schemaRunForAdmission(`{"dgdr_run":{"namespace":"other"},"optimizer":{"max_trials":-1}}`, map[string]string{"updated": "true"}),
				wantWebhookErrors: []string{
					"spec.aisimulate.inputOverrides.dgdr_run.namespace: Forbidden: is managed by the operator and cannot be overridden",
				},
			},
		},
		{
			name: "spec update is rejected by the API server",
			test: admissionTestCase{
				oldObject:    v1beta2RunForAdmission(v1beta2AdmissionModel, nil),
				object:       v1beta2RunForAdmission(alternateAdmissionModel, nil),
				wantCELError: `<nil>: Invalid value: spec is immutable`,
			},
		},
		{
			name: "repairing invalid input also requires a new immutable Run",
			test: admissionTestCase{
				seedWithoutWebhook: true,
				oldObject:          schemaRunForAdmission(`{"optimizer":{"max_trials":-1}}`, nil),
				object:             schemaRunForAdmission(`{"optimizer":{"max_trials":2}}`, nil),
				wantCELError:       `<nil>: Invalid value: spec is immutable`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := runAdmissionTest(t, tt.test)
			if actual == nil {
				return
			}
			if actual.GetGeneration() != 1 {
				t.Fatalf("generation = %d, want 1 because no Run spec mutation is allowed", actual.GetGeneration())
			}
			if tt.check != nil {
				tt.check(t, actual)
			}
		})
	}
}

func v1beta2RunForAdmission(model string, labels map[string]string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	metric := nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")
	return &nvidiacomv1beta2.DynamoGraphDeploymentRun{
		APIVersion: nvidiacomv1beta2.GroupVersion.String(),
		Kind:       "DynamoGraphDeploymentRun",
		Name:       "test-v1beta2-dgdr-run", Namespace: "default", Labels: labels,
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRunSpec{NativeSearchInput: nvidiacomv1beta2.NativeSearchInput{
			Model:    &nvidiacomv1beta2.ModelSpec{Name: model},
			Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
			Hardware: &nvidiacomv1beta2.HardwareSpec{
				GPU:            &nvidiacomv1beta2.GPUHardwareSpec{SKU: new("h200_sxm"), Budget: new(int32(8))},
				NumGPUsPerNode: new(int32(8)),
			},
			Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
				InputTokens: new(int32(2)), OutputTokens: new(int32(1)),
				Load: &nvidiacomv1beta2.StaticLoadSpec{Concurrency: new(int32(1))},
			}},
			Objective: &nvidiacomv1beta2.ObjectiveSpec{
				Mode: nvidiacomv1beta2.ObjectiveModeOptimize, Metric: &metric, SLA: &nvidiacomv1beta2.ObjectiveSLASpec{},
			},
			Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
		}},
	}
}

func v1beta2ModelOnlyRunForAdmission(model string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := v1beta2RunForAdmission(model, nil)
	run.Spec.NativeSearchInput = nvidiacomv1beta2.NativeSearchInput{
		Model: &nvidiacomv1beta2.ModelSpec{Name: model},
	}
	return run
}

func schemaRunForAdmission(overrides string, labels map[string]string) *nvidiacomv1beta2.DynamoGraphDeploymentRun {
	run := v1beta2RunForAdmission(v1beta2AdmissionModel, labels)
	run.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{
		InputOverrides: &runtime.RawExtension{Raw: []byte(overrides)},
	}
	return run
}
