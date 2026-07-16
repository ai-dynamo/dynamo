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
	"slices"
	"strings"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	admissionv1 "k8s.io/api/admission/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoModelValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamomodels.yaml")

	tests := []struct {
		name          string
		model         *nvidiacomv1alpha1.DynamoModel
		oldModel      *nvidiacomv1alpha1.DynamoModel
		mutateRequest func(*testing.T, map[string]any)
		wantSchemaErr string
		wantCELErr    string
		wantWebhook   []string
		wantWarnings  []string
	}{
		// Source-version schema and CEL boundaries.
		{
			name:  "valid base model",
			model: dynamoModelForAdmission(nil),
		},
		{
			name:  "missing modelName is rejected by source schema",
			model: dynamoModelForAdmission(nil),
			mutateRequest: func(t *testing.T, request map[string]any) {
				t.Helper()
				delete(request["spec"].(map[string]any), "modelName")
			},
			wantSchemaErr: "spec.modelName: Required value",
		},
		{
			name: "unsupported modelType is rejected by source schema",
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.ModelType = "unknown"
			}),
			wantSchemaErr: `spec.modelType: Unsupported value: "unknown": supported values: "base", "lora", "adapter"`,
		},

		// Structural create rules and accepted source schemes.
		{
			name:  "valid LoRA model with S3 source",
			model: loraModelForAdmission("s3://my-bucket/lora-adapter"),
		},
		{
			name:  "valid LoRA model with Hugging Face source",
			model: loraModelForAdmission("hf://organization/model-name"),
		},
		{
			name:  "valid LoRA model with local source",
			model: loraModelForAdmission("file:///local/path"),
		},
		{
			name: "empty required names aggregate in API declaration order",
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.ModelName = ""
				model.Spec.BaseModelName = ""
			}),
			wantWebhook: []string{
				"spec.modelName: Required value: must not be empty",
				"spec.baseModelName: Required value: must not be empty",
			},
		},
		{
			name: "LoRA model requires source",
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.ModelType = "lora"
			}),
			wantWebhook: []string{`spec.source: Required value: is required when spec.modelType is "lora"`},
		},
		{
			name:  "LoRA model requires non-empty source URI",
			model: loraModelForAdmission(""),
			wantWebhook: []string{
				`spec.source.uri: Required value: must be specified when spec.modelType is "lora"`,
			},
		},
		{
			name:  "LoRA model rejects HTTP source",
			model: loraModelForAdmission("http://example.com/model"),
			wantWebhook: []string{
				`spec.source.uri: Invalid value: "http://example.com/model": must start with "s3://", "hf://", or "file:///"`,
			},
		},
		{
			name:  "LoRA model rejects file URI with host",
			model: loraModelForAdmission("file://host/local/path"),
			wantWebhook: []string{
				`spec.source.uri: Invalid value: "file://host/local/path": must start with "s3://", "hf://", or "file:///"`,
			},
		},
		{
			name:  "LoRA model rejects bare file URI",
			model: loraModelForAdmission("file://"),
			wantWebhook: []string{
				`spec.source.uri: Invalid value: "file://": must start with "s3://", "hf://", or "file:///"`,
			},
		},
		{
			name: "non-LoRA model ignores source semantics for compatibility",
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.Source = &nvidiacomv1alpha1.ModelSource{URI: "http://example.com/model"}
			}),
		},

		// Structural update rules and warnings.
		{
			name:     "unchanged model is accepted",
			oldModel: dynamoModelForAdmission(nil),
			model:    dynamoModelForAdmission(nil),
		},
		{
			name:     "modelName may change",
			oldModel: dynamoModelForAdmission(nil),
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.ModelName = "Qwen/Qwen3-0.6B-renamed"
			}),
		},
		{
			name:     "baseModelName is immutable and warns",
			oldModel: dynamoModelForAdmission(nil),
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.BaseModelName = "Qwen/Qwen3-8B"
			}),
			wantWebhook: []string{
				`spec.baseModelName: Invalid value: "Qwen/Qwen3-8B": is immutable and cannot be changed after creation`,
			},
			wantWarnings: []string{"Changing spec.baseModelName will break endpoint discovery"},
		},
		{
			name:     "modelType is immutable and warns",
			oldModel: dynamoModelForAdmission(nil),
			model:    loraModelForAdmission("s3://bucket/adapter"),
			wantWebhook: []string{
				`spec.modelType: Invalid value: "lora": is immutable and cannot be changed after creation`,
			},
			wantWarnings: []string{"Changing spec.modelType may cause unexpected behavior"},
		},
		{
			name:     "independent immutable fields aggregate and warn in API declaration order",
			oldModel: dynamoModelForAdmission(nil),
			model: loraModelForAdmission("s3://bucket/adapter", func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.BaseModelName = "Qwen/Qwen3-8B"
			}),
			wantWebhook: []string{
				`spec.baseModelName: Invalid value: "Qwen/Qwen3-8B": is immutable and cannot be changed after creation`,
				`spec.modelType: Invalid value: "lora": is immutable and cannot be changed after creation`,
			},
			wantWarnings: []string{
				"Changing spec.baseModelName will break endpoint discovery",
				"Changing spec.modelType may cause unexpected behavior",
			},
		},
		{
			name:     "LoRA source URI may change",
			oldModel: loraModelForAdmission("s3://bucket/adapter-v1"),
			model:    loraModelForAdmission("s3://bucket/adapter-v2"),
		},
		{
			name:     "create semantics also apply on update",
			oldModel: loraModelForAdmission("s3://bucket/adapter-v1"),
			model:    loraModelForAdmission("http://example.com/adapter"),
			wantWebhook: []string{
				`spec.source.uri: Invalid value: "http://example.com/adapter": must start with "s3://", "hf://", or "file:///"`,
			},
		},
		{
			name:     "deleting model skips update validation",
			oldModel: dynamoModelForAdmission(nil),
			model: dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
				model.Spec.ModelType = "lora"
				model.DeletionTimestamp = &metav1.Time{Time: time.Unix(1, 0)}
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			current := admissionUnstructured(t, tt.model)
			if tt.mutateRequest != nil {
				tt.mutateRequest(t, current)
			}
			var old map[string]any
			if tt.oldModel != nil {
				old = admissionUnstructured(t, tt.oldModel)
			}

			version := admissionSourceVersion(t, tt.model)
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

			handler := NewDynamoModelHandler()
			ctx := dgdAdmissionContext(dynamoModelAdmissionOperation(tt.oldModel), nvidiacomv1alpha1.GroupVersion.WithKind("DynamoModel"))
			var warnings []string
			var err error
			if tt.oldModel == nil {
				warnings, err = handler.ValidateCreate(ctx, tt.model.DeepCopy())
			} else {
				warnings, err = handler.ValidateUpdate(ctx, tt.oldModel.DeepCopy(), tt.model.DeepCopy())
			}
			assertWebhookErrors(t, err, tt.wantWebhook)
			if !slices.Equal(warnings, tt.wantWarnings) {
				t.Fatalf("webhook warnings = %v, want %v", warnings, tt.wantWarnings)
			}
		})
	}
}

func TestDynamoModelHandlerBoundaryErrorsRemainRegular(t *testing.T) {
	handler := NewDynamoModelHandler()
	_, err := handler.ValidateCreate(t.Context(), &runtime.Unknown{})
	if err == nil || !strings.Contains(err.Error(), "expected DynamoModel") {
		t.Fatalf("ValidateCreate() error = %v, want cast error", err)
	}
	if k8serrors.IsInvalid(err) {
		t.Fatalf("ValidateCreate() error = %v, want regular boundary error", err)
	}
}

func dynamoModelForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoModel),
) *nvidiacomv1alpha1.DynamoModel {
	model := &nvidiacomv1alpha1.DynamoModel{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoModel",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "test-model", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoModelSpec{
			ModelName:     "Qwen/Qwen3-0.6B",
			BaseModelName: "Qwen/Qwen3-0.6B",
			ModelType:     "base",
		},
	}
	if mutate != nil {
		mutate(model)
	}
	return model
}

func loraModelForAdmission(
	uri string,
	mutations ...func(*nvidiacomv1alpha1.DynamoModel),
) *nvidiacomv1alpha1.DynamoModel {
	return dynamoModelForAdmission(func(model *nvidiacomv1alpha1.DynamoModel) {
		model.Spec.ModelType = "lora"
		model.Spec.Source = &nvidiacomv1alpha1.ModelSource{URI: uri}
		for _, mutate := range mutations {
			mutate(model)
		}
	})
}

func dynamoModelAdmissionOperation(oldModel *nvidiacomv1alpha1.DynamoModel) admissionv1.Operation {
	if oldModel == nil {
		return admissionv1.Create
	}
	return admissionv1.Update
}
