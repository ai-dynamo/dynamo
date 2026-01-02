/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoModelValidator_Validate(t *testing.T) {
	tests := []struct {
		name    string
		model   *nvidiacomv1alpha1.DynamoModel
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid base model",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-model",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			wantErr: false,
		},
		{
			name: "valid lora model with s3 source",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "s3://my-bucket/lora-adapter",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid lora model with hf source",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "hf://organization/model-name",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "missing modelName",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-model",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			wantErr: true,
			errMsg:  "spec.modelName is required",
		},
		{
			name: "missing baseModelName",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-model",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "",
					ModelType:     "base",
				},
			},
			wantErr: true,
			errMsg:  "spec.baseModelName is required",
		},
		{
			name: "lora without source",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source:        nil,
				},
			},
			wantErr: true,
			errMsg:  "spec.source is required when modelType is 'lora'",
		},
		{
			name: "lora with empty URI",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "",
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.source.uri must be specified when modelType is 'lora'",
		},
		{
			name: "lora with invalid URI scheme",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "http://example.com/model",
					},
				},
			},
			wantErr: true,
			errMsg:  "source URI must start with 's3://' or 'hf://', got: http://example.com/model",
		},
		{
			name: "lora with file:// URI scheme",
			model: &nvidiacomv1alpha1.DynamoModel{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lora",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "file:///local/path",
					},
				},
			},
			wantErr: true,
			errMsg:  "source URI must start with 's3://' or 'hf://', got: file:///local/path",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoModelValidator(tt.model)
			_, err := validator.Validate()

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoModelValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("DynamoModelValidator.Validate() error message = %v, want %v", err.Error(), tt.errMsg)
			}
		})
	}
}

func TestDynamoModelValidator_ValidateUpdate(t *testing.T) {
	tests := []struct {
		name            string
		oldModel        *nvidiacomv1alpha1.DynamoModel
		newModel        *nvidiacomv1alpha1.DynamoModel
		wantErr         bool
		wantWarnings    bool
		errMsg          string
		expectedWarnMsg string
	}{
		{
			name: "no changes",
			oldModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			newModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			wantErr: false,
		},
		{
			name: "changing modelType",
			oldModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			newModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "s3://bucket/adapter",
					},
				},
			},
			wantErr:         true,
			wantWarnings:    true,
			errMsg:          "spec.modelType is immutable and cannot be changed after creation",
			expectedWarnMsg: "Changing spec.modelType may cause unexpected behavior",
		},
		{
			name: "changing baseModelName",
			oldModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			newModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-70b",
					ModelType:     "base",
				},
			},
			wantErr:         true,
			wantWarnings:    true,
			errMsg:          "spec.baseModelName is immutable and cannot be changed after creation",
			expectedWarnMsg: "Changing spec.baseModelName will break endpoint discovery",
		},
		{
			name: "changing modelName is allowed",
			oldModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			newModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-renamed",
					BaseModelName: "llama-3-8b",
					ModelType:     "base",
				},
			},
			wantErr: false,
		},
		{
			name: "updating source URI for lora is allowed",
			oldModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "s3://bucket/adapter-v1",
					},
				},
			},
			newModel: &nvidiacomv1alpha1.DynamoModel{
				Spec: nvidiacomv1alpha1.DynamoModelSpec{
					ModelName:     "llama-3-8b-custom",
					BaseModelName: "llama-3-8b",
					ModelType:     "lora",
					Source: &nvidiacomv1alpha1.ModelSource{
						URI: "s3://bucket/adapter-v2",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoModelValidator(tt.newModel)
			warnings, err := validator.ValidateUpdate(tt.oldModel)

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoModelValidator.ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("DynamoModelValidator.ValidateUpdate() error message = %v, want %v", err.Error(), tt.errMsg)
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("DynamoModelValidator.ValidateUpdate() expected warnings but got none")
			}

			if tt.wantWarnings && len(warnings) > 0 && warnings[0] != tt.expectedWarnMsg {
				t.Errorf("DynamoModelValidator.ValidateUpdate() warning = %v, want %v", warnings[0], tt.expectedWarnMsg)
			}
		})
	}
}
