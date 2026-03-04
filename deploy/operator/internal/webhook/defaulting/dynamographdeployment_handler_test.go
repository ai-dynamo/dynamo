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

package defaulting

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// admissionCtx builds a context carrying an admission request for the given operation.
func admissionCtx(op admissionv1.Operation) context.Context {
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
		},
	})
}

func TestDGDDefaulter_Default(t *testing.T) {
	const testVersion = "0.8.0"

	tests := []struct {
		name            string
		operatorVersion string
		ctx             context.Context
		dgd             *nvidiacomv1alpha1.DynamoGraphDeployment
		wantAnnotation  string
		wantErr         bool
	}{
		{
			name:            "CREATE stamps operator version on new DGD without annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE stamps operator version on DGD with existing annotations",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						"some-other-annotation": "some-value",
					},
				},
			},
			wantAnnotation: testVersion,
		},
		{
			name:            "CREATE does not overwrite pre-existing origin version",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Create),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "UPDATE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "UPDATE preserves existing annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Update),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.7.0",
					},
				},
			},
			wantAnnotation: "0.7.0",
		},
		{
			name:            "DELETE does not stamp annotation",
			operatorVersion: testVersion,
			ctx:             admissionCtx(admissionv1.Delete),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
		{
			name:            "no admission request in context skips defaulting gracefully",
			operatorVersion: testVersion,
			ctx:             context.Background(),
			dgd: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dgd",
					Namespace: "default",
				},
			},
			wantAnnotation: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDGDDefaulter(tt.operatorVersion)

			err := defaulter.Default(tt.ctx, tt.dgd)
			if (err != nil) != tt.wantErr {
				t.Errorf("Default() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			got := ""
			if tt.dgd.Annotations != nil {
				got = tt.dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion]
			}

			if got != tt.wantAnnotation {
				t.Errorf("annotation %q = %q, want %q",
					consts.KubeAnnotationDynamoOperatorOriginVersion, got, tt.wantAnnotation)
			}
		})
	}
}
