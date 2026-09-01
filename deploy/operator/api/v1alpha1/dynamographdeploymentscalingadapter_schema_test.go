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

package v1alpha1

import (
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	apitest "k8s.io/apiextensions-apiserver/pkg/test"
)

func TestDGDSAScaleTargetSchemaRequiresExactlyOneTarget(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed")
	}
	crdPath := filepath.Join(filepath.Dir(thisFile), "../../config/crd/bases/nvidia.com_dynamographdeploymentscalingadapters.yaml")
	validators := apitest.VersionValidatorsFromFile(t, crdPath)

	tests := []struct {
		name          string
		version       string
		componentKey  string
		componentSet  bool
		groupSet      bool
		wantRejection bool
	}{
		{name: "v1alpha1 service", version: "v1alpha1", componentKey: "serviceName", componentSet: true},
		{name: "v1alpha1 component group", version: "v1alpha1", componentKey: "serviceName", groupSet: true},
		{name: "v1alpha1 neither", version: "v1alpha1", componentKey: "serviceName", wantRejection: true},
		{name: "v1alpha1 both", version: "v1alpha1", componentKey: "serviceName", componentSet: true, groupSet: true, wantRejection: true},
		{name: "v1beta1 component", version: "v1beta1", componentKey: "componentName", componentSet: true},
		{name: "v1beta1 component group", version: "v1beta1", componentKey: "componentName", groupSet: true},
		{name: "v1beta1 neither", version: "v1beta1", componentKey: "componentName", wantRejection: true},
		{name: "v1beta1 both", version: "v1beta1", componentKey: "componentName", componentSet: true, groupSet: true, wantRejection: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build a scaling adapter with the requested target and replica count")
			dgdRef := map[string]any{"name": "graph"}
			if test.componentSet {
				dgdRef[test.componentKey] = "worker"
			}
			if test.groupSet {
				dgdRef["componentGroupName"] = "workers"
			}
			obj := map[string]any{
				"apiVersion": "nvidia.com/" + test.version,
				"kind":       "DynamoGraphDeploymentScalingAdapter",
				"metadata":   map[string]any{"name": "adapter"},
				"spec":       map[string]any{"replicas": int64(1), "dgdRef": dgdRef},
			}

			t.Log("Run the generated schema and collect the target-specific CEL errors")
			errs := validators[test.version](obj, nil)
			gotRejection := false
			for _, err := range errs {
				if strings.Contains(err.Error(), "exactly one of "+test.componentKey+" or componentGroupName must be set") {
					gotRejection = true
					break
				}
			}

			t.Log("Verify the expected rejection and require valid cases to have no schema errors")
			if gotRejection != test.wantRejection {
				t.Fatalf("CEL target rejection = %t, want %t; errors: %v", gotRejection, test.wantRejection, errs)
			}
		})
	}
}
