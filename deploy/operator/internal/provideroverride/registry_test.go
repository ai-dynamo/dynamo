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

package provideroverride

import (
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

func TestGroveAPIVersionMatchesCompiledSchema(t *testing.T) {
	t.Log("Compare the persisted override version with the compiled Grove types")
	if got := grovev1alpha1.SchemeGroupVersion.String(); got != GroveAPIVersion {
		t.Fatalf("compiled Grove apiVersion = %q, supported override version = %q", got, GroveAPIVersion)
	}
}

func TestExpectedTarget(t *testing.T) {
	t.Log("Build component shapes that exercise each target-resolution branch")
	singleNode := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{}
	multinode := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		Multinode: &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
	}
	forcedScalingGroup := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		Experimental: &nvidiacomv1beta1.ExperimentalSpec{
			Grove: &nvidiacomv1beta1.GroveSpec{ForceScalingGroup: true},
		},
	}
	interPodGMS := &nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		Multinode: &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		Experimental: &nvidiacomv1beta1.ExperimentalSpec{
			GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
				Mode: nvidiacomv1beta1.GMSModeInterPod,
			},
		},
	}

	t.Log("Define successful and rejected provider-context resolutions")
	tests := []struct {
		name      string
		provider  string
		version   string
		scope     Scope
		component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
		want      string
		wantErr   string
	}{
		{name: "root PodCliqueSet", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeRoot, want: TargetPodCliqueSet},
		{name: "single-node component PodClique", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeComponent, component: singleNode, want: TargetPodCliqueTemplateSpec},
		{name: "multinode component scaling group", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeComponent, component: multinode, want: TargetPodCliqueScalingGroupConfig},
		{name: "forced component scaling group", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeComponent, component: forcedScalingGroup, want: TargetPodCliqueScalingGroupConfig},
		{name: "multinode leader PodClique", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeMultinodeLeader, component: multinode, want: TargetPodCliqueTemplateSpec},
		{name: "multinode worker PodClique", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeMultinodeWorker, component: multinode, want: TargetPodCliqueTemplateSpec},
		{name: "component provider unsupported", provider: consts.WorkloadProviderComponent, version: GroveAPIVersion, scope: ScopeRoot, wantErr: "does not support provider overrides"},
		{name: "version unsupported", provider: consts.WorkloadProviderGrove, version: "grove.io/v2", scope: ScopeRoot, wantErr: "unsupported Grove apiVersion"},
		{name: "role requires multinode", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeMultinodeLeader, component: singleNode, wantErr: "requires a multinode component"},
		{name: "GMS roles unsupported", provider: consts.WorkloadProviderGrove, version: GroveAPIVersion, scope: ScopeMultinodeWorker, component: interPodGMS, wantErr: "not supported for inter-pod GMS"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Resolve the registered target for the provider context")
			got, err := ExpectedTarget(tt.provider, tt.version, tt.scope, tt.component)

			t.Log("Verify the target or field-specific resolution failure")
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ExpectedTarget() error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ExpectedTarget() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("ExpectedTarget() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDefaultTarget(t *testing.T) {
	t.Log("Default an omitted root target from its provider context")
	override := &nvidiacomv1beta1.ProviderOverride{APIVersion: GroveAPIVersion}
	DefaultTarget(override, consts.WorkloadProviderGrove, ScopeRoot, nil)
	if override.Target != TargetPodCliqueSet {
		t.Fatalf("target = %q, want %q", override.Target, TargetPodCliqueSet)
	}

	t.Log("Preserve an explicitly selected target")
	override.Target = "explicit"
	DefaultTarget(override, consts.WorkloadProviderGrove, ScopeRoot, nil)
	if override.Target != "explicit" {
		t.Fatalf("explicit target was overwritten with %q", override.Target)
	}
}

func TestValidateValue(t *testing.T) {
	t.Log("Define valid fragments and representative schema or ownership failures")
	tests := []struct {
		name               string
		target             string
		value              string
		wantErr            []string
		ownershipViolation bool
	}{
		{
			name:   "root topology fragment preserves provider fields",
			target: TargetPodCliqueSet,
			value:  `{"spec":{"template":{"topologyConstraint":{"packDomain":"rack","futureProviderField":{"enabled":true}}}}}`,
		},
		{
			name:   "embedded topology fragment",
			target: TargetPodCliqueTemplateSpec,
			value:  `{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"host"}}}`,
		},
		{
			name:   "scaling group topology fragment",
			target: TargetPodCliqueScalingGroupConfig,
			value:  `{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}`,
		},
		{
			name:               "root structural field is rejected",
			target:             TargetPodCliqueSet,
			value:              `{"metadata":{"labels":{"unsafe":"true"}},"spec":{"template":{"topologyConstraint":{}}}}`,
			wantErr:            []string{"metadata"},
			ownershipViolation: true,
		},
		{
			name:               "template structural field is rejected",
			target:             TargetPodCliqueTemplateSpec,
			value:              `{"replicas":3,"topologyConstraint":{}}`,
			wantErr:            []string{"replicas"},
			ownershipViolation: true,
		},
		{
			name:    "topology fragment is required",
			target:  TargetPodCliqueTemplateSpec,
			value:   `{}`,
			wantErr: []string{"topologyConstraint"},
		},
		{
			name:    "value must be an object",
			target:  TargetPodCliqueTemplateSpec,
			value:   `[]`,
			wantErr: []string{"must be a JSON object"},
		},
		{
			name:    "known provider field uses registered type",
			target:  TargetPodCliqueTemplateSpec,
			value:   `{"topologyConstraint":{"pack":"rack"}}`,
			wantErr: []string{"does not match the registered PodCliqueTemplateSpec schema"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Validate the sparse value against its registered schema and ownership boundary")
			errs := ValidateValue(tt.target, []byte(tt.value))

			t.Log("Verify deterministic errors and ownership classification")
			if len(errs) != len(tt.wantErr) {
				t.Fatalf("ValidateValue() errors = %v, want %v", errs, tt.wantErr)
			}
			for i, want := range tt.wantErr {
				if !strings.Contains(errs[i].Error(), want) {
					t.Errorf("error[%d] = %q, want containing %q", i, errs[i].Error(), want)
				}
				if errs[i].OwnershipViolation != tt.ownershipViolation {
					t.Errorf("error[%d].OwnershipViolation = %t, want %t", i, errs[i].OwnershipViolation, tt.ownershipViolation)
				}
			}
		})
	}
}

func TestWritesGroveTopology(t *testing.T) {
	t.Log("Build an embedded Grove topology fragment")
	override := &nvidiacomv1beta1.ProviderOverride{
		Value: apiextensionsv1.JSON{Raw: []byte(`{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}`)},
	}

	t.Log("Detect topology only when the fragment matches the target shape")
	if !WritesGroveTopology(TargetPodCliqueTemplateSpec, override.Value.Raw) {
		t.Fatal("WritesGroveTopology() = false, want true")
	}
	if WritesGroveTopology(TargetPodCliqueSet, override.Value.Raw) {
		t.Fatal("WritesGroveTopology() used an embedded fragment for a root target")
	}
}
