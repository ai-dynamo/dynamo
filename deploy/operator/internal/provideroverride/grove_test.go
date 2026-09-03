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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestComposeGroveOverrides(t *testing.T) {
	t.Log("Build a DGD with overrides at every supported Grove provider context")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: providerOverrideFixture(
				TargetPodCliqueSet,
				`{"spec":{"template":{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"},"futureProviderField":{"enabled":true},"nullableProviderField":null}}}}`,
			),
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "frontend",
					ProviderOverride: providerOverrideFixture(
						TargetPodCliqueTemplateSpec,
						`{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"host"}}}`,
					),
				},
				{
					ComponentName: "worker",
					ProviderOverride: providerOverrideFixture(
						TargetPodCliqueScalingGroupConfig,
						`{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}`,
					),
					Multinode: &nvidiacomv1beta1.MultinodeSpec{
						NodeCount: 2,
						Leader: &nvidiacomv1beta1.MultinodeRoleSpec{ProviderOverride: providerOverrideFixture(
							TargetPodCliqueTemplateSpec,
							`{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"host"}}}`,
						)},
						Worker: &nvidiacomv1beta1.MultinodeRoleSpec{ProviderOverride: providerOverrideFixture(
							TargetPodCliqueTemplateSpec,
							`{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"host"},"newProviderField":"preserved"}}`,
						)},
					},
				},
			},
		},
	}

	t.Log("Render the Grove destinations that receive the sparse fragments")
	desired := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Replicas: 1,
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{Name: "frontend"},
					{Name: "worker-" + consts.GroveRoleSuffixLeader},
					{Name: "worker-" + consts.GroveRoleSuffixWorker},
				},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
					{Name: "worker", CliqueNames: []string{"worker-" + consts.GroveRoleSuffixLeader, "worker-" + consts.GroveRoleSuffixWorker}},
				},
			},
		},
	}

	t.Log("Compose all provider-native fragments into the rendered object")
	got, err := ComposeGroveOverrides(dgd, desired)
	if err != nil {
		t.Fatalf("ComposeGroveOverrides() error = %v", err)
	}
	if got.GetAPIVersion() != GroveAPIVersion || got.GetKind() != TargetPodCliqueSet {
		t.Fatalf("GVK = %s %s, want %s %s", got.GetAPIVersion(), got.GetKind(), GroveAPIVersion, TargetPodCliqueSet)
	}
	if desired.Spec.Template.TopologyConstraint != nil {
		t.Fatal("ComposeGroveOverrides() mutated the rendered input")
	}

	t.Log("Verify each fragment reached only its resolved destination")
	assertNestedValue(t, got.Object, true, "spec", "template", "topologyConstraint", "futureProviderField", "enabled")
	assertNestedValue(t, got.Object, nil, "spec", "template", "topologyConstraint", "nullableProviderField")
	assertNamedNestedValue(t, got.Object, []string{"spec", "template", "cliques"}, "frontend", "host", "topologyConstraint", "pack", "required")
	assertNamedNestedValue(t, got.Object, []string{"spec", "template", "podCliqueScalingGroups"}, "worker", "rack", "topologyConstraint", "pack", "required")
	assertNamedNestedValue(t, got.Object, []string{"spec", "template", "cliques"}, "worker-"+consts.GroveRoleSuffixLeader, "host", "topologyConstraint", "pack", "required")
	assertNamedNestedValue(t, got.Object, []string{"spec", "template", "cliques"}, "worker-"+consts.GroveRoleSuffixWorker, "preserved", "topologyConstraint", "newProviderField")
	assertNestedValue(t, got.Object, int64(1), "spec", "replicas")
}

func TestComposeGroveOverridesRejectsMissingDestination(t *testing.T) {
	t.Log("Build an override whose generated component destination is absent")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "missing",
					ProviderOverride: &nvidiacomv1beta1.ProviderOverride{
						APIVersion: GroveAPIVersion,
						Target:     TargetPodCliqueTemplateSpec,
						Value:      apiextensionsv1.JSON{Raw: []byte(`{"topologyConstraint":{}}`)},
					},
				},
			},
		},
	}

	t.Log("Render a PodCliqueSet containing a different destination")
	desired := &grovev1alpha1.PodCliqueSet{
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "other"}},
			},
		},
	}

	t.Log("Reject the override with the missing generated destination in the error")
	_, err := ComposeGroveOverrides(dgd, desired)
	if err == nil {
		t.Fatal("ComposeGroveOverrides() error = nil, want missing generated destination error")
	}
	if !strings.Contains(err.Error(), `generated destination spec.template.cliques["missing"] was not found`) {
		t.Fatalf("ComposeGroveOverrides() error = %v, want missing generated destination error", err)
	}
}

func TestComposeGroveOverridesRejectsUnsupportedTarget(t *testing.T) {
	t.Log("Build a root override with a target no longer registered by this release")
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: providerOverrideFixture("RetiredPodCliqueSet", `{}`),
		},
	}

	t.Log("Reject the stale target before applying the provider value")
	_, err := ComposeGroveOverrides(dgd, &grovev1alpha1.PodCliqueSet{})
	if err == nil || !strings.Contains(err.Error(), "unsupported Grove target") {
		t.Fatalf("ComposeGroveOverrides() error = %v, want unsupported Grove target", err)
	}
}

func providerOverrideFixture(target, value string) *nvidiacomv1beta1.ProviderOverride {
	return &nvidiacomv1beta1.ProviderOverride{
		APIVersion: GroveAPIVersion,
		Target:     target,
		Value:      apiextensionsv1.JSON{Raw: []byte(value)},
	}
}

func assertNestedValue(t *testing.T, object map[string]interface{}, want interface{}, fields ...string) {
	t.Helper()
	got, found, err := unstructured.NestedFieldNoCopy(object, fields...)
	if err != nil || !found || got != want {
		t.Fatalf("value at %v = (%v, found=%t, err=%v), want %v", fields, got, found, err, want)
	}
}

func assertNamedNestedValue(
	t *testing.T,
	object map[string]interface{},
	listPath []string,
	name string,
	want interface{},
	fields ...string,
) {
	t.Helper()
	items, found, err := unstructured.NestedSlice(object, listPath...)
	if err != nil || !found {
		t.Fatalf("list at %v = (found=%t, err=%v)", listPath, found, err)
	}
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]interface{})
		if !ok || item["name"] != name {
			continue
		}
		assertNestedValue(t, item, want, fields...)
		return
	}
	t.Fatalf("item %q not found at %v", name, listPath)
}
