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
	"fmt"
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	sigsjson "sigs.k8s.io/json"
)

// HasGroveTopologyOverrides reports whether any DGD provider context writes a
// Grove topology constraint. dgd must not be nil.
func HasGroveTopologyOverrides(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	// Check the root before traversing component and role contexts.
	if overrideWritesGroveTopology(dgd.Spec.ProviderOverride) {
		return true
	}

	// Stop as soon as one component provider context writes topology.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if overrideWritesGroveTopology(component.ProviderOverride) {
			return true
		}
		if component.Multinode == nil {
			continue
		}
		if component.Multinode.Leader != nil &&
			overrideWritesGroveTopology(component.Multinode.Leader.ProviderOverride) {
			return true
		}
		if component.Multinode.Worker != nil &&
			overrideWritesGroveTopology(component.Multinode.Worker.ProviderOverride) {
			return true
		}
	}
	return false
}

func overrideWritesGroveTopology(override *nvidiacomv1beta1.ProviderOverride) bool {
	return override != nil && WritesGroveTopology(override.Target, override.Value.Raw)
}

// ComposeGroveOverrides converts a fully rendered PodCliqueSet to unstructured
// form and inserts each provider-owned subtree at its resolved destination.
// Keeping the result unstructured preserves provider fields that are newer
// than the Grove Go types compiled into Dynamo. dgd and desired must not be
// nil.
func ComposeGroveOverrides(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *grovev1alpha1.PodCliqueSet,
) (*unstructured.Unstructured, error) {
	// Preserve the rendered object as unstructured data before applying fragments.
	object, err := runtime.DefaultUnstructuredConverter.ToUnstructured(desired)
	if err != nil {
		return nil, fmt.Errorf("convert rendered PodCliqueSet to unstructured: %w", err)
	}
	result := &unstructured.Unstructured{Object: object}
	result.SetAPIVersion(GroveAPIVersion)
	result.SetKind(TargetPodCliqueSet)

	// Apply the root fragment before the more specific component destinations.
	if err := applyGroveRootOverride(result, dgd.Spec.ProviderOverride); err != nil {
		return nil, fmt.Errorf("spec.providerOverride: %w", err)
	}

	// Resolve each component and role fragment to its generated Grove destination.
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		componentPath := fmt.Sprintf("spec.components[%d]", i)
		if err := applyGroveComponentOverride(result, component, component.ProviderOverride); err != nil {
			return nil, fmt.Errorf("%s.providerOverride: %w", componentPath, err)
		}
		if component.Multinode == nil {
			continue
		}
		if component.Multinode.Leader != nil {
			if err := applyGroveRoleOverride(
				result,
				component,
				ScopeMultinodeLeader,
				component.Multinode.Leader.ProviderOverride,
			); err != nil {
				return nil, fmt.Errorf("%s.multinode.leader.providerOverride: %w", componentPath, err)
			}
		}
		if component.Multinode.Worker != nil {
			if err := applyGroveRoleOverride(
				result,
				component,
				ScopeMultinodeWorker,
				component.Multinode.Worker.ProviderOverride,
			); err != nil {
				return nil, fmt.Errorf("%s.multinode.worker.providerOverride: %w", componentPath, err)
			}
		}
	}
	return result, nil
}

// applyGroveRootOverride applies an optional root fragment. result must not be
// nil; a nil override is an intentional no-op.
func applyGroveRootOverride(result *unstructured.Unstructured, override *nvidiacomv1beta1.ProviderOverride) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted identity before inserting the root-owned subtree.
	if err := validateOverrideIdentity(override, ScopeRoot, nil); err != nil {
		return err
	}

	// Replace the registered opaque subtree without defining merge or null semantics.
	topologyConstraint, err := groveTopologyConstraint(override)
	if err != nil {
		return err
	}
	return unstructured.SetNestedField(
		result.Object,
		topologyConstraint,
		"spec",
		"template",
		"topologyConstraint",
	)
}

// applyGroveComponentOverride applies an optional component fragment. result
// and component must not be nil; a nil override is an intentional no-op.
func applyGroveComponentOverride(
	result *unstructured.Unstructured,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	override *nvidiacomv1beta1.ProviderOverride,
) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted identity before selecting its generated list.
	if err := validateOverrideIdentity(override, ScopeComponent, component); err != nil {
		return err
	}

	// Route the fragment to the list implied by its embedded Grove target.
	name := strings.ToLower(component.ComponentName)
	switch override.Target {
	case TargetPodCliqueTemplateSpec:
		return setNamedGroveTopologyConstraint(
			result,
			[]string{"spec", "template", "cliques"},
			name,
			override,
		)
	case TargetPodCliqueScalingGroupConfig:
		return setNamedGroveTopologyConstraint(
			result,
			[]string{"spec", "template", "podCliqueScalingGroups"},
			name,
			override,
		)
	default:
		return fmt.Errorf("unsupported target %q", override.Target)
	}
}

// applyGroveRoleOverride applies an optional multinode role fragment. result
// and component must not be nil; a nil override is an intentional no-op.
func applyGroveRoleOverride(
	result *unstructured.Unstructured,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	scope Scope,
	override *nvidiacomv1beta1.ProviderOverride,
) error {
	// A missing fragment is an intentional no-op for this provider context.
	if override == nil {
		return nil
	}

	// Verify the persisted role identity before resolving its generated name.
	if err := validateOverrideIdentity(override, scope, component); err != nil {
		return err
	}
	suffix := consts.GroveRoleSuffixLeader
	if scope == ScopeMultinodeWorker {
		suffix = consts.GroveRoleSuffixWorker
	}

	// Insert the PCLQ topology subtree named for the selected multinode role.
	return setNamedGroveTopologyConstraint(
		result,
		[]string{"spec", "template", "cliques"},
		strings.ToLower(component.ComponentName+"-"+suffix),
		override,
	)
}

// validateOverrideIdentity checks one persisted provider context. override
// must not be nil; component may be nil only for ScopeRoot.
func validateOverrideIdentity(
	override *nvidiacomv1beta1.ProviderOverride,
	scope Scope,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) error {
	// Re-resolve the registered target and reject stale or forged identities.
	expected, err := ExpectedTarget(consts.WorkloadProviderGrove, override.APIVersion, scope, component)
	if err != nil {
		return err
	}
	if override.Target != expected {
		return fmt.Errorf("unsupported Grove target %q; resolved target is %q", override.Target, expected)
	}

	// Recheck value ownership before the controller mutates provider resources.
	if valueErrs := ValidateValue(override.Target, override.Value.Raw); len(valueErrs) != 0 {
		return fmt.Errorf("value is invalid: %s", valueErrs[0].Error())
	}
	return nil
}

// setNamedGroveTopologyConstraint sets the registered opaque subtree on one
// named embedded target. result and override must not be nil.
func setNamedGroveTopologyConstraint(
	result *unstructured.Unstructured,
	path []string,
	name string,
	override *nvidiacomv1beta1.ProviderOverride,
) error {
	// Read the generated list without assuming the destination exists.
	items, found, err := unstructured.NestedSlice(result.Object, path...)
	if err != nil {
		return fmt.Errorf("read %s: %w", strings.Join(path, "."), err)
	}
	if !found {
		return fmt.Errorf("generated destination %s[%q] was not found", strings.Join(path, "."), name)
	}

	// Decode the raw subtree once before locating its generated destination.
	topologyConstraint, err := groveTopologyConstraint(override)
	if err != nil {
		return err
	}

	// Set only the generated entry whose stable name matches the DGD context.
	for i := range items {
		item, ok := items[i].(map[string]interface{})
		if !ok || item["name"] != name {
			continue
		}
		item["topologyConstraint"] = topologyConstraint
		items[i] = item
		return unstructured.SetNestedSlice(result.Object, items, path...)
	}
	return fmt.Errorf("generated destination %s[%q] was not found", strings.Join(path, "."), name)
}

// groveTopologyConstraint extracts the raw opaque subtree registered for one
// Grove target. override must not be nil.
func groveTopologyConstraint(override *nvidiacomv1beta1.ProviderOverride) (interface{}, error) {
	// Decode without a provider struct so unknown fields and explicit nulls survive.
	var value map[string]interface{}
	if err := sigsjson.UnmarshalCaseSensitivePreserveInts(override.Value.Raw, &value); err != nil {
		return nil, fmt.Errorf("decode value: %w", err)
	}

	// Select the registered subtree path for the standalone or embedded target.
	path := []string{"topologyConstraint"}
	if override.Target == TargetPodCliqueSet {
		path = []string{"spec", "template", "topologyConstraint"}
	}
	topologyConstraint, found, err := unstructured.NestedFieldNoCopy(value, path...)
	if err != nil {
		return nil, fmt.Errorf("read value.%s: %w", strings.Join(path, "."), err)
	}
	if !found {
		return nil, fmt.Errorf("value.%s is required", strings.Join(path, "."))
	}
	return topologyConstraint, nil
}
