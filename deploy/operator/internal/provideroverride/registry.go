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

// Package provideroverride defines the provider-schema targets and ownership
// boundaries shared by DGD defaulting, validation, and reconciliation.
package provideroverride

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

// Scope identifies one stable DGD provider context. It deliberately does not
// contain generated Kubernetes resource names.
type Scope string

const (
	ScopeRoot            Scope = "root"
	ScopeComponent       Scope = "component"
	ScopeMultinodeLeader Scope = "multinode.leader"
	ScopeMultinodeWorker Scope = "multinode.worker"
)

const (
	// TargetPodCliqueSet identifies the Grove root resource schema.
	TargetPodCliqueSet = "PodCliqueSet"
	// TargetPodCliqueTemplateSpec identifies the Grove embedded PCLQ template schema.
	TargetPodCliqueTemplateSpec = "PodCliqueTemplateSpec"
	// TargetPodCliqueScalingGroupConfig identifies the Grove embedded PCSG config schema.
	TargetPodCliqueScalingGroupConfig = "PodCliqueScalingGroupConfig"
)

// GroveAPIVersion is the Grove schema version supported by persisted overrides.
const GroveAPIVersion = "grove.io/v1alpha1"

// ValueError reports an invalid path inside override.value.
type ValueError struct {
	Path               string
	Detail             string
	OwnershipViolation bool
}

// Error implements error.
func (e ValueError) Error() string {
	if e.Path == "" {
		return e.Detail
	}
	return fmt.Sprintf("%s: %s", e.Path, e.Detail)
}

// ExpectedTarget returns the one provider schema target allowed for a DGD
// provider context and component shape. component may be nil only for
// ScopeRoot.
func ExpectedTarget(
	provider string,
	apiVersion string,
	scope Scope,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) (string, error) {
	// Accept only a provider and schema version registered by this release.
	if provider != consts.WorkloadProviderGrove {
		return "", fmt.Errorf("workload provider %q does not support provider overrides", provider)
	}
	if apiVersion != GroveAPIVersion {
		return "", fmt.Errorf("unsupported Grove apiVersion %q; supported value is %q", apiVersion, GroveAPIVersion)
	}

	// Resolve the native resource or embedded schema from the stable DGD context.
	switch scope {
	case ScopeRoot:
		return TargetPodCliqueSet, nil
	case ScopeComponent:
		if component == nil {
			return "", fmt.Errorf("component context is required for scope %q", scope)
		}
		if component.UsesPCSG() {
			return TargetPodCliqueScalingGroupConfig, nil
		}
		return TargetPodCliqueTemplateSpec, nil
	case ScopeMultinodeLeader, ScopeMultinodeWorker:
		if component == nil || !component.IsMultinode() {
			return "", fmt.Errorf("scope %q requires a multinode component", scope)
		}
		if component.IsInterPodGMSEnabled() {
			return "", fmt.Errorf("scope %q is not supported for inter-pod GMS components", scope)
		}
		return TargetPodCliqueTemplateSpec, nil
	default:
		return "", fmt.Errorf("unsupported provider override scope %q", scope)
	}
}

// DefaultTarget persists the target when the provider context has one
// unambiguous registered target. Unsupported contexts remain unchanged so the
// validating webhook can return a field-specific error. override must not be
// nil; component may be nil only for ScopeRoot.
func DefaultTarget(
	override *nvidiacomv1beta1.ProviderOverride,
	provider string,
	scope Scope,
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) {
	// Preserve explicitly selected targets.
	if override.Target != "" {
		return
	}

	// Persist a target only when this release can resolve it unambiguously.
	target, err := ExpectedTarget(provider, override.APIVersion, scope, component)
	if err == nil {
		override.Target = target
	}
}

// ValidateValue verifies that an override writes only provider-owned paths for
// its registered target. Descendants of topologyConstraint remain opaque so a
// newer provider field survives an older Dynamo schema.
func ValidateValue(target string, raw []byte) []ValueError {
	// Require an object before decoding it into a registered provider type.
	root, err := decodeObject(raw, "")
	if err != nil {
		return []ValueError{*err}
	}
	if valueErr := validateKnownGroveShape(target, raw); valueErr != nil {
		return []ValueError{*valueErr}
	}

	// Enforce the Dynamo-owned boundary for the resolved provider target.
	switch target {
	case TargetPodCliqueSet:
		return validatePodCliqueSetValue(root)
	case TargetPodCliqueTemplateSpec, TargetPodCliqueScalingGroupConfig:
		return validateTopologyOwner(root, "")
	default:
		return []ValueError{{Detail: fmt.Sprintf("target %q has no registered ownership policy", target)}}
	}
}

// validateKnownGroveShape checks fields understood by this Dynamo release
// against the exact Grove resource or embedded Go type. encoding/json ignores
// unknown fields so forward-compatible descendants remain opaque.
func validateKnownGroveShape(target string, raw []byte) *ValueError {
	// Select the exact standalone or embedded type registered for this target.
	var destination any
	switch target {
	case TargetPodCliqueSet:
		destination = &grovev1alpha1.PodCliqueSet{}
	case TargetPodCliqueTemplateSpec:
		destination = &grovev1alpha1.PodCliqueTemplateSpec{}
	case TargetPodCliqueScalingGroupConfig:
		destination = &grovev1alpha1.PodCliqueScalingGroupConfig{}
	default:
		return nil
	}

	// Validate known fields while leaving newer provider descendants opaque.
	if err := json.Unmarshal(raw, destination); err != nil {
		return &ValueError{Detail: fmt.Sprintf("does not match the registered %s schema: %v", target, err)}
	}
	return nil
}

// WritesGroveTopology reports whether a registered Grove override contains a
// topologyConstraint subtree. Invalid values return false and are reported by
// ValidateValue separately.
func WritesGroveTopology(target string, raw []byte) bool {
	// Invalid values are reported by ValidateValue, not by composition detection.
	root, err := decodeObject(raw, "")
	if err != nil {
		return false
	}

	// Locate topology at the path owned by the registered target.
	switch target {
	case TargetPodCliqueSet:
		spec, ok := childObject(root, "spec")
		if !ok {
			return false
		}
		template, ok := childObject(spec, "template")
		if !ok {
			return false
		}
		_, exists := template["topologyConstraint"]
		return exists
	case TargetPodCliqueTemplateSpec, TargetPodCliqueScalingGroupConfig:
		_, exists := root["topologyConstraint"]
		return exists
	default:
		return false
	}
}

func validatePodCliqueSetValue(root map[string]json.RawMessage) []ValueError {
	// Walk the sparse root fragment down to its topology-owning template.
	errs := rejectUnknown(root, "", "spec")
	spec, valueErr := requiredObject(root, "spec", "spec")
	if valueErr != nil {
		return append(errs, *valueErr)
	}
	errs = append(errs, rejectUnknown(spec, "spec", "template")...)
	template, valueErr := requiredObject(spec, "template", "spec.template")
	if valueErr != nil {
		return append(errs, *valueErr)
	}
	errs = append(errs, validateTopologyOwner(template, "spec.template")...)
	return errs
}

func validateTopologyOwner(root map[string]json.RawMessage, path string) []ValueError {
	// Permit only the provider-owned topology subtree at embedded targets.
	errs := rejectUnknown(root, path, "topologyConstraint")
	topologyPath := joinPath(path, "topologyConstraint")
	_, valueErr := requiredObject(root, "topologyConstraint", topologyPath)
	if valueErr != nil {
		errs = append(errs, *valueErr)
	}
	return errs
}

func decodeObject(raw []byte, path string) (map[string]json.RawMessage, *ValueError) {
	// Reject empty and non-object JSON before returning a traversable map.
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, &ValueError{Path: path, Detail: "must be a JSON object"}
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, &ValueError{Path: path, Detail: fmt.Sprintf("must be a JSON object: %v", err)}
	}
	if object == nil {
		return nil, &ValueError{Path: path, Detail: "must be a JSON object"}
	}
	return object, nil
}

func requiredObject(root map[string]json.RawMessage, key, path string) (map[string]json.RawMessage, *ValueError) {
	// Distinguish an absent required subtree from a malformed one.
	raw, exists := root[key]
	if !exists {
		return nil, &ValueError{Path: path, Detail: "is required"}
	}
	return decodeObject(raw, path)
}

func childObject(root map[string]json.RawMessage, key string) (map[string]json.RawMessage, bool) {
	// Treat absent or malformed optional children as unavailable.
	raw, exists := root[key]
	if !exists {
		return nil, false
	}
	object, valueErr := decodeObject(raw, key)
	return object, valueErr == nil
}

func rejectUnknown(root map[string]json.RawMessage, path string, allowed ...string) []ValueError {
	// Build the exact set of fields Dynamo delegates to the provider.
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedSet[key] = struct{}{}
	}

	// Sort rejected paths so admission errors remain deterministic.
	unknown := make([]string, 0)
	for key := range root {
		if _, ok := allowedSet[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	sort.Strings(unknown)

	// Classify every remaining field as an ownership violation.
	errs := make([]ValueError, 0, len(unknown))
	for _, key := range unknown {
		errs = append(errs, ValueError{
			Path:               joinPath(path, key),
			Detail:             "is Dynamo-owned or not enabled for provider override",
			OwnershipViolation: true,
		})
	}
	return errs
}

func joinPath(parent, child string) string {
	if parent == "" {
		return child
	}
	return parent + "." + child
}
