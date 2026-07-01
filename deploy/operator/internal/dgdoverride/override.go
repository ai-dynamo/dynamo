/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package dgdoverride applies versioned partial overrides to complete
// DynamoGraphDeployment blueprints.
package dgdoverride

import (
	"fmt"
	"sort"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
)

const dgdKind = "DynamoGraphDeployment"

var (
	alphaGVK = nvidiacomv1alpha1.GroupVersion.WithKind(dgdKind)
	betaGVK  = nvidiacomv1beta1.GroupVersion.WithKind(dgdKind)
)

// Warning describes a non-fatal part of an override that was ignored to
// preserve the identity or topology of the generated blueprint.
type Warning struct {
	Path    string
	Message string
}

func (w Warning) String() string {
	if w.Path == "" {
		return w.Message
	}
	return w.Path + ": " + w.Message
}

// Merger applies DGD overrides using the structural schema generated from the
// same CRD served by the operator.
type Merger struct {
	typeConverter managedfields.TypeConverter
}

// New creates a DGD override merger for the supported alpha and beta APIs.
func New() (*Merger, error) {
	typeConverter, err := newTypeConverter()
	if err != nil {
		return nil, err
	}
	return &Merger{typeConverter: typeConverter}, nil
}

// Apply overlays a partial DGD override onto a complete DGD blueprint.
//
// The merge happens in the override's API version. If the versions differ,
// Apply converts the complete blueprint before merging and converts the
// complete result back afterward. The returned object therefore always has
// the same GVK as blueprint. Neither input is mutated. Structural schema
// validation runs here; admission defaults and CEL validation remain the API
// server's responsibility when the resulting DGD is submitted.
func (m *Merger) Apply(
	blueprint *unstructured.Unstructured,
	override *unstructured.Unstructured,
) (*unstructured.Unstructured, []Warning, error) {
	if m == nil || m.typeConverter == nil {
		return nil, nil, fmt.Errorf("DGD override merger is not initialized")
	}

	blueprintGVK, err := validateDGD(blueprint, "blueprint")
	if err != nil {
		return nil, nil, err
	}
	overrideGVK, err := validateDGD(override, "override")
	if err != nil {
		return nil, nil, err
	}
	if _, err := m.typeConverter.ObjectToTyped(blueprint); err != nil {
		return nil, nil, fmt.Errorf("validate %s blueprint: %w", blueprintGVK.GroupVersion(), err)
	}

	working := blueprint.DeepCopy()
	if blueprintGVK != overrideGVK {
		working, err = convertDGD(working, overrideGVK)
		if err != nil {
			return nil, nil, fmt.Errorf(
				"convert complete blueprint from %s to %s: %w",
				blueprintGVK.GroupVersion(),
				overrideGVK.GroupVersion(),
				err,
			)
		}
	}

	partial, warnings, err := prepareOverride(working, override, overrideGVK)
	if err != nil {
		return nil, warnings, err
	}

	baseTyped, err := m.typeConverter.ObjectToTyped(working)
	if err != nil {
		return nil, warnings, fmt.Errorf("validate converted %s blueprint: %w", overrideGVK.GroupVersion(), err)
	}
	partialTyped, err := m.typeConverter.ObjectToTyped(partial)
	if err != nil {
		return nil, warnings, fmt.Errorf("validate %s override: %w", overrideGVK.GroupVersion(), err)
	}
	mergedTyped, err := baseTyped.Merge(partialTyped)
	if err != nil {
		return nil, warnings, fmt.Errorf("merge %s override: %w", overrideGVK.GroupVersion(), err)
	}
	mergedObject, err := m.typeConverter.TypedToObject(mergedTyped)
	if err != nil {
		return nil, warnings, fmt.Errorf("materialize merged %s DGD: %w", overrideGVK.GroupVersion(), err)
	}
	merged, ok := mergedObject.(*unstructured.Unstructured)
	if !ok {
		return nil, warnings, fmt.Errorf("materialize merged DGD: expected unstructured object, got %T", mergedObject)
	}
	merged.SetGroupVersionKind(overrideGVK)

	if blueprintGVK != overrideGVK {
		merged, err = convertDGD(merged, blueprintGVK)
		if err != nil {
			return nil, warnings, fmt.Errorf(
				"convert complete merged DGD from %s back to %s: %w",
				overrideGVK.GroupVersion(),
				blueprintGVK.GroupVersion(),
				err,
			)
		}
	}
	merged.SetGroupVersionKind(blueprintGVK)
	if _, err := m.typeConverter.ObjectToTyped(merged); err != nil {
		return nil, warnings, fmt.Errorf("validate final %s DGD: %w", blueprintGVK.GroupVersion(), err)
	}
	return merged, warnings, nil
}

func validateDGD(object *unstructured.Unstructured, role string) (schema.GroupVersionKind, error) {
	if object == nil {
		return schema.GroupVersionKind{}, fmt.Errorf("%s must not be nil", role)
	}
	gvk := object.GroupVersionKind()
	if gvk == alphaGVK || gvk == betaGVK {
		return gvk, nil
	}
	return schema.GroupVersionKind{}, fmt.Errorf(
		"%s must be %s or %s, got apiVersion %q kind %q",
		role,
		alphaGVK.GroupVersion(),
		betaGVK.GroupVersion(),
		object.GetAPIVersion(),
		object.GetKind(),
	)
}

func convertDGD(object *unstructured.Unstructured, target schema.GroupVersionKind) (*unstructured.Unstructured, error) {
	source := object.GroupVersionKind()
	if source == target {
		return object.DeepCopy(), nil
	}

	var converted runtime.Object
	switch {
	case source == alphaGVK && target == betaGVK:
		alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, alpha); err != nil {
			return nil, fmt.Errorf("decode alpha DGD: %w", err)
		}
		beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := alpha.ConvertTo(beta); err != nil {
			return nil, fmt.Errorf("convert alpha DGD to beta: %w", err)
		}
		converted = beta
	case source == betaGVK && target == alphaGVK:
		beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, beta); err != nil {
			return nil, fmt.Errorf("decode beta DGD: %w", err)
		}
		alpha := &nvidiacomv1alpha1.DynamoGraphDeployment{}
		if err := alpha.ConvertFrom(beta); err != nil {
			return nil, fmt.Errorf("convert beta DGD to alpha: %w", err)
		}
		converted = alpha
	default:
		return nil, fmt.Errorf("unsupported DGD conversion from %s to %s", source, target)
	}

	content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(converted)
	if err != nil {
		return nil, fmt.Errorf("encode converted DGD: %w", err)
	}
	result := &unstructured.Unstructured{Object: content}
	result.SetGroupVersionKind(target)
	return result, nil
}

func prepareOverride(
	blueprint *unstructured.Unstructured,
	override *unstructured.Unstructured,
	gvk schema.GroupVersionKind,
) (*unstructured.Unstructured, []Warning, error) {
	partial := override.DeepCopy()
	if _, found := partial.Object["status"]; found {
		return nil, nil, fmt.Errorf("override status is not supported")
	}
	warnings, err := sanitizeMetadata(partial)
	if err != nil {
		return nil, warnings, err
	}
	if err := rejectNullValues(partial.Object, ""); err != nil {
		return nil, warnings, err
	}

	switch gvk {
	case alphaGVK:
		more, err := prepareAlphaServices(blueprint, partial)
		warnings = append(warnings, more...)
		if err != nil {
			return nil, warnings, err
		}
	case betaGVK:
		more, err := prepareBetaComponents(blueprint, partial)
		warnings = append(warnings, more...)
		if err != nil {
			return nil, warnings, err
		}
	default:
		return nil, warnings, fmt.Errorf("unsupported override GVK %s", gvk)
	}

	return partial, warnings, nil
}

func sanitizeMetadata(override *unstructured.Unstructured) ([]Warning, error) {
	metadata, found, err := unstructured.NestedMap(override.Object, "metadata")
	if err != nil {
		return nil, fmt.Errorf("override metadata must be an object: %w", err)
	}
	if !found {
		return nil, nil
	}

	allowed := map[string]interface{}{}
	for _, key := range []string{"annotations", "labels"} {
		if value, ok := metadata[key]; ok {
			allowed[key] = value
		}
	}
	warnings := make([]Warning, 0, 2)
	// Cross-version conversion stores round-trip state under this operator-owned
	// prefix. Letting an override replace it could silently corrupt preserved fields.
	if value, ok := allowed["annotations"]; ok {
		if annotations, ok := value.(map[string]interface{}); ok {
			reserved := make([]string, 0)
			for key := range annotations {
				if strings.HasPrefix(key, "nvidia.com/dgd-") {
					reserved = append(reserved, key)
					delete(annotations, key)
				}
			}
			sort.Strings(reserved)
			if len(annotations) == 0 {
				delete(allowed, "annotations")
			}
			if len(reserved) > 0 {
				warnings = append(warnings, Warning{
					Path:    "metadata.annotations",
					Message: "ignored reserved operator keys: " + strings.Join(reserved, ", "),
				})
			}
		}
	}

	ignored := make([]string, 0, len(metadata))
	for key := range metadata {
		if key != "annotations" && key != "labels" {
			ignored = append(ignored, key)
		}
	}
	sort.Strings(ignored)

	if len(allowed) == 0 {
		unstructured.RemoveNestedField(override.Object, "metadata")
	} else if err := unstructured.SetNestedMap(override.Object, allowed, "metadata"); err != nil {
		return nil, fmt.Errorf("sanitize override metadata: %w", err)
	}
	if len(ignored) == 0 {
		return warnings, nil
	}
	warnings = append([]Warning{{
		Path:    "metadata",
		Message: "ignored identity/runtime fields: " + strings.Join(ignored, ", "),
	}}, warnings...)
	return warnings, nil
}

func rejectNullValues(value interface{}, path string) error {
	switch typed := value.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			childPath := key
			if path != "" {
				childPath = path + "." + key
			}
			if typed[key] == nil {
				return fmt.Errorf("override %s must not be null; field deletion is not supported", childPath)
			}
			if err := rejectNullValues(typed[key], childPath); err != nil {
				return err
			}
		}
	case []interface{}:
		for i, item := range typed {
			childPath := fmt.Sprintf("%s[%d]", path, i)
			if item == nil {
				return fmt.Errorf("override %s must not be null; field deletion is not supported", childPath)
			}
			if err := rejectNullValues(item, childPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func prepareAlphaServices(
	blueprint *unstructured.Unstructured,
	override *unstructured.Unstructured,
) ([]Warning, error) {
	baseServices, _, err := unstructured.NestedMap(blueprint.Object, "spec", "services")
	if err != nil {
		return nil, fmt.Errorf("alpha blueprint spec.services must be an object: %w", err)
	}
	overrideServices, found, err := unstructured.NestedMap(override.Object, "spec", "services")
	if err != nil {
		return nil, fmt.Errorf("alpha override spec.services must be an object: %w", err)
	}
	if !found {
		return nil, nil
	}

	names := make([]string, 0, len(overrideServices))
	for name := range overrideServices {
		names = append(names, name)
	}
	sort.Strings(names)

	warnings := make([]Warning, 0)
	for _, name := range names {
		if _, exists := baseServices[name]; !exists {
			delete(overrideServices, name)
			warnings = append(warnings, Warning{
				Path:    "spec.services." + name,
				Message: "ignored because the generated blueprint has no such service",
			})
			continue
		}
		if name == "Frontend" || name == "Planner" {
			continue
		}
		if err := appendAlphaWorkerArgs(baseServices[name], overrideServices[name]); err != nil {
			return warnings, fmt.Errorf("spec.services.%s.extraPodSpec.mainContainer.args: %w", name, err)
		}
	}

	if err := unstructured.SetNestedMap(override.Object, overrideServices, "spec", "services"); err != nil {
		return warnings, fmt.Errorf("prepare alpha override services: %w", err)
	}
	return warnings, nil
}

// appendAlphaWorkerArgs preserves the legacy v1alpha1 profiler contract, where
// worker args extend the generated command line. V1beta1 intentionally follows
// the CRD's atomic-list semantics and replaces container args instead.
func appendAlphaWorkerArgs(baseValue, overrideValue interface{}) error {
	base, ok := baseValue.(map[string]interface{})
	if !ok {
		return fmt.Errorf("blueprint service must be an object, got %T", baseValue)
	}
	partial, ok := overrideValue.(map[string]interface{})
	if !ok {
		return fmt.Errorf("override service must be an object, got %T", overrideValue)
	}

	overrideArgs, found, err := unstructured.NestedStringSlice(
		partial,
		"extraPodSpec",
		"mainContainer",
		"args",
	)
	if err != nil {
		return fmt.Errorf("must be a list of strings: %w", err)
	}
	if !found {
		return nil
	}
	baseArgs, _, err := unstructured.NestedStringSlice(
		base,
		"extraPodSpec",
		"mainContainer",
		"args",
	)
	if err != nil {
		return fmt.Errorf("blueprint value must be a list of strings: %w", err)
	}
	combined := append(append([]string(nil), baseArgs...), overrideArgs...)
	if err := unstructured.SetNestedStringSlice(
		partial,
		combined,
		"extraPodSpec",
		"mainContainer",
		"args",
	); err != nil {
		return fmt.Errorf("set combined arguments: %w", err)
	}
	return nil
}

func prepareBetaComponents(
	blueprint *unstructured.Unstructured,
	override *unstructured.Unstructured,
) ([]Warning, error) {
	baseComponents, _, err := unstructured.NestedSlice(blueprint.Object, "spec", "components")
	if err != nil {
		return nil, fmt.Errorf("beta blueprint spec.components must be a list: %w", err)
	}
	overrideComponents, found, err := unstructured.NestedSlice(override.Object, "spec", "components")
	if err != nil {
		return nil, fmt.Errorf("beta override spec.components must be a list: %w", err)
	}
	if !found {
		return nil, nil
	}

	baseNames := make(map[string]struct{}, len(baseComponents))
	for i, value := range baseComponents {
		component, ok := value.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("beta blueprint spec.components[%d] must be an object, got %T", i, value)
		}
		name, ok := component["name"].(string)
		if !ok || name == "" {
			return nil, fmt.Errorf("beta blueprint spec.components[%d].name must be a non-empty string", i)
		}
		baseNames[name] = struct{}{}
	}

	filtered := make([]interface{}, 0, len(overrideComponents))
	warnings := make([]Warning, 0)
	for i, value := range overrideComponents {
		component, ok := value.(map[string]interface{})
		if !ok {
			return warnings, fmt.Errorf("beta override spec.components[%d] must be an object, got %T", i, value)
		}
		name, ok := component["name"].(string)
		if !ok || name == "" {
			return warnings, fmt.Errorf("beta override spec.components[%d].name must be a non-empty string", i)
		}
		if _, exists := baseNames[name]; !exists {
			warnings = append(warnings, Warning{
				Path:    fmt.Sprintf("spec.components[name=%s]", name),
				Message: "ignored because the generated blueprint has no such component",
			})
			continue
		}
		filtered = append(filtered, component)
	}

	if err := unstructured.SetNestedSlice(override.Object, filtered, "spec", "components"); err != nil {
		return warnings, fmt.Errorf("prepare beta override components: %w", err)
	}
	return warnings, nil
}
