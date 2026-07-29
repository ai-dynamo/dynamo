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

// MarshalJSON overrides for the public v1beta1 API types.
//
// # Why these overrides exist
//
// Several of our v1beta1 spec fields are native corev1 types that contain
// embedded value-typed sub-structs (for example `corev1.PodTemplateSpec`
// carries a value-typed `metav1.ObjectMeta`, and `corev1.Container` carries a
// value-typed `corev1.ResourceRequirements`). Go's `encoding/json` does NOT
// omit zero-valued struct fields (`,omitempty` only applies to nil pointers,
// nil interfaces, empty slices/maps, and numeric/string zeros), so a vanilla
// `json.Marshal` of a conversion webhook response always emits artefacts
// like:
//
//	"podTemplate": { "metadata": {}, "spec": { "containers": [ { "resources": {}, ... } ] } }
//
// even when the user's source YAML -- and therefore the
// `kubectl.kubernetes.io/last-applied-configuration` annotation kubectl
// stamps onto the object -- contains no `metadata` or `resources` keys. That
// asymmetry breaks kubectl client-side apply's two-way merge diff against
// the stored `last-applied-configuration` and causes `.metadata.generation`
// to bump on every `kubectl apply` of an unchanged manifest.
//
// # The subtle failure mode we have to avoid
//
// A naive "just delete every empty map" pass would silently break sentinel
// types like `v1beta1.ScalingAdapter`, which is a deliberately empty struct
// whose *presence* at `components[i].scalingAdapter: {}` opts a component
// into the DGDSA autoscaling path. Stripping that `{}` would turn the wire
// view into an opt-out view, breaking autoscaling without any type-checker
// warning. Any future "marker" struct we add to the API would hit the same
// pitfall.
//
// # How this package distinguishes artefact from sentinel
//
// The distinguishing property is encoded in the Go type, not the JSON wire:
// sentinels are declared as POINTERS (`*ScalingAdapter` with `,omitempty`)
// so that `nil` correctly disappears from the JSON and a non-nil empty
// struct signals "user opted in". Encoder artefacts are declared as
// value-typed fields (`ObjectMeta metav1.ObjectMeta` inline on
// `corev1.PodTemplateSpec`, `Resources corev1.ResourceRequirements` on
// `corev1.Container`) because the upstream corev1/metav1 package authors
// didn't reach for pointers -- and Go's `omitempty` silently ignores them.
//
// The normalizer below walks the serialized JSON in lockstep with the Go
// type tree via reflection and strips a map entry iff (a) its JSON value
// is an empty object AND (b) the corresponding Go field is a non-pointer
// struct type. Pointer-typed empty objects are preserved as sentinels.
// Adding a new API field with a value-typed sub-struct automatically
// participates in the normalization without further code changes, and
// adding a new pointer-typed sentinel struct is automatically preserved.
//
// # Scope
//
// MarshalJSON is declared on the Go type, so the normalization runs on
// every serialization path: conversion webhook responses, direct
// `json.Marshal(obj)` calls, controller-runtime client writes, manifests
// dumped by `%+v`-style logging, Prometheus string exporters, and so on.
// Semantics are preserved by construction -- round-tripping through
// `Unmarshal(Marshal(x))` returns a Go value equal to `x` under
// `reflect.DeepEqual` -- so the broader scope is benign, but worth
// keeping in mind when debugging unexpected wire shapes anywhere in the
// operator.
//
// # Key ordering
//
// The normalizer round-trips the serialized bytes through `map[string]any`
// to drive the reflection walk, and `encoding/json` marshals maps with
// alphabetically sorted keys. Output therefore uses alphabetical key order
// instead of Go struct-declaration order. Kubernetes tooling is
// key-order-agnostic (apiserver decode, SMD, CSA, etcd storage, kubectl
// diff all ignore ordering), but any downstream consumer that
// byte-compares raw JSON (for example, content hashing for
// change detection) will observe a one-time reshuffle.

package v1beta1

import (
	"encoding/json"
	"reflect"
	"strings"
	"sync"
)

// normalizeV1Beta1JSON strips Go-encoder empty-struct artefacts from the
// serialized JSON of a v1beta1 root object. The walk is driven by
// reflection over `rootType` so that the artefact vs. sentinel distinction
// follows the Go field declarations rather than a hand-maintained allowlist
// of JSON paths. Errors only arise from malformed JSON produced by the
// embedded encoder, which would indicate a bug in the standard library
// rather than user input.
func normalizeV1Beta1JSON(raw []byte, rootType reflect.Type) ([]byte, error) {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, err
	}
	stripEmptyValueStructs(rootType, v)
	return json.Marshal(v)
}

// stripEmptyValueStructs walks `jsonVal` in lockstep with `goType`, deleting
// any map entry whose JSON value is an empty object and whose corresponding
// Go field is a non-pointer struct type. The deletion cascades: if all
// children of a value-typed sub-struct field themselves reduce to `{}`, the
// parent collapses to `{}` too, which then trips the same check at its own
// parent. This is intentional and matches the user-YAML-intent side of the
// CSA diff (value-typed artefacts are never user-authored).
//
// Unknown JSON keys (fields that exist on the wire but not in the Go type)
// are preserved untouched: the reflection walk just stops descending. This
// covers `x-kubernetes-preserve-unknown-fields` payloads like
// `DynamoGraphDeploymentRequest.Template.Spec.Topology.*` without risk of
// mutilation.
func stripEmptyValueStructs(goType reflect.Type, jsonVal any) {
	goType = derefToConcrete(goType)
	if goType == nil {
		return
	}

	switch jv := jsonVal.(type) {
	case map[string]any:
		switch goType.Kind() {
		case reflect.Struct:
			stripStructFields(goType, jv)
		case reflect.Map:
			elemType := goType.Elem()
			for k := range jv {
				stripEmptyValueStructs(elemType, jv[k])
			}
		}
	case []any:
		if goType.Kind() == reflect.Slice || goType.Kind() == reflect.Array {
			elemType := goType.Elem()
			for i := range jv {
				stripEmptyValueStructs(elemType, jv[i])
			}
		}
	}
}

// stripStructFields applies the artefact rule to a single struct-typed JSON
// object. Inputs are guaranteed by the caller to be a (map, struct type)
// pair. The fields lookup is cached by type to keep the recursive walk
// linear-ish in object size.
func stripStructFields(goType reflect.Type, jv map[string]any) {
	fields := jsonFieldsOf(goType)
	for key, child := range jv {
		fi, known := fields[key]
		if !known {
			continue
		}
		stripEmptyValueStructs(fi.Type, child)
		// The artefact rule: a JSON `{}` is stripped iff the corresponding
		// Go field is a value-typed struct (non-pointer). Pointer-typed
		// empty objects are preserved as sentinels (see ScalingAdapter).
		if m, ok := child.(map[string]any); ok && len(m) == 0 && fi.Type.Kind() == reflect.Struct {
			delete(jv, key)
		}
	}
}

// derefToConcrete follows pointer indirection so callers can reason about
// the underlying struct/map/slice kind without worrying about whether the
// Go field was declared as a pointer. Interface types bail out: we don't
// know their concrete type until runtime and our walk is type-driven.
func derefToConcrete(t reflect.Type) reflect.Type {
	for t != nil && t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t != nil && t.Kind() == reflect.Interface {
		return nil
	}
	return t
}

// jsonFieldsOf returns a map from JSON field name to the Go StructField
// metadata for `t`, honoring `json:"..."` tags and promoting embedded
// anonymous struct fields that use the no-name tag form (`json:",inline"`,
// `json:",omitempty"`, or absent tag) exactly as `encoding/json` does.
// Embedded fields with an explicit JSON name (e.g. `metav1.ObjectMeta
// \`json:"metadata,omitempty"\“) are treated as named fields, not promoted.
// Cached per type to keep repeated walks cheap.
var fieldCache sync.Map // map[reflect.Type]map[string]reflect.StructField

func jsonFieldsOf(t reflect.Type) map[string]reflect.StructField {
	if cached, ok := fieldCache.Load(t); ok {
		return cached.(map[string]reflect.StructField)
	}
	out := make(map[string]reflect.StructField)
	collectFields(t, out)
	fieldCache.Store(t, out)
	return out
}

func collectFields(t reflect.Type, out map[string]reflect.StructField) {
	var anonymous []reflect.StructField
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name := tag
		if comma := strings.IndexByte(tag, ','); comma >= 0 {
			name = tag[:comma]
		}
		if name == "" && f.Anonymous {
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				anonymous = append(anonymous, f)
			}
			continue
		}
		if name == "" {
			name = f.Name
		}
		out[name] = f
	}
	for _, f := range anonymous {
		ft := f.Type
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}
		promoted := make(map[string]reflect.StructField)
		collectFields(ft, promoted)
		for name, promotedField := range promoted {
			if _, exists := out[name]; !exists {
				out[name] = promotedField
			}
		}
	}
}

// Type aliases used to disable MarshalJSON recursion: json.Marshal on the
// aliased type uses only the default struct encoder, never the custom
// method defined on the original type.
//
// The MarshalJSON methods below use VALUE receivers on purpose. The Go
// encoder only invokes a custom Marshaler when the value it is currently
// encoding is addressable (for pointer receivers) OR when the method has
// a value receiver. Code paths that serialize our types through reflection
// or by-value (list-item iteration, `runtime.Object` interface values,
// `%+v` formatting, etc.) often do not hold an addressable pointer, so a
// pointer-receiver MarshalJSON would silently fall back to the default
// struct encoder and skip the normalization. Value receivers close that
// hole at the cost of one struct copy per marshal, which is negligible
// compared to the JSON encode that follows.

type dynamoGraphDeploymentForMarshal DynamoGraphDeployment

// MarshalJSON serializes a DynamoGraphDeployment and applies the v1beta1
// normalization pass. See the package-level comment for the rationale.
func (d DynamoGraphDeployment) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(dynamoGraphDeploymentForMarshal(d))
	if err != nil {
		return nil, err
	}
	return normalizeV1Beta1JSON(raw, reflect.TypeOf(DynamoGraphDeployment{}))
}

type dynamoComponentDeploymentForMarshal DynamoComponentDeployment

// MarshalJSON serializes a DynamoComponentDeployment and applies the v1beta1
// normalization pass. See the package-level comment for the rationale.
func (d DynamoComponentDeployment) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(dynamoComponentDeploymentForMarshal(d))
	if err != nil {
		return nil, err
	}
	return normalizeV1Beta1JSON(raw, reflect.TypeOf(DynamoComponentDeployment{}))
}

type dynamoGraphDeploymentRequestForMarshal DynamoGraphDeploymentRequest

// MarshalJSON serializes a DynamoGraphDeploymentRequest and applies the
// v1beta1 normalization pass. DGDR does not currently carry pod templates
// in its request payload, but its Template field can host a synthesized
// DGD that does; normalizing here keeps the three root kinds behaving
// uniformly and avoids surprising field drift if DGDR ever gains a
// pod-template-bearing field.
func (d DynamoGraphDeploymentRequest) MarshalJSON() ([]byte, error) {
	raw, err := json.Marshal(dynamoGraphDeploymentRequestForMarshal(d))
	if err != nil {
		return nil, err
	}
	return normalizeV1Beta1JSON(raw, reflect.TypeOf(DynamoGraphDeploymentRequest{}))
}
