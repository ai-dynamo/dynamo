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

package api

import (
	"fmt"
	"maps"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

type roundTripHubObject interface {
	conversion.Hub
	metav1.Object
	runtime.Object
}

type roundTripSpokeObject[S any] interface {
	*S
	conversion.Convertible
	metav1.Object
	runtime.Object
}

// fuzzMutatedSpokeCarrier checks this flow:
//
//	hub -> spoke -(mutation)-> hub -> spoke
//
// The spoke object is the "carrier": it contains fields converted from the hub
// plus preservation annotations for hub-only data. The test mutates that spoke
// object without deleting those annotations, converts spoke -> hub -> spoke,
// and expects the final spoke to match the mutated carrier modulo only the
// reserved preservation annotations.
func fuzzMutatedSpokeCarrier[
	H roundTripHubObject,
	S any,
	PS roundTripSpokeObject[S],
](t *testing.T, name string, newHub func() H) {
	t.Helper()
	t.Logf("hub->spoke-(mutation)->hub->spoke %s seed=%d iters=%d", name, *fuzzSeed, *fuzzIters)
	f := newRoundTripFiller(*fuzzSeed)
	for i := 0; i < *fuzzIters; i++ {
		// Stage 1, hub -> spoke: build the spoke carrier with hub-only data preserved in annotations.
		in := newHub()
		f.Fill(in)
		carrier := PS(new(S))
		if err := carrier.ConvertFrom(in); err != nil {
			t.Fatalf("%s iter %d ConvertFrom: %v\ninput=%s", name, i, err, mustJSON(in))
		}

		// Stage 2, mutation: update the carrier without deleting existing values or annotations.
		patch := PS(new(S))
		f.Fill(patch)
		sparseFuzzInto(carrier, patch)
		carrierBeforeYAML := toYAML(t, carrier)

		// Stage 3, spoke -> hub: convert away from the mutated carrier.
		hub := newHub()
		if err := carrier.ConvertTo(hub); err != nil {
			t.Fatalf("%s iter %d ConvertTo mutated spoke: %v\ninput=%s", name, i, err, mustJSON(carrier))
		}

		// Stage 3 invariant: conversion must not mutate the carrier in memory.
		if diff := cmp.Diff(carrierBeforeYAML, toYAML(t, carrier)); diff != "" {
			t.Fatalf("%s iter %d ConvertTo mutated spoke input (-before +after):\n%s\ninput=%s", name, i, diff, mustJSON(carrier))
		}

		// Stage 4, hub -> spoke: compare all live fields; preservation annotations may differ.
		out := PS(new(S))
		if err := out.ConvertFrom(hub); err != nil {
			t.Fatalf("%s iter %d ConvertFrom restored spoke: %v\ninput=%s", name, i, err, mustJSON(carrier))
		}
		if diff := cmp.Diff(
			comparableRoundTripObject(carrier),
			comparableRoundTripObject(out),
			cmpopts.EquateEmpty(),
		); diff != "" {
			t.Fatalf("%s iter %d hub->mutated-spoke->hub->spoke mismatch (-want +got):\n%s\ninput=%s", name, i, diff, mustJSON(carrier))
		}
	}
}

// fuzzMutatedHubCarrier checks this flow:
//
//	spoke -> hub -(mutation)-> spoke -> hub
//
// The hub object is the "carrier": it contains fields converted from the spoke
// plus preservation annotations for spoke-only data. The test mutates that hub
// object without deleting those annotations, converts hub -> spoke -> hub, and
// expects the final hub to match the mutated carrier modulo only the reserved
// preservation annotations.
func fuzzMutatedHubCarrier[
	H roundTripHubObject,
	S any,
	PS roundTripSpokeObject[S],
](t *testing.T, name string, newHub func() H) {
	t.Helper()
	t.Logf("spoke->hub-(mutation)->spoke->hub %s seed=%d iters=%d", name, *fuzzSeed, *fuzzIters)
	f := newRoundTripFiller(*fuzzSeed)
	for i := 0; i < *fuzzIters; i++ {
		// Stage 1, spoke -> hub: build the hub carrier with spoke-only data preserved in annotations.
		in := PS(new(S))
		f.Fill(in)
		carrier := newHub()
		if err := in.ConvertTo(carrier); err != nil {
			t.Fatalf("%s iter %d ConvertTo: %v\ninput=%s", name, i, err, mustJSON(in))
		}

		// Stage 2, mutation: update the carrier without deleting existing values or annotations.
		patch := newHub()
		f.Fill(patch)
		sparseFuzzInto(carrier, patch)
		carrierBeforeYAML := toYAML(t, carrier)

		// Stage 3, hub -> spoke: convert away from the mutated carrier.
		spoke := PS(new(S))
		if err := spoke.ConvertFrom(carrier); err != nil {
			t.Fatalf("%s iter %d ConvertFrom mutated hub: %v\ninput=%s", name, i, err, mustJSON(carrier))
		}

		// Stage 3 invariant: conversion must not mutate the carrier in memory.
		if diff := cmp.Diff(carrierBeforeYAML, toYAML(t, carrier)); diff != "" {
			t.Fatalf("%s iter %d ConvertFrom mutated hub input (-before +after):\n%s\ninput=%s", name, i, diff, mustJSON(carrier))
		}

		// Stage 4, spoke -> hub: compare all live fields; preservation annotations may differ.
		out := newHub()
		if err := spoke.ConvertTo(out); err != nil {
			t.Fatalf("%s iter %d ConvertTo restored hub: %v\ninput=%s", name, i, err, mustJSON(carrier))
		}
		if diff := cmp.Diff(
			comparableRoundTripObject(carrier),
			comparableRoundTripObject(out),
			cmpopts.EquateEmpty(),
		); diff != "" {
			t.Fatalf("%s iter %d spoke->mutated-hub->spoke->hub mismatch (-want +got):\n%s\ninput=%s", name, i, diff, mustJSON(carrier))
		}
	}
}

func TestFuzzRoundTripMutability(t *testing.T) {
	t.Run("DGD/hub-to-mutated-spoke", func(t *testing.T) {
		fuzzMutatedSpokeCarrier[*v1beta1.DynamoGraphDeployment, v1alpha1.DynamoGraphDeployment](t, "DGD",
			func() *v1beta1.DynamoGraphDeployment { return &v1beta1.DynamoGraphDeployment{} },
		)
	})
	t.Run("DGD/spoke-to-mutated-hub", func(t *testing.T) {
		fuzzMutatedHubCarrier[*v1beta1.DynamoGraphDeployment, v1alpha1.DynamoGraphDeployment](t, "DGD",
			func() *v1beta1.DynamoGraphDeployment { return &v1beta1.DynamoGraphDeployment{} },
		)
	})
	t.Run("DCD/hub-to-mutated-spoke", func(t *testing.T) {
		fuzzMutatedSpokeCarrier[*v1beta1.DynamoComponentDeployment, v1alpha1.DynamoComponentDeployment](t, "DCD",
			func() *v1beta1.DynamoComponentDeployment { return &v1beta1.DynamoComponentDeployment{} },
		)
	})
	t.Run("DCD/spoke-to-mutated-hub", func(t *testing.T) {
		fuzzMutatedHubCarrier[*v1beta1.DynamoComponentDeployment, v1alpha1.DynamoComponentDeployment](t, "DCD",
			func() *v1beta1.DynamoComponentDeployment { return &v1beta1.DynamoComponentDeployment{} },
		)
	})
	t.Run("DGDSA/hub-to-mutated-spoke", func(t *testing.T) {
		fuzzMutatedSpokeCarrier[*v1beta1.DynamoGraphDeploymentScalingAdapter, v1alpha1.DynamoGraphDeploymentScalingAdapter](t, "DGDSA",
			func() *v1beta1.DynamoGraphDeploymentScalingAdapter {
				return &v1beta1.DynamoGraphDeploymentScalingAdapter{}
			},
		)
	})
	t.Run("DGDSA/spoke-to-mutated-hub", func(t *testing.T) {
		fuzzMutatedHubCarrier[*v1beta1.DynamoGraphDeploymentScalingAdapter, v1alpha1.DynamoGraphDeploymentScalingAdapter](t, "DGDSA",
			func() *v1beta1.DynamoGraphDeploymentScalingAdapter {
				return &v1beta1.DynamoGraphDeploymentScalingAdapter{}
			},
		)
	})
}

func TestSparseFuzzInto(t *testing.T) {
	type nested struct {
		Value string
	}
	type item struct {
		Name   string
		Nested nested
	}
	type sample struct {
		Scalar      string
		Ptr         *nested
		Items       []item
		Bytes       []byte
		Annotations map[string]string
		Labels      map[string]nested
	}

	t.Run("recurses into existing values", func(t *testing.T) {
		dst := &sample{
			Scalar: "old",
			Ptr:    &nested{Value: "old-ptr"},
			Items: []item{
				{Name: "old-name", Nested: nested{Value: "old-item"}},
			},
			Bytes: []byte("old-bytes"),
			Labels: map[string]nested{
				"existing": {Value: "old-label"},
			},
		}
		patch := &sample{
			Scalar: "new",
			Ptr:    &nested{Value: "new-ptr"},
			Items: []item{
				{Name: "new-name", Nested: nested{Value: "new-item"}},
				{Name: "appended", Nested: nested{Value: "new-extra"}},
			},
			Bytes: []byte("new-bytes"),
			Labels: map[string]nested{
				"added": {Value: "new-label"},
			},
		}

		sparseFuzzInto(dst, patch)

		want := &sample{
			Scalar: "new",
			Ptr:    &nested{Value: "new-ptr"},
			Items: []item{
				{Name: "new-name", Nested: nested{Value: "new-item"}},
				{Name: "appended", Nested: nested{Value: "new-extra"}},
			},
			Bytes: []byte("new-bytes"),
			Labels: map[string]nested{
				"existing": {Value: "new-label"},
				"added":    {Value: "new-label"},
			},
		}
		if diff := cmp.Diff(want, dst); diff != "" {
			t.Fatalf("sparseFuzzInto mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("does not delete reserved annotations", func(t *testing.T) {
		dst := &sample{
			Annotations: map[string]string{
				"nvidia.com/dgd-preserved": "keep",
				"user":                     "old",
			},
		}
		patch := &sample{
			Annotations: map[string]string{
				"added": "new",
			},
		}

		sparseFuzzInto(dst, patch)

		want := map[string]string{
			"nvidia.com/dgd-preserved": "keep",
			"user":                     "new",
			"added":                    "new",
		}
		if diff := cmp.Diff(want, dst.Annotations); diff != "" {
			t.Fatalf("Annotations mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil and empty patches do not clear existing fields", func(t *testing.T) {
		dst := &sample{
			Ptr:   &nested{Value: "keep-ptr"},
			Items: []item{{Name: "keep-item"}},
			Annotations: map[string]string{
				"user": "keep",
			},
		}
		patch := &sample{
			Ptr:         nil,
			Items:       nil,
			Annotations: map[string]string{},
		}

		sparseFuzzInto(dst, patch)

		want := &sample{
			Ptr:   &nested{Value: "keep-ptr"},
			Items: []item{{Name: "keep-item"}},
			Annotations: map[string]string{
				"user": "keep",
			},
		}
		if diff := cmp.Diff(want, dst); diff != "" {
			t.Fatalf("sparseFuzzInto mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("deep copies appended values", func(t *testing.T) {
		dst := &sample{}
		patch := &sample{
			Items: []item{{Name: "copied"}},
			Bytes: []byte("copied"),
			Labels: map[string]nested{
				"copied": {Value: "before"},
			},
		}

		sparseFuzzInto(dst, patch)
		patch.Items[0].Name = "mutated"
		patch.Bytes[0] = 'm'
		patch.Labels["copied"] = nested{Value: "after"}

		if got := dst.Items[0].Name; got != "copied" {
			t.Fatalf("copied item name = %q, want copied", got)
		}
		if got := string(dst.Bytes); got != "copied" {
			t.Fatalf("copied bytes = %q, want copied", got)
		}
		if got := dst.Labels["copied"].Value; got != "before" {
			t.Fatalf("copied label = %q, want before", got)
		}
	})
}

func comparableRoundTripObject(obj runtime.Object) runtime.Object {
	out := obj.DeepCopyObject()
	meta, ok := out.(metav1.Object)
	if !ok {
		panic(fmt.Sprintf("%T does not implement metav1.Object", out))
	}
	meta.SetAnnotations(scrubReservedAnnotations(maps.Clone(meta.GetAnnotations())))
	normalizeComparableRoundTripValue(reflect.ValueOf(out))
	return out
}

// sparseFuzzInto applies a generated patch without clearing existing values or
// reserved preservation annotations. Unlike mergo-style value merging, this is
// a fuzz mutator: it recursively edits existing objects, including slice items,
// so stale annotation overlays are exercised on deep fields.
func sparseFuzzInto(dst, patch any) {
	sparseMergeValue(reflect.ValueOf(dst), reflect.ValueOf(patch))
}

func sparseMergeValue(dst, patch reflect.Value) {
	if !dst.IsValid() || !patch.IsValid() || dst.Type() != patch.Type() {
		return
	}

	switch dst.Kind() {
	case reflect.Pointer:
		sparseMergePointer(dst, patch)
	case reflect.Interface:
		sparseMergeInterface(dst, patch)
	case reflect.Struct:
		sparseMergeStruct(dst, patch)
	case reflect.Slice:
		sparseMergeSlice(dst, patch)
	case reflect.Array:
		sparseMergeArray(dst, patch)
	case reflect.Map:
		sparseMergeMap(dst, patch)
	default:
		if dst.CanSet() {
			dst.Set(patch)
		}
	}
}

func sparseMergePointer(dst, patch reflect.Value) {
	if patch.IsNil() {
		return
	}
	if dst.IsNil() {
		if dst.CanSet() {
			dst.Set(deepCopyValue(patch))
		}
		return
	}
	sparseMergeValue(dst.Elem(), patch.Elem())
}

func sparseMergeInterface(dst, patch reflect.Value) {
	if patch.IsNil() {
		return
	}
	if dst.CanSet() {
		dst.Set(deepCopyValue(patch))
	}
}

func sparseMergeStruct(dst, patch reflect.Value) {
	if isAtomicSparseMergeType(dst.Type()) {
		if dst.CanSet() && !patch.IsZero() {
			dst.Set(deepCopyValue(patch))
		}
		return
	}
	if hasUnexportedFields(dst.Type()) {
		if dst.CanSet() && !patch.IsZero() {
			dst.Set(patch)
		}
		return
	}
	for i := 0; i < dst.NumField(); i++ {
		field := dst.Field(i)
		if !field.CanSet() {
			continue
		}
		sparseMergeValue(field, patch.Field(i))
	}
}

func sparseMergeSlice(dst, patch reflect.Value) {
	if patch.IsNil() || patch.Len() == 0 {
		return
	}
	if dst.Type().Elem().Kind() == reflect.Uint8 {
		if dst.CanSet() {
			dst.Set(deepCopyValue(patch))
		}
		return
	}
	if dst.IsNil() || dst.Len() == 0 {
		if dst.CanSet() {
			dst.Set(deepCopyValue(patch))
		}
		return
	}
	for i := 0; i < min(dst.Len(), patch.Len()); i++ {
		sparseMergeValue(dst.Index(i), patch.Index(i))
	}
	if patch.Len() > dst.Len() && dst.CanSet() {
		for i := dst.Len(); i < patch.Len(); i++ {
			dst.Set(reflect.Append(dst, deepCopyValue(patch.Index(i))))
		}
	}
}

func sparseMergeArray(dst, patch reflect.Value) {
	for i := 0; i < min(dst.Len(), patch.Len()); i++ {
		sparseMergeValue(dst.Index(i), patch.Index(i))
	}
}

func sparseMergeMap(dst, patch reflect.Value) {
	if patch.IsNil() || patch.Len() == 0 {
		return
	}
	if dst.IsNil() {
		if !dst.CanSet() {
			return
		}
		dst.Set(reflect.MakeMap(dst.Type()))
	}

	patchKeys := patch.MapKeys()
	for i, dstKey := range dst.MapKeys() {
		if isReservedAnnotationKeyValue(dstKey) {
			continue
		}
		patchValue := patch.MapIndex(patchKeys[i%len(patchKeys)])
		mergedValue := deepCopyValue(dst.MapIndex(dstKey))
		sparseMergeValue(mergedValue, patchValue)
		dst.SetMapIndex(dstKey, mergedValue)
	}
	for _, patchKey := range patchKeys {
		if isReservedAnnotationKeyValue(patchKey) || dst.MapIndex(patchKey).IsValid() {
			continue
		}
		dst.SetMapIndex(patchKey, deepCopyValue(patch.MapIndex(patchKey)))
	}
}

func deepCopyValue(v reflect.Value) reflect.Value {
	out := reflect.New(v.Type()).Elem()
	if !v.IsValid() {
		return out
	}

	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return out
		}
		cp := reflect.New(v.Type().Elem())
		cp.Elem().Set(deepCopyValue(v.Elem()))
		out.Set(cp)
	case reflect.Interface:
		if v.IsNil() {
			return out
		}
		out.Set(deepCopyValue(v.Elem()))
	case reflect.Slice:
		if v.IsNil() {
			return out
		}
		cp := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			cp.Index(i).Set(deepCopyValue(v.Index(i)))
		}
		out.Set(cp)
	case reflect.Map:
		if v.IsNil() {
			return out
		}
		cp := reflect.MakeMapWithSize(v.Type(), v.Len())
		for _, key := range v.MapKeys() {
			cp.SetMapIndex(key, deepCopyValue(v.MapIndex(key)))
		}
		out.Set(cp)
	default:
		out.Set(v)
	}
	return out
}

func isAtomicSparseMergeType(t reflect.Type) bool {
	return t == reflect.TypeOf(intstr.IntOrString{})
}

func normalizeComparableRoundTripValue(v reflect.Value) {
	if !v.IsValid() {
		return
	}
	if v.Type() == reflect.TypeOf(intstr.IntOrString{}) {
		if !v.CanSet() {
			return
		}
		ios := v.Interface().(intstr.IntOrString)
		if ios.Type == intstr.String {
			ios.IntVal = 0
		} else {
			ios.StrVal = ""
		}
		v.Set(reflect.ValueOf(ios))
		return
	}
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			normalizeComparableRoundTripValue(v.Elem())
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			field := v.Field(i)
			if field.CanSet() || field.Kind() == reflect.Pointer || field.Kind() == reflect.Interface {
				normalizeComparableRoundTripValue(field)
			}
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			normalizeComparableRoundTripValue(v.Index(i))
		}
	case reflect.Map:
		for _, key := range v.MapKeys() {
			value := deepCopyValue(v.MapIndex(key))
			normalizeComparableRoundTripValue(value)
			v.SetMapIndex(key, value)
		}
	}
}

func hasUnexportedFields(t reflect.Type) bool {
	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).PkgPath != "" {
			return true
		}
	}
	return false
}

func isReservedAnnotationKeyValue(v reflect.Value) bool {
	if v.Kind() != reflect.String {
		return false
	}
	key := v.String()
	for _, prefix := range reservedAnnotationPrefixes {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}
