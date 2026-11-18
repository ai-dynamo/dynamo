/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
)

func TestDynamoGraphDeploymentRequestInputOverridesApplyAtomically(t *testing.T) {
	env := admissionEnv.ForTest(t)
	admissionGate.set(features.Gates{GPUDiscovery: true})
	setAdmissionH100Topology()
	fixture := beta2SchemaDGDRForAdmission(`{"engine":{"mode":"aggregated","workers":{"prefill":null,"decode":null}},"optimizer":{"max_trials":4}}`, nil)
	resourceClient := newAdmissionResourceClient(t, env, fixture, "", &warningRecorder{})
	object, _ := admissionObject(t, fixture, env.Namespace(), nil)
	payload, err := json.Marshal(object.Object)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Apply the complete patch document and persist its literal nested nulls")
	created, err := resourceClient.Patch(t.Context(), object.GetName(), types.ApplyPatchType, payload, metav1.PatchOptions{FieldManager: "search-author"})
	if err != nil {
		t.Fatalf("apply the initial request: %v", err)
	}
	created, err = resourceClient.Get(t.Context(), created.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read the persisted request: %v", err)
	}
	for _, role := range []string{"prefill", "decode"} {
		value, found, err := unstructured.NestedFieldNoCopy(created.Object, "spec", "aisim", "inputOverrides", "engine", "workers", role)
		if err != nil || !found || value != nil {
			t.Fatalf("stored %s deletion marker = %#v, found=%v err=%v", role, value, found, err)
		}
	}

	t.Log("A different manager cannot independently own a disjoint part of the patch")
	replacement := map[string]any{"optimizer": map[string]any{"parallelism": int64(2)}}
	patch := map[string]any{
		"apiVersion": "nvidia.com/v1beta2", "kind": "DynamoGraphDeploymentRequest",
		"metadata": map[string]any{"name": object.GetName(), "namespace": env.Namespace()},
		"spec":     map[string]any{"aisim": map[string]any{"inputOverrides": replacement}},
	}
	payload, err = json.Marshal(patch)
	if err != nil {
		t.Fatal(err)
	}
	_, err = resourceClient.Patch(t.Context(), object.GetName(), types.ApplyPatchType, payload, metav1.PatchOptions{FieldManager: "search-editor"})
	if !apierrors.IsConflict(err) || !strings.Contains(err.Error(), ".spec.aisim.inputOverrides") {
		t.Fatalf("disjoint patch returned %v, want an inputOverrides ownership conflict", err)
	}

	t.Log("Forced takeover replaces the entire document rather than retaining old subtrees")
	updated, err := resourceClient.Patch(t.Context(), object.GetName(), types.ApplyPatchType, payload, metav1.PatchOptions{FieldManager: "search-editor", Force: new(true)})
	if err != nil {
		t.Fatalf("take over the patch document: %v", err)
	}
	got, found, err := unstructured.NestedMap(updated.Object, "spec", "aisim", "inputOverrides")
	if err != nil || !found || !reflect.DeepEqual(got, replacement) {
		t.Fatalf("inputOverrides = %#v, found=%v err=%v, want %#v", got, found, err, replacement)
	}
}
