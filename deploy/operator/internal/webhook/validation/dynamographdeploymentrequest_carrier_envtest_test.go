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

package validation_test

import (
	"bytes"
	"encoding/json"
	"sort"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
)

// A request created through its owning API version has to stay writable through every other
// served version. A metadata-only patch through the foreign view must not be rejected, and it
// must neither rewrite the spec nor bump the generation. A default that materialises a native
// field next to a conversion carrier breaks this, so it is covered at the API-server boundary.
func TestDynamoGraphDeploymentRequestCarrierObjectsStayWritable(t *testing.T) {
	t.Run("native v1beta1 written through v1beta2", func(t *testing.T) {
		carrierRoundTrip(t, betaDGDRForAdmission(nil), beta2DGDRForAdmission(nil), "v1beta1")
	})
	t.Run("native v1beta2 written through v1beta1", func(t *testing.T) {
		carrierRoundTrip(t, beta2DGDRForAdmission(setBeta2AdmissionHardware), betaDGDRForAdmission(nil), "v1beta2")
	})
	t.Run("native v1beta2 written through v1alpha1", func(t *testing.T) {
		carrierRoundTrip(t, beta2DGDRForAdmission(setBeta2AdmissionHardware), alphaDGDRForAdmission(nil), "v1beta2")
	})
}

func carrierRoundTrip(t *testing.T, ownerFixture, otherFixture runtime.Object, carrierKey string) {
	t.Helper()
	env := admissionEnv.ForTest(t)
	admissionGate.set(features.Gates{GPUDiscovery: true})
	setAdmissionH100Topology()
	warnings := &warningRecorder{}
	ownerClient := newAdmissionResourceClient(t, env, ownerFixture, "", warnings)
	otherClient := newAdmissionResourceClient(t, env, otherFixture, "", warnings)

	t.Log("Create the request through the version that owns its representation")
	object, _ := admissionObject(t, ownerFixture, env.Namespace(), nil)
	created, err := ownerClient.Create(t.Context(), object, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create through the owning version: %v", err)
	}
	name := created.GetName()

	t.Log("Read it through the other version and expect the conversion carrier")
	foreign, err := otherClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read through the other version: %v", err)
	}
	spec, _ := foreign.Object["spec"].(map[string]any)
	if _, ok := spec[carrierKey]; !ok {
		keys := make([]string, 0, len(spec))
		for key := range spec {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		t.Fatalf("foreign view has spec keys %v, want the %s carrier", keys, carrierKey)
	}

	t.Log("Patch metadata through each version")
	for label, client := range map[string]dynamic.ResourceInterface{"foreign": otherClient, "owner": ownerClient} {
		patch := []byte(`{"metadata":{"labels":{"carrier-probe-` + label + `":"1"}}}`)
		if _, err := client.Patch(t.Context(), name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
			t.Fatalf("metadata-only patch through the %s version: %v", label, err)
		}
	}

	t.Log("The spec and generation must be untouched")
	after, err := ownerClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read back through the owning version: %v", err)
	}
	if after.GetGeneration() != created.GetGeneration() {
		t.Fatalf("generation = %d, want %d: a metadata-only patch rewrote the spec", after.GetGeneration(), created.GetGeneration())
	}

	t.Log("Write the unchanged foreign view back with a full update; the spec must be byte-stable")
	beforeSpec, err := json.Marshal(foreign.Object["spec"])
	if err != nil {
		t.Fatalf("encode the foreign spec: %v", err)
	}
	current, err := otherClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read the foreign view before the update: %v", err)
	}
	updated, err := otherClient.Update(t.Context(), current, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("full update of the unchanged foreign view: %v", err)
	}
	afterSpec, err := json.Marshal(updated.Object["spec"])
	if err != nil {
		t.Fatalf("encode the updated spec: %v", err)
	}
	if !bytes.Equal(beforeSpec, afterSpec) {
		t.Fatalf("the foreign spec changed across an unchanged update:\nbefore: %s\nafter:  %s", beforeSpec, afterSpec)
	}
	if updated.GetGeneration() != created.GetGeneration() {
		t.Fatalf("generation = %d, want %d: an unchanged update rewrote the spec", updated.GetGeneration(), created.GetGeneration())
	}
}
