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

package dgdrinput

import (
	"fmt"
	"strings"
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestValidateSpecAcceptsDefaultedRequest(t *testing.T) {
	if errs := ValidateSpec(defaulted(), DefaultOptions()); len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
}

func TestValidateSpecReportsUnresolvedHardware(t *testing.T) {
	spec := defaulted()
	spec.Hardware = nil
	errs := ValidateSpec(spec, DefaultOptions())
	if len(errs) == 0 || !strings.Contains(fmt.Sprint(errs), "hardware") {
		t.Fatalf("want a hardware error, got %v", errs)
	}
}

func TestValidateSpecReportsConflictAndMergedInputErrors(t *testing.T) {
	spec := defaulted()
	spec.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{
		Raw: []byte(`{"goal":{"target":"ttft"},"workload":{"concurrency":4}}`),
	}}
	errs := ValidateSpec(spec, DefaultOptions())
	// Typed metric ThroughputPerGPU disagrees with the native ttft target, and the native
	// concurrency adds a second load mode next to the typed KV load ratio.
	if len(errs) < 2 {
		t.Fatalf("want conflict and workload errors, got %v", errs)
	}
}

func TestValidateSpecRejectsMalformedParameters(t *testing.T) {
	spec := defaulted()
	spec.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(`[1]`)}}
	errs := ValidateSpec(spec, DefaultOptions())
	if len(errs) != 1 || errs[0].Field != "spec.aisimulate.parameters" {
		t.Fatalf("got %v", errs)
	}
}

func TestNewViolationsIgnoresPreexistingErrors(t *testing.T) {
	bad := defaulted()
	bad.Hardware = nil
	old := ValidateSpec(bad, DefaultOptions())
	if len(old) == 0 {
		t.Fatal("setup: expected a violation")
	}
	// Same violation on update: nothing new.
	if got := NewViolations(ValidateSpec(bad, DefaultOptions()), old); len(got) != 0 {
		t.Fatalf("preexisting violation must not block: %v", got)
	}
	// A different, newly introduced violation is still reported.
	worse := defaulted()
	worse.Hardware = nil
	worse.Backends = []v1beta2.Backend{"nope"}
	got := NewViolations(ValidateSpec(worse, DefaultOptions()), old)
	if len(got) == 0 {
		t.Fatal("new violation must be reported")
	}
	// A fresh create against no previous errors reports everything.
	if got := NewViolations(old, nil); len(got) != len(old) {
		t.Fatalf("got %d want %d", len(got), len(old))
	}
}
