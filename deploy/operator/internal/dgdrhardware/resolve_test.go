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

package dgdrhardware

import (
	"context"
	"errors"
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
)

type fakeDiscoverer struct {
	found   Discovered
	err     error
	calls   int
	filters []string
}

func (f *fakeDiscoverer) Discover(_ context.Context, filterSKU string) (Discovered, error) {
	f.calls++
	f.filters = append(f.filters, filterSKU)
	return f.found, f.err
}

func ptr[T any](v T) *T { return &v }

func spec(sku *string, budget *int32) *v1beta2.DynamoGraphDeploymentRequestSpec {
	return &v1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    v1beta2.ModelSpec{Name: "m"},
		Hardware: &v1beta2.HardwareSpec{GPU: &v1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}},
	}
}

func TestResolve(t *testing.T) {
	cluster := Discovered{SKU: "h100_sxm", GPUs: 16}
	cases := []struct {
		name       string
		spec       *v1beta2.DynamoGraphDeploymentRequestSpec
		found      Discovered
		want       Defaults
		wantCalls  int
		wantFilter string
	}{
		{"nothing set", spec(nil, nil), cluster, Defaults{SKU: "h100_sxm", Budget: 16}, 1, ""},
		{"no hardware at all", &v1beta2.DynamoGraphDeploymentRequestSpec{}, cluster, Defaults{SKU: "h100_sxm", Budget: 16}, 1, ""},
		{"sku set, budget missing", spec(ptr("h200_sxm"), nil), cluster, Defaults{Budget: 16}, 1, "h200_sxm"},
		{"budget set, sku missing", spec(nil, ptr(int32(4))), cluster, Defaults{SKU: "h100_sxm"}, 1, ""},
		{"both set", spec(ptr("h200_sxm"), ptr(int32(4))), cluster, Defaults{}, 0, ""},
		{"budget is capped", spec(nil, nil), Discovered{SKU: "h100_sxm", GPUs: 64}, Defaults{SKU: "h100_sxm", Budget: 32}, 1, ""},
		{"unknown model leaves the sku", spec(nil, nil), Discovered{GPUs: 8}, Defaults{Budget: 8}, 1, ""},
		{"no gpus leaves the budget", spec(ptr("h100_sxm"), nil), Discovered{SKU: "h100_sxm"}, Defaults{}, 1, "h100_sxm"},
		{"empty sku counts as missing", spec(ptr(""), ptr(int32(2))), cluster, Defaults{SKU: "h100_sxm"}, 1, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			discoverer := &fakeDiscoverer{found: tc.found}
			got, err := Resolve(context.Background(), discoverer, tc.spec, Options{})
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("got %+v want %+v", got, tc.want)
			}
			if discoverer.calls != tc.wantCalls {
				t.Fatalf("discovery calls: got %d want %d", discoverer.calls, tc.wantCalls)
			}
			if tc.wantCalls > 0 && discoverer.filters[0] != tc.wantFilter {
				t.Fatalf("filter: got %q want %q", discoverer.filters[0], tc.wantFilter)
			}
		})
	}
}

func TestResolveHonorsNativeHardware(t *testing.T) {
	s := spec(nil, nil)
	s.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{
		Raw: []byte(`{"search_space":{"hardware_sku":"b200_sxm"}}`),
	}}
	discoverer := &fakeDiscoverer{found: Discovered{SKU: "h100_sxm", GPUs: 8}}
	got, err := Resolve(context.Background(), discoverer, s, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if got != (Defaults{Budget: 8}) {
		t.Fatalf("native SKU must be kept, got %+v", got)
	}
	if discoverer.filters[0] != "b200_sxm" {
		t.Fatalf("discovery should filter by the native SKU, got %q", discoverer.filters[0])
	}
}

func TestResolveMaxBudgetOption(t *testing.T) {
	got, err := Resolve(context.Background(), &fakeDiscoverer{found: Discovered{SKU: "x", GPUs: 20}}, spec(nil, nil), Options{MaxBudget: 8})
	if err != nil || got.Budget != 8 {
		t.Fatalf("got %+v %v", got, err)
	}
}

func TestResolveErrors(t *testing.T) {
	boom := errors.New("dcgm unreachable")
	if _, err := Resolve(context.Background(), &fakeDiscoverer{err: boom}, spec(nil, nil), Options{}); !errors.Is(err, boom) {
		t.Fatalf("discovery error not returned: %v", err)
	}
	if _, err := Resolve(context.Background(), nil, spec(nil, nil), Options{}); err == nil {
		t.Fatal("a nil discoverer must be an error")
	}
	bad := spec(nil, nil)
	bad.AISimulate = &v1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(`[1]`)}}
	if _, err := Resolve(context.Background(), &fakeDiscoverer{}, bad, Options{}); err == nil {
		t.Fatal("malformed parameters must be an error")
	}
	// Complete hardware never consults the discoverer, so a nil one is fine.
	if got, err := Resolve(context.Background(), nil, spec(ptr("a"), ptr(int32(1))), Options{}); err != nil || got != (Defaults{}) {
		t.Fatalf("complete hardware: %+v %v", got, err)
	}
}

func TestApply(t *testing.T) {
	t.Run("fills only what is missing", func(t *testing.T) {
		s := spec(ptr("h200_sxm"), nil)
		Apply(s, Defaults{SKU: "h100_sxm", Budget: 16})
		if *s.Hardware.GPU.SKU != "h200_sxm" || *s.Hardware.GPU.Budget != 16 {
			t.Fatalf("got %s %d", *s.Hardware.GPU.SKU, *s.Hardware.GPU.Budget)
		}
	})
	t.Run("creates the nested objects", func(t *testing.T) {
		s := &v1beta2.DynamoGraphDeploymentRequestSpec{}
		Apply(s, Defaults{SKU: "h100_sxm", Budget: 8})
		if s.Hardware == nil || s.Hardware.GPU == nil || *s.Hardware.GPU.SKU != "h100_sxm" || *s.Hardware.GPU.Budget != 8 {
			t.Fatalf("got %+v", s.Hardware)
		}
	})
	t.Run("zero defaults leave the spec alone", func(t *testing.T) {
		s := &v1beta2.DynamoGraphDeploymentRequestSpec{}
		Apply(s, Defaults{})
		if s.Hardware != nil {
			t.Fatal("created hardware for empty defaults")
		}
	})
}
