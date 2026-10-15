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

package defaulting

import (
	"context"
	"errors"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrhardware"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
	admissionv1 "k8s.io/api/admission/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

type fakeHardware struct {
	found   dgdrhardware.Discovered
	err     error
	calls   int
	filters []string
}

func (f *fakeHardware) Discover(_ context.Context, filterSKU string) (dgdrhardware.Discovered, error) {
	f.calls++
	f.filters = append(f.filters, filterSKU)
	return f.found, f.err
}

func v1b2Ptr[T any](v T) *T { return &v }

// modelOnly is a model-only Request after the CRD's structural defaults.
func modelOnly() *nvidiacomv1beta2.DynamoGraphDeploymentRequest {
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequest{
		Spec: nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
			Model:    nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
			Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
			Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
				ISL: v1b2Ptr(int32(4000)), OSL: v1b2Ptr(int32(1000)), KVLoadRatio: nvidiacomv1beta2.KVLoadRatio{0, 1},
			}},
			Objective: &nvidiacomv1beta2.ObjectiveSpec{
				Mode: nvidiacomv1beta2.ObjectiveModeOptimize,
				SLA:  &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: v1b2Ptr(2000.0), ITLMs: v1b2Ptr(30.0)},
			},
			Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
		},
	}
}

func admissionContext(op admissionv1.Operation) context.Context {
	gvk := nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK
	return admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: op,
			Kind:      metav1.GroupVersionKind{Group: gvk.Group, Version: gvk.Version, Kind: gvk.Kind},
		},
	})
}

func newDefaulter(hardware dgdrhardware.Discoverer) *DGDRV1Beta2Defaulter {
	return NewDGDRV1Beta2Defaulter(DGDRV1Beta2DefaulterOptions{
		Hardware:        hardware,
		AISimulateImage: "registry.example.com/aisimulate-runner:0.13.0",
	})
}

func TestV1Beta2DefaulterFillsAModelOnlyRequest(t *testing.T) {
	hardware := &fakeHardware{found: dgdrhardware.Discovered{SKU: "h200_sxm", GPUs: 64}}
	dgdr := modelOnly()
	if err := newDefaulter(hardware).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}

	spec := dgdr.Spec
	if spec.Objective.Metric == nil || *spec.Objective.Metric != "ThroughputPerGPU" {
		t.Fatalf("metric not defaulted: %v", spec.Objective.Metric)
	}
	if spec.AISimulate == nil || spec.AISimulate.Image != "registry.example.com/aisimulate-runner:0.13.0" {
		t.Fatalf("image not defaulted: %+v", spec.AISimulate)
	}
	gpuSpec := spec.Hardware.GPU
	if *gpuSpec.SKU != "h200_sxm" || *gpuSpec.Budget != 32 {
		t.Fatalf("hardware not defaulted (budget capped at 32): %s %d", *gpuSpec.SKU, *gpuSpec.Budget)
	}

	input, err := dgdrinput.Build(&spec, dgdrinput.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if errs := dgdrinput.Validate(input); len(errs) > 0 {
		t.Fatalf("a defaulted model-only Request must validate: %v", errs)
	}
}

func TestV1Beta2DefaulterKeepsExplicitValues(t *testing.T) {
	hardware := &fakeHardware{found: dgdrhardware.Discovered{SKU: "h100_sxm", GPUs: 8}}
	dgdr := modelOnly()
	dgdr.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1b2Ptr("b200_sxm"), Budget: v1b2Ptr(int32(4))}}
	dgdr.Spec.Objective.Metric = v1b2Ptr(nvidiacomv1beta2.ObjectiveMetric("TTFT"))
	dgdr.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Image: "my/image:1.0.0"}

	if err := newDefaulter(hardware).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}
	if hardware.calls != 0 {
		t.Fatal("discovered hardware although it was fully specified")
	}
	if *dgdr.Spec.Hardware.GPU.SKU != "b200_sxm" || *dgdr.Spec.Hardware.GPU.Budget != 4 ||
		*dgdr.Spec.Objective.Metric != "TTFT" || dgdr.Spec.AISimulate.Image != "my/image:1.0.0" {
		t.Fatalf("explicit values were overwritten: %+v", dgdr.Spec)
	}
}

func TestV1Beta2DefaulterDiscoversOnlyWhatIsMissing(t *testing.T) {
	hardware := &fakeHardware{found: dgdrhardware.Discovered{SKU: "h100_sxm", GPUs: 16}}
	dgdr := modelOnly()
	dgdr.Spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1b2Ptr("h200_sxm")}}
	if err := newDefaulter(hardware).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}
	if hardware.filters[0] != "h200_sxm" {
		t.Fatalf("discovery must be restricted to the requested SKU, got %q", hardware.filters[0])
	}
	if *dgdr.Spec.Hardware.GPU.SKU != "h200_sxm" || *dgdr.Spec.Hardware.GPU.Budget != 16 {
		t.Fatalf("got %s %d", *dgdr.Spec.Hardware.GPU.SKU, *dgdr.Spec.Hardware.GPU.Budget)
	}
}

func TestV1Beta2DefaulterLeavesTheTargetToTheNativeInput(t *testing.T) {
	dgdr := modelOnly()
	dgdr.Spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{
		Parameters: &runtime.RawExtension{Raw: []byte(`{"goal":{"target":"ttft"}}`)},
	}
	if err := newDefaulter(nil).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}
	if dgdr.Spec.Objective.Metric != nil {
		t.Fatalf("defaulted a metric next to goal.target: %v", *dgdr.Spec.Objective.Metric)
	}
	if dgdr.Spec.AISimulate.Image == "" {
		t.Fatal("the image should still be defaulted next to native parameters")
	}
}

func TestV1Beta2DefaulterDiscoveryFailureIsNotAnAdmissionFailure(t *testing.T) {
	dgdr := modelOnly()
	hardware := &fakeHardware{err: errors.New("dcgm unreachable")}
	if err := newDefaulter(hardware).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatalf("a discovery failure must not fail admission: %v", err)
	}
	if dgdr.Spec.Hardware != nil {
		t.Fatalf("hardware set despite the failure: %+v", dgdr.Spec.Hardware)
	}
	if dgdr.Spec.Objective.Metric == nil {
		t.Fatal("the metric default must not depend on discovery")
	}
}

func TestV1Beta2DefaulterWithoutDiscovery(t *testing.T) {
	dgdr := modelOnly()
	if err := newDefaulter(nil).Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}
	if dgdr.Spec.Hardware != nil {
		t.Fatal("hardware set without a discoverer")
	}
}

func TestV1Beta2DefaulterAdmitsUpdatesUnchanged(t *testing.T) {
	hardware := &fakeHardware{found: dgdrhardware.Discovered{SKU: "h100_sxm", GPUs: 8}}
	dgdr := modelOnly()
	if err := newDefaulter(hardware).Default(admissionContext(admissionv1.Update), dgdr); err != nil {
		t.Fatal(err)
	}
	if hardware.calls != 0 || dgdr.Spec.Objective.Metric != nil || dgdr.Spec.AISimulate != nil {
		t.Fatalf("an update was defaulted: %+v", dgdr.Spec)
	}
}

func TestV1Beta2DefaulterWithoutConfiguredImage(t *testing.T) {
	dgdr := modelOnly()
	defaulter := NewDGDRV1Beta2Defaulter(DGDRV1Beta2DefaulterOptions{})
	if err := defaulter.Default(admissionContext(admissionv1.Create), dgdr); err != nil {
		t.Fatal(err)
	}
	if dgdr.Spec.AISimulate != nil {
		t.Fatalf("created aisimulate without an image to put in it: %+v", dgdr.Spec.AISimulate)
	}
}

func TestV1Beta2DefaulterRejectsOtherVersions(t *testing.T) {
	ctx := admission.NewContextWithRequest(context.Background(), admission.Request{
		AdmissionRequest: admissionv1.AdmissionRequest{
			Operation: admissionv1.Create,
			Kind:      metav1.GroupVersionKind{Group: "nvidia.com", Version: "v1beta1", Kind: "DynamoGraphDeploymentRequest"},
		},
	})
	if err := newDefaulter(nil).Default(ctx, modelOnly()); err == nil {
		t.Fatal("accepted a v1beta1 admission request")
	}
	if err := newDefaulter(nil).Default(admissionContext(admissionv1.Create), &runtime.Unknown{}); err == nil {
		t.Fatal("accepted a foreign object")
	}
}
