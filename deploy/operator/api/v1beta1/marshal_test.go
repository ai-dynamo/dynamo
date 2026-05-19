/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1beta1

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// unmarshalToMap is a small helper so the assertions below read against the
// actual wire shape rather than Go struct re-decodes (which would silently
// re-introduce the artefacts the normalizer just removed).
func unmarshalToMap(t *testing.T, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal json to map: %v", err)
	}
	return m
}

type embeddedJSONCollision struct {
	Same *struct{} `json:"same,omitempty"`
}

type outerJSONCollision struct {
	Same struct{} `json:"same,omitempty"`
	embeddedJSONCollision
}

func TestNormalizeJSON_OuterFieldWinsEmbeddedCollision(t *testing.T) {
	normalized, err := normalizeV1Beta1JSON([]byte(`{"same":{}}`), reflect.TypeOf(outerJSONCollision{}))
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if string(normalized) != "{}" {
		t.Fatalf("expected outer value-typed field to win collision and be stripped, got %s", string(normalized))
	}
}

// TestDGDMarshal_StripsEmptyPodTemplateMetadata locks in the fix for the
// kubectl-apply generation-bump regression: when the stored v1alpha1 object
// has no ExtraPodMetadata, the v1beta1 projection built by buildPodTemplateTo
// contains a zero-valued corev1.PodTemplateSpec.ObjectMeta, which vanilla
// Go json.Marshal would render as `"metadata": {}`. That artefact desyncs
// kubectl's last-applied-configuration diff from the live view and triggers
// a .metadata.generation bump on every `kubectl apply` of an unchanged
// manifest. After MarshalJSON normalization the key must be absent.
func TestDGDMarshal_StripsEmptyPodTemplateMetadata(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "nvidia.com/v1beta1",
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "sglang-agg", Namespace: "jsm"},
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "decode",
					ComponentType: ComponentType("worker"),
					PodTemplate: &corev1.PodTemplateSpec{
						// Deliberately no ObjectMeta fields set: this
						// mirrors the ConvertTo output for a stored
						// v1alpha1 object with no ExtraPodMetadata.
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "main",
									Image: "my-registry/sglang-runtime:my-tag",
									Resources: corev1.ResourceRequirements{
										Limits: corev1.ResourceList{
											"nvidia.com/gpu": resource.MustParse("1"),
										},
									},
								},
							},
						},
					}},
			},
		},
	}

	b, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("marshaled: %s", string(b))

	m := unmarshalToMap(t, b)
	svcs := m["spec"].(map[string]any)["components"].([]any)
	pt := svcs[0].(map[string]any)["podTemplate"].(map[string]any)
	if _, ok := pt["metadata"]; ok {
		t.Errorf("expected podTemplate.metadata to be absent after normalization, got: %v", pt["metadata"])
	}
}

// TestDGDMarshal_PreservesNonEmptyPodTemplateMetadata verifies that the
// normalizer only drops the empty-map case. A user who sets labels or
// annotations on podTemplate.metadata must still see that metadata in the
// serialized v1beta1 object, otherwise the operator would silently drop user
// intent through the webhook.
func TestDGDMarshal_PreservesNonEmptyPodTemplateMetadata(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "decode",
					PodTemplate: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{"k": "v"},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{Name: "main", Image: "x"}},
						},
					}},
			},
		},
	}
	b, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	pt := m["spec"].(map[string]any)["components"].([]any)[0].(map[string]any)["podTemplate"].(map[string]any)
	md, ok := pt["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("expected podTemplate.metadata to be preserved when user set labels, got: %v", pt["metadata"])
	}
	labels, ok := md["labels"].(map[string]any)
	if !ok || labels["k"] != "v" {
		t.Errorf("expected labels.k == v, got %v", md["labels"])
	}
}

// TestDGDMarshal_StripsEmptyContainerResources covers the second source of
// Go-injected empty-map noise: corev1.Container.Resources is value-typed, so
// every container without explicit requests/limits picks up a `"resources":
// {}` in the marshaled output. Same generation-bump mechanism as
// podTemplate.metadata; same fix.
func TestDGDMarshal_StripsEmptyContainerResources(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "frontend",
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							InitContainers: []corev1.Container{
								{Name: "wait", Image: "busybox"},
							},
							Containers: []corev1.Container{
								{Name: "main", Image: "frontend"},
							},
						},
					}},
			},
		},
	}
	b, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	spec := m["spec"].(map[string]any)["components"].([]any)[0].(map[string]any)["podTemplate"].(map[string]any)["spec"].(map[string]any)
	if c0, ok := spec["containers"].([]any)[0].(map[string]any); ok {
		if _, has := c0["resources"]; has {
			t.Errorf("expected empty resources to be stripped on containers[0], got: %v", c0["resources"])
		}
	}
	if c0, ok := spec["initContainers"].([]any)[0].(map[string]any); ok {
		if _, has := c0["resources"]; has {
			t.Errorf("expected empty resources to be stripped on initContainers[0], got: %v", c0["resources"])
		}
	}
}

// TestDGDMarshal_PreservesNonEmptyContainerResources ensures the resources
// pruner only drops `{}` and not maps with any of the known sub-fields set.
func TestDGDMarshal_PreservesNonEmptyContainerResources(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "main",
								Image: "x",
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										"nvidia.com/gpu": resource.MustParse("1"),
									},
								},
							}},
						},
					}},
			},
		},
	}
	b, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	c := m["spec"].(map[string]any)["components"].([]any)[0].(map[string]any)["podTemplate"].(map[string]any)["spec"].(map[string]any)["containers"].([]any)[0].(map[string]any)
	res, ok := c["resources"].(map[string]any)
	if !ok {
		t.Fatalf("expected resources to be preserved when limits are set, got: %v", c["resources"])
	}
	if limits, _ := res["limits"].(map[string]any); limits["nvidia.com/gpu"] != "1" {
		t.Errorf("limits[nvidia.com/gpu] = %v, want \"1\"", limits["nvidia.com/gpu"])
	}
}

// TestDCDMarshal_StripsEmptyPodTemplateMetadata mirrors the DGD test for the
// standalone DynamoComponentDeployment kind, which uses the same shared spec
// and therefore the same Go-encoder artefacts.
func TestDCDMarshal_StripsEmptyPodTemplateMetadata(t *testing.T) {
	dcd := &DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "nvidia.com/v1beta1",
			Kind:       "DynamoComponentDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "n"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "x",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
							},
						}},
					},
				},
			},
		},
	}
	b, err := json.Marshal(dcd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	pt := m["spec"].(map[string]any)["podTemplate"].(map[string]any)
	if _, ok := pt["metadata"]; ok {
		t.Errorf("expected DCD podTemplate.metadata to be absent, got: %v", pt["metadata"])
	}
}

// TestDGDMarshal_PreservesScalingAdapterSentinel is the regression test that
// protects presence-sensitive marker types from the normalization pass. The
// v1beta1 API uses a deliberately empty struct (`ScalingAdapter`) whose
// *presence* at `services[i].scalingAdapter: {}` opts a service into the
// DGDSA autoscaling path. Because the field is declared as a POINTER
// (`*ScalingAdapter` with `,omitempty`), Go's encoder already elides it when
// nil, so any `{}` that makes it to the wire was put there intentionally by
// the user -- stripping it would silently turn an opt-in into an opt-out.
// The reflection-driven stripper keys on "value-typed struct field" to
// distinguish encoder artefacts from sentinels; this test asserts that
// rule holds.
func TestDGDMarshal_PreservesScalingAdapterSentinel(t *testing.T) {
	dgd := &DynamoGraphDeployment{
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{{
				ComponentName:  "decode",
				ScalingAdapter: &ScalingAdapter{},
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "x"}}},
				}}},
		},
	}
	b, err := json.Marshal(dgd)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("marshaled: %s", string(b))
	m := unmarshalToMap(t, b)
	comp := m["spec"].(map[string]any)["components"].([]any)[0].(map[string]any)
	sa, ok := comp["scalingAdapter"]
	if !ok {
		t.Fatalf("expected scalingAdapter sentinel to be preserved, but field was stripped")
	}
	if saMap, isMap := sa.(map[string]any); !isMap || len(saMap) != 0 {
		t.Errorf("expected scalingAdapter to remain an empty object, got: %v", sa)
	}
}

// TestMarshal_RoundTrip confirms two properties of the custom MarshalJSON:
//
//  1. Semantic equality: after one normalization pass the Go object is
//     preserved at `cmp.Diff` resolution across an arbitrary number of
//     further marshal/unmarshal cycles -- the pruning happens only at the
//     JSON boundary, not in the Go types.
//  2. Byte-level idempotency: re-marshaling the canonical form yields the
//     same bytes. This is the property kubectl apply relies on -- if the
//     second marshal reshuffled or re-injected an artefact, the CSA diff
//     against `last-applied-configuration` would bump `.metadata.generation`
//     again. We assert the invariant directly so any future regression in
//     the normalizer shows up here rather than in a live cluster.
//
// The "canonicalization" pass is necessary because some field types carry
// optional encoder caches that only the encode/decode cycle settles (for
// example, `resource.Quantity` lazily materializes its canonical string on
// first Marshal). Without that pre-pass `cmp.Diff` would flag a benign
// cache-population difference.
func TestMarshal_RoundTrip(t *testing.T) {
	orig := &DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "agg",
			Namespace: "jsm",
			Labels:    map[string]string{"app": "dynamo"},
			Annotations: map[string]string{
				"kubectl.kubernetes.io/last-applied-configuration": "{}",
			},
		},
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "decode",
					ComponentType: ComponentType("worker"),
					Replicas:      ptrInt32(2),
					PodTemplate: &corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"role": "decode"}},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  "main",
								Image: "my-registry/sglang-runtime:my-tag",
								Env:   []corev1.EnvVar{{Name: "X", Value: "1"}},
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
								},
							}},
						},
					}},
				{
					ComponentName:  "prefill",
					ScalingAdapter: &ScalingAdapter{},
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "x"}}},
					}},
			},
		},
	}

	// First pass: settle encoder-side caches (resource.Quantity, etc.) so
	// that the subsequent equality check compares canonical forms.
	b1, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("first marshal: %v", err)
	}
	var canonical DynamoGraphDeployment
	if err := json.Unmarshal(b1, &canonical); err != nil {
		t.Fatalf("first unmarshal: %v", err)
	}

	b2, err := json.Marshal(&canonical)
	if err != nil {
		t.Fatalf("second marshal: %v", err)
	}
	var back DynamoGraphDeployment
	if err := json.Unmarshal(b2, &back); err != nil {
		t.Fatalf("second unmarshal: %v", err)
	}

	if diff := cmp.Diff(canonical, back); diff != "" {
		t.Errorf("round-trip drift in Go objects (-canonical +back):\n%s", diff)
	}
	if !bytes.Equal(b1, b2) {
		t.Errorf("marshal not idempotent -- this is the exact property kubectl apply depends on.\nfirst:  %s\nsecond: %s", b1, b2)
	}
}

func ptrInt32(v int32) *int32 { return &v }

// TestDGDListMarshal_StripsArtifactsOnItems proves that the value-receiver
// MarshalJSON is actually invoked on list items. `encoding/json` encodes
// slice elements via non-addressable reflect.Values; a pointer-receiver
// MarshalJSON would silently fall back to the default struct encoder for
// these elements and reintroduce `metadata: {}` on every listed object --
// exactly the bug the fix is designed to prevent, just on List reads
// instead of Get reads. If this test fails, check that MarshalJSON is
// declared with a VALUE receiver on DynamoGraphDeployment.
func TestDGDListMarshal_StripsArtifactsOnItems(t *testing.T) {
	list := &DynamoGraphDeploymentList{
		TypeMeta: metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeploymentList"},
		Items: []DynamoGraphDeployment{
			newDGDWithEmptyPodTemplateMetadata("agg-a"),
			newDGDWithEmptyPodTemplateMetadata("agg-b"),
		},
	}
	b, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	items, ok := m["items"].([]any)
	if !ok || len(items) != 2 {
		t.Fatalf("items missing or wrong length: %v", m["items"])
	}
	for i, raw := range items {
		pt := raw.(map[string]any)["spec"].(map[string]any)["components"].([]any)[0].(map[string]any)["podTemplate"].(map[string]any)
		if _, has := pt["metadata"]; has {
			t.Errorf("items[%d] podTemplate.metadata should be stripped, got: %v", i, pt["metadata"])
		}
	}
}

// TestDCDListMarshal_StripsArtifactsOnItems is the DCD counterpart to
// TestDGDListMarshal_StripsArtifactsOnItems. Same receiver-kind regression
// guard, different root type.
func TestDCDListMarshal_StripsArtifactsOnItems(t *testing.T) {
	item := DynamoComponentDeployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoComponentDeployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "n"},
		Spec: DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "x"}}},
				},
			},
		},
	}
	list := &DynamoComponentDeploymentList{
		TypeMeta: metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoComponentDeploymentList"},
		Items:    []DynamoComponentDeployment{item, item},
	}
	b, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	items := m["items"].([]any)
	for i, raw := range items {
		pt := raw.(map[string]any)["spec"].(map[string]any)["podTemplate"].(map[string]any)
		if _, has := pt["metadata"]; has {
			t.Errorf("DCDList items[%d] podTemplate.metadata should be stripped, got: %v", i, pt["metadata"])
		}
	}
}

// TestDGDRListMarshal_NoArtifactsInRootFields sanity-checks that a DGDR
// list item is normalized by the custom MarshalJSON. DGDR has no embedded
// PodTemplateSpec today, so there is no `podTemplate.metadata` on its wire
// shape; instead we assert that none of the root-level Go-encoder artefact
// patterns we know about (empty `metadata` on the top-level TypeMeta-bearing
// struct, empty `status`) sneak in. The test's real value is guarding the
// value-receiver contract for the third root kind.
func TestDGDRListMarshal_NoArtifactsInRootFields(t *testing.T) {
	item := DynamoGraphDeploymentRequest{
		TypeMeta:   metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeploymentRequest"},
		ObjectMeta: metav1.ObjectMeta{Name: "req", Namespace: "ns"},
		Spec:       DynamoGraphDeploymentRequestSpec{Model: "m"},
	}
	list := &DynamoGraphDeploymentRequestList{
		TypeMeta: metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeploymentRequestList"},
		Items:    []DynamoGraphDeploymentRequest{item},
	}
	b, err := json.Marshal(list)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	items := m["items"].([]any)
	root := items[0].(map[string]any)
	// metav1.ObjectMeta is value-typed on the root, but a non-empty
	// ObjectMeta (Name/Namespace set) must still appear on the wire.
	if _, ok := root["metadata"]; !ok {
		t.Errorf("DGDR list item: populated metadata should be preserved, got root=%v", root)
	}
	// Status is a value-typed struct with no fields set; it should be
	// stripped by the normalizer, proving MarshalJSON fired on the item.
	if _, ok := root["status"]; ok {
		t.Errorf("DGDR list item: empty status should be stripped (value-receiver MarshalJSON not firing?), got: %v", root["status"])
	}
}

// TestDGDRMarshal_PreservesUnknownFieldsPayload locks in the unknown-fields
// contract for `runtime.RawExtension`-bearing fields (which carry
// `+kubebuilder:pruning:PreserveUnknownFields` on the type definition).
// The normalizer's reflection walk descends into named Go fields only;
// payloads stored as RawExtension are opaque to our Go type tree and must
// be passed through byte-identical so that consumer-defined empty objects
// (e.g. `spec.nested.empty: {}` inside a user-supplied DGD override) are
// not mistaken for encoder artefacts and stripped.
func TestDGDRMarshal_PreservesUnknownFieldsPayload(t *testing.T) {
	payload := json.RawMessage(`{"apiVersion":"nvidia.com/v1alpha1","kind":"DynamoGraphDeployment","spec":{"services":{"decode":{"extraPodSpec":{"marker":{}}}}}}`)
	dgdr := &DynamoGraphDeploymentRequest{
		TypeMeta:   metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeploymentRequest"},
		ObjectMeta: metav1.ObjectMeta{Name: "req", Namespace: "ns"},
		Spec: DynamoGraphDeploymentRequestSpec{
			Model: "meta-llama/Llama-3-8B",
			Overrides: &OverridesSpec{
				DGD: &runtime.RawExtension{Raw: payload},
			},
		},
	}
	b, err := json.Marshal(dgdr)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	m := unmarshalToMap(t, b)
	dgd := m["spec"].(map[string]any)["overrides"].(map[string]any)["dgd"].(map[string]any)
	if dgd["apiVersion"] != "nvidia.com/v1alpha1" || dgd["kind"] != "DynamoGraphDeployment" {
		t.Fatalf("RawExtension payload apiVersion/kind mangled: %v", dgd)
	}
	decode := dgd["spec"].(map[string]any)["services"].(map[string]any)["decode"].(map[string]any)
	extra := decode["extraPodSpec"].(map[string]any)
	marker, ok := extra["marker"]
	if !ok {
		t.Fatalf("user-supplied `marker: {}` inside RawExtension payload was stripped; the normalizer must not descend into unknown-field payloads")
	}
	if mm, isMap := marker.(map[string]any); !isMap || len(mm) != 0 {
		t.Errorf("expected marker to remain {}, got: %v", marker)
	}
}

// newDGDWithEmptyPodTemplateMetadata is a compact DGD builder used by the
// list-item tests. The shape deliberately matches what the v1alpha1 -> v1beta1
// conversion pipeline produces for a stored object with no ExtraPodMetadata:
// an all-zero PodTemplateSpec.ObjectMeta that a vanilla encoder would emit
// as `"metadata": {}`.
func newDGDWithEmptyPodTemplateMetadata(name string) DynamoGraphDeployment {
	return DynamoGraphDeployment{
		TypeMeta:   metav1.TypeMeta{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "jsm"},
		Spec: DynamoGraphDeploymentSpec{
			Components: []DynamoComponentDeploymentSharedSpec{{
				ComponentName: "decode",
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:  "main",
							Image: "my-registry/sglang-runtime:my-tag",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")},
							},
						}},
					},
				}}},
		},
	}
}

// BenchmarkMarshal_DGD_Aggregate records a per-op cost baseline for the
// reflection-driven normalization. The shape mirrors a realistic
// aggregated-serve DGD (one service, one container, resource limits set)
// so future refactors can detect unintended regressions in either the
// cached field-lookup path or the map round-trip.
func BenchmarkMarshal_DGD_Aggregate(b *testing.B) {
	dgd := newDGDWithEmptyPodTemplateMetadata("agg")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := json.Marshal(&dgd); err != nil {
			b.Fatal(err)
		}
	}
}
