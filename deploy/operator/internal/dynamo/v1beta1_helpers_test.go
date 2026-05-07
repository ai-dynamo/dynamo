package dynamo

import (
	"encoding/json"
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestGetDCDComponentNamePrefersSpecOverLegacyMetadata(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "metadata-name",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoComponent: "label-component",
			},
			Annotations: map[string]string{
				preservedDCDServiceNameAnnotation: "annotation-component",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "spec-component",
			},
		},
	}

	if got, want := GetDCDComponentName(dcd), "spec-component"; got != want {
		t.Fatalf("GetDCDComponentName() = %q, want %q", got, want)
	}
}

func TestGetDCDComponentNameLegacyFallbacks(t *testing.T) {
	tests := []struct {
		name string
		dcd  *v1beta1.DynamoComponentDeployment
		want string
	}{
		{
			name: "nil",
			want: "",
		},
		{
			name: "label",
			dcd: &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "metadata-name",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoComponent: "label-component",
					},
					Annotations: map[string]string{
						preservedDCDServiceNameAnnotation: "annotation-component",
					},
				},
			},
			want: "label-component",
		},
		{
			name: "annotation",
			dcd: &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "metadata-name",
					Annotations: map[string]string{
						preservedDCDServiceNameAnnotation: "annotation-component",
					},
				},
			},
			want: "annotation-component",
		},
		{
			name: "metadata name",
			dcd: &v1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "metadata-name"},
			},
			want: "metadata-name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDCDComponentName(tt.dcd); got != tt.want {
				t.Fatalf("GetDCDComponentName() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPreservedAlphaSpecTakesPrecedenceOverLegacyCarrierAnnotations(t *testing.T) {
	dynamoNamespace := "canonical-namespace"
	alphaSpec := v1alpha1.DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
			Annotations:      map[string]string{"canonical-annotation": "kept"},
			Labels:           map[string]string{"canonical-label": "kept"},
			DynamoNamespace:  &dynamoNamespace,
			SubComponentType: "canonical-sub",
			Ingress: &v1alpha1.IngressSpec{
				Enabled:                    true,
				Host:                       "canonical.example.com",
				IngressControllerClassName: ptr.To("nginx"),
			},
		},
	}
	dcd := dcdWithPreservedAlphaSpec(t, alphaSpec, true)
	dcd.Labels = map[string]string{
		commonconsts.KubeLabelDynamoNamespace: "label-namespace",
	}
	dcd.Annotations[preservedDCDDynamoNamespaceAnnotation] = "stale-namespace"
	dcd.Annotations[preservedDCDSubComponentTypeAnnotation] = "stale-sub"
	dcd.Annotations[PreservedDCDAnnotationsAnnotation] = mustJSON(t, map[string]string{"stale-annotation": "ignored"})
	dcd.Annotations[PreservedDCDLabelsAnnotation] = mustJSON(t, map[string]string{"stale-label": "ignored"})
	dcd.Annotations[PreservedDCDIngressAnnotation] = mustJSON(t, IngressSpec{Enabled: true, Host: "stale.example.com"})

	if got := GetDCDDynamoNamespace(dcd); got != "canonical-namespace" {
		t.Fatalf("GetDCDDynamoNamespace() = %q, want canonical-namespace", got)
	}
	if got := GetDCDSubComponentType(dcd); got != "canonical-sub" {
		t.Fatalf("GetDCDSubComponentType() = %q, want canonical-sub", got)
	}
	if got, want := GetDCDPreservedAlphaAnnotations(dcd), alphaSpec.Annotations; !maps.Equal(got, want) {
		t.Fatalf("GetDCDPreservedAlphaAnnotations() = %#v, want %#v", got, want)
	}
	if got, want := GetDCDPreservedAlphaLabels(dcd), alphaSpec.Labels; !maps.Equal(got, want) {
		t.Fatalf("GetDCDPreservedAlphaLabels() = %#v, want %#v", got, want)
	}
	ingressSpec, ok, err := GetDCDPreservedAlphaIngressSpec(dcd)
	if err != nil {
		t.Fatalf("GetDCDPreservedAlphaIngressSpec() error = %v", err)
	}
	if !ok || !ingressSpec.Enabled || ingressSpec.Host != "canonical.example.com" || ingressSpec.IngressControllerClassName == nil || *ingressSpec.IngressControllerClassName != "nginx" {
		t.Fatalf("GetDCDPreservedAlphaIngressSpec() = (%#v, %v), want canonical ingress", ingressSpec, ok)
	}
}

func TestPreservedAlphaHelpersFallbackToLegacyCarrierAnnotations(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "test-ns",
			Annotations: map[string]string{
				preservedDCDDynamoNamespaceAnnotation:  "legacy-namespace",
				preservedDCDSubComponentTypeAnnotation: "legacy-sub",
				PreservedDCDAnnotationsAnnotation:      mustJSON(t, map[string]string{"legacy-annotation": "kept"}),
				PreservedDCDLabelsAnnotation:           mustJSON(t, map[string]string{"legacy-label": "kept"}),
				PreservedDCDIngressAnnotation:          mustJSON(t, IngressSpec{Enabled: true, Host: "legacy.example.com"}),
			},
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoNamespace: "label-namespace",
			},
		},
	}

	if got := GetDCDDynamoNamespace(dcd); got != "legacy-namespace" {
		t.Fatalf("GetDCDDynamoNamespace() = %q, want legacy-namespace", got)
	}
	if got := GetDCDSubComponentType(dcd); got != "legacy-sub" {
		t.Fatalf("GetDCDSubComponentType() = %q, want legacy-sub", got)
	}
	if got, want := GetDCDPreservedAlphaAnnotations(dcd), map[string]string{"legacy-annotation": "kept"}; !maps.Equal(got, want) {
		t.Fatalf("GetDCDPreservedAlphaAnnotations() = %#v, want %#v", got, want)
	}
	if got, want := GetDCDPreservedAlphaLabels(dcd), map[string]string{"legacy-label": "kept"}; !maps.Equal(got, want) {
		t.Fatalf("GetDCDPreservedAlphaLabels() = %#v, want %#v", got, want)
	}
	ingressSpec, ok, err := GetDCDPreservedAlphaIngressSpec(dcd)
	if err != nil {
		t.Fatalf("GetDCDPreservedAlphaIngressSpec() error = %v", err)
	}
	if !ok || ingressSpec.Host != "legacy.example.com" {
		t.Fatalf("GetDCDPreservedAlphaIngressSpec() = (%#v, %v), want legacy ingress", ingressSpec, ok)
	}
}

func TestGetDCDPreservedAlphaSpecSupportsEnvelopeAndRawSpec(t *testing.T) {
	dynamoNamespace := "raw-namespace"
	alphaSpec := v1alpha1.DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
			DynamoNamespace:  &dynamoNamespace,
			SubComponentType: "raw-sub",
		},
	}

	envelopeDCD := dcdWithPreservedAlphaSpec(t, alphaSpec, true)
	if got := getDCDPreservedAlphaSpec(envelopeDCD); got == nil || got.SubComponentType != "raw-sub" {
		t.Fatalf("envelope getDCDPreservedAlphaSpec() = %#v, want raw-sub", got)
	}

	rawDCD := dcdWithPreservedAlphaSpec(t, alphaSpec, false)
	if got := getDCDPreservedAlphaSpec(rawDCD); got == nil || got.DynamoNamespace == nil || *got.DynamoNamespace != "raw-namespace" {
		t.Fatalf("raw getDCDPreservedAlphaSpec() = %#v, want raw-namespace", got)
	}
}

func TestGetDCDPreservedAlphaIngressSpecReturnsMalformedAnnotationError(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				PreservedDCDIngressAnnotation: "{",
			},
		},
	}

	if _, _, err := GetDCDPreservedAlphaIngressSpec(dcd); err == nil {
		t.Fatalf("GetDCDPreservedAlphaIngressSpec() error = nil, want malformed JSON error")
	}
}

func TestMergeLowPriorityMetadata(t *testing.T) {
	got := mergeLowPriorityMetadata(
		map[string]string{"existing": "kept", "shared": "winner"},
		map[string]string{"shared": "ignored", "new": "added"},
	)
	want := map[string]string{"existing": "kept", "shared": "winner", "new": "added"}
	if !maps.Equal(got, want) {
		t.Fatalf("mergeLowPriorityMetadata() = %#v, want %#v", got, want)
	}
}

func dcdWithPreservedAlphaSpec(t *testing.T, spec v1alpha1.DynamoComponentDeploymentSpec, envelope bool) *v1beta1.DynamoComponentDeployment {
	t.Helper()

	var value any = spec
	if envelope {
		value = struct {
			Spec v1alpha1.DynamoComponentDeploymentSpec `json:"spec"`
		}{Spec: spec}
	}
	return &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "dcd",
			Annotations: map[string]string{
				preservedDCDAlphaSpecAnnotation: mustJSON(t, value),
			},
		},
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()

	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal(%#v) error = %v", value, err)
	}
	return string(data)
}
