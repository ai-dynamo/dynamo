package dynamo

import (
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestComponentsByNameNil(t *testing.T) {
	if got := ComponentsByName(nil); len(got) != 0 {
		t.Fatalf("ComponentsByName(nil) = %#v, want empty map", got)
	}
}

func TestGetDCDComponentNamePrefersSpecOverLegacyMetadata(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "metadata-name",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoComponent: "label-component",
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

func TestGetDCDComponentNameIgnoresStalePreservedServiceName(t *testing.T) {
	const (
		specComponentName  = "live-beta-component"
		labelComponentName = "stable-label-component"
	)
	dcd := dcdFromAlpha(t, v1alpha1.DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
			ServiceName: "stale-alpha-service-name",
		},
	})
	dcd.Spec.ComponentName = specComponentName
	dcd.Labels = map[string]string{
		commonconsts.KubeLabelDynamoComponent: labelComponentName,
	}

	if got, want := GetDCDComponentName(dcd), specComponentName; got != want {
		t.Fatalf("GetDCDComponentName() = %q, want %q", got, want)
	}

	dcd.Spec.ComponentName = ""
	if got, want := GetDCDComponentName(dcd), labelComponentName; got != want {
		t.Fatalf("GetDCDComponentName() without spec name = %q, want %q", got, want)
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
				},
			},
			want: "label-component",
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

func TestDCDAlphaCompatibilityHelpersReadThroughAPIConversion(t *testing.T) {
	dynamoNamespace := "canonical-namespace"
	dcd := dcdFromAlpha(t, v1alpha1.DynamoComponentDeploymentSpec{
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
	})
	dcd.Labels = map[string]string{
		commonconsts.KubeLabelDynamoNamespace: "label-namespace",
	}

	if got := GetDCDDynamoNamespace(dcd); got != "canonical-namespace" {
		t.Fatalf("GetDCDDynamoNamespace() = %q, want canonical-namespace", got)
	}
	if got := GetDCDSubComponentType(dcd); got != "canonical-sub" {
		t.Fatalf("GetDCDSubComponentType() = %q, want canonical-sub", got)
	}
	if got, want := GetDCDPreservedAlphaAnnotations(dcd), map[string]string{"canonical-annotation": "kept"}; !maps.Equal(got, want) {
		t.Fatalf("GetDCDPreservedAlphaAnnotations() = %#v, want %#v", got, want)
	}
	if got, want := GetDCDPreservedAlphaLabels(dcd), map[string]string{"canonical-label": "kept"}; !maps.Equal(got, want) {
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

func TestGetDCDWorkloadComponentTypePreservesLegacyAlphaWorkerSelector(t *testing.T) {
	dcd := dcdFromAlpha(t, v1alpha1.DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
			ComponentType:    commonconsts.ComponentTypeWorker,
			SubComponentType: commonconsts.ComponentTypeDecode,
		},
	})
	dcd.Labels = map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
		commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
		commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
		commonconsts.KubeLabelDynamoSubComponentType:    commonconsts.ComponentTypeDecode,
	}

	if got := GetDCDWorkloadComponentType(dcd); got != commonconsts.ComponentTypeWorker {
		t.Fatalf("GetDCDWorkloadComponentType() = %q, want worker", got)
	}

	dcd.Labels = nil
	if got := GetDCDWorkloadComponentType(dcd); got != commonconsts.ComponentTypeDecode {
		t.Fatalf("GetDCDWorkloadComponentType() without legacy labels = %q, want decode", got)
	}
}

func TestToAlphaCheckpointConfigSetsNilIdentityThroughConverter(t *testing.T) {
	got := ToAlphaCheckpointConfig(&v1beta1.ComponentCheckpointConfig{
		Mode:          v1beta1.CheckpointMode("auto"),
		CheckpointRef: ptr.To("checkpoint"),
	})
	if got == nil {
		t.Fatalf("ToAlphaCheckpointConfig() = nil")
	}
	if got.Identity != nil {
		t.Fatalf("ToAlphaCheckpointConfig().Identity = %#v, want nil", got.Identity)
	}
}

func TestToBetaSharedMemorySize(t *testing.T) {
	size := resource.MustParse("2Gi")
	tests := []struct {
		name string
		src  *v1alpha1.SharedMemorySpec
		want string
	}{
		{name: "nil"},
		{name: "zero size", src: &v1alpha1.SharedMemorySpec{}},
		{name: "disabled", src: &v1alpha1.SharedMemorySpec{Disabled: true}, want: "0"},
		{name: "size", src: &v1alpha1.SharedMemorySpec{Size: size}, want: "2Gi"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ToBetaSharedMemorySize(tt.src)
			if tt.want == "" {
				if got != nil {
					t.Fatalf("ToBetaSharedMemorySize() = %s, want nil", got.String())
				}
				return
			}
			if got == nil || got.String() != tt.want {
				t.Fatalf("ToBetaSharedMemorySize() = %v, want %s", got, tt.want)
			}
		})
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

func dcdFromAlpha(t *testing.T, spec v1alpha1.DynamoComponentDeploymentSpec) *v1beta1.DynamoComponentDeployment {
	t.Helper()

	alpha := &v1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "test-ns",
		},
		Spec: spec,
	}
	beta := &v1beta1.DynamoComponentDeployment{}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("ConvertTo(v1beta1) error = %v", err)
	}
	return beta
}
