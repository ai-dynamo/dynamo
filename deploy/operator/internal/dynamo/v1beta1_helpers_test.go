package dynamo

import (
	"context"
	"maps"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
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

func TestComponentRuntimeNamespace(t *testing.T) {
	tests := []struct {
		name             string
		componentType    string
		workerHashSuffix string
		want             string
	}{
		{
			name:             "worker appends hash suffix",
			componentType:    commonconsts.ComponentTypeWorker,
			workerHashSuffix: "abc123",
			want:             "base-abc123",
		},
		{
			name:             "decode appends hash suffix",
			componentType:    commonconsts.ComponentTypeDecode,
			workerHashSuffix: "abc123",
			want:             "base-abc123",
		},
		{
			name:             "frontend ignores hash suffix",
			componentType:    commonconsts.ComponentTypeFrontend,
			workerHashSuffix: "abc123",
			want:             "base",
		},
		{
			name:             "legacy worker hash remains active suffix",
			componentType:    commonconsts.ComponentTypeWorker,
			workerHashSuffix: commonconsts.LegacyWorkerHash,
			want:             "base-legacy",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComponentRuntimeNamespace("base", tt.componentType, tt.workerHashSuffix)
			if got != tt.want {
				t.Fatalf("ComponentRuntimeNamespace() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetGroveRuntimeNamespaceUsesCanonicalWorkerHash(t *testing.T) {
	tests := []struct {
		name         string
		enableSuffix bool
		wantSuffix   bool
	}{
		{
			name: "suffix disabled uses base namespace",
		},
		{
			name:         "suffix enabled uses canonical worker hash",
			enableSuffix: true,
			wantSuffix:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			annotations := map[string]string(nil)
			if tt.enableSuffix {
				annotations = map[string]string{commonconsts.AnnotationGroveWorkerHashSuffixEnabled: "true"}
			}
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "grove",
					Namespace:   "k8s",
					Annotations: annotations,
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "worker",
						ComponentType: commonconsts.ComponentTypeWorker,
					}},
				},
			}
			component := &dgd.Spec.Components[0]
			want := dgd.GetDynamoNamespaceForComponent(component)
			if tt.wantSuffix {
				hash, err := ComputeDGDWorkersSpecHash(dgd)
				if err != nil {
					t.Fatalf("ComputeDGDWorkersSpecHash() error = %v", err)
				}
				want += "-" + hash
			}

			t.Log("Read the namespace against a Grove worker that completed the accepted PCS revision.")
			reader, acceptedPCSRevisionHash := completedGroveWorkerReader(t, dgd, component)
			got, err := GetGroveRuntimeNamespace(context.Background(), reader, dgd, component, acceptedPCSRevisionHash)
			if err != nil {
				t.Fatalf("GetGroveRuntimeNamespace() error = %v", err)
			}
			if got != want {
				t.Fatalf("GetGroveRuntimeNamespace() = %q, want %q", got, want)
			}
		})
	}
}

func TestGetGroveRuntimeNamespacePreservesActiveWorkerNamespaceUntilTargetRevisionCommits(t *testing.T) {
	t.Log("Build a suffixed worker with a namespace from the active generation")
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "grove",
			Namespace: "k8s",
			Annotations: map[string]string{
				commonconsts.AnnotationGroveWorkerHashSuffixEnabled: commonconsts.KubeLabelValueTrue,
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: commonconsts.ComponentTypeWorker,
			}},
		},
	}
	dgd.Status.Components = map[string]v1beta1.ComponentReplicaStatus{
		"worker": {RuntimeNamespace: "active-worker-namespace"},
	}
	component := dgd.GetComponentByName("worker")

	g := gomega.NewWithT(t)
	var reader client.Reader = newFakeGroveClient(g)
	t.Log("Keep the active namespace while the accepted PCS revision is pending.")
	var acceptedPCSRevisionHash *string
	got, err := GetGroveRuntimeNamespace(context.Background(), reader, dgd, component, acceptedPCSRevisionHash)
	if err != nil {
		t.Fatalf("GetGroveRuntimeNamespace() error = %v", err)
	}
	if got != "active-worker-namespace" {
		t.Fatalf("GetGroveRuntimeNamespace() = %q, want active worker namespace", got)
	}

	t.Log("Publish the desired namespace after the child completes the accepted PCS revision.")
	hash, err := ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash() error = %v", err)
	}
	want := ComponentRuntimeNamespace(dgd.GetDynamoNamespaceForComponent(component), string(component.ComponentType), hash)
	reader, acceptedPCSRevisionHash = completedGroveWorkerReader(t, dgd, component)
	got, err = GetGroveRuntimeNamespace(context.Background(), reader, dgd, component, acceptedPCSRevisionHash)
	if err != nil {
		t.Fatalf("GetGroveRuntimeNamespace() error = %v", err)
	}
	if got != want {
		t.Fatalf("GetGroveRuntimeNamespace() = %q, want %q", got, want)
	}
}

func completedGroveWorkerReader(
	t *testing.T,
	dgd *v1beta1.DynamoGraphDeployment,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) (client.Reader, *string) {
	t.Helper()
	g := gomega.NewWithT(t)
	targetRevision := "target-revision"
	completedAt := metav1.Now()

	podClique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Name:      GroveComponentResourceName(dgd, component.ComponentName),
			Namespace: dgd.Namespace,
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1},
		Status: grovev1alpha1.PodCliqueStatus{
			Replicas:                          1,
			UpdatedReplicas:                   1,
			CurrentPodCliqueSetGenerationHash: &targetRevision,
			UpdateProgress:                    &grovev1alpha1.PodCliqueUpdateProgress{UpdateEndedAt: &completedAt},
		},
	}
	return newFakeGroveClient(g, podClique), &targetRevision
}

func TestGetDCDRuntimeNamespaceUsesMetadataWorkerHashBeforePodTemplate(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "k8s",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoNamespace:  "base",
				commonconsts.KubeLabelDynamoWorkerHash: "abc123",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoWorkerHash: "pod-template-hash",
						},
					},
				},
			},
		},
	}

	if got := GetDCDEffectiveWorkerHash(dcd); got != "abc123" {
		t.Fatalf("GetDCDEffectiveWorkerHash() = %q, want abc123", got)
	}
	if got := GetDCDRuntimeNamespace(dcd); got != "base-abc123" {
		t.Fatalf("GetDCDRuntimeNamespace() = %q, want base-abc123", got)
	}
}

func TestGetDCDRuntimeNamespaceUsesPodTemplateWorkerHashWhenMetadataMissing(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "k8s",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoNamespace: "base",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoWorkerHash: "abc123",
						},
					},
				},
			},
		},
	}

	if got := GetDCDEffectiveWorkerHash(dcd); got != "abc123" {
		t.Fatalf("GetDCDEffectiveWorkerHash() = %q, want abc123", got)
	}
	if got := GetDCDRuntimeNamespace(dcd); got != "base-abc123" {
		t.Fatalf("GetDCDRuntimeNamespace() = %q, want base-abc123", got)
	}
}

func TestGetDCDRuntimeNamespaceUsesLegacyMetadataOnlyHash(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "k8s",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoNamespace:  "base",
				commonconsts.KubeLabelDynamoWorkerHash: commonconsts.LegacyWorkerHash,
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
			},
		},
	}

	if got := GetDCDEffectiveWorkerHash(dcd); got != commonconsts.LegacyWorkerHash {
		t.Fatalf("GetDCDEffectiveWorkerHash() = %q, want legacy", got)
	}
	if got := GetDCDRuntimeNamespace(dcd); got != "base-legacy" {
		t.Fatalf("GetDCDRuntimeNamespace() = %q, want base-legacy", got)
	}
}

func TestGetDCDRuntimeNamespaceUsesLegacyPodTemplateHash(t *testing.T) {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "k8s",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoNamespace: "base",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoWorkerHash: commonconsts.LegacyWorkerHash,
						},
					},
				},
			},
		},
	}

	if got := GetDCDEffectiveWorkerHash(dcd); got != commonconsts.LegacyWorkerHash {
		t.Fatalf("GetDCDEffectiveWorkerHash() = %q, want legacy", got)
	}
	if got := GetDCDRuntimeNamespace(dcd); got != "base-legacy" {
		t.Fatalf("GetDCDRuntimeNamespace() = %q, want base-legacy", got)
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
		Enabled:       true,
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
