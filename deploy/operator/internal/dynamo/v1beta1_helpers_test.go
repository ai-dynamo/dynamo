package dynamo

import (
	"maps"
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestEffectiveComponentForRoleManualWorkerOverrides(t *testing.T) {
	terminationGracePeriod := int64(30)
	leaderClaimTemplate := "leader-claim"
	workerClaimTemplate := "worker-claim"
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      map[string]string{"role": "leader"},
				Annotations: map[string]string{"source": "leader"},
			},
			Spec: corev1.PodSpec{
				NodeSelector: map[string]string{"pool": "leader"},
				Tolerations:  []corev1.Toleration{{Key: "leader"}},
				ResourceClaims: []corev1.PodResourceClaim{{
					Name:                      "devices",
					ResourceClaimTemplateName: &leaderClaimTemplate,
				}},
				ImagePullSecrets:              []corev1.LocalObjectReference{{Name: "leader-secret"}},
				TerminationGracePeriodSeconds: &terminationGracePeriod,
				Containers: []corev1.Container{
					{
						Name:    commonconsts.MainContainerName,
						Image:   "leader:1.4.0",
						Command: []string{"python3", "-m", "dynamo.vllm"},
						Args:    []string{"--node-rank=0"},
						Env:     []corev1.EnvVar{{Name: "ENGINE_ROLE", Value: "leader"}},
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1")},
							Claims: []corev1.ResourceClaim{{Name: "leader-devices"}},
						},
					},
					{Name: "metrics", Image: "metrics:1.0.0"},
				},
			},
		},
		Multinode: &v1beta1.MultinodeSpec{
			Mode:      v1beta1.MultinodeModeManual,
			NodeCount: 2,
			Worker: &v1beta1.MultinodeWorkerSpec{PodTemplateOverrides: &v1beta1.MultinodePodTemplateOverrides{
				Metadata: &v1beta1.MultinodePodTemplateMetadataOverrides{
					Labels:      ptr.To(map[string]string{"role": "worker"}),
					Annotations: ptr.To(map[string]string{"source": "worker"}),
				},
				Spec: &v1beta1.MultinodePodSpecOverrides{
					NodeSelector: ptr.To(map[string]string{"pool": "worker"}),
					Tolerations:  ptr.To([]corev1.Toleration{{Key: "worker"}}),
					ResourceClaims: ptr.To([]corev1.PodResourceClaim{{
						Name:                      "devices",
						ResourceClaimTemplateName: &workerClaimTemplate,
					}}),
					ImagePullSecrets: ptr.To([]corev1.LocalObjectReference{{Name: "worker-secret"}}),
					Containers: []v1beta1.MultinodeContainerOverride{{
						Name:    commonconsts.MainContainerName,
						Image:   ptr.To("worker:1.4.0"),
						Command: ptr.To([]string{"python3", "-m", "dynamo.vllm"}),
						Args:    ptr.To([]string{"--node-rank=1", "--headless"}),
						Env:     ptr.To([]corev1.EnvVar{{Name: "ENGINE_ROLE", Value: "worker"}}),
						Resources: &v1beta1.MultinodeContainerResourceOverrides{
							Claims: ptr.To([]corev1.ResourceClaim{{Name: "worker-devices"}}),
						},
					}},
				},
			}},
		},
	}

	leader, err := EffectiveComponentForRole(component, RoleLeader)
	if err != nil {
		t.Fatalf("resolve leader: %v", err)
	}
	worker, err := EffectiveComponentForRole(component, RoleWorker)
	if err != nil {
		t.Fatalf("resolve worker: %v", err)
	}
	if !reflect.DeepEqual(leader.PodTemplate, component.PodTemplate) {
		t.Fatalf("leader template changed: got %#v, want %#v", leader.PodTemplate, component.PodTemplate)
	}
	if !maps.Equal(worker.PodTemplate.Labels, map[string]string{"role": "worker"}) ||
		!maps.Equal(worker.PodTemplate.Annotations, map[string]string{"source": "worker"}) {
		t.Fatalf("worker metadata = labels %v annotations %v", worker.PodTemplate.Labels, worker.PodTemplate.Annotations)
	}
	if !maps.Equal(worker.PodTemplate.Spec.NodeSelector, map[string]string{"pool": "worker"}) ||
		!reflect.DeepEqual(worker.PodTemplate.Spec.Tolerations, []corev1.Toleration{{Key: "worker"}}) ||
		!reflect.DeepEqual(worker.PodTemplate.Spec.ImagePullSecrets, []corev1.LocalObjectReference{{Name: "worker-secret"}}) {
		t.Fatalf("worker pod overrides were not applied: %#v", worker.PodTemplate.Spec)
	}
	if got := *worker.PodTemplate.Spec.ResourceClaims[0].ResourceClaimTemplateName; got != workerClaimTemplate {
		t.Fatalf("worker resource claim template = %q, want %q", got, workerClaimTemplate)
	}
	if worker.PodTemplate.Spec.TerminationGracePeriodSeconds == nil || *worker.PodTemplate.Spec.TerminationGracePeriodSeconds != terminationGracePeriod {
		t.Fatal("unrelated pod fields were not inherited")
	}
	main := GetMainContainer(worker)
	if main == nil {
		t.Fatal("effective worker has no main container")
	}
	if main.Image != "worker:1.4.0" || !reflect.DeepEqual(main.Args, []string{"--node-rank=1", "--headless"}) ||
		!reflect.DeepEqual(main.Env, []corev1.EnvVar{{Name: "ENGINE_ROLE", Value: "worker"}}) ||
		!reflect.DeepEqual(main.Resources.Claims, []corev1.ResourceClaim{{Name: "worker-devices"}}) {
		t.Fatalf("worker main container override = %#v", main)
	}
	if gpuLimit := main.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]; gpuLimit.Cmp(resource.MustParse("1")) != 0 {
		t.Fatalf("unrelated main-container GPU limit = %s, want 1", gpuLimit.String())
	}
	if len(worker.PodTemplate.Spec.Containers) != 2 || worker.PodTemplate.Spec.Containers[1].Name != "metrics" {
		t.Fatalf("sidecars were not inherited: %#v", worker.PodTemplate.Spec.Containers)
	}
	if GetMainContainer(component).Image != "leader:1.4.0" {
		t.Fatal("effective template resolution mutated the source component")
	}
}

func TestEffectiveComponentForRoleManualWorkerExplicitEmptyOverrides(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"leader": "true"}, Annotations: map[string]string{"leader": "true"}},
			Spec: corev1.PodSpec{
				NodeSelector:     map[string]string{"pool": "leader"},
				Tolerations:      []corev1.Toleration{{Key: "leader"}},
				ResourceClaims:   []corev1.PodResourceClaim{{Name: "devices"}},
				ImagePullSecrets: []corev1.LocalObjectReference{{Name: "registry"}},
				Containers: []corev1.Container{{
					Name:    commonconsts.MainContainerName,
					Image:   "runtime:1.4.0",
					Command: []string{"python3"},
					Args:    []string{"leader"},
					Env:     []corev1.EnvVar{{Name: "LEADER", Value: "true"}},
					Resources: corev1.ResourceRequirements{
						Claims: []corev1.ResourceClaim{{Name: "devices"}},
					},
				}},
			},
		},
		Multinode: &v1beta1.MultinodeSpec{
			Mode: v1beta1.MultinodeModeManual,
			Worker: &v1beta1.MultinodeWorkerSpec{PodTemplateOverrides: &v1beta1.MultinodePodTemplateOverrides{
				Metadata: &v1beta1.MultinodePodTemplateMetadataOverrides{
					Labels:      ptr.To(map[string]string{}),
					Annotations: ptr.To(map[string]string{}),
				},
				Spec: &v1beta1.MultinodePodSpecOverrides{
					NodeSelector:     ptr.To(map[string]string{}),
					Tolerations:      ptr.To([]corev1.Toleration{}),
					ResourceClaims:   ptr.To([]corev1.PodResourceClaim{}),
					ImagePullSecrets: ptr.To([]corev1.LocalObjectReference{}),
					Containers: []v1beta1.MultinodeContainerOverride{{
						Name:      commonconsts.MainContainerName,
						Command:   ptr.To([]string{}),
						Args:      ptr.To([]string{}),
						Env:       ptr.To([]corev1.EnvVar{}),
						Resources: &v1beta1.MultinodeContainerResourceOverrides{Claims: ptr.To([]corev1.ResourceClaim{})},
					}},
				},
			}},
		},
	}

	worker, err := EffectiveComponentForRole(component, RoleWorker)
	if err != nil {
		t.Fatalf("resolve worker: %v", err)
	}
	main := GetMainContainer(worker)
	if len(worker.PodTemplate.Labels) != 0 || len(worker.PodTemplate.Annotations) != 0 ||
		len(worker.PodTemplate.Spec.NodeSelector) != 0 || len(worker.PodTemplate.Spec.Tolerations) != 0 ||
		len(worker.PodTemplate.Spec.ResourceClaims) != 0 || len(worker.PodTemplate.Spec.ImagePullSecrets) != 0 ||
		len(main.Command) != 0 || len(main.Args) != 0 || len(main.Env) != 0 || len(main.Resources.Claims) != 0 {
		t.Fatalf("explicit empty overrides did not clear inherited values: %#v", worker.PodTemplate)
	}
	if main.Image != "runtime:1.4.0" {
		t.Fatalf("omitted image override did not inherit leader image: %q", main.Image)
	}
}

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
