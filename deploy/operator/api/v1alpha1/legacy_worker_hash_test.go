/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"
	"unicode/utf8"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

const currentWorkerHashAnnotation = "nvidia.com/current-worker-hash"

func TestComputeDGDWorkersSpecHashGolden(t *testing.T) {
	for _, tt := range legacyWorkerHashGoldenCases() {
		t.Run(tt.name, func(t *testing.T) {
			got1, err := ComputeDGDWorkersSpecHash(tt.dgd)
			if err != nil {
				t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
			}
			got2, err := ComputeDGDWorkersSpecHash(tt.dgd)
			if err != nil {
				t.Fatalf("second ComputeDGDWorkersSpecHash: %v", err)
			}
			if got1 != got2 {
				t.Fatalf("hash is not stable: first %q, second %q", got1, got2)
			}
			if got1 != tt.want {
				t.Fatalf("ComputeDGDWorkersSpecHash() = %q, want golden %q", got1, tt.want)
			}
		})
	}
}

type legacyWorkerHashGoldenCase struct {
	name string
	dgd  *DynamoGraphDeployment
	want string
}

// legacyWorkerHashGoldenCases are golden values from the v1.1.x worker-hash
// algorithm in deploy/operator/internal/dynamo/hash.go. Changing any value here
// is a rollout-compatibility change and needs an explicit migration plan.
func legacyWorkerHashGoldenCases() []legacyWorkerHashGoldenCase {
	return []legacyWorkerHashGoldenCase{
		{
			name: "worker",
			dgd:  legacyWorkerHashDGD(),
			want: "9b66accc",
		},
		{
			name: "resource metadata and non-workers ignored",
			dgd:  legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges(),
			want: "9b66accc",
		},
		{
			name: "pod metadata included",
			dgd:  legacyWorkerHashDGDWithPodMetadataChanges(),
			want: "af8a6c60",
		},
		{
			name: "no workers",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {ComponentType: commonconsts.ComponentTypeFrontend},
			}),
			want: "44136fa3",
		},
		{
			name: "nil services",
			dgd:  legacyWorkerHashDGDFromServices(nil),
			want: "44136fa3",
		},
		{
			name: "worker ordering",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"z-worker": {ComponentType: commonconsts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "Z", Value: "1"}}},
				"a-worker": {ComponentType: commonconsts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "A", Value: "1"}}},
			}),
			want: "59cae8b3",
		},
		{
			name: "all worker component types",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"decode":  {ComponentType: commonconsts.ComponentTypeDecode, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "decode"}}},
				"prefill": {ComponentType: commonconsts.ComponentTypePrefill, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "prefill"}}},
				"worker":  {ComponentType: commonconsts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "ROLE", Value: "worker"}}},
			}),
			want: "b175ee30",
		},
		{
			name: "main container name only",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					ExtraPodSpec: &ExtraPodSpec{
						MainContainer: &corev1.Container{Name: commonconsts.MainContainerName},
					},
				},
			}),
			want: "0c322ce0",
		},
		{
			name: "main container rich",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					ExtraPodSpec: &ExtraPodSpec{
						MainContainer: &corev1.Container{
							Name:    commonconsts.MainContainerName,
							Image:   "worker:1",
							Command: []string{"python", "-m", "server"},
							Args:    []string{"--model", "qwen"},
							Env:     []corev1.EnvVar{{Name: "EXTRA", Value: "true"}},
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("2")},
							},
						},
					},
				},
			}),
			want: "9e64367a",
		},
		{
			name: "pod spec rich",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					ExtraPodSpec: &ExtraPodSpec{
						PodSpec: &corev1.PodSpec{
							NodeSelector: map[string]string{"gpu": "true"},
							Tolerations: []corev1.Toleration{{
								Key:      "nvidia.com/gpu",
								Operator: corev1.TolerationOpExists,
							}},
							Volumes: []corev1.Volume{{
								Name: "cache",
								VolumeSource: corev1.VolumeSource{
									EmptyDir: &corev1.EmptyDirVolumeSource{},
								},
							}},
						},
					},
				},
			}),
			want: "1fcd5d7c",
		},
		{
			name: "resources requests limits claims",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					Resources: &Resources{
						Requests: &ResourceItem{CPU: "1", Memory: "4Gi", GPU: "1", GPUType: "nvidia.com/gpu"},
						Limits:   &ResourceItem{CPU: "2", Memory: "8Gi", GPU: "1"},
						Claims:   []corev1.ResourceClaim{{Name: "gpu-claim"}},
					},
				},
			}),
			want: "a4172e4e",
		},
		{
			name: "envs volume mounts secret",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:  commonconsts.ComponentTypeWorker,
					Envs:           []corev1.EnvVar{{Name: "MODEL", Value: "llama"}},
					EnvFromSecret:  ptr.To("worker-secret"),
					VolumeMounts:   []VolumeMount{{Name: "models", MountPoint: "/models", UseAsCompilationCache: true}},
					SharedMemory:   &SharedMemorySpec{Size: resource.MustParse("2Gi")},
					LivenessProbe:  &corev1.Probe{InitialDelaySeconds: 5},
					ReadinessProbe: &corev1.Probe{TimeoutSeconds: 3},
				},
			}),
			want: "a0ceefd2",
		},
		{
			name: "multiple compilation cache volume mounts",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					VolumeMounts: []VolumeMount{
						{Name: "model-cache", MountPoint: "/models", UseAsCompilationCache: true},
						{Name: "compile-cache", MountPoint: "/compile", UseAsCompilationCache: true},
					},
				},
			}),
			want: "5a3c0f65",
		},
		{
			name: "ignored scaling ingress model",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:  commonconsts.ComponentTypeWorker,
					Annotations:    map[string]string{"ignored": "true"},
					Labels:         map[string]string{"ignored": "true"},
					Replicas:       ptr.To(int32(3)),
					Autoscaling:    &Autoscaling{Enabled: true, MinReplicas: 1, MaxReplicas: 10},
					Ingress:        &IngressSpec{Enabled: true, Host: "example.com"},
					ModelRef:       &ModelReference{Name: "model"},
					ScalingAdapter: &ScalingAdapter{Enabled: true},
				},
			}),
			want: "769fa7c7",
		},
		{
			name: "multinode sidecar checkpoint",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					Multinode:     &MultinodeSpec{NodeCount: 2},
					FrontendSidecar: &FrontendSidecarSpec{
						Image: "frontend:1",
						Args:  []string{"--router-mode", "direct"},
					},
					Checkpoint: &ServiceCheckpointConfig{Enabled: true},
				},
			}),
			want: "bf66a8e3",
		},
		{
			name: "probe handler",
			dgd: legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					ReadinessProbe: &corev1.Probe{
						ProbeHandler: corev1.ProbeHandler{
							HTTPGet: &corev1.HTTPGetAction{
								Path: "/health",
								Port: intstr.FromString("http"),
							},
						},
					},
				},
			}),
			want: "8ddbbadc",
		},
	}
}

func TestComputeDGDWorkersSpecHashNil(t *testing.T) {
	if _, err := ComputeDGDWorkersSpecHash(nil); err == nil {
		t.Fatal("ComputeDGDWorkersSpecHash(nil) error = nil, want error")
	}
}

func TestDGDConvertToPreservesWorkerHashAnnotationOpaquely(t *testing.T) {
	src := legacyWorkerHashDGD()
	src.Annotations = map[string]string{
		currentWorkerHashAnnotation: "controller-owned-hash",
		"user":                      "kept",
	}

	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}

	if got := hub.Annotations[currentWorkerHashAnnotation]; got != "controller-owned-hash" {
		t.Fatalf("%s = %q, want controller-owned-hash", currentWorkerHashAnnotation, got)
	}
	if got := hub.Annotations["user"]; got != "kept" {
		t.Fatalf("user annotation = %q, want kept", got)
	}
}

func TestComputeDGDWorkersSpecHashConversionRoundTripExact(t *testing.T) {
	alpha := legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
		"worker": {
			ComponentType: commonconsts.ComponentTypeWorker,
			ExtraPodSpec: &ExtraPodSpec{
				MainContainer: &corev1.Container{Name: commonconsts.MainContainerName},
			},
		},
	})

	directHash, err := ComputeDGDWorkersSpecHash(alpha)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash(alpha): %v", err)
	}
	roundTripHash, err := legacyWorkerHashAfterBetaRoundTrip(alpha)
	if err != nil {
		t.Fatalf("legacyWorkerHashAfterBetaRoundTrip: %v", err)
	}

	if directHash != "0c322ce0" {
		t.Fatalf("direct alpha hash = %q, want 0c322ce0", directHash)
	}
	if roundTripHash != directHash {
		t.Fatalf("round-tripped alpha hash = %q, want direct alpha hash %q", roundTripHash, directHash)
	}
}

func TestComputeDGDWorkersSpecHashConversionRoundTripStableSubset(t *testing.T) {
	for _, tt := range legacyWorkerHashGoldenCases() {
		t.Run(tt.name, func(t *testing.T) {
			roundTripHash, err := legacyWorkerHashAfterBetaRoundTrip(tt.dgd)
			if err != nil {
				t.Fatalf("legacyWorkerHashAfterBetaRoundTrip: %v", err)
			}
			if roundTripHash != tt.want {
				t.Fatalf("round-tripped alpha hash = %q, want direct golden %q", roundTripHash, tt.want)
			}
		})
	}
}

func TestComputeDGDWorkersSpecHashTracksPodMetadata(t *testing.T) {
	baseHash, err := ComputeDGDWorkersSpecHash(legacyWorkerHashDGD())
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}

	mutated := legacyWorkerHashDGDWithPodMetadataChanges()

	got, err := ComputeDGDWorkersSpecHash(mutated)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}
	if got == baseHash {
		t.Fatalf("pod metadata change did not change preserved legacy worker hash: %q", got)
	}
}

func TestComputeDGDWorkersSpecHashIgnoresResourceMetadataAndNonWorkers(t *testing.T) {
	baseHash, err := ComputeDGDWorkersSpecHash(legacyWorkerHashDGD())
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}

	mutated := legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges()

	got, err := ComputeDGDWorkersSpecHash(mutated)
	if err != nil {
		t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
	}
	if got != baseHash {
		t.Fatalf("non-pod-template metadata/non-worker changes changed preserved legacy worker hash: got %q, want %q", got, baseHash)
	}
}

func FuzzComputeDGDWorkersSpecHashDeterministic(f *testing.F) {
	f.Add("worker", "MODEL", "llama", "checksum/config", "base", false)
	f.Add("prefill", "ROLE", "prefill", "rollout", "v1", true)
	f.Add("decode", "", "", "", "", false)

	f.Fuzz(func(t *testing.T, componentType, envName, envValue, metadataKey, metadataValue string, includeMainContainer bool) {
		switch componentType {
		case commonconsts.ComponentTypeWorker, commonconsts.ComponentTypePrefill, commonconsts.ComponentTypeDecode:
		default:
			componentType = commonconsts.ComponentTypeWorker
		}

		spec := &DynamoComponentDeploymentSharedSpec{
			ComponentType: componentType,
			Envs:          []corev1.EnvVar{{Name: envName, Value: envValue}},
			ExtraPodMetadata: &ExtraPodMetadata{
				Labels: map[string]string{metadataKey: metadataValue},
			},
		}
		if includeMainContainer {
			spec.ExtraPodSpec = &ExtraPodSpec{
				MainContainer: &corev1.Container{Name: commonconsts.MainContainerName, Image: envValue},
			}
		}

		dgd := legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
			"worker": spec,
		})
		got1, err := ComputeDGDWorkersSpecHash(dgd)
		if err != nil {
			t.Fatalf("ComputeDGDWorkersSpecHash: %v", err)
		}
		got2, err := ComputeDGDWorkersSpecHash(dgd)
		if err != nil {
			t.Fatalf("second ComputeDGDWorkersSpecHash: %v", err)
		}
		if got1 != got2 {
			t.Fatalf("hash is not deterministic: first %q, second %q", got1, got2)
		}
		if len(got1) != 8 {
			t.Fatalf("hash length = %d, want 8: %q", len(got1), got1)
		}
	})
}

func FuzzComputeDGDWorkersSpecHashConversionRoundTripStableSubset(f *testing.F) {
	f.Add("worker", "MODEL", "llama", "rollout", "base", "1", "4Gi")
	f.Add("prefill", "ROLE", "prefill", "checksum", "v1", "", "")
	f.Add("decode", "", "", "", "", "250m", "1Gi")

	f.Fuzz(func(t *testing.T, componentType, envName, envValue, metadataKey, metadataValue, cpu, memory string) {
		for _, value := range []string{componentType, envName, envValue, metadataKey, metadataValue, cpu, memory} {
			if !utf8.ValidString(value) {
				t.Skip("Kubernetes JSON strings are valid UTF-8")
			}
		}
		if cpu != "" {
			if _, err := resource.ParseQuantity(cpu); err != nil {
				t.Skip("invalid CPU quantity")
			}
		}
		if memory != "" {
			if _, err := resource.ParseQuantity(memory); err != nil {
				t.Skip("invalid memory quantity")
			}
		}
		switch componentType {
		case commonconsts.ComponentTypeWorker, commonconsts.ComponentTypePrefill, commonconsts.ComponentTypeDecode:
		default:
			componentType = commonconsts.ComponentTypeWorker
		}

		alpha := legacyWorkerHashDGDFromServices(map[string]*DynamoComponentDeploymentSharedSpec{
			"worker": {
				ComponentType: componentType,
				Envs:          []corev1.EnvVar{{Name: envName, Value: envValue}},
				ExtraPodMetadata: &ExtraPodMetadata{
					Labels: map[string]string{metadataKey: metadataValue},
				},
				Resources: &Resources{
					Requests: &ResourceItem{CPU: cpu, Memory: memory},
				},
			},
		})

		directHash, err := ComputeDGDWorkersSpecHash(alpha)
		if err != nil {
			t.Fatalf("ComputeDGDWorkersSpecHash(alpha): %v", err)
		}
		roundTripHash, err := legacyWorkerHashAfterBetaRoundTrip(alpha)
		if err != nil {
			t.Fatalf("legacyWorkerHashAfterBetaRoundTrip: %v", err)
		}
		if directHash != roundTripHash {
			t.Fatalf("round-trip changed stable-subset hash: direct %q, round-trip %q", directHash, roundTripHash)
		}
	})
}

func legacyWorkerHashAfterBetaRoundTrip(src *DynamoGraphDeployment) (string, error) {
	hub := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(hub); err != nil {
		return "", err
	}

	roundTripped := &DynamoGraphDeployment{}
	if err := roundTripped.ConvertFrom(hub); err != nil {
		return "", err
	}
	return ComputeDGDWorkersSpecHash(roundTripped)
}

func legacyWorkerHashDGDFromServices(services map[string]*DynamoComponentDeploymentSharedSpec) *DynamoGraphDeployment {
	return &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "legacy-worker-hash",
			Namespace: "ns",
		},
		Spec: DynamoGraphDeploymentSpec{
			Services: services,
		},
	}
}

func legacyWorkerHashDGD() *DynamoGraphDeployment {
	return &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "legacy-worker-hash",
			Namespace: "ns",
		},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: commonconsts.ComponentTypeFrontend,
					Envs:          []corev1.EnvVar{{Name: "FRONTEND_ONLY", Value: "ignored"}},
				},
				"worker": {
					ComponentType: commonconsts.ComponentTypeWorker,
					Annotations:   map[string]string{"resource": "base"},
					Labels:        map[string]string{"resource": "base"},
					Replicas:      ptr.To(int32(2)),
					Envs:          []corev1.EnvVar{{Name: "MODEL", Value: "llama"}},
					ExtraPodMetadata: &ExtraPodMetadata{
						Labels:      map[string]string{"rollout": "base"},
						Annotations: map[string]string{"checksum/config": "base"},
					},
				},
			},
		},
	}
}

func legacyWorkerHashDGDWithPodMetadataChanges() *DynamoGraphDeployment {
	mutated := legacyWorkerHashDGD()
	mutated.Spec.Services["worker"].ExtraPodMetadata = &ExtraPodMetadata{
		Labels:      map[string]string{"rollout": "changed"},
		Annotations: map[string]string{"checksum/config": "changed"},
	}
	return mutated
}

func legacyWorkerHashDGDWithResourceMetadataAndNonWorkerChanges() *DynamoGraphDeployment {
	mutated := legacyWorkerHashDGD()
	mutated.Spec.Services["worker"].Annotations = map[string]string{"resource": "changed"}
	mutated.Spec.Services["worker"].Labels = map[string]string{"resource": "changed"}
	mutated.Spec.Services["worker"].Replicas = ptr.To(int32(99))
	mutated.Spec.Services["worker"].Ingress = &IngressSpec{
		Enabled: true,
		Host:    "changed.example.com",
	}
	mutated.Spec.Services["frontend"].Envs = []corev1.EnvVar{{Name: "FRONTEND_ONLY", Value: "changed"}}
	return mutated
}
