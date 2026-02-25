/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func baseJob() *batchv1.Job {
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "profile-test",
			Namespace: "default",
			Labels: map[string]string{
				nvidiacomv1beta1.LabelApp:       nvidiacomv1beta1.LabelValueDynamoProfiler,
				nvidiacomv1beta1.LabelDGDR:      "my-dgdr",
				nvidiacomv1beta1.LabelManagedBy: nvidiacomv1beta1.LabelValueDynamoOperator,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: ptr.To[int32](3),
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						nvidiacomv1beta1.LabelApp:       nvidiacomv1beta1.LabelValueDynamoProfiler,
						nvidiacomv1beta1.LabelDGDR:      "my-dgdr",
						nvidiacomv1beta1.LabelManagedBy: nvidiacomv1beta1.LabelValueDynamoOperator,
					},
				},
				Spec: corev1.PodSpec{
					ServiceAccountName: "dgdr-profiling-job",
					RestartPolicy:      corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:    "profiler",
							Image:   "profiler:latest",
							Command: []string{"python", "-m", "dynamo.profiler"},
							Env: []corev1.EnvVar{
								{Name: "OUTPUT_DIR", Value: "/output"},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: "profiling-output", MountPath: "/output"},
							},
						},
						{
							Name:  "output-copier",
							Image: "busybox:latest",
						},
					},
					Volumes: []corev1.Volume{
						{Name: "profiling-output", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
					},
					ImagePullSecrets: []corev1.LocalObjectReference{
						{Name: "nvcr-imagepullsecret"},
					},
				},
			},
		},
	}
}

func TestApplyProfilingJobOverrides_NilOverrides(t *testing.T) {
	job := baseJob()
	original := job.Spec.BackoffLimit
	applyProfilingJobOverrides(job, nil)
	if job.Spec.BackoffLimit != original {
		t.Error("nil overrides should leave job unchanged")
	}
}

func TestApplyProfilingJobOverrides_BackoffLimit(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		BackoffLimit: ptr.To[int32](10),
	})
	if *job.Spec.BackoffLimit != 10 {
		t.Errorf("expected BackoffLimit=10, got %d", *job.Spec.BackoffLimit)
	}
}

func TestApplyProfilingJobOverrides_ActiveDeadlineSeconds(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		ActiveDeadlineSeconds: ptr.To[int64](3600),
	})
	if job.Spec.ActiveDeadlineSeconds == nil || *job.Spec.ActiveDeadlineSeconds != 3600 {
		t.Errorf("expected ActiveDeadlineSeconds=3600, got %v", job.Spec.ActiveDeadlineSeconds)
	}
}

func TestApplyProfilingJobOverrides_TTLSecondsAfterFinished(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		TTLSecondsAfterFinished: ptr.To[int32](600),
	})
	if job.Spec.TTLSecondsAfterFinished == nil || *job.Spec.TTLSecondsAfterFinished != 600 {
		t.Errorf("expected TTLSecondsAfterFinished=600, got %v", job.Spec.TTLSecondsAfterFinished)
	}
}

func TestApplyProfilingJobOverrides_Completions(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Completions: ptr.To[int32](5),
	})
	if job.Spec.Completions == nil || *job.Spec.Completions != 5 {
		t.Errorf("expected Completions=5, got %v", job.Spec.Completions)
	}
}

func TestApplyProfilingJobOverrides_Parallelism(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Parallelism: ptr.To[int32](2),
	})
	if job.Spec.Parallelism == nil || *job.Spec.Parallelism != 2 {
		t.Errorf("expected Parallelism=2, got %v", job.Spec.Parallelism)
	}
}

func TestApplyProfilingJobOverrides_Suspend(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Suspend: ptr.To(true),
	})
	if job.Spec.Suspend == nil || !*job.Spec.Suspend {
		t.Error("expected Suspend=true")
	}
}

func TestApplyProfilingJobOverrides_LabelsProtected(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					nvidiacomv1beta1.LabelApp:  "attacker",
					nvidiacomv1beta1.LabelDGDR: "attacker",
					"team":                     "ml-platform",
					"env":                      "staging",
				},
			},
		},
	})
	tmplLabels := job.Spec.Template.Labels
	if tmplLabels[nvidiacomv1beta1.LabelApp] != nvidiacomv1beta1.LabelValueDynamoProfiler {
		t.Errorf("protected label %q was overwritten", nvidiacomv1beta1.LabelApp)
	}
	if tmplLabels[nvidiacomv1beta1.LabelDGDR] != "my-dgdr" {
		t.Errorf("protected label %q was overwritten", nvidiacomv1beta1.LabelDGDR)
	}
	if tmplLabels["team"] != "ml-platform" {
		t.Error("user label 'team' was not applied")
	}
	if tmplLabels["env"] != "staging" {
		t.Error("user label 'env' was not applied")
	}
}

func TestApplyProfilingJobOverrides_Annotations(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					"prometheus.io/scrape": "true",
					"custom/note":         "profiling-test",
				},
			},
		},
	})
	ann := job.Spec.Template.Annotations
	if ann["prometheus.io/scrape"] != "true" {
		t.Error("annotation prometheus.io/scrape not applied")
	}
	if ann["custom/note"] != "profiling-test" {
		t.Error("annotation custom/note not applied")
	}
}

func TestApplyProfilingJobOverrides_Tolerations(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Tolerations: []corev1.Toleration{
					{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule},
				},
			},
		},
	})
	if len(job.Spec.Template.Spec.Tolerations) != 1 || job.Spec.Template.Spec.Tolerations[0].Key != "nvidia.com/gpu" {
		t.Error("tolerations not applied")
	}
}

func TestApplyProfilingJobOverrides_NodeSelector(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				NodeSelector: map[string]string{"gpu-type": "H100"},
			},
		},
	})
	if job.Spec.Template.Spec.NodeSelector["gpu-type"] != "H100" {
		t.Error("nodeSelector not applied")
	}
}

func TestApplyProfilingJobOverrides_Affinity(t *testing.T) {
	job := baseJob()
	affinity := &corev1.Affinity{
		NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{Key: "gpu", Operator: corev1.NodeSelectorOpIn, Values: []string{"H100"}},
						},
					},
				},
			},
		},
	}
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{Affinity: affinity},
		},
	})
	if job.Spec.Template.Spec.Affinity != affinity {
		t.Error("affinity not applied")
	}
}

func TestApplyProfilingJobOverrides_PriorityClassName(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{PriorityClassName: "high-priority"},
		},
	})
	if job.Spec.Template.Spec.PriorityClassName != "high-priority" {
		t.Error("priorityClassName not applied")
	}
}

func TestApplyProfilingJobOverrides_ServiceAccountName(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{ServiceAccountName: "custom-sa"},
		},
	})
	if job.Spec.Template.Spec.ServiceAccountName != "custom-sa" {
		t.Error("serviceAccountName not applied")
	}
}

func TestApplyProfilingJobOverrides_ImagePullSecrets_Merge(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "nvcr-imagepullsecret"},
					{Name: "my-registry-secret"},
				},
			},
		},
	})
	secrets := job.Spec.Template.Spec.ImagePullSecrets
	if len(secrets) != 2 {
		t.Fatalf("expected 2 secrets (base deduped + new appended), got %d: %v", len(secrets), secrets)
	}
	if secrets[0].Name != "nvcr-imagepullsecret" {
		t.Errorf("expected base secret first, got %s", secrets[0].Name)
	}
	if secrets[1].Name != "my-registry-secret" {
		t.Errorf("expected override secret second, got %s", secrets[1].Name)
	}
}

func TestApplyProfilingJobOverrides_ImagePullSecrets_NoDuplicates(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "nvcr-imagepullsecret"},
				},
			},
		},
	})
	secrets := job.Spec.Template.Spec.ImagePullSecrets
	if len(secrets) != 1 {
		t.Fatalf("expected 1 secret (duplicate should not be added), got %d: %v", len(secrets), secrets)
	}
}

func TestApplyProfilingJobOverrides_RuntimeClassName(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{RuntimeClassName: ptr.To("nvidia")},
		},
	})
	if job.Spec.Template.Spec.RuntimeClassName == nil || *job.Spec.Template.Spec.RuntimeClassName != "nvidia" {
		t.Error("runtimeClassName not applied")
	}
}

func TestApplyProfilingJobOverrides_DNSConfigAndPolicy(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				DNSPolicy: corev1.DNSNone,
				DNSConfig: &corev1.PodDNSConfig{
					Nameservers: []string{"8.8.8.8"},
				},
			},
		},
	})
	if job.Spec.Template.Spec.DNSPolicy != corev1.DNSNone {
		t.Error("dnsPolicy not applied")
	}
	if job.Spec.Template.Spec.DNSConfig == nil || len(job.Spec.Template.Spec.DNSConfig.Nameservers) != 1 {
		t.Error("dnsConfig not applied")
	}
}

func TestApplyProfilingJobOverrides_VolumesDedup(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Volumes: []corev1.Volume{
					{Name: "profiling-output", VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "user-pvc"},
					}},
					{Name: "extra-config", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				},
			},
		},
	})
	vols := job.Spec.Template.Spec.Volumes
	if len(vols) != 2 {
		t.Fatalf("expected 2 volumes (deduped + appended), got %d", len(vols))
	}
	if vols[0].Name != "profiling-output" || vols[0].PersistentVolumeClaim == nil {
		t.Error("volume 'profiling-output' was not replaced by user override")
	}
	if vols[1].Name != "extra-config" {
		t.Error("new volume 'extra-config' was not appended")
	}
}

func TestApplyProfilingJobOverrides_InitContainers(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{
					{Name: "model-downloader", Image: "downloader:v1"},
				},
			},
		},
	})
	if len(job.Spec.Template.Spec.InitContainers) != 1 || job.Spec.Template.Spec.InitContainers[0].Name != "model-downloader" {
		t.Error("initContainers not appended")
	}
}

func TestApplyProfilingJobOverrides_ContainerResources(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("4"),
								corev1.ResourceMemory: resource.MustParse("16Gi"),
							},
						},
					},
				},
			},
		},
	})
	limits := job.Spec.Template.Spec.Containers[0].Resources.Limits
	if limits.Cpu().String() != "4" {
		t.Errorf("expected CPU limit=4, got %s", limits.Cpu().String())
	}
	if limits.Memory().String() != "16Gi" {
		t.Errorf("expected memory limit=16Gi, got %s", limits.Memory().String())
	}
}

func TestApplyProfilingJobOverrides_ContainerImage(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Image: "custom-profiler:v2"},
				},
			},
		},
	})
	if job.Spec.Template.Spec.Containers[0].Image != "custom-profiler:v2" {
		t.Errorf("expected image=custom-profiler:v2, got %s", job.Spec.Template.Spec.Containers[0].Image)
	}
}

func TestApplyProfilingJobOverrides_ContainerEnvDedup(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Env: []corev1.EnvVar{
							{Name: "OUTPUT_DIR", Value: "/custom-output"},
							{Name: "HF_TOKEN", Value: "secret"},
						},
					},
				},
			},
		},
	})
	envs := job.Spec.Template.Spec.Containers[0].Env
	if len(envs) != 2 {
		t.Fatalf("expected 2 env vars (deduped + appended), got %d", len(envs))
	}
	envMap := make(map[string]string, len(envs))
	for _, e := range envs {
		envMap[e.Name] = e.Value
	}
	if envMap["OUTPUT_DIR"] != "/custom-output" {
		t.Error("env OUTPUT_DIR not overridden by user")
	}
	if envMap["HF_TOKEN"] != "secret" {
		t.Error("env HF_TOKEN not appended")
	}
}

func TestApplyProfilingJobOverrides_ContainerVolumeMountsDedup(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						VolumeMounts: []corev1.VolumeMount{
							{Name: "profiling-output", MountPath: "/custom-output"},
							{Name: "hf-cache", MountPath: "/root/.cache/huggingface"},
						},
					},
				},
			},
		},
	})
	mounts := job.Spec.Template.Spec.Containers[0].VolumeMounts
	if len(mounts) != 2 {
		t.Fatalf("expected 2 volume mounts, got %d", len(mounts))
	}
	mountMap := make(map[string]string, len(mounts))
	for _, m := range mounts {
		mountMap[m.Name] = m.MountPath
	}
	if mountMap["profiling-output"] != "/custom-output" {
		t.Error("mount 'profiling-output' not overridden")
	}
	if mountMap["hf-cache"] != "/root/.cache/huggingface" {
		t.Error("mount 'hf-cache' not appended")
	}
}

func TestApplyProfilingJobOverrides_ContainerEnvFrom(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						EnvFrom: []corev1.EnvFromSource{
							{ConfigMapRef: &corev1.ConfigMapEnvSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: "profiler-env"},
							}},
						},
					},
				},
			},
		},
	})
	if len(job.Spec.Template.Spec.Containers[0].EnvFrom) != 1 {
		t.Error("envFrom not appended")
	}
}

func TestApplyProfilingJobOverrides_ContainerSecurityContext(t *testing.T) {
	job := baseJob()
	sc := &corev1.SecurityContext{
		Privileged: ptr.To(false),
		RunAsUser:  ptr.To[int64](2000),
	}
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{SecurityContext: sc},
				},
			},
		},
	})
	if job.Spec.Template.Spec.Containers[0].SecurityContext != sc {
		t.Error("container securityContext not applied")
	}
}

func TestApplyProfilingJobOverrides_CommandAndArgsPreserved(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Image: "custom:v1",
						Env:   []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
					},
				},
			},
		},
	})
	c := job.Spec.Template.Spec.Containers[0]
	if len(c.Command) == 0 || c.Command[0] != "python" {
		t.Error("command was unexpectedly overwritten")
	}
}

func TestApplyProfilingJobOverrides_SidecarUntouched(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Image: "custom:v1"},
				},
			},
		},
	})
	if len(job.Spec.Template.Spec.Containers) != 2 {
		t.Fatal("sidecar container was removed")
	}
	if job.Spec.Template.Spec.Containers[1].Name != "output-copier" {
		t.Error("sidecar container was modified")
	}
	if job.Spec.Template.Spec.Containers[1].Image != "busybox:latest" {
		t.Error("sidecar image was modified")
	}
}

func TestApplyProfilingJobOverrides_Combined(t *testing.T) {
	job := baseJob()
	applyProfilingJobOverrides(job, &batchv1.JobSpec{
		BackoffLimit:            ptr.To[int32](5),
		ActiveDeadlineSeconds:   ptr.To[int64](7200),
		TTLSecondsAfterFinished: ptr.To[int32](300),
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels:      map[string]string{"team": "infra", nvidiacomv1beta1.LabelApp: "should-be-ignored"},
				Annotations: map[string]string{"note": "combined-test"},
			},
			Spec: corev1.PodSpec{
				Tolerations:        []corev1.Toleration{{Key: "gpu", Operator: corev1.TolerationOpExists}},
				NodeSelector:       map[string]string{"zone": "us-west"},
				PriorityClassName:  "batch",
				ServiceAccountName: "custom-sa",
				Volumes: []corev1.Volume{
					{Name: "scratch", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
				},
				Containers: []corev1.Container{
					{
						Image: "profiler:v2",
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("8")},
						},
						Env: []corev1.EnvVar{
							{Name: "OUTPUT_DIR", Value: "/v2-output"},
							{Name: "CUDA_VISIBLE_DEVICES", Value: "0,1"},
						},
						VolumeMounts: []corev1.VolumeMount{
							{Name: "scratch", MountPath: "/scratch"},
						},
					},
				},
			},
		},
	})

	if *job.Spec.BackoffLimit != 5 {
		t.Errorf("BackoffLimit: want 5, got %d", *job.Spec.BackoffLimit)
	}
	if *job.Spec.ActiveDeadlineSeconds != 7200 {
		t.Errorf("ActiveDeadlineSeconds: want 7200, got %d", *job.Spec.ActiveDeadlineSeconds)
	}
	if *job.Spec.TTLSecondsAfterFinished != 300 {
		t.Errorf("TTLSecondsAfterFinished: want 300, got %d", *job.Spec.TTLSecondsAfterFinished)
	}

	tmpl := job.Spec.Template
	if tmpl.Labels[nvidiacomv1beta1.LabelApp] != nvidiacomv1beta1.LabelValueDynamoProfiler {
		t.Error("protected label was overwritten in combined test")
	}
	if tmpl.Labels["team"] != "infra" {
		t.Error("user label 'team' missing")
	}
	if tmpl.Annotations["note"] != "combined-test" {
		t.Error("annotation missing")
	}

	spec := tmpl.Spec
	if len(spec.Tolerations) != 1 {
		t.Error("tolerations wrong")
	}
	if spec.NodeSelector["zone"] != "us-west" {
		t.Error("nodeSelector wrong")
	}
	if spec.PriorityClassName != "batch" {
		t.Error("priorityClassName wrong")
	}
	if spec.ServiceAccountName != "custom-sa" {
		t.Error("serviceAccountName wrong")
	}

	if len(spec.Volumes) != 2 {
		t.Errorf("expected 2 volumes, got %d", len(spec.Volumes))
	}

	profiler := spec.Containers[0]
	if profiler.Image != "profiler:v2" {
		t.Errorf("image: want profiler:v2, got %s", profiler.Image)
	}
	if profiler.Command[0] != "python" {
		t.Error("command was overwritten")
	}
	if profiler.Resources.Limits.Cpu().String() != "8" {
		t.Error("resources wrong")
	}
	if len(profiler.Env) != 2 {
		t.Errorf("expected 2 envs (OUTPUT_DIR deduped + CUDA_VISIBLE_DEVICES appended), got %d", len(profiler.Env))
	}
	if len(profiler.VolumeMounts) != 2 {
		t.Errorf("expected 2 volume mounts, got %d", len(profiler.VolumeMounts))
	}

	sidecar := spec.Containers[1]
	if sidecar.Name != "output-copier" || sidecar.Image != "busybox:latest" {
		t.Error("sidecar was modified")
	}
}

func TestMergeNamedSlice_EmptyOverrides(t *testing.T) {
	base := []corev1.EnvVar{{Name: "A", Value: "1"}}
	result := mergeNamedSlice(base, nil, func(e corev1.EnvVar) string { return e.Name })
	if len(result) != 1 {
		t.Errorf("expected 1, got %d", len(result))
	}
}

func TestMergeNamedSlice_EmptyBase(t *testing.T) {
	overrides := []corev1.EnvVar{{Name: "A", Value: "1"}}
	result := mergeNamedSlice(nil, overrides, func(e corev1.EnvVar) string { return e.Name })
	if len(result) != 1 || result[0].Value != "1" {
		t.Errorf("expected [A=1], got %v", result)
	}
}

func TestMergeNamedSlice_PreservesOrder(t *testing.T) {
	base := []corev1.EnvVar{
		{Name: "B", Value: "2"},
		{Name: "A", Value: "1"},
	}
	overrides := []corev1.EnvVar{
		{Name: "A", Value: "override"},
		{Name: "C", Value: "3"},
	}
	result := mergeNamedSlice(base, overrides, func(e corev1.EnvVar) string { return e.Name })
	if len(result) != 3 {
		t.Fatalf("expected 3, got %d", len(result))
	}
	if result[0].Name != "B" || result[1].Name != "A" || result[2].Name != "C" {
		t.Errorf("order wrong: %v", result)
	}
	if result[1].Value != "override" {
		t.Errorf("A not overridden: %s", result[1].Value)
	}
}
