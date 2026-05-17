// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"math"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewRestorePod(t *testing.T) {
	readinessProbe := &corev1.Probe{PeriodSeconds: 7, TimeoutSeconds: 3}
	livenessProbe := &corev1.Probe{InitialDelaySeconds: 11}
	startupProbe := &corev1.Probe{FailureThreshold: 120}
	restorePod, err := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "worker",
			Labels: map[string]string{"existing": "label"},
			Annotations: map[string]string{
				"existing":                 "annotation",
				TargetContainersAnnotation: "main",
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
			Containers: []corev1.Container{{
				Name:           "main",
				Image:          "test:latest",
				Command:        []string{"python3", "-m", "dynamo.vllm"},
				Args:           []string{"--model", "Qwen"},
				ReadinessProbe: readinessProbe.DeepCopy(),
				LivenessProbe:  livenessProbe.DeepCopy(),
				StartupProbe:   startupProbe.DeepCopy(),
			}},
		},
	}, PodOptions{
		Namespace:       "test-ns",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Storage: Storage{
			Type:     StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		},
		SeccompProfile: DefaultSeccompLocalhostProfile,
	})
	if err != nil {
		t.Fatalf("NewRestorePod returned error: %v", err)
	}

	if restorePod.Name != "worker" || restorePod.Namespace != "test-ns" {
		t.Fatalf("unexpected restore pod identity: %#v", restorePod.ObjectMeta)
	}
	if restorePod.Labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint id label: %#v", restorePod.Labels)
	}
	if _, has := restorePod.Labels[CheckpointSourceLabel]; has {
		t.Fatalf("restore pod must not carry the checkpoint-source label: %#v", restorePod.Labels)
	}
	if restorePod.Annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation: %#v", restorePod.Annotations)
	}
	if restorePod.Annotations[TargetContainersAnnotation] != "main" {
		t.Fatalf("expected target-containers annotation to be preserved: %#v", restorePod.Annotations)
	}
	if restorePod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %#v", restorePod.Spec.RestartPolicy)
	}
	main := &restorePod.Spec.Containers[0]
	if len(main.Command) != 2 || main.Command[0] != "sleep" || main.Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", main.Command)
	}
	if main.Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", main.Args)
	}
	if main.ReadinessProbe == nil {
		t.Fatalf("expected readiness probe to be preserved")
	}
	if got := main.ReadinessProbe.PeriodSeconds; got != readinessProbe.PeriodSeconds {
		t.Fatalf("expected readiness probe period %d, got %d", readinessProbe.PeriodSeconds, got)
	}
	if main.LivenessProbe == nil {
		t.Fatalf("expected liveness probe to be preserved")
	}
	if got := main.LivenessProbe.InitialDelaySeconds; got != livenessProbe.InitialDelaySeconds {
		t.Fatalf("expected liveness initial delay %d, got %d", livenessProbe.InitialDelaySeconds, got)
	}
	if main.StartupProbe == nil {
		t.Fatalf("expected startup probe to gate restore completion")
	}
	assertRestoreStartupGate(t, main.StartupProbe)
	if restorePod.Spec.SecurityContext == nil || restorePod.Spec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", restorePod.Spec.SecurityContext)
	}
	if len(restorePod.Spec.Volumes) != 2 {
		t.Fatalf("expected checkpoint and snapshot-control volumes, got %#v", restorePod.Spec.Volumes)
	}
	if len(main.VolumeMounts) != 2 {
		t.Fatalf("expected checkpoint and snapshot-control mounts, got %#v", main.VolumeMounts)
	}
	foundMount := false
	for _, m := range main.VolumeMounts {
		if m.Name == SnapshotControlVolumeName {
			foundMount = true
			if m.SubPath != "main" {
				t.Fatalf("expected subPath=main, got %q", m.SubPath)
			}
			break
		}
	}
	if !foundMount {
		t.Fatalf("expected %s mount, got %#v", SnapshotControlVolumeName, main.VolumeMounts)
	}
	foundEnv := false
	for _, e := range main.Env {
		if e.Name == SnapshotControlDirEnv {
			foundEnv = true
			break
		}
	}
	if !foundEnv {
		t.Fatalf("expected %s env, got %#v", SnapshotControlDirEnv, main.Env)
	}
}

func TestNewRestorePodShapesMultipleTargets(t *testing.T) {
	restorePod, err := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "failover-worker",
			Annotations: map[string]string{TargetContainersAnnotation: "engine-0,engine-1"},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "engine-0", Image: "test:latest", Command: []string{"python3"}, Args: []string{"--serve"}},
				{Name: "engine-1", Image: "test:latest", Command: []string{"python3"}, Args: []string{"--serve"}},
				{Name: "sidecar", Image: "sidecar:latest", Command: []string{"sidecar"}, Args: []string{"run"}},
			},
		},
	}, PodOptions{
		Namespace:       "test-ns",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Storage: Storage{
			Type:     StorageTypePVC,
			PVCName:  "snapshot-pvc",
			BasePath: "/checkpoints",
		},
		SeccompProfile: DefaultSeccompLocalhostProfile,
	})
	if err != nil {
		t.Fatalf("NewRestorePod returned error: %v", err)
	}

	for _, name := range []string{"engine-0", "engine-1"} {
		c := findRestoreContainer(t, restorePod.Spec.Containers, name)
		if len(c.Command) != 2 || c.Command[0] != "sleep" || c.Command[1] != "infinity" {
			t.Fatalf("expected placeholder command on %s, got %#v", name, c.Command)
		}
		if c.Args != nil {
			t.Fatalf("expected args cleared on %s, got %#v", name, c.Args)
		}
		assertRestoreStartupGate(t, c.StartupProbe)
		found := false
		for _, m := range c.VolumeMounts {
			if m.Name == SnapshotControlVolumeName {
				found = true
				if m.SubPath != name {
					t.Fatalf("expected subPath=%s on %s, got %q", name, name, m.SubPath)
				}
			}
		}
		if !found {
			t.Fatalf("expected %s mount on %s, got %#v", SnapshotControlVolumeName, name, c.VolumeMounts)
		}
	}

	sidecar := findRestoreContainer(t, restorePod.Spec.Containers, "sidecar")
	if len(sidecar.Command) != 1 || sidecar.Command[0] != "sidecar" {
		t.Fatalf("sidecar command must not be rewritten, got %#v", sidecar.Command)
	}
	for _, m := range sidecar.VolumeMounts {
		if m.Name == SnapshotControlVolumeName {
			t.Fatalf("sidecar must not get a control mount: %#v", sidecar.VolumeMounts)
		}
	}
	for _, e := range sidecar.Env {
		if e.Name == SnapshotControlDirEnv {
			t.Fatalf("sidecar must not get a control env: %#v", sidecar.Env)
		}
	}
}

func TestNewRestorePodRequiresAnnotation(t *testing.T) {
	_, err := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "no-annotation"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
		},
	}, PodOptions{
		Namespace:       "test-ns",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Storage:         Storage{Type: StorageTypePVC, PVCName: "snapshot-pvc", BasePath: "/checkpoints"},
		SeccompProfile:  DefaultSeccompLocalhostProfile,
	})
	if err == nil || !strings.Contains(err.Error(), TargetContainersAnnotation) {
		t.Fatalf("expected missing-annotation error, got %v", err)
	}
}

func TestNewRestorePodRejectsUnknownContainer(t *testing.T) {
	_, err := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "bad-target",
			Annotations: map[string]string{TargetContainersAnnotation: "ghost"},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
		},
	}, PodOptions{
		Namespace:       "test-ns",
		CheckpointID:    "hash",
		ArtifactVersion: "2",
		Storage:         Storage{Type: StorageTypePVC, PVCName: "snapshot-pvc", BasePath: "/checkpoints"},
		SeccompProfile:  DefaultSeccompLocalhostProfile,
	})
	if err == nil || !strings.Contains(err.Error(), `"ghost"`) {
		t.Fatalf("expected unknown-container error, got %v", err)
	}
}

func TestPrepareRestorePodSpec(t *testing.T) {
	podSpec := corev1.PodSpec{}
	readinessProbe := &corev1.Probe{PeriodSeconds: 13, SuccessThreshold: 1}
	livenessProbe := &corev1.Probe{TimeoutSeconds: 5}
	startupProbe := &corev1.Probe{FailureThreshold: 60}
	podSpec.Containers = []corev1.Container{{
		Name:           "main",
		Command:        []string{"python3", "-m", "dynamo.vllm"},
		Args:           []string{"--model", "Qwen"},
		ReadinessProbe: readinessProbe.DeepCopy(),
		LivenessProbe:  livenessProbe.DeepCopy(),
		StartupProbe:   startupProbe.DeepCopy(),
	}}

	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}
	annotations := map[string]string{TargetContainersAnnotation: "main"}
	if err := PrepareRestorePodSpec(&podSpec, annotations, storage, DefaultSeccompLocalhostProfile, true); err != nil {
		t.Fatalf("first PrepareRestorePodSpec error: %v", err)
	}
	if err := PrepareRestorePodSpec(&podSpec, annotations, storage, DefaultSeccompLocalhostProfile, true); err != nil {
		t.Fatalf("second PrepareRestorePodSpec error: %v", err)
	}

	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", podSpec.SecurityContext)
	}
	if len(podSpec.Volumes) != 2 {
		t.Fatalf("expected checkpoint and snapshot-control volumes, got %#v", podSpec.Volumes)
	}
	container := &podSpec.Containers[0]
	if len(container.VolumeMounts) != 2 {
		t.Fatalf("expected checkpoint and snapshot-control mounts, got %#v", container.VolumeMounts)
	}
	volCount := 0
	for _, v := range podSpec.Volumes {
		if v.Name == SnapshotControlVolumeName {
			volCount++
		}
	}
	if volCount != 1 {
		t.Fatalf("expected single %s volume after repeated calls, got %#v", SnapshotControlVolumeName, podSpec.Volumes)
	}
	mountCount := 0
	for _, m := range container.VolumeMounts {
		if m.Name == SnapshotControlVolumeName {
			mountCount++
		}
	}
	if mountCount != 1 {
		t.Fatalf("expected single %s mount after repeated calls, got %#v", SnapshotControlVolumeName, container.VolumeMounts)
	}
	envCount := 0
	for _, e := range container.Env {
		if e.Name == SnapshotControlDirEnv {
			envCount++
		}
	}
	if envCount != 1 {
		t.Fatalf("expected single %s env after repeated calls, got %#v", SnapshotControlDirEnv, container.Env)
	}
	if len(container.Command) != 2 || container.Command[0] != "sleep" || container.Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", container.Command)
	}
	if container.Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", container.Args)
	}
	if container.ReadinessProbe == nil {
		t.Fatalf("expected readiness probe to be preserved")
	}
	if got := container.ReadinessProbe.PeriodSeconds; got != readinessProbe.PeriodSeconds {
		t.Fatalf("expected readiness probe period %d, got %d", readinessProbe.PeriodSeconds, got)
	}
	if container.LivenessProbe == nil {
		t.Fatalf("expected liveness probe to be preserved")
	}
	if got := container.LivenessProbe.TimeoutSeconds; got != livenessProbe.TimeoutSeconds {
		t.Fatalf("expected liveness timeout %d, got %d", livenessProbe.TimeoutSeconds, got)
	}
	if container.StartupProbe == nil {
		t.Fatalf("expected startup probe to gate restore completion")
	}
	assertRestoreStartupGate(t, container.StartupProbe)
}

func TestPrepareRestorePodSpecSynthesizesStartupProbeFromLiveness(t *testing.T) {
	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: "/livez"},
		},
		PeriodSeconds:    5,
		TimeoutSeconds:   4,
		FailureThreshold: 2,
	}
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:          "main",
			Command:       []string{"python3", "-m", "dynamo.vllm"},
			Args:          []string{"--model", "Qwen"},
			LivenessProbe: livenessProbe.DeepCopy(),
		}},
	}

	if err := PrepareRestorePodSpec(&podSpec, map[string]string{TargetContainersAnnotation: "main"}, Storage{}, "", true); err != nil {
		t.Fatalf("PrepareRestorePodSpec error: %v", err)
	}

	container := &podSpec.Containers[0]
	if container.LivenessProbe == nil {
		t.Fatalf("expected liveness probe to be preserved")
	}
	if container.StartupProbe == nil {
		t.Fatalf("expected startup probe to gate restore completion")
	}
	assertRestoreStartupGate(t, container.StartupProbe)
	if container.StartupProbe.HTTPGet == nil || container.StartupProbe.HTTPGet.Path != "/livez" {
		t.Fatalf("expected synthesized startup probe to inherit liveness HTTPGet handler, got %#v", container.StartupProbe)
	}
}

func TestPrepareRestorePodSpecSynthesizesStartupProbeFromReadiness(t *testing.T) {
	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: []string{"cat", "/tmp/ready"}},
		},
		PeriodSeconds:    13,
		SuccessThreshold: 3,
		FailureThreshold: 4,
	}
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:           "main",
			Command:        []string{"python3", "-m", "dynamo.vllm"},
			Args:           []string{"--model", "Qwen"},
			ReadinessProbe: readinessProbe.DeepCopy(),
		}},
	}

	if err := PrepareRestorePodSpec(&podSpec, map[string]string{TargetContainersAnnotation: "main"}, Storage{}, "", true); err != nil {
		t.Fatalf("PrepareRestorePodSpec error: %v", err)
	}

	container := &podSpec.Containers[0]
	if container.ReadinessProbe == nil {
		t.Fatalf("expected readiness probe to be preserved")
	}
	if got := container.ReadinessProbe.SuccessThreshold; got != readinessProbe.SuccessThreshold {
		t.Fatalf("expected readiness success threshold %d, got %d", readinessProbe.SuccessThreshold, got)
	}
	if container.StartupProbe == nil {
		t.Fatalf("expected startup probe to gate restore completion")
	}
	assertRestoreStartupGate(t, container.StartupProbe)
	if container.StartupProbe.Exec == nil || len(container.StartupProbe.Exec.Command) != 2 ||
		container.StartupProbe.Exec.Command[0] != "cat" || container.StartupProbe.Exec.Command[1] != "/tmp/ready" {
		t.Fatalf("expected synthesized startup probe to inherit readiness Exec command, got %#v", container.StartupProbe)
	}
}

func TestPrepareRestorePodSpecFallsBackToSentinelWhenNoProbe(t *testing.T) {
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    "main",
			Command: []string{"python3", "-m", "dynamo.vllm"},
			Args:    []string{"--model", "Qwen"},
		}},
	}

	if err := PrepareRestorePodSpec(&podSpec, map[string]string{TargetContainersAnnotation: "main"}, Storage{}, "", true); err != nil {
		t.Fatalf("PrepareRestorePodSpec error: %v", err)
	}

	container := &podSpec.Containers[0]
	if container.StartupProbe == nil {
		t.Fatalf("expected sentinel-cat fallback startup probe to be installed")
	}
	assertRestoreStartupGate(t, container.StartupProbe)
	if container.StartupProbe.Exec == nil {
		t.Fatalf("expected fallback startup probe to use Exec handler, got %#v", container.StartupProbe)
	}
	want := []string{"cat", SnapshotControlMountPath + "/" + RestoreCompleteFile}
	if len(container.StartupProbe.Exec.Command) != len(want) {
		t.Fatalf("fallback startup probe command = %#v, want %#v", container.StartupProbe.Exec.Command, want)
	}
	for i := range want {
		if container.StartupProbe.Exec.Command[i] != want[i] {
			t.Fatalf("fallback startup probe command = %#v, want %#v", container.StartupProbe.Exec.Command, want)
		}
	}
}

// assertRestoreStartupGate verifies the load-bearing invariants every restore
// StartupProbe must satisfy: effectively infinite retries during CRIU restore
// and SuccessThreshold=1.
func assertRestoreStartupGate(t *testing.T, probe *corev1.Probe) {
	t.Helper()
	if probe == nil {
		t.Fatalf("expected non-nil startup probe")
	}
	if got := probe.FailureThreshold; got != math.MaxInt32 {
		t.Fatalf("expected startup failure threshold %d, got %d", math.MaxInt32, got)
	}
	if got := probe.SuccessThreshold; got != 1 {
		t.Fatalf("expected startup success threshold 1, got %d", got)
	}
}

func validRestoreSpecFixture(profile string, targets ...string) (*corev1.PodSpec, map[string]string) {
	if len(targets) == 0 {
		targets = []string{"main"}
	}
	containers := make([]corev1.Container, 0, len(targets))
	for _, name := range targets {
		container := corev1.Container{
			Name: name,
			VolumeMounts: []corev1.VolumeMount{
				{Name: CheckpointVolumeName, MountPath: "/checkpoints"},
				{Name: SnapshotControlVolumeName, MountPath: SnapshotControlMountPath, SubPath: name},
			},
			Env: []corev1.EnvVar{{Name: SnapshotControlDirEnv, Value: SnapshotControlMountPath}},
		}
		ensureRestoreStartupProbe(&container)
		containers = append(containers, container)
	}
	return &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type:             corev1.SeccompProfileTypeLocalhost,
				LocalhostProfile: &profile,
			},
		},
		Volumes: []corev1.Volume{
			{
				Name: CheckpointVolumeName,
				VolumeSource: corev1.VolumeSource{
					PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
						ClaimName: "snapshot-pvc",
					},
				},
			},
			{
				Name:         SnapshotControlVolumeName,
				VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
			},
		},
		Containers: containers,
	}, map[string]string{TargetContainersAnnotation: FormatTargetContainers(targets)}
}

func TestValidateRestorePodSpec(t *testing.T) {
	profile := DefaultSeccompLocalhostProfile
	podSpec, annotations := validRestoreSpecFixture(profile)
	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, annotations, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected restore pod spec to be valid, got %v", err)
	}

	badSpec := podSpec.DeepCopy()
	badSpec.Volumes = []corev1.Volume{badSpec.Volumes[1]}
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing checkpoint-storage volume for PVC snapshot-pvc" {
		t.Fatalf("expected missing volume error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Containers[0].VolumeMounts = []corev1.VolumeMount{badSpec.Containers[0].VolumeMounts[1]}
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != `missing checkpoint-storage mount at /checkpoints on container "main"` {
		t.Fatalf("expected missing mount error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Volumes = []corev1.Volume{badSpec.Volumes[0]}
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != fmt.Sprintf("missing %s emptyDir volume; add it via snapshotprotocol.EnsureControlVolume", SnapshotControlVolumeName) {
		t.Fatalf("expected missing control volume error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Containers[0].VolumeMounts = []corev1.VolumeMount{badSpec.Containers[0].VolumeMounts[0]}
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != fmt.Sprintf(`missing %s mount at %s on container "main"`, SnapshotControlVolumeName, SnapshotControlMountPath) {
		t.Fatalf("expected missing control mount error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Containers[0].Env = nil
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != fmt.Sprintf(`missing %s env var on container "main"`, SnapshotControlDirEnv) {
		t.Fatalf("expected missing control env error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Containers[0].StartupProbe = nil
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != `missing restore-complete startup probe on container "main"` {
		t.Fatalf("expected missing restore startup probe error, got %v", err)
	}

	// A non-sentinel startup probe is now accepted: ensureRestoreStartupProbe
	// synthesizes the probe from the workload's existing Startup/Liveness/
	// Readiness handler, so the validator only checks for presence, not a
	// fixed exec-command shape. Only fully-missing probes are rejected.
	okSpec := podSpec.DeepCopy()
	okSpec.Containers[0].StartupProbe = &corev1.Probe{
		ProbeHandler:     corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/livez"}},
		PeriodSeconds:    5,
		FailureThreshold: math.MaxInt32,
		SuccessThreshold: 1,
	}
	if err := ValidateRestorePodSpec(okSpec, annotations, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected synthesized HTTPGet startup probe to validate, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.SecurityContext = nil
	if err := ValidateRestorePodSpec(badSpec, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing localhost seccomp profile" {
		t.Fatalf("expected missing seccomp error, got %v", err)
	}

	if err := ValidateRestorePodSpec(podSpec, map[string]string{}, storage, DefaultSeccompLocalhostProfile); err == nil || !strings.Contains(err.Error(), TargetContainersAnnotation) {
		t.Fatalf("expected missing-annotation error, got %v", err)
	}
}

func TestValidateRestorePodSpecMultipleTargets(t *testing.T) {
	podSpec, annotations := validRestoreSpecFixture(DefaultSeccompLocalhostProfile, "engine-0", "engine-1")
	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}
	if err := ValidateRestorePodSpec(podSpec, annotations, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected multi-target validation to pass, got %v", err)
	}

	// Drop the engine-1 control mount → validation should fail for that target specifically.
	bad := podSpec.DeepCopy()
	for i := range bad.Containers {
		if bad.Containers[i].Name == "engine-1" {
			bad.Containers[i].VolumeMounts = []corev1.VolumeMount{bad.Containers[i].VolumeMounts[0]}
		}
	}
	if err := ValidateRestorePodSpec(bad, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || !strings.Contains(err.Error(), `"engine-1"`) {
		t.Fatalf("expected missing-mount error on engine-1, got %v", err)
	}

	badSubPath := podSpec.DeepCopy()
	for i := range badSubPath.Containers {
		if badSubPath.Containers[i].Name == "engine-1" {
			for j := range badSubPath.Containers[i].VolumeMounts {
				if badSubPath.Containers[i].VolumeMounts[j].Name == SnapshotControlVolumeName {
					badSubPath.Containers[i].VolumeMounts[j].SubPath = "wrong-engine"
				}
			}
		}
	}
	if err := ValidateRestorePodSpec(badSubPath, annotations, storage, DefaultSeccompLocalhostProfile); err == nil || !strings.Contains(err.Error(), `"engine-1"`) || !strings.Contains(err.Error(), "SubPath") {
		t.Fatalf("expected bad subPath error on engine-1, got %v", err)
	}
}

func TestValidateRestorePodSpecAllowsWorkerWithSidecars(t *testing.T) {
	podSpec, annotations := validRestoreSpecFixture(DefaultSeccompLocalhostProfile, "worker")
	podSpec.Containers = append(podSpec.Containers, corev1.Container{Name: "sidecar"})

	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, annotations, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected worker with sidecars to validate, got %v", err)
	}
}

func TestDiscoverStorageFromDaemonSetsUsesCheckpointsVolume(t *testing.T) {
	daemonSet := appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: "snapshot-agent", Namespace: "test-ns"},
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name: SnapshotAgentContainerName,
						VolumeMounts: []corev1.VolumeMount{
							{Name: "cache", MountPath: "/cache"},
							{Name: SnapshotAgentVolumeName, MountPath: "/checkpoints"},
						},
					}},
					Volumes: []corev1.Volume{
						{
							Name: "cache",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "cache-pvc"},
							},
						},
						{
							Name: SnapshotAgentVolumeName,
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "snapshot-pvc"},
							},
						},
					},
				},
			},
		},
	}

	storage, err := DiscoverStorageFromDaemonSets("test-ns", []appsv1.DaemonSet{daemonSet})
	if err != nil {
		t.Fatalf("expected daemonset storage discovery to succeed, got %v", err)
	}
	if storage.PVCName != "snapshot-pvc" || storage.BasePath != "/checkpoints" {
		t.Fatalf("expected snapshot PVC discovery, got %#v", storage)
	}
}

func findRestoreContainer(t *testing.T, containers []corev1.Container, name string) *corev1.Container {
	t.Helper()
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	t.Fatalf("container %q not found in spec", name)
	return nil
}
