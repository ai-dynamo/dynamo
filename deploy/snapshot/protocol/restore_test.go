package protocol

import (
	"math"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewRestorePod(t *testing.T) {
	readinessProbe := &corev1.Probe{PeriodSeconds: 7, TimeoutSeconds: 3}
	livenessProbe := &corev1.Probe{InitialDelaySeconds: 11}
	startupProbe := &corev1.Probe{FailureThreshold: 120}
	restorePod := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "worker",
			Labels:      map[string]string{"existing": "label"},
			Annotations: map[string]string{"existing": "annotation"},
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

	if restorePod.Name != "worker" || restorePod.Namespace != "test-ns" {
		t.Fatalf("unexpected restore pod identity: %#v", restorePod.ObjectMeta)
	}
	if restorePod.Labels[RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label: %#v", restorePod.Labels)
	}
	if restorePod.Labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint id label: %#v", restorePod.Labels)
	}
	if restorePod.Annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation: %#v", restorePod.Annotations)
	}
	if restorePod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %#v", restorePod.Spec.RestartPolicy)
	}
	if len(restorePod.Spec.Containers[0].Command) != 2 || restorePod.Spec.Containers[0].Command[0] != "sleep" || restorePod.Spec.Containers[0].Command[1] != "infinity" {
		t.Fatalf("expected placeholder command, got %#v", restorePod.Spec.Containers[0].Command)
	}
	if restorePod.Spec.Containers[0].Args != nil {
		t.Fatalf("expected restore args to be cleared: %#v", restorePod.Spec.Containers[0].Args)
	}
	if restorePod.Spec.Containers[0].ReadinessProbe == nil {
		t.Fatalf("expected readiness probe to be preserved")
	}
	if got := restorePod.Spec.Containers[0].ReadinessProbe.PeriodSeconds; got != readinessProbe.PeriodSeconds {
		t.Fatalf("expected readiness probe period %d, got %d", readinessProbe.PeriodSeconds, got)
	}
	if restorePod.Spec.Containers[0].LivenessProbe == nil {
		t.Fatalf("expected liveness probe to be preserved")
	}
	if got := restorePod.Spec.Containers[0].LivenessProbe.InitialDelaySeconds; got != livenessProbe.InitialDelaySeconds {
		t.Fatalf("expected liveness initial delay %d, got %d", livenessProbe.InitialDelaySeconds, got)
	}
	if restorePod.Spec.Containers[0].StartupProbe == nil {
		t.Fatalf("expected startup probe to be preserved")
	}
	if got := restorePod.Spec.Containers[0].StartupProbe.FailureThreshold; got != math.MaxInt32 {
		t.Fatalf("expected startup failure threshold %d, got %d", math.MaxInt32, got)
	}
	if restorePod.Spec.SecurityContext == nil || restorePod.Spec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", restorePod.Spec.SecurityContext)
	}
	if len(restorePod.Spec.Volumes) != 1 {
		t.Fatalf("expected checkpoint volume, got %#v", restorePod.Spec.Volumes)
	}
	if len(restorePod.Spec.Containers[0].VolumeMounts) != 1 {
		t.Fatalf("expected checkpoint mount, got %#v", restorePod.Spec.Containers[0].VolumeMounts)
	}
}

func TestPrepareRestorePodSpec(t *testing.T) {
	podSpec := corev1.PodSpec{}
	readinessProbe := &corev1.Probe{PeriodSeconds: 13, SuccessThreshold: 1}
	livenessProbe := &corev1.Probe{TimeoutSeconds: 5}
	startupProbe := &corev1.Probe{FailureThreshold: 60}
	container := corev1.Container{
		Command:        []string{"python3", "-m", "dynamo.vllm"},
		Args:           []string{"--model", "Qwen"},
		ReadinessProbe: readinessProbe.DeepCopy(),
		LivenessProbe:  livenessProbe.DeepCopy(),
		StartupProbe:   startupProbe.DeepCopy(),
	}

	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}
	PrepareRestorePodSpec(&podSpec, &container, storage, DefaultSeccompLocalhostProfile, true)
	PrepareRestorePodSpec(&podSpec, &container, storage, DefaultSeccompLocalhostProfile, true)

	if podSpec.SecurityContext == nil || podSpec.SecurityContext.SeccompProfile == nil {
		t.Fatalf("expected seccomp profile to be injected: %#v", podSpec.SecurityContext)
	}
	if len(podSpec.Volumes) != 1 {
		t.Fatalf("expected checkpoint volume, got %#v", podSpec.Volumes)
	}
	if len(container.VolumeMounts) != 1 {
		t.Fatalf("expected checkpoint mount, got %#v", container.VolumeMounts)
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
		t.Fatalf("expected startup probe to be preserved")
	}
	if got := container.StartupProbe.FailureThreshold; got != math.MaxInt32 {
		t.Fatalf("expected startup failure threshold %d, got %d", math.MaxInt32, got)
	}
	if got := container.StartupProbe.SuccessThreshold; got != 1 {
		t.Fatalf("expected startup success threshold 1, got %d", got)
	}
}

func TestPrepareRestorePodSpecSynthesizesStartupProbeFromLiveness(t *testing.T) {
	podSpec := corev1.PodSpec{}
	livenessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{Path: "/livez"},
		},
		PeriodSeconds:    5,
		TimeoutSeconds:   4,
		FailureThreshold: 2,
	}
	container := corev1.Container{
		Command:       []string{"python3", "-m", "dynamo.vllm"},
		Args:          []string{"--model", "Qwen"},
		LivenessProbe: livenessProbe.DeepCopy(),
	}

	PrepareRestorePodSpec(&podSpec, &container, Storage{}, "", true)

	if container.LivenessProbe == nil {
		t.Fatalf("expected liveness probe to be preserved")
	}
	if container.StartupProbe == nil {
		t.Fatalf("expected startup probe to be synthesized")
	}
	if container.StartupProbe.HTTPGet == nil || container.StartupProbe.HTTPGet.Path != "/livez" {
		t.Fatalf("expected startup probe HTTP path /livez, got %#v", container.StartupProbe.HTTPGet)
	}
	if got := container.StartupProbe.FailureThreshold; got != math.MaxInt32 {
		t.Fatalf("expected startup failure threshold %d, got %d", math.MaxInt32, got)
	}
	if got := container.StartupProbe.SuccessThreshold; got != 1 {
		t.Fatalf("expected startup success threshold 1, got %d", got)
	}
}

func TestNewRestorePodTargetsMainContainerWhenSidecarsPresent(t *testing.T) {
	restorePod := NewRestorePod(&corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "worker"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "sidecar", Image: "sidecar:latest", Command: []string{"sidecar"}, Args: []string{"run"}},
				{Name: "main", Image: "test:latest", Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
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

	if got := restorePod.Spec.Containers[0].Command; len(got) != 1 || got[0] != "sidecar" {
		t.Fatalf("expected sidecar command to remain unchanged, got %#v", got)
	}
	if got := restorePod.Spec.Containers[1].Command; len(got) != 2 || got[0] != "sleep" || got[1] != "infinity" {
		t.Fatalf("expected main container placeholder command, got %#v", got)
	}
	if restorePod.Spec.Containers[1].Args != nil {
		t.Fatalf("expected main container args to be cleared: %#v", restorePod.Spec.Containers[1].Args)
	}
}

func TestPrepareRestorePodSpecSynthesizesStartupProbeFromReadiness(t *testing.T) {
	podSpec := corev1.PodSpec{}
	readinessProbe := &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: []string{"cat", "/tmp/ready"}},
		},
		PeriodSeconds:    13,
		SuccessThreshold: 3,
		FailureThreshold: 4,
	}
	container := corev1.Container{
		Command:        []string{"python3", "-m", "dynamo.vllm"},
		Args:           []string{"--model", "Qwen"},
		ReadinessProbe: readinessProbe.DeepCopy(),
	}

	PrepareRestorePodSpec(&podSpec, &container, Storage{}, "", true)

	if container.ReadinessProbe == nil {
		t.Fatalf("expected readiness probe to be preserved")
	}
	if got := container.ReadinessProbe.SuccessThreshold; got != readinessProbe.SuccessThreshold {
		t.Fatalf("expected readiness success threshold %d, got %d", readinessProbe.SuccessThreshold, got)
	}
	if container.StartupProbe == nil {
		t.Fatalf("expected startup probe to be synthesized")
	}
	if container.StartupProbe.Exec == nil || len(container.StartupProbe.Exec.Command) != 2 || container.StartupProbe.Exec.Command[0] != "cat" || container.StartupProbe.Exec.Command[1] != "/tmp/ready" {
		t.Fatalf("expected startup probe exec command to match readiness probe: %#v", container.StartupProbe.Exec)
	}
	if got := container.StartupProbe.FailureThreshold; got != math.MaxInt32 {
		t.Fatalf("expected startup failure threshold %d, got %d", math.MaxInt32, got)
	}
	if got := container.StartupProbe.SuccessThreshold; got != 1 {
		t.Fatalf("expected startup success threshold 1, got %d", got)
	}
}

func TestValidateRestorePodSpec(t *testing.T) {
	profile := DefaultSeccompLocalhostProfile
	podSpec := &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type:             corev1.SeccompProfileTypeLocalhost,
				LocalhostProfile: &profile,
			},
		},
		Volumes: []corev1.Volume{{
			Name: CheckpointVolumeName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: "snapshot-pvc",
				},
			},
		}},
		Containers: []corev1.Container{{
			Name: "main",
			VolumeMounts: []corev1.VolumeMount{{
				Name:      CheckpointVolumeName,
				MountPath: "/checkpoints",
			}},
		}},
	}
	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected restore pod spec to be valid, got %v", err)
	}

	badSpec := podSpec.DeepCopy()
	badSpec.Volumes = nil
	if err := ValidateRestorePodSpec(badSpec, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing checkpoint-storage volume for PVC snapshot-pvc" {
		t.Fatalf("expected missing volume error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.Containers[0].VolumeMounts = nil
	if err := ValidateRestorePodSpec(badSpec, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing checkpoint-storage mount at /checkpoints" {
		t.Fatalf("expected missing mount error, got %v", err)
	}

	badSpec = podSpec.DeepCopy()
	badSpec.SecurityContext = nil
	if err := ValidateRestorePodSpec(badSpec, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "missing localhost seccomp profile" {
		t.Fatalf("expected missing seccomp error, got %v", err)
	}
}

func TestValidateRestorePodSpecRequiresMainContainerWhenMultiContainer(t *testing.T) {
	profile := DefaultSeccompLocalhostProfile
	podSpec := &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type:             corev1.SeccompProfileTypeLocalhost,
				LocalhostProfile: &profile,
			},
		},
		Volumes: []corev1.Volume{{
			Name: CheckpointVolumeName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: "snapshot-pvc",
				},
			},
		}},
		Containers: []corev1.Container{
			{
				Name: "worker",
				VolumeMounts: []corev1.VolumeMount{{
					Name:      CheckpointVolumeName,
					MountPath: "/checkpoints",
				}},
			},
			{Name: "sidecar"},
		},
	}

	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, storage, DefaultSeccompLocalhostProfile); err == nil || err.Error() != "restore target must include a worker container named main" {
		t.Fatalf("expected multi-container restore target without main to be rejected, got %v", err)
	}
}

func TestValidateRestorePodSpecAllowsMainContainerWithSidecars(t *testing.T) {
	profile := DefaultSeccompLocalhostProfile
	podSpec := &corev1.PodSpec{
		SecurityContext: &corev1.PodSecurityContext{
			SeccompProfile: &corev1.SeccompProfile{
				Type:             corev1.SeccompProfileTypeLocalhost,
				LocalhostProfile: &profile,
			},
		},
		Volumes: []corev1.Volume{{
			Name: CheckpointVolumeName,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: "snapshot-pvc",
				},
			},
		}},
		Containers: []corev1.Container{
			{Name: "sidecar"},
			{
				Name: "main",
				VolumeMounts: []corev1.VolumeMount{{
					Name:      CheckpointVolumeName,
					MountPath: "/checkpoints",
				}},
			},
		},
	}

	storage := Storage{
		Type:     StorageTypePVC,
		PVCName:  "snapshot-pvc",
		BasePath: "/checkpoints",
	}

	if err := ValidateRestorePodSpec(podSpec, storage, DefaultSeccompLocalhostProfile); err != nil {
		t.Fatalf("expected main container with sidecars to validate, got %v", err)
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
