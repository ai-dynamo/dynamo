package kube

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
)

func TestSnapshotStorageFromDaemonSetPrefersCheckpointMount(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "agent",
							VolumeMounts: []corev1.VolumeMount{
								{Name: "debug-driver", MountPath: "/host-lib"},
								{Name: "checkpoints", MountPath: "/checkpoints"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "debug-driver",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: "debug-driver-pvc",
								},
							},
						},
						{
							Name: "checkpoints",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: "snapshot-pvc",
								},
							},
						},
					},
				},
			},
		},
	}

	storage, ok := SnapshotStorageFromDaemonSet(daemonSet)
	if !ok {
		t.Fatalf("expected snapshot storage to be discovered")
	}
	if storage.PVCName != "snapshot-pvc" {
		t.Fatalf("expected snapshot-pvc, got %q", storage.PVCName)
	}
	if storage.BasePath != "/checkpoints" {
		t.Fatalf("expected /checkpoints, got %q", storage.BasePath)
	}
}

func TestSnapshotStorageFromDaemonSetFallsBackToOnlyPVCMount(t *testing.T) {
	daemonSet := &appsv1.DaemonSet{
		Spec: appsv1.DaemonSetSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "agent",
							VolumeMounts: []corev1.VolumeMount{
								{Name: "checkpoints", MountPath: "/custom-checkpoints"},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "checkpoints",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: "snapshot-pvc",
								},
							},
						},
					},
				},
			},
		},
	}

	storage, ok := SnapshotStorageFromDaemonSet(daemonSet)
	if !ok {
		t.Fatalf("expected snapshot storage to be discovered")
	}
	if storage.PVCName != "snapshot-pvc" {
		t.Fatalf("expected snapshot-pvc, got %q", storage.PVCName)
	}
	if storage.BasePath != "/custom-checkpoints" {
		t.Fatalf("expected /custom-checkpoints, got %q", storage.BasePath)
	}
}
