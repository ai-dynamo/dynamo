package k8s

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

// TestGetVolumeDetails tests the getVolumeDetails function with common volume types
func TestGetVolumeDetails(t *testing.T) {
	tests := []struct {
		name            string
		volume          *corev1.Volume
		expectedType    string
		expectedCMName  string
		expectedSecret  string
		expectedPVCName string
	}{
		{
			name:         "emptyDir",
			volume:       &corev1.Volume{VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
			expectedType: "emptyDir",
		},
		{
			name:           "configMap",
			volume:         &corev1.Volume{VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "my-config"}}}},
			expectedType:   "configMap",
			expectedCMName: "my-config",
		},
		{
			name:           "secret",
			volume:         &corev1.Volume{VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "my-secret"}}},
			expectedType:   "secret",
			expectedSecret: "my-secret",
		},
		{
			name:            "persistentVolumeClaim",
			volume:          &corev1.Volume{VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "my-pvc"}}},
			expectedType:    "persistentVolumeClaim",
			expectedPVCName: "my-pvc",
		},
		{
			name:         "hostPath",
			volume:       &corev1.Volume{VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/var/lib"}}},
			expectedType: "hostPath",
		},
		{
			name:         "projected",
			volume:       &corev1.Volume{VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{}}},
			expectedType: "projected",
		},
		{
			name:         "downwardAPI",
			volume:       &corev1.Volume{VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{}}},
			expectedType: "downwardAPI",
		},
		{
			name:         "csi",
			volume:       &corev1.Volume{VolumeSource: corev1.VolumeSource{CSI: &corev1.CSIVolumeSource{Driver: "csi.example.com"}}},
			expectedType: "csi",
		},
		{
			name:         "unknown - no volume source",
			volume:       &corev1.Volume{},
			expectedType: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getVolumeDetails(tt.volume)

			if result.volumeType != tt.expectedType {
				t.Errorf("volumeType = %q, want %q", result.volumeType, tt.expectedType)
			}
			if tt.expectedCMName != "" && result.configMapName != tt.expectedCMName {
				t.Errorf("configMapName = %q, want %q", result.configMapName, tt.expectedCMName)
			}
			if tt.expectedSecret != "" && result.secretName != tt.expectedSecret {
				t.Errorf("secretName = %q, want %q", result.secretName, tt.expectedSecret)
			}
			if tt.expectedPVCName != "" && result.pvcName != tt.expectedPVCName {
				t.Errorf("pvcName = %q, want %q", result.pvcName, tt.expectedPVCName)
			}
		})
	}
}

// TestDetectVolumeTypeFromPath tests path pattern matching for K8s volume types
func TestDetectVolumeTypeFromPath(t *testing.T) {
	tests := []struct {
		name           string
		hostPath       string
		expectedType   string
		expectedVolume string
	}{
		{
			name:           "emptyDir volume",
			hostPath:       "/var/lib/kubelet/pods/abc123/volumes/kubernetes.io~empty-dir/cache-volume",
			expectedType:   "emptyDir",
			expectedVolume: "cache-volume",
		},
		{
			name:           "configMap volume",
			hostPath:       "/var/lib/kubelet/pods/abc123/volumes/kubernetes.io~configmap/config-vol",
			expectedType:   "configMap",
			expectedVolume: "config-vol",
		},
		{
			name:           "secret volume",
			hostPath:       "/var/lib/kubelet/pods/abc123/volumes/kubernetes.io~secret/secret-vol",
			expectedType:   "secret",
			expectedVolume: "secret-vol",
		},
		{
			name:           "projected volume",
			hostPath:       "/var/lib/kubelet/pods/abc123/volumes/kubernetes.io~projected/kube-api-access",
			expectedType:   "projected",
			expectedVolume: "kube-api-access",
		},
		{
			name:           "persistentVolumeClaim volume",
			hostPath:       "/var/lib/kubelet/pods/abc123/volumes/kubernetes.io~persistentvolumeclaim/data-pvc",
			expectedType:   "persistentVolumeClaim",
			expectedVolume: "data-pvc",
		},
		{
			name:           "unknown volume type",
			hostPath:       "/some/other/path/without/kubernetes/pattern",
			expectedType:   "unknown",
			expectedVolume: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			volType, volName := DetectVolumeTypeFromPath(tt.hostPath)

			if volType != tt.expectedType {
				t.Errorf("volumeType = %q, want %q", volType, tt.expectedType)
			}
			if volName != tt.expectedVolume {
				t.Errorf("volumeName = %q, want %q", volName, tt.expectedVolume)
			}
		})
	}
}

// TestExtractVolumeInfo tests the ExtractVolumeInfo function with various pod configurations
func TestExtractVolumeInfo(t *testing.T) {
	tests := []struct {
		name          string
		pod           *corev1.Pod
		containerName string
		wantErr       bool
		wantCount     int
	}{
		{
			name: "single emptyDir mount",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes:    []corev1.Volume{{Name: "cache", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}},
					Containers: []corev1.Container{{Name: "app", VolumeMounts: []corev1.VolumeMount{{Name: "cache", MountPath: "/cache"}}}},
				},
			},
			containerName: "app",
			wantCount:     1,
		},
		{
			name: "multiple volume types",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{Name: "cache", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
						{Name: "config", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "app-config"}}}},
						{Name: "data", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "data-pvc"}}},
					},
					Containers: []corev1.Container{
						{
							Name: "app",
							VolumeMounts: []corev1.VolumeMount{
								{Name: "cache", MountPath: "/cache"},
								{Name: "config", MountPath: "/etc/config", ReadOnly: true},
								{Name: "data", MountPath: "/data"},
							},
						},
					},
				},
			},
			containerName: "app",
			wantCount:     3,
		},
		{
			name: "container not found",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "app"}},
				},
			},
			containerName: "nonexistent",
			wantErr:       true,
		},
		{
			name: "no volume mounts",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Volumes:    []corev1.Volume{{Name: "unused", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}},
					Containers: []corev1.Container{{Name: "app"}},
				},
			},
			containerName: "app",
			wantCount:     0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractVolumeInfo(tt.pod, tt.containerName)

			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if len(got) != tt.wantCount {
				t.Errorf("got %d volumes, want %d", len(got), tt.wantCount)
			}
		})
	}
}
