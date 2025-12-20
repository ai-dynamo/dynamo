package controller_common

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
)

func TestCanonicalizePodSpec(t *testing.T) {
	tests := []struct {
		name     string
		input    *corev1.PodSpec
		expected *corev1.PodSpec
	}{
		{
			name: "sorts containers by name",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "zebra"},
					{Name: "alpha"},
					{Name: "beta"},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "alpha"},
					{Name: "beta"},
					{Name: "zebra"},
				},
			},
		},
		{
			name: "sorts init containers by name",
			input: &corev1.PodSpec{
				InitContainers: []corev1.Container{
					{Name: "init-zebra"},
					{Name: "init-alpha"},
				},
			},
			expected: &corev1.PodSpec{
				InitContainers: []corev1.Container{
					{Name: "init-alpha"},
					{Name: "init-zebra"},
				},
			},
		},
		{
			name: "sorts ephemeral containers by name",
			input: &corev1.PodSpec{
				EphemeralContainers: []corev1.EphemeralContainer{
					{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug-zebra"}},
					{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug-alpha"}},
				},
			},
			expected: &corev1.PodSpec{
				EphemeralContainers: []corev1.EphemeralContainer{
					{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug-alpha"}},
					{EphemeralContainerCommon: corev1.EphemeralContainerCommon{Name: "debug-zebra"}},
				},
			},
		},
		{
			name: "sorts environment variables by name",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						Env: []corev1.EnvVar{
							{Name: "ZOO", Value: "zebra"},
							{Name: "ALPHA", Value: "apple"},
							{Name: "BETA", Value: "banana"},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						Env: []corev1.EnvVar{
							{Name: "ALPHA", Value: "apple"},
							{Name: "BETA", Value: "banana"},
							{Name: "ZOO", Value: "zebra"},
						},
					},
				},
			},
		},
		{
			name: "sorts envFrom by source type and name",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						EnvFrom: []corev1.EnvFromSource{
							{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-z"}}},
							{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"}}},
							{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-a"}}},
							{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-z"}}},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						EnvFrom: []corev1.EnvFromSource{
							{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"}}},
							{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-z"}}},
							{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-a"}}},
							{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-z"}}},
						},
					},
				},
			},
		},
		{
			name: "sorts container ports by name then port number",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						Ports: []corev1.ContainerPort{
							{Name: "http", ContainerPort: 8080},
							{Name: "grpc", ContainerPort: 9090},
							{Name: "grpc", ContainerPort: 8080},
							{Name: "debug", ContainerPort: 8080},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						Ports: []corev1.ContainerPort{
							{Name: "debug", ContainerPort: 8080},
							{Name: "grpc", ContainerPort: 8080},
							{Name: "grpc", ContainerPort: 9090},
							{Name: "http", ContainerPort: 8080},
						},
					},
				},
			},
		},
		{
			name: "sorts volume mounts by name then mount path",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						VolumeMounts: []corev1.VolumeMount{
							{Name: "vol1", MountPath: "/data2"},
							{Name: "vol2", MountPath: "/data1"},
							{Name: "vol1", MountPath: "/data1"},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						VolumeMounts: []corev1.VolumeMount{
							{Name: "vol1", MountPath: "/data1"},
							{Name: "vol1", MountPath: "/data2"},
							{Name: "vol2", MountPath: "/data1"},
						},
					},
				},
			},
		},
		{
			name: "sorts security context capabilities",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						SecurityContext: &corev1.SecurityContext{
							Capabilities: &corev1.Capabilities{
								Add:  []corev1.Capability{"SYS_ADMIN", "NET_ADMIN", "CHOWN"},
								Drop: []corev1.Capability{"ALL", "SETUID", "KILL"},
							},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						SecurityContext: &corev1.SecurityContext{
							Capabilities: &corev1.Capabilities{
								Add:  []corev1.Capability{"CHOWN", "NET_ADMIN", "SYS_ADMIN"},
								Drop: []corev1.Capability{"ALL", "KILL", "SETUID"},
							},
						},
					},
				},
			},
		},
		{
			name: "sorts image pull secrets by name",
			input: &corev1.PodSpec{
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "registry-z"},
					{Name: "registry-a"},
					{Name: "registry-b"},
					{Name: "registry-a"},
				},
			},
			expected: &corev1.PodSpec{
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "registry-a"},
					{Name: "registry-b"},
					{Name: "registry-z"},
				},
			},
		},
		{
			name: "sorts nil image pull secrets",
			input: &corev1.PodSpec{
				ImagePullSecrets: nil,
			},
			expected: &corev1.PodSpec{
				ImagePullSecrets: nil,
			},
		},
		{
			name: "sorts volumes by name",
			input: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{Name: "vol-z"},
					{Name: "vol-a"},
					{Name: "vol-b"},
				},
			},
			expected: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{Name: "vol-a"},
					{Name: "vol-b"},
					{Name: "vol-z"},
				},
			},
		},
		{
			name: "sorts configmap volume items by key then path",
			input: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "config",
						VolumeSource: corev1.VolumeSource{
							ConfigMap: &corev1.ConfigMapVolumeSource{
								Items: []corev1.KeyToPath{
									{Key: "app.conf", Path: "config/app.conf"},
									{Key: "db.conf", Path: "config/db.conf"},
									{Key: "app.conf", Path: "backup/app.conf"},
								},
							},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "config",
						VolumeSource: corev1.VolumeSource{
							ConfigMap: &corev1.ConfigMapVolumeSource{
								Items: []corev1.KeyToPath{
									{Key: "app.conf", Path: "backup/app.conf"},
									{Key: "app.conf", Path: "config/app.conf"},
									{Key: "db.conf", Path: "config/db.conf"},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "sorts secret volume items by key then path",
			input: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "secret",
						VolumeSource: corev1.VolumeSource{
							Secret: &corev1.SecretVolumeSource{
								Items: []corev1.KeyToPath{
									{Key: "tls.key", Path: "tls/server.key"},
									{Key: "tls.crt", Path: "tls/server.crt"},
									{Key: "tls.key", Path: "backup/server.key"},
								},
							},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "secret",
						VolumeSource: corev1.VolumeSource{
							Secret: &corev1.SecretVolumeSource{
								Items: []corev1.KeyToPath{
									{Key: "tls.crt", Path: "tls/server.crt"},
									{Key: "tls.key", Path: "backup/server.key"},
									{Key: "tls.key", Path: "tls/server.key"},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "sorts downward API items by path",
			input: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "downward",
						VolumeSource: corev1.VolumeSource{
							DownwardAPI: &corev1.DownwardAPIVolumeSource{
								Items: []corev1.DownwardAPIVolumeFile{
									{Path: "metadata/name"},
									{Path: "metadata/annotations"},
									{Path: "limits/cpu"},
								},
							},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "downward",
						VolumeSource: corev1.VolumeSource{
							DownwardAPI: &corev1.DownwardAPIVolumeSource{
								Items: []corev1.DownwardAPIVolumeFile{
									{Path: "limits/cpu"},
									{Path: "metadata/annotations"},
									{Path: "metadata/name"},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "sorts projected volume sources and their items",
			input: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "projected",
						VolumeSource: corev1.VolumeSource{
							Projected: &corev1.ProjectedVolumeSource{
								Sources: []corev1.VolumeProjection{
									{
										Secret: &corev1.SecretProjection{
											LocalObjectReference: corev1.LocalObjectReference{Name: "secret-z"},
											Items: []corev1.KeyToPath{
												{Key: "password", Path: "auth/password"},
												{Key: "username", Path: "auth/username"},
											},
										},
									},
									{
										ConfigMap: &corev1.ConfigMapProjection{
											LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"},
											Items: []corev1.KeyToPath{
												{Key: "db.conf", Path: "config/db.conf"},
												{Key: "app.conf", Path: "config/app.conf"},
											},
										},
									},
									{
										DownwardAPI: &corev1.DownwardAPIProjection{
											Items: []corev1.DownwardAPIVolumeFile{
												{Path: "metadata/name"},
												{Path: "limits/cpu"},
											},
										},
									},
									{
										ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
											Audience: "api.example.com",
											Path:     "tokens/api",
										},
									},
								},
							},
						},
					},
				},
			},
			expected: &corev1.PodSpec{
				Volumes: []corev1.Volume{
					{
						Name: "projected",
						VolumeSource: corev1.VolumeSource{
							Projected: &corev1.ProjectedVolumeSource{
								Sources: []corev1.VolumeProjection{
									{
										ConfigMap: &corev1.ConfigMapProjection{
											LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"},
											Items: []corev1.KeyToPath{
												{Key: "app.conf", Path: "config/app.conf"},
												{Key: "db.conf", Path: "config/db.conf"},
											},
										},
									},
									{
										DownwardAPI: &corev1.DownwardAPIProjection{
											Items: []corev1.DownwardAPIVolumeFile{
												{Path: "limits/cpu"},
												{Path: "metadata/name"},
											},
										},
									},
									{
										ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
											Audience: "api.example.com",
											Path:     "tokens/api",
										},
									},
									{
										Secret: &corev1.SecretProjection{
											LocalObjectReference: corev1.LocalObjectReference{Name: "secret-z"},
											Items: []corev1.KeyToPath{
												{Key: "password", Path: "auth/password"},
												{Key: "username", Path: "auth/username"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "sorts tolerations by key, operator, value, effect, seconds",
			input: &corev1.PodSpec{
				Tolerations: []corev1.Toleration{
					{
						Key:      "node-type",
						Operator: corev1.TolerationOpEqual,
						Value:    "gpu",
						Effect:   corev1.TaintEffectNoSchedule,
					},
					{
						Key:      "node-role",
						Operator: corev1.TolerationOpEqual,
						Value:    "master",
						Effect:   corev1.TaintEffectNoSchedule,
					},
					{
						Key:      "node-role",
						Operator: corev1.TolerationOpExists,
						Effect:   corev1.TaintEffectNoSchedule,
					},
				},
			},
			expected: &corev1.PodSpec{
				Tolerations: []corev1.Toleration{
					{
						Key:      "node-role",
						Operator: corev1.TolerationOpEqual,
						Value:    "master",
						Effect:   corev1.TaintEffectNoSchedule,
					},
					{
						Key:      "node-role",
						Operator: corev1.TolerationOpExists,
						Effect:   corev1.TaintEffectNoSchedule,
					},
					{
						Key:      "node-type",
						Operator: corev1.TolerationOpEqual,
						Value:    "gpu",
						Effect:   corev1.TaintEffectNoSchedule,
					},
				},
			},
		},
		{
			name: "sorts topology spread constraints by topology key, when unsatisfiable, max skew",
			input: &corev1.PodSpec{
				TopologySpreadConstraints: []corev1.TopologySpreadConstraint{
					{
						TopologyKey:       "kubernetes.io/zone",
						WhenUnsatisfiable: corev1.DoNotSchedule,
						MaxSkew:           2,
					},
					{
						TopologyKey:       "kubernetes.io/hostname",
						WhenUnsatisfiable: corev1.DoNotSchedule,
						MaxSkew:           1,
					},
					{
						TopologyKey:       "kubernetes.io/hostname",
						WhenUnsatisfiable: corev1.ScheduleAnyway,
						MaxSkew:           1,
					},
				},
			},
			expected: &corev1.PodSpec{
				TopologySpreadConstraints: []corev1.TopologySpreadConstraint{
					{
						TopologyKey:       "kubernetes.io/hostname",
						WhenUnsatisfiable: corev1.DoNotSchedule,
						MaxSkew:           1,
					},
					{
						TopologyKey:       "kubernetes.io/hostname",
						WhenUnsatisfiable: corev1.ScheduleAnyway,
						MaxSkew:           1,
					},
					{
						TopologyKey:       "kubernetes.io/zone",
						WhenUnsatisfiable: corev1.DoNotSchedule,
						MaxSkew:           2,
					},
				},
			},
		},
		{
			name: "sorts host aliases by IP and hostnames within each alias",
			input: &corev1.PodSpec{
				HostAliases: []corev1.HostAlias{
					{
						IP:        "192.168.1.2",
						Hostnames: []string{"web2.example.com", "api2.example.com"},
					},
					{
						IP:        "192.168.1.1",
						Hostnames: []string{"web1.example.com", "api1.example.com", "admin1.example.com"},
					},
				},
			},
			expected: &corev1.PodSpec{
				HostAliases: []corev1.HostAlias{
					{
						IP:        "192.168.1.1",
						Hostnames: []string{"admin1.example.com", "api1.example.com", "web1.example.com"},
					},
					{
						IP:        "192.168.1.2",
						Hostnames: []string{"api2.example.com", "web2.example.com"},
					},
				},
			},
		},
		{
			name: "sorts DNS config options, nameservers, and searches",
			input: &corev1.PodSpec{
				DNSConfig: &corev1.PodDNSConfig{
					Options: []corev1.PodDNSConfigOption{
						{Name: "timeout", Value: func() *string { s := "5"; return &s }()},
						{Name: "attempts", Value: func() *string { s := "3"; return &s }()},
						{Name: "ndots", Value: func() *string { s := "2"; return &s }()},
					},
					Nameservers: []string{"8.8.8.8", "1.1.1.1", "8.8.4.4"},
					Searches:    []string{"example.com", "cluster.local", "app.local"},
				},
			},
			expected: &corev1.PodSpec{
				DNSConfig: &corev1.PodDNSConfig{
					Options: []corev1.PodDNSConfigOption{
						{Name: "attempts", Value: func() *string { s := "3"; return &s }()},
						{Name: "ndots", Value: func() *string { s := "2"; return &s }()},
						{Name: "timeout", Value: func() *string { s := "5"; return &s }()},
					},
					Nameservers: []string{"1.1.1.1", "8.8.4.4", "8.8.8.8"},
					Searches:    []string{"app.local", "cluster.local", "example.com"},
				},
			},
		},
		{
			name: "handles nil pointer values gracefully",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						SecurityContext: &corev1.SecurityContext{
							Capabilities: nil,
						},
					},
				},
				DNSConfig: &corev1.PodDNSConfig{
					Options: []corev1.PodDNSConfigOption{
						{Name: "timeout", Value: nil},
						{Name: "attempts", Value: func() *string { s := "3"; return &s }()},
					},
				},
				Tolerations: []corev1.Toleration{
					{
						Key:               "test",
						TolerationSeconds: nil,
					},
					{
						Key:               "test2",
						TolerationSeconds: func() *int64 { s := int64(300); return &s }(),
					},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "test",
						SecurityContext: &corev1.SecurityContext{
							Capabilities: nil,
						},
					},
				},
				DNSConfig: &corev1.PodDNSConfig{
					Options: []corev1.PodDNSConfigOption{
						{Name: "attempts", Value: func() *string { s := "3"; return &s }()},
						{Name: "timeout", Value: nil},
					},
				},
				Tolerations: []corev1.Toleration{
					{
						Key:               "test",
						TolerationSeconds: nil,
					},
					{
						Key:               "test2",
						TolerationSeconds: func() *int64 { s := int64(300); return &s }(),
					},
				},
			},
		},
		{
			name: "returns original podspec when already sorted",
			input: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "alpha",
						Env: []corev1.EnvVar{
							{Name: "A", Value: "1"},
							{Name: "B", Value: "2"},
						},
					},
					{Name: "beta"},
				},
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "secret-a"},
					{Name: "secret-b"},
				},
			},
			expected: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name: "alpha",
						Env: []corev1.EnvVar{
							{Name: "A", Value: "1"},
							{Name: "B", Value: "2"},
						},
					},
					{Name: "beta"},
				},
				ImagePullSecrets: []corev1.LocalObjectReference{
					{Name: "secret-a"},
					{Name: "secret-b"},
				},
			},
		},
		{
			name: "handles empty slices gracefully",
			input: &corev1.PodSpec{
				Containers:       []corev1.Container{},
				InitContainers:   []corev1.Container{},
				ImagePullSecrets: []corev1.LocalObjectReference{},
				Volumes:          []corev1.Volume{},
				Tolerations:      []corev1.Toleration{},
			},
			expected: &corev1.PodSpec{
				Containers:       []corev1.Container{},
				InitContainers:   []corev1.Container{},
				ImagePullSecrets: []corev1.LocalObjectReference{},
				Volumes:          []corev1.Volume{},
				Tolerations:      []corev1.Toleration{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CanonicalizePodSpec(tt.input)

			// Verify the function returns the same instance
			assert.Same(t, tt.input, result, "function should return the same PodSpec instance")

			// Verify the sorting is correct
			assert.Equal(t, tt.expected, result, "PodSpec should be sorted correctly")
		})
	}
}

func TestCanonicalizePodSpec_Idempotent(t *testing.T) {
	// Create a complex, unsorted PodSpec
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name: "zebra",
				Env: []corev1.EnvVar{
					{Name: "Z_VAR", Value: "z"},
					{Name: "A_VAR", Value: "a"},
				},
				Ports: []corev1.ContainerPort{
					{Name: "http", ContainerPort: 8080},
					{Name: "grpc", ContainerPort: 9090},
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "vol2", MountPath: "/data2"},
					{Name: "vol1", MountPath: "/data1"},
				},
			},
			{Name: "alpha"},
		},
		InitContainers: []corev1.Container{
			{Name: "init-zebra"},
			{Name: "init-alpha"},
		},
		ImagePullSecrets: []corev1.LocalObjectReference{
			{Name: "secret-z"},
			{Name: "secret-a"},
		},
		Volumes: []corev1.Volume{
			{Name: "vol-z"},
			{Name: "vol-a"},
		},
		Tolerations: []corev1.Toleration{
			{Key: "node-z"},
			{Key: "node-a"},
		},
	}

	// First canonicalization
	result1 := CanonicalizePodSpec(podSpec)

	// Second canonicalization on the same object
	result2 := CanonicalizePodSpec(result1)

	// Should be identical after second canonicalization
	assert.Equal(t, result1, result2, "CanonicalizePodSpec should be idempotent")

	// Verify containers are sorted
	assert.Equal(t, "alpha", result2.Containers[0].Name)
	assert.Equal(t, "zebra", result2.Containers[1].Name)

	// Verify env vars within containers are sorted
	assert.Equal(t, "A_VAR", result2.Containers[1].Env[0].Name)
	assert.Equal(t, "Z_VAR", result2.Containers[1].Env[1].Name)

	// Verify ports are sorted
	assert.Equal(t, "grpc", result2.Containers[1].Ports[0].Name)
	assert.Equal(t, "http", result2.Containers[1].Ports[1].Name)

	// Verify volume mounts are sorted
	assert.Equal(t, "vol1", result2.Containers[1].VolumeMounts[0].Name)
	assert.Equal(t, "vol2", result2.Containers[1].VolumeMounts[1].Name)
}

func TestCanonicalizePodSpec_EnvFromSortPriority(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name: "test",
				EnvFrom: []corev1.EnvFromSource{
					{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-b"}}},
					{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-b"}}},
					{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-a"}}},
					{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"}}},
					// Test duplicate names for secondary sort
					{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"}}},
					{SecretRef: &corev1.SecretEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-a"}}},
				},
			},
		},
	}

	result := CanonicalizePodSpec(podSpec)

	// ConfigMaps should come before Secrets (cm: < sec:)
	// Within each type, sorted by name
	expected := []string{
		"cm:config-a:",  // ConfigMap config-a
		"cm:config-a:",  // ConfigMap config-a (duplicate)
		"cm:config-b:",  // ConfigMap config-b
		"sec:secret-a:", // Secret secret-a
		"sec:secret-a:", // Secret secret-a (duplicate)
		"sec:secret-b:", // Secret secret-b
	}

	envFromKey := func(e corev1.EnvFromSource) string {
		if e.ConfigMapRef != nil {
			return "cm:" + e.ConfigMapRef.Name + ":"
		}
		if e.SecretRef != nil {
			return "sec:" + e.SecretRef.Name + ":"
		}
		return "other:"
	}

	for i, envFrom := range result.Containers[0].EnvFrom {
		assert.Equal(t, expected[i], envFromKey(envFrom), "EnvFrom at index %d should match expected sort order", i)
	}
}

func TestCanonicalizePodSpec_TolerationSecondsHandling(t *testing.T) {
	sec300 := int64(300)
	sec600 := int64(600)

	podSpec := &corev1.PodSpec{
		Tolerations: []corev1.Toleration{
			{Key: "key1", TolerationSeconds: &sec600},
			{Key: "key1", TolerationSeconds: nil},
			{Key: "key1", TolerationSeconds: &sec300},
		},
	}

	result := CanonicalizePodSpec(podSpec)

	// Should be sorted by TolerationSeconds: nil (0) < 300 < 600
	assert.Nil(t, result.Tolerations[0].TolerationSeconds)
	assert.Equal(t, int64(300), *result.Tolerations[1].TolerationSeconds)
	assert.Equal(t, int64(600), *result.Tolerations[2].TolerationSeconds)
}

func TestCanonicalizePodSpec_ProjectedVolumeSourcePriority(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Volumes: []corev1.Volume{
			{
				Name: "projected",
				VolumeSource: corev1.VolumeSource{
					Projected: &corev1.ProjectedVolumeSource{
						Sources: []corev1.VolumeProjection{
							{Secret: &corev1.SecretProjection{LocalObjectReference: corev1.LocalObjectReference{Name: "secret-a"}}},
							{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Audience: "zz.example.com"}},
							{DownwardAPI: &corev1.DownwardAPIProjection{}},
							{ConfigMap: &corev1.ConfigMapProjection{LocalObjectReference: corev1.LocalObjectReference{Name: "config-z"}}},
							{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{Audience: "aa.example.com"}},
							{ConfigMap: &corev1.ConfigMapProjection{LocalObjectReference: corev1.LocalObjectReference{Name: "config-a"}}},
						},
					},
				},
			},
		},
	}

	result := CanonicalizePodSpec(podSpec)

	// Expected sort order: cm: < downward: < sat: < sec:
	// Within same type, sorted by name/audience
	getProjectionKey := func(p corev1.VolumeProjection) string {
		if p.ConfigMap != nil {
			return "cm:" + p.ConfigMap.Name
		}
		if p.Secret != nil {
			return "sec:" + p.Secret.Name
		}
		if p.DownwardAPI != nil {
			return "downward:"
		}
		if p.ServiceAccountToken != nil {
			return "sat:" + p.ServiceAccountToken.Audience
		}
		return "z:other"
	}

	expected := []string{
		"cm:config-a",
		"cm:config-z",
		"downward:",
		"sat:aa.example.com",
		"sat:zz.example.com",
		"sec:secret-a",
	}

	sources := result.Volumes[0].VolumeSource.Projected.Sources
	for i, source := range sources {
		assert.Equal(t, expected[i], getProjectionKey(source), "Projected source at index %d should match expected sort order", i)
	}
}
