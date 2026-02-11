package main

import (
	"os"
	"testing"
)

// TestGetEnv tests environment variable retrieval with defaults
func TestGetEnv(t *testing.T) {
	tests := []struct {
		name         string
		key          string
		envValue     string
		setEnv       bool
		defaultValue string
		want         string
	}{
		{
			name:         "env var set",
			key:          "TEST_VAR_SET",
			envValue:     "custom_value",
			setEnv:       true,
			defaultValue: "default",
			want:         "custom_value",
		},
		{
			name:         "env var not set - use default",
			key:          "TEST_VAR_UNSET",
			setEnv:       false,
			defaultValue: "default_value",
			want:         "default_value",
		},
		{
			name:         "env var empty string - use default",
			key:          "TEST_VAR_EMPTY",
			envValue:     "",
			setEnv:       true,
			defaultValue: "default",
			want:         "default",
		},
		{
			name:         "empty default value",
			key:          "TEST_VAR_UNSET2",
			setEnv:       false,
			defaultValue: "",
			want:         "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clean up before test
			os.Unsetenv(tt.key)

			if tt.setEnv {
				os.Setenv(tt.key, tt.envValue)
				defer os.Unsetenv(tt.key)
			}

			got := getEnv(tt.key, tt.defaultValue)
			if got != tt.want {
				t.Errorf("getEnv(%q, %q) = %q, want %q", tt.key, tt.defaultValue, got, tt.want)
			}
		})
	}
}

// TestImagesCompatible tests image name compatibility checking
func TestImagesCompatible(t *testing.T) {
	tests := []struct {
		name             string
		checkpointImage  string
		placeholderImage string
		want             bool
	}{
		// Exact matches
		{
			name:             "exact match - simple",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "nginx:alpine",
			want:             true,
		},
		{
			name:             "exact match - with registry",
			checkpointImage:  "docker.io/library/nginx:alpine",
			placeholderImage: "docker.io/library/nginx:alpine",
			want:             true,
		},

		// Registry normalization - docker.io/library/
		{
			name:             "checkpoint has docker.io/library prefix",
			checkpointImage:  "docker.io/library/nginx:alpine",
			placeholderImage: "nginx:alpine",
			want:             true,
		},
		{
			name:             "placeholder has docker.io/library prefix",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "docker.io/library/nginx:alpine",
			want:             true,
		},
		{
			name:             "both have docker.io/library prefix - same image",
			checkpointImage:  "docker.io/library/ubuntu:22.04",
			placeholderImage: "docker.io/library/ubuntu:22.04",
			want:             true,
		},

		// Registry normalization - docker.io/
		{
			name:             "checkpoint has docker.io prefix",
			checkpointImage:  "docker.io/nginx:alpine",
			placeholderImage: "nginx:alpine",
			want:             true,
		},
		{
			name:             "placeholder has docker.io prefix",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "docker.io/nginx:alpine",
			want:             true,
		},

		// Placeholder naming convention
		{
			name:             "placeholder convention - simple",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "criu-placeholder-nginx-alpine",
			want:             true,
		},
		{
			name:             "placeholder convention - with tag",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "criu-placeholder-nginx-alpine:v1",
			want:             true,
		},
		{
			name:             "placeholder convention - ubuntu",
			checkpointImage:  "ubuntu:22.04",
			placeholderImage: "criu-placeholder-ubuntu-22.04",
			want:             true,
		},
		{
			name:             "placeholder convention - with registry normalization",
			checkpointImage:  "docker.io/library/nginx:alpine",
			placeholderImage: "criu-placeholder-nginx-alpine",
			want:             true,
		},
		{
			name:             "placeholder convention - multi-part image",
			checkpointImage:  "myregistry.io/myorg/app:v1.0",
			placeholderImage: "criu-placeholder-myregistry.io-myorg-app-v1.0",
			want:             true,
		},

		// Mismatches
		{
			name:             "different images",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "ubuntu:22.04",
			want:             false,
		},
		{
			name:             "different tags",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "nginx:latest",
			want:             false,
		},
		{
			name:             "wrong placeholder convention",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "criu-placeholder-ubuntu-22.04",
			want:             false,
		},
		{
			name:             "placeholder without prefix",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "nginx-alpine",
			want:             false,
		},
		{
			name:             "empty checkpoint image",
			checkpointImage:  "",
			placeholderImage: "nginx:alpine",
			want:             false,
		},
		{
			name:             "empty placeholder image",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "",
			want:             false,
		},
		{
			name:             "both empty",
			checkpointImage:  "",
			placeholderImage: "",
			want:             true,
		},

		// Edge cases
		{
			name:             "registry with different paths",
			checkpointImage:  "docker.io/library/nginx:alpine",
			placeholderImage: "docker.io/myuser/nginx:alpine",
			want:             false,
		},
		{
			name:             "image with slashes in org",
			checkpointImage:  "myregistry.io/org/team/app:v1",
			placeholderImage: "criu-placeholder-myregistry.io-org-team-app-v1",
			want:             true,
		},
		{
			name:             "placeholder with extra suffix",
			checkpointImage:  "nginx:alpine",
			placeholderImage: "criu-placeholder-nginx-alpine-extra",
			want:             false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := imagesCompatible(tt.checkpointImage, tt.placeholderImage)
			if got != tt.want {
				t.Errorf("imagesCompatible(%q, %q) = %v, want %v",
					tt.checkpointImage, tt.placeholderImage, got, tt.want)
			}
		})
	}
}
