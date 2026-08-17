// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package executor

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

func TestRequiredSourceMounts(t *testing.T) {
	candidates := []nvidiacomv1alpha1.CheckpointSourceMount{
		{Path: "/model-cache", Volume: "model-cache", VolumeSource: "PersistentVolumeClaim/model-cache"},
		{Path: "/scratch", Volume: "scratch", VolumeSource: "EmptyDir"},
		{Path: "/not-externalized", Volume: "ignored", VolumeSource: "ConfigMap/ignored"},
	}

	mounts := requiredSourceMounts(candidates, map[string]string{
		"/model-cache":         "/model-cache",
		"/scratch":             "/scratch",
		"/etc/hosts":           "/etc/hosts",
		"/var/run/secrets/k8s": "/var/run/secrets/k8s",
	})

	assert.Equal(t, candidates[:2], mounts)
}

func TestRequiredSourceMountsRecordsKnownZero(t *testing.T) {
	mounts := requiredSourceMounts(nil, nil)

	require.NotNil(t, mounts)
	assert.Empty(t, mounts)
}

func TestRequiredSourceMountsMatchesRunAliases(t *testing.T) {
	candidates := []nvidiacomv1alpha1.CheckpointSourceMount{
		{Path: "/var/run/model-cache", Volume: "model-cache", VolumeSource: "PersistentVolumeClaim/model-cache"},
		{Path: "/run/credentials", Volume: "credentials", VolumeSource: "Secret/credentials"},
	}

	mounts := requiredSourceMounts(candidates, map[string]string{
		"/run/model-cache":     "/run/model-cache",
		"/var/run/credentials": "/var/run/credentials",
		"/etc/hosts":           "/etc/hosts",
	})

	assert.Equal(t, candidates, mounts)
}

func TestRequiredSourceMountsMatchesAliasedCandidates(t *testing.T) {
	candidates := []nvidiacomv1alpha1.CheckpointSourceMount{
		{Path: "/run/data", Volume: "first", VolumeSource: "EmptyDir"},
		{Path: "/var/run/data", Volume: "second", VolumeSource: "EmptyDir"},
	}

	mounts := requiredSourceMounts(candidates, map[string]string{"/run/data": "/run/data"})

	assert.Equal(t, candidates, mounts)
}
