/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestApplyCyborgWeightsPathPrecedesAuthoredReferences(t *testing.T) {
	t.Parallel()

	t.Log("Define authored bindings that depend on the projected checkpoint location")
	projection := &ModelProjection{checkpoint: &dynamov1beta1.LPXCheckpoint{
		Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace,
		Model:    "openai/gpt-oss-20b",
		Revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
	}}
	authored := []corev1.EnvVar{
		{Name: "CYBORG_ARGS", Value: "--weights-path $(CYBORG_WEIGHTS_PATH)"},
		{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	want := append([]corev1.EnvVar{{
		Name:  cyborgWeightsPathEnv,
		Value: "/nfs/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
	}}, authored...)
	for _, test := range []struct {
		name string
		env  []corev1.EnvVar
	}{
		{name: "missing binding", env: authored},
		{name: "stale binding after reference", env: append(append([]corev1.EnvVar(nil), authored...), corev1.EnvVar{
			Name: cyborgWeightsPathEnv, Value: "/nfs/huggingface/hub/models--openai--gpt-oss-20b/snapshots/stale",
		})},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish the snapshot path before references while preserving other bindings")
			container := corev1.Container{Env: test.env}
			applyCyborgWeightsPath(&container, projection, "/nfs")
			require.Equal(t, want, container.Env)

			t.Log("Repeat rendering without duplicating or reordering environment bindings")
			applyCyborgWeightsPath(&container, projection, "/nfs")
			require.Equal(t, want, container.Env)
		})
	}
}

func TestApplyCyborgWeightsPathWithoutCheckpointKeepsAuthoredValue(t *testing.T) {
	t.Parallel()

	t.Log("Leave an authored weights path in place when no checkpoint is selected")
	authored := []corev1.EnvVar{{Name: cyborgWeightsPathEnv, Value: "/nfs/authored"}}
	container := corev1.Container{Env: append([]corev1.EnvVar(nil), authored...)}
	applyCyborgWeightsPath(&container, &ModelProjection{}, "/nfs")
	require.Equal(t, authored, container.Env)
}

func TestCheckpointSnapshotPath(t *testing.T) {
	t.Parallel()

	t.Log("Map repository IDs to the Hugging Face cache layout under model storage")
	for _, test := range []struct {
		model string
		want  string
	}{
		{model: "openai/gpt-oss-20b", want: "/mnt/models/models--openai--gpt-oss-20b/snapshots/0123456789abcdef0123456789abcdef01234567"},
		{model: "gpt2", want: "/mnt/models/models--gpt2/snapshots/0123456789abcdef0123456789abcdef01234567"},
	} {
		checkpoint := &dynamov1beta1.LPXCheckpoint{Model: test.model, Revision: "0123456789abcdef0123456789abcdef01234567"}
		require.Equal(t, test.want, checkpointSnapshotPath(checkpoint, "/mnt/models"))
	}
}

func TestProjectModelCheckpoint(t *testing.T) {
	t.Parallel()
	checkpoint := &dynamov1beta1.LPXCheckpoint{
		Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace,
		Model:    "openai/gpt-oss-20b",
		Revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
	}
	for _, test := range []struct {
		name     string
		pipeline Pipeline
		wantErr  string
	}{
		{name: "hybrid conductor receives the checkpoint", pipeline: PipelineLPX},
		{
			name: "LPU-only pipeline has no Cyborg conductor", pipeline: PipelineSingle,
			wantErr: "unsupported LPX runtime: checkpoint requires a hybrid build with a Cyborg conductor",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a hybrid build with a checkpoint")
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			normalized.build.CompilationMode = BuildCompilationModeHybrid
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: test.pipeline, Models: []string{"default"}, BuildSnapshot: normalized,
				Checkpoint: checkpoint,
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Retain the checkpoint for Cyborg without changing the workload digest")
			require.Same(t, checkpoint, projections[0].checkpoint)
			baseline := projectTestBuild(t, normalized, test.pipeline)
			require.Equal(t, baseline.Digest(), projections[0].Digest())
		})
	}
}
