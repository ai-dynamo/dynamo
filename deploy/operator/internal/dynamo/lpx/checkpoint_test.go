/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"capnproto.org/go/capnp/v3"
	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
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
	manifestCheckpoint := &dynamov1beta1.LPXCheckpoint{
		Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace,
		Model:    "openai/gpt-oss-20b",
		Revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
	}
	override := &dynamov1beta1.LPXCheckpoint{
		Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace,
		Model:    "openai/gpt-oss-20b",
		Revision: "0123456789abcdef0123456789abcdef01234567",
	}
	for _, test := range []struct {
		name               string
		pipeline           Pipeline
		manifestCheckpoint *dynamov1beta1.LPXCheckpoint
		override           *dynamov1beta1.LPXCheckpoint
		want               *dynamov1beta1.LPXCheckpoint
		wantErr            string
	}{
		{name: "hybrid build uses its manifest checkpoint", pipeline: PipelineLPX, manifestCheckpoint: manifestCheckpoint, want: manifestCheckpoint},
		{name: "override replaces the manifest checkpoint", pipeline: PipelineLPX, manifestCheckpoint: manifestCheckpoint, override: override, want: override},
		{name: "override applies without a manifest checkpoint", pipeline: PipelineLPX, override: override, want: override},
		{name: "hybrid build without a checkpoint projects none", pipeline: PipelineLPX},
		{name: "LPU-only build ignores its manifest checkpoint", pipeline: PipelineSingle, manifestCheckpoint: manifestCheckpoint},
		{
			name: "LPU-only build rejects an override", pipeline: PipelineSingle, override: override,
			wantErr: "unsupported LPX runtime: experimental.checkpoint requires a hybrid build with a Cyborg conductor",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a hybrid build whose manifest may name a checkpoint")
			normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
			normalized.build.CompilationMode = BuildCompilationModeHybrid
			normalized.build.Checkpoint = test.manifestCheckpoint
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: test.pipeline, Models: []string{"default"}, BuildSnapshot: normalized,
				CheckpointOverride: test.override,
			})
			if test.wantErr != "" {
				require.EqualError(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Project the selected checkpoint without changing the workload digest")
			require.Same(t, test.want, projections[0].checkpoint)
			baseline := projectTestBuild(t, normalized, test.pipeline)
			require.Equal(t, baseline.Digest(), projections[0].Digest())
		})
	}
}

func TestCheckpointFromManifestV2(t *testing.T) {
	const revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
	for _, test := range []struct {
		name     string
		absent   bool
		provider manifestcapnpv2.CheckpointProvider
		model    string
		revision string
		want     *dynamov1beta1.LPXCheckpoint
		wantErr  string
	}{
		{name: "absent checkpoint", absent: true},
		{
			name: "pinned Hugging Face checkpoint", model: "openai/gpt-oss-20b", revision: revision,
			want: &dynamov1beta1.LPXCheckpoint{Provider: dynamov1beta1.LPXCheckpointProviderHuggingFace, Model: "openai/gpt-oss-20b", Revision: revision},
		},
		{name: "unknown provider", provider: 1, model: "openai/gpt-oss-20b", revision: revision, wantErr: "checkpoint.provider 1 is not supported"},
		{name: "path traversal", model: "openai/../gpt-oss-20b", revision: revision, wantErr: `checkpoint.model "openai/../gpt-oss-20b" is not a valid Hugging Face repository ID`},
		{name: "cache layout alias", model: "openai/gpt--oss", revision: revision, wantErr: `checkpoint.model "openai/gpt--oss" is not a valid Hugging Face repository ID`},
		{name: "git suffix", model: "openai/gpt-oss.git", revision: revision, wantErr: `checkpoint.model "openai/gpt-oss.git" is not a valid Hugging Face repository ID`},
		{name: "branch revision", model: "openai/gpt-oss-20b", revision: "main", wantErr: `checkpoint.revision "main" is not a full commit SHA`},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Encode a manifest with the selected checkpoint")
			_, segment := capnp.NewSingleSegmentMessage(nil)
			manifest, err := manifestcapnpv2.NewRootManifest(segment)
			require.NoError(t, err)
			if !test.absent {
				checkpoint, err := manifest.NewCheckpoint()
				require.NoError(t, err)
				checkpoint.SetProvider(test.provider)
				require.NoError(t, checkpoint.SetModel(test.model))
				require.NoError(t, checkpoint.SetRevision(test.revision))
			}

			t.Log("Accept only a pinned Hugging Face repository snapshot")
			checkpoint, err := checkpointFromManifestV2(manifest)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, checkpoint)
		})
	}
}
