// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpoint

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestSnapshotCompatibilityRetainsEnvironmentDependencies(t *testing.T) {
	for _, origin := range []string{"", "1.7.0", "1.8.0"} {
		for _, test := range []struct {
			name       string
			before     []corev1.EnvVar
			after      []corev1.EnvVar
			args       []string
			wantChange bool
		}{
			{name: "excluded dependency moves after retained value", wantChange: true,
				before: []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}, {Name: "MODEL_ID", Value: "$(DYN_SERVED_MODEL_NAME)"}},
				after:  []corev1.EnvVar{{Name: "MODEL_ID", Value: "$(DYN_SERVED_MODEL_NAME)"}, {Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}}},
			{name: "transitive dependency changes", wantChange: true,
				before: []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_SERVED_MODEL_NAME)"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}},
				after:  []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-b"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_SERVED_MODEL_NAME)"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}}},
			{name: "later duplicate did not supply the transitive value",
				before: []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_SERVED_MODEL_NAME)"}, {Name: "DYN_SERVED_MODEL_NAME", Value: "unused-a"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}},
				after:  []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_SERVED_MODEL_NAME)"}, {Name: "DYN_SERVED_MODEL_NAME", Value: "unused-b"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}}},
			{name: "self-reference reads its preceding definition", wantChange: true,
				before: []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "a"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_NAMESPACE)/child"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}},
				after:  []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "b"}, {Name: "DYN_NAMESPACE", Value: "$(DYN_NAMESPACE)/child"}, {Name: "MODEL_ID", Value: "$(DYN_NAMESPACE)"}}},
			{name: "escaped reference is not a dependency",
				before: []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "a"}, {Name: "MODEL_ID", Value: "$$(DYN_NAMESPACE)"}},
				after:  []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "b"}, {Name: "MODEL_ID", Value: "$$(DYN_NAMESPACE)"}}},
			{name: "unrelated restored values stay excluded",
				before: []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "a"}, {Name: "MODEL_ID", Value: "model"}},
				after:  []corev1.EnvVar{{Name: "DYN_NAMESPACE", Value: "b"}, {Name: "MODEL_ID", Value: "model"}}},
			{name: "argv expansion captures a restored input", wantChange: true, args: []string{"--model", "$(DYN_SERVED_MODEL_NAME)"},
				before: []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}},
				after:  []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-b"}}},
		} {
			t.Run(origin+"/"+test.name, func(t *testing.T) {
				t.Log("Compare captured inputs while preserving the legacy compatibility identity")
				before := snapshotCompatibilityTestPodTemplate("graph", "ns", "worker")
				before.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin}
				before.Spec.Containers[0].Env = test.before
				if test.args != nil {
					before.Spec.Containers[0].Args = test.args
				}
				after := before.DeepCopy()
				after.Spec.Containers[0].Env = test.after
				original := before.DeepCopy()
				beforeHash, err := ComputeSnapshotCompatibilityHash(&before, "main", "vllm", "disabled", "", origin == "1.8.0")
				require.NoError(t, err)
				afterHash, err := ComputeSnapshotCompatibilityHash(after, "main", "vllm", "disabled", "", origin == "1.8.0")
				require.NoError(t, err)
				if origin == "1.8.0" && test.wantChange {
					require.NotEqual(t, beforeHash, afterHash)
				} else {
					require.Equal(t, beforeHash, afterHash)
				}
				require.Equal(t, original, &before)
			})
		}
	}
}
