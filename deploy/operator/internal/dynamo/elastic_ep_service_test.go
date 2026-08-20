/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"strings"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestElasticEPLeaderServiceName(t *testing.T) {
	longComponentServiceName := "a-very-long-dynamo-graph-deployment-name-with-a-long-component-name"

	tests := []struct {
		name                 string
		componentServiceName string
		want                 string
	}{
		{
			name:                 "appends the ray suffix",
			componentServiceName: "my-dgd-worker",
			want:                 "my-dgd-worker-ray",
		},
		{
			name:                 "normalizes dots that a Service name may not carry",
			componentServiceName: "my-dgd-qwen3-0.6b",
			want:                 "my-dgd-qwen3-0-6b-ray",
		},
		{
			name:                 "keeps a name that lands exactly on the limit",
			componentServiceName: strings.Repeat("a", maxServiceNameLength-len("-ray")),
			want:                 strings.Repeat("a", maxServiceNameLength-len("-ray")) + "-ray",
		},
		{
			name:                 "truncates past the limit with a deterministic hash suffix",
			componentServiceName: longComponentServiceName,
			want:                 "a-very-long-dynamo-graph-deployment-name-with-a-long-compo-502d",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Derive the leader Service name from the component Service name")
			got := ElasticEPLeaderServiceName(tt.componentServiceName)

			t.Log("Verify the derived name and that it stays a valid Service name")
			if got != tt.want {
				t.Errorf("ElasticEPLeaderServiceName(%q) = %q, want %q", tt.componentServiceName, got, tt.want)
			}
			if len(got) > maxServiceNameLength {
				t.Errorf("name %q is %d chars, want at most %d", got, len(got), maxServiceNameLength)
			}
		})
	}
}

func TestElasticEPLeaderServiceNameIsStable(t *testing.T) {
	t.Log("Derive a truncated name twice from the same input")
	overLimit := "a-very-long-dynamo-graph-deployment-name-with-a-long-component-name"
	first := ElasticEPLeaderServiceName(overLimit)
	second := ElasticEPLeaderServiceName(overLimit)

	t.Log("Verify the truncation is deterministic, because the follower derives this address independently")
	if first != second {
		t.Errorf("ElasticEPLeaderServiceName is not deterministic: %q then %q", first, second)
	}
}

func TestGenerateElasticEPHeadlessService(t *testing.T) {
	t.Log("Generate the leader Service for a single-pod elastic-EP component")
	svc := GenerateElasticEPHeadlessService(ComponentServiceParams{
		ServiceName:     "my-dgd-worker",
		Namespace:       "ns",
		ComponentType:   "worker",
		DynamoNamespace: "ns-my-dgd",
		ComponentName:   "worker",
		Labels:          map[string]string{"app": "x"},
		Annotations:     map[string]string{"a": "b"},
	})

	t.Log("Verify the Service identity")
	if got, want := svc.Name, "my-dgd-worker-ray"; got != want {
		t.Errorf("Name = %q, want %q", got, want)
	}
	if svc.Namespace != "ns" {
		t.Errorf("Namespace = %q, want ns", svc.Namespace)
	}

	t.Log("Verify it is headless, because Ray's multi-port head<->worker traffic needs a direct pod address rather than a load-balanced ClusterIP")
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Errorf("ClusterIP = %q, want %q", svc.Spec.ClusterIP, corev1.ClusterIPNone)
	}

	t.Log("Verify not-ready addresses are published, because the follower must reach the Ray head before the leader is Ready (the engine only starts once ranks join)")
	if !svc.Spec.PublishNotReadyAddresses {
		t.Error("PublishNotReadyAddresses = false, want true (else follower<->leader deadlocks)")
	}

	t.Log("Verify the selector points at the elastic-EP component, which the caller gates to a single pod")
	wantSelector := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: "worker",
		commonconsts.KubeLabelDynamoNamespace:     "ns-my-dgd",
		commonconsts.KubeLabelDynamoComponent:     "worker",
	}
	for key, want := range wantSelector {
		if got := svc.Spec.Selector[key]; got != want {
			t.Errorf("Selector[%q] = %q, want %q", key, got, want)
		}
	}

	t.Log("Verify both ports are published: Ray GCS for the raylet join and the system port for the follower's /live gate")
	ports := map[string]int32{}
	for _, port := range svc.Spec.Ports {
		ports[port.Name] = port.Port
	}
	if ports["ray-gcs"] != 6379 {
		t.Errorf("ray-gcs port = %d, want 6379", ports["ray-gcs"])
	}
	if ports[commonconsts.DynamoSystemPortName] != commonconsts.DynamoSystemPort {
		t.Errorf(
			"system port = %d, want %d",
			ports[commonconsts.DynamoSystemPortName], commonconsts.DynamoSystemPort,
		)
	}

	t.Log("Verify user labels and annotations are carried through")
	if svc.Labels["app"] != "x" {
		t.Errorf("Labels[app] = %q, want x", svc.Labels["app"])
	}
	if svc.Annotations["a"] != "b" {
		t.Errorf("Annotations[a] = %q, want b", svc.Annotations["a"])
	}
}
