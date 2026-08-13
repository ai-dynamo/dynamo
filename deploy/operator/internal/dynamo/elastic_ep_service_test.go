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
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestElasticEPLeaderServiceName(t *testing.T) {
	if got, want := ElasticEPLeaderServiceName("my-dgd-worker"), "my-dgd-worker-ray"; got != want {
		t.Fatalf("ElasticEPLeaderServiceName = %q, want %q", got, want)
	}
}

func TestGenerateElasticEPHeadlessService(t *testing.T) {
	svc := GenerateElasticEPHeadlessService(ComponentServiceParams{
		ServiceName:     "my-dgd-worker",
		Namespace:       "ns",
		ComponentType:   "worker",
		DynamoNamespace: "ns-my-dgd",
		ComponentName:   "worker",
		Labels:          map[string]string{"app": "x"},
		Annotations:     map[string]string{"a": "b"},
	})

	if got, want := svc.Name, "my-dgd-worker-ray"; got != want {
		t.Errorf("Name = %q, want %q", got, want)
	}
	if svc.Namespace != "ns" {
		t.Errorf("Namespace = %q, want ns", svc.Namespace)
	}

	// Headless: Ray's multi-port head<->worker traffic needs a direct pod address,
	// not a load-balanced ClusterIP.
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Errorf("ClusterIP = %q, want %q", svc.Spec.ClusterIP, corev1.ClusterIPNone)
	}
	// The follower must reach the Ray head before the leader is Ready (the engine only
	// starts once ranks join), so not-ready addresses must be published.
	if !svc.Spec.PublishNotReadyAddresses {
		t.Error("PublishNotReadyAddresses = false, want true (else follower<->leader deadlocks)")
	}

	// Selector points at the elastic-EP component (the single-pod leader today).
	wantSel := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: "worker",
		commonconsts.KubeLabelDynamoNamespace:     "ns-my-dgd",
		commonconsts.KubeLabelDynamoComponent:     "worker",
	}
	for k, v := range wantSel {
		if svc.Spec.Selector[k] != v {
			t.Errorf("Selector[%q] = %q, want %q", k, svc.Spec.Selector[k], v)
		}
	}

	// Ports: Ray GCS (6379) + the leader system/health port (9090).
	ports := map[string]int32{}
	for _, p := range svc.Spec.Ports {
		ports[p.Name] = p.Port
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

	// User labels/annotations are carried through.
	if svc.Labels["app"] != "x" {
		t.Errorf("Labels[app] = %q, want x", svc.Labels["app"])
	}
	if svc.Annotations["a"] != "b" {
		t.Errorf("Annotations[a] = %q, want b", svc.Annotations["a"])
	}
}
