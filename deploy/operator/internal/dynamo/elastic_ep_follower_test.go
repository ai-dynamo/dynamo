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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func elasticEPComponent() *v1beta1.DynamoComponentDeploymentSharedSpec {
	return &v1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:    commonconsts.MainContainerName,
					Command: []string{"python3"},
					Args: []string{
						"-m", "dynamo.vllm",
						"--enable-elastic-ep",
						"--data-parallel-backend", "ray",
					},
				}},
			},
		},
	}
}

func TestExpandRolesForComponent_EmitsElasticEPFollower(t *testing.T) {
	roles := expandRolesForComponent("decode", nil, 1, elasticEPComponent())

	var main, follower *ServiceRole
	for i := range roles {
		switch roles[i].Role {
		case RoleMain:
			main = &roles[i]
		case RoleFollower:
			follower = &roles[i]
		}
	}
	if main == nil {
		t.Fatal("expected a RoleMain leader")
	}
	if follower == nil {
		t.Fatal("expected a RoleFollower clique")
	}
	if follower.Replicas != 0 {
		t.Errorf("follower Replicas = %d, want 0 (rests at zero, scaled on demand)", follower.Replicas)
	}
	if want := "decode-" + commonconsts.GroveRoleSuffixFollower; follower.Name != want {
		t.Errorf("follower Name = %q, want %q", follower.Name, want)
	}
}

func TestExpandRolesForComponent_NoFollowerForNonElasticEP(t *testing.T) {
	plain := &v1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name: commonconsts.MainContainerName,
					Args: []string{"-m", "dynamo.vllm"},
				}},
			},
		},
	}
	for _, r := range expandRolesForComponent("decode", nil, 1, plain) {
		if r.Role == RoleFollower {
			t.Errorf("a non-elastic-EP component must not get a follower; got %+v", r)
		}
	}
}

func TestInjectElasticEPRayLaunchFlags_Follower(t *testing.T) {
	container := &corev1.Container{
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm", "--enable-elastic-ep", "--data-parallel-backend", "ray"},
	}

	if !injectElasticEPRayLaunchFlags(container, RoleFollower, "my-worker", nil) {
		t.Fatal("expected the follower launch to be injected")
	}
	if len(container.Args) != 1 {
		t.Fatalf("expected 1 arg (the shell script), got %d", len(container.Args))
	}
	script := container.Args[0]

	// Joins the Phase-3 leader Service (<component>-ray), not a gang leader hostname.
	if !strings.Contains(script, "ray start --address=my-worker-ray:6379") {
		t.Errorf("follower must join the elastic-EP leader Service; got: %s", script)
	}
	if !strings.Contains(script, `--node-ip-address="$POD_IP"`) {
		t.Errorf("follower must pin --node-ip-address to POD_IP; got: %s", script)
	}
	if !strings.Contains(script, "/live") {
		t.Errorf("follower must health-gate on the leader /live; got: %s", script)
	}
	// The follower never serves; the leader spawns the real DP-rank worker on its GPU.
	// The serve flags (e.g. --enable-elastic-ep) must not survive into the follower's
	// command -- only the Ray join runs. ("dynamo.vllm" appears in the /live wait log,
	// so assert on a serve-only flag instead.)
	if strings.Contains(script, "--enable-elastic-ep") {
		t.Errorf("follower must NOT run the vLLM serve command; got: %s", script)
	}
	if len(container.Command) != 2 || container.Command[0] != "/bin/sh" {
		t.Errorf("expected [/bin/sh -c] command, got %v", container.Command)
	}
}

func TestInjectElasticEPFollowerAntiAffinity(t *testing.T) {
	podSpec := &corev1.PodSpec{}
	injectElasticEPFollowerAntiAffinity(podSpec, "decode", "ns-mydgd")

	if podSpec.Affinity == nil || podSpec.Affinity.PodAntiAffinity == nil {
		t.Fatal("expected pod anti-affinity to be set")
	}
	terms := podSpec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if len(terms) != 1 {
		t.Fatalf("expected 1 required anti-affinity term, got %d", len(terms))
	}
	if terms[0].TopologyKey != "kubernetes.io/hostname" {
		t.Errorf("topologyKey = %q, want kubernetes.io/hostname", terms[0].TopologyKey)
	}
	sel := terms[0].LabelSelector.MatchLabels
	if sel[commonconsts.KubeLabelDynamoComponent] != "decode" ||
		sel[commonconsts.KubeLabelDynamoNamespace] != "ns-mydgd" {
		t.Errorf("selector = %v, want component=decode namespace=ns-mydgd", sel)
	}
}

func TestInjectElasticEPFollowerAntiAffinity_PreservesUserAffinity(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{}},
	}
	injectElasticEPFollowerAntiAffinity(podSpec, "decode", "ns-mydgd")

	if podSpec.Affinity.NodeAffinity == nil {
		t.Error("user-supplied NodeAffinity was dropped")
	}
	if len(podSpec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 1 {
		t.Error("the required anti-affinity term was not added")
	}
}
