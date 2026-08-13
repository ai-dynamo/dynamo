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

// The follower is NOT a Grove clique: Grove rejects minAvailable:0 (grove#676), so a
// follower that rests at zero would gang-block the leader. expandRolesForComponent must
// therefore emit only the leader role; the follower is rendered on the non-Grove pathway
// (synthesizeElasticEPFollowerDCD). Revisit once grove#686 lands.
func TestExpandRolesForComponent_NoGroveFollowerForElasticEP(t *testing.T) {
	for _, r := range expandRolesForComponent("decode", nil, 1, elasticEPComponent()) {
		if r.Role == RoleFollower {
			t.Errorf("elastic-EP must NOT emit a Grove follower clique (grove#676); got %+v", r)
		}
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

// leaderDCD builds a single-pod elastic-EP leader DynamoComponentDeployment as
// GenerateDynamoComponentsDeployments would, so synthesizeElasticEPFollowerDCD can derive
// the follower from it.
func leaderDCD() *v1beta1.DynamoComponentDeployment {
	return &v1beta1.DynamoComponentDeployment{
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: *elasticEPComponent(),
		},
	}
}

func TestSynthesizeElasticEPFollowerDCD(t *testing.T) {
	leader := leaderDCD()
	leader.Name = "mydgd-decode"
	leader.Labels = map[string]string{commonconsts.KubeLabelDynamoComponent: "decode"}

	follower := synthesizeElasticEPFollowerDCD(leader, "decode")
	if follower == nil {
		t.Fatal("expected a follower DCD for an elastic-EP leader")
	}
	if want := "mydgd-decode-" + commonconsts.GroveRoleSuffixFollower; follower.Name != want {
		t.Errorf("follower Name = %q, want %q", follower.Name, want)
	}
	if follower.Spec.Replicas == nil || *follower.Spec.Replicas != 0 {
		t.Errorf("follower Replicas = %v, want 0 (rests at zero, scaled on demand)", follower.Spec.Replicas)
	}
	if want := "decode-" + commonconsts.GroveRoleSuffixFollower; follower.Labels[commonconsts.KubeLabelDynamoComponent] != want {
		t.Errorf("follower component label = %q, want %q (distinct so its resources never collide with the leader)",
			follower.Labels[commonconsts.KubeLabelDynamoComponent], want)
	}
	if follower.Annotations[commonconsts.KubeAnnotationElasticEPFollower] != commonconsts.KubeLabelValueTrue {
		t.Errorf("follower must carry the marker annotation so the renderer picks RoleFollower; got %v", follower.Annotations)
	}
	// The leader is untouched by the derivation.
	if leader.Spec.Replicas != nil {
		t.Errorf("leader Replicas must not be mutated by follower synthesis; got %v", leader.Spec.Replicas)
	}
}

func TestSynthesizeElasticEPFollowerDCD_NilForNonElasticEP(t *testing.T) {
	plain := &v1beta1.DynamoComponentDeployment{
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name: commonconsts.MainContainerName,
							Args: []string{"-m", "dynamo.vllm"},
						}},
					},
				},
			},
		},
	}
	if got := synthesizeElasticEPFollowerDCD(plain, "decode"); got != nil {
		t.Errorf("a non-elastic-EP leader must not synthesize a follower; got %+v", got)
	}
}

func TestInjectElasticEPRayLaunchFlags_Follower(t *testing.T) {
	// Both the Grove role name (<leader>) and the non-Grove synthesized name
	// (<leader>-flw) must resolve to the same leader headless Service <leader>-ray.
	for _, serviceName := range []string{"my-worker", "my-worker-" + commonconsts.GroveRoleSuffixFollower} {
		container := &corev1.Container{
			Command: []string{"python3"},
			Args:    []string{"-m", "dynamo.vllm", "--enable-elastic-ep", "--data-parallel-backend", "ray"},
		}

		if !injectElasticEPRayLaunchFlags(container, RoleFollower, serviceName, nil) {
			t.Fatalf("[%s] expected the follower launch to be injected", serviceName)
		}
		if len(container.Args) != 1 {
			t.Fatalf("[%s] expected 1 arg (the shell script), got %d", serviceName, len(container.Args))
		}
		script := container.Args[0]

		// Joins the leader Service (<leader>-ray), not the follower's own name.
		if !strings.Contains(script, "ray start --address=my-worker-ray:6379") {
			t.Errorf("[%s] follower must join the leader Service my-worker-ray; got: %s", serviceName, script)
		}
		if !strings.Contains(script, `--node-ip-address="$POD_IP"`) {
			t.Errorf("[%s] follower must pin --node-ip-address to POD_IP; got: %s", serviceName, script)
		}
		if !strings.Contains(script, "/live") {
			t.Errorf("[%s] follower must health-gate on the leader /live; got: %s", serviceName, script)
		}
		// The follower never serves; the leader spawns the real DP-rank worker on its GPU.
		// The serve flags must not survive into the follower's command. ("dynamo.vllm"
		// appears in the /live wait log, so assert on a serve-only flag instead.)
		if strings.Contains(script, "--enable-elastic-ep") {
			t.Errorf("[%s] follower must NOT run the vLLM serve command; got: %s", serviceName, script)
		}
		if len(container.Command) != 2 || container.Command[0] != "/bin/sh" {
			t.Errorf("[%s] expected [/bin/sh -c] command, got %v", serviceName, container.Command)
		}
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
