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

// vllmComponent builds a single-pod vLLM component whose main container carries the
// given extra args, so each test can describe the launch it wants at a high level.
func vllmComponent(extraArgs ...string) *v1beta1.DynamoComponentDeploymentSharedSpec {
	return &v1beta1.DynamoComponentDeploymentSharedSpec{
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:    commonconsts.MainContainerName,
					Command: []string{"python3"},
					Args:    append([]string{"-m", "dynamo.vllm"}, extraArgs...),
				}},
			},
		},
	}
}

func elasticEPComponent() *v1beta1.DynamoComponentDeploymentSharedSpec {
	return vllmComponent("--enable-elastic-ep", "--data-parallel-backend", "ray")
}

// The follower is NOT a Grove clique: Grove rejects minAvailable:0 (grove#676), so a
// follower that rests at zero would gang-block the leader. expandRolesForComponent must
// therefore emit only the leader role; the follower is rendered on the non-Grove pathway
// (synthesizeElasticEPFollowerDCD). Revisit once grove#686 lands.
func TestExpandRolesForComponent_NeverEmitsFollower(t *testing.T) {
	tests := []struct {
		name      string
		component *v1beta1.DynamoComponentDeploymentSharedSpec
	}{
		{name: "elastic EP on the ray backend", component: elasticEPComponent()},
		{name: "plain single-pod vLLM", component: vllmComponent()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("expanding the component's Grove roles")
			roles := expandRolesForComponent("decode", nil, 1, tt.component)

			t.Log("no role may be RoleFollower: a minAvailable:0 clique gang-blocks the leader")
			for _, r := range roles {
				if r.Role == RoleFollower {
					t.Errorf("must NOT emit a Grove follower clique (grove#676); got %+v", r)
				}
			}
		})
	}
}

// leaderDCD builds a single-pod elastic-EP leader as GenerateDynamoComponentsDeployments
// would. It sets both Spec.ComponentName and the label because GetDCDComponentName
// prefers the former: the follower must override both, or the worker hash sees two DCDs
// named "decode".
func leaderDCD(component *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoComponentDeployment {
	dcd := &v1beta1.DynamoComponentDeployment{
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: *component,
		},
	}
	dcd.Name = "mydgd-decode"
	dcd.Spec.ComponentName = "decode"
	dcd.Labels = map[string]string{commonconsts.KubeLabelDynamoComponent: "decode"}
	return dcd
}

func TestSynthesizeElasticEPFollowerDCD_OnlyForElasticEP(t *testing.T) {
	tests := []struct {
		name          string
		component     *v1beta1.DynamoComponentDeploymentSharedSpec
		wantSynthesis bool
	}{
		{name: "elastic EP on the ray backend gets a follower", component: elasticEPComponent(), wantSynthesis: true},
		{name: "plain vLLM gets none", component: vllmComponent(), wantSynthesis: false},
		{name: "elastic EP without the ray backend gets none", component: vllmComponent("--enable-elastic-ep"), wantSynthesis: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("deriving the follower from the leader DCD")
			follower := synthesizeElasticEPFollowerDCD(leaderDCD(tt.component), "decode")

			t.Log("only an elastic-EP Ray launch has a Ray cluster for a follower to join")
			if gotSynthesis := follower != nil; gotSynthesis != tt.wantSynthesis {
				t.Fatalf("synthesized = %v, want %v (got %+v)", gotSynthesis, tt.wantSynthesis, follower)
			}
		})
	}
}

func TestSynthesizeElasticEPFollowerDCD_DerivesADistinctIdentity(t *testing.T) {
	leader := leaderDCD(elasticEPComponent())
	wantSuffixed := "decode-" + commonconsts.GroveRoleSuffixFollower

	t.Log("deriving the follower from a single-pod elastic-EP leader")
	follower := synthesizeElasticEPFollowerDCD(leader, "decode")
	if follower == nil {
		t.Fatal("expected a follower DCD for an elastic-EP leader")
	}

	t.Log("its resource name and component identity must both carry the -flw suffix, so its Deployment, Service, selector, and worker hash never collide with the leader's")
	if want := "mydgd-decode-" + commonconsts.GroveRoleSuffixFollower; follower.Name != want {
		t.Errorf("follower Name = %q, want %q", follower.Name, want)
	}
	if got := GetDCDComponentName(follower); got != wantSuffixed {
		t.Errorf("GetDCDComponentName(follower) = %q, want %q (distinct from the leader for the worker hash)", got, wantSuffixed)
	}
	if got := follower.Labels[commonconsts.KubeLabelDynamoComponent]; got != wantSuffixed {
		t.Errorf("follower component label = %q, want %q (keeps the leader's headless Service from selecting it)", got, wantSuffixed)
	}

	t.Log("it rests at zero replicas so it never gang-blocks the leader, and is scaled on demand")
	if follower.Spec.Replicas == nil || *follower.Spec.Replicas != 0 {
		t.Errorf("follower Replicas = %v, want 0", follower.Spec.Replicas)
	}

	t.Log("the marker annotation is what makes the workload renderer pick RoleFollower over the leader's serve command")
	if follower.Annotations[commonconsts.KubeAnnotationElasticEPFollower] != commonconsts.KubeLabelValueTrue {
		t.Errorf("follower must carry the marker annotation; got %v", follower.Annotations)
	}

	t.Log("the leader itself is untouched by the derivation")
	if leader.Spec.Replicas != nil {
		t.Errorf("leader Replicas must not be mutated by follower synthesis; got %v", leader.Spec.Replicas)
	}
}

// Both the Grove role name (<leader>) and the non-Grove synthesized name (<leader>-flw)
// must resolve to the same leader headless Service <leader>-ray.
func TestInjectElasticEPRayLaunchFlags_Follower(t *testing.T) {
	tests := []struct {
		name        string
		serviceName string
	}{
		{name: "grove role name", serviceName: "my-worker"},
		{name: "non-grove synthesized name", serviceName: "my-worker-" + commonconsts.GroveRoleSuffixFollower},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{
				Command: []string{"python3"},
				Args:    []string{"-m", "dynamo.vllm", "--enable-elastic-ep", "--data-parallel-backend", "ray"},
			}

			t.Log("rewriting the follower's launch command")
			if !injectElasticEPRayLaunchFlags(container, RoleFollower, tt.serviceName, nil) {
				t.Fatal("expected the follower launch to be injected")
			}
			if len(container.Args) != 1 {
				t.Fatalf("expected 1 arg (the shell script), got %d", len(container.Args))
			}
			script := container.Args[0]

			t.Log("it joins the leader Service (<leader>-ray), never its own name, and pins its Ray node address to the pod IP")
			if !strings.Contains(script, "ray start --address=my-worker-ray:6379") {
				t.Errorf("follower must join the leader Service my-worker-ray; got: %s", script)
			}
			if !strings.Contains(script, `--node-ip-address="$POD_IP"`) {
				t.Errorf("follower must pin --node-ip-address to POD_IP; got: %s", script)
			}

			t.Log("it health-gates on the leader /live so the join lands after the leader has placed its data-parallel group")
			if !strings.Contains(script, "/live") {
				t.Errorf("follower must health-gate on the leader /live; got: %s", script)
			}

			t.Log("the follower never serves -- the leader spawns the real DP-rank worker on its GPU as a Ray actor -- so the serve flags must not survive")
			// "dynamo.vllm" appears in the /live wait log, so assert on a serve-only flag.
			if strings.Contains(script, "--enable-elastic-ep") {
				t.Errorf("follower must NOT run the vLLM serve command; got: %s", script)
			}
			if len(container.Command) != 2 || container.Command[0] != "/bin/sh" {
				t.Errorf("expected [/bin/sh -c] command, got %v", container.Command)
			}
		})
	}
}

func TestInjectElasticEPFollowerAntiAffinity(t *testing.T) {
	tests := []struct {
		name             string
		podSpec          *corev1.PodSpec
		wantNodeAffinity bool
	}{
		{name: "empty pod spec", podSpec: &corev1.PodSpec{}},
		{
			name:             "preserves user-supplied affinity",
			podSpec:          &corev1.PodSpec{Affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{}}},
			wantNodeAffinity: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("injecting the one-pod-per-node anti-affinity")
			injectElasticEPFollowerAntiAffinity(tt.podSpec, "decode", "ns-mydgd")

			t.Log("exactly one required term, keyed on hostname, so each data-parallel rank gets the cross-node NVLink the EP collective needs")
			if tt.podSpec.Affinity == nil || tt.podSpec.Affinity.PodAntiAffinity == nil {
				t.Fatal("expected pod anti-affinity to be set")
			}
			terms := tt.podSpec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			if len(terms) != 1 {
				t.Fatalf("expected 1 required anti-affinity term, got %d", len(terms))
			}
			if terms[0].TopologyKey != "kubernetes.io/hostname" {
				t.Errorf("topologyKey = %q, want kubernetes.io/hostname", terms[0].TopologyKey)
			}

			t.Log("the term selects this component's pods by their identity labels")
			sel := terms[0].LabelSelector.MatchLabels
			if sel[commonconsts.KubeLabelDynamoComponent] != "decode" ||
				sel[commonconsts.KubeLabelDynamoNamespace] != "ns-mydgd" {
				t.Errorf("selector = %v, want component=decode namespace=ns-mydgd", sel)
			}

			if tt.wantNodeAffinity {
				t.Log("the user's own affinity rules survive the injection")
				if tt.podSpec.Affinity.NodeAffinity == nil {
					t.Error("user-supplied NodeAffinity was dropped")
				}
			}
		})
	}
}
