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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

// vllmComponent builds a single-pod vLLM component whose main container carries the
// given extra args, so each test can describe the launch it wants at a high level.
// The leader every case in this file builds around. The follower derives its own
// identity and both placement terms from these, so they are shared by all tests.
const (
	leaderComponent       = "decode"
	leaderDynamoNamespace = "ns-mydgd"
)

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
			roles := expandRolesForComponent(leaderComponent, nil, 1, tt.component)

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
	dcd.Spec.ComponentName = leaderComponent
	dcd.Labels = map[string]string{commonconsts.KubeLabelDynamoComponent: leaderComponent}
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
		// The Service renderer emits the leader's headless Service only for the
		// single-pod shape. Synthesizing outside that gate leaves a follower waiting on
		// an address that is never created (replicas > 1) or that reconciles through the
		// LWS path, where the marker never routes to RoleFollower (multinode).
		{
			name:          "replicas > 1 gets none: each replica is its own Ray head",
			component:     withReplicas(elasticEPComponent(), 2),
			wantSynthesis: false,
		},
		{
			name:          "multinode gets none: it reaches its leader through the framework hostname",
			component:     withNodeCount(elasticEPComponent(), 2),
			wantSynthesis: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("deriving the follower from the leader DCD")
			follower := synthesizeElasticEPFollowerDCD(leaderDCD(tt.component), leaderComponent)

			t.Log("a follower is synthesized only for a shape whose leader Service is emitted")
			if gotSynthesis := follower != nil; gotSynthesis != tt.wantSynthesis {
				t.Fatalf("synthesized = %v, want %v (got %+v)", gotSynthesis, tt.wantSynthesis, follower)
			}
		})
	}
}

func withReplicas(c *v1beta1.DynamoComponentDeploymentSharedSpec, n int32) *v1beta1.DynamoComponentDeploymentSharedSpec {
	c.Replicas = ptr.To(n)
	return c
}

func withNodeCount(c *v1beta1.DynamoComponentDeploymentSharedSpec, n int32) *v1beta1.DynamoComponentDeploymentSharedSpec {
	c.Multinode = &v1beta1.MultinodeSpec{NodeCount: n}
	return c
}

// A user may legitimately declare a component whose name equals a derived follower
// identity. Storing the derived DCD unchecked would drop one of the two from
// rendering, worker hashing, and status depending on map order, so generation fails.
func TestGenerateDynamoComponentsDeployments_RejectsFollowerNameCollision(t *testing.T) {
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "mydgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: leaderComponent,
					ComponentType: commonconsts.ComponentTypeDecode,
					Replicas:      ptr.To(int32(1)),
					PodTemplate:   elasticEPComponent().PodTemplate,
				},
				{
					ComponentName: leaderComponent + "-" + commonconsts.GroveRoleSuffixFollower,
					ComponentType: commonconsts.ComponentTypeDecode,
					Replicas:      ptr.To(int32(1)),
					PodTemplate:   vllmComponent().PodTemplate,
				},
			},
		},
	}

	t.Log("generate with an elastic-EP leader and a declared component of the derived name")
	_, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})

	t.Log("generation fails loudly rather than silently dropping one of them")
	if err == nil {
		t.Fatal("expected a collision error, got nil")
	}
	if !strings.Contains(err.Error(), "collides") {
		t.Errorf("error should name the collision; got: %v", err)
	}
}

func TestSynthesizeElasticEPFollowerDCD_DerivesADistinctIdentity(t *testing.T) {
	leader := leaderDCD(elasticEPComponent())
	wantSuffixed := "decode-" + commonconsts.GroveRoleSuffixFollower

	t.Log("deriving the follower from a single-pod elastic-EP leader")
	follower := synthesizeElasticEPFollowerDCD(leader, leaderComponent)
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

	t.Log("it carries the load-bearing clique affinity pinning it into the LEADER's NVLink partition")
	if follower.Spec.PodTemplate == nil || follower.Spec.PodTemplate.Spec.Affinity == nil ||
		follower.Spec.PodTemplate.Spec.Affinity.PodAffinity == nil {
		t.Fatal("follower must carry a pod affinity")
	}
	clique := follower.Spec.PodTemplate.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if len(clique) != 1 || clique[0].TopologyKey != commonconsts.NodeLabelGPUClique {
		t.Fatalf("expected 1 required clique affinity on %q, got %+v", commonconsts.NodeLabelGPUClique, clique)
	}
	if got := clique[0].LabelSelector.MatchLabels[commonconsts.KubeLabelDynamoComponent]; got != leaderComponent {
		t.Errorf("clique affinity must select the LEADER (decode), got %q", got)
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

func TestInjectElasticEPFollowerAffinity(t *testing.T) {
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
			t.Log("injecting the follower placement terms (leader component = decode)")
			injectElasticEPFollowerAffinity(tt.podSpec, leaderComponent, leaderDynamoNamespace)
			if tt.podSpec.Affinity == nil {
				t.Fatal("expected affinity to be set")
			}

			t.Log("LOAD-BEARING: a required pod affinity on nvidia.com/gpu.clique selecting the leader, pinning the follower into the leader's NVLink partition")
			if tt.podSpec.Affinity.PodAffinity == nil {
				t.Fatal("expected pod affinity (clique) to be set")
			}
			aff := tt.podSpec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			if len(aff) != 1 {
				t.Fatalf("expected 1 required clique affinity term, got %d", len(aff))
			}
			if aff[0].TopologyKey != commonconsts.NodeLabelGPUClique {
				t.Errorf("clique topologyKey = %q, want %q", aff[0].TopologyKey, commonconsts.NodeLabelGPUClique)
			}
			if s := aff[0].LabelSelector.MatchLabels; s[commonconsts.KubeLabelDynamoComponent] != leaderComponent ||
				s[commonconsts.KubeLabelDynamoNamespace] != leaderDynamoNamespace {
				t.Errorf("clique selector = %v, want leader component=decode namespace=ns-mydgd", s)
			}

			t.Log("one-pod-per-node: a required pod anti-affinity on hostname selecting the leader")
			if tt.podSpec.Affinity.PodAntiAffinity == nil {
				t.Fatal("expected pod anti-affinity to be set")
			}
			anti := tt.podSpec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			if len(anti) != 1 {
				t.Fatalf("expected 1 required anti-affinity term, got %d", len(anti))
			}
			if anti[0].TopologyKey != "kubernetes.io/hostname" {
				t.Errorf("anti-affinity topologyKey = %q, want kubernetes.io/hostname", anti[0].TopologyKey)
			}
			if s := anti[0].LabelSelector.MatchLabels; s[commonconsts.KubeLabelDynamoComponent] != leaderComponent ||
				s[commonconsts.KubeLabelDynamoNamespace] != leaderDynamoNamespace {
				t.Errorf("anti-affinity selector = %v, want leader component=decode namespace=ns-mydgd", s)
			}

			if tt.wantNodeAffinity {
				t.Log("user-supplied affinity is merged, not overwritten")
				if tt.podSpec.Affinity.NodeAffinity == nil {
					t.Error("user-supplied NodeAffinity was dropped")
				}
			}
		})
	}
}
