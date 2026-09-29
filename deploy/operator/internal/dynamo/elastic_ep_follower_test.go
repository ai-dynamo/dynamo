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
	"regexp"
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
	leaderDCDName         = "mydgd-decode"
)

func vllmComponent(extraArgs ...string) *v1beta1.DynamoComponentDeploymentSharedSpec {
	return &v1beta1.DynamoComponentDeploymentSharedSpec{
		// Elastic EP is a worker topology, and synthesis now requires it: the leader is
		// the engine heading the Ray cluster, so a planner or frontend carrying the same
		// flags must not derive a follower.
		ComponentType: commonconsts.ComponentTypeWorker,
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

// The follower is NOT a Grove clique: Grove rejects minAvailable:0 (grove#676), and a
// follower must be able to rest at zero -- a single-rank leader seeds one there, and any
// follower can be scaled there -- which would gang-block the leader. expandRolesForComponent must
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
		// injectElasticEPRayLaunchFlags leaves a Command-less leader alone rather than
		// emit a shell command with no executable, so no Ray head is started. A follower
		// here would poll a /live endpoint that never comes up.
		{
			name:          "leader without an explicit Command gets none: it never starts a Ray head",
			component:     withoutCommand(elasticEPComponent()),
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

// The follower is a deep copy, so it inherits spec.experimental.checkpoint verbatim. An
// explicit checkpointRef there is resolved by the renderer no matter what the reconciler
// looks up, and would restore-shape the follower's main container -- so once scaled it
// would restore a leader engine image instead of running its bare Ray join.
func TestSynthesizeElasticEPFollowerDCD_StripsCheckpointConfig(t *testing.T) {
	leader := leaderDCD(elasticEPComponent())
	checkpointRef := "leader-checkpoint"
	leader.Spec.Experimental = &v1beta1.ExperimentalSpec{
		Checkpoint: &v1beta1.ComponentCheckpointConfig{
			Enabled:       true,
			CheckpointRef: &checkpointRef,
		},
	}

	t.Log("derive the follower from a leader with an explicit checkpoint reference")
	follower := synthesizeElasticEPFollowerDCD(leader, leaderComponent)
	if follower == nil {
		t.Fatal("expected a follower to be synthesized")
	}

	t.Log("the inherited checkpoint config is dropped, so nothing restore-shapes the follower")
	if follower.Spec.Experimental != nil && follower.Spec.Experimental.Checkpoint != nil {
		t.Errorf("follower kept checkpoint config %+v; it would restore a leader engine instead of running its Ray join",
			follower.Spec.Experimental.Checkpoint)
	}

	t.Log("the leader's own checkpoint config is untouched by the derivation")
	if leader.Spec.Experimental.Checkpoint == nil || !leader.Spec.Experimental.Checkpoint.Enabled {
		t.Error("leader lost its checkpoint config during follower synthesis")
	}
}

// Admission accepts the elastic-EP launch flags on any component, so without a
// component-type gate a global-vLLM graph could put them on a planner or frontend and
// have a follower derived for it.
func TestIsSinglePodElasticEPShape_RequiresAWorkerComponent(t *testing.T) {
	tests := []struct {
		name          string
		componentType v1beta1.ComponentType
		want          bool
	}{
		{name: "worker qualifies", componentType: commonconsts.ComponentTypeWorker, want: true},
		{name: "decode worker qualifies", componentType: commonconsts.ComponentTypeDecode, want: true},
		{name: "prefill worker qualifies", componentType: commonconsts.ComponentTypePrefill, want: true},
		{name: "frontend does not", componentType: commonconsts.ComponentTypeFrontend, want: false},
		{name: "planner does not", componentType: commonconsts.ComponentTypePlanner, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			component := elasticEPComponent()
			component.ComponentType = tt.componentType

			t.Log("elastic EP is a worker topology: the leader is the engine heading the Ray cluster")
			if got := IsSinglePodElasticEPShape(component); got != tt.want {
				t.Errorf("IsSinglePodElasticEPShape(%s) = %v, want %v", tt.componentType, got, tt.want)
			}
		})
	}
}

// Infrastructure keyed by component name -- GMS DRA claim templates, checkpoint info --
// is only created for components declared in the DGD. The follower is derived, not
// declared, so resolving under its own "<leader>-flw" name finds nothing: the claim
// template never exists and the pod cannot schedule, and the checkpoint lookup silently
// returns nil. It must resolve under the leader's identity instead.
func TestElasticEPComponentIdentity(t *testing.T) {
	tests := []struct {
		name          string
		component     *v1beta1.DynamoComponentDeploymentSharedSpec
		componentName string
		want          string
	}{
		{
			name:          "a declared component resolves under its own name",
			component:     elasticEPComponent(),
			componentName: leaderComponent,
			want:          leaderComponent,
		},
		{
			name: "a synthesized follower resolves under the leader's name",
			component: func() *v1beta1.DynamoComponentDeploymentSharedSpec {
				follower := synthesizeElasticEPFollowerDCD(leaderDCD(elasticEPComponent()), leaderComponent)
				if follower == nil {
					t.Fatal("expected a follower to be synthesized")
				}
				return &follower.Spec.DynamoComponentDeploymentSharedSpec
			}(),
			componentName: leaderComponent + "-" + commonconsts.GroveRoleSuffixFollower,
			want:          leaderComponent,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("resolve the identity that component-keyed infrastructure is looked up under")
			got := ElasticEPComponentIdentity(tt.component, tt.componentName)

			t.Log("only a name a declared component owns can resolve to real infrastructure")
			if got != tt.want {
				t.Errorf("ElasticEPComponentIdentity = %q, want %q", got, tt.want)
			}
		})
	}
}

// A 60-63 character leader name is itself valid, but appending the follower suffix pushes
// the result past the 63-character DNS-1123 limit, and the API server then rejects the
// generated DCD and stalls the whole reconcile.
func TestElasticEPFollowerName_BoundedToKubeLimit(t *testing.T) {
	tests := []struct {
		name       string
		leaderName string
	}{
		{name: "short name keeps the plain suffix", leaderName: "decode"},
		{name: "exactly at the limit", leaderName: strings.Repeat("a", maxKubeNameLength)},
		{name: "one under the limit", leaderName: strings.Repeat("b", maxKubeNameLength-1)},
		{name: "well over the limit", leaderName: strings.Repeat("c", maxKubeNameLength+40)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("derive the follower identity from the leader's")
			got := elasticEPFollowerName(tt.leaderName)

			t.Log("it always fits the Kubernetes name limit and stays a valid DNS-1123 label")
			if len(got) > maxKubeNameLength {
				t.Errorf("name %q is %d chars, over the %d limit", got, len(got), maxKubeNameLength)
			}
			if !regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`).MatchString(got) {
				t.Errorf("name %q is not a valid DNS-1123 label", got)
			}

			t.Log("it stays recognizable as a follower and is stable across calls")
			if !strings.HasSuffix(got, "-"+commonconsts.GroveRoleSuffixFollower) {
				t.Errorf("name %q lost the follower suffix", got)
			}
			if again := elasticEPFollowerName(tt.leaderName); again != got {
				t.Errorf("not deterministic: %q then %q", got, again)
			}
		})
	}

	t.Log("distinct over-long leaders still derive distinct followers")
	a := elasticEPFollowerName(strings.Repeat("d", 70) + "-one")
	b := elasticEPFollowerName(strings.Repeat("d", 70) + "-two")
	if a == b {
		t.Errorf("truncation collapsed two distinct leaders onto %q", a)
	}
}

func withReplicas(c *v1beta1.DynamoComponentDeploymentSharedSpec, n int32) *v1beta1.DynamoComponentDeploymentSharedSpec {
	c.Replicas = ptr.To(n)
	return c
}

// withoutCommand models a component whose real entrypoint is the image ENTRYPOINT.
func withoutCommand(c *v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoComponentDeploymentSharedSpec {
	c.PodTemplate.Spec.Containers[0].Command = nil
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

	// This leader declares no --data-parallel-size, so its declared width is one rank and
	// the follower is seeded at zero. A leader that declares dp=N seeds it at N-1 instead --
	// see TestSynthesizeElasticEPFollowerDCD_SeedsDeclaredWidth.
	t.Log("with no declared width the follower is seeded at zero and scaled on demand")
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

// The follower joins exactly the address it is handed. It must never derive the leader
// Service name from its own identity: that name is DGD- and generation-scoped and may be
// hash-truncated, so any recomputation here is a chance for the joiner to drift from the
// Service the reconciler actually emitted.
func TestInjectElasticEPRayLaunchFlags_Follower(t *testing.T) {
	tests := []struct {
		name          string
		serviceName   string
		leaderService string
	}{
		{
			name:          "uses the carried address",
			serviceName:   "my-worker-" + commonconsts.GroveRoleSuffixFollower,
			leaderService: "my-worker-ray",
		},
		{
			// A truncated follower name shares no prefix with the leader Service, so
			// this case fails outright if the address is ever rebuilt from serviceName.
			name:          "address is unrelated to the follower's own name",
			serviceName:   "trunc-a1b2-" + commonconsts.GroveRoleSuffixFollower,
			leaderService: "mydgd-my-worker-9f2c-ray",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{
				Command: []string{"python3"},
				Args:    []string{"-m", "dynamo.vllm", "--enable-elastic-ep", "--data-parallel-backend", "ray"},
			}

			t.Log("rewriting the follower's launch command")
			if !injectElasticEPRayLaunchFlags(container, RoleFollower, tt.serviceName, nil, tt.leaderService, 0) {
				t.Fatal("expected the follower launch to be injected")
			}
			if len(container.Args) != 1 {
				t.Fatalf("expected 1 arg (the shell script), got %d", len(container.Args))
			}
			script := container.Args[0]

			t.Log("it joins the carried leader address verbatim and pins its Ray node address to the pod IP")
			if !strings.Contains(script, "ray start --address="+tt.leaderService+":6379") {
				t.Errorf("follower must join %s; got: %s", tt.leaderService, script)
			}
			if !strings.Contains(script, `--node-ip-address="$POD_IP"`) {
				t.Errorf("follower must pin --node-ip-address to POD_IP; got: %s", script)
			}

			t.Log("nothing in the command is derived from the follower's own component name")
			if strings.Contains(script, tt.serviceName) {
				t.Errorf("follower must not build its address from its own name %q; got: %s", tt.serviceName, script)
			}

			t.Log("it gates on the leader's Ray head, not the leader's engine, so a full-width launch can converge")
			if !strings.Contains(script, "create_connection((") || !strings.Contains(script, VLLMPort) {
				t.Errorf("follower must gate on a TCP connect to the leader Ray port; got: %s", script)
			}
			// Gating on /live is what the multinode RoleWorker arm does, and it deadlocks a
			// launch sized to --data-parallel-size: the leader cannot reach /live until the
			// ranks it is waiting for have joined, and they will not join until /live answers.
			if strings.Contains(script, "/live") {
				t.Errorf("follower must not gate on the leader engine /live; got: %s", script)
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
			injectElasticEPFollowerAffinity(tt.podSpec, leaderComponent, leaderDynamoNamespace, leaderDCDName)
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

			// Without this the term also matches the OLD leader mid-rollout, so the
			// follower could be pinned into the old leader's partition while joining the
			// new leader's Service -- the Ray Service is narrowed the same way.
			t.Log("both terms are narrowed to one leader generation, as the Ray Service is")
			for _, term := range []corev1.PodAffinityTerm{aff[0], tt.podSpec.Affinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution[0]} {
				if got := term.LabelSelector.MatchLabels[commonconsts.KubeLabelDynamoSelector]; got != leaderDCDName {
					t.Errorf("term on %q has selector %q, want the leader DCD name %q", term.TopologyKey, got, leaderDCDName)
				}
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

// TestSynthesizeElasticEPFollowerDCD_DoesNotTouchASingleRankLeader guards the upgrade
// path for every deployment that derives no followers.
//
// The follower-count annotation lives on the leader's POD TEMPLATE, so writing it changes
// the pod hash and rolls the deployment. A leader with no followers behaves identically
// whether the annotation is absent or "0" -- elasticEPSynthesizedFollowers maps both to
// zero -- so stamping "0" would restart every existing single-rank elastic-EP deployment
// on operator upgrade and buy nothing.
//
// Seen for real on dynamo-aws-gb300: the upgrade rolled two serving deployments into new
// generations that could not schedule, and neither served again until capacity was freed.
//
// Mutation check: stamping unconditionally fails this.
func TestSynthesizeElasticEPFollowerDCD_DoesNotTouchASingleRankLeader(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
	}{
		{name: "no --data-parallel-size", args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray"}},
		{name: "dp=1", args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "1"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			leader := leaderDCD(vllmComponent(tt.args...))
			before := GetPodTemplateAnnotations(&leader.Spec.DynamoComponentDeploymentSharedSpec)
			beforeCount := len(before)

			synthesizeElasticEPFollowerDCD(leader, leaderComponent)

			after := GetPodTemplateAnnotations(&leader.Spec.DynamoComponentDeploymentSharedSpec)
			if _, stamped := after[commonconsts.KubeAnnotationElasticEPFollowerReplicas]; stamped {
				t.Errorf("a leader with no followers must not gain the follower-count annotation; "+
					"it changes the pod hash and rolls a serving deployment for no behaviour change (got %v)", after)
			}
			if len(after) != beforeCount {
				t.Errorf("the leader's pod-template annotations changed (%d -> %d): %v", beforeCount, len(after), after)
			}
		})
	}

	// And the positive case, so this cannot pass by never stamping at all.
	t.Run("dp=4 does stamp, because the behaviour genuinely changes", func(t *testing.T) {
		leader := leaderDCD(vllmComponent("--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "4"))
		synthesizeElasticEPFollowerDCD(leader, leaderComponent)
		got := GetPodTemplateAnnotations(&leader.Spec.DynamoComponentDeploymentSharedSpec)[commonconsts.KubeAnnotationElasticEPFollowerReplicas]
		if got != "3" {
			t.Errorf("follower-count annotation = %q, want \"3\"", got)
		}
	})
}

// TestElasticEPLeaderDoesNotWaitWithoutSynthesizedFollowers is the regression guard for a
// bug this PR introduced and a risk review caught before it shipped.
//
// The width wait and the --data-parallel-size-local pin were originally keyed on
// --data-parallel-size. That flag says how many data-parallel RANKS the engine wants; it
// says nothing about how many PODS they run in. A single pod with several GPUs runs them
// intra-pod, and two shipped shapes do exactly that:
//
//   - the Grove pathway, which renders this same RoleMain arm but deliberately never
//     synthesizes a follower (grove#676); Grove is the DEFAULT provider on any cluster
//     where the Grove API is installed and the DGD does not opt out
//   - any component with replicas > 1, where IsSinglePodElasticEPShape declines and no
//     follower is derived either
//
// The first shape is what any dp > 1 component becomes when it does not set
// nvidia.com/enable-grove "false": Grove is its provider, so it renders one pod and no
// follower is ever synthesized. Keyed on the flag, such a manifest waited 20 minutes for a
// second Ray node nothing would ever create and then exited 1 -- a working deployment
// turned into a CrashLoopBackOff. (The shipped demo fixture hit this before it was given
// an explicit Grove opt-out.)
//
// So the trigger is the follower count synthesis actually stamped, and its absence must
// render the leader exactly as it rendered before any of this.
//
// Mutation check: re-keying either on getFlagValue(..., dataParallelSizeFlag) fails every
// subtest here.
func TestElasticEPLeaderDoesNotWaitWithoutSynthesizedFollowers(t *testing.T) {
	// Renders WITHOUT going through synthesis, which is what Grove and replicas > 1 do.
	renderUnsynthesized := func(t *testing.T, component *v1beta1.DynamoComponentDeploymentSharedSpec) string {
		t.Helper()
		container := GetMainContainer(component).DeepCopy()
		if err := (&VLLMBackend{}).UpdateContainer(
			container, 1, RoleMain, component, "test-service",
			&GroveMultinodeDeployer{}, staticContainerGPUCount(4),
		); err != nil {
			t.Fatalf("UpdateContainer: %v", err)
		}
		return strings.Join(container.Args, " ")
	}

	elasticDP2 := func() *v1beta1.DynamoComponentDeploymentSharedSpec {
		return vllmComponent("--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "2")
	}

	for _, tt := range []struct {
		name      string
		component *v1beta1.DynamoComponentDeploymentSharedSpec
	}{
		{
			// The shipped demo manifest's shape: Grove renders it, Grove creates no
			// follower, so there is never a second Ray node.
			name:      "Grove leader: no follower is ever synthesized",
			component: elasticDP2(),
		},
		{
			// IsSinglePodElasticEPShape declines at replicas > 1, so no follower is
			// derived, yet every replica still renders as RoleMain.
			name:      "replicas > 1: no follower is derived either",
			component: withReplicas(elasticDP2(), 2),
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			script := renderUnsynthesized(t, tt.component)

			if strings.Contains(script, "ray.nodes()") {
				t.Errorf("leader must NOT wait for Ray nodes that nothing will create; got: %s", script)
			}
			if strings.Contains(script, dataParallelSizeLocalFlag) {
				t.Errorf("leader must NOT be pinned to one local rank when its ranks run intra-pod; got: %s", script)
			}
			// The Ray head itself still renders -- that shipped in #12943 and is not
			// conditional on any of this.
			if !strings.Contains(script, "ray start --head") {
				t.Errorf("the Ray head must still be injected; got: %s", script)
			}
		})
	}
}

// TestElasticEPLeaderGetsDataParallelSizeLocal pins the second half of the sizing rule.
//
// ElasticEPFollowerReplicas derives the follower count from --data-parallel-size on the
// rule "one pod is one node is one rank". That same rule fixes the leader at exactly one
// local rank, and the operator now says so rather than leaving it to the user. When the
// two halves disagreed, vLLM put every rank on the DP master and aborted with
//
//	ValueError: Not enough resources to allocate 4 DP ranks on DP master node <ip>,
//	            possible to fit 1 DP ranks.
//
// while three follower pods sat idle in the Ray cluster it had ignored. Caught on
// dynamo-aws-gb300; the error never names the missing flag.
//
// Mutation check: deleting the injectElasticEPDataParallelSizeLocal call fails the dp=4
// subtest and nothing else.
func TestElasticEPLeaderGetsDataParallelSizeLocal(t *testing.T) {
	// followers is what synthesis stamped on the leader, and it -- not
	// --data-parallel-size -- is what licenses the pin. Rendering through
	// synthesizeElasticEPFollowerDCD rather than hand-setting the annotation keeps the
	// two halves honest: if synthesis stops stamping it, these tests go red.
	render := func(t *testing.T, extraArgs ...string) string {
		t.Helper()
		component := vllmComponent(append([]string{"--enable-elastic-ep", "--data-parallel-backend", "ray"}, extraArgs...)...)
		leader := leaderDCD(component)
		synthesizeElasticEPFollowerDCD(leader, leaderComponent)
		leaderSpec := &leader.Spec.DynamoComponentDeploymentSharedSpec
		container := GetMainContainer(leaderSpec).DeepCopy()
		if err := (&VLLMBackend{}).UpdateContainer(
			container, 1, RoleMain, leaderSpec, "test-service",
			&GroveMultinodeDeployer{}, staticContainerGPUCount(1),
		); err != nil {
			t.Fatalf("UpdateContainer: %v", err)
		}
		return strings.Join(container.Args, " ")
	}

	t.Run("a multi-rank leader is pinned to one local rank", func(t *testing.T) {
		script := render(t, "--data-parallel-size", "4")
		if !strings.Contains(script, "--data-parallel-size-local 1") {
			t.Errorf("leader must be pinned to one local rank, or vLLM packs all 4 onto the DP master; got: %s", script)
		}
	})

	// An explicit choice wins. The operator supplies a default; it does not overrule a
	// user who has deliberately asked for a different split.
	t.Run("an explicit local size is left alone", func(t *testing.T) {
		script := render(t, "--data-parallel-size", "4", "--data-parallel-size-local", "2")
		if !strings.Contains(script, "--data-parallel-size-local 2") {
			t.Errorf("the user's explicit local size must survive; got: %s", script)
		}
		if strings.Contains(script, "--data-parallel-size-local 1") {
			t.Errorf("the operator must not append a second local size; got: %s", script)
		}
	})

	// No-regression: a single-rank leader has no followers, so the local split is vLLM's
	// business and the rendered command must not change.
	for _, tt := range []struct {
		name string
		args []string
	}{
		{name: "no --data-parallel-size", args: nil},
		{name: "dp=1", args: []string{"--data-parallel-size", "1"}},
	} {
		t.Run(tt.name+" gets no local size", func(t *testing.T) {
			if script := render(t, tt.args...); strings.Contains(script, dataParallelSizeLocalFlag) {
				t.Errorf("a single-rank leader must not gain a local size; got: %s", script)
			}
		})
	}
}

// TestElasticEPLeaderWaitsForDeclaredWidth pins the other half of a full-width launch.
//
// Seeding N-1 followers is not enough on its own. vLLM's create_dp_placement_groups reads
// the Ray cluster ONCE at engine start and allocates one placement group per rank against
// whatever it finds; nothing retries. So if the leader starts the engine as soon as its
// own Ray head is up, the launch becomes a race against N-1 pods being scheduled, pulled
// and joined, and it loses intermittently:
//
//	ValueError: Not enough resources to allocate 4 placement groups,
//	            only created 2 placement groups
//
// Observed on dynamo-aws-gb300: the same manifest reached 4 Ray nodes on one attempt and
// 2 on the next. The gate is what makes it deterministic.
//
// Mutation check: deleting the widthGate branch fails the dp=4 subtest and nothing else.
func TestElasticEPLeaderWaitsForDeclaredWidth(t *testing.T) {
	// Rendered through synthesis, because the wait is licensed by the follower count
	// synthesis stamps on the leader -- not by --data-parallel-size. A leader whose
	// followers were never created (Grove, or replicas > 1) must get no wait at all;
	// TestElasticEPLeaderDoesNotWaitWithoutSynthesizedFollowers covers that directly.
	render := func(t *testing.T, extraArgs ...string) string {
		t.Helper()
		component := vllmComponent(append([]string{"--enable-elastic-ep", "--data-parallel-backend", "ray"}, extraArgs...)...)
		leader := leaderDCD(component)
		synthesizeElasticEPFollowerDCD(leader, leaderComponent)
		leaderSpec := &leader.Spec.DynamoComponentDeploymentSharedSpec
		container := GetMainContainer(leaderSpec).DeepCopy()
		if err := (&VLLMBackend{}).UpdateContainer(
			container, 1, RoleMain, leaderSpec, "test-service",
			&GroveMultinodeDeployer{}, staticContainerGPUCount(1),
		); err != nil {
			t.Fatalf("UpdateContainer: %v", err)
		}
		return strings.Join(container.Args, " ")
	}

	t.Run("dp=4 waits for four Ray nodes before starting the engine", func(t *testing.T) {
		script := render(t, "--data-parallel-size", "4")
		if !strings.Contains(script, "-ge 4 ]") {
			t.Errorf("leader must wait for its declared width before exec'ing vLLM; got: %s", script)
		}
		if !strings.Contains(script, "ray.nodes()") {
			t.Errorf("the wait must count live Ray nodes, matching the one-pod-per-rank sizing rule; got: %s", script)
		}
		// The wait has to sit between the head coming up and the engine starting. If it
		// landed after the engine it would be useless, and before the head it could never
		// pass -- ray.init needs the head.
		headIdx := strings.Index(script, "create_connection")
		waitIdx := strings.Index(script, "ray.nodes()")
		engineIdx := strings.Index(script, "dynamo.vllm")
		if !(headIdx < waitIdx && waitIdx < engineIdx) {
			t.Errorf("the width wait must come after the Ray head is up and before the engine starts; got: %s", script)
		}
	})

	// The no-regression case. Everything that predates this change declares a single rank
	// or no width at all, so it must render byte-for-byte as before -- a wait for one node
	// would be satisfied by the leader itself, but emitting it at all is a pod-template
	// change, and a pod-template change rolls a serving deployment.
	for _, tt := range []struct {
		name string
		args []string
	}{
		{name: "no --data-parallel-size emits no wait", args: nil},
		{name: "dp=1 emits no wait", args: []string{"--data-parallel-size", "1"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if script := render(t, tt.args...); strings.Contains(script, "ray.nodes()") {
				t.Errorf("a single-rank leader must not gain a width wait; got: %s", script)
			}
		})
	}
}

// TestSynthesizeElasticEPFollowerDCD_SeedsDeclaredWidth pins that synthesis actually
// *uses* the sizing rule, not merely that the rule computes correctly in isolation.
//
// Testing ElasticEPFollowerReplicas alone does not cover this: reverting
// synthesizeElasticEPFollowerDCD to a hardcoded zero leaves that test passing, because
// the helper is still correct -- it has simply stopped being called. Caught by mutation.
func TestSynthesizeElasticEPFollowerDCD_SeedsDeclaredWidth(t *testing.T) {
	leader := leaderDCD(vllmComponent(
		"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "4",
	))

	follower := synthesizeElasticEPFollowerDCD(leader, leaderComponent)
	if follower == nil {
		t.Fatal("no follower derived for a declared elastic-EP leader")
	}

	t.Log("EP16 at TP4 is four ranks: the leader holds rank 0, so three followers")
	if got := ptr.Deref(follower.Spec.Replicas, -1); got != 3 {
		t.Errorf("follower replicas = %d, want 3 for --data-parallel-size 4; "+
			"a hardcoded seed makes the declared width unreachable", got)
	}
}

// TestElasticEPFollowerReplicasTracksDeclaredWidth pins the sizing rule: one pod per node
// per data-parallel rank, with the leader holding rank 0.
//
// This is the launch footprint, and it is what makes an EP16 deployment four pods rather
// than one. Before this the follower was seeded at zero unconditionally, so a leader
// declaring --data-parallel-size 4 rendered a single pod and could never reach the width
// it asked for.
func TestElasticEPFollowerReplicasTracksDeclaredWidth(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		want int32
	}{
		{
			// The pre-existing shape: no declared width, so nothing is derived and every
			// deployment that predates this change renders exactly as it did.
			name: "no --data-parallel-size yields no followers",
			args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray"},
			want: 0,
		},
		{
			name: "dp=1 is the leader alone",
			args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "1"},
			want: 0,
		},
		{
			name: "dp=2 is the leader plus one follower",
			args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "2"},
			want: 1,
		},
		{
			// EP16 at TP4: four ranks, four pods, one leader and three followers.
			name: "dp=4 is the leader plus three followers",
			args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "4"},
			want: 3,
		},
		{
			name: "an unparseable value falls back to the leader alone",
			args: []string{"--enable-elastic-ep", "--data-parallel-backend", "ray", "--data-parallel-size", "four"},
			want: 0,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := ElasticEPFollowerReplicas(&corev1.Container{
				Name:    commonconsts.MainContainerName,
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Args:    tt.args,
			})
			if got != tt.want {
				t.Errorf("ElasticEPFollowerReplicas() = %d, want %d (args: %v)", got, tt.want, tt.args)
			}
		})
	}
}

// TestElasticEPSizingReadsTheSameCommandLineAsDetection pins the two halves of this feature
// to one answer about where a flag may live.
//
// IsElasticEPRayLaunch scans Command AND Args, so it enables the whole path for a manifest
// that writes the vLLM invocation in `command:`. Sizing used to scan Args only, so for that
// same manifest it saw no --data-parallel-size and defaulted to one rank. The result was a
// declared width of N rendering as ONE pod: no follower, no width gate, no local-rank pin --
// and vLLM then aborting because it placed every rank on the DP master.
//
// That manifest shape is not exotic: validateElasticEPRequiresCommand REJECTS an empty
// command, so users are actively pushed toward writing the invocation there.
//
// Mutation check: reverting either read to getExpandedArgs fails the matching subtest here
// and nothing else.
func TestElasticEPSizingReadsTheSameCommandLineAsDetection(t *testing.T) {
	// The whole invocation in Command, Args empty -- what a user writes when they follow the
	// "elastic EP requires an explicit container command" admission error literally.
	inCommand := &corev1.Container{
		Name: commonconsts.MainContainerName,
		Command: []string{
			"python3", "-m", "dynamo.vllm",
			"--model", "deepseek-ai/DeepSeek-V2-Lite",
			"--enable-elastic-ep",
			"--data-parallel-backend", "ray",
			"--data-parallel-size", "4",
		},
	}

	t.Run("detection and sizing agree for flags in Command", func(t *testing.T) {
		if !IsElasticEPRayLaunch(inCommand) {
			t.Fatal("IsElasticEPRayLaunch did not detect flags in Command; " +
				"this test's premise is that it does")
		}
		if got := ElasticEPFollowerReplicas(inCommand); got != 3 {
			t.Errorf("ElasticEPFollowerReplicas() = %d, want 3: detection enabled the path "+
				"from Command, so sizing must read Command too -- otherwise a declared "+
				"--data-parallel-size 4 renders as one pod and the engine aborts", got)
		}
	})

	t.Run("the local-rank pin is injected for flags in Command", func(t *testing.T) {
		container := inCommand.DeepCopy()
		injectElasticEPDataParallelSizeLocal(container)
		joined := strings.Join(append(append([]string{}, container.Command...), container.Args...), " ")
		if !strings.Contains(joined, dataParallelSizeLocalFlag+" 1") {
			t.Errorf("--data-parallel-size-local 1 was not injected for a Command-only "+
				"invocation; got %q. Followers without the pin are unreachable: vLLM puts "+
				"every rank on the DP master and aborts", joined)
		}
	})

	t.Run("the standard shell-wrapped shape is unchanged", func(t *testing.T) {
		// Every shipped elastic-EP manifest uses this form, so it is the one that must not
		// move: command is the shell, args carry the invocation.
		shellWrapped := &corev1.Container{
			Name:    commonconsts.MainContainerName,
			Command: []string{"/bin/sh", "-c"},
			Args: []string{
				"python3 -m dynamo.vllm --enable-elastic-ep --data-parallel-backend ray " +
					"--data-parallel-size 4",
			},
		}
		if got := ElasticEPFollowerReplicas(shellWrapped); got != 3 {
			t.Errorf("ElasticEPFollowerReplicas() = %d, want 3 for the shipped manifest shape", got)
		}
	})
}

// TestSynthesizeElasticEPFollowerDCD_AffinityUsesPodStampedNamespace pins the source of
// the dynamo-namespace value in the follower's placement terms.
//
// Two accessors disagree. GetDCDDynamoNamespace prefers the shared spec's dynamoNamespace
// and only falls back to the DCD's label; the renderer stamps pods with the former. A
// graph that sets the deprecated v1alpha1 dynamoNamespace therefore gets pods labelled
// "ep-gate" while its DCD label reads "<k8s-namespace>-ep-gate".
//
// Selecting on the label produced a term matching no pod. That is worse than it sounds:
// the affinity is not merely wrong, it is unsatisfiable, so the capability check reads it
// as "the leader is not in an NVLink partition" and drops it. The follower then schedules
// anywhere, silently losing the NVLink guarantee the term exists to provide. Caught on
// dynamo-aws-gb300; fixtures where both values agree cannot see it.
func TestSynthesizeElasticEPFollowerDCD_AffinityUsesPodStampedNamespace(t *testing.T) {
	const (
		podStamped = "ep-gate"
		dcdLabel   = "tzulingk-ft-tests-ep-gate"
	)

	leader := leaderDCD(elasticEPComponent())
	leader.Labels[commonconsts.KubeLabelDynamoNamespace] = dcdLabel
	// GetDCDDynamoNamespace resolves through the v1alpha1 sparse-save annotation, which is
	// where a graph using the deprecated dynamoNamespace field keeps it, and which is what
	// the renderer stamps on the pods. Setting it apart from the label reproduces the
	// divergence seen on the cluster.
	if leader.Annotations == nil {
		leader.Annotations = map[string]string{}
	}
	leader.Annotations["nvidia.com/dcd-spec"] = `{"dynamoNamespace":"` + podStamped + `"}`

	if got := GetDCDDynamoNamespace(leader); got != podStamped {
		t.Fatalf("fixture does not diverge: GetDCDDynamoNamespace = %q, want %q", got, podStamped)
	}

	t.Log("Synthesize the follower from a leader whose two namespace values disagree")
	follower := synthesizeElasticEPFollowerDCD(leader, leaderComponent)
	if follower == nil {
		t.Fatal("no follower derived")
	}

	terms := follower.Spec.PodTemplate.Spec.Affinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	if len(terms) == 0 {
		t.Fatal("no required pod-affinity term was injected")
	}

	t.Log("The term selects the namespace the pods carry, not the DCD's label")
	got := terms[0].LabelSelector.MatchLabels[commonconsts.KubeLabelDynamoNamespace]
	if got != podStamped {
		t.Errorf("affinity selects dynamo-namespace %q, want %q (the value stamped on pods); "+
			"selecting %q matches no pod, so the term is unsatisfiable and gets dropped",
			got, podStamped, dcdLabel)
	}
}
