package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestSGLangSidecarPublisherPlacement(t *testing.T) {
	const publisher = `{"publisher":"zmq","endpoint":"tcp://*:5557"}`
	tests := []struct {
		name          string
		nodes         int32
		args          []string
		workerRuntime bool
		wantError     string
	}{
		{name: "TP", nodes: 2, args: []string{"--tp-size", "2"}},
		{name: "legacy DP", nodes: 2, args: []string{"--tp-size", "2", "--dp-size", "2", "--enable-dp-attention"}, workerRuntime: true},
		{name: "canonical DP", nodes: 2, args: []string{"--tp-size", "2", "--attn-dp-size", "2"}, workerRuntime: true},
		{name: "canonical equals", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2"}, workerRuntime: true},
		{name: "canonical long aliases", nodes: 2, args: []string{"--tensor-parallel-size", "2", "--attention-data-parallel-size", "2"}, workerRuntime: true},
		{name: "canonical long equals", nodes: 2, args: []string{"--tensor-parallel-size=2", "--attention-data-parallel-size=2"}, workerRuntime: true},
		{name: "legacy long alias", nodes: 2, args: []string{"--tp-size", "2", "--data-parallel-size=2", "--enable-dp-attention"}, workerRuntime: true},
		{name: "legacy width one", nodes: 2, args: []string{"--tp-size", "2", "--dp-size", "1", "--enable-dp-attention"}},
		{name: "canonical overrides legacy", nodes: 2, args: []string{"--tp-size", "2", "--attn-dp-size", "2", "--enable-dp-attention"}, workerRuntime: true},
		{name: "legacy fills canonical one", nodes: 2, args: []string{"--tp-size", "2", "--attn-dp-size=1", "--dp-size=2", "--enable-dp-attention"}, workerRuntime: true},
		{name: "last alias wins", nodes: 2, args: []string{"--tp-size", "2", "--attn-dp-size=2", "--attention-data-parallel-size=1"}},
		{name: "multiple local publishers", nodes: 2, args: []string{"--tp-size=8", "--attn-dp-size=4"}, workerRuntime: true},
		{name: "attention CP", nodes: 2, args: []string{"--tp-size=8", "--attn-dp-size=2", "--attention-context-parallel-size=2"}, workerRuntime: true},
		{name: "PP follower has no publisher", nodes: 2, args: []string{"--tp-size=2", "--pipeline-parallel-size=2", "--attn-dp-size=2"}},
		{name: "PP stages within each Pod", nodes: 2, args: []string{"--tp-size=2", "--pp-size=4", "--attn-dp-size=2"}},
		{name: "no KV config", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", "--kv-events-config="}},
		{name: "disabled publisher", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", `--kv-events-config={"publisher":"null"}`}},
		{name: "legacy DP with KV disabled", nodes: 2, args: []string{"--tp-size=2", "--dp-size=2", "--enable-dp-attention", "--kv-events-config="}},
		{name: "default null publisher", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", "--kv-events-config={}"}},
		{name: "bound loopback source", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", `--kv-events-config={"publisher":"zmq","endpoint":"tcp://127.0.0.1:5557","bind":true}`}, workerRuntime: true},
		{name: "connect publisher has no local source", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", `--kv-events-config={"publisher":"zmq","endpoint":"tcp://collector:5557"}`}},
		{name: "explicit connect has no local source", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", `--kv-events-config={"publisher":"zmq","bind":false}`}},
		{name: "invalid publisher JSON", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", "--kv-events-config=$(KV_CONFIG)"}, wantError: "literal JSON"},
		{name: "unknown publisher", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=2", `--kv-events-config={"publisher":"custom"}`}, wantError: "cannot determine local KV sources"},
		{name: "mixed followers minimal", nodes: 4, args: []string{"--tp-size=4", "--attn-dp-size=2"}, wantError: "mixed publisher and non-publisher followers"},
		{name: "mixed followers review repro", nodes: 4, args: []string{"--tp-size=16", "--dp-size=2", "--enable-dp-attention"}, wantError: "mixed publisher and non-publisher followers"},
		{name: "mixed PP followers", nodes: 4, args: []string{"--tp-size=4", "--pp-size=2", "--attn-dp-size=2"}, wantError: "mixed publisher and non-publisher followers"},
		{name: "non-divisible attention width", nodes: 2, args: []string{"--tp-size=4", "--attn-dp-size=3"}, wantError: "must divide"},
		{name: "non-divisible node partition", nodes: 3, args: []string{"--tp-size=4", "--attn-dp-size=2"}, wantError: "must divide"},
		{name: "invalid zero", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=0"}, wantError: "positive integer"},
		{name: "invalid negative", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=-1"}, wantError: "positive integer"},
		{name: "missing size", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size"}, wantError: "requires a value"},
		{name: "expanded topology", nodes: 2, args: []string{"--tp-size=2", "--attn-dp-size=$(DP)"}, wantError: "positive integer"},
		{name: "model DP is not attention DP", nodes: 2, args: []string{"--tp-size=2", "--dp-size=2"}, wantError: "model data parallelism"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Author a shared template with explicit publisher configuration")
			args := append([]string{"--model-path", "model", "--kv-events-config", publisher}, tt.args...)
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "engine", ComponentType: v1beta1.ComponentTypeWorker,
				Multinode: &v1beta1.MultinodeSpec{NodeCount: tt.nodes},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: "main", Image: "engine:custom", Command: []string{"python3", "-m", "sglang.launch_server"}, Args: args}},
					InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:1.6.0", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)}},
				}},
			}
			original := component.DeepCopy()

			t.Log("Both renderers must use the same publisher placement")
			for _, provider := range []commonconsts.MultinodeDeploymentType{commonconsts.MultinodeDeploymentTypeGrove, commonconsts.MultinodeDeploymentTypeLWS} {
				for _, role := range []Role{RoleLeader, RoleWorker} {
					pod, err := GenerateBasePodSpec(component, BackendFrameworkSGLang, nil, "graph", "ns", role, tt.nodes, &configv1alpha1.OperatorConfiguration{}, provider, "engine", nil, staticContainerGPUCount(1))
					if tt.wantError != "" {
						require.ErrorContains(t, err, tt.wantError)
						continue
					}
					require.NoError(t, err)
					want := 0
					if role == RoleLeader || tt.workerRuntime {
						want = 1
					}
					require.Len(t, pod.InitContainers, want)
				}
			}
			require.Equal(t, original, component)
		})
	}
}
