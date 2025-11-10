package dynamo

import (
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// Mock MultinodeDeployer for testing with no shell interpretation needed
type MockSimpleDeployer struct{}

func (m *MockSimpleDeployer) GetLeaderHostname(serviceName string) string {
	return "leader.example.com"
}

func (m *MockSimpleDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, numberOfNodes)
	hostnames[0] = m.GetLeaderHostname(serviceName)
	for i := int32(1); i < numberOfNodes; i++ {
		hostnames[i] = "worker" + string('0'+i) + ".example.com"
	}
	return hostnames
}

func (m *MockSimpleDeployer) GetNodeRank() (string, bool) {
	return "1", false // simple rank, no shell interpretation needed
}

// Mock MultinodeDeployer for testing with shell interpretation needed
type MockShellDeployer struct{}

func (m *MockShellDeployer) GetLeaderHostname(serviceName string) string {
	return "$(LEADER_HOST)"
}

func (m *MockShellDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, numberOfNodes)
	hostnames[0] = m.GetLeaderHostname(serviceName)
	for i := int32(1); i < numberOfNodes; i++ {
		hostnames[i] = "$(WORKER_" + string('0'+i) + "_HOST)"
	}
	return hostnames
}

func (m *MockShellDeployer) GetNodeRank() (string, bool) {
	return "$(WORKER_INDEX)", true // needs shell interpretation
}

func TestSGLangBackend_PythonCommandInjection(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name              string
		numberOfNodes     int32
		role              Role
		multinodeDeployer MultinodeDeployer
		initialCommand    []string
		initialArgs       []string
		expectedCommand   []string
		expectedArgs      []string
		description       string
	}{
		{
			name:              "single node python command no changes",
			numberOfNodes:     1,
			role:              RoleMain,
			multinodeDeployer: &MockSimpleDeployer{},
			initialCommand:    []string{"python3"},
			initialArgs:       []string{"-m", "dynamo.sglang"},
			expectedCommand:   []string{"python3"},
			expectedArgs:      []string{"-m", "dynamo.sglang"},
			description:       "Single node should not modify python commands",
		},
		{
			name:              "python command simple deployer - direct append",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockSimpleDeployer{},
			initialCommand:    []string{"python3"},
			initialArgs:       []string{"-m", "dynamo.sglang", "--model", "llama"},
			expectedCommand:   []string{"python3"},
			expectedArgs:      []string{"-m", "dynamo.sglang", "--model", "llama", "--dist-init-addr", "leader.example.com:29500", "--nnodes", "2", "--node-rank", "1"},
			description:       "Direct python command with simple deployer should append flags",
		},
		{
			name:              "python command shell deployer - shell wrapping",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"python3"},
			initialArgs:       []string{"-m", "dynamo.sglang", "--model", "llama"},
			expectedCommand:   []string{"sh", "-c"},
			expectedArgs:      []string{"exec python3 -m dynamo.sglang --model llama --dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)"},
			description:       "Direct python command with shell deployer should wrap with sh -c exec",
		},
		{
			name:              "python command leader role - always simple",
			numberOfNodes:     3,
			role:              RoleLeader,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"python"},
			initialArgs:       []string{"-m", "dynamo.sglang"},
			expectedCommand:   []string{"python"},
			expectedArgs:      []string{"-m", "dynamo.sglang", "--dist-init-addr", "$(LEADER_HOST):29500", "--nnodes", "3", "--node-rank", "0"},
			description:       "Leader role should never use shell wrapping",
		},
		{
			name:              "python3.11 variant supported",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockSimpleDeployer{},
			initialCommand:    []string{"python3.11"},
			initialArgs:       []string{"-m", "dynamo.sglang"},
			expectedCommand:   []string{"python3.11"},
			expectedArgs:      []string{"-m", "dynamo.sglang", "--dist-init-addr", "leader.example.com:29500", "--nnodes", "2", "--node-rank", "1"},
			description:       "Python version variants should be recognized",
		},
		{
			name:              "absolute path python command supported",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockSimpleDeployer{},
			initialCommand:    []string{"/usr/bin/python3.8"},
			initialArgs:       []string{"-m", "dynamo.sglang", "--model", "llama"},
			expectedCommand:   []string{"/usr/bin/python3.8"},
			expectedArgs:      []string{"-m", "dynamo.sglang", "--model", "llama", "--dist-init-addr", "leader.example.com:29500", "--nnodes", "2", "--node-rank", "1"},
			description:       "Absolute path Python commands should be recognized",
		},
		{
			name:              "pyenv shims python command supported",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"/home/user/.pyenv/shims/python3.9"},
			initialArgs:       []string{"-m", "dynamo.sglang"},
			expectedCommand:   []string{"sh", "-c"},
			expectedArgs:      []string{"exec /home/user/.pyenv/shims/python3.9 -m dynamo.sglang --dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)"},
			description:       "Pyenv shims Python paths should be recognized and wrapped with shell",
		},
		{
			name:              "python command with module in command array - simple deployer",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockSimpleDeployer{},
			initialCommand:    []string{"python3", "-m", "dynamo.sglang"},
			initialArgs:       []string{"--model-path", "Qwen/Qwen3-0.6B", "--tp-size", "8"},
			expectedCommand:   []string{"python3", "-m", "dynamo.sglang"},
			expectedArgs:      []string{"--model-path", "Qwen/Qwen3-0.6B", "--tp-size", "8", "--dist-init-addr", "leader.example.com:29500", "--nnodes", "2", "--node-rank", "1"},
			description:       "Multi-element python command should have flags appended to args",
		},
		{
			name:              "python command with module in command array - shell deployer",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"python3", "-m", "dynamo.sglang"},
			initialArgs:       []string{"--model-path", "Qwen/Qwen3-0.6B"},
			expectedCommand:   []string{"sh", "-c"},
			expectedArgs:      []string{"exec python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)"},
			description:       "Multi-element python command with shell deployer should wrap entire command",
		},
		{
			name:              "python command with no args - shell deployer",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"python3", "-m", "dynamo.sglang"},
			initialArgs:       []string{},
			expectedCommand:   []string{"sh", "-c"},
			expectedArgs:      []string{"exec python3 -m dynamo.sglang --dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)"},
			description:       "Multi-element python command with no args should still work with shell wrapper",
		},
		{
			name:              "non-python command multinode unchanged",
			numberOfNodes:     2,
			role:              RoleWorker,
			multinodeDeployer: &MockShellDeployer{},
			initialCommand:    []string{"java"},
			initialArgs:       []string{"-jar", "app.jar"},
			expectedCommand:   []string{"java"},
			expectedArgs:      []string{"-jar", "app.jar"}, // Args remain separate, no python found, no changes
			description:       "Non-python commands should remain unchanged (no flattening)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{
				Command: append([]string{}, tt.initialCommand...),
				Args:    append([]string{}, tt.initialArgs...),
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, &v1alpha1.DynamoComponentDeploymentSharedSpec{}, "test-service", tt.multinodeDeployer)

			if !reflect.DeepEqual(container.Command, tt.expectedCommand) {
				t.Errorf("UpdateContainer() command = %v, want %v", container.Command, tt.expectedCommand)
			}

			if !reflect.DeepEqual(container.Args, tt.expectedArgs) {
				t.Errorf("UpdateContainer() args = %v, want %v", container.Args, tt.expectedArgs)
			}
		})
	}
}

func TestSGLangBackend_ShellCommandInjection(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name              string
		numberOfNodes     int32
		role              Role
		multinodeDeployer MultinodeDeployer
		initialCommand    []string
		initialArgs       []string
		expectedArgs      []string
		description       string
	}{
		{
			name:              "single node shell command not modified",
			numberOfNodes:     1,
			role:              RoleMain,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang"},
			expectedArgs:      []string{"python -m dynamo.sglang"},
			description:       "Single node should not modify shell commands",
		},
		{
			name:              "multinode shell command with regex injection",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang"},
			expectedArgs:      []string{"python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 2 --node-rank 0"},
			description:       "Shell commands should use regex injection for python commands",
		},
		{
			name:              "multinode shell command with complex pipeline",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"echo blah | wc -l && python -m dynamo.sglang && ls -al"},
			expectedArgs:      []string{"echo blah | wc -l && python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 2 --node-rank 0 && ls -al"},
			description:       "Complex shell commands should inject flags only into python part",
		},
		{
			name:              "shell command worker with grove env vars",
			numberOfNodes:     3,
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang"},
			expectedArgs:      []string{"python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1))"},
			description:       "Shell command worker should get grove env vars in node rank",
		},
		{
			name:              "shell command with LWS deployer",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &LWSMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang"},
			expectedArgs:      []string{"python -m dynamo.sglang --dist-init-addr $(LWS_LEADER_ADDRESS):29500 --nnodes 2 --node-rank 0"},
			description:       "LWS shell commands should use LWS variables",
		},
		{
			name:              "shell command with pipes",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang | tee /tmp/log"},
			expectedArgs:      []string{"python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 2 --node-rank 0 | tee /tmp/log"},
			description:       "Shell commands with pipes should inject flags before pipe",
		},
		{
			name:              "shell command multiple args individual processing",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"echo start", "python -m dynamo.sglang", "echo done"},
			expectedArgs:      []string{"echo start", "python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 2 --node-rank 0", "echo done"},
			description:       "Shell commands with multiple args should process each individually, modify only the python arg",
		},
		{
			name:              "shell command no sglang modules unchanged",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"echo hello", "python -m some.other.module"},
			expectedArgs:      []string{"echo hello", "python -m some.other.module"},
			description:       "Shell commands without sglang modules should remain unchanged (args stay separate)",
		},
		{
			name:              "shell command stops after first python injection",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialCommand:    []string{"sh", "-c"},
			initialArgs:       []string{"python -m dynamo.sglang", "python -m dynamo.sglang --other-flags"},
			expectedArgs:      []string{"python -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 2 --node-rank 0", "python -m dynamo.sglang --other-flags"},
			description:       "Should stop processing after first successful python flag injection",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{
				Command: append([]string{}, tt.initialCommand...),
				Args:    append([]string{}, tt.initialArgs...),
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, &v1alpha1.DynamoComponentDeploymentSharedSpec{}, "test-service", tt.multinodeDeployer)

			if !reflect.DeepEqual(container.Args, tt.expectedArgs) {
				t.Errorf("UpdateContainer() args = %v, want %v", container.Args, tt.expectedArgs)
			}

			expectedCommand := tt.initialCommand
			if !reflect.DeepEqual(container.Command, expectedCommand) {
				t.Errorf("UpdateContainer() should preserve shell command, got: %v, want: %v", container.Command, expectedCommand)
			}
		})
	}
}

func TestIsPythonCommand(t *testing.T) {
	tests := []struct {
		cmd      string
		expected bool
	}{
		// Base python commands
		{"python", true},
		{"python3", true},
		{"python2", true},
		{"python3.11", true},
		{"python2.7", true},
		{"python3.12.1", true},

		// Absolute paths
		{"/usr/bin/python", true},
		{"/usr/bin/python3", true},
		{"/usr/bin/python3.8", true},
		{"/usr/bin/python2.7", true},
		{"/opt/python/bin/python3.11", true},
		{"/usr/local/bin/python3.12.1", true},
		{"/home/user/.pyenv/shims/python3.9", true},
		{"./python3", true},
		{"../bin/python", true},
		{"bin/python3.10", true},

		// Invalid cases
		{"java", false},
		{"sh", false},
		{"node", false},
		{"python-config", false},          // hyphen makes it not a python interpreter
		{"/usr/bin/python-config", false}, // hyphen in absolute path
		{"", false},
		{"python ", false},          // space makes it invalid
		{"/usr/bin/python ", false}, // space in absolute path
		{"pythonx", false},          // extra characters
		{"/usr/bin/pythonx", false}, // extra characters in absolute path
		{"/usr/bin/", false},        // empty filename
		{"python/", false},          // directory, not executable
		{"/usr/bin/python/", false}, // directory path
	}

	for _, tt := range tests {
		t.Run(tt.cmd, func(t *testing.T) {
			result := isPythonCommand(tt.cmd)
			if result != tt.expected {
				t.Errorf("isPythonCommand(%q) = %v, want %v", tt.cmd, result, tt.expected)
			}
		})
	}
}

func TestSGLangBackend_GetMultinodeFlags(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name               string
		numberOfNodes      int32
		role               Role
		multinodeDeployer  MultinodeDeployer
		expectedFlags      string
		expectedNeedsShell bool
		description        string
	}{
		{
			name:               "leader role never needs shell",
			numberOfNodes:      2,
			role:               RoleLeader,
			multinodeDeployer:  &MockShellDeployer{},
			expectedFlags:      "--dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank 0",
			expectedNeedsShell: false,
			description:        "Leader should always use rank 0 and no shell interpretation",
		},
		{
			name:               "worker with simple deployer",
			numberOfNodes:      3,
			role:               RoleWorker,
			multinodeDeployer:  &MockSimpleDeployer{},
			expectedFlags:      "--dist-init-addr leader.example.com:29500 --nnodes 3 --node-rank 1",
			expectedNeedsShell: false,
			description:        "Simple deployer should not need shell interpretation",
		},
		{
			name:               "worker with shell deployer",
			numberOfNodes:      2,
			role:               RoleWorker,
			multinodeDeployer:  &MockShellDeployer{},
			expectedFlags:      "--dist-init-addr $(LEADER_HOST):29500 --nnodes 2 --node-rank $(WORKER_INDEX)",
			expectedNeedsShell: true,
			description:        "Shell deployer should need shell interpretation for workers",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags, needsShell := backend.getMultinodeFlags(tt.numberOfNodes, tt.role, "test-service", tt.multinodeDeployer)

			if flags != tt.expectedFlags {
				t.Errorf("getMultinodeFlags() flags = %q, want %q", flags, tt.expectedFlags)
			}

			if needsShell != tt.expectedNeedsShell {
				t.Errorf("getMultinodeFlags() needsShell = %v, want %v", needsShell, tt.expectedNeedsShell)
			}
		})
	}
}

func TestSGLangBackend_ProbeRemoval(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name                string
		numberOfNodes       int32
		role                Role
		multinodeDeployer   MultinodeDeployer
		expectProbesRemoved bool
	}{
		{
			name:                "single node does not remove probes",
			numberOfNodes:       1,
			role:                RoleMain,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			expectProbesRemoved: false,
		},
		{
			name:                "multinode leader does not remove probes",
			numberOfNodes:       2,
			role:                RoleLeader,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			expectProbesRemoved: false,
		},
		{
			name:                "multinode worker removes probes",
			numberOfNodes:       2,
			role:                RoleWorker,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode main role does not remove probes",
			numberOfNodes:       2,
			role:                RoleMain,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			expectProbesRemoved: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			livenessProbe := &corev1.Probe{InitialDelaySeconds: 30}
			readinessProbe := &corev1.Probe{InitialDelaySeconds: 10}
			startupProbe := &corev1.Probe{InitialDelaySeconds: 5}

			container := &corev1.Container{
				Args:           []string{"python -m dynamo.sglang"},
				LivenessProbe:  livenessProbe,
				ReadinessProbe: readinessProbe,
				StartupProbe:   startupProbe,
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, &v1alpha1.DynamoComponentDeploymentSharedSpec{}, "test-service", tt.multinodeDeployer)

			if tt.expectProbesRemoved {
				if container.LivenessProbe != nil {
					t.Errorf("Expected LivenessProbe to be removed, but it was not")
				}
				if container.ReadinessProbe != nil {
					t.Errorf("Expected ReadinessProbe to be removed, but it was not")
				}
				if container.StartupProbe != nil {
					t.Errorf("Expected StartupProbe to be removed, but it was not")
				}
			} else {
				if container.LivenessProbe == nil {
					t.Errorf("Expected LivenessProbe to be preserved, but it was removed")
				}
				if container.ReadinessProbe == nil {
					t.Errorf("Expected ReadinessProbe to be preserved, but it was removed")
				}
				if container.StartupProbe == nil {
					t.Errorf("Expected StartupProbe to be preserved, but it was removed")
				}
			}
		})
	}
}

func TestSGLangBackend_UpdateContainer_UseAsCompilationCache(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name                       string
		component                  *v1alpha1.DynamoComponentDeploymentSharedSpec
		volumeMounts               []corev1.VolumeMount
		expectNoEnvVarChanges      bool
		expectLoggedPartialSupport bool
	}{
		{
			name: "SGLang backend with useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "sglang-cache",
						MountPoint:            "/cache/sglang",
						UseAsCompilationCache: true,
					},
				},
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true, // SGLang doesn't set env vars yet
			expectLoggedPartialSupport: true,
		},
		{
			name: "SGLang backend with useAsCompilationCache at custom volume mount",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "custom-cache",
						MountPoint:            "/custom/cache/path",
						UseAsCompilationCache: true,
					},
				},
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true, // SGLang doesn't set env vars yet
			expectLoggedPartialSupport: true,
		},
		{
			name: "SGLang backend without useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:       "regular-volume",
						MountPoint: "/data",
					},
				},
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true,
			expectLoggedPartialSupport: false,
		},
		{
			name: "SGLang backend with no volume mounts",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: nil,
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true,
			expectLoggedPartialSupport: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a container with initial state including volume mounts
			container := &corev1.Container{
				Env:          []corev1.EnvVar{},
				VolumeMounts: tt.volumeMounts,
			}

			// Store original env vars for comparison
			originalEnvCount := len(container.Env)

			// Call UpdateContainer (single node to avoid multinode logic)
			backend.UpdateContainer(container, 1, RoleMain, tt.component, "test-service", &GroveMultinodeDeployer{})

			if tt.expectNoEnvVarChanges {
				// Check that no new environment variables were added
				if len(container.Env) != originalEnvCount {
					t.Errorf("Expected no environment variable changes, but env count changed from %d to %d", originalEnvCount, len(container.Env))
				}
			}

		})
	}
}
