/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestPowerGateWrapPreservesFinalRenderedCommandAndArgs(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
	}{
		{
			name:    "direct command",
			command: []string{"python3", "-m", "dynamo.vllm"},
			args:    []string{"--model", "Qwen/Qwen3-0.6B"},
		},
		{
			name:    "final shell command",
			command: []string{"/bin/sh", "-c"},
			args:    []string{"setup && exec python3 -m dynamo.trtllm | tee /dev/stderr"},
		},
		{
			name:    "command only preserves image default args",
			command: []string{"python3", "-m", "dynamo.vllm"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build a final rendered PodSpec with opaque command and args tokens")
			originalCommand := append([]string(nil), test.command...)
			originalArgs := append([]string(nil), test.args...)
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name:    commonconsts.MainContainerName,
				Command: append([]string(nil), originalCommand...),
				Args:    append([]string(nil), originalArgs...),
			}}}

			t.Log("Apply the gate after rendering")
			err := ApplyPowerGate(podSpec, testPowerGateInputs())
			if err != nil {
				t.Fatalf("ApplyPowerGate() error = %v", err)
			}

			t.Log("Verify structural prefixing and byte-for-byte args preservation")
			mainContainer := &podSpec.Containers[0]
			wantCommand := append([]string{PowerGateExecutable, PowerGateSeparator}, originalCommand...)
			if !reflect.DeepEqual(mainContainer.Command, wantCommand) {
				t.Fatalf("command = %#v, want %#v", mainContainer.Command, wantCommand)
			}
			if !reflect.DeepEqual(mainContainer.Args, originalArgs) {
				t.Fatalf("args = %#v, want unchanged %#v", mainContainer.Args, originalArgs)
			}

			t.Log("Verify immutable gate inputs and the read-only Downward API projection")
			assertPowerGateEnvironment(t, mainContainer.Env)
			if len(mainContainer.VolumeMounts) != 1 || mainContainer.VolumeMounts[0].Name != PowerGateVolumeName ||
				mainContainer.VolumeMounts[0].MountPath != PowerGateVolumeMountPath || !mainContainer.VolumeMounts[0].ReadOnly {
				t.Fatalf("power gate volume mount = %#v", mainContainer.VolumeMounts)
			}
			if len(podSpec.Volumes) != 1 || podSpec.Volumes[0].DownwardAPI == nil {
				t.Fatalf("power gate volume = %#v", podSpec.Volumes)
			}
			items := podSpec.Volumes[0].DownwardAPI.Items
			if len(items) != 2 || items[0].Path != PowerGatePodUIDPath || items[0].FieldRef.FieldPath != "metadata.uid" ||
				items[1].Path != PowerGateReportPath || items[1].FieldRef.FieldPath != powerGateReportAnnotationFieldPath {
				t.Fatalf("Downward API items = %#v", items)
			}
			if mainContainer.TerminationMessagePath != PowerGateTerminationMessagePath ||
				mainContainer.TerminationMessagePolicy != corev1.TerminationMessageReadFile {
				t.Fatalf("termination message configuration = (%q, %q)", mainContainer.TerminationMessagePath, mainContainer.TerminationMessagePolicy)
			}
		})
	}
}

func TestPowerGateWrapAfterTRTLLMSetup(t *testing.T) {
	t.Log("Render the final multinode TensorRT-LLM leader command")
	component := betaComponent(t, &v1alpha1.DynamoComponentDeploymentSharedSpec{
		Resources: &v1alpha1.Resources{
			Requests: &v1alpha1.ResourceItem{GPU: "1"},
		},
	})
	mainContainer := corev1.Container{
		Name:    commonconsts.MainContainerName,
		Command: []string{"python3", "-m", "dynamo.trtllm"},
		Args:    []string{"--model-path", "Qwen/Qwen3-0.6B"},
	}
	backend := &TRTLLMBackend{MpiRunSecretName: "trtllm-ssh"}
	backend.UpdateContainer(
		&mainContainer,
		2,
		RoleLeader,
		component,
		"decode",
		&GroveMultinodeDeployer{},
		staticContainerGPUCount(1),
	)
	finalCommand := append([]string(nil), mainContainer.Command...)
	finalArgs := append([]string(nil), mainContainer.Args...)
	podSpec := &corev1.PodSpec{Containers: []corev1.Container{mainContainer}}

	t.Log("Apply the gate only after TensorRT-LLM has completed its rewrite")
	if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err != nil {
		t.Fatalf("ApplyPowerGate() error = %v", err)
	}

	t.Log("Verify the complete TensorRT-LLM output remains opaque and unchanged")
	wantCommand := append([]string{PowerGateExecutable, PowerGateSeparator}, finalCommand...)
	if !reflect.DeepEqual(podSpec.Containers[0].Command, wantCommand) {
		t.Fatalf("command = %#v, want %#v", podSpec.Containers[0].Command, wantCommand)
	}
	if !reflect.DeepEqual(podSpec.Containers[0].Args, finalArgs) {
		t.Fatalf("args = %#v, want unchanged %#v", podSpec.Containers[0].Args, finalArgs)
	}
}

func TestReservedEnvCollisionRejectsBeforeMutation(t *testing.T) {
	reservedNames := []string{
		PowerGateDGDUIDEnv,
		PowerGateComponentEnv,
		PowerGateExpectedGPUCountEnv,
		PowerGateInGateBoundWattsPerGPUEnv,
	}

	for _, reservedName := range reservedNames {
		t.Run(reservedName, func(t *testing.T) {
			t.Log("Build a final container containing a user-owned reserved variable")
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name:    commonconsts.MainContainerName,
				Command: []string{"python3"},
				Args:    []string{"-m", "dynamo.vllm"},
				Env:     []corev1.EnvVar{{Name: reservedName, Value: "user-value"}},
			}}}
			before := podSpec.DeepCopy()

			t.Log("Reject the collision without partially mutating the PodSpec")
			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err == nil {
				t.Fatal("ApplyPowerGate() error = nil, want reserved collision")
			}
			if !reflect.DeepEqual(podSpec, before) {
				t.Fatalf("PodSpec changed on collision: got %#v want %#v", podSpec, before)
			}
		})
	}
}

func TestPowerGateWrapRejectsReservedMountOverlap(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*corev1.PodSpec)
	}{
		{
			name: "report file subPath",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers[0].VolumeMounts = []corev1.VolumeMount{{
					Name: "forged-report", MountPath: PowerGateVolumeMountPath + "/report", SubPath: "report",
				}}
			},
		},
		{
			name: "parent directory",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "parent", MountPath: "/var/run/dynamo"}}
			},
		},
		{
			name: "overlapping device path",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers[0].VolumeDevices = []corev1.VolumeDevice{{
					Name: "forged-device", DevicePath: PowerGateVolumeMountPath + "/pod-uid",
				}}
			},
		},
		{
			name: "sidecar reserved volume name",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers = append(podSpec.Containers, corev1.Container{
					Name: "sidecar", VolumeMounts: []corev1.VolumeMount{{Name: PowerGateVolumeName, MountPath: "/data"}},
				})
			},
		},
		{
			name: "ephemeral reserved volume device name",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.EphemeralContainers = []corev1.EphemeralContainer{{
					EphemeralContainerCommon: corev1.EphemeralContainerCommon{
						Name: "debug",
						VolumeDevices: []corev1.VolumeDevice{{
							Name: PowerGateVolumeName, DevicePath: "/dev/debug-gate",
						}},
					},
				}}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build a final PodSpec with an overlapping user-controlled mount")
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName, Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"},
			}}}
			test.mutate(podSpec)
			before := podSpec.DeepCopy()

			t.Log("Reject the shadowing path or volume reference before mutation")
			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err == nil {
				t.Fatal("ApplyPowerGate() error = nil, want reserved mount collision")
			}
			if !reflect.DeepEqual(podSpec, before) {
				t.Fatalf("PodSpec changed on collision: got %#v want %#v", podSpec, before)
			}
		})
	}
}

func TestPowerGateWrapRejectsAlternateMainExecPaths(t *testing.T) {
	execAction := func() *corev1.ExecAction {
		return &corev1.ExecAction{Command: []string{"python3", "-m", "dynamo.vllm"}}
	}
	tests := []struct {
		name   string
		mutate func(*corev1.Container)
	}{
		{
			name: "post-start exec",
			mutate: func(container *corev1.Container) {
				container.Lifecycle = &corev1.Lifecycle{
					PostStart: &corev1.LifecycleHandler{Exec: execAction()},
				}
			},
		},
		{
			name: "pre-stop exec",
			mutate: func(container *corev1.Container) {
				container.Lifecycle = &corev1.Lifecycle{
					PreStop: &corev1.LifecycleHandler{Exec: execAction()},
				}
			},
		},
		{
			name: "startup exec probe",
			mutate: func(container *corev1.Container) {
				container.StartupProbe = &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: execAction()}}
			},
		},
		{
			name: "readiness exec probe",
			mutate: func(container *corev1.Container) {
				container.ReadinessProbe = &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: execAction()}}
			},
		},
		{
			name: "liveness exec probe",
			mutate: func(container *corev1.Container) {
				container.LivenessProbe = &corev1.Probe{ProbeHandler: corev1.ProbeHandler{Exec: execAction()}}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName, Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"},
			}}}
			test.mutate(&podSpec.Containers[0])
			before := podSpec.DeepCopy()

			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err == nil {
				t.Fatal("ApplyPowerGate() error = nil, want alternate exec path rejection")
			}
			if !reflect.DeepEqual(podSpec, before) {
				t.Fatalf("PodSpec changed on rejection: got %#v want %#v", podSpec, before)
			}
		})
	}
}

func TestPowerGateWrapRejectsTerminationMessageCollisions(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*corev1.Container)
	}{
		{
			name: "custom termination path",
			mutate: func(container *corev1.Container) {
				container.TerminationMessagePath = "/tmp/user-termination-log"
			},
		},
		{
			name: "fallback log policy",
			mutate: func(container *corev1.Container) {
				container.TerminationMessagePolicy = corev1.TerminationMessageFallbackToLogsOnError
			},
		},
		{
			name: "termination file shadow",
			mutate: func(container *corev1.Container) {
				container.VolumeMounts = []corev1.VolumeMount{{Name: "shadow", MountPath: PowerGateTerminationMessagePath, SubPath: "message"}}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName, Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"},
			}}}
			test.mutate(&podSpec.Containers[0])
			before := podSpec.DeepCopy()

			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err == nil {
				t.Fatal("ApplyPowerGate() error = nil, want termination-message collision rejection")
			}
			if !reflect.DeepEqual(podSpec, before) {
				t.Fatalf("PodSpec changed on rejection: got %#v want %#v", podSpec, before)
			}
		})
	}
}

func TestPowerGateWrapRejectsGPUAllocationsOutsideMain(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*corev1.PodSpec)
	}{
		{
			name: "GPU sidecar",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers = append(podSpec.Containers, corev1.Container{
					Name: "sidecar",
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
						corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
					}},
				})
			},
		},
		{
			name: "MIG init container",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.InitContainers = []corev1.Container{{
					Name: "prepare",
					Resources: corev1.ResourceRequirements{Requests: corev1.ResourceList{
						corev1.ResourceName("nvidia.com/mig-1g.10gb"): resource.MustParse("1"),
					}},
				}}
			},
		},
		{
			name: "time-sliced GPU init container",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.InitContainers = []corev1.Container{{
					Name: "prepare",
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
						corev1.ResourceName("nvidia.com/gpu.shared"): resource.MustParse("1"),
					}},
				}}
			},
		},
		{
			name: "ephemeral GPU container",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.EphemeralContainers = []corev1.EphemeralContainer{{
					EphemeralContainerCommon: corev1.EphemeralContainerCommon{
						Name: "debug",
						Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
							corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
						}},
					},
				}}
			},
		},
		{
			name: "DRA claim",
			mutate: func(podSpec *corev1.PodSpec) {
				podSpec.Containers[0].Resources.Claims = []corev1.ResourceClaim{{Name: "gpu"}}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName, Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"},
			}}}
			test.mutate(podSpec)
			before := podSpec.DeepCopy()

			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err == nil {
				t.Fatal("ApplyPowerGate() error = nil, want unsupported GPU allocation rejection")
			}
			if !reflect.DeepEqual(podSpec, before) {
				t.Fatalf("PodSpec changed on rejection: got %#v want %#v", podSpec, before)
			}
		})
	}
}

func TestDetectBackendBehindGate(t *testing.T) {
	tests := []struct {
		name    string
		command []string
		args    []string
	}{
		{name: "direct", command: []string{"python3"}, args: []string{"-m", "dynamo.vllm"}},
		{name: "shell", command: []string{"/bin/sh", "-c"}, args: []string{"python3 -m dynamo.vllm"}},
		{name: "piped", command: []string{"/bin/sh", "-c"}, args: []string{"python3 -m dynamo.vllm | tee /tmp/backend.log"}},
		{name: "chained", command: []string{"/bin/sh", "-c"}, args: []string{"setup-model && exec python3 -m dynamo.vllm"}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Wrap the final custom command without parsing its content")
			originalArgs := append([]string(nil), test.args...)
			podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
				Name:    commonconsts.MainContainerName,
				Command: append([]string(nil), test.command...),
				Args:    append([]string(nil), test.args...),
			}}}
			if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err != nil {
				t.Fatalf("ApplyPowerGate() error = %v", err)
			}

			t.Log("Detect the original backend through the structural prefix")
			mainContainer := podSpec.Containers[0]
			framework, err := DetectBackendFrameworkFromArgs(mainContainer.Command, mainContainer.Args)
			if err != nil {
				t.Fatalf("DetectBackendFrameworkFromArgs() error = %v", err)
			}
			if framework != BackendFrameworkVLLM {
				t.Fatalf("framework = %q, want %q", framework, BackendFrameworkVLLM)
			}
			if !reflect.DeepEqual(mainContainer.Args, originalArgs) {
				t.Fatalf("args = %#v, want unchanged %#v", mainContainer.Args, originalArgs)
			}
		})
	}
}

func TestLoRABehindGate(t *testing.T) {
	t.Log("Build the gated live-Pod command consumed by the LoRA fallback")
	podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name:    commonconsts.MainContainerName,
		Command: []string{"python3"},
		Args:    []string{"-m", "dynamo.vllm", "--enable-lora"},
	}}}
	if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err != nil {
		t.Fatalf("ApplyPowerGate() error = %v", err)
	}

	t.Log("Use the same backend detector as live-Pod LoRA classification")
	mainContainer := podSpec.Containers[0]
	framework, err := DetectBackendFrameworkFromArgs(mainContainer.Command, mainContainer.Args)
	if err != nil {
		t.Fatalf("DetectBackendFrameworkFromArgs() error = %v", err)
	}
	if framework != BackendFrameworkVLLM {
		t.Fatalf("framework = %q, want %q", framework, BackendFrameworkVLLM)
	}
}

func TestMissingWrapperFailsBeforeOriginal(t *testing.T) {
	t.Log("Wrap an original command that would create an observable marker")
	marker := filepath.Join(t.TempDir(), "original-started")
	originalCommand := []string{"/bin/sh", "-c"}
	originalArgs := []string{"touch " + marker}
	podSpec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name:    commonconsts.MainContainerName,
		Command: append([]string(nil), originalCommand...),
		Args:    append([]string(nil), originalArgs...),
	}}}
	if err := ApplyPowerGate(podSpec, testPowerGateInputs()); err != nil {
		t.Fatalf("ApplyPowerGate() error = %v", err)
	}

	t.Log("Verify Kubernetes must resolve the gate executable before any original token")
	mainContainer := podSpec.Containers[0]
	wantCommand := append([]string{PowerGateExecutable, PowerGateSeparator}, originalCommand...)
	if !reflect.DeepEqual(mainContainer.Command, wantCommand) {
		t.Fatalf("command = %#v, want %#v", mainContainer.Command, wantCommand)
	}
	if !reflect.DeepEqual(mainContainer.Args, originalArgs) {
		t.Fatalf("args = %#v, want unchanged %#v", mainContainer.Args, originalArgs)
	}

	t.Log("Execute with a PATH that cannot resolve the wrapper")
	t.Setenv("PATH", t.TempDir())
	commandArgs := append([]string(nil), mainContainer.Command[1:]...)
	commandArgs = append(commandArgs, mainContainer.Args...)
	if err := exec.Command(mainContainer.Command[0], commandArgs...).Run(); err == nil {
		t.Fatal("missing wrapper execution error = nil")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatalf("original command marker stat error = %v, want not found", err)
	}
}

func testPowerGateInputs() PowerGateInputs {
	return PowerGateInputs{
		DGDUID:                   "dgd-uid",
		Component:                "decode",
		ExpectedPhysicalGPUCount: 2,
		InGateBoundWattsPerGPU:   350,
	}
}

func assertPowerGateEnvironment(t *testing.T, env []corev1.EnvVar) {
	t.Helper()

	want := map[string]string{
		PowerGateDGDUIDEnv:                 "dgd-uid",
		PowerGateComponentEnv:              "decode",
		PowerGateExpectedGPUCountEnv:       "2",
		PowerGateInGateBoundWattsPerGPUEnv: "350",
	}
	if len(env) != len(want) {
		t.Fatalf("power gate env length = %d, want %d: %#v", len(env), len(want), env)
	}
	for _, item := range env {
		if want[item.Name] != item.Value {
			t.Fatalf("power gate env %q = %q, want %q", item.Name, item.Value, want[item.Name])
		}
	}
}
