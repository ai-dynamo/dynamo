package runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	specs "github.com/opencontainers/runtime-spec/specs-go"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	remote "k8s.io/cri-client/pkg"
)

const (
	crioConnectTimeout = 2 * time.Second
	crioCallTimeout    = 10 * time.Second
)

// CRIORuntime resolves container identity via the CRI-O CRI gRPC socket.
type CRIORuntime struct {
	svc internalapi.RuntimeService
}

func NewCRIORuntime(socket string) (*CRIORuntime, error) {
	svc, err := remote.NewRemoteRuntimeService(socket, crioConnectTimeout, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to dial CRI-O at %s: %w", socket, err)
	}
	return &CRIORuntime{svc: svc}, nil
}

// Close is a no-op: k8s.io/cri-client's RuntimeService interface doesn't
// expose one. The gRPC connection is released at process exit.
func (r *CRIORuntime) Close() error { return nil }

func (r *CRIORuntime) ResolveContainer(ctx context.Context, id string) (int, *specs.Spec, error) {
	ctx, cancel := context.WithTimeout(ctx, crioCallTimeout)
	defer cancel()

	resp, err := r.svc.ContainerStatus(ctx, id, true)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get container status for %s: %w", id, err)
	}
	return parseCRIOStatus(id, resp.GetInfo())
}

// ResolveContainerByPod picks the first RUNNING container matching the pod +
// container-name label filter; errors if none qualify.
func (r *CRIORuntime) ResolveContainerByPod(ctx context.Context, podName, podNamespace, containerName string) (int, *specs.Spec, error) {
	running, err := r.findRunningContainerByPod(ctx, podName, podNamespace, containerName)
	if err != nil {
		return 0, nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, crioCallTimeout)
	defer cancel()

	statusResp, err := r.svc.ContainerStatus(ctx, running.GetId(), true)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to get status for container %s (pod %s/%s): %w", running.GetId(), podNamespace, podName, err)
	}
	return parseCRIOStatus(running.GetId(), statusResp.GetInfo())
}

func (r *CRIORuntime) ResolveContainerIDByPod(ctx context.Context, podName, podNamespace, containerName string) (string, error) {
	running, err := r.findRunningContainerByPod(ctx, podName, podNamespace, containerName)
	if err != nil {
		return "", err
	}
	return running.GetId(), nil
}

// findRunningContainerByPod picks the first RUNNING container matching the pod
// + container-name label filter; errors if none qualify.
func (r *CRIORuntime) findRunningContainerByPod(ctx context.Context, podName, podNamespace, containerName string) (*runtimeapi.Container, error) {
	ctx, cancel := context.WithTimeout(ctx, crioCallTimeout)
	defer cancel()

	filter := &runtimeapi.ContainerFilter{
		LabelSelector: map[string]string{
			"io.kubernetes.pod.name":       podName,
			"io.kubernetes.pod.namespace":  podNamespace,
			"io.kubernetes.container.name": containerName,
		},
	}
	containers, err := r.svc.ListContainers(ctx, filter)
	if err != nil {
		return nil, fmt.Errorf("failed to list containers for pod %s/%s: %w", podNamespace, podName, err)
	}
	if len(containers) == 0 {
		return nil, fmt.Errorf("no container found for pod %s/%s container %s", podNamespace, podName, containerName)
	}

	var running *runtimeapi.Container
	for _, c := range containers {
		if c.GetState() == runtimeapi.ContainerState_CONTAINER_RUNNING {
			running = c
			break
		}
	}
	if running == nil {
		return nil, fmt.Errorf("no running container found for pod %s/%s container %s (%d candidates)", podNamespace, podName, containerName, len(containers))
	}
	return running, nil
}

// crioInfo is the JSON shape CRI-O writes into ContainerStatus.Info["info"]
// when verbose=true. The outer Info map is part of CRI; the "info" key and
// this JSON shape are CRI-O-specific (crictl, kubelet checkpointing, and
// podman all parse it the same way).
type crioInfo struct {
	Pid         int         `json:"pid"`
	RuntimeSpec *specs.Spec `json:"runtimeSpec,omitempty"`
}

func parseCRIOStatus(id string, info map[string]string) (int, *specs.Spec, error) {
	raw := info["info"]
	if raw == "" {
		return 0, nil, fmt.Errorf("CRI-O Info[info] missing for %s", id)
	}
	var parsed crioInfo
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return 0, nil, fmt.Errorf("failed to parse CRI-O Info[info] JSON for %s: %w", id, err)
	}
	if parsed.Pid <= 0 {
		return 0, nil, fmt.Errorf("CRI-O Info[info] for %s missing pid", id)
	}
	if parsed.RuntimeSpec == nil {
		return 0, nil, fmt.Errorf("CRI-O Info[info] for %s missing runtimeSpec", id)
	}
	return parsed.Pid, parsed.RuntimeSpec, nil
}
