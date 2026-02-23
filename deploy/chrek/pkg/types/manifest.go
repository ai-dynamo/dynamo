package types

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"gopkg.in/yaml.v3"
)

const manifestFilename = "manifest.yaml"

// CheckpointManifest is saved as manifest.yaml at checkpoint time and loaded at restore.
type CheckpointManifest struct {
	CheckpointHash string    `yaml:"checkpointHash"`
	CreatedAt      time.Time `yaml:"createdAt"`

	CRIUDump   CRIUDumpManifest  `yaml:"criuDump"`
	K8s        SourcePodManifest `yaml:"k8s"`
	Overlay OverlayManifest   `yaml:"overlay"`
	CUDA       CUDAManifest      `yaml:"cudaRestore,omitempty"`
}

func NewCheckpointManifest(
	checkpointHash string,
	criuDump CRIUDumpManifest,
	k8s SourcePodManifest,
	overlay OverlayManifest,
) *CheckpointManifest {
	return &CheckpointManifest{
		CheckpointHash: checkpointHash,
		CreatedAt:      time.Now().UTC(),
		CRIUDump:       criuDump,
		K8s:            k8s,
		Overlay:     overlay,
	}
}

// CRIUDumpManifest stores the resolved dump-time CRIU mount plan used for restore.
type CRIUDumpManifest struct {
	CRIU     CRIUSettings      `yaml:"criu"`
	ExtMnt   map[string]string `yaml:"extMnt,omitempty"`
	External []string          `yaml:"external,omitempty"`
	SkipMnt  []string          `yaml:"skipMnt,omitempty"`
}

func NewCRIUDumpManifest(criuOpts *criurpc.CriuOpts, settings CRIUSettings) CRIUDumpManifest {
	m := CRIUDumpManifest{CRIU: settings}
	if criuOpts == nil {
		return m
	}

	m.ExtMnt = make(map[string]string, len(criuOpts.ExtMnt))
	for _, mount := range criuOpts.ExtMnt {
		if mount == nil || mount.GetKey() == "" {
			continue
		}
		m.ExtMnt[mount.GetKey()] = mount.GetVal()
	}
	if len(m.ExtMnt) == 0 {
		m.ExtMnt = nil
	}
	m.External = append([]string(nil), criuOpts.External...)
	m.SkipMnt = append([]string(nil), criuOpts.SkipMnt...)
	return m
}

// SourcePodManifest records the source pod identity at checkpoint time.
type SourcePodManifest struct {
	ContainerID  string `yaml:"containerId"`
	PID          int    `yaml:"pid"`
	SourceNode   string `yaml:"sourceNode"`
	PodName      string `yaml:"podName"`
	PodNamespace string `yaml:"podNamespace"`

	// StdioFDs holds readlink targets for FDs 0, 1, 2 (e.g. "pipe:[12345]").
	StdioFDs []string `yaml:"stdioFDs,omitempty"`
}

func NewSourcePodManifest(containerID string, pid int, sourceNode, podName, podNamespace string, stdioFDs []string) SourcePodManifest {
	return SourcePodManifest{
		ContainerID:  containerID,
		PID:          pid,
		SourceNode:   sourceNode,
		PodName:      podName,
		PodNamespace: podNamespace,
		StdioFDs:     append([]string(nil), stdioFDs...),
	}
}

// OverlayManifest holds runtime overlay state captured at checkpoint time.
type OverlayManifest struct {
	Exclusions     OverlaySettings `yaml:"exclusions"`
	UpperDir       string           `yaml:"upperDir,omitempty"`
	ExternalPaths  []string         `yaml:"externalPaths,omitempty"`
	BindMountDests []string         `yaml:"bindMountDests,omitempty"`
}

func NewOverlayManifest(exclusions OverlaySettings, upperDir string, ociSpec *specs.Spec) OverlayManifest {
	meta := OverlayManifest{
		Exclusions: exclusions,
		UpperDir:   upperDir,
	}
	if ociSpec == nil {
		return meta
	}

	if ociSpec.Linux != nil {
		meta.ExternalPaths = make([]string, 0, len(ociSpec.Linux.MaskedPaths)+len(ociSpec.Linux.ReadonlyPaths))
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.MaskedPaths...)
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.ReadonlyPaths...)
	}
	for _, m := range ociSpec.Mounts {
		if m.Type == "bind" {
			meta.BindMountDests = append(meta.BindMountDests, m.Destination)
		}
	}
	return meta
}

// CUDAManifest captures CUDA state from checkpoint time for restore.
type CUDAManifest struct {
	PIDs           []int    `yaml:"pids"`
	SourceGPUUUIDs []string `yaml:"sourceGpuUuids"`
}

func NewCUDAManifest(pids []int, sourceGPUUUIDs []string) CUDAManifest {
	return CUDAManifest{
		PIDs:           append([]int(nil), pids...),
		SourceGPUUUIDs: append([]string(nil), sourceGPUUUIDs...),
	}
}

func (m CUDAManifest) IsEmpty() bool {
	return len(m.PIDs) == 0
}

// WriteManifest writes a checkpoint manifest file in the checkpoint directory.
func WriteManifest(checkpointDir string, data *CheckpointManifest) error {
	content, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint manifest: %w", err)
	}

	manifestPath := filepath.Join(checkpointDir, manifestFilename)
	if err := os.WriteFile(manifestPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return nil
}

// ReadManifest reads checkpoint manifest from a checkpoint directory.
func ReadManifest(checkpointDir string) (*CheckpointManifest, error) {
	manifestPath := filepath.Join(checkpointDir, manifestFilename)

	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	var data CheckpointManifest
	if err := yaml.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint manifest: %w", err)
	}

	return &data, nil
}
