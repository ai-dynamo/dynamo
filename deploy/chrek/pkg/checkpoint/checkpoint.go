// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// ContainerInfoSnapshot holds runtime/container info needed for checkpointing.
type ContainerInfoSnapshot struct {
	PID        int
	RootFS     string
	UpperDir   string
	OCISpec    *specs.Spec
	MountInfo  []MountInfo
	Namespaces map[NamespaceType]*NamespaceInfo
}

// CheckpointManifest is saved as manifest.yaml at checkpoint time and loaded at restore.
type CheckpointManifest struct {
	CheckpointID string    `yaml:"checkpointId"`
	CreatedAt    time.Time `yaml:"createdAt"`

	CRIUDump   CRIUDumpManifest         `yaml:"criuDump"`
	K8s        SourcePodManifest        `yaml:"k8s"`
	Filesystem FilesystemManifest       `yaml:"filesystem"`
	Namespaces []NamespaceManifestEntry `yaml:"namespaces"`
}

// NewCheckpointManifest assembles a CheckpointManifest from per-module builders.
func NewCheckpointManifest(
	checkpointID string,
	criuDump CRIUDumpManifest,
	k8s SourcePodManifest,
	filesystem FilesystemManifest,
	namespaces []NamespaceManifestEntry,
) *CheckpointManifest {
	return &CheckpointManifest{
		CheckpointID: checkpointID,
		CreatedAt:    time.Now().UTC(),
		CRIUDump:     criuDump,
		K8s:          k8s,
		Filesystem:   filesystem,
		Namespaces:   namespaces,
	}
}

// CheckpointRequest holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointRequest struct {
	ContainerID   string
	ContainerName string // K8s container name (for K8s API volume type lookup)
	CheckpointID  string
	CheckpointDir string
	NodeName      string
	PodName       string
	PodNamespace  string
}

// CheckpointOutcome contains the result of a checkpoint operation.
type CheckpointOutcome struct {
	CheckpointID  string
	CheckpointDir string
	Data          *CheckpointManifest
}

// Checkpointer performs CRIU checkpoint operations
type Checkpointer struct {
	discoveryClient *DiscoveryClient
	log             *logrus.Entry
}

// NewCheckpointer creates a new checkpointer
func NewCheckpointer(discoveryClient *DiscoveryClient) *Checkpointer {
	return &Checkpointer{
		discoveryClient: discoveryClient,
		log:             logrus.WithField("component", "checkpointer"),
	}
}

// Checkpoint performs a CRIU dump of a container.
// The operation has three phases: introspect, configure, capture.
func (c *Checkpointer) Checkpoint(ctx context.Context, req CheckpointRequest, spec *CheckpointSpec) (*CheckpointOutcome, error) {
	if spec == nil {
		return nil, fmt.Errorf("checkpoint spec is required")
	}
	checkpointStart := time.Now()
	c.log.Info("=== Starting checkpoint operation ===")

	checkpointDir := filepath.Join(req.CheckpointDir, req.CheckpointID)
	if err := os.MkdirAll(checkpointDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// Open image directory FD for CRIU — must stay open through both configure and capture
	// phases since CRIU's swrk child process inherits this FD.
	imageDir, imageDirFD, err := common.OpenPathForCRIU(checkpointDir)
	if err != nil {
		return nil, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()

	// Phase 1: Introspect container state
	state, err := c.introspect(ctx, req.ContainerID)
	if err != nil {
		return nil, err
	}

	// Phase 2: Configure CRIU options and build checkpoint manifest.
	criuOpts, data, err := c.configure(state, req, spec, checkpointDir, imageDirFD)
	if err != nil {
		return nil, err
	}

	// Phase 3: Capture — CRIU dump, /dev/shm, rootfs diff
	criuDumpDuration, err := c.capture(criuOpts, data, state, checkpointDir)
	if err != nil {
		return nil, err
	}

	totalDuration := time.Since(checkpointStart)
	c.log.WithFields(logrus.Fields{
		"total_duration":     totalDuration,
		"criu_dump_duration": criuDumpDuration,
	}).Info("=== Checkpoint operation completed ===")

	return &CheckpointOutcome{
		CheckpointID:  req.CheckpointID,
		CheckpointDir: checkpointDir,
		Data:          data,
	}, nil
}

// introspect resolves the container and gathers all runtime state from containerd and /proc.
func (c *Checkpointer) introspect(ctx context.Context, containerID string) (*ContainerInfoSnapshot, error) {
	pid, ociSpec, err := c.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}

	rootFS, err := GetRootFS(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}
	upperDir, err := GetOverlayUpperDir(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get overlay upperdir: %w", err)
	}
	mountInfo, err := ReadMountInfoFromHostProcPath(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}
	namespaces, err := GetAllNamespaces(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get namespaces: %w", err)
	}

	return &ContainerInfoSnapshot{
		PID:        pid,
		RootFS:     rootFS,
		UpperDir:   upperDir,
		OCISpec:    ociSpec,
		MountInfo:  mountInfo,
		Namespaces: namespaces,
	}, nil
}

// configure builds CRIU options and checkpoint manifest from runtime snapshot and spec.
func (c *Checkpointer) configure(
	state *ContainerInfoSnapshot,
	req CheckpointRequest,
	spec *CheckpointSpec,
	checkpointDir string,
	imageDirFD int32,
) (*criurpc.CriuOpts, *CheckpointManifest, error) {
	criuOpts, err := BuildCRIUDumpOptions(
		&spec.CRIU,
		state.PID,
		imageDirFD,
		state.RootFS,
		state.MountInfo,
		state.OCISpec,
		state.Namespaces,
	)
	if err != nil {
		return nil, nil, err
	}

	// Write CRIU config file (for options unavailable via RPC)
	configPath := filepath.Join(checkpointDir, CheckpointCRIUConfFilename)
	if err := os.WriteFile(configPath, []byte(spec.CRIU.GenerateCRIUConfContent()), 0644); err != nil {
		return nil, nil, fmt.Errorf("failed to write CRIU config file: %w", err)
	}
	criuOpts.ConfigFile = proto.String(configPath)

	// Build and save the checkpoint manifest.
	manifest := NewCheckpointManifest(
		req.CheckpointID,
		NewCRIUDumpManifest(criuOpts, spec.CRIU),
		NewSourcePodManifest(req, state.PID),
		NewFilesystemManifest(spec.RootfsExclusions, state.UpperDir, state.OCISpec),
		NewNamespaceManifestEntries(state.Namespaces),
	)

	if err := WriteCheckpointManifest(checkpointDir, manifest); err != nil {
		return nil, nil, fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return criuOpts, manifest, nil
}

// capture executes the CRIU dump and post-dump captures (/dev/shm, rootfs diff).
// Returns the CRIU dump duration for timing reporting.
func (c *Checkpointer) capture(
	criuOpts *criurpc.CriuOpts,
	data *CheckpointManifest,
	state *ContainerInfoSnapshot,
	checkpointDir string,
) (time.Duration, error) {
	criuDumpDuration, err := ExecuteCRIUDump(criuOpts, checkpointDir, c.log)
	if err != nil {
		return 0, err
	}

	// Capture /dev/shm contents (must happen after dump for final process state)
	if err := CaptureDevShm(state.PID, checkpointDir, c.log); err != nil {
		c.log.WithError(err).Warn("Failed to capture /dev/shm contents")
	}

	// Capture rootfs diff and deleted files
	CaptureRootfsState(state.UpperDir, checkpointDir, data, c.log)

	return criuDumpDuration, nil
}
