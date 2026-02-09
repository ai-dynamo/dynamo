// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// CheckpointMetadata is saved as metadata.yaml at checkpoint time, loaded at restore.
type CheckpointMetadata struct {
	CheckpointID string    `yaml:"checkpointId"`
	CreatedAt    time.Time `yaml:"createdAt"`

	CRIU       CRIUConfig          `yaml:"criu"`
	K8s        K8sMetadata         `yaml:"k8s"`
	Filesystem FilesystemMetadata  `yaml:"filesystem"`
	Mounts     []MountMetadata     `yaml:"mounts"`
	Namespaces []NamespaceMetadata `yaml:"namespaces"`
}

// NewCheckpointMetadata assembles a CheckpointMetadata from per-module builders.
func NewCheckpointMetadata(
	checkpointID string,
	criuCfg CRIUConfig,
	k8s K8sMetadata,
	filesystem FilesystemMetadata,
	mounts []MountMetadata,
	namespaces []NamespaceMetadata,
) *CheckpointMetadata {
	return &CheckpointMetadata{
		CheckpointID: checkpointID,
		CreatedAt:    time.Now().UTC(),
		CRIU:         criuCfg,
		K8s:          k8s,
		Filesystem:   filesystem,
		Mounts:       mounts,
		Namespaces:   namespaces,
	}
}

// CheckpointParams holds per-checkpoint identifiers for a checkpoint operation.
type CheckpointParams struct {
	ContainerID   string
	ContainerName string // K8s container name (for K8s API volume type lookup)
	CheckpointID  string
	CheckpointDir string
	NodeName      string
	PodName       string
	PodNamespace  string
}

// Result contains the result of a checkpoint operation
type Result struct {
	CheckpointID  string
	CheckpointDir string
	Data          *CheckpointMetadata
}

// containerState holds all introspected container state needed for checkpoint.
type containerState struct {
	ContainerInfo *ContainerInfo
	PID           int
	RootFS        string
	UpperDir      string
	Mounts        []MountMapping
	Namespaces    map[NamespaceType]*NamespaceInfo
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
func (c *Checkpointer) Checkpoint(ctx context.Context, params CheckpointParams, cfg *Config) (*Result, error) {
	if cfg == nil {
		return nil, fmt.Errorf("checkpoint config is required")
	}
	checkpointStart := time.Now()
	c.log.Info("=== Starting checkpoint operation ===")

	checkpointDir := filepath.Join(params.CheckpointDir, params.CheckpointID)
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
	state, err := c.introspect(ctx, params.ContainerID)
	if err != nil {
		return nil, err
	}

	// Phase 2: Configure CRIU options and build checkpoint metadata
	criuOpts, data, err := c.configure(state, params, cfg, checkpointDir, imageDirFD)
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

	return &Result{
		CheckpointID:  params.CheckpointID,
		CheckpointDir: checkpointDir,
		Data:          data,
	}, nil
}

// introspect resolves the container and gathers all runtime state from containerd and /proc.
func (c *Checkpointer) introspect(ctx context.Context, containerID string) (*containerState, error) {
	containerInfo, err := c.discoveryClient.ResolveContainer(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}
	pid := int(containerInfo.PID)

	rootFS, err := GetRootFS(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}
	upperDir, err := GetOverlayUpperDir(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get overlay upperdir: %w", err)
	}
	mounts, err := GetKubernetesVolumeMounts(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get mounts: %w", err)
	}
	namespaces, err := GetAllNamespaces(pid)
	if err != nil {
		return nil, fmt.Errorf("failed to get namespaces: %w", err)
	}

	return &containerState{
		ContainerInfo: containerInfo,
		PID:           pid,
		RootFS:        rootFS,
		UpperDir:      upperDir,
		Mounts:        mounts,
		Namespaces:    namespaces,
	}, nil
}

// configure builds CRIU options and checkpoint metadata from introspected state and config.
func (c *Checkpointer) configure(
	state *containerState,
	params CheckpointParams,
	cfg *Config,
	checkpointDir string,
	imageDirFD int32,
) (*criurpc.CriuOpts, *CheckpointMetadata, error) {
	// Build CRIU options from config
	criuOpts := BuildCRIUOpts(&cfg.CRIU, CRIUDumpParams{
		PID:        state.PID,
		ImageDirFD: imageDirFD,
		RootFS:     state.RootFS,
	})

	// Write CRIU config file (for options unavailable via RPC)
	configPath := filepath.Join(checkpointDir, CheckpointCRIUConfFilename)
	if err := os.WriteFile(configPath, []byte(cfg.CRIU.GenerateCRIUConfContent()), 0644); err != nil {
		return nil, nil, fmt.Errorf("failed to write CRIU config file: %w", err)
	}
	criuOpts.ConfigFile = proto.String(configPath)

	// Configure external mounts and namespaces
	if err := ConfigureExternalMounts(criuOpts, state.PID, state.ContainerInfo); err != nil {
		return nil, nil, err
	}
	ConfigureExternalNamespaces(criuOpts, state.Namespaces, cfg.CRIU.ExternalMounts)

	// Configure mounts to skip (for cross-node restore)
	if _, err := ConfigureSkipMounts(criuOpts, state.PID, cfg.SkipMountPrefixes); err != nil {
		return nil, nil, err
	}

	// Build and save checkpoint metadata
	oci := extractOCIState(state.ContainerInfo)
	metadata := NewCheckpointMetadata(
		params.CheckpointID,
		cfg.CRIU,
		NewK8sMetadata(params, state.PID, state.ContainerInfo),
		NewFilesystemMetadata(cfg.RootfsExclusions, state.UpperDir, oci),
		NewMountMetadata(state.Mounts, oci),
		NewNamespaceMetadata(state.Namespaces),
	)

	if err := SaveCheckpointMetadata(checkpointDir, metadata); err != nil {
		return nil, nil, fmt.Errorf("failed to save checkpoint metadata: %w", err)
	}

	return criuOpts, metadata, nil
}

// capture executes the CRIU dump and post-dump captures (/dev/shm, rootfs diff).
// Returns the CRIU dump duration for timing reporting.
func (c *Checkpointer) capture(
	criuOpts *criurpc.CriuOpts,
	data *CheckpointMetadata,
	state *containerState,
	checkpointDir string,
) (time.Duration, error) {
	// Execute CRIU dump
	criuDumpStart := time.Now()
	criuClient := criu.MakeCriu()
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		c.log.WithField("duration", time.Since(criuDumpStart)).Error("CRIU dump failed")
		c.logDumpFailureArtifacts(checkpointDir)
		return 0, fmt.Errorf("CRIU dump failed: %w", err)
	}
	criuDumpDuration := time.Since(criuDumpStart)
	c.log.WithField("duration", criuDumpDuration).Info("CRIU dump completed")

	// Capture /dev/shm contents (must happen after dump for final process state)
	if err := CaptureDevShm(state.PID, checkpointDir, c.log); err != nil {
		c.log.WithError(err).Warn("Failed to capture /dev/shm contents")
	}

	// Capture rootfs diff and deleted files
	CaptureRootfsState(state.UpperDir, checkpointDir, data, c.log)

	return criuDumpDuration, nil
}

func (c *Checkpointer) logDumpFailureArtifacts(checkpointDir string) {
	dumpLogPath := filepath.Join(checkpointDir, DumpLogFilename)
	dumpLog, err := os.ReadFile(dumpLogPath)
	if err != nil {
		c.log.WithError(err).WithField("path", dumpLogPath).Warn("Could not read CRIU dump log")
	} else {
		c.log.Error("=== CRIU DUMP LOG START ===")
		c.log.Error(string(dumpLog))
		c.log.Error("=== CRIU DUMP LOG END ===")
	}

	entries, err := os.ReadDir(checkpointDir)
	if err != nil {
		c.log.WithError(err).WithField("path", checkpointDir).Warn("Could not list checkpoint directory")
		return
	}

	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		info, infoErr := entry.Info()
		if infoErr != nil {
			files = append(files, entry.Name())
			continue
		}
		files = append(files, fmt.Sprintf("%s (%d bytes)", entry.Name(), info.Size()))
	}

	c.log.WithFields(logrus.Fields{
		"checkpoint_dir": checkpointDir,
		"files":          files,
	}).Error("Checkpoint artifacts present after CRIU dump failure")
}
