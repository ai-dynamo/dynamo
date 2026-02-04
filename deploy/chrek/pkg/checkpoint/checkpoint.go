// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// CheckpointParams holds per-checkpoint identifiers for a checkpoint operation.
// This is separate from config.CheckpointConfig which holds static CRIU settings.
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
	Data          *config.CheckpointData
}

// Checkpointer performs CRIU checkpoint operations
type Checkpointer struct {
	discoveryClient *checkpointk8s.DiscoveryClient
	k8sClient       *checkpointk8s.K8sClient // Optional: for accurate volume type discovery from K8s API
	hostProc        string
	log             *logrus.Entry
}

// NewCheckpointer creates a new checkpointer
func NewCheckpointer(discoveryClient *checkpointk8s.DiscoveryClient, hostProc string) *Checkpointer {
	if hostProc == "" {
		hostProc = os.Getenv("HOST_PROC")
		if hostProc == "" {
			hostProc = "/proc"
		}
	}
	return &Checkpointer{
		discoveryClient: discoveryClient,
		hostProc:        hostProc,
		log:             logrus.WithField("component", "checkpointer"),
	}
}

// WithK8sClient sets an optional Kubernetes client for accurate volume type discovery.
// When set, volume types are fetched from the K8s API instead of being inferred from paths.
func (c *Checkpointer) WithK8sClient(client *checkpointk8s.K8sClient) *Checkpointer {
	c.k8sClient = client
	return c
}

// Checkpoint performs a CRIU dump of a container
func (c *Checkpointer) Checkpoint(ctx context.Context, params CheckpointParams, cfg *config.CheckpointConfig) (*Result, error) {
	checkpointStart := time.Now()
	c.log.Info("=== Starting checkpoint operation ===")

	// 1. Resolve container to get PID
	resolveStart := time.Now()
	containerInfo, err := c.discoveryClient.ResolveContainer(ctx, params.ContainerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}
	pid := int(containerInfo.PID)
	c.log.WithField("duration", time.Since(resolveStart)).Info("Container resolution completed")

	// 2. Create checkpoint directory
	checkpointDir := config.GetCheckpointDir(params.CheckpointDir, params.CheckpointID)
	if err := os.MkdirAll(checkpointDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// 3. Introspect container state
	introspectStart := time.Now()
	rootFS, err := GetRootFS(pid, c.hostProc)
	if err != nil {
		return nil, fmt.Errorf("failed to get rootfs: %w", err)
	}
	mounts, err := GetKubernetesVolumeMounts(pid, c.hostProc)
	if err != nil {
		return nil, fmt.Errorf("failed to get mounts: %w", err)
	}
	namespaces, err := GetAllNamespaces(pid, c.hostProc)
	if err != nil {
		return nil, fmt.Errorf("failed to get namespaces: %w", err)
	}
	c.log.WithField("duration", time.Since(introspectStart)).Info("Container introspection completed")

	// 4. Open image directory FD
	imageDir, imageDirFD, err := OpenImageDir(checkpointDir)
	if err != nil {
		return nil, err
	}
	defer imageDir.Close()

	// 5. Build CRIU options from config
	criuParams := CRIUDumpParams{
		PID:        pid,
		ImageDirFD: imageDirFD,
		RootFS:     rootFS,
	}
	criuOpts := BuildCRIUOpts(&cfg.CRIU, criuParams)

	// 6. Create CRIU config file if needed (for options not available via RPC)
	if cfg.CRIU.NeedsCRIUConfFile() {
		if err := cfg.CRIU.Validate(); err != nil {
			return nil, err
		}
		configPath := filepath.Join(checkpointDir, "criu.conf")
		configContent := cfg.CRIU.GenerateCRIUConfContent()
		if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
			return nil, fmt.Errorf("failed to write CRIU config file: %w", err)
		}
		criuOpts.ConfigFile = proto.String(configPath)
		c.log.WithFields(logrus.Fields{
			"config_path": configPath,
			"lib_dir":     cfg.CRIU.LibDir,
		}).Info("Created CRIU config file")
	}

	// 7. Configure external mounts and namespaces
	if err := ConfigureExternalMounts(criuOpts, pid, c.hostProc, containerInfo); err != nil {
		return nil, err
	}
	netNsInode := ConfigureExternalNamespaces(criuOpts, namespaces, cfg.CRIU.ExternalMounts)
	if netNsInode > 0 {
		c.log.WithField("inode", netNsInode).Debug("Marked network namespace as external")
	}
	for _, extMount := range cfg.CRIU.ExternalMounts {
		c.log.WithField("external", extMount).Debug("Added external mount mapping")
	}

	// 8. Get overlay upperdir for rootfs diff capture
	upperDir, upperDirErr := GetOverlayUpperDir(pid, c.hostProc)
	if upperDirErr != nil {
		c.log.WithError(upperDirErr).Warn("Could not get overlay upperdir - rootfs diff will not be captured")
	} else {
		c.log.WithField("upperdir", upperDir).Debug("Found overlay upperdir")
	}

	// 9. Build checkpoint data with config and container state
	metaCfg := MetadataBuilderConfig{
		CheckpointID:  params.CheckpointID,
		NodeName:      params.NodeName,
		ContainerID:   params.ContainerID,
		ContainerName: params.ContainerName,
		PodName:       params.PodName,
		PodNamespace:  params.PodNamespace,
		PID:           pid,
	}
	data := BuildCheckpointData(ctx, metaCfg, cfg, containerInfo, mounts, namespaces, c.k8sClient, c.log)
	if upperDir != "" {
		data.UpperDir = upperDir
	}

	// Populate external mounts from CRIU options (computed by ConfigureExternalNamespaces)
	// These are needed for restore to use the exact same external mount mappings
	data.CRIU.ExternalMounts = cfg.CRIU.ExternalMounts

	if err := config.SaveCheckpointData(checkpointDir, data); err != nil {
		return nil, fmt.Errorf("failed to save checkpoint data: %w", err)
	}

	// 10. Remove semaphores from /dev/shm before checkpoint
	// Semaphores cause CRIU restore to fail with "Can't link dev/shm/link_remap.X -> dev/shm/sem.Y"
	RemoveSemaphores(pid, c.hostProc, c.log)

	// 11. Execute CRIU dump via go-criu
	criuDumpStart := time.Now()
	criuClient := criu.MakeCriu()
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		c.log.WithField("duration", time.Since(criuDumpStart)).Error("CRIU dump failed")
		return nil, fmt.Errorf("CRIU dump failed: %w", err)
	}
	criuDumpDuration := time.Since(criuDumpStart)
	c.log.WithField("duration", criuDumpDuration).Info("CRIU dump completed successfully")

	// 12. Capture /dev/shm contents (excluding semaphores)
	// This must happen after CRIU dump since we want the final process state
	shmCaptureStart := time.Now()
	if err := CaptureDevShm(pid, c.hostProc, checkpointDir, c.log); err != nil {
		c.log.WithError(err).Warn("Failed to capture /dev/shm contents")
	}
	c.log.WithField("duration", time.Since(shmCaptureStart)).Info("/dev/shm capture completed")

	// 13. Capture rootfs diff and deleted files
	rootfsCaptureStart := time.Now()
	CaptureRootfsState(upperDir, checkpointDir, data, c.log)
	c.log.WithField("duration", time.Since(rootfsCaptureStart)).Info("Rootfs capture completed")

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

