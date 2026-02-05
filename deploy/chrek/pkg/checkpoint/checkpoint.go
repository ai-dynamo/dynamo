// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// Options configures the checkpoint operation
type Options struct {
	ContainerID   string
	ContainerName string // K8s container name (for K8s API volume type lookup)
	CheckpointID  string
	CheckpointDir string
	NodeName      string
	PodName       string
	PodNamespace  string

	// CRIU options (from environment variables)
	GhostLimit uint32 // From CRIU_GHOST_LIMIT: ghost file size limit in bytes (0 = CRIU default)
	Timeout    uint32 // From CRIU_TIMEOUT: timeout in seconds (0 = no timeout)

	// GPU/CUDA checkpoint options
	CUDAPluginDir  string   // Path to CRIU CUDA plugin (e.g., /home/mmshin/work/criu/plugins/cuda)
	ExternalMounts []string // Additional external mount mappings (e.g., "mnt[path]:path")
}

// Result contains the result of a checkpoint operation
type Result struct {
	CheckpointID  string
	CheckpointDir string
	Metadata      *common.CheckpointMetadata
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
func (c *Checkpointer) Checkpoint(ctx context.Context, opts Options) (*Result, error) {
	checkpointStart := time.Now()
	c.log.Info("=== Starting checkpoint operation ===")

	// 1. Resolve container to get PID
	resolveStart := time.Now()
	containerInfo, err := c.discoveryClient.ResolveContainer(ctx, opts.ContainerID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve container: %w", err)
	}
	pid := int(containerInfo.PID)
	c.log.WithField("duration", time.Since(resolveStart)).Info("Container resolution completed")

	// 2. Create checkpoint directory
	checkpointDir := common.GetCheckpointDir(opts.CheckpointDir, opts.CheckpointID)
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

	// 5. Build CRIU options
	criuOpts := BuildCRIUOptsFromCheckpointOpts(opts, pid, imageDirFD, rootFS)

	// 6. Create CRIU config file for CUDA plugin (libdir is not available via RPC)
	if opts.CUDAPluginDir != "" {
		if opts.Timeout == 0 {
			return nil, fmt.Errorf("CRIU_TIMEOUT environment variable must be set for CUDA checkpoints")
		}
		configPath := filepath.Join(checkpointDir, "criu.conf")
		configContent := fmt.Sprintf(`enable-external-masters
libdir %s
tcp-close
link-remap
timeout %d
allow-uprobes
skip-in-flight
`, opts.CUDAPluginDir, opts.Timeout)
		if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
			return nil, fmt.Errorf("failed to write CRIU config file: %w", err)
		}
		criuOpts.ConfigFile = proto.String(configPath)
		c.log.WithFields(logrus.Fields{
			"config_path": configPath,
			"plugin_dir":  opts.CUDAPluginDir,
		}).Info("Created CRIU config file for CUDA plugin")
	}

	// 7. Configure external mounts and namespaces
	if err := ConfigureExternalMounts(criuOpts, pid, c.hostProc, containerInfo); err != nil {
		return nil, err
	}
	netNsInode := ConfigureExternalNamespaces(criuOpts, namespaces, opts.ExternalMounts)
	if netNsInode > 0 {
		c.log.WithField("inode", netNsInode).Debug("Marked network namespace as external")
	}
	for _, extMount := range opts.ExternalMounts {
		c.log.WithField("external", extMount).Debug("Added external mount mapping")
	}

	// 8. Get overlay upperdir for rootfs diff capture
	upperDir, upperDirErr := GetOverlayUpperDir(pid, c.hostProc)
	if upperDirErr != nil {
		c.log.WithError(upperDirErr).Warn("Could not get overlay upperdir - rootfs diff will not be captured")
	} else {
		c.log.WithField("upperdir", upperDir).Debug("Found overlay upperdir")
	}

	// 9. Build and save initial metadata before dump
	metaCfg := MetadataBuilderConfig{
		CheckpointID:  opts.CheckpointID,
		NodeName:      opts.NodeName,
		ContainerID:   opts.ContainerID,
		ContainerName: opts.ContainerName,
		PodName:       opts.PodName,
		PodNamespace:  opts.PodNamespace,
		PID:           pid,
		CUDAPluginDir: opts.CUDAPluginDir,
	}
	meta := BuildCheckpointMetadata(ctx, metaCfg, containerInfo, mounts, namespaces, c.k8sClient, c.log)
	if upperDir != "" {
		meta.UpperDir = upperDir
	}
	if err := common.SaveMetadata(checkpointDir, meta); err != nil {
		return nil, fmt.Errorf("failed to save metadata: %w", err)
	}

	// 10. Remove semaphores from /dev/shm before checkpoint
	// Semaphores cause CRIU restore to fail with "Can't link dev/shm/link_remap.X -> dev/shm/sem.Y"
	if err := c.removeSemaphores(pid); err != nil {
		return nil, fmt.Errorf("failed to remove semaphores: %w", err)
	}

	// 11. Execute CRIU dump via go-criu
	criuDumpStart := time.Now()
	criuClient := criu.MakeCriu()
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		c.log.WithField("duration", time.Since(criuDumpStart)).Error("CRIU dump failed")
		return nil, fmt.Errorf("CRIU dump failed: %w", err)
	}
	criuDumpDuration := time.Since(criuDumpStart)
	c.log.WithField("duration", criuDumpDuration).Info("CRIU dump completed successfully")

	// 12. Capture rootfs diff and deleted files
	rootfsCaptureStart := time.Now()
	CaptureRootfsState(upperDir, checkpointDir, meta, c.log)
	c.log.WithField("duration", time.Since(rootfsCaptureStart)).Info("Rootfs capture completed")

	totalDuration := time.Since(checkpointStart)
	c.log.WithFields(logrus.Fields{
		"total_duration":     totalDuration,
		"criu_dump_duration": criuDumpDuration,
	}).Info("=== Checkpoint operation completed ===")

	return &Result{
		CheckpointID:  opts.CheckpointID,
		CheckpointDir: checkpointDir,
		Metadata:      meta,
	}, nil
}

// removeSemaphores removes POSIX semaphores from the container's /dev/shm.
// Semaphores can cause issues during CRIU checkpoint/restore because they
// maintain kernel state that may not transfer correctly between processes.
// This accesses the container's filesystem via /proc/<pid>/root/dev/shm/.
func (c *Checkpointer) removeSemaphores(pid int) error {
	shmPath := filepath.Join(c.hostProc, fmt.Sprintf("%d/root/dev/shm", pid))

	entries, err := os.ReadDir(shmPath)
	if err != nil {
		// It's okay if /dev/shm doesn't exist (container may not have it)
		c.log.WithError(err).Debug("Could not read container /dev/shm (may not exist)")
		return nil
	}

	var removed []string
	var errors []error
	for _, entry := range entries {
		name := entry.Name()
		if strings.HasPrefix(name, "sem.") {
			semPath := filepath.Join(shmPath, name)
			if err := os.Remove(semPath); err != nil {
				c.log.WithError(err).WithField("semaphore", name).Error("Failed to remove semaphore")
				errors = append(errors, fmt.Errorf("failed to remove semaphore %s: %w", name, err))
			} else {
				removed = append(removed, name)
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to remove %d semaphore(s): %v", len(errors), errors)
	}

	if len(removed) > 0 {
		c.log.WithFields(logrus.Fields{
			"count":      len(removed),
			"semaphores": removed,
		}).Info("Removed semaphores from container /dev/shm before checkpoint")
	} else {
		c.log.Debug("No semaphores found in container /dev/shm")
	}

	return nil
}
