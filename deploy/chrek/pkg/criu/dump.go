package criu

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	criulib "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

const (
	dumpLogFilename = "dump.log"
	criuConfFilename = "criu.conf"
)

// BuildDumpOptions creates CRIU options from the container snapshot and settings.
// It also writes the criu.conf file for options that cannot be passed via RPC.
// The ImagesDirFd is left unset — ExecuteDump opens it at dump time.
func BuildDumpOptions(
	state *types.CheckpointContainerSnapshot,
	settings *types.CRIUSettings,
	checkpointDir string,
	log logr.Logger,
) (*criurpc.CriuOpts, error) {
	var maskedPaths []string
	if state.OCISpec != nil && state.OCISpec.Linux != nil {
		maskedPaths = state.OCISpec.Linux.MaskedPaths
	}

	externalized, skipped := common.BuildMountPolicy(state.Mounts, state.RootFS, maskedPaths)
	log.V(1).Info("Resolved mount policy for CRIU dump",
		"externalized_count", len(externalized),
		"skipped_count", len(skipped),
	)

	criuOpts := &criurpc.CriuOpts{
		Pid:     proto.Int32(int32(state.PID)),
		Root:    proto.String(state.RootFS),
		LogFile: proto.String(dumpLogFilename),
		// Always externalize network namespace
		External: []string{fmt.Sprintf("net[%d]:extNetNs", state.NetNSInode)},
	}
	criuOpts.ExtMnt = toExtMountMaps(externalized)
	criuOpts.SkipMnt = skipped

	if state.HostCgroupPath != "" {
		criuOpts.FreezeCgroup = proto.String(state.HostCgroupPath)
	}

	if settings == nil {
		return criuOpts, nil
	}

	if err := applyCommonSettings(criuOpts, settings); err != nil {
		return nil, err
	}

	// Dump-only options
	criuOpts.LeaveRunning = proto.Bool(settings.LeaveRunning)
	criuOpts.OrphanPtsMaster = proto.Bool(settings.OrphanPtsMaster)
	criuOpts.ExtMasters = proto.Bool(settings.ExtMasters)
	criuOpts.AutoDedup = proto.Bool(settings.AutoDedup)
	criuOpts.LazyPages = proto.Bool(settings.LazyPages)

	if settings.GhostLimit > 0 {
		criuOpts.GhostLimit = proto.Uint32(settings.GhostLimit)
	}

	// Write criu.conf for options that cannot be passed via RPC.
	if confContent := buildCRIUConf(settings); confContent != "" {
		confPath := filepath.Join(checkpointDir, criuConfFilename)
		if err := os.WriteFile(confPath, []byte(confContent), 0644); err != nil {
			return nil, fmt.Errorf("failed to write criu.conf: %w", err)
		}
		criuOpts.ConfigFile = proto.String(confPath)
	}

	return criuOpts, nil
}

// ExecuteDump opens the image directory FD, runs the CRIU dump, and cleans up.
func ExecuteDump(
	criuOpts *criurpc.CriuOpts,
	checkpointDir string,
	settings *types.CRIUSettings,
	log logr.Logger,
) (time.Duration, error) {
	imageDir, imageDirFD, err := openPathForCRIU(checkpointDir)
	if err != nil {
		return 0, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()
	criuOpts.ImagesDirFd = proto.Int32(imageDirFD)

	criuDumpStart := time.Now()
	criuClient := criulib.MakeCriu()
	if settings != nil && strings.TrimSpace(settings.BinaryPath) != "" {
		if _, err := os.Stat(settings.BinaryPath); err != nil {
			return 0, fmt.Errorf("criu binary not found at %s: %w", settings.BinaryPath, err)
		}
		criuClient.SetCriuPath(settings.BinaryPath)
	}
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		dumpDuration := time.Since(criuDumpStart)
		log.Error(err, "CRIU dump failed",
			"duration", dumpDuration,
			"checkpoint_dir", checkpointDir,
			"dump_log_path", fmt.Sprintf("%s/%s", checkpointDir, dumpLogFilename),
		)
		return 0, fmt.Errorf("CRIU dump failed: %w", err)
	}

	criuDumpDuration := time.Since(criuDumpStart)
	log.Info("CRIU dump completed", "duration", criuDumpDuration)
	return criuDumpDuration, nil
}

func buildCRIUConf(c *types.CRIUSettings) string {
	if c == nil {
		return ""
	}
	var content string
	if c.LibDir != "" {
		content += "libdir " + c.LibDir + "\n"
	}
	if c.AllowUprobes {
		content += "allow-uprobes\n"
	}
	if c.SkipInFlight {
		content += "skip-in-flight\n"
	}
	return content
}
