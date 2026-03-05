package criu

import (
	"fmt"
	"os"
	"strings"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// parseManageCgroupsMode normalizes and validates the CRIU cgroup mode setting.
func parseManageCgroupsMode(raw string) (criurpc.CriuCgMode, string, error) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	switch mode {
	case "":
		// Default to SOFT when unset (matches Helm default of "soft")
		return criurpc.CriuCgMode_SOFT, "soft", nil
	case "ignore":
		return criurpc.CriuCgMode_IGNORE, "ignore", nil
	case "soft":
		return criurpc.CriuCgMode_SOFT, mode, nil
	case "full":
		return criurpc.CriuCgMode_FULL, mode, nil
	case "strict":
		return criurpc.CriuCgMode_STRICT, mode, nil
	default:
		return criurpc.CriuCgMode_IGNORE, "", fmt.Errorf("invalid manageCgroupsMode %q", raw)
	}
}

func shouldSetCgroupRoot(cgMode criurpc.CriuCgMode) bool {
	switch cgMode {
	case criurpc.CriuCgMode_SOFT, criurpc.CriuCgMode_FULL, criurpc.CriuCgMode_STRICT:
		return true
	default:
		return false
	}
}

// applyCommonSettings sets CRIU options shared between dump and restore.
func applyCommonSettings(opts *criurpc.CriuOpts, settings *types.CRIUSettings) error {
	opts.LogLevel = proto.Int32(settings.LogLevel)
	opts.ShellJob = proto.Bool(settings.ShellJob)
	opts.TcpClose = proto.Bool(settings.TcpClose)
	opts.FileLocks = proto.Bool(settings.FileLocks)
	opts.ExtUnixSk = proto.Bool(settings.ExtUnixSk)
	opts.LinkRemap = proto.Bool(settings.LinkRemap)

	opts.ManageCgroups = proto.Bool(true)
	cgMode, _, err := parseManageCgroupsMode(settings.ManageCgroupsMode)
	if err != nil {
		return fmt.Errorf("invalid cgroup mode: %w", err)
	}
	opts.ManageCgroupsMode = &cgMode
	return nil
}

// openPathForCRIU opens a path (directory or file) and clears the CLOEXEC flag
// so the FD can be inherited by CRIU child processes.
// Returns the opened file and its FD. Caller must close the file when done.
// The caller must also retain the *os.File reference for the entire lifetime the
// raw FD is in use — if the *os.File is garbage collected, Go's finalizer will
// close the underlying FD.
func openPathForCRIU(path string) (*os.File, int32, error) {
	dir, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to open %s: %w", path, err)
	}

	// Clear CLOEXEC so the FD is inherited by CRIU child process.
	// Go's os.Open() sets O_CLOEXEC by default, but go-criu's swrk mode
	// requires the FD to be inherited.
	if _, err := unix.FcntlInt(dir.Fd(), unix.F_SETFD, 0); err != nil {
		dir.Close()
		return nil, 0, fmt.Errorf("failed to clear CLOEXEC on %s: %w", path, err)
	}

	return dir, int32(dir.Fd()), nil
}

// toExtMountMaps converts the mount policy's externalized map to CRIU protobuf entries.
func toExtMountMaps(extMap map[string]string) []*criurpc.ExtMountMap {
	entries := make([]*criurpc.ExtMountMap, 0, len(extMap))
	for key, val := range extMap {
		entries = append(entries, &criurpc.ExtMountMap{
			Key: proto.String(key),
			Val: proto.String(val),
		})
	}
	return entries
}

