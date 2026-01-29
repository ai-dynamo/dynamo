// criu.go provides shared CRIU utilities used by both checkpoint and restore.
package common

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"golang.org/x/sys/unix"
)

// OpenDirForCRIU opens a directory and clears the CLOEXEC flag so the FD
// can be inherited by CRIU child processes.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenDirForCRIU(path string) (*os.File, int32, error) {
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

// DefaultMaskedPaths returns the standard OCI masked paths.
// These paths are typically masked (made inaccessible) in containers.
// Used as fallback when checkpoint metadata doesn't include OCI-derived paths.
func DefaultMaskedPaths() []string {
	return []string{
		"/proc/bus",
		"/proc/fs",
		"/proc/irq",
		"/proc/sys",
		"/proc/sysrq-trigger",
		"/proc/acpi",
		"/proc/kcore",
		"/proc/keys",
		"/proc/latency_stats",
		"/proc/timer_list",
		"/proc/scsi",
		"/proc/interrupts",
		"/proc/asound",
		"/sys/firmware",
		"/sys/devices/virtual/powercap",
	}
}

// DefaultReadonlyPaths returns the standard OCI readonly paths.
// These paths are typically mounted read-only in containers.
func DefaultReadonlyPaths() []string {
	return []string{
		"/proc/bus",
		"/proc/fs",
		"/proc/irq",
		"/proc/sys",
		"/proc/sysrq-trigger",
	}
}

// CRIUMountPoint represents a parsed mount point from /proc/pid/mountinfo.
type CRIUMountPoint struct {
	MountID   string // Mount ID
	ParentID  string // Parent mount ID
	Path      string // Mount point path (container-side)
	Root      string // Root within filesystem (host-side for bind mounts)
	FSType    string // Filesystem type
	Source    string // Mount source
	Options   string // Mount options
	SuperOpts string // Super block options
}

// ParseMountInfoFile parses a mountinfo file and returns all mount points.
// This is used by both checkpoint (to mark mounts as external) and
// restore (to generate external mount mappings).
func ParseMountInfoFile(path string) ([]CRIUMountPoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	var mounts []CRIUMountPoint
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		mount, err := parseCRIUMountInfoLine(scanner.Text())
		if err != nil {
			continue // Skip malformed lines
		}
		mounts = append(mounts, mount)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading mountinfo: %w", err)
	}

	return mounts, nil
}

// GetMountPointPaths returns just the mount point paths from a mountinfo file.
// This is a convenience function when you only need the paths.
func GetMountPointPaths(path string) ([]string, error) {
	mounts, err := ParseMountInfoFile(path)
	if err != nil {
		return nil, err
	}

	paths := make([]string, 0, len(mounts))
	for _, m := range mounts {
		paths = append(paths, m.Path)
	}
	return paths, nil
}

// parseCRIUMountInfoLine parses a single line from mountinfo.
// mountinfo format:
// 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
// (1)(2)(3)   (4)   (5)      (6)     (7)   (8) (9)   (10)         (11)
func parseCRIUMountInfoLine(line string) (CRIUMountPoint, error) {
	fields := strings.Fields(line)
	if len(fields) < 10 {
		return CRIUMountPoint{}, fmt.Errorf("malformed mountinfo line")
	}

	// Find separator (-) to get fstype and source
	sepIdx := -1
	for i, f := range fields {
		if f == "-" {
			sepIdx = i
			break
		}
	}

	if sepIdx == -1 || sepIdx+2 >= len(fields) {
		return CRIUMountPoint{}, fmt.Errorf("malformed mountinfo line (no separator)")
	}

	superOpts := ""
	if sepIdx+3 < len(fields) {
		superOpts = fields[sepIdx+3]
	}

	return CRIUMountPoint{
		MountID:   fields[0],
		ParentID:  fields[1],
		Root:      fields[3],
		Path:      fields[4],
		Options:   fields[5],
		FSType:    fields[sepIdx+1],
		Source:    fields[sepIdx+2],
		SuperOpts: superOpts,
	}, nil
}
