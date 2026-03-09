package common

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

)

const HostCgroupPath = "/sys/fs/cgroup"

// ResolveCgroupRootFromHostPID reads the unified cgroup v2 path for a PID via /host/proc.
func ResolveCgroupRootFromHostPID(pid int) (string, error) {
	cgroupFile := filepath.Join(HostProcPath, strconv.Itoa(pid), "cgroup")
	data, err := os.ReadFile(cgroupFile)
	if err != nil {
		return "", fmt.Errorf("failed reading %s: %w", cgroupFile, err)
	}

	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "0::") {
			continue
		}
		path := strings.TrimPrefix(line, "0::")
		if path == "" {
			return "/", nil
		}
		if !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		return filepath.Clean(path), nil
	}

	return "", fmt.Errorf("unified cgroup entry not found in %s", cgroupFile)
}
