package criu

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

const nvidiaDevicePrefix = "dev["

func dumpNVIDIAExternalDevices(rootFS string) ([]string, error) {
	patterns := []string{
		filepath.Join(rootFS, "dev", "nvidia*"),
		filepath.Join(rootFS, "dev", "nvidia-caps", "nvidia-cap*"),
	}

	seen := map[string]struct{}{}
	var external []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, fmt.Errorf("failed to glob NVIDIA device pattern %s: %w", pattern, err)
		}
		for _, path := range matches {
			info, err := os.Lstat(path)
			if err != nil {
				continue
			}
			if info.Mode()&os.ModeCharDevice == 0 {
				continue
			}
			stat, ok := info.Sys().(*syscall.Stat_t)
			if !ok {
				continue
			}

			name := filepath.Base(path)
			spec := fmt.Sprintf("dev[%d/%d]:%s", unix.Major(uint64(stat.Rdev)), unix.Minor(uint64(stat.Rdev)), name)
			if _, ok := seen[spec]; ok {
				continue
			}
			seen[spec] = struct{}{}
			external = append(external, spec)
		}
	}

	sort.Strings(external)
	return external, nil
}

func restoreNVIDIAExternalDevices(dumpExternal []string) []string {
	var external []string
	for _, spec := range dumpExternal {
		name, ok := parseDumpExternalDeviceName(spec)
		if !ok || !isNVIDIADeviceName(name) {
			continue
		}

		devicePath := filepath.Join("/dev", name)
		if strings.HasPrefix(name, "nvidia-cap") {
			devicePath = filepath.Join("/dev/nvidia-caps", name)
		}
		external = append(external, fmt.Sprintf("dev[%s]:%s", name, devicePath))
	}

	sort.Strings(external)
	return external
}

func parseDumpExternalDeviceName(spec string) (string, bool) {
	if !strings.HasPrefix(spec, nvidiaDevicePrefix) {
		return "", false
	}
	_, name, ok := strings.Cut(spec, "]:")
	if !ok || name == "" || strings.Contains(name, "/") {
		return "", false
	}
	return name, true
}

func isNVIDIADeviceName(name string) bool {
	if name == "nvidiactl" || name == "nvidia-uvm" || name == "nvidia-uvm-tools" || name == "nvidia-modeset" {
		return true
	}
	if strings.HasPrefix(name, "nvidia-cap") {
		return true
	}
	if !strings.HasPrefix(name, "nvidia") {
		return false
	}
	for _, r := range strings.TrimPrefix(name, "nvidia") {
		if r < '0' || r > '9' {
			return false
		}
	}
	return len(name) > len("nvidia")
}
