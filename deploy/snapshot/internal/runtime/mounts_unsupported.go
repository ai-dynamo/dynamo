//go:build !linux

package runtime

import (
	"fmt"
	"runtime"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// ReadMountInfo is unsupported outside Linux.
func ReadMountInfo(_ int) ([]types.MountInfo, error) {
	return nil, fmt.Errorf("reading mountinfo is not supported on %s", runtime.GOOS)
}

// RemountProcSys is unsupported outside Linux.
func RemountProcSys(_ bool) error {
	return fmt.Errorf("remounting /proc/sys is not supported on %s", runtime.GOOS)
}
