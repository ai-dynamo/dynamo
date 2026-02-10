package common

import (
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
)

// AssertCommonK8sFlags verifies that common K8s CRIU flags are set correctly.
// These flags are shared between checkpoint and restore operations.
func AssertCommonK8sFlags(t *testing.T, opts *criurpc.CriuOpts) {
	t.Helper()

	if !opts.GetShellJob() {
		t.Error("ShellJob should be true")
	}
	if !opts.GetTcpClose() {
		t.Error("TcpClose should be true")
	}
	if !opts.GetFileLocks() {
		t.Error("FileLocks should be true")
	}
	if !opts.GetExtUnixSk() {
		t.Error("ExtUnixSk should be true")
	}
	if !opts.GetManageCgroups() {
		t.Error("ManageCgroups should be true")
	}
}
