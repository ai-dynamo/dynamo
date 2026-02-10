// Package common provides shared utilities for CRIU operations.
package common

import (
	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"
)

// SetCommonK8sFlags sets CRIU flags that are always-on for K8s environments.
// These flags are shared between checkpoint and restore operations.
func SetCommonK8sFlags(opts *criurpc.CriuOpts) {
	opts.ShellJob = proto.Bool(true)   // Containers are often session leaders
	opts.TcpClose = proto.Bool(true)   // Pod IPs change on restore/migration
	opts.FileLocks = proto.Bool(true)  // Applications use file locks
	opts.ExtUnixSk = proto.Bool(true)  // Containers have external Unix sockets
	opts.ManageCgroups = proto.Bool(true)
}
