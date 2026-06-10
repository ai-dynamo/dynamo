package nccl

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
)

func TestRunProcessTreeActionRunsHooksConcurrently(t *testing.T) {
	procRoot, err := os.MkdirTemp("/tmp", "dhook")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(procRoot)
	})
	var connected atomic.Int32
	release := make(chan struct{})

	for _, pid := range []int{101, 102} {
		controlDir := filepath.Join(procRoot, strconv.Itoa(pid), "root", "snapshot-control", "snapshot-hook")
		if err := os.MkdirAll(controlDir, 0o755); err != nil {
			t.Fatal(err)
		}
		listener, err := net.Listen("unix", filepath.Join(controlDir, fmt.Sprintf("%d.sock", pid)))
		if err != nil {
			t.Fatal(err)
		}
		defer listener.Close()

		go func(listener net.Listener) {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			defer conn.Close()
			if _, err := bufio.NewReader(conn).ReadString('\n'); err != nil {
				return
			}
			connected.Add(1)
			<-release
			_, _ = conn.Write([]byte("ok 0\n"))
		}(listener)
	}

	done := make(chan error, 1)
	go func() {
		_, err := runProcessTreeActionWithProcRoot(context.Background(), []int{101, 102}, procRoot, actionPrepare, logr.Discard())
		done <- err
	}()

	deadline := time.After(2 * time.Second)
	for connected.Load() != 2 {
		select {
		case err := <-done:
			if err != nil {
				t.Fatalf("action returned early: %v", err)
			}
			t.Fatal("action completed before both hook servers were connected")
		case <-deadline:
			t.Fatalf("timed out waiting for concurrent hook connections; connected=%d", connected.Load())
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}

	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("action failed: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for hook action to complete")
	}
}
