package executor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func TestPromoteCheckpointRejectsCanceledContext(t *testing.T) {
	t.Run("before replacement", func(t *testing.T) {
		root := t.TempDir()
		staging := filepath.Join(root, "staging")
		final := filepath.Join(root, "final")
		if err := os.Mkdir(staging, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(final, 0700); err != nil {
			t.Fatal(err)
		}
		old := filepath.Join(final, "old")
		if err := os.WriteFile(old, []byte("old"), 0600); err != nil {
			t.Fatal(err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := promoteCheckpoint(ctx, staging, final)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("promoteCheckpoint() error = %v, want context.Canceled", err)
		}
		if _, err := os.Stat(old); err != nil {
			t.Fatalf("previous checkpoint was changed: %v", err)
		}
		if _, err := os.Stat(staging); err != nil {
			t.Fatalf("staged checkpoint was promoted: %v", err)
		}
	})

	t.Run("immediately before rename", func(t *testing.T) {
		root := t.TempDir()
		staging := filepath.Join(root, "staging")
		final := filepath.Join(root, "final")
		if err := os.Mkdir(staging, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(final, 0700); err != nil {
			t.Fatal(err)
		}
		ctx := newCancelWhenMissingContext(final)
		err := promoteCheckpoint(ctx, staging, final)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("promoteCheckpoint() error = %v, want context.Canceled", err)
		}
		if _, err := os.Stat(staging); err != nil {
			t.Fatalf("staged checkpoint was promoted: %v", err)
		}
		if _, err := os.Stat(final); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("checkpoint was published after cancellation: %v", err)
		}
	})
}

type cancelWhenMissingContext struct {
	context.Context
	path   string
	cancel context.CancelFunc
	once   sync.Once
}

func newCancelWhenMissingContext(path string) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	return &cancelWhenMissingContext{
		Context: ctx,
		path:    path,
		cancel:  cancel,
	}
}

func (c *cancelWhenMissingContext) Err() error {
	if _, err := os.Lstat(c.path); errors.Is(err, os.ErrNotExist) {
		c.once.Do(c.cancel)
	}
	return c.Context.Err()
}
