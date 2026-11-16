/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package history

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// NewLocal stores history as files under dir. It is the development and test backend;
// a Deployment using it needs a PersistentVolume to keep history across restarts.
func NewLocal(dir string) (Store, error) {
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("create history dir: %w", err)
	}
	return &kvStore{kv: &localKV{root: dir}}, nil
}

type localKV struct {
	root string
}

// resolve rejects keys that would escape the root.
func (l *localKV) resolve(key string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(key))
	if clean == "." || strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) {
		return "", fmt.Errorf("invalid history key %q", key)
	}
	return filepath.Join(l.root, clean), nil
}

func (l *localKV) Put(_ context.Context, key string, data []byte) error {
	file, err := l.resolve(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(file), 0o750); err != nil {
		return err
	}
	tmp := file + ".tmp"
	if err := os.WriteFile(tmp, data, 0o640); err != nil {
		return err
	}
	return os.Rename(tmp, file)
}

func (l *localKV) Get(_ context.Context, key string) ([]byte, error) {
	file, err := l.resolve(key)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(file)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, ErrNotFound
	}
	return data, err
}

func (l *localKV) List(_ context.Context, prefix string) ([]string, error) {
	var keys []string
	err := filepath.WalkDir(l.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || strings.HasSuffix(p, ".tmp") {
			return nil
		}
		rel, err := filepath.Rel(l.root, p)
		if err != nil {
			return err
		}
		key := filepath.ToSlash(rel)
		if strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
		return nil
	})
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return keys, err
}
