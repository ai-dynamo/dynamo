/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdoverride"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"
)

type options struct {
	blueprintPath string
	overridePath  string
	outputPath    string
	installPath   string
}

func main() {
	if err := run(os.Args[1:], os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintf(os.Stderr, "dgd-apply-overrides: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stderr io.Writer) error {
	opts, err := parseOptions(args, stderr)
	if err != nil {
		return err
	}
	if opts.installPath != "" {
		if err := installSelf(opts.installPath); err != nil {
			return fmt.Errorf("install dgd-apply-overrides to %q: %w", opts.installPath, err)
		}
		return nil
	}

	blueprint, err := readDGD(opts.blueprintPath, "blueprint")
	if err != nil {
		return err
	}
	override, err := readDGD(opts.overridePath, "override")
	if err != nil {
		return err
	}

	effective, warnings, err := dgdoverride.Apply(blueprint, override)
	if err != nil {
		return fmt.Errorf("apply DGD override: %w", err)
	}
	for _, warning := range warnings {
		if _, err := fmt.Fprintf(stderr, "warning: %s\n", warning); err != nil {
			return fmt.Errorf("write override warning: %w", err)
		}
	}

	manifest, err := yaml.Marshal(effective.Object)
	if err != nil {
		return fmt.Errorf("encode effective DGD: %w", err)
	}
	if err := writeFileAtomically(opts.outputPath, manifest, 0o644); err != nil {
		return fmt.Errorf("write effective DGD %q: %w", opts.outputPath, err)
	}
	return nil
}

func parseOptions(args []string, stderr io.Writer) (options, error) {
	opts := options{}
	flags := flag.NewFlagSet("dgd-apply-overrides", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&opts.blueprintPath, "blueprint", "", "Path to the complete DGD blueprint YAML")
	flags.StringVar(&opts.overridePath, "override", "", "Path to the partial DGD override JSON or YAML")
	flags.StringVar(&opts.outputPath, "output", "", "Path for the effective DGD YAML")
	flags.StringVar(&opts.installPath, "install-to", "", "Copy this executable to the given path and exit")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	if opts.installPath != "" {
		if opts.blueprintPath != "" || opts.overridePath != "" || opts.outputPath != "" {
			return options{}, fmt.Errorf("--install-to cannot be combined with --blueprint, --override, or --output")
		}
		return opts, nil
	}

	missing := make([]string, 0, 3)
	if opts.blueprintPath == "" {
		missing = append(missing, "--blueprint")
	}
	if opts.overridePath == "" {
		missing = append(missing, "--override")
	}
	if opts.outputPath == "" {
		missing = append(missing, "--output")
	}
	if len(missing) != 0 {
		return options{}, fmt.Errorf("required flags missing: %s", strings.Join(missing, ", "))
	}
	return opts, nil
}

func readDGD(path string, role string) (*unstructured.Unstructured, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s %q: %w", role, path, err)
	}
	return decodeDGD(data, role)
}

func decodeDGD(data []byte, role string) (*unstructured.Unstructured, error) {
	jsonData, err := yaml.YAMLToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("decode %s YAML: %w", role, err)
	}
	object, _, err := unstructured.UnstructuredJSONScheme.Decode(jsonData, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("decode %s object: %w", role, err)
	}
	dgd, ok := object.(*unstructured.Unstructured)
	if !ok {
		return nil, fmt.Errorf("decode %s: expected an object, got %T", role, object)
	}
	return dgd, nil
}

func installSelf(path string) error {
	executablePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("locate current executable: %w", err)
	}
	executable, err := os.Open(executablePath)
	if err != nil {
		return fmt.Errorf("open current executable: %w", err)
	}
	defer func() {
		_ = executable.Close()
	}()

	return writeAtomically(path, 0o755, func(destination io.Writer) error {
		_, err := io.Copy(destination, executable)
		return err
	})
}

func writeFileAtomically(path string, data []byte, mode os.FileMode) error {
	return writeAtomically(path, mode, func(destination io.Writer) error {
		_, err := destination.Write(data)
		return err
	})
}

func writeAtomically(path string, mode os.FileMode, write func(io.Writer) error) error {
	directory := filepath.Dir(path)
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		_ = os.Remove(temporaryPath)
	}()

	if err := write(temporary); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Chmod(mode); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, path)
}
