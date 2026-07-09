/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdoverride"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

type options struct {
	installPath          string
	printProtocolVersion bool
}

type applyRequest struct {
	Blueprint json.RawMessage `json:"blueprint"`
	Override  json.RawMessage `json:"override"`
}

const protocolVersion = "1"

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintf(os.Stderr, "dgd-apply-overrides: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdin io.Reader, stdout io.Writer, stderr io.Writer) error {
	opts, err := parseOptions(args, stderr)
	if err != nil {
		return err
	}
	if opts.printProtocolVersion {
		_, err := fmt.Fprintln(stdout, protocolVersion)
		return err
	}
	if opts.installPath != "" {
		if err := installSelf(opts.installPath); err != nil {
			return fmt.Errorf("install dgd-apply-overrides to %q: %w", opts.installPath, err)
		}
		return nil
	}

	request, err := readApplyRequest(stdin)
	if err != nil {
		return err
	}
	blueprint, err := decodeDGD(request.Blueprint, "request blueprint")
	if err != nil {
		return err
	}
	override, err := decodeDGD(request.Override, "request override")
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

	if err := json.NewEncoder(stdout).Encode(effective.Object); err != nil {
		return fmt.Errorf("encode effective DGD response: %w", err)
	}
	return nil
}

func parseOptions(args []string, stderr io.Writer) (options, error) {
	opts := options{}
	flags := flag.NewFlagSet("dgd-apply-overrides", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.StringVar(&opts.installPath, "install-to", "", "Copy this executable to the given path and exit")
	flags.BoolVar(&opts.printProtocolVersion, "protocol-version", false, "Print the CLI protocol version and exit")
	if err := flags.Parse(args); err != nil {
		return options{}, err
	}
	if flags.NArg() != 0 {
		return options{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	if opts.installPath != "" || opts.printProtocolVersion {
		if opts.installPath != "" && opts.printProtocolVersion {
			return options{}, fmt.Errorf("--install-to and --protocol-version are mutually exclusive")
		}
	}
	return opts, nil
}

func readApplyRequest(reader io.Reader) (applyRequest, error) {
	decoder := json.NewDecoder(reader)
	decoder.DisallowUnknownFields()

	request := applyRequest{}
	if err := decoder.Decode(&request); err != nil {
		return applyRequest{}, fmt.Errorf("decode request JSON: %w", err)
	}
	if len(request.Blueprint) == 0 || string(request.Blueprint) == "null" {
		return applyRequest{}, errors.New("request blueprint is required")
	}
	if len(request.Override) == 0 || string(request.Override) == "null" {
		return applyRequest{}, errors.New("request override is required")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return applyRequest{}, errors.New("decode request JSON: multiple JSON values are not allowed")
		}
		return applyRequest{}, fmt.Errorf("decode request JSON: %w", err)
	}
	return request, nil
}

func decodeDGD(data []byte, role string) (*unstructured.Unstructured, error) {
	object, _, err := unstructured.UnstructuredJSONScheme.Decode(data, nil, nil)
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
