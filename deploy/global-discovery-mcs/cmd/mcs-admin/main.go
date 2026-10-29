// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Command mcs-admin inspects and repairs a registry store directly, without a
// running mcs-registry:
//
//	mcs-admin [store flags] describe [--clusterset NAME] [--output text|json]
//	mcs-admin [store flags] delete-cluster --clusterset NAME --cluster NAME --yes
//
// Store flags select the backend as mcs-registry does (--store,
// --dynamodb-table, --dynamodb-endpoint, --region), with the same MCS_
// environment fallbacks. Commands live in a table; a backend-specific
// command such as the DynamoDB store's create-table registers itself from
// its own file with registerCommand.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"slices"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/storefactory"
)

func main() {
	os.Exit(run(context.Background(), os.Args[1:], os.LookupEnv, os.Stdout, os.Stderr, storefactory.Open, time.Now))
}

type opener func(context.Context, *config.Config) (store.Store, error)

// cmdEnv is what a command runs with.
type cmdEnv struct {
	cfg    *config.Config
	stdout io.Writer
	stderr io.Writer
	now    time.Time
	open   opener
	st     store.Store
}

// store opens the configured store once. A fresh in-process memory store is
// always empty, so it warns when that is the store that opens.
func (e *cmdEnv) store(ctx context.Context) (store.Store, error) {
	if e.st == nil {
		if e.cfg.Store == config.StoreMemory {
			fmt.Fprintln(e.stderr, "mcs-admin: warning: the memory store is empty in a fresh process; "+
				"use --store=dynamodb to inspect a deployment")
		}
		st, err := e.open(ctx, e.cfg)
		if err != nil {
			return nil, err
		}
		e.st = st
	}
	return e.st, nil
}

// command is one mcs-admin subcommand.
type command struct {
	summary string
	run     func(ctx context.Context, e *cmdEnv, args []string) error
}

var commands = map[string]command{}

// registerCommand adds a subcommand.
func registerCommand(name string, c command) { commands[name] = c }

func init() {
	registerCommand("describe", command{
		summary: "list clustersets, or the records of one clusterset",
		run: func(ctx context.Context, e *cmdEnv, args []string) error {
			return describe(ctx, e, args)
		},
	})
	registerCommand("delete-cluster", command{
		summary: "delete a cluster and everything it owns",
		run: func(ctx context.Context, e *cmdEnv, args []string) error {
			return deleteCluster(ctx, e, args)
		},
	})
}

func usage(w io.Writer, fs *flag.FlagSet) {
	fmt.Fprint(w, "Usage: mcs-admin [store flags] <command> [command flags]\n\nCommands:\n")
	for _, name := range slices.Sorted(maps.Keys(commands)) {
		fmt.Fprintf(w, "  %-15s %s\n", name, commands[name].summary)
	}
	fmt.Fprint(w, "\nStore flags:\n")
	fs.PrintDefaults()
}

func run(
	ctx context.Context, args []string, lookupEnv func(string) (string, bool),
	stdout, stderr io.Writer, open opener, now func() time.Time,
) int {
	cfg := config.Default()
	fs := flag.NewFlagSet("mcs-admin", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { usage(stderr, fs) }
	fs.StringVar(&cfg.Store, "store", cfg.Store, "store backend: memory or dynamodb")
	fs.StringVar(&cfg.DynamoDBTable, "dynamodb-table", cfg.DynamoDBTable, "DynamoDB table name")
	fs.StringVar(&cfg.DynamoDBEndpoint, "dynamodb-endpoint", cfg.DynamoDBEndpoint,
		"DynamoDB endpoint override, for dynamodb-local")
	fs.StringVar(&cfg.Region, "region", cfg.Region, "AWS region; falls back to AWS_REGION")
	fs.DurationVar(&cfg.LeaseSkew, "lease-skew", cfg.LeaseSkew, "clock skew allowance when showing lease states")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if err := config.ApplyEnv(fs, lookupEnv); err != nil {
		fmt.Fprintf(stderr, "mcs-admin: %v\n", err)
		return 2
	}
	if cfg.Region == "" {
		cfg.Region, _ = lookupEnv("AWS_REGION")
	}
	if fs.NArg() == 0 {
		fs.Usage()
		return 2
	}
	name := fs.Arg(0)
	cmd, ok := commands[name]
	if !ok {
		fmt.Fprintf(stderr, "mcs-admin: unknown command %q\n", name)
		fs.Usage()
		return 2
	}

	e := &cmdEnv{cfg: cfg, stdout: stdout, stderr: stderr, now: now(), open: open}
	defer func() {
		if e.st != nil {
			_ = e.st.Close()
		}
	}()
	if err := cmd.run(ctx, e, fs.Args()[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		fmt.Fprintf(stderr, "mcs-admin %s: %v\n", name, err)
		if _, ok := errors.AsType[usageError](err); ok {
			return 2
		}
		return 1
	}
	return 0
}

type usageError struct{ error }

// parseError is a flag parse error: -h and --help pass through as
// flag.ErrHelp, everything else is a usage error.
func parseError(err error) error {
	if errors.Is(err, flag.ErrHelp) {
		return err
	}
	return usageError{err}
}

// newFlagSet returns the flag set of a subcommand. -h and --help print the
// subcommand's usage and flags to stderr and make Parse return
// flag.ErrHelp; other parse errors print the usage too, and run reports the
// error once.
func newFlagSet(name, synopsis string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: mcs-admin [store flags] %s %s\n\n%s\n\nFlags:\n",
			name, synopsis, commands[name].summary)
		fs.SetOutput(stderr)
		fs.PrintDefaults()
		fs.SetOutput(io.Discard)
	}
	return fs
}

// description is the JSON output of describe for one clusterset.
type description struct {
	ClusterSet   string           `json:"clusterset"`
	Clusters     []clusterRow     `json:"clusters"`
	Exports      []exportRow      `json:"exports"`
	EndpointSets []endpointSetRow `json:"endpointSets"`
	Counts       map[string]int   `json:"counts"`
}

type clusterRow struct {
	Cluster    string    `json:"cluster"`
	Region     string    `json:"region"`
	LeaseState string    `json:"leaseState"`
	Epoch      uint64    `json:"epoch"`
	RenewedAt  time.Time `json:"renewedAt"`
	ExpiresAt  time.Time `json:"expiresAt"`
	Generation uint64    `json:"generation"`
	Agent      string    `json:"agent,omitempty"`
}

type exportRow struct {
	Cluster    string    `json:"cluster"`
	Namespace  string    `json:"namespace"`
	Name       string    `json:"name"`
	Type       string    `json:"type"`
	Generation uint64    `json:"generation"`
	CreatedAt  time.Time `json:"exportCreationTimestamp"`
}

type endpointSetRow struct {
	Cluster    string `json:"cluster"`
	Namespace  string `json:"namespace"`
	Name       string `json:"name"`
	Endpoints  int    `json:"endpoints"`
	Ready      int    `json:"ready"`
	Generation uint64 `json:"generation"`
}

func leaseState(l store.Lease, now time.Time, skew time.Duration) string {
	if l.Epoch == 0 {
		return ""
	}
	return string(registry.EvaluateLease(l.RenewedAt, l.TTL, now, skew))
}

func describeClusterSet(snap store.ClusterSetSnapshot, now time.Time, skew time.Duration) description {
	d := description{ClusterSet: snap.ClusterSet, Counts: map[string]int{}}
	for _, c := range snap.Clusters {
		row := clusterRow{
			Cluster: c.Key.Cluster, Region: c.Region, LeaseState: leaseState(c.Lease, now, skew), Epoch: c.Lease.Epoch,
			RenewedAt: c.Lease.RenewedAt, ExpiresAt: c.Lease.ExpiresAt(), Generation: c.Generation, Agent: c.AgentID,
		}
		d.Clusters = append(d.Clusters, row)
		if row.LeaseState == string(registry.LeaseStateActive) {
			d.Counts["activeClusters"]++
		}
	}
	for _, e := range snap.Exports {
		d.Exports = append(d.Exports, exportRow{
			Cluster: e.Key.Cluster, Namespace: e.Key.Namespace, Name: e.Key.Name, Type: string(e.Spec.Type),
			Generation: e.Generation, CreatedAt: e.Spec.ExportCreationTimestamp,
		})
	}
	for _, s := range snap.EndpointSets {
		row := endpointSetRow{
			Cluster: s.Key.Cluster, Namespace: s.Key.Namespace, Name: s.Key.Name,
			Endpoints: len(s.Endpoints), Generation: s.Generation,
		}
		for _, e := range s.Endpoints {
			if e.Conditions.Ready {
				row.Ready++
			}
		}
		d.EndpointSets = append(d.EndpointSets, row)
		d.Counts["endpoints"] += row.Endpoints
	}
	d.Counts["clusters"] = len(d.Clusters)
	d.Counts["exports"] = len(d.Exports)
	d.Counts["endpointSets"] = len(d.EndpointSets)
	return d
}

func formatTime(t time.Time) string {
	if t.IsZero() {
		return "-"
	}
	return t.UTC().Format(time.RFC3339)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func printText(w io.Writer, d description) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintf(tw, "Clusterset %s: %d clusters (%d active), %d exports, %d endpoint sets (%d endpoints)\n\n",
		d.ClusterSet, d.Counts["clusters"], d.Counts["activeClusters"], d.Counts["exports"], d.Counts["endpointSets"],
		d.Counts["endpoints"])
	if len(d.Clusters) > 0 {
		fmt.Fprintln(tw, "CLUSTER\tREGION\tLEASE\tEPOCH\tRENEWED\tEXPIRES\tGENERATION")
		for _, c := range d.Clusters {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%s\t%s\t%d\n", c.Cluster, orDash(c.Region), c.LeaseState, c.Epoch,
				formatTime(c.RenewedAt), formatTime(c.ExpiresAt), c.Generation)
		}
		fmt.Fprintln(tw)
	}
	if len(d.Exports) > 0 {
		fmt.Fprintln(tw, "EXPORT\tCLUSTER\tTYPE\tGENERATION\tCREATED")
		for _, e := range d.Exports {
			fmt.Fprintf(tw, "%s/%s\t%s\t%s\t%d\t%s\n", e.Namespace, e.Name, e.Cluster, orDash(e.Type),
				e.Generation, formatTime(e.CreatedAt))
		}
		fmt.Fprintln(tw)
	}
	if len(d.EndpointSets) > 0 {
		fmt.Fprintln(tw, "ENDPOINT SET\tCLUSTER\tENDPOINTS\tREADY\tGENERATION")
		for _, s := range d.EndpointSets {
			fmt.Fprintf(tw, "%s/%s\t%s\t%d\t%d\t%d\n", s.Namespace, s.Name, s.Cluster, s.Endpoints, s.Ready, s.Generation)
		}
	}
	_ = tw.Flush()
}

func describe(ctx context.Context, e *cmdEnv, args []string) error {
	fs := newFlagSet("describe", "[--clusterset NAME] [--output text|json]", e.stderr)
	clusterSet := fs.String("clusterset", "", "clusterset to describe; all clustersets when empty")
	output := fs.String("output", "text", "output format: text or json")
	if err := fs.Parse(args); err != nil {
		return parseError(err)
	}
	if *output != "text" && *output != "json" {
		return usageError{fmt.Errorf("--output %q: want text or json", *output)}
	}
	st, err := e.store(ctx)
	if err != nil {
		return err
	}
	w := e.stdout
	names := []string{*clusterSet}
	if *clusterSet == "" {
		if names, err = st.ListClusterSets(ctx); err != nil {
			return fmt.Errorf("list clustersets: %w", err)
		}
	}
	var out []description
	for _, name := range names {
		snap, err := st.LoadClusterSet(ctx, name)
		if err != nil {
			return fmt.Errorf("load clusterset %q: %w", name, err)
		}
		snap.ClusterSet = name
		out = append(out, describeClusterSet(snap, e.now, e.cfg.LeaseSkew))
	}
	if *output == "json" {
		enc := json.NewEncoder(w)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	if len(out) == 0 {
		fmt.Fprintln(w, "No clustersets.")
		return nil
	}
	for i, d := range out {
		if i > 0 {
			fmt.Fprintln(w, strings.Repeat("-", 40))
		}
		printText(w, d)
	}
	return nil
}

func deleteCluster(ctx context.Context, e *cmdEnv, args []string) error {
	fs := newFlagSet("delete-cluster", "--clusterset NAME --cluster NAME [--yes]", e.stderr)
	clusterSet := fs.String("clusterset", "", "clusterset of the cluster")
	cluster := fs.String("cluster", "", "cluster to delete")
	yes := fs.Bool("yes", false, "confirm the deletion")
	if err := fs.Parse(args); err != nil {
		return parseError(err)
	}
	if !registry.IsDNSLabel(*clusterSet) || !registry.IsDNSLabel(*cluster) {
		return usageError{errors.New("--clusterset and --cluster must be DNS labels")}
	}
	st, err := e.store(ctx)
	if err != nil {
		return err
	}
	w := e.stdout
	snap, err := st.LoadClusterSet(ctx, *clusterSet)
	if err != nil {
		return fmt.Errorf("load clusterset %q: %w", *clusterSet, err)
	}
	var exports, sets int
	for _, e := range snap.Exports {
		if e.Key.Cluster == *cluster {
			exports++
		}
	}
	for _, s := range snap.EndpointSets {
		if s.Key.Cluster == *cluster {
			sets++
		}
	}
	summary := fmt.Sprintf("cluster %s/%s: %d exports, %d endpoint sets", *clusterSet, *cluster, exports, sets)
	if !*yes {
		return usageError{fmt.Errorf("would delete %s; pass --yes to delete", summary)}
	}
	if err := st.DeleteCluster(ctx, store.ClusterKey{ClusterSet: *clusterSet, Cluster: *cluster}); err != nil {
		return fmt.Errorf("delete cluster: %w", err)
	}
	fmt.Fprintf(w, "Deleted %s.\n", summary)
	return nil
}
