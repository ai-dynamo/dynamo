// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command line of the `dynamo-ft-controller` binary.

use std::collections::{BTreeMap, BTreeSet};
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail, ensure};
use clap::{Parser, ValueEnum};
use reqwest::Url;

use crate::controller::{MAX_DURATION, Settings};
use crate::decision::Policy;

/// How often a standing problem (ranks that never answer, ranks counted dead
/// while every survivor is healthy) is logged again.
pub const REMIND_EVERY: Duration = Duration::from_secs(30);

/// One decider per vLLM fault-tolerance world. Watches every rank's front
/// end and recovers the world through vLLM's fault-tolerance API, or
/// escalates (exit code 3) when it cannot.
#[derive(Clone, Debug, Parser)]
#[command(name = "dynamo-ft-controller", version)]
pub struct Cli {
    /// Name of the world; used in request ids and state file names.
    #[arg(long)]
    pub world: String,

    /// A rank's front-end base URL as RANK=URL; repeat for every rank (for
    /// example 0=http://10.0.0.5:8000). RANK is the global data-parallel
    /// rank, which the front end reports as its engine id.
    #[arg(long = "rank", value_name = "RANK=URL", required = true, value_parser = parse_rank_url)]
    pub ranks: Vec<(u32, Url)>,

    /// Address RANK hosts the coordination store on (dp_master_ip), as
    /// RANK=HOST. Default: the host of the rank's URL, which must then not be
    /// a loopback address. An explicit loopback address is accepted, for
    /// single-node tests.
    #[arg(long = "store-host", value_name = "RANK=HOST", value_parser = parse_rank_host)]
    pub store_hosts: Vec<(u32, String)>,

    /// Rank hosting the coordination store when the world starts. vLLM 0.34.0
    /// always hosts it on the lowest live rank, so this must be the lowest
    /// --rank, which is also the default.
    #[arg(long)]
    pub initial_store_rank: Option<u32>,

    /// Port of the first store rebuild; each rebuild takes the next port.
    /// Worlds that share nodes need ranges that do not overlap.
    #[arg(long, default_value_t = 29650)]
    pub store_port_base: u16,

    /// Time between status polls.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "500ms")]
    pub poll_interval: Duration,

    /// Timeout of each status poll and recovery order.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "2s")]
    pub call_timeout: Duration,

    /// A rank whose status endpoint fails this long counts as dead.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "10s")]
    pub dead_after: Duration,

    /// Wake survivors still healthy this long after a fault is suspected.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "2s")]
    pub wake_after: Duration,

    /// Longest gap between two wakes of one rank. The gap doubles from
    /// --wake-after up to this and stays there while the fault lasts.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "30s")]
    pub wake_max_gap: Duration,

    /// Timeout of a wake request, whose answer is ignored.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "2s")]
    pub wake_timeout: Duration,

    /// vLLM's round check-in timeout (10 s in 0.34.0). A round that no
    /// survivor has taken up after this counts as lost, and a new round with
    /// a new request id may follow.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "10s")]
    pub checkin_timeout: Duration,

    /// Recovery rounds per incident before escalating.
    #[arg(long, default_value_t = 3)]
    pub max_rounds: u32,

    /// Give up on a round that has not settled after this. Default: a round
    /// may run until the recovery deadline.
    #[arg(long, value_parser = humantime::parse_duration)]
    pub round_timeout: Option<Duration>,

    /// Escalate when the world is not recovered this long after the first
    /// survivor fault, which is when vLLM's own recovery clock starts. Keep
    /// it under vLLM's engine_recovery_timeout_sec (120 s).
    #[arg(long, value_parser = humantime::parse_duration, default_value = "110s")]
    pub recovery_deadline: Duration,

    /// Send {"abort": true} with retry rounds. vLLM 0.34.0 ignores it.
    #[arg(long)]
    pub retry_abort: bool,

    /// Directory for the controller epoch and the last round record.
    #[arg(long, required_unless_present = "dry_run")]
    pub state_dir: Option<PathBuf>,

    /// Command run through `sh -c` when escalating. {world} and {reason}
    /// become single-quoted shell words.
    #[arg(long, value_name = "TEMPLATE")]
    pub escalate_cmd: Option<String>,

    /// Kill the escalation command after this long.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "30s")]
    pub escalate_timeout: Duration,

    /// Log decisions only: send no wake or order, run no escalation command,
    /// write no state. Prints FT_CONTROLLER_WOULD_ESCALATE where it would
    /// escalate and goes on watching.
    #[arg(long)]
    pub dry_run: bool,

    /// Log format on stderr.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

/// A validated member of the world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub url: Url,
    pub store_host: String,
}

impl Cli {
    /// Checks the arguments together and returns the members by rank.
    pub fn members(&self) -> Result<BTreeMap<u32, Member>> {
        check_world_name(&self.world)?;
        let mut members = BTreeMap::new();
        for (rank, url) in &self.ranks {
            let member = Member {
                url: url.clone(),
                store_host: url_host(url)?,
            };
            if members.insert(*rank, member).is_some() {
                bail!("--rank {rank} is given more than once");
            }
        }
        let mut urls = BTreeMap::new();
        for (&rank, member) in &members {
            let url = member.url.as_str().trim_end_matches('/').to_string();
            if let Some(other) = urls.insert(url, rank) {
                bail!(
                    "--rank {other} and --rank {rank} have the same URL {}",
                    member.url
                );
            }
        }
        let mut explicit = BTreeSet::new();
        for (rank, host) in &self.store_hosts {
            ensure!(
                explicit.insert(*rank),
                "--store-host for rank {rank} is given more than once"
            );
            let Some(member) = members.get_mut(rank) else {
                bail!("--store-host names rank {rank}, which is not a member of the world");
            };
            member.store_host = host.clone();
        }
        for (rank, member) in &members {
            if !explicit.contains(rank) && is_local_address(&member.store_host) {
                bail!(
                    "the URL of rank {rank} has the loopback address {}, which the other ranks cannot use as dp_master_ip; \
                     pass --store-host {rank}=<address the other ranks reach> (an explicit loopback address is accepted, for single-node tests)",
                    member.store_host
                );
            }
        }
        let Some(&lowest) = members.keys().next() else {
            bail!("at least one --rank is required");
        };
        if let Some(initial) = self.initial_store_rank {
            ensure!(
                initial == lowest,
                "--initial-store-rank {initial} must be the lowest --rank ({lowest}): vLLM 0.34.0 hosts the coordination store on the lowest live rank"
            );
        }
        let positive = [
            ("--poll-interval", self.poll_interval),
            ("--call-timeout", self.call_timeout),
            ("--dead-after", self.dead_after),
            ("--wake-max-gap", self.wake_max_gap),
            ("--wake-timeout", self.wake_timeout),
            ("--checkin-timeout", self.checkin_timeout),
            ("--recovery-deadline", self.recovery_deadline),
            ("--escalate-timeout", self.escalate_timeout),
        ];
        let round_timeout = self.round_timeout.map(|value| ("--round-timeout", value));
        for (flag, value) in positive.into_iter().chain(round_timeout) {
            ensure!(!value.is_zero(), "{flag} must be positive");
            check_bounded(flag, value)?;
        }
        check_bounded("--wake-after", self.wake_after)?;
        ensure!(self.max_rounds >= 1, "--max-rounds must be at least 1");
        Ok(members)
    }

    pub fn settings(&self) -> Settings {
        Settings {
            world: self.world.clone(),
            poll_interval: self.poll_interval,
            policy: Policy {
                dead_after: self.dead_after,
                wake_after: self.wake_after,
                wake_max_gap: self.wake_max_gap,
                checkin_timeout: self.checkin_timeout,
                max_rounds: self.max_rounds,
                recovery_deadline: self.recovery_deadline,
            },
            round_timeout: self.round_timeout,
            retry_abort: self.retry_abort,
            store_port_base: self.store_port_base,
            dry_run: self.dry_run,
            remind_every: REMIND_EVERY,
        }
    }
}

/// Durations are bounded, so no deadline computation can overflow.
fn check_bounded(flag: &str, value: Duration) -> Result<()> {
    ensure!(
        value <= MAX_DURATION,
        "{flag} must be at most {}",
        humantime::format_duration(MAX_DURATION)
    );
    Ok(())
}

/// World names go into request ids and file names.
fn check_world_name(world: &str) -> Result<()> {
    ensure!(
        !world.is_empty() && world.len() <= 64,
        "--world must be 1 to 64 characters"
    );
    ensure!(
        world
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "--world {world:?} may only use ASCII letters, digits, '-', '_' and '.'"
    );
    ensure!(
        world.starts_with(|c: char| c.is_ascii_alphanumeric()),
        "--world {world:?} must start with a letter or digit"
    );
    Ok(())
}

/// The host of `url`, without the brackets of an IPv6 literal.
fn url_host(url: &Url) -> Result<String> {
    let Some(host) = url.host_str() else {
        bail!("{url} has no host");
    };
    let host = host
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(host);
    Ok(host.to_string())
}

/// Loopback and unspecified addresses, which no other rank can connect to.
fn is_local_address(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    host.parse::<IpAddr>().is_ok_and(|ip| {
        let ip = ip.to_canonical();
        ip.is_loopback() || ip.is_unspecified()
    })
}

fn split_rank(arg: &str, what: &str) -> Result<(u32, String), String> {
    let Some((rank, value)) = arg.split_once('=') else {
        return Err(format!("expected RANK={what}, got {arg:?}"));
    };
    let rank = rank
        .trim()
        .parse()
        .map_err(|error| format!("bad rank {rank:?}: {error}"))?;
    Ok((rank, value.trim().to_string()))
}

fn parse_rank_url(arg: &str) -> Result<(u32, Url), String> {
    let (rank, url) = split_rank(arg, "URL")?;
    let url: Url = url
        .parse()
        .map_err(|error| format!("bad URL {url:?}: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("{url} is not an http(s) URL"));
    }
    if url.host_str().is_none() {
        return Err(format!("{url} has no host"));
    }
    Ok((rank, url))
}

fn parse_rank_host(arg: &str) -> Result<(u32, String), String> {
    let (rank, host) = split_rank(arg, "HOST")?;
    if host.is_empty() || host.contains(char::is_whitespace) {
        return Err(format!("bad store host {host:?}"));
    }
    Ok((rank, host))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(
            [
                "dynamo-ft-controller",
                "--world",
                "w1",
                "--state-dir",
                "/tmp/ft",
            ]
            .iter()
            .chain(args),
        )
    }

    fn members_error(args: &[&str]) -> String {
        parse(args).unwrap().members().unwrap_err().to_string()
    }

    #[test]
    fn ranks_and_store_hosts_build_the_membership() {
        let cli = parse(&[
            "--rank",
            "0=http://10.0.0.5:8000",
            "--rank",
            "1=http://[fd00::6]:8000",
            "--rank",
            "2=http://pod-2.world:8000",
            "--store-host",
            "2=10.1.0.2",
        ])
        .unwrap();
        let members = cli.members().unwrap();
        let hosts: Vec<&str> = members.values().map(|m| m.store_host.as_str()).collect();
        assert_eq!(hosts, ["10.0.0.5", "fd00::6", "10.1.0.2"]);
        let settings = cli.settings();
        assert_eq!(settings.poll_interval, Duration::from_millis(500));
        assert_eq!(settings.policy.dead_after, Duration::from_secs(10));
        assert_eq!(settings.policy.wake_max_gap, Duration::from_secs(30));
        assert_eq!(settings.policy.checkin_timeout, Duration::from_secs(10));
        assert_eq!(settings.policy.recovery_deadline, Duration::from_secs(110));
        assert_eq!(settings.round_timeout, None, "rounds run to the deadline");
        assert_eq!(settings.store_port_base, 29650);
        assert_eq!(settings.policy.max_rounds, 3);
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn rejects_inconsistent_membership() {
        let duplicate = members_error(&["--rank", "0=http://a:1", "--rank", "0=http://b:1"]);
        assert!(duplicate.contains("more than once"), "{duplicate}");

        let same_url = members_error(&["--rank", "0=http://a:1", "--rank", "1=http://a:1/"]);
        assert!(same_url.contains("same URL"), "{same_url}");

        let stranger = members_error(&["--rank", "0=http://a:1", "--store-host", "1=10.0.0.1"]);
        assert!(stranger.contains("not a member"), "{stranger}");

        let two_ranks = ["--rank", "1=http://a:1", "--rank", "2=http://b:1"];
        let store = members_error(&[&two_ranks[..], &["--initial-store-rank", "2"]].concat());
        assert!(
            store.contains("--initial-store-rank 2 must be the lowest --rank (1)"),
            "{store}"
        );
        let lowest = parse(&[&two_ranks[..], &["--initial-store-rank", "1"]].concat()).unwrap();
        assert!(lowest.members().is_ok());
        assert!(parse(&two_ranks).unwrap().members().is_ok());

        let world = Cli::try_parse_from([
            "x",
            "--world",
            "a b",
            "--state-dir",
            "/tmp",
            "--rank",
            "0=http://a:1",
        ])
        .unwrap();
        assert!(world.members().unwrap_err().to_string().contains("--world"));
    }

    #[test]
    fn loopback_store_hosts_must_be_explicit() {
        for url in [
            "http://127.0.0.1:8000",
            "http://127.1.2.3:8000",
            "http://localhost:8000",
            "http://[::1]:8000",
            "http://[::ffff:127.0.0.1]:8000",
            "http://0.0.0.0:8000",
        ] {
            let rank = format!("0={url}");
            let error = members_error(&["--rank", &rank]);
            assert!(error.contains("--store-host 0="), "{url}: {error}");
        }
        let single_node = parse(&[
            "--rank",
            "0=http://127.0.0.1:8000",
            "--rank",
            "1=http://127.0.0.1:8001",
            "--store-host",
            "0=127.0.0.1",
            "--store-host",
            "1=127.0.0.1",
        ])
        .unwrap();
        let members = single_node.members().unwrap();
        assert!(members.values().all(|m| m.store_host == "127.0.0.1"));
    }

    #[test]
    fn durations_are_positive_and_bounded() {
        let absurd = members_error(&[
            "--rank",
            "0=http://a:1",
            "--recovery-deadline",
            "1000000000y",
        ]);
        assert!(
            absurd.contains("--recovery-deadline must be at most"),
            "{absurd}"
        );
        let zero = members_error(&["--rank", "0=http://a:1", "--round-timeout", "0s"]);
        assert!(zero.contains("--round-timeout must be positive"), "{zero}");
        let wake = members_error(&["--rank", "0=http://a:1", "--wake-after", "2days"]);
        assert!(wake.contains("--wake-after"), "{wake}");
        let explicit = parse(&["--rank", "0=http://a:1", "--round-timeout", "90s"]).unwrap();
        assert!(explicit.members().is_ok());
        assert_eq!(
            explicit.settings().round_timeout,
            Some(Duration::from_secs(90))
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(parse(&["--rank", "http://a:1"]).is_err());
        assert!(parse(&["--rank", "x=http://a:1"]).is_err());
        assert!(parse(&["--rank", "0=ftp://a:1"]).is_err());
        assert!(parse(&["--rank", "0=http://a:1", "--dead-after", "ten"]).is_err());
        assert!(parse(&[]).is_err(), "at least one --rank is required");
        // --state-dir is required unless --dry-run.
        assert!(Cli::try_parse_from(["x", "--world", "w", "--rank", "0=http://a:1"]).is_err());
        assert!(
            Cli::try_parse_from(["x", "--world", "w", "--rank", "0=http://a:1", "--dry-run"])
                .is_ok()
        );
    }
}
