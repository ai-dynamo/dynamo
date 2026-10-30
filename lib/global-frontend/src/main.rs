// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_frontend` and the crate's README.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
use dynamo_global_frontend::{
    Metrics, PoolSnapshot, RelayGatedPoolCatalog, app, discover_pools, follow_relays,
    measure_round_trips, serve, serve_relay_views,
};
use dynamo_global_router::{
    AffinitySelector, InFlightTracker, LeastInFlightSelector, Pool, PoolSelector, PoolSignals,
    RoundRobinSelector,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

/// How often the round trip to each pool is measured.
const ROUND_TRIP_INTERVAL: Duration = Duration::from_secs(30);
/// How often expired session pins are dropped.
const PIN_SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// How each request picks a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RouterPolicy {
    /// Every pool serving the model in turn.
    RoundRobin,
    /// The pool with the fewest of this process's requests in flight.
    LeastInFlight,
    /// Keep each session on one pool, nearest region first, and move it only
    /// for a pool with much more room (see `--headroom-factor`). Reads each
    /// pool's load from its Relay (`--pool-relay`).
    Affinity,
}

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Accept Relay-initiated presence streams on this port and route only
    /// to pools with a live Relay view. Uses a trusted network, without mTLS.
    #[arg(long)]
    relay_view_port: Option<u16>,
    /// Read candidate pools from Global Discovery and require live Relay views.
    /// Router discovery must use a probe independent of pool availability (such as /live).
    #[arg(long, conflicts_with = "pools", requires = "relay_view_port")]
    global_discovery_url: Option<Url>,
    /// A pool as `<cluster>/<namespace>/<DGD name>=<model>@<frontend URL>`,
    /// where the cluster is the ID the pool's PoolRelay runs with. Separate
    /// several frontend URLs with commas, and repeat the flag for each pool.
    /// The cluster also names the pool's region.
    #[arg(long = "pool", value_parser = parse_pool, required_unless_present = "global_discovery_url")]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
    #[arg(long, value_enum, default_value_t = RouterPolicy::RoundRobin)]
    router_policy: RouterPolicy,
    /// The region this process runs in. Affinity prefers it, and metrics
    /// distinguish local requests from requests sent to other regions.
    #[arg(long, required_if_eq("router_policy", "affinity"))]
    region: Option<String>,
    /// A pool's PoolRelay, which streams the pool's load, as
    /// `<cluster>/<namespace>/<DGD name>=<URL>`. Repeat the flag for each pool;
    /// each pool needs a Relay of its own, since a Relay reports the load of
    /// everything it watches as one.
    #[arg(long = "pool-relay", value_parser = parse_pool_relay)]
    pool_relays: Vec<(PoolKey, Url)>,
    /// `affinity`: a request leaves its pool only for one with more than this
    /// many times its spare capacity, or when its pool is full.
    #[arg(long, default_value_t = 2.0)]
    headroom_factor: f64,
    /// `affinity`: how long a session keeps its pool without a request.
    #[arg(long, default_value_t = 1800)]
    session_idle_secs: u64,
}

fn parse_key(key: &str) -> Result<PoolKey, String> {
    let [cluster, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(format!(
            "expected <cluster>/<namespace>/<DGD name>, got {key}"
        ));
    };
    PoolKey::new(cluster, namespace, dgd).map_err(|error| error.to_string())
}

fn parse_url(url: &str, what: &str) -> Result<Url, String> {
    let parsed = url
        .parse::<Url>()
        .map_err(|error| format!("{url}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{what} must be an HTTP(S) URL: {url}"));
    }
    Ok(parsed)
}

fn parse_pool_relay(value: &str) -> Result<(PoolKey, Url), String> {
    let (key, relay) = value
        .split_once('=')
        .ok_or_else(|| format!("expected <cluster>/<namespace>/<DGD name>=<URL>, got {value}"))?;
    let relay = parse_url(relay, "pool relay")?;
    // The Relay client is built without TLS.
    if relay.scheme() != "http" {
        return Err(format!("pool relay must be an http:// URL: {relay}"));
    }
    Ok((parse_key(key)?, relay))
}

/// Gives each pool its Relay. Every Relay must name a configured pool, and
/// no two pools may share one.
fn attach_relays(pools: &mut [Pool], relays: &[(PoolKey, Url)]) -> Result<()> {
    for (key, relay) in relays {
        if pools.iter().any(|pool| pool.relay.as_ref() == Some(relay)) {
            bail!("--pool-relay {relay} is given for two pools; each pool needs its own Relay");
        }
        let id = V1PoolIdDeriver.derive(key);
        let Some(pool) = pools.iter_mut().find(|pool| pool.id == id) else {
            bail!("--pool-relay {relay} names a pool no --pool gives");
        };
        if pool.relay.replace(relay.clone()).is_some() {
            bail!("--pool-relay is given twice for pool {id}");
        }
    }
    Ok(())
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape = || {
        format!("expected <cluster>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}")
    };
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    if model.trim().is_empty() {
        return Err("pool model must not be empty".to_owned());
    }
    let key = parse_key(key)?;
    let frontends = frontends
        .split(',')
        .map(|url| parse_url(url, "pool frontend"))
        .collect::<Result<Vec<_>, _>>()?;
    // Until pools come from Global Discovery, which reports each pool's
    // region, the cluster names it.
    Ok(Pool::new(&key, model, key.cluster_id(), frontends))
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let mut args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }
    attach_relays(&mut args.pools, &args.pool_relays)?;
    if args.relay_view_port.is_some() && !args.pool_relays.is_empty() {
        bail!("--pool-relay cannot be combined with Relay-initiated receiving");
    }
    if !(args.headroom_factor >= 1.0 && args.headroom_factor.is_finite()) {
        bail!("--headroom-factor must be at least 1");
    }

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });

    // Keep the explicit catalog sender alive for the static-pool mode.
    let (_catalog, pools) = watch::channel(Arc::new(PoolSnapshot::new(args.pools)));
    let pools = if let Some(url) = args.global_discovery_url {
        let client = dynamo_global_discovery_client::GlobalDiscoveryClient::new(
            url,
            Duration::from_secs(5),
        )?;
        discover_pools(client, Duration::from_secs(5), shutdown.child_token())
    } else {
        pools
    };
    let in_flight = Arc::new(InFlightTracker::new());
    let signals = Arc::new(PoolSignals::new(in_flight.clone()));
    let relay_catalog = if args.relay_view_port.is_some() {
        Some(Arc::new(RelayGatedPoolCatalog::with_signals(
            pools.clone(),
            shutdown.child_token(),
            Some(signals.clone()),
        )))
    } else if !args.pool_relays.is_empty() {
        Some(Arc::new(RelayGatedPoolCatalog::for_subscriptions(
            pools.clone(),
            shutdown.child_token(),
            signals.clone(),
        )))
    } else {
        None
    };
    let receiving = if let Some(port) = args.relay_view_port {
        let listener = tokio::net::TcpListener::bind((args.host.as_str(), port))
            .await
            .context("listen for Relay views")?;
        Some((
            listener,
            Arc::clone(relay_catalog.as_ref().context("missing Relay catalog")?),
        ))
    } else {
        if let Some(catalog) = &relay_catalog {
            tokio::spawn(follow_relays(
                pools.clone(),
                Arc::clone(catalog),
                shutdown.child_token(),
            ));
        }
        None
    };
    let pools = relay_catalog
        .as_ref()
        .map_or(pools, |catalog| catalog.subscribe());
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let metrics = Metrics::new(args.region.clone(), in_flight.clone());
    let selector: Arc<dyn PoolSelector> = match args.router_policy {
        RouterPolicy::RoundRobin => Arc::new(RoundRobinSelector::new()),
        RouterPolicy::LeastInFlight => Arc::new(LeastInFlightSelector::new(in_flight.clone())),
        RouterPolicy::Affinity => {
            let region = args.region.clone().context("--region is required")?;
            if args.pool_relays.is_empty() && args.relay_view_port.is_none() {
                tracing::warn!(
                    "no --pool-relay is given, so affinity knows no pool's load \
                     and keeps each session on its nearest pool"
                );
            }
            tokio::spawn(measure_round_trips(
                pools.clone(),
                signals.clone(),
                ROUND_TRIP_INTERVAL,
                shutdown.child_token(),
            ));
            let idle = Duration::from_secs(args.session_idle_secs);
            let selector = Arc::new(AffinitySelector::new(
                signals,
                region,
                args.headroom_factor,
                idle,
            ));
            tokio::spawn(sweep_pins(selector.clone(), shutdown.child_token()));
            selector
        }
    };
    let app = app(pools, selector, in_flight, http, metrics);
    let address = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .with_context(|| format!("listen on {address}"))?;
    tracing::info!(%address, "Global Frontend listening");
    if let Some((relay_listener, catalog)) = receiving {
        let http = async {
            let result = serve(listener, app, shutdown.clone()).await;
            shutdown.cancel();
            result.context("serve the Global Frontend")
        };
        let relay = async {
            let result = serve_relay_views(relay_listener, catalog, shutdown.clone()).await;
            shutdown.cancel();
            result.context("serve Relay views")
        };
        tokio::try_join!(http, relay)?;
        Ok(())
    } else {
        serve(listener, app, shutdown)
            .await
            .context("serve the Global Frontend")
    }
}

/// Drops expired session pins once a `PIN_SWEEP_INTERVAL`, off the request
/// path, until `cancel` fires.
async fn sweep_pins(selector: Arc<AffinitySelector>, cancel: CancellationToken) {
    let mut ticks = tokio::time::interval(PIN_SWEEP_INTERVAL);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => selector.sweep(),
        }
    }
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_paths_and_multiple_frontends() {
        let pool =
            parse_pool("ohio/dynamo/blue=Qwen/Qwen3@http://blue:8000,https://green:8000").unwrap();
        assert_eq!(pool.model, "Qwen/Qwen3");
        assert_eq!(pool.frontends.len(), 2);
    }

    #[test]
    fn rejects_missing_models_and_non_http_frontends() {
        for value in [
            "ohio/ns/pool=@http://pool:8000",
            "ohio/ns/pool=m@file:///tmp/pool",
            "ohio/ns/pool=m@mailto:user@example.com",
        ] {
            assert!(parse_pool(value).is_err(), "{value}");
        }
    }

    #[test]
    fn parses_pool_relays_and_requires_a_region_for_affinity() {
        let (key, relay) = parse_pool_relay("ohio/dynamo/blue=http://relay:50051").unwrap();
        assert_eq!(key.dgd_name(), "blue");
        assert_eq!(relay.as_str(), "http://relay:50051/");
        assert!(parse_pool_relay("ohio/dynamo=http://relay:50051").is_err());
        assert!(parse_pool_relay("ohio/dynamo/blue=https://relay:50051").is_err());

        // Each pool needs its own Relay.
        let mut pools = ["blue", "green"]
            .map(|dgd| parse_pool(&format!("ohio/dynamo/{dgd}=qwen@http://{dgd}:8000")).unwrap());
        let relays = ["blue", "green"]
            .map(|dgd| parse_pool_relay(&format!("ohio/dynamo/{dgd}=http://relay:50051")).unwrap());
        assert!(attach_relays(&mut pools, &relays).is_err());
        assert!(attach_relays(&mut pools[1..], &relays[1..]).is_ok());

        let pool = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";
        let affinity = ["global-frontend", pool, "--router-policy", "affinity"];
        assert!(Args::try_parse_from(affinity).is_err());
        let args = Args::try_parse_from([&affinity[..], &["--region", "ohio"]].concat()).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::Affinity);
        let args = Args::try_parse_from(["global-frontend", pool]).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::RoundRobin);
    }

    #[test]
    fn requires_a_pool() {
        assert!(Args::try_parse_from(["global-frontend"]).is_err());
        let discovery = [
            "global-frontend",
            "--global-discovery-url",
            "http://discovery:8080",
        ];
        assert!(Args::try_parse_from(discovery).is_err());
        assert!(
            Args::try_parse_from([&discovery[..], &["--relay-view-port", "50051"]].concat())
                .is_ok()
        );
        assert!(
            Args::try_parse_from(
                [
                    &discovery[..],
                    &[
                        "--relay-view-port",
                        "50051",
                        "--pool",
                        "east/dynamo/blue=qwen@http://pool:8000"
                    ],
                ]
                .concat()
            )
            .is_err()
        );
    }
}
