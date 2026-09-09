// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only queue/cache experiment, NOT an inference-engine performance prediction.
//! Every policy receives the same arrivals. Cache contents and queues evolve from its own actions.

use dynamo_custom_policy_example_adaptive::controller::{
    AdaptiveRouter, Algorithm, Candidate, Parameters, pick_weighted,
};
use std::{collections::VecDeque, time::Duration};

const WEIGHTS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
const SLO: f64 = 0.5;
const DISCOUNT: f64 = 0.999;

#[derive(Clone, Copy)]
struct Credit {
    context: usize,
    arm: usize,
    issued: u64,
}

/// Bucketed discounted UCB, experimental feedback consumer, deliberately not catalog registered.
struct Bandit {
    sums: [[f64; 5]; 4],
    counts: [[f64; 5]; 4],
    pending: [[usize; 5]; 4],
    tick: u64,
}

impl Bandit {
    fn new() -> Self {
        Self {
            sums: [[0.0; 5]; 4],
            counts: [[0.0; 5]; 4],
            pending: [[0; 5]; 4],
            tick: 0,
        }
    }

    fn choose(&mut self, context: usize) -> Credit {
        self.tick += 1;
        for c in 0..4 {
            for a in 0..5 {
                self.sums[c][a] *= DISCOUNT;
                self.counts[c][a] *= DISCOUNT;
            }
        }
        let total: f64 = self.counts[context].iter().sum::<f64>()
            + self.pending[context].iter().sum::<usize>() as f64;
        let score = |arm: usize| {
            let n = self.counts[context][arm];
            let pending = self.pending[context][arm] as f64;
            if n + pending == 0.0 {
                f64::INFINITY
            } else {
                let mean = if n > 0.0 {
                    self.sums[context][arm] / n
                } else {
                    0.5
                };
                mean + 0.4 * ((total + 1.0).ln() / (n + pending)).sqrt()
            }
        };
        let arm = (0..5)
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .unwrap();
        self.pending[context][arm] += 1;
        Credit {
            context,
            arm,
            issued: self.tick,
        }
    }

    fn observe(&mut self, credit: Credit, ttft: f64) {
        let Credit {
            context,
            arm,
            issued,
        } = credit;
        self.pending[context][arm] -= 1;
        // Attribute delayed feedback to the decision's arm/context, discounted by decision age.
        let age_weight = DISCOUNT.powf((self.tick - issued) as f64);
        self.counts[context][arm] += age_weight;
        self.sums[context][arm] += age_weight / (1.0 + ttft / SLO);
    }
}

#[derive(Clone)]
struct Request {
    at: f64,
    prefix: usize,
    input: usize,
    output: usize,
    phase: usize,
}
struct Job {
    request: Request,
    credit: Option<Credit>,
}
struct Running {
    job: Job,
    first: f64,
    done: f64,
    sent: bool,
}
struct Worker {
    queue: VecDeque<Job>,
    running: Option<Running>,
    cache: VecDeque<usize>,
    speed: f64,
}

impl Worker {
    fn load(&self) -> usize {
        self.queue.len() + usize::from(self.running.is_some())
    }
    fn start(&mut self, now: f64) {
        if self.running.is_some() {
            return;
        }
        if let Some(job) = self.queue.pop_front() {
            let hit = self.cache.contains(&job.request.prefix);
            let prefill = (0.004
                + job.request.input as f64 * if hit { 0.1 } else { 1.0 } / 30_000.0)
                / self.speed;
            let first = now + prefill;
            let done = first + job.request.output as f64 / (400.0 * self.speed);
            self.running = Some(Running {
                job,
                first,
                done,
                sent: false,
            });
        }
    }
}

#[derive(Default)]
struct Metrics {
    ttfts: Vec<f64>,
    completed: usize,
    good: usize,
    hits: usize,
    weight_sum: f64,
    arrivals: usize,
    last_done: f64,
}

fn advance(workers: &mut [Worker], until: f64, bandit: &mut Bandit, metrics: &mut [Metrics]) {
    loop {
        let next = workers
            .iter()
            .enumerate()
            .filter_map(|(i, worker)| {
                worker
                    .running
                    .as_ref()
                    .map(|r| (i, if r.sent { r.done } else { r.first }))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((index, now)) = next else { break };
        if now > until {
            break;
        }
        let worker = &mut workers[index];
        let running = worker.running.as_mut().unwrap();
        let metric = &mut metrics[running.job.request.phase];
        if !running.sent {
            let ttft = now - running.job.request.at;
            metric.ttfts.push(ttft);
            metric.good += usize::from(ttft <= SLO);
            if let Some(credit) = running.job.credit {
                bandit.observe(credit, ttft);
            }
            let prefix = running.job.request.prefix;
            worker.cache.retain(|&p| p != prefix);
            worker.cache.push_back(prefix);
            if worker.cache.len() > 32 {
                worker.cache.pop_front();
            }
            running.sent = true;
        } else {
            metric.completed += 1;
            metric.last_done = now;
            worker.running = None;
            worker.start(now);
        }
    }
}

fn trace(scenario: &str, seed: u64) -> Vec<Request> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut now = 0.0;
    (0..10_000)
        .map(|i| {
            let phase = i / 2000;
            let rate = match phase {
                0 | 4 => 8.0,
                1 | 3 => 55.0,
                _ => 30.0,
            };
            now += -(1.0 - rng.f64()).ln() / rate;
            let prefix = match scenario {
                "cold-burst" => i + 10_000,
                "cache-working-set" => rng.usize(..256),
                "mixed" if i % 3 == 0 => i + 10_000,
                _ if phase == 1 || phase == 3 => {
                    if rng.f64() < 0.85 {
                        0
                    } else {
                        rng.usize(..64)
                    }
                }
                _ => rng.usize(..64),
            };
            let (input, output) = match scenario {
                "decode-heavy" => (256, 128),
                "cache-working-set" => (16384, 16),
                _ => (4096, 16),
            };
            Request {
                at: now,
                prefix,
                input,
                output,
                phase,
            }
        })
        .collect()
}

fn run(scenario: &str, policy: &str, seed: u64, trace: &[Request]) {
    let mut workers: Vec<_> = (0..8)
        .map(|i| Worker {
            queue: VecDeque::new(),
            running: None,
            cache: if scenario == "cache-working-set" {
                (i * 32..(i + 1) * 32).collect()
            } else {
                (i * 8..(i + 1) * 8).collect()
            },
            speed: if scenario == "heterogeneous" && i < 2 {
                0.4
            } else {
                1.0
            },
        })
        .collect();
    let mut adaptive = AdaptiveRouter::new(Parameters {
        algorithm: if policy == "aimd" {
            Algorithm::Aimd
        } else {
            Algorithm::Sigmoid
        },
        seed: Some(seed),
        ..Default::default()
    })
    .unwrap();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut bandit = Bandit::new();
    let mut metrics: Vec<Metrics> = (0..5).map(|_| Metrics::default()).collect();
    for request in trace {
        advance(&mut workers, request.at, &mut bandit, &mut metrics);
        let rows: Vec<_> = workers
            .iter()
            .map(|w| Candidate {
                affinity: if w.cache.contains(&request.prefix) {
                    0.9
                } else {
                    0.0
                },
                active_requests: w.load(),
            })
            .collect();
        let min = rows.iter().map(|r| r.active_requests).min().unwrap();
        let max = rows.iter().map(|r| r.active_requests).max().unwrap();
        let credit = if policy.contains("ducb") {
            let context = if policy == "contextual-ducb" {
                usize::from(rows.iter().any(|r| r.affinity > 0.5))
                    + 2 * usize::from(rows.iter().map(|r| r.active_requests).sum::<usize>() > 64)
            } else {
                0
            };
            Some(bandit.choose(context))
        } else {
            None
        };
        let (selected, weight) = if policy == "sigmoid" || policy == "aimd" {
            let selected = adaptive
                .select(Duration::from_secs_f64(request.at), rows.iter().copied())
                .unwrap();
            (selected, adaptive.snapshot().distribution_weight)
        } else {
            let weight = credit.map_or_else(
                || match policy {
                    "static-0.1" => 0.1,
                    "static-0.5" => 0.5,
                    "static-0.9" => 0.9,
                    "least-load" => 1.0,
                    _ => unreachable!(),
                },
                |c| WEIGHTS[c.arm],
            );
            (
                pick_weighted(rows.iter().copied(), min, max, weight, 8.0, &mut rng).unwrap(),
                weight,
            )
        };
        let m = &mut metrics[request.phase];
        m.hits += usize::from(rows[selected].affinity > 0.0);
        m.weight_sum += weight;
        m.arrivals += 1;
        workers[selected].queue.push_back(Job {
            request: request.clone(),
            credit,
        });
        workers[selected].start(request.at);
    }
    advance(&mut workers, f64::INFINITY, &mut bandit, &mut metrics);
    assert!(workers.iter().all(|w| w.load() == 0));
    assert!(bandit.pending.iter().flatten().all(|&n| n == 0));
    let all_count: usize = metrics.iter().map(|m| m.completed).sum();
    let all_good: usize = metrics.iter().map(|m| m.good).sum();
    let all_hits: usize = metrics.iter().map(|m| m.hits).sum();
    let all_weights: f64 = metrics.iter().map(|m| m.weight_sum).sum();
    let all_done = metrics.iter().map(|m| m.last_done).fold(0.0, f64::max);
    let mut all_ttfts: Vec<f64> = metrics
        .iter()
        .flat_map(|m| m.ttfts.iter().copied())
        .collect();
    all_ttfts.sort_by(f64::total_cmp);
    let all_mean = all_ttfts.iter().sum::<f64>() / all_count as f64;
    let all_p99 = all_ttfts[all_count * 99 / 100];
    println!(
        "{scenario},{policy},{seed},all,{all_count},{all_mean:.6},{all_p99:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        all_count as f64 / (all_done - trace[0].at),
        all_good as f64 / all_count as f64,
        all_hits as f64 / all_count as f64,
        all_weights / all_count as f64,
        trace.last().unwrap().at - trace[0].at
    );
    for (phase, m) in metrics.iter_mut().enumerate() {
        assert_eq!(m.completed, m.arrivals);
        assert_eq!(m.ttfts.len(), m.arrivals);
        m.ttfts.sort_by(f64::total_cmp);
        let mean = m.ttfts.iter().sum::<f64>() / m.arrivals as f64;
        let p99 = m.ttfts[(m.ttfts.len() * 99 / 100).min(m.ttfts.len() - 1)];
        let first = trace[phase * 2000].at;
        let last = trace[(phase + 1) * 2000 - 1].at;
        // Cohort drain throughput: includes queue drain beyond the phase's final arrival.
        let throughput = m.completed as f64 / (m.last_done - first);
        println!(
            "{scenario},{policy},{seed},{phase},{},{mean:.6},{p99:.6},{throughput:.6},{:.6},{:.6},{:.6},{:.6}",
            m.completed,
            m.good as f64 / m.arrivals as f64,
            m.hits as f64 / m.arrivals as f64,
            m.weight_sum / m.arrivals as f64,
            last - first
        );
    }
}

fn main() {
    println!(
        "scenario,policy,seed,phase,completed,mean_ttft_s,p99_ttft_s,cohort_req_s,slo_fraction,hit_fraction,mean_distribution_weight,arrival_span_s"
    );
    for seed in 1..=5 {
        for scenario in [
            "hotspot-shift",
            "cold-burst",
            "decode-heavy",
            "mixed",
            "heterogeneous",
            "cache-working-set",
        ] {
            let trace = trace(scenario, seed);
            for policy in [
                "static-0.1",
                "static-0.5",
                "static-0.9",
                "least-load",
                "sigmoid",
                "aimd",
                "ducb",
                "contextual-ducb",
            ] {
                run(scenario, policy, seed, &trace);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_exploration_and_out_of_order_credit() {
        let mut bandit = Bandit::new();
        let credits: Vec<_> = (0..5).map(|_| bandit.choose(2)).collect();
        let mut arms: Vec<_> = credits.iter().map(|c| c.arm).collect();
        arms.sort();
        assert_eq!(arms, [0, 1, 2, 3, 4]);
        for &credit in credits.iter().rev() {
            bandit.observe(credit, 0.1);
        }
        assert!(bandit.pending.iter().flatten().all(|&n| n == 0));
        assert_eq!(bandit.counts[0], [0.0; 5]);
        assert!(bandit.counts[2].iter().all(|&n| n > 0.99 && n <= 1.0));
    }

    #[test]
    fn learns_a_new_best_arm_after_a_reward_shift() {
        let mut bandit = Bandit::new();
        for best in [0, 4] {
            let mut late_best = 0;
            for i in 0..10_000 {
                let credit = bandit.choose(0);
                bandit.observe(credit, if credit.arm == best { 0.01 } else { 2.0 });
                late_best += usize::from(i > 9000 && credit.arm == best);
            }
            assert!(late_best > 900);
        }
    }
}
