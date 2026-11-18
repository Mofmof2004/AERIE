//! The Mission 2 traffic log on the 300-drone swarm, seeds 0 to 99: does every seed build,
//! and what does a listener see per role?
//!
//! Run: cargo run --release -p swarmsim --example traffic_stats
//!
//! Prints, averaged over the seeds:
//! - the traffic graph: pairs that talked, mean degree, pieces, messages per flight;
//! - per role: messages sent and received per drone, and the share of what it sends that
//!   it forwards for others (sent − own reports, heartbeats and commands it originates);
//! - reports dropped because the sender is cut off from every compute drone.

use std::time::Instant;

use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;
use swarmsim::traffic::TrafficConfig;

fn main() {
    let config = SimConfig::with_squads(30);
    let traffic = TrafficConfig::default();
    let seeds = 0..100_u64;
    let start = Instant::now();

    let mut pairs = 0.0;
    let mut degree = 0.0;
    let mut pieces = 0.0;
    let mut messages = 0.0;
    let mut dropped = 0.0;
    let mut generated = 0.0;
    let mut sent = [0.0_f64; 4];
    let mut received = [0.0_f64; 4];
    let mut count = [0.0_f64; 4];

    for seed in seeds.clone() {
        let snap = Snapshot::build(seed, &config);
        let log = snap.traffic(&traffic);
        let g = log.undirected();
        let mut comps = g.components();
        comps.sort_unstable();
        comps.dedup();

        pairs += g.edges.len() as f64;
        degree += 2.0 * g.edges.len() as f64 / snap.n() as f64;
        pieces += comps.len() as f64;
        messages += log.total() as f64;
        dropped += log.dropped as f64;
        generated += log.dropped as f64
            + snap
                .swarm
                .iter()
                .map(|d| traffic.report[d.role as usize] * traffic.steps as f64)
                .sum::<f64>();

        let (s, r) = log.sent_received();
        for (i, d) in snap.swarm.iter().enumerate() {
            let k = d.role as usize;
            sent[k] += s[i] as f64;
            received[k] += r[i] as f64;
            count[k] += 1.0;
        }
    }

    let n = seeds.count() as f64;
    println!(
        "300 drones ({} × {} km), {} seeds, {:.1} s",
        config.arena_w,
        config.arena_h,
        n,
        start.elapsed().as_secs_f64()
    );
    println!(
        "traffic graph: {:.0} pairs talked, mean degree {:.2}, {:.2} pieces, {:.0} messages per flight",
        pairs / n,
        degree / n,
        pieces / n,
        messages / n
    );
    println!(
        "reports dropped (no compute drone reachable): {:.1} % of all reports",
        100.0 * dropped / generated.max(1.0)
    );
    println!(
        "\n{:<12} {:>10} {:>10} {:>10}",
        "role", "sent", "received", "drones"
    );
    for role in Role::ALL {
        let k = role as usize;
        println!(
            "{:<12} {:>10.0} {:>10.0} {:>10.1}",
            format!("{role:?}"),
            sent[k] / count[k],
            received[k] / count[k],
            count[k] / n
        );
    }
    println!("\nsent / received: messages per drone per flight, averaged over drones of that role");
}
