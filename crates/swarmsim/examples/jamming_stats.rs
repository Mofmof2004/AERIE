//! Runs the whole of step 2 on seeds 0 to 99, in the dataset order (spawn, place,
//! truth, readings, all from one RNG), and prints what jamming does to each swarm.
//! The evidence for decision D02.
//!
//! Run: cargo run --release -p swarmsim --example jamming_stats
//!
//! Columns:
//! - jammed:    share of drones whose true power is at least the threshold
//! - relays:    jammed relays (the backbone, see D01)
//! - lost:      share of links removed by fail_links
//! - pieces:    components among the free drones (jammed drones are isolated and not counted)
//! - largest:   size of the biggest component after jamming
//! - stranded:  free drones outside the biggest component
//! - disagree:  drones whose reading is on the other side of the threshold from their label

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::jamming::{fail_links, jammed_labels, place_jamming, readings, true_power};
use swarmsim::spawn::spawn_swarm;

fn main() {
    let config = SimConfig::default();
    let n_seeds: u64 = 100;
    let threshold = config.jam_threshold * config.jam_p0;

    println!(
        "{:>4} {:>7} {:>6} {:>6} {:>6} {:>7} {:>8} {:>8}",
        "seed", "jammed", "relays", "lost", "pieces", "largest", "stranded", "disagree"
    );

    let mut jammed_shares: Vec<f64> = Vec::new();
    let mut sum_lost = 0.0;
    let mut sum_stranded = 0.0;
    let mut sum_disagree = 0.0;
    let mut sum_relays = 0.0;
    let mut split_seeds = 0;
    let mut split_before = 0;

    for seed in 0..n_seeds {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(&config, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &config, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, &config);
        let reads = readings(&truth, &config, &mut rng);
        let labels = jammed_labels(&truth, &config);

        let graph = epsilon_graph(&swarm, &config);
        let after = fail_links(&graph, &labels);
        let before_stats = stats(&graph, &swarm);
        let after_stats = stats(&after, &swarm);

        let n = swarm.len();
        let n_jammed = labels.iter().filter(|&&l| l).count();
        let relays = swarm
            .iter()
            .zip(&labels)
            .filter(|(d, l)| **l && d.role == Role::Relay)
            .count();
        let lost = 1.0 - after_stats.n_edges as f64 / before_stats.n_edges as f64;
        let pieces = after_stats.n_components - n_jammed;
        let stranded = n - n_jammed - after_stats.largest_component;
        let disagree = reads
            .iter()
            .zip(&labels)
            .filter(|(r, l)| (**r >= threshold) != **l)
            .count();

        println!(
            "{:>4} {:>6.0}% {:>6} {:>5.0}% {:>6} {:>7} {:>8} {:>8}",
            seed,
            100.0 * n_jammed as f64 / n as f64,
            relays,
            100.0 * lost,
            pieces,
            after_stats.largest_component,
            stranded,
            disagree
        );

        jammed_shares.push(n_jammed as f64 / n as f64);
        sum_lost += lost;
        sum_stranded += stranded as f64;
        sum_disagree += disagree as f64;
        sum_relays += relays as f64;
        if before_stats.n_components > 1 {
            split_before += 1;
        }
        if pieces > 1 {
            split_seeds += 1;
        }
    }

    let k = n_seeds as f64;
    let min = jammed_shares.iter().copied().fold(f64::INFINITY, f64::min);
    let max = jammed_shares.iter().copied().fold(0.0, f64::max);
    let mean = jammed_shares.iter().sum::<f64>() / k;
    println!();
    println!(
        "jammed share: min {:.0} %, mean {:.1} %, max {:.0} %",
        100.0 * min,
        100.0 * mean,
        100.0 * max
    );
    println!("relays jammed per seed: {:.2}", sum_relays / k);
    println!("links lost: {:.1} %", 100.0 * sum_lost / k);
    println!("free drones stranded (outside the largest piece): {:.2}", sum_stranded / k);
    println!(
        "free drones split into 2+ pieces: {split_seeds} / {n_seeds} seeds (before jamming: {split_before})"
    );
    println!("reading on the wrong side of the threshold: {:.2} drones per seed", sum_disagree / k);
}
