//! Calibrates the link model (D03): runs step 2 with `live_graph` on seeds 0 to 99 for
//! several values of the noise floor N0 and the signal scales, one row per setting.
//!
//! Run: cargo run --release -p swarmsim --example link_stats
//!
//! Columns (averages over the seeds):
//! - lost:     share of geometric links that are not live
//! - degree:   mean degree of the live graph (geometric graph: about 8.7)
//! - far:      share of links still live when both ends are at least 12 km from the jammer
//!   (the long tail: should stay close to 100 %)
//! - zone:     share of links still live when both ends are in the jammed zone (label)
//! - split:    seeds where the live graph has 2+ pieces of at least 2 drones
//! - strand:   drones outside the largest piece
//! - cut1:     one-way links cut by jamming (geometric pairs)
//! - range1:   one-way directions from the range difference (relay heard beyond 2.5 km)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::jamming::{jammed_labels, place_jamming, true_power};
use swarmsim::links::live_graph;
use swarmsim::spawn::spawn_swarm;

/// The settings to compare: (N0, recon, rf, relay, compute scales).
const SETTINGS: [(f64, f64, f64, f64, f64); 12] = [
    (0.05, 0.5, 0.7, 0.3, 1.0),
    (0.1, 0.5, 0.7, 0.3, 1.0),
    (0.2, 0.5, 0.7, 0.3, 1.0),
    (0.5, 0.5, 0.7, 0.3, 1.0), // chosen (D03)
    (1.0, 0.5, 0.7, 0.3, 1.0),
    (0.2, 0.25, 0.35, 0.3, 0.5),
    (0.5, 0.25, 0.35, 0.3, 0.5),
    (0.2, 1.0, 1.0, 1.0, 1.0),
    (0.5, 1.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 1.0, 1.0),
    (2.0, 0.5, 0.7, 0.3, 1.0),
    (2.0, 1.0, 1.0, 1.0, 1.0),
];

fn main() {
    let n_seeds: u64 = 100;
    println!(
        "{:>5} {:>5} {:>5} {:>5} {:>5} | {:>5} {:>6} {:>5} {:>5} {:>6} {:>6} {:>5} {:>6}",
        "N0",
        "recon",
        "rf",
        "relay",
        "comp",
        "lost",
        "degree",
        "far",
        "zone",
        "split",
        "strand",
        "cut1",
        "range1"
    );

    for &(n0, s_recon, s_rf, s_relay, s_compute) in &SETTINGS {
        let config = SimConfig {
            noise_floor: n0,
            signal_scale_recon: s_recon,
            signal_scale_rf: s_rf,
            signal_scale_relay: s_relay,
            signal_scale_compute: s_compute,
            ..SimConfig::default()
        };
        let mut sum_lost = 0.0;
        let mut sum_degree = 0.0;
        let (mut far_total, mut far_live) = (0usize, 0usize);
        let (mut zone_total, mut zone_live) = (0usize, 0usize);
        let mut split = 0;
        let mut sum_stranded = 0.0;
        let (mut sum_cut, mut sum_range) = (0.0, 0.0);

        for seed in 0..n_seeds {
            // The dataset order: spawn, place, truth.
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let swarm = spawn_swarm(&config, &mut rng);
            let (jammer, ridge) = place_jamming(&swarm, &config, &mut rng);
            let truth = true_power(&swarm, &jammer, &ridge, &config);
            let labels = jammed_labels(&truth, &config);
            let geometric = epsilon_graph(&swarm, &config);
            let live = live_graph(&geometric, &swarm, &truth, &config);
            let n = swarm.len();

            sum_lost += 1.0 - live.graph.edges.len() as f64 / geometric.edges.len() as f64;
            sum_degree += 2.0 * live.graph.edges.len() as f64 / n as f64;

            let is_live =
                |a: u32, b: u32| live.graph.edges.iter().any(|e| (e.src, e.dst) == (a, b));
            let far = |i: usize| swarm[i].position.dist(jammer.position) >= 12.0;
            for e in &geometric.edges {
                let (a, b) = (e.src as usize, e.dst as usize);
                if far(a) && far(b) {
                    far_total += 1;
                    far_live += is_live(e.src, e.dst) as usize;
                }
                if labels[a] && labels[b] {
                    zone_total += 1;
                    zone_live += is_live(e.src, e.dst) as usize;
                }
            }

            // Pieces of the live graph.
            let comp = live.graph.components();
            let mut sizes = vec![0usize; n];
            for &c in &comp {
                sizes[c] += 1;
            }
            let largest = sizes.iter().copied().max().unwrap_or(0);
            if sizes.iter().filter(|&&s| s >= 2).count() >= 2 {
                split += 1;
            }
            sum_stranded += (n - largest) as f64;

            sum_cut += live.cut_one_way.len() as f64;
            sum_range += live.range_one_way.len() as f64;
        }

        let k = n_seeds as f64;
        let pct = |live: usize, total: usize| 100.0 * live as f64 / total.max(1) as f64;
        println!(
            "{:>5} {:>5} {:>5} {:>5} {:>5} | {:>4.0}% {:>6.2} {:>4.0}% {:>4.0}% {:>6} {:>6.1} {:>5.1} {:>6.1}",
            n0,
            s_recon,
            s_rf,
            s_relay,
            s_compute,
            100.0 * sum_lost / k,
            sum_degree / k,
            pct(far_live, far_total),
            pct(zone_live, zone_total),
            split,
            sum_stranded / k,
            sum_cut / k,
            sum_range / k
        );
    }
}
