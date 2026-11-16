//! Prints the epsilon graph stats for seeds 0 to 9, to compare with the plan's
//! targets: mean degree 8 to 10, one component, about 25 % of links between squads.
//!
//! Run: cargo run --release -p swarmsim --example epsilon_stats

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::spawn::spawn_swarm;

fn main() {
    let config: SimConfig = SimConfig::default();
    let n_seeds: u64 = 10;

    // Header: each column has a fixed width so the rows line up.
    println!(
        "{:>4} {:>6} {:>8} {:>6} {:>8} {:>9} {:>9}",
        "seed", "links", "degree", "comps", "largest", "isolated", "between"
    );

    // Running totals for the average row.
    let mut sum_degree: f64 = 0.0;
    let mut sum_components: f64 = 0.0;
    let mut sum_between: f64 = 0.0;

    for seed in 0..n_seeds {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let swarm = spawn_swarm(&config, &mut rng);
        let graph = epsilon_graph(&swarm, &config);
        let stats = stats(&graph, &swarm);

        println!(
            "{:>4} {:>6} {:>8.2} {:>6} {:>8} {:>9} {:>8.0}%",
            seed,
            stats.n_edges,
            stats.mean_degree,
            stats.n_components,
            stats.largest_component,
            stats.isolated,
            100.0 * stats.cross_squad_share
        );

        sum_degree += stats.mean_degree;
        sum_components += stats.n_components as f64;
        sum_between += stats.cross_squad_share;
    }

    let n = n_seeds as f64;
    println!(
        "{:>4} {:>6} {:>8.2} {:>6.1} {:>8} {:>9} {:>8.0}%",
        "mean",
        "",
        sum_degree / n,
        sum_components / n,
        "",
        "",
        100.0 * sum_between / n
    );
    println!("target        8 to 10      1                     ~25%");
}
