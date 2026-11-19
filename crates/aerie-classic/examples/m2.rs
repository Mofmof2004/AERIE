//! Mission 2 runner: DeepWalk and Node2Vec (experiment 1: q = 0.5 and q = 2) on the
//! 300-drone traffic graph, scored the plan's three ways
//! (`aerie_classic::m2`; the swarm viewer's Mission 2 view runs the same thing with a
//! button).
//!
//! Run: cargo run --release -p aerie-classic --example m2
//!
//! Test seeds 90-99, default DeepWalk settings. Prints the training pairs per epoch and
//! the time of one DeepWalk run (first seed), then the table (mean ± spread over the
//! seeds); writes results/m2/embeddings.csv.

use std::time::Instant;

use aerie_classic::embed::deepwalk::{DeepWalkConfig, deepwalk};
use aerie_classic::embed::walks::random_walks;
use aerie_classic::m2::{
    EXPERIMENT_BIASES, Progress, plan_config, summarize, sweep, table, write_rows,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::snapshot::{Snapshot, TEST_SEEDS};
use swarmsim::traffic::TrafficConfig;

/// Training pairs in one epoch: every position of every walk, with every other position
/// at most `window` steps away.
fn pairs_per_epoch(walks: &[Vec<usize>], window: usize) -> usize {
    walks
        .iter()
        .map(|w| {
            (0..w.len())
                .map(|t| t.min(window) + (w.len() - 1 - t).min(window))
                .sum::<usize>()
        })
        .sum()
}

fn main() -> std::io::Result<()> {
    let sim = plan_config();
    let dw = DeepWalkConfig::default();
    let started = Instant::now();

    // Pairs per epoch and time per run, on the first seed (same walks as deepwalk()).
    let first = Snapshot::build(TEST_SEEDS.start, &sim);
    let graph = first.traffic(&TrafficConfig::default()).undirected();
    let mut walk_rng = ChaCha8Rng::seed_from_u64(TEST_SEEDS.start);
    walk_rng.set_stream(1);
    let walks = random_walks(&graph, dw.walks_per_node, dw.walk_length, &mut walk_rng);
    let short = walks.iter().filter(|w| w.len() < dw.walk_length).count();
    let t = Instant::now();
    deepwalk(&graph, &dw, TEST_SEEDS.start);
    println!(
        "Seed {}: {} drones, {} links, {} walks ({short} cut short), {} training pairs per \
         epoch, one DeepWalk run {:.2} s\n",
        TEST_SEEDS.start,
        graph.n,
        graph.edges.len(),
        walks.len(),
        pairs_per_epoch(&walks, dw.skipgram.window),
        t.elapsed().as_secs_f64()
    );

    let rows = sweep(
        TEST_SEEDS,
        &sim,
        &dw,
        &EXPERIMENT_BIASES,
        &Progress::default(),
    )
    .map_err(std::io::Error::other)?;
    println!(
        "DeepWalk and Node2Vec on the traffic graph, {} drones, test seeds {}-{} (mean ± spread over seeds)\n",
        graph.n,
        TEST_SEEDS.start,
        TEST_SEEDS.end - 1
    );
    print!("{}", table(&summarize(&rows)));
    let path = write_rows(&rows)?;
    println!(
        "\nWrote {} ({:.1} s in all)",
        path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
