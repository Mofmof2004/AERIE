//! Tests for the kNN graph (union rule, D00).
//!
//! The tests fix the ranges at 5 km / 9 km so the weights do not change when the
//! defaults are tuned. k is set per test.

use super::knn_graph;
use crate::config::SimConfig;
use crate::drone::{Drone, DroneId, Role, SquadId, Vec2};
use crate::spawn::spawn_swarm;
use aerie_graph::EdgeList;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Ranges 5 / 9 km and the given k.
fn config_k(k: usize) -> SimConfig {
    SimConfig {
        range_short: 5.0,
        range_long: 9.0,
        knn_k: k,
        ..SimConfig::default()
    }
}

/// Recon drones at the given points, ids in order.
fn recon_at(points: &[(f64, f64)]) -> Vec<Drone> {
    points
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| Drone::new(DroneId(i as u32), SquadId(0), Role::Recon, Vec2 { x, y }))
        .collect()
}

/// The (src, dst) pairs of a graph, in stored order.
fn pairs(g: &EdgeList) -> Vec<(u32, u32)> {
    g.edges.iter().map(|e| (e.src, e.dst)).collect()
}

/// Number of links of every node.
fn link_counts(g: &EdgeList) -> Vec<usize> {
    g.neighbours().iter().map(|v| v.len()).collect()
}

#[test]
fn line_0_1_3_7_with_k1_is_a_path() {
    // Picks: 0 -> 1, 1 -> 0, 3 -> 1, 7 -> 3. Union: 0-1, 1-3, 3-7.
    let swarm = recon_at(&[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (7.0, 0.0)]);
    let g = knn_graph(&swarm, &config_k(1));
    assert_eq!(pairs(&g), vec![(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn union_keeps_one_way_picks_that_mutual_would_drop() {
    // Only 0-1 is a mutual pick; the other two links exist because one side picked.
    // A mutual graph would leave drones at x = 3 and x = 7 isolated.
    let swarm = recon_at(&[(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (7.0, 0.0)]);
    let g = knn_graph(&swarm, &config_k(1));
    assert!(link_counts(&g).iter().all(|&c| c >= 1), "nobody isolated");
}

#[test]
fn a_hub_can_have_more_than_2k_links() {
    // Centre at the origin, four drones 1 km away in four directions (sqrt 2 km
    // apart from each other). With k = 1 all four pick the centre.
    let swarm = recon_at(&[(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]);
    let g = knn_graph(&swarm, &config_k(1));
    assert_eq!(link_counts(&g), vec![4, 1, 1, 1, 1]);
}

#[test]
fn k_at_least_n_minus_1_gives_the_complete_graph() {
    let swarm = recon_at(&[(0.0, 0.0), (1.0, 0.0), (4.0, 3.0)]);
    let g = knn_graph(&swarm, &config_k(5));
    assert_eq!(pairs(&g), vec![(0, 1), (0, 2), (1, 2)]);
}

#[test]
fn fewer_than_two_drones_gives_no_edges() {
    let empty = knn_graph(&[], &config_k(5));
    assert_eq!((empty.n, empty.edges.len()), (0, 0));
    let one = knn_graph(&recon_at(&[(1.0, 1.0)]), &config_k(5));
    assert_eq!((one.n, one.edges.len()), (1, 0));
}

#[test]
fn ties_go_to_the_lower_index() {
    // Drone 0 sits between drone 1 (x = 1) and drone 2 (x = -1), both 1 km away.
    // The stable sort keeps index order, so 0 picks 1. Drone 2 still picks 0.
    let swarm = recon_at(&[(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]);
    let g = knn_graph(&swarm, &config_k(1));
    assert_eq!(pairs(&g), vec![(0, 1), (0, 2)]);
}

#[test]
fn weight_is_the_epsilon_kernel_even_beyond_radio_range() {
    // Two recon drones 10 km apart: out of radio range, but each is the other's
    // nearest neighbour. sigma = 5 / 2, so weight = exp(-100 / 12.5) = exp(-8).
    let swarm = recon_at(&[(0.0, 0.0), (10.0, 0.0)]);
    let g = knn_graph(&swarm, &config_k(1));
    assert_eq!(g.edges.len(), 1);
    assert!((g.edges[0].weight - (-8.0_f64).exp()).abs() < 1e-12);

    // Inside range it matches the planned value: 4 km recon pair ~ 0.278.
    let near = recon_at(&[(0.0, 0.0), (4.0, 0.0)]);
    let w = knn_graph(&near, &config_k(1)).edges[0].weight;
    assert!((w - 0.278).abs() < 1e-3);
}

#[test]
fn spawned_swarm_degrees_and_invariants() {
    let cfg = config_k(5);
    let swarm = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(0));
    let g = knn_graph(&swarm, &cfg);
    let n = swarm.len();
    assert_eq!(g.n as usize, n);

    // Stored once, src < dst, sorted, no duplicates.
    let p = pairs(&g);
    assert!(p.iter().all(|&(s, d)| s < d));
    assert!(p.windows(2).all(|w| w[0] < w[1]), "sorted and unique");

    // Every drone keeps its own k picks; the mean degree is between k and 2k.
    let counts = link_counts(&g);
    assert!(counts.iter().all(|&c| c >= 5));
    let mean = 2.0 * g.edges.len() as f64 / n as f64;
    assert!((5.0..=10.0).contains(&mean), "mean degree {mean}");
}

#[test]
fn spawned_swarm_graph_is_deterministic() {
    let cfg = config_k(5);
    let s1 = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(3));
    let s2 = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(3));
    assert_eq!(knn_graph(&s1, &cfg), knn_graph(&s2, &cfg));
}
