//! Tests for the epsilon-neighbourhood graph.
//!
//! The tests fix the ranges at short 5 km (sigma 2.5) and relay 9 km (sigma 4.5)
//! with `ranges_5_9()`, so they do not change when the defaults are tuned.
//! Planned values: recon pair at 4 km weighs exp(-16 / 12.5) ~ 0.278,
//! recon-relay at 6 km is not linked (min range 5), relay pair at 8 km
//! weighs exp(-64 / 40.5) ~ 0.206.

use super::{epsilon_graph, textbook_epsilon_graph};
use crate::config::SimConfig;
use crate::drone::{Drone, DroneId, Role, SquadId, Vec2};
use crate::spawn::spawn_swarm;
use aerie_graph::EdgeList;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Default config with the ranges these tests were written for.
fn ranges_5_9() -> SimConfig {
    SimConfig {
        range_short: 5.0,
        range_long: 9.0,
        ..SimConfig::default()
    }
}

/// A drone with the given role at (x, y), in squad 0.
fn drone(id: u32, role: Role, x: f64, y: f64) -> Drone {
    Drone::new(DroneId(id), SquadId(0), role, Vec2 { x, y })
}

/// Two drones `d` km apart on the x axis.
fn pair(a: Role, b: Role, d: f64) -> Vec<Drone> {
    vec![drone(0, a, 0.0, 0.0), drone(1, b, d, 0.0)]
}

/// Weight of the edge between `src` and `dst`, if it exists.
fn weight(g: &EdgeList, src: u32, dst: u32) -> Option<f64> {
    g.edges
        .iter()
        .find(|e| e.src == src && e.dst == dst)
        .map(|e| e.weight)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

// ------------------------------------------------------------ epsilon_graph

#[test]
fn recon_pair_at_4_km() {
    let g = epsilon_graph(&pair(Role::Recon, Role::Recon, 4.0), &ranges_5_9());
    let w = weight(&g, 0, 1).expect("4 km is within the 5 km range");
    assert!(close(w, (-16.0_f64 / 12.5).exp()));
    assert!((w - 0.278).abs() < 1e-3);
}

#[test]
fn recon_relay_at_6_km_is_not_linked() {
    // The relay could reach 9 km, but the recon drone only 5 km: the min rule wins.
    let g = epsilon_graph(&pair(Role::Recon, Role::Relay, 6.0), &ranges_5_9());
    assert!(g.edges.is_empty());
}

#[test]
fn relay_pair_at_8_km() {
    let g = epsilon_graph(&pair(Role::Relay, Role::Relay, 8.0), &ranges_5_9());
    let w = weight(&g, 0, 1).expect("8 km is within the 9 km relay range");
    assert!(close(w, (-64.0_f64 / 40.5).exp()));
    assert!((w - 0.206).abs() < 1e-3);
}

#[test]
fn exactly_at_the_range_is_linked_with_weight_exp_minus_2() {
    // "At most the range": d = epsilon still links. With sigma = epsilon / 2,
    // the weight there is exp(-eps^2 / (2 (eps/2)^2)) = exp(-2), for any epsilon.
    for (role, d) in [(Role::Recon, 5.0), (Role::Relay, 9.0)] {
        let g = epsilon_graph(&pair(role, role, d), &ranges_5_9());
        let w = weight(&g, 0, 1).expect("d = range must link");
        assert!(close(w, (-2.0_f64).exp()));
    }
}

#[test]
fn just_beyond_the_range_is_not_linked() {
    let g = epsilon_graph(&pair(Role::Recon, Role::Recon, 5.001), &ranges_5_9());
    assert!(g.edges.is_empty());
}

#[test]
fn same_place_weighs_one() {
    let g = epsilon_graph(&pair(Role::Compute, Role::Recon, 0.0), &ranges_5_9());
    assert!(close(weight(&g, 0, 1).unwrap(), 1.0));
}

#[test]
fn role_order_does_not_matter() {
    let cfg = ranges_5_9();
    let a = epsilon_graph(&pair(Role::Recon, Role::Relay, 4.0), &cfg);
    let b = epsilon_graph(&pair(Role::Relay, Role::Recon, 4.0), &cfg);
    assert_eq!(a, b);
}

#[test]
fn ranges_come_from_the_config() {
    let cfg = SimConfig {
        range_short: 3.0,
        ..SimConfig::default()
    };
    let g = epsilon_graph(&pair(Role::Recon, Role::Recon, 4.0), &cfg);
    assert!(g.edges.is_empty());
}

#[test]
fn n_counts_isolated_drones_and_empty_swarm_is_fine() {
    let swarm = vec![
        drone(0, Role::Recon, 0.0, 0.0),
        drone(1, Role::Recon, 1.0, 0.0),
        drone(2, Role::Recon, 30.0, 0.0), // far from everyone
    ];
    let g = epsilon_graph(&swarm, &ranges_5_9());
    assert_eq!(g.n, 3);
    assert_eq!(g.edges.len(), 1);

    let empty = epsilon_graph(&[], &ranges_5_9());
    assert_eq!(empty.n, 0);
    assert!(empty.edges.is_empty());
}

#[test]
fn triangle_gives_three_edges_each_stored_once() {
    let swarm = vec![
        drone(0, Role::Recon, 0.0, 0.0),
        drone(1, Role::Recon, 3.0, 0.0),
        drone(2, Role::Recon, 0.0, 4.0),
    ];
    let g = epsilon_graph(&swarm, &ranges_5_9());
    // 0-1 is 3 km, 0-2 is 4 km, 1-2 is 5 km (a 3-4-5 triangle): all linked.
    assert_eq!(g.edges.len(), 3);
    assert!(weight(&g, 0, 1).is_some());
    assert!(weight(&g, 0, 2).is_some());
    assert!(weight(&g, 1, 2).is_some());
}

#[test]
fn spawned_swarm_respects_the_edge_list_invariants() {
    let cfg = SimConfig::default();
    let swarm = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(0));
    let g = epsilon_graph(&swarm, &cfg);
    assert_eq!(g.n as usize, swarm.len());
    assert!(!g.edges.is_empty());

    let mut seen = std::collections::HashSet::new();
    for e in &g.edges {
        assert!(e.src < e.dst, "stored once, src < dst");
        assert!(seen.insert((e.src, e.dst)), "no duplicate edges");
        assert!(e.weight >= (-2.0_f64).exp() - 1e-12 && e.weight <= 1.0);
        let (a, b) = (&swarm[e.src as usize], &swarm[e.dst as usize]);
        let range = a.role.radio_range(&cfg).min(b.role.radio_range(&cfg));
        assert!(a.position.dist(b.position) <= range);
    }
}

#[test]
fn spawned_swarm_graph_is_deterministic() {
    let cfg = SimConfig::default();
    let s1 = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(7));
    let s2 = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(7));
    assert_eq!(epsilon_graph(&s1, &cfg), epsilon_graph(&s2, &cfg));
}

// --------------------------------------------------- textbook_epsilon_graph

#[test]
fn textbook_matches_swarm_when_parameters_agree() {
    // A recon pair with eps = 5, sigma = 2.5 is the same in both builders.
    let swarm = pair(Role::Recon, Role::Recon, 4.0);
    let a = textbook_epsilon_graph(&swarm, 5.0, 2.5);
    let b = epsilon_graph(&swarm, &ranges_5_9());
    assert_eq!(a, b);
}

#[test]
fn textbook_ignores_roles() {
    // One eps for everyone: the 8 km relay pair is lost at eps = 5...
    let relays = pair(Role::Relay, Role::Relay, 8.0);
    assert!(textbook_epsilon_graph(&relays, 5.0, 2.5).edges.is_empty());
    // ...and the 6 km recon-relay pair links at eps = 7, which D00 refuses.
    let mixed = pair(Role::Recon, Role::Relay, 6.0);
    assert_eq!(textbook_epsilon_graph(&mixed, 7.0, 3.5).edges.len(), 1);
}

#[test]
fn textbook_sigma_only_changes_weights() {
    let swarm = pair(Role::Recon, Role::Recon, 2.0);
    let narrow = textbook_epsilon_graph(&swarm, 5.0, 1.0);
    let wide = textbook_epsilon_graph(&swarm, 5.0, 4.0);
    assert_eq!(narrow.edges.len(), wide.edges.len());
    // exp(-4 / 2) vs exp(-4 / 32): a wider kernel gives a heavier edge.
    assert!(close(narrow.edges[0].weight, (-2.0_f64).exp()));
    assert!(close(wide.edges[0].weight, (-0.125_f64).exp()));
}
