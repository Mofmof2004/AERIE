//! The link model (decision D03): a link direction works when the sender's signal
//! beats noise plus jamming at the receiver. The live graph is the part of the
//! geometric graph where both directions work. One-way directions are kept too, in two
//! separate lists (decision D03, option b): geometric links cut one way by jamming, and
//! directions between drones whose ranges differ (a loud relay heard by a recon drone
//! that cannot answer). A mission picks the list it needs.

use crate::config::SimConfig;
use crate::drone::Drone;
use aerie_graph::Edge;
use aerie_graph::EdgeList;

/// Signal from `sender` at `receiver`, divided by noise plus jamming at the receiver.
/// The direction works when this is at least 1. Uses the sender's power and scale.
fn margin(sender: &Drone, receiver: &Drone, jam_at_receiver: f64, config: &SimConfig) -> f64 {
    let d = sender.position.dist(receiver.position);
    let signal =
        sender.role.tx_power(config) / (1.0 + (d / sender.role.signal_scale(config)).powi(2));
    signal / (config.noise_floor + jam_at_receiver)
}

/// The comm graph under jamming.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveGraph {
    /// Links that work in both directions, with their Gaussian weights. Same `n` as the
    /// geometric graph: drone `i` is still index `i`.
    pub graph: EdgeList,
    /// One per edge of `graph`, in the same order: the smaller of the two margins
    /// (how close the link is to breaking; always at least 1).
    pub margins: Vec<f64>,
    /// Geometric links that jamming cuts one way, as (sender, receiver), in edge order.
    /// The receiver is the jammed end: it can be heard but cannot hear.
    pub cut_one_way: Vec<(u32, u32)>,
    /// Directions between drones that are not geometric neighbours, as (sender, receiver),
    /// in (i, j) order: the longer-range drone (a relay) is heard beyond the other's range.
    /// Many: about 720 per seed with the D03 defaults, against about 370 live links.
    pub range_one_way: Vec<(u32, u32)>,
}

/// Which links survive jamming, and which directions work one way only. `truth` is the
/// true jamming power at each drone.
///
/// Two-way links can only come from the geometric graph (`epsilon_graph`, the smaller of
/// the two ranges): jamming only shrinks ranges. One-way directions also come from pairs
/// outside it, within the larger range: there only the longer-range drone can be heard.
pub fn live_graph(
    geometric: &EdgeList,
    swarm: &[Drone],
    truth: &[f64],
    config: &SimConfig,
) -> LiveGraph {
    let mut edges: Vec<Edge> = Vec::new();
    let mut margins: Vec<f64> = Vec::new();
    let mut cut_one_way: Vec<(u32, u32)> = Vec::new();
    let mut range_one_way: Vec<(u32, u32)> = Vec::new();

    for edge in &geometric.edges {
        let i = edge.src as usize;
        let j = edge.dst as usize;

        // Each direction is jammed at its receiver.
        let m_ij = margin(&swarm[i], &swarm[j], truth[j], config);
        let m_ji = margin(&swarm[j], &swarm[i], truth[i], config);

        if m_ij >= 1.0 && m_ji >= 1.0 {
            edges.push(edge.clone());
            margins.push(m_ij.min(m_ji));
        } else if m_ij >= 1.0 {
            cut_one_way.push((edge.src, edge.dst));
        } else if m_ji >= 1.0 {
            cut_one_way.push((edge.dst, edge.src));
        }
        // Neither direction works: the link is dead.
    }

    // Pairs outside the geometric graph but within the larger range. The shorter-range
    // drone cannot reach that far, jammed or not, so only one direction can work: from
    // the longer-range drone. Same `<=` as epsilon_graph, so no pair is counted twice.
    let n = swarm.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = swarm[i].position.dist(swarm[j].position);
            let range_i = swarm[i].role.radio_range(config);
            let range_j = swarm[j].role.radio_range(config);
            if d <= range_i.min(range_j) || d > range_i.max(range_j) {
                continue; // a geometric pair (done above), or out of reach of both
            }
            let (sender, receiver) = if range_i > range_j { (i, j) } else { (j, i) };
            if margin(&swarm[sender], &swarm[receiver], truth[receiver], config) >= 1.0 {
                range_one_way.push((sender as u32, receiver as u32));
            }
        }
    }

    let live_graph = EdgeList {
        n: geometric.n,
        edges,
    };
    LiveGraph {
        graph: live_graph,
        margins,
        cut_one_way,
        range_one_way,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drone::{DroneId, Role, SquadId, Vec2};
    use crate::graphs::epsilon::epsilon_graph;
    use crate::jamming::{place_jamming, true_power};
    use crate::spawn::spawn_swarm;
    use aerie_graph::Edge;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Pinned values, so tuning the defaults cannot break these tests:
    /// N0 = 0.05, recon scale 1 (tx 0.3625), relay scale 0.3, ranges 2.5 / 16 km.
    fn config() -> SimConfig {
        SimConfig {
            noise_floor: 0.05,
            range_short: 2.5,
            range_long: 16.0,
            signal_scale_recon: 1.0,
            signal_scale_rf: 1.0,
            signal_scale_relay: 0.3,
            signal_scale_compute: 1.0,
            ..SimConfig::default()
        }
    }

    fn drone(i: u32, role: Role, x: f64) -> Drone {
        Drone::new(DroneId(i), SquadId(0), role, Vec2 { x, y: 0.0 })
    }

    /// Two drones on the x axis, `d` km apart, linked in the geometric graph.
    fn pair(role: Role, d: f64) -> (Vec<Drone>, EdgeList) {
        let swarm = vec![drone(0, role, 0.0), drone(1, role, d)];
        let graph = EdgeList {
            n: 2,
            edges: vec![Edge {
                src: 0,
                dst: 1,
                weight: 0.5,
            }],
        };
        (swarm, graph)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn is_geometric(g: &EdgeList, a: usize, b: usize) -> bool {
        let pair = (a.min(b) as u32, a.max(b) as u32);
        g.edges.iter().any(|e| (e.src, e.dst) == pair)
    }

    /// Two drones on the x axis with no geometric link: everything comes from the
    /// larger-range pass.
    fn apart(a: Role, b: Role, d: f64, jam: [f64; 2]) -> LiveGraph {
        let cfg = config();
        let swarm = vec![drone(0, a, 0.0), drone(1, b, d)];
        let geometric = crate::graphs::epsilon::epsilon_graph(&swarm, &cfg);
        assert!(geometric.edges.is_empty());
        live_graph(&geometric, &swarm, &jam, &cfg)
    }

    #[test]
    fn hand_margins_for_two_recon_drones() {
        // margin = tx / (1 + d²) / (N0 + J), tx = 0.05 * (1 + 2.5²) = 0.3625.
        let cfg = config();
        let (a, b) = (drone(0, Role::Recon, 0.0), drone(1, Role::Recon, 2.5));
        assert!(close(margin(&a, &b, 0.0, &cfg), 1.0)); // at the range: exactly 1
        let b = drone(1, Role::Recon, 1.0);
        assert!(close(margin(&a, &b, 0.0, &cfg), 3.625)); // 0.3625 / 2 / 0.05
        assert!(close(margin(&a, &b, 0.05, &cfg), 1.8125)); // jamming = N0 halves it
    }

    #[test]
    fn no_jammer_gives_the_geometric_graph() {
        // The plan's first check: with zero jamming, every geometric link is live.
        let scales = SimConfig {
            signal_scale_recon: 0.5,
            signal_scale_rf: 0.7,
            signal_scale_relay: 0.3,
            signal_scale_compute: 1.0,
            ..SimConfig::default()
        };
        for cfg in [config(), scales] {
            for seed in 0..10 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let swarm = spawn_swarm(&cfg, &mut rng);
                let geometric = epsilon_graph(&swarm, &cfg);
                let live = live_graph(&geometric, &swarm, &vec![0.0; swarm.len()], &cfg);
                assert_eq!(live.graph, geometric, "seed {seed}");
                // With no jammer nothing is cut; the only one-way directions are relays
                // heard beyond the short range: never a geometric pair, sent by a relay.
                assert!(live.cut_one_way.is_empty());
                for &(s, r) in &live.range_one_way {
                    let (s, r) = (s as usize, r as usize);
                    assert_eq!(swarm[s].role, Role::Relay);
                    assert_ne!(swarm[r].role, Role::Relay);
                    assert!(!is_geometric(&geometric, s, r));
                }
            }
        }
    }

    #[test]
    fn near_the_jammer_a_drone_is_heard_but_cannot_hear() {
        // Drone 0 sits in full jamming (J = 1), drone 1 in none, 1 km apart.
        // 0 -> 1: 0.3625 / 2 / 0.05 = 3.6, works. 1 -> 0: 0.3625 / 2 / 1.05 = 0.17, fails.
        let cfg = config();
        let (swarm, graph) = pair(Role::Recon, 1.0);
        let live = live_graph(&graph, &swarm, &[1.0, 0.0], &cfg);
        assert!(live.graph.edges.is_empty());
        assert_eq!(live.cut_one_way, vec![(0, 1)]);
        assert!(live.range_one_way.is_empty());
        // Swapped jamming: the one-way link points the other way.
        let live = live_graph(&graph, &swarm, &[0.0, 1.0], &cfg);
        assert_eq!(live.cut_one_way, vec![(1, 0)]);
    }

    #[test]
    fn both_ends_jammed_kills_the_link_completely() {
        let cfg = config();
        let (swarm, graph) = pair(Role::Recon, 1.0);
        let live = live_graph(&graph, &swarm, &[1.0, 1.0], &cfg);
        assert!(live.graph.edges.is_empty());
        assert!(live.margins.is_empty());
        assert!(live.cut_one_way.is_empty()); // dead, not one-way
        assert!(live.range_one_way.is_empty());
        assert_eq!(live.graph.n, 2); // drones stay
    }

    #[test]
    fn relays_outlast_recon_drones() {
        // Same distance (2 km), same jamming at both ends (0.5): recon dies, relays live.
        // Recon: 0.3625 / 5 / 0.55 = 0.13. Relay: tx 142.3, 142.3 / 45.4 / 0.55 = 5.7.
        let cfg = config();
        let jam = [0.5, 0.5];
        let (swarm, graph) = pair(Role::Recon, 2.0);
        assert!(
            live_graph(&graph, &swarm, &jam, &cfg)
                .graph
                .edges
                .is_empty()
        );
        let (swarm, graph) = pair(Role::Relay, 2.0);
        let live = live_graph(&graph, &swarm, &jam, &cfg);
        assert_eq!(live.graph.edges.len(), 1);
        assert!(live.margins[0] > 5.0);
    }

    #[test]
    fn stored_margin_is_the_weaker_direction() {
        // A relay and a recon drone 1 km apart, no jamming: the recon's weaker signal
        // (1 -> 0) sets the stored margin.
        let cfg = config();
        let swarm = vec![drone(0, Role::Relay, 0.0), drone(1, Role::Recon, 1.0)];
        let (_, graph) = pair(Role::Recon, 1.0);
        let live = live_graph(&graph, &swarm, &[0.0, 0.0], &cfg);
        let relay_to_recon = margin(&swarm[0], &swarm[1], 0.0, &cfg);
        let recon_to_relay = margin(&swarm[1], &swarm[0], 0.0, &cfg);
        assert!(recon_to_relay < relay_to_recon);
        assert_eq!(live.margins, vec![recon_to_relay]);
    }

    #[test]
    fn placed_jamming_invariants() {
        // On real seeds: live is inside geometric, one margin per live edge, every
        // margin at least 1, and n unchanged.
        let cfg = SimConfig::default();
        for seed in 0..10 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let swarm = spawn_swarm(&cfg, &mut rng);
            let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
            let truth = true_power(&swarm, &jammer, &ridge, &cfg);
            let geometric = epsilon_graph(&swarm, &cfg);
            let live = live_graph(&geometric, &swarm, &truth, &cfg);
            assert_eq!(live.graph.n, geometric.n);
            assert_eq!(live.margins.len(), live.graph.edges.len());
            assert!(live.margins.iter().all(|&m| m >= 1.0));
            assert!(live.graph.edges.iter().all(|e| geometric.edges.contains(e)));
            // Cut links are geometric pairs, range directions never are.
            let geo = |&&(a, b): &&(u32, u32)| is_geometric(&geometric, a as usize, b as usize);
            assert!(live.cut_one_way.iter().all(|p| geo(&p)));
            assert!(!live.range_one_way.iter().any(|p| geo(&p)));
            // Live links and cut links share the geometric pairs.
            assert!(live.graph.edges.len() + live.cut_one_way.len() <= geometric.edges.len());
            // No pair appears twice across the two lists.
            let all: Vec<(u32, u32)> = live
                .cut_one_way
                .iter()
                .chain(&live.range_one_way)
                .map(|&(a, b)| (a.min(b), a.max(b)))
                .collect();
            let mut pairs = all.clone();
            pairs.sort();
            pairs.dedup();
            assert_eq!(pairs.len(), all.len());
            assert!(
                live.graph.edges.len() < geometric.edges.len(),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn deterministic() {
        let cfg = SimConfig::default();
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(3);
            let swarm = spawn_swarm(&cfg, &mut rng);
            let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
            let truth = true_power(&swarm, &jammer, &ridge, &cfg);
            live_graph(&epsilon_graph(&swarm, &cfg), &swarm, &truth, &cfg)
        };
        assert_eq!(run(), run());
    }

    // ------------------------------------- option (b): one-way beyond the shorter range

    #[test]
    fn a_relay_is_heard_beyond_the_recon_range() {
        // Prediction 1. Relay tx = 0.05 (1 + (16 / 0.3)²) = 142.3; at 5 km the signal is
        // 142.3 / (1 + (5 / 0.3)²) = 0.51, margin 0.51 / 0.05 = 10.2. Recon cannot answer.
        let live = apart(Role::Relay, Role::Recon, 5.0, [0.0, 0.0]);
        assert_eq!(live.range_one_way, vec![(0, 1)]);
        assert!(live.cut_one_way.is_empty());
        assert!(live.graph.edges.is_empty());
        // Same pair, listed the other way round: the sender is still the relay.
        let live = apart(Role::Recon, Role::Relay, 5.0, [0.0, 0.0]);
        assert_eq!(live.range_one_way, vec![(1, 0)]);
    }

    #[test]
    fn beyond_both_ranges_nothing_is_heard() {
        // Prediction 2: two relays 20 km apart (range 16). Prediction 3: two recon
        // drones 4 km apart (range 2.5). Same ranges, so no extra pass either.
        assert!(
            apart(Role::Relay, Role::Relay, 20.0, [0.0, 0.0])
                .range_one_way
                .is_empty()
        );
        assert!(
            apart(Role::Recon, Role::Recon, 4.0, [0.0, 0.0])
                .range_one_way
                .is_empty()
        );
        // A relay 17 km away is beyond its own 16 km range too.
        assert!(
            apart(Role::Relay, Role::Recon, 17.0, [0.0, 0.0])
                .range_one_way
                .is_empty()
        );
    }

    #[test]
    fn jamming_at_the_recon_drowns_the_relay() {
        // Prediction 4: jamming 1.0 at the recon drone: 0.51 / 1.05 = 0.49 < 1.
        let live = apart(Role::Relay, Role::Recon, 5.0, [0.0, 1.0]);
        assert!(live.range_one_way.is_empty());
        // Jamming at the relay does not matter: it only sends.
        let live = apart(Role::Relay, Role::Recon, 5.0, [1.0, 0.0]);
        assert_eq!(live.range_one_way, vec![(0, 1)]);
    }
}
