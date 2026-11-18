//! The traffic log (Mission 2): who transmits to whom during one flight, as a listener
//! would record it. The listener never sees the content, only sender, receiver and count.
//!
//! The swarm is on station (frozen, no motion yet), and every message travels over the
//! live graph, one hop at a time. Over `steps` time steps:
//! - **Heartbeats**: each drone pings each live neighbour.
//! - **Reports**: each drone sends its readings to its nearest compute drone (fewest
//!   hops), along a shortest path; every drone on the path forwards it.
//! - **Commands**: each compute drone sends orders back to the drones it serves, along the
//!   same paths reversed.
//! - **Fusion**: compute drones share results with every other compute drone they can
//!   reach, which crosses the relay backbone between squads.
//!
//! Counts are Poisson draws (rate × steps per flow), so the log is noisy like a real one.
//! The roles leave fingerprints on purpose: relays forward a lot, compute drones are
//! where reports end, recon drones talk most. Mission 2 measures what a listener can
//! recover from that.

use std::collections::{BTreeMap, VecDeque};

use aerie_graph::{Edge, EdgeList};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Poisson};

use crate::drone::{Drone, Role};

/// How busy the swarm is. Rates are messages per time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrafficConfig {
    /// Length of the flight, time steps.
    pub steps: u32,
    /// Heartbeats from a drone to each live neighbour.
    pub heartbeat: f64,
    /// Reports from one drone, by role (recon, RF sensing, relay, compute).
    pub report: [f64; 4],
    /// Commands from a compute drone to each drone it serves.
    pub command: f64,
    /// Fusion messages from a compute drone to each other reachable compute drone.
    pub fusion: f64,
}

impl Default for TrafficConfig {
    fn default() -> Self {
        TrafficConfig {
            steps: 200,
            heartbeat: 0.05,
            report: [0.5, 0.4, 0.05, 0.0],
            command: 0.05,
            fusion: 0.1,
        }
    }
}

/// Messages from `src` to `dst` (a live link, one direction) over the flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flow {
    pub src: u32,
    pub dst: u32,
    pub count: u32,
}

/// What the listener recorded.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrafficLog {
    /// Number of drones.
    pub n: u32,
    /// One entry per (sender, receiver) pair that exchanged messages, sorted by
    /// (src, dst). Only live links appear.
    pub flows: Vec<Flow>,
    /// Reports that found no compute drone (the sender is cut off from all of them).
    pub dropped: u32,
}

impl TrafficLog {
    /// The undirected traffic graph: one edge per pair that talked, weight = messages in
    /// both directions. Same storage as every other graph (src < dst, sorted).
    pub fn undirected(&self) -> EdgeList {
        let mut total: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for f in &self.flows {
            *total
                .entry((f.src.min(f.dst), f.src.max(f.dst)))
                .or_default() += f.count;
        }
        EdgeList {
            n: self.n,
            edges: total
                .into_iter()
                .map(|((src, dst), c)| Edge {
                    src,
                    dst,
                    weight: c as f64,
                })
                .collect(),
        }
    }

    /// Messages sent and received by each drone.
    pub fn sent_received(&self) -> (Vec<u32>, Vec<u32>) {
        let mut sent = vec![0; self.n as usize];
        let mut received = vec![0; self.n as usize];
        for f in &self.flows {
            sent[f.src as usize] += f.count;
            received[f.dst as usize] += f.count;
        }
        (sent, received)
    }

    pub fn total(&self) -> u64 {
        self.flows.iter().map(|f| f.count as u64).sum()
    }
}

/// Records one flight over the live graph `live`. Deterministic for a given `rng` state.
pub fn traffic_log(
    swarm: &[Drone],
    live: &EdgeList,
    config: &TrafficConfig,
    rng: &mut ChaCha8Rng,
) -> TrafficLog {
    let n = swarm.len();
    let neighbours = neighbour_lists(live, n);
    let mut counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut add_path = |path: &[usize], k: u32| {
        if k == 0 {
            return;
        }
        for hop in path.windows(2) {
            *counts.entry((hop[0] as u32, hop[1] as u32)).or_default() += k;
        }
    };
    let steps = config.steps as f64;

    // Heartbeats: every live link, both directions.
    for e in &live.edges {
        let (a, b) = (e.src as usize, e.dst as usize);
        add_path(&[a, b], draw(config.heartbeat * steps, rng));
        add_path(&[b, a], draw(config.heartbeat * steps, rng));
    }

    // Shortest paths from every compute drone (BFS trees), used by reports, commands and
    // fusion. A tree's parent pointers give the path back to its root.
    let computes: Vec<usize> = (0..n).filter(|&i| swarm[i].role == Role::Compute).collect();
    let trees: Vec<Bfs> = computes.iter().map(|&c| bfs(&neighbours, c)).collect();

    let mut dropped = 0;
    for (i, drone) in swarm.iter().enumerate() {
        let rate = config.report[drone.role as usize];
        // The nearest compute drone in hops; ties go to the first one (smallest index).
        let nearest = (0..computes.len())
            .filter_map(|k| trees[k].dist[i].map(|d| (d, k)))
            .min();
        match nearest {
            // A compute drone serves itself: nothing goes over the air.
            Some((0, _)) => {}
            Some((_, k)) => {
                let up = trees[k].path_to_root(i); // i, ..., compute
                add_path(&up, draw(rate * steps, rng));
                let down: Vec<usize> = up.iter().rev().copied().collect();
                add_path(&down, draw(config.command * steps, rng));
            }
            None => dropped += draw(rate * steps, rng),
        }
    }

    // Fusion between compute drones, both directions, over the shortest path.
    for (a, &ca) in computes.iter().enumerate() {
        for &cb in &computes {
            if ca != cb && trees[a].dist[cb].is_some() {
                let path: Vec<usize> = trees[a].path_to_root(cb).into_iter().rev().collect();
                add_path(&path, draw(config.fusion * steps, rng));
            }
        }
    }

    TrafficLog {
        n: n as u32,
        flows: counts
            .into_iter()
            .map(|((src, dst), count)| Flow { src, dst, count })
            .collect(),
        dropped,
    }
}

/// One Poisson draw with mean `mean` (0 for a non-positive mean).
fn draw(mean: f64, rng: &mut ChaCha8Rng) -> u32 {
    if mean <= 0.0 {
        return 0;
    }
    Poisson::new(mean).map_or(0, |p| p.sample(rng) as u32)
}

/// Neighbours of each drone, sorted, from an undirected edge list.
fn neighbour_lists(graph: &EdgeList, n: usize) -> Vec<Vec<usize>> {
    let mut nb = vec![Vec::new(); n];
    for e in &graph.edges {
        nb[e.src as usize].push(e.dst as usize);
        nb[e.dst as usize].push(e.src as usize);
    }
    for list in &mut nb {
        list.sort_unstable();
    }
    nb
}

/// A breadth-first search tree from one root.
struct Bfs {
    /// Hops from the root, `None` if unreachable.
    dist: Vec<Option<u32>>,
    /// The neighbour one hop closer to the root (the root points to itself).
    parent: Vec<usize>,
}

impl Bfs {
    /// The drones from `i` to the root, both included. `i` must be reachable.
    fn path_to_root(&self, mut i: usize) -> Vec<usize> {
        let mut path = vec![i];
        while self.dist[i] != Some(0) {
            i = self.parent[i];
            path.push(i);
        }
        path
    }
}

/// BFS from `root`. Neighbours are visited in index order, so ties between shortest
/// paths always go the same way.
fn bfs(neighbours: &[Vec<usize>], root: usize) -> Bfs {
    let n = neighbours.len();
    let mut dist = vec![None; n];
    let mut parent: Vec<usize> = (0..n).collect();
    dist[root] = Some(0);
    let mut queue = VecDeque::from([root]);
    while let Some(i) = queue.pop_front() {
        let d = dist[i].expect("queued drones are reached");
        for &j in &neighbours[i] {
            if dist[j].is_none() {
                dist[j] = Some(d + 1);
                parent[j] = i;
                queue.push_back(j);
            }
        }
    }
    Bfs { dist, parent }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drone::{DroneId, SquadId, Vec2};
    use rand::SeedableRng;

    fn drone(i: u32, role: Role) -> Drone {
        Drone::new(
            DroneId(i),
            SquadId(0),
            role,
            Vec2 {
                x: i as f64,
                y: 0.0,
            },
        )
    }

    fn graph(n: u32, links: &[(u32, u32)]) -> EdgeList {
        EdgeList {
            n,
            edges: links
                .iter()
                .map(|&(src, dst)| Edge {
                    src,
                    dst,
                    weight: 1.0,
                })
                .collect(),
        }
    }

    fn count(log: &TrafficLog, src: u32, dst: u32) -> u32 {
        log.flows
            .iter()
            .find(|f| (f.src, f.dst) == (src, dst))
            .map_or(0, |f| f.count)
    }

    /// Only one kind of traffic switched on.
    fn only(heartbeat: f64, report: f64, command: f64, fusion: f64) -> TrafficConfig {
        TrafficConfig {
            steps: 1000,
            heartbeat,
            report: [report; 4],
            command,
            fusion,
        }
    }

    #[test]
    fn reports_flow_to_the_compute_drone_and_relays_forward_them() {
        // 0 recon - 1 relay - 2 compute: drone 0's reports cross 0→1 and 1→2; the relay
        // forwards them on top of its own, so 1→2 carries more than 0→1.
        let swarm = [
            drone(0, Role::Recon),
            drone(1, Role::Relay),
            drone(2, Role::Compute),
        ];
        let g = graph(3, &[(0, 1), (1, 2)]);
        let log = traffic_log(
            &swarm,
            &g,
            &only(0.0, 1.0, 0.0, 0.0),
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(count(&log, 0, 1) > 800);
        assert!(count(&log, 1, 2) > count(&log, 0, 1));
        assert_eq!(
            count(&log, 1, 0),
            0,
            "nothing flows away from the compute drone"
        );
        assert_eq!(log.dropped, 0);
    }

    #[test]
    fn commands_go_back_down_the_same_path() {
        let swarm = [
            drone(0, Role::Recon),
            drone(1, Role::Relay),
            drone(2, Role::Compute),
        ];
        let g = graph(3, &[(0, 1), (1, 2)]);
        let log = traffic_log(
            &swarm,
            &g,
            &only(0.0, 0.0, 1.0, 0.0),
            &mut ChaCha8Rng::seed_from_u64(0),
        );
        assert!(
            count(&log, 2, 1) > count(&log, 1, 0),
            "2→1 carries orders for 0 and 1"
        );
        assert!(count(&log, 1, 0) > 800);
        assert_eq!(count(&log, 0, 1), 0);
    }

    #[test]
    fn messages_only_use_live_links() {
        let swarm: Vec<Drone> = (0..6)
            .map(|i| {
                drone(
                    i,
                    if i % 3 == 2 {
                        Role::Compute
                    } else {
                        Role::Recon
                    },
                )
            })
            .collect();
        let g = graph(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let log = traffic_log(
            &swarm,
            &g,
            &TrafficConfig::default(),
            &mut ChaCha8Rng::seed_from_u64(4),
        );
        for f in &log.flows {
            let (a, b) = (f.src.min(f.dst), f.src.max(f.dst));
            assert!(g.edges.iter().any(|e| (e.src, e.dst) == (a, b)), "{f:?}");
        }
        assert_eq!(log.undirected().n, 6);
    }

    #[test]
    fn a_drone_cut_off_from_every_compute_drone_drops_its_reports() {
        // 0 - 1 (no compute) and 2 compute alone: 0 and 1 cannot deliver.
        let swarm = [
            drone(0, Role::Recon),
            drone(1, Role::Recon),
            drone(2, Role::Compute),
        ];
        let g = graph(3, &[(0, 1)]);
        let log = traffic_log(
            &swarm,
            &g,
            &only(0.0, 1.0, 0.0, 0.0),
            &mut ChaCha8Rng::seed_from_u64(1),
        );
        assert!(log.dropped > 1500);
        assert_eq!(log.total(), 0);
    }

    #[test]
    fn fusion_crosses_between_compute_drones_both_ways() {
        // 0 compute - 1 relay - 2 compute.
        let swarm = [
            drone(0, Role::Compute),
            drone(1, Role::Relay),
            drone(2, Role::Compute),
        ];
        let g = graph(3, &[(0, 1), (1, 2)]);
        let log = traffic_log(
            &swarm,
            &g,
            &only(0.0, 0.0, 0.0, 1.0),
            &mut ChaCha8Rng::seed_from_u64(2),
        );
        for (a, b) in [(0, 1), (1, 2), (2, 1), (1, 0)] {
            assert!(count(&log, a, b) > 800, "{a}→{b}");
        }
    }

    #[test]
    fn same_rng_same_log_and_undirected_sums_both_directions() {
        let swarm = [drone(0, Role::Recon), drone(1, Role::Compute)];
        let g = graph(2, &[(0, 1)]);
        let cfg = TrafficConfig::default();
        let a = traffic_log(&swarm, &g, &cfg, &mut ChaCha8Rng::seed_from_u64(9));
        let b = traffic_log(&swarm, &g, &cfg, &mut ChaCha8Rng::seed_from_u64(9));
        assert_eq!(a, b);
        let u = a.undirected();
        assert_eq!(u.edges.len(), 1);
        assert_eq!(
            u.edges[0].weight,
            (count(&a, 0, 1) + count(&a, 1, 0)) as f64
        );
        let (sent, received) = a.sent_received();
        assert_eq!(sent.iter().sum::<u32>(), received.iter().sum::<u32>());
    }
}
