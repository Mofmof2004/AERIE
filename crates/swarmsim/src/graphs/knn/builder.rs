//! The k-nearest-neighbour builder.

use crate::config::SimConfig;
use crate::drone::Drone;
use aerie_graph::{Edge, EdgeList};

/// kNN graph, union rule (D00): each drone picks its `config.knn_k` nearest drones,
/// and i, j are linked when either one picked the other. Weight: the same Gaussian
/// kernel as the epsilon graph, sigma = half the smaller of the two radio ranges.
/// Unlike the epsilon graph, links can be longer than the radio range.
pub fn knn_graph(swarm: &[Drone], config: &SimConfig) -> EdgeList {
    let n = swarm.len();

    if n < 2 {
        return EdgeList {
            n: n as u32,
            edges: Vec::new(),
        };
    }

    let k = config.knn_k.min(n - 1);

    let mut pairs: Vec<(usize, usize)> = Vec::new();

    for src in 0..n {
        let mut candidates: Vec<(f64, usize)> = Vec::new();

        for dest in 0..n {
            if dest != src {
                let distance = swarm[src].position.dist(swarm[dest].position);
                candidates.push((distance, dest));
            }
        }

        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_distance, dest) in candidates.iter().take(k) {
            pairs.push((src.min(dest), src.max(dest)));
        }
    }

    pairs.sort();
    pairs.dedup();

    let mut edges: Vec<Edge> = Vec::new();
    for (i, j) in pairs {
        let a = &swarm[i];
        let b = &swarm[j];
        let link_range = a.role.radio_range(config).min(b.role.radio_range(config));
        let sigma = link_range / 2.0;
        let distance = a.position.dist(b.position);
        let weight = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
        edges.push(Edge {
            src: i as u32,
            dst: j as u32,
            weight,
        });
    }

    EdgeList { n: n as u32, edges }
}
