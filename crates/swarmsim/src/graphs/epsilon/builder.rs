//! The epsilon-neighbourhood builder.

use crate::config::SimConfig;
use crate::drone::Drone;
use aerie_graph::{Edge, EdgeList};

/// Textbook epsilon graph: one `epsilon` and one Gaussian width `kernel` (sigma)
/// for every pair, whatever the roles. Used by the graph playground GUI.
pub fn textbook_epsilon_graph(swarm: &[Drone], epsilon: f64, kernel: f64) -> EdgeList {
    let mut edges: Vec<Edge> = Vec::new();
    let n = swarm.len();

    let mut src: usize = 0;
    while src < n {
        let mut dest: usize = &src + 1;
        while dest < n {
            let x_diff = (swarm[src].position.x - swarm[dest].position.x)
                * (swarm[src].position.x - swarm[dest].position.x);
            let y_diff = (swarm[src].position.y - swarm[dest].position.y)
                * (swarm[src].position.y - swarm[dest].position.y);
            let distance: f64 = (x_diff + y_diff).sqrt();

            if distance < epsilon {
                let edge_weight: f64 = (-(distance * distance) / (2.0 * kernel * kernel)).exp();
                let new_edge: Edge = Edge {
                    src: src as u32,
                    dst: dest as u32,
                    weight: edge_weight,
                };
                edges.push(new_edge);
            }
            dest += 1;
        }
        src += 1
    }
    let edgeslist: EdgeList = EdgeList { n: n as u32, edges };
    edgeslist
}

/// Builds the comm graph: drones `i < j` are linked when their distance is at most
/// the smaller of their two radio ranges. Weight: exp(-d^2 / (2 sigma^2)), with
/// sigma = half that link range. Nodes are indices into `swarm`, not drone IDs.
pub fn epsilon_graph(swarm: &[Drone], config: &SimConfig) -> EdgeList {
    let n = swarm.len();
    let mut edges: Vec<Edge> = Vec::new();

    for src in 0..n {
        for dst in (src + 1)..n {
            let a = &swarm[src];
            let b = &swarm[dst];

            // Per-pair epsilon: a link must work in both directions.
            let epsilon = a.role.radio_range(config).min(b.role.radio_range(config));
            let distance = a.position.dist(b.position);

            if distance <= epsilon {
                let sigma = epsilon / 2.0;
                let weight = (-(distance * distance) / (2.0 * sigma * sigma)).exp();
                edges.push(Edge {
                    src: src as u32,
                    dst: dst as u32,
                    weight,
                });
            }
        }
    }

    EdgeList { n: n as u32, edges }
}
