//! Summary numbers for a comm graph, to check it against the plan's targets:
//! mean degree 8 to 10, one connected component, about a quarter of links between squads.

use crate::drone::Drone;
use aerie_graph::EdgeList;

/// Summary of one graph built over one swarm.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    pub n_nodes: usize,
    pub n_edges: usize,
    /// Links per drone: 2 * edges / nodes (each link touches two drones).
    pub mean_degree: f64,
    /// Average link weight, 0 if there are no links.
    pub mean_weight: f64,
    pub n_components: usize,
    /// Number of drones in the biggest component.
    pub largest_component: usize,
    /// Drones with no link at all (components of size 1).
    pub isolated: usize,
    /// Fraction of links joining two different squads, from 0 to 1.
    pub cross_squad_share: f64,
}

/// Computes the stats of `graph`. `swarm` is only used to know each drone's squad;
/// node `i` of the graph is `swarm[i]`.
pub fn stats(graph: &EdgeList, swarm: &[Drone]) -> GraphStats {
    let n_nodes: usize = graph.n as usize;
    let n_edges: usize = graph.edges.len();

    // Float division by zero gives NaN, not a crash: guard both divisions.
    let mean_degree: f64 = if n_nodes == 0 {
        0.0
    } else {
        (n_edges as f64 * 2.0) / n_nodes as f64
    };

    let mut weight_sum: f64 = 0.0;
    let mut cross_squad: usize = 0;
    for edge in &graph.edges {
        weight_sum += edge.weight;
        if swarm[edge.src as usize].squad != swarm[edge.dst as usize].squad {
            cross_squad += 1;
        }
    }
    let (mean_weight, cross_squad_share) = if n_edges == 0 {
        (0.0, 0.0)
    } else {
        (
            weight_sum / n_edges as f64,
            cross_squad as f64 / n_edges as f64,
        )
    };

    // components() gives one label per node, numbered 0, 1, 2, ...
    let graph_components: Vec<usize> = graph.components();
    let n_components: usize = if n_nodes == 0 {
        0
    } else {
        *graph_components.iter().max().unwrap_or(&0) + 1
    };

    // How many drones carry each label.
    let mut sizes: Vec<usize> = vec![0; n_components];
    for label in &graph_components {
        sizes[*label] += 1;
    }
    let largest_component: usize = *sizes.iter().max().unwrap_or(&0);
    let isolated: usize = sizes.iter().filter(|size| **size == 1).count();

    GraphStats {
        n_nodes,
        n_edges,
        mean_degree,
        mean_weight,
        n_components,
        largest_component,
        isolated,
        cross_squad_share,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drone::{DroneId, Role, SquadId, Vec2};
    use aerie_graph::Edge;

    fn drone(id: u32, squad: u16) -> Drone {
        Drone::new(
            DroneId(id),
            SquadId(squad),
            Role::Recon,
            Vec2 { x: 0.0, y: 0.0 },
        )
    }

    fn edge(src: u32, dst: u32, weight: f64) -> Edge {
        Edge { src, dst, weight }
    }

    /// Squad 0: drones 0, 1, 2 (triangle). Squad 1: drones 3, 4.
    /// Links: the triangle, 2-3 across squads, drone 4 alone.
    fn fixture() -> (EdgeList, Vec<Drone>) {
        let swarm = vec![
            drone(0, 0),
            drone(1, 0),
            drone(2, 0),
            drone(3, 1),
            drone(4, 1),
        ];
        let graph = EdgeList {
            n: 5,
            edges: vec![
                edge(0, 1, 1.0),
                edge(0, 2, 0.5),
                edge(1, 2, 0.5),
                edge(2, 3, 0.2),
            ],
        };
        (graph, swarm)
    }

    #[test]
    fn counts_and_means() {
        let (graph, swarm) = fixture();
        let s = stats(&graph, &swarm);
        assert_eq!(s.n_nodes, 5);
        assert_eq!(s.n_edges, 4);
        assert!((s.mean_degree - 8.0 / 5.0).abs() < 1e-12);
        assert!((s.mean_weight - 2.2 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn components_largest_and_isolated() {
        let (graph, swarm) = fixture();
        let s = stats(&graph, &swarm);
        // {0, 1, 2, 3} and {4}.
        assert_eq!(s.n_components, 2);
        assert_eq!(s.largest_component, 4);
        assert_eq!(s.isolated, 1);
    }

    #[test]
    fn cross_squad_share_counts_links_between_squads() {
        let (graph, swarm) = fixture();
        let s = stats(&graph, &swarm);
        // Only 2-3 crosses: 1 link out of 4.
        assert!((s.cross_squad_share - 0.25).abs() < 1e-12);
    }

    #[test]
    fn no_links_and_no_drones_give_zeros_not_nan() {
        let swarm = vec![drone(0, 0), drone(1, 0)];
        let s = stats(
            &EdgeList {
                n: 2,
                edges: vec![],
            },
            &swarm,
        );
        assert_eq!(s.mean_degree, 0.0);
        assert_eq!(s.mean_weight, 0.0);
        assert_eq!(s.cross_squad_share, 0.0);
        assert_eq!(s.n_components, 2);
        assert_eq!(s.isolated, 2);

        let empty = stats(
            &EdgeList {
                n: 0,
                edges: vec![],
            },
            &[],
        );
        assert_eq!(empty.n_components, 0);
        assert_eq!(empty.largest_component, 0);
        assert_eq!(empty.mean_degree, 0.0);
    }
}
