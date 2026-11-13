//! Graph storage shared by every AERIE crate.
//!
//! Graphs are stored as weighted edge lists, the format every mission reads.
//! This crate knows nothing about drones: nodes are plain indices.

use serde::{Deserialize, Serialize};

/// One weighted link between two nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Index of the first endpoint. For undirected graphs, `src < dst`.
    pub src: u32,
    /// Index of the second endpoint.
    pub dst: u32,
    /// Link weight, e.g. exp(-d^2 / (2 sigma^2)). Absent links are simply not stored.
    pub weight: f64,
}

/// A weighted graph stored as a list of edges.
///
/// Conventions:
/// - Undirected: each link is stored once, with `src < dst`.
/// - Nodes are indices `0..n` in this snapshot, not drone IDs.
///   The mapping from index to `DroneId` lives in `swarmsim`.
/// - `n` is stored explicitly so isolated nodes (degree 0) still exist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeList {
    /// Number of nodes.
    pub n: u32,
    /// The links. No self-loops, no duplicates.
    pub edges: Vec<Edge>,
}

impl EdgeList {
    /// Weighted degree of every node: the sum of the weights of its links.
    ///
    /// Returns a vector of length `n`. Isolated nodes get 0.0.
    /// This is the diagonal of the degree matrix D in L = D - A.
    pub fn degrees(&self) -> Vec<f64> {
        let mut degrees = vec![0.0; self.n as usize];
        // Borrow the edges (&) instead of moving them out of self.
        // Each undirected link adds its weight to both endpoints.
        for edge in &self.edges {
            degrees[edge.src as usize] += edge.weight;
            degrees[edge.dst as usize] += edge.weight;
        }
        degrees
    }
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degrees_triangle_plus_isolated() {
        // build the triangle 0-1-2 plus isolated node 3, all weights 1.0
        let g = EdgeList {
            n: 4,
            edges: vec![
                Edge {
                    src: 0,
                    dst: 1,
                    weight: 1.0,
                },
                Edge {
                    src: 1,
                    dst: 2,
                    weight: 1.0,
                },
                Edge {
                    src: 0,
                    dst: 2,
                    weight: 1.0,
                },
            ],
        };

        assert_eq!(g.degrees(), vec![2.0, 2.0, 2.0, 0.0]);
    }
}
