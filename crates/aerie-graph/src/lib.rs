//! Graph storage shared by every AERIE crate.
//!
//! Graphs are stored as weighted edge lists, the format every mission reads.
//! This crate knows nothing about drones: nodes are plain indices.

use faer::Mat;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

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

    /// Adjacency matrix A (Mission 1): `A[i][j]` is the weight of the link i-j, 0 when
    /// there is none. n x n and dense (a faer `Mat`).
    ///
    /// Symmetric, because each undirected link is written at both (i, j) and (j, i).
    /// Zero diagonal, because there are no self-loops. Row i sums to `degrees()[i]`.
    ///
    /// Dense costs n² memory: fine at 100 drones, too much at 100k (Mission 9 needs CSR).
    pub fn adjacency_matrix(&self) -> Mat<f64> {
        let n = self.n as usize;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        for edge in &self.edges {
            a[(edge.src as usize, edge.dst as usize)] = edge.weight;
            a[(edge.dst as usize, edge.src as usize)] = edge.weight;
        }

        a
    }

    /// Degree matrix D (Mission 1): diagonal, `D[i][i]` = weighted degree of node i (the
    /// sum of row i of A), zeros everywhere else. A degree belongs to one node, so there
    /// is nothing off the diagonal.
    ///
    /// Built from A with a double loop (n² steps); `degrees()` gives the same diagonal
    /// from the edges alone, and the tests check that they agree.
    pub fn degree_matrix(&self) -> Mat<f64> {
        let n = self.n as usize;
        let mut d: Mat<f64> = Mat::zeros(n, n);
        let a: Mat<f64> = self.adjacency_matrix();
        for i in 0..n {
            for j in 0..n {
                d[(i, i)] += a[(i, j)];
            }
        }

        d
    }
    /// Adjacency list: entry `i` holds the indices of the nodes linked to node `i`.
    ///
    /// Returns a vector of length `n`. Isolated nodes get an empty list.
    /// Each link appears in both endpoints' lists, since the graph is undirected.
    /// The order inside a list follows the order of `edges`, so do not rely on it.
    /// Weights are ignored: this answers "who", not "how strongly".
    pub fn neighbours(&self) -> Vec<Vec<usize>> {
        let mut neighbours = vec![Vec::new(); self.n as usize];

        // Push each endpoint into the other's list.
        for edge in &self.edges {
            neighbours[edge.src as usize].push(edge.dst as usize);
            neighbours[edge.dst as usize].push(edge.src as usize);
        }

        neighbours
    }

    /// Connected components by breadth-first search.
    ///
    /// Returns a vector of length `n`: entry `i` is the component number of node `i`.
    /// Nodes that can reach each other share a number. Numbers start at 0 and follow
    /// the order of discovery: the component of node 0 is 0, the next unvisited node
    /// starts component 1, and so on. The number of components is the maximum + 1.
    ///
    /// Runs in O(N + M): each node enters the queue once, each link is seen twice.
    pub fn components(&self) -> Vec<usize> {
        // `None` marks unvisited nodes during BFS; every node is `Some` when it finishes.
        let mut components: Vec<Option<usize>> = vec![None; self.n as usize];
        // Built once: looking up neighbours inside the loop must be instant.
        let neighbours = self.neighbours();
        let mut component_count: usize = 0;
        let mut queue = VecDeque::new();

        // Every node not reached yet starts a new component.
        for s in 0..self.n as usize {
            if components[s].is_none() {
                queue.push_back(s);
                components[s] = Some(component_count);
                // Mark nodes when they are pushed, not when popped,
                // so no node is queued twice.
                while let Some(u) = queue.pop_front() {
                    for v in &neighbours[u] {
                        if components[*v].is_none() {
                            components[*v] = Some(component_count);
                            queue.push_back(*v);
                        }
                    }
                }
                component_count += 1;
            }
        }
        let mut result = Vec::new();
        for slot in components {
            // BFS visited every node, so every slot is Some
            result.push(slot.unwrap());
        }
        result
    }
}

// Unit tests: compiled only by `cargo test`.
#[cfg(test)]
mod tests {
    use super::*;

    /// Test fixture: triangle 0-1-2 plus isolated node 3, all weights 1.0.
    /// Edge order (0,1), (1,2), (0,2) fixes the neighbour order the tests expect.
    fn triangle_plus_isolated() -> EdgeList {
        EdgeList {
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
        }
    }

    #[test]
    fn degrees_triangle_plus_isolated() {
        // Each triangle node has two links of weight 1.0; node 3 has none.
        let g: EdgeList = triangle_plus_isolated();
        assert_eq!(g.degrees(), vec![2.0, 2.0, 2.0, 0.0]);
    }

    #[test]
    fn neighbours_triangle_plus_isolated() {
        // Hand trace: (0,1) -> [[1],[0],[],[]]; (1,2) -> [[1],[0,2],[1],[]];
        // (0,2) -> [[1,2],[0,2],[1,0],[]]. Node 3 stays empty.
        let g: EdgeList = triangle_plus_isolated();
        assert_eq!(
            g.neighbours(),
            vec![vec![1, 2], vec![0, 2], vec![1, 0], vec![]]
        );
    }

    #[test]
    fn components_triangle_plus_isolated() {
        // BFS from node 0 reaches 1 and 2 (component 0); node 3 starts component 1.
        let g: EdgeList = triangle_plus_isolated();
        assert_eq!(g.components(), vec![0, 0, 0, 1]);
    }

    #[test]
    fn components_interleaved() {
        // Components whose members are not consecutive indices: {0, 3} and {1, 2}.
        // BFS from node 0 reaches 3 (component 0); node 1 starts component 1 and reaches 2.
        // Catches code that numbers nodes by position instead of by reachability.
        let g = EdgeList {
            n: 4,
            edges: vec![
                Edge {
                    src: 0,
                    dst: 3,
                    weight: 1.0,
                },
                Edge {
                    src: 1,
                    dst: 2,
                    weight: 1.0,
                },
            ],
        };
        assert_eq!(g.components(), vec![0, 1, 1, 0]);
    }

    /// Checks that `m` equals the grid `expected`, cell by cell, within 1e-12.
    fn assert_mat(m: &Mat<f64>, expected: &[&[f64]]) {
        assert_eq!(m.nrows(), expected.len(), "rows");
        for (i, row) in expected.iter().enumerate() {
            assert_eq!(m.ncols(), row.len(), "columns");
            for (j, &want) in row.iter().enumerate() {
                let got = m[(i, j)];
                assert!(
                    (got - want).abs() < 1e-12,
                    "({i}, {j}): got {got}, want {want}"
                );
            }
        }
    }

    /// Path 0 - 1 - 2 with two different weights, so a mix-up of rows shows.
    fn weighted_path() -> EdgeList {
        EdgeList {
            n: 3,
            edges: vec![
                Edge {
                    src: 0,
                    dst: 1,
                    weight: 0.5,
                },
                Edge {
                    src: 1,
                    dst: 2,
                    weight: 2.0,
                },
            ],
        }
    }

    #[test]
    fn adjacency_triangle_plus_isolated() {
        // Each triangle node links to the other two with weight 1; node 3 has no links,
        // so its row and column are zero. Nothing on the diagonal.
        let a = triangle_plus_isolated().adjacency_matrix();
        assert_mat(
            &a,
            &[
                &[0.0, 1.0, 1.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0],
                &[1.0, 1.0, 0.0, 0.0],
                &[0.0, 0.0, 0.0, 0.0],
            ],
        );
    }

    #[test]
    fn adjacency_keeps_the_weights_and_is_symmetric() {
        let a = weighted_path().adjacency_matrix();
        assert_mat(&a, &[&[0.0, 0.5, 0.0], &[0.5, 0.0, 2.0], &[0.0, 2.0, 0.0]]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(a[(i, j)], a[(j, i)]);
            }
        }
    }

    #[test]
    fn degree_triangle_plus_isolated() {
        // Diagonal = degrees [2, 2, 2, 0], zeros elsewhere.
        let d = triangle_plus_isolated().degree_matrix();
        assert_mat(
            &d,
            &[
                &[2.0, 0.0, 0.0, 0.0],
                &[0.0, 2.0, 0.0, 0.0],
                &[0.0, 0.0, 2.0, 0.0],
                &[0.0, 0.0, 0.0, 0.0],
            ],
        );
    }

    #[test]
    fn degree_matrix_agrees_with_degrees_and_row_sums_of_a() {
        // Two separate pieces of code must agree: D from A (double loop) and
        // degrees() from the edge list. Weighted path: [0.5, 2.5, 2.0].
        for g in [weighted_path(), triangle_plus_isolated()] {
            let d = g.degree_matrix();
            let a = g.adjacency_matrix();
            let deg = g.degrees();
            let n = g.n as usize;
            for i in 0..n {
                assert!((d[(i, i)] - deg[i]).abs() < 1e-12);
                let row_sum: f64 = (0..n).map(|j| a[(i, j)]).sum();
                assert!((d[(i, i)] - row_sum).abs() < 1e-12);
                for j in 0..n {
                    if i != j {
                        assert_eq!(d[(i, j)], 0.0, "off-diagonal ({i}, {j})");
                    }
                }
            }
        }
        assert_eq!(weighted_path().degrees(), vec![0.5, 2.5, 2.0]);
    }

    #[test]
    fn empty_graph_gives_empty_matrices() {
        let g = EdgeList {
            n: 0,
            edges: vec![],
        };
        assert_eq!(g.adjacency_matrix().nrows(), 0);
        assert_eq!(g.degree_matrix().ncols(), 0);
    }
}
