//! Node2Vec (Grover & Leskovec, 2016): DeepWalk with biased walks (return parameter p,
//! in-out parameter q), then the same skip-gram.
//!
//! A DeepWalk step only looks at where the walk is (v). A Node2Vec step also looks at
//! where it came from (t), and multiplies each neighbour x's link weight by a bias:
//! - 1/p if x is t (going back);
//! - 1 if x is also a neighbour of t (staying near t);
//! - 1/q otherwise (moving away from t).
//!
//! With p = q = 1 every bias is 1: the walks, and so Z, are DeepWalk's exactly.

use aerie_graph::EdgeList;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

use super::skipgram::{SkipGramConfig, train_skipgram};

/// Settings for Node2Vec: the bias, the walks, then skip-gram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node2VecConfig {
    /// Return parameter: a step back to t is weighted 1/p. Large p: rarely go back.
    pub p: f64,
    /// In-out parameter: a step away from t is weighted 1/q. Small q: move outward.
    pub q: f64,
    /// Walks started from every drone.
    pub walks_per_node: usize,
    /// Drones per walk, the start included.
    pub walk_length: usize,
    /// Everything about the training: dim, window, negatives, epochs, lr.
    pub skipgram: SkipGramConfig,
}

impl Default for Node2VecConfig {
    /// p = q = 1 (DeepWalk), then DeepWalk's defaults: 10 walks of 20 drones per drone,
    /// skip-gram's own defaults.
    fn default() -> Self {
        Node2VecConfig {
            p: 1.0,
            q: 1.0,
            walks_per_node: 10,
            walk_length: 20,
            skipgram: SkipGramConfig::default(),
        }
    }
}

/// `walks_per_node` rounds of one walk from every drone, each up to `walk_length` drones,
/// biased by p and q (module doc). The first step of a walk has no t: it is a plain
/// weighted step. A drone with no links ends its walk at once.
///
/// Same order, same neighbour order and same random-number use as `random_walks` (one
/// number per step): with p = q = 1 and the same `rng` state, the walks are identical.
/// Weights and p, q must be positive.
pub fn biased_walks(
    graph: &EdgeList,
    walks_per_node: usize,
    walk_length: usize,
    p: f64,
    q: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<usize>> {
    let n = graph.n as usize;

    // Neighbour lists, built once: (neighbour, weight), each link added to both ends.
    let mut neighbours: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut linked: HashSet<(usize, usize)> = HashSet::new();

    for edge in &graph.edges {
        let (a, b) = (edge.src as usize, edge.dst as usize);
        neighbours[a].push((b, edge.weight));
        neighbours[b].push((a, edge.weight));
        linked.insert((a, b));
        linked.insert((b, a));
    }

    let mut walks: Vec<Vec<usize>> = Vec::with_capacity(walks_per_node * n);
    for _round in 0..walks_per_node {
        for start in 0..n {
            let mut walk = vec![start];
            let mut current = start;
            let mut previous: Option<usize> = None; // no previous drone before the first step

            for _step in 1..walk_length {
                let options = &neighbours[current];

                if options.is_empty() {
                    break; // isolated drone: the walk stops here
                }

                // Each neighbour's link weight times its bias.
                let weights: Vec<f64> = options
                    .iter()
                    .map(|&(x, w)| {
                        let bias = match previous {
                            None => 1.0,                                // first step: plain DeepWalk
                            Some(t) if x == t => 1.0 / p,               // back where we came from
                            Some(t) if linked.contains(&(t, x)) => 1.0, // also next to t
                            Some(_) => 1.0 / q,                         // further away from t
                        };
                        w * bias
                    })
                    .collect();

                // Same pick as random_walks: one random number, subtract until below 0.
                let total: f64 = weights.iter().sum();
                let mut u = rng.random_range(0.0..total);
                let mut next = options[options.len() - 1].0; // fallback for rounding
                for (&(x, _), &w) in options.iter().zip(&weights) {
                    u -= w;
                    if u < 0.0 {
                        next = x;
                        break;
                    }
                }
                walk.push(next);
                previous = Some(current);
                current = next;
            }
            walks.push(walk);
        }
    }
    walks
}

/// One vector per drone, learned from biased walks on `graph`. Returns Z: `graph.n`
/// vectors of `config.skipgram.dim` numbers, in drone order.
///
/// Same random streams as `deepwalk` (1 for the walks, 2 for the training), so with
/// p = q = 1 it returns `deepwalk`'s Z exactly (the plan's unit test).
pub fn node2vec(graph: &EdgeList, config: &Node2VecConfig, seed: u64) -> Vec<Vec<f64>> {
    let mut walk_rng = ChaCha8Rng::seed_from_u64(seed);
    walk_rng.set_stream(1);
    let mut train_rng = ChaCha8Rng::seed_from_u64(seed);
    train_rng.set_stream(2);

    let walks = biased_walks(
        graph,
        config.walks_per_node,
        config.walk_length,
        config.p,
        config.q,
        &mut walk_rng,
    );
    train_skipgram(&walks, graph.n as usize, &config.skipgram, &mut train_rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::deepwalk::{DeepWalkConfig, deepwalk};
    use crate::embed::walks::random_walks;
    use aerie_graph::Edge;

    fn graph(n: u32, links: &[(u32, u32, f64)]) -> EdgeList {
        EdgeList {
            n,
            edges: links
                .iter()
                .map(|&(src, dst, weight)| Edge { src, dst, weight })
                .collect(),
        }
    }

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    /// Two 6-drone cliques (0–5 and 6–11) joined by one bridge 5–6.
    fn two_groups() -> EdgeList {
        let mut links = Vec::new();
        for g in [0_u32, 6] {
            for a in g..g + 6 {
                for b in a + 1..g + 6 {
                    links.push((a, b, 1.0));
                }
            }
        }
        links.push((5, 6, 1.0));
        graph(12, &links)
    }

    fn ring(n: u32) -> EdgeList {
        let links: Vec<(u32, u32, f64)> = (0..n)
            .map(|i| (i.min((i + 1) % n), i.max((i + 1) % n), 1.0))
            .collect();
        graph(n, &links)
    }

    #[test]
    fn default_is_deepwalk() {
        let c = Node2VecConfig::default();
        let d = DeepWalkConfig::default();
        assert_eq!((c.p, c.q), (1.0, 1.0));
        assert_eq!(
            (c.walks_per_node, c.walk_length),
            (d.walks_per_node, d.walk_length)
        );
        assert_eq!(c.skipgram, d.skipgram);
    }

    #[test]
    fn with_p_and_q_at_1_the_walks_are_random_walks() {
        // Weighted, with a triangle, a leaf and an isolated drone (5).
        let g = graph(
            6,
            &[
                (0, 1, 3.0),
                (0, 2, 1.0),
                (1, 2, 0.5),
                (2, 3, 2.0),
                (3, 4, 1.0),
            ],
        );
        assert_eq!(
            biased_walks(&g, 5, 12, 1.0, 1.0, &mut rng(1)),
            random_walks(&g, 5, 12, &mut rng(1))
        );
    }

    #[test]
    fn with_p_and_q_at_1_node2vec_is_deepwalk() {
        // The plan's unit test: same Z, number for number.
        let g = two_groups();
        let skipgram = SkipGramConfig {
            dim: 8,
            window: 3,
            ..Default::default()
        };
        let n2v = Node2VecConfig {
            walks_per_node: 4,
            walk_length: 10,
            skipgram,
            ..Default::default()
        };
        let dw = DeepWalkConfig {
            walks_per_node: 4,
            walk_length: 10,
            skipgram,
        };
        assert_eq!(node2vec(&g, &n2v, 7), deepwalk(&g, &dw, 7));
    }

    #[test]
    fn the_second_step_follows_the_biases() {
        // t = 0, v = 1; from v: back to 0, to 2 (also next to 0), to 3 (not next to 0).
        // All weights 1, p = 2, q = 0.5: biases 1/2, 1, 2 -> chances 1/7, 2/7, 4/7.
        let g = graph(4, &[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 1.0)]);
        let walks = biased_walks(&g, 30_000, 3, 2.0, 0.5, &mut rng(2));
        let from_0_via_1: Vec<&Vec<usize>> =
            walks.iter().filter(|w| w[0] == 0 && w[1] == 1).collect();
        let share = |x: usize| {
            from_0_via_1.iter().filter(|w| w[2] == x).count() as f64 / from_0_via_1.len() as f64
        };
        assert!(from_0_via_1.len() > 10_000);
        for (x, expected) in [(0, 1.0 / 7.0), (2, 2.0 / 7.0), (3, 4.0 / 7.0)] {
            assert!(
                (share(x) - expected).abs() < 0.015,
                "to {x}: {} vs {expected}",
                share(x)
            );
        }
    }

    #[test]
    fn a_huge_p_never_goes_back() {
        let walks = biased_walks(&ring(10), 20, 15, 1e12, 1.0, &mut rng(3));
        for w in &walks {
            assert!(w.windows(3).all(|s| s[2] != s[0]), "{w:?} stepped back");
        }
    }

    #[test]
    fn a_small_q_explores_more_than_a_large_q() {
        // On a ring, the only choices are back (1/p) or on (1/q): a small q keeps going,
        // a large q bounces between two drones.
        let distinct = |q: f64| {
            let walks = biased_walks(&ring(30), 10, 20, 1.0, q, &mut rng(4));
            walks
                .iter()
                .map(|w| {
                    let mut s = w.clone();
                    s.sort_unstable();
                    s.dedup();
                    s.len() as f64
                })
                .sum::<f64>()
                / walks.len() as f64
        };
        let (outward, local) = (distinct(0.25), distinct(4.0));
        assert!(outward > 2.0 * local, "{outward} vs {local}");
    }

    #[test]
    fn shapes_reproducible_and_isolated_drones() {
        let g = graph(5, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]); // 4 has no link
        let walks = biased_walks(&g, 3, 8, 0.5, 2.0, &mut rng(5));
        assert_eq!(walks.len(), 15);
        assert_eq!(walks[4], vec![4]);
        for (k, w) in walks.iter().enumerate() {
            assert_eq!(w[0], k % 5, "walk r * n + i starts at drone i");
            assert!(w.len() <= 8);
        }
        assert_eq!(walks, biased_walks(&g, 3, 8, 0.5, 2.0, &mut rng(5)));
        let config = Node2VecConfig {
            p: 0.5,
            q: 2.0,
            walks_per_node: 3,
            walk_length: 8,
            skipgram: SkipGramConfig {
                dim: 4,
                ..Default::default()
            },
        };
        let z = node2vec(&g, &config, 6);
        assert_eq!(z.len(), 5);
        assert!(
            z.iter()
                .all(|r| r.len() == 4 && r.iter().all(|x| x.is_finite()))
        );
        assert_eq!(z, node2vec(&g, &config, 6));
    }
}
