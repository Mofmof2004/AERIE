//! DeepWalk (Perozzi et al., 2014): random walks on the graph, fed to skip-gram.
//!
//! The walks turn the graph into "sentences" of drone ids; skip-gram then learns a vector
//! per drone so that drones met close together on the walks get vectors with a high dot
//! product. Encoder: the lookup table Z. Decoder: the dot product z_u · z_v.
//! Node2Vec is the same thing with biased walks.

use aerie_graph::EdgeList;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::skipgram::{SkipGramConfig, train_skipgram};
use super::walks::random_walks;

/// Settings for DeepWalk: the walks, then skip-gram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepWalkConfig {
    /// Walks started from every drone.
    pub walks_per_node: usize,
    /// Drones per walk, the start included.
    pub walk_length: usize,
    /// Everything about the training: dim, window, negatives, epochs, lr.
    pub skipgram: SkipGramConfig,
}

impl Default for DeepWalkConfig {
    /// The plan's starting point for Mission 2: 10 walks of 20 drones from every drone,
    /// then skip-gram's own defaults (dim 32, window 5, 5 negatives, lr 0.025).
    fn default() -> Self {
        DeepWalkConfig {
            walks_per_node: 10,
            walk_length: 20,
            skipgram: SkipGramConfig::default(),
        }
    }
}

/// One vector per drone, learned from random walks on `graph`. Returns Z: `graph.n`
/// vectors of `config.skipgram.dim` numbers, in drone order.
///
/// The walks and the training each get their own random stream from `seed` (streams 1
/// and 2): changing the walks never changes Z's random start, so two settings differ only
/// by what they are meant to change. Same graph, config and seed, same Z.
///
/// A drone with no links walks in place (a walk of itself only): it never makes a pair
/// and keeps its random start vector.
pub fn deepwalk(graph: &EdgeList, config: &DeepWalkConfig, seed: u64) -> Vec<Vec<f64>> {
    let mut walk_rng = ChaCha8Rng::seed_from_u64(seed);
    walk_rng.set_stream(1);
    let mut train_rng = ChaCha8Rng::seed_from_u64(seed);
    train_rng.set_stream(2);

    let walks = random_walks(
        graph,
        config.walks_per_node,
        config.walk_length,
        &mut walk_rng,
    );
    train_skipgram(&walks, graph.n as usize, &config.skipgram, &mut train_rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::skipgram::dot;
    use aerie_graph::Edge;

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
    }

    /// Two 6-drone cliques (0–5 and 6–11) joined by one bridge 5–6.
    fn two_groups() -> EdgeList {
        let mut edges = Vec::new();
        for group in [0_u32, 6] {
            for a in group..group + 6 {
                for b in a + 1..group + 6 {
                    edges.push(Edge {
                        src: a,
                        dst: b,
                        weight: 1.0,
                    });
                }
            }
        }
        edges.push(Edge {
            src: 5,
            dst: 6,
            weight: 1.0,
        });
        EdgeList { n: 12, edges }
    }

    /// Small and fast: 5 walks of 10 per drone, window 3.
    fn small(dim: usize, epochs: usize) -> DeepWalkConfig {
        DeepWalkConfig {
            walks_per_node: 5,
            walk_length: 10,
            skipgram: SkipGramConfig {
                dim,
                window: 3,
                epochs,
                ..Default::default()
            },
        }
    }

    #[test]
    fn default_is_the_plans_starting_point() {
        let c = DeepWalkConfig::default();
        assert_eq!((c.walks_per_node, c.walk_length), (10, 20));
        assert_eq!(
            (c.skipgram.dim, c.skipgram.window, c.skipgram.negatives),
            (32, 5, 5)
        );
        assert_eq!(c.skipgram.lr, 0.025);
    }

    #[test]
    fn one_vector_of_dim_numbers_per_drone() {
        let z = deepwalk(&two_groups(), &small(8, 1), 0);
        assert_eq!(z.len(), 12);
        assert!(z.iter().all(|row| row.len() == 8));
        assert!(z.iter().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn same_seed_same_z_other_seed_other_z() {
        let g = two_groups();
        let config = small(8, 1);
        assert_eq!(deepwalk(&g, &config, 3), deepwalk(&g, &config, 3));
        assert_ne!(deepwalk(&g, &config, 3), deepwalk(&g, &config, 4));
    }

    #[test]
    fn is_exactly_walks_then_skipgram_on_streams_1_and_2() {
        let g = two_groups();
        let config = small(8, 2);
        let mut walk_rng = ChaCha8Rng::seed_from_u64(7);
        walk_rng.set_stream(1);
        let mut train_rng = ChaCha8Rng::seed_from_u64(7);
        train_rng.set_stream(2);
        let walks = random_walks(&g, 5, 10, &mut walk_rng);
        let by_hand = train_skipgram(&walks, 12, &config.skipgram, &mut train_rng);
        assert_eq!(deepwalk(&g, &config, 7), by_hand);
    }

    #[test]
    fn more_walks_do_not_change_the_random_start() {
        // With 0 epochs nothing is trained, so Z is its random start. Separate streams:
        // the walks change, the start doesn't.
        let g = two_groups();
        let mut few = small(8, 0);
        few.walks_per_node = 1;
        let mut many = small(8, 0);
        many.walks_per_node = 20;
        assert_eq!(deepwalk(&g, &few, 9), deepwalk(&g, &many, 9));
    }

    #[test]
    fn each_drones_nearest_vector_is_a_group_mate() {
        let g = two_groups();
        let config = DeepWalkConfig {
            walks_per_node: 10,
            walk_length: 10,
            skipgram: SkipGramConfig {
                dim: 16,
                window: 2,
                epochs: 3,
                ..Default::default()
            },
        };
        let z = deepwalk(&g, &config, 11);
        let mut right = 0;
        for a in 0..12 {
            let best = (0..12)
                .filter(|&b| b != a)
                .max_by(|&x, &y| cosine(&z[a], &z[x]).total_cmp(&cosine(&z[a], &z[y])))
                .unwrap();
            if (a < 6) == (best < 6) {
                right += 1;
            }
        }
        assert!(
            right >= 11,
            "{right}/12 drones have a group-mate as nearest neighbour"
        );
    }

    #[test]
    fn a_drone_with_no_links_keeps_its_start_vector() {
        // Drone 12 has no links: its walks are just [12], so it never makes a pair.
        let mut g = two_groups();
        g.n = 13;
        let trained = deepwalk(&g, &small(8, 2), 5);
        let start = deepwalk(&g, &small(8, 0), 5);
        assert_eq!(trained[12], start[12]);
        assert_ne!(trained[0], start[0]);
    }
}
