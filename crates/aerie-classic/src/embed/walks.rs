//! Random walks over a weighted graph: the "sentences" DeepWalk learns from.

use aerie_graph::EdgeList;
use rand::RngExt;
use rand_chacha::ChaCha8Rng;

/// `walks_per_node` rounds of one walk from every drone, each walk up to `walk_length`
/// drones long (its start included, so `walk_length - 1` steps).
///
/// Each step moves to a neighbour with probability proportional to the link weight:
/// row `i` of the transition matrix P = D⁻¹A from Mission 1. On the traffic graph, a pair
/// that exchanged 10 times more messages is 10 times more likely to be the next step.
/// A drone with no links ends its walk at once (a walk of length 1).
///
/// Weights must be positive. The walks come out round by round, drone 0 first in each
/// round, so walk `r * n + i` starts at drone `i`. Same `rng` state, same walks.
pub fn random_walks(
    graph: &EdgeList,
    walks_per_node: usize,
    walk_length: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<usize>> {
    let n = graph.n as usize;

    // Neighbour lists, built once: (neighbour, weight), each link added to both ends.
    let mut neighbours: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for edge in &graph.edges {
        let (a, b) = (edge.src as usize, edge.dst as usize);
        neighbours[a].push((b, edge.weight));
        neighbours[b].push((a, edge.weight));
    }

    let mut walks: Vec<Vec<usize>> = Vec::with_capacity(walks_per_node * n);
    for _round in 0..walks_per_node {
        for start in 0..n {
            let mut walk = vec![start];
            let mut current = start;
            for _step in 1..walk_length {
                let options = &neighbours[current];
                if options.is_empty() {
                    break; // isolated drone: the walk stops here
                }
                // Pick a neighbour with probability proportional to its weight.
                let total: f64 = options.iter().map(|&(_, w)| w).sum();
                let mut u = rng.random_range(0.0..total);
                let mut next = options[options.len() - 1].0; // fallback for rounding
                for &(j, w) in options {
                    u -= w;
                    if u < 0.0 {
                        next = j;
                        break;
                    }
                }
                walk.push(next);
                current = next;
            }
            walks.push(walk);
        }
    }
    walks
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_graph::Edge;
    use rand::SeedableRng;

    fn graph(n: u32, links: &[(u32, u32, f64)]) -> EdgeList {
        EdgeList {
            n,
            edges: links
                .iter()
                .map(|&(src, dst, weight)| Edge { src, dst, weight })
                .collect(),
        }
    }

    fn rng() -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(7)
    }

    #[test]
    fn counts_lengths_and_starts() {
        // A path 0 - 1 - 2 - 3: every walk has the full length, and walk r·n + i starts
        // at drone i.
        let g = graph(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        let walks = random_walks(&g, 3, 6, &mut rng());
        assert_eq!(walks.len(), 3 * 4);
        for (k, w) in walks.iter().enumerate() {
            assert_eq!(w.len(), 6);
            assert_eq!(w[0], k % 4);
        }
    }

    #[test]
    fn every_step_follows_a_link() {
        let links = [
            (0, 1, 2.0),
            (1, 2, 1.0),
            (0, 2, 5.0),
            (2, 3, 0.5),
            (3, 4, 1.0),
        ];
        let g = graph(5, &links);
        for w in random_walks(&g, 20, 10, &mut rng()) {
            for step in w.windows(2) {
                let (a, b) = (step[0].min(step[1]) as u32, step[0].max(step[1]) as u32);
                assert!(
                    links.iter().any(|&(s, d, _)| (s, d) == (a, b)),
                    "{} → {} is not a link",
                    step[0],
                    step[1]
                );
            }
        }
    }

    #[test]
    fn an_isolated_drone_walks_nowhere() {
        let g = graph(3, &[(0, 1, 1.0)]);
        let walks = random_walks(&g, 2, 5, &mut rng());
        assert_eq!(walks[2], [2]);
        assert_eq!(walks[5], [2]);
        assert_eq!(walks[0].len(), 5);
    }

    #[test]
    fn step_frequencies_match_the_transition_matrix() {
        // The triangle from the task: 0-1 weight 1, 1-2 weight 3, 0-2 weight 1. From
        // drone 1, P says 1/4 to drone 0 and 3/4 to drone 2. Over many steps, the share
        // of moves i → j must match P[i][j] from Mission 1.
        let g = graph(3, &[(0, 1, 1.0), (1, 2, 3.0), (0, 2, 1.0)]);
        let p = g.transition_matrix();
        let mut moves = [[0.0_f64; 3]; 3];
        for w in random_walks(&g, 2000, 20, &mut rng()) {
            for step in w.windows(2) {
                moves[step[0]][step[1]] += 1.0;
            }
        }
        for i in 0..3 {
            let out: f64 = moves[i].iter().sum();
            for j in 0..3 {
                let share = moves[i][j] / out;
                assert!(
                    (share - p[(i, j)]).abs() < 0.01,
                    "{i} → {j}: walks {share:.3}, P {:.3}",
                    p[(i, j)]
                );
            }
        }
        assert!((p[(1, 0)] - 0.25).abs() < 1e-12 && (p[(1, 2)] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn same_rng_same_walks() {
        let g = graph(4, &[(0, 1, 1.0), (1, 2, 2.0), (2, 3, 1.0), (0, 3, 4.0)]);
        let a = random_walks(&g, 5, 8, &mut ChaCha8Rng::seed_from_u64(1));
        let b = random_walks(&g, 5, 8, &mut ChaCha8Rng::seed_from_u64(1));
        let c = random_walks(&g, 5, 8, &mut ChaCha8Rng::seed_from_u64(2));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
