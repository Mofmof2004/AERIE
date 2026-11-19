//! Skip-gram with negative sampling: the learning half of DeepWalk.
//!
//! Every drone gets two vectors of `d` numbers: `z` (its embedding, the result) and `c` (its
//! context vector, used only while training). For a drone `u` and a drone `v` seen near it
//! in a walk, training raises `z_u · z_v`; for a few random drones `n`, it lowers `z_u · c_n`.
//! Drones that keep meeting in walks end up with similar `z`.

use rand::RngExt;
use rand_chacha::ChaCha8Rng;

/// The logistic function σ(x) = 1 / (1 + e⁻ˣ): squashes any number into (0, 1).
/// σ(0) = 0.5, and σ(−x) = 1 − σ(x).
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The dot product a · b = a₀b₀ + a₁b₁ + …: one number. Both slices must have the same length.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut result: f64 = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}

/// The loss of one training example and its gradients. Changes nothing.
///
/// The example is a drone `u` (embedding `z_u`), a drone `v` that appeared near it in a walk
/// (its **context** vector, named `z_v` here but it comes from C, not Z), and random drones `n₁ … n_k` (contexts `c_neg`):
///
/// ```text
/// L = −log σ(z_u · z_v) − Σₖ log σ(−z_u · c_nₖ)
///
/// with g_v = σ(z_u · z_v) − 1   and   g_k = σ(z_u · c_nₖ):
/// ∂L/∂z_u  = g_v · z_v + Σₖ g_k · c_nₖ
/// ∂L/∂z_v  = g_v · z_u
/// ∂L/∂c_nₖ = g_k · z_u
/// ```
///
/// Returns `(loss, ∂L/∂z_u, ∂L/∂z_v, [∂L/∂c_n₁, …])`, the last in the order of `c_neg`.
/// Every gradient is a number times the other vector of its dot product, all computed from
/// the vectors as given: apply the SGD step only after, or `z_u` sees half-updated contexts.
pub fn pair_loss_and_grads(
    z_u: &[f64],
    z_v: &[f64],
    c_neg: &[&[f64]],
) -> (f64, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let d = z_u.len();

    // Positive pair: pull u and v together.
    let x = dot(z_u, z_v);
    let g_v = sigmoid(x) - 1.0;
    let mut loss = -sigmoid(x).ln();

    let grad_zv: Vec<f64> = z_u.iter().map(|&z| g_v * z).collect(); // g_v · z_u
    let mut grad_z: Vec<f64> = z_v.iter().map(|&c| g_v * c).collect(); // g_v · z_v, then add the negatives

    // Negatives: push u away from each one.
    let mut grad_neg: Vec<Vec<f64>> = Vec::new();
    for &c_n in c_neg {
        let x = dot(z_u, c_n);
        let g_k = sigmoid(x);
        loss -= sigmoid(-x).ln();

        for i in 0..d {
            grad_z[i] += g_k * c_n[i]; // + g_k · c_nₖ
        }
        grad_neg.push(z_u.iter().map(|&z| g_k * z).collect()); // g_k · z_u
    }
    (loss, grad_z, grad_zv, grad_neg)
}

/// Settings for skip-gram training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkipGramConfig {
    pub dim: usize,       // numbers per vector
    pub window: usize,    // how far apart in a walk two drones count as a pair
    pub negatives: usize, // random drones per positive pair
    pub epochs: usize,    // passes over all the walks
    pub lr: f64,          // SGD step size
}

impl Default for SkipGramConfig {
    fn default() -> Self {
        SkipGramConfig {
            dim: 32,
            window: 5,
            negatives: 5,
            epochs: 1,
            lr: 0.025,
        }
    }
}

/// Learns one vector per drone from the walks. Returns Z: n vectors of `dim` numbers.
///
/// Z starts random in ±0.5/dim, C at zero (as in Word2Vec). For every position `t` of every
/// walk, each drone `v` at most `window` steps away (either side) makes a positive pair
/// (u = walk[t], v), with `negatives` random drones drawn in proportion to
/// (appearances in all the walks)^0.75, never `u` or `v` themselves (Word2Vec only skips
/// `v`; skipping `u` too matters on small graphs, where u is drawn often). One plain SGD
/// step per pair with a fixed `lr`; `epochs` passes over all the walks.
///
/// The walks must visit at least 3 different drones, or the draw for negatives never ends:
/// on a graph of 2 drones, every draw is u or v.
///
/// `n` is the number of drones: every id in `walks` must be below it. A drone that never
/// appears keeps its random start vector. Same walks and `rng` state, same Z.
pub fn train_skipgram(
    walks: &[Vec<usize>],
    n: usize,
    config: &SkipGramConfig,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f64>> {
    let dim = config.dim;
    let half = 0.5 / config.dim as f64; // dim is usize, so convert it to f64 first

    let mut z: Vec<Vec<f64>> = Vec::new();

    for _ in 0..n {
        let mut row = Vec::new();
        for _ in 0..config.dim {
            row.push(rng.random_range(-half..half));
        }
        z.push(row);
    }

    let mut c: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];

    // 2. Negative table: running total of (appearances in the walks)^0.75.
    let mut counts = vec![0.0_f64; n];
    for walk in walks {
        for &d in walk {
            counts[d] += 1.0;
        }
    }

    let mut cum: Vec<f64> = Vec::new();
    let mut total = 0.0;
    for &count in &counts {
        total += count.powf(0.75);
        cum.push(total);
    }

    for _ in 0..config.epochs {
        for walk in walks {
            for t in 0..walk.len() {
                let u = walk[t];
                let lo = t.saturating_sub(config.window); // t − window, but never below 0
                let hi = (t + config.window).min(walk.len() - 1); // never past the walk's end

                for window_pos in lo..=hi {
                    if window_pos != t {
                        let v = walk[window_pos];
                        let mut neg: Vec<usize> = Vec::new();
                        while neg.len() < config.negatives {
                            let r = rng.random_range(0.0..total);
                            let k = cum.partition_point(|&x| x <= r);
                            if k != v && k != u {
                                neg.push(k);
                            }
                        }
                        let negs: Vec<&[f64]> = neg.iter().map(|&k| c[k].as_slice()).collect();
                        let (_loss, grad_z, grad_zv, grad_neg) =
                            pair_loss_and_grads(&z[u], &c[v], &negs);

                        for i in 0..dim {
                            z[u][i] -= config.lr * grad_z[i];
                            c[v][i] -= config.lr * grad_zv[i];
                        }

                        for j in 0..neg.len() {
                            let k = neg[j];
                            let g = &grad_neg[j];
                            for i in 0..dim {
                                c[k][i] -= config.lr * g[i];
                            }
                        }
                    }
                }
            }
        }
    }

    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::walks::random_walks;
    use aerie_graph::{Edge, EdgeList};
    use rand::SeedableRng;

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

    #[test]
    fn shapes_and_reproducible() {
        let g = two_groups();
        let walks = random_walks(&g, 2, 10, &mut ChaCha8Rng::seed_from_u64(1));
        let config = SkipGramConfig {
            dim: 8,
            ..Default::default()
        };
        let a = train_skipgram(&walks, 12, &config, &mut ChaCha8Rng::seed_from_u64(2));
        let b = train_skipgram(&walks, 12, &config, &mut ChaCha8Rng::seed_from_u64(2));
        assert_eq!(a.len(), 12);
        assert!(a.iter().all(|row| row.len() == 8));
        assert!(a.iter().flatten().all(|x| x.is_finite()));
        assert_eq!(a, b);
    }

    #[test]
    fn drones_in_the_same_group_get_similar_vectors() {
        // The core claim of DeepWalk: drones that meet in walks end up close. Mean cosine
        // inside a group must beat mean cosine across the bridge, by a clear margin.
        // Margin: over 30 seeds at dim 16 the gap is 0.35–0.48 when u is never a negative
        // (0.58–0.77 when it can be, as in Word2Vec), so 0.25 leaves room for the seed.
        let g = two_groups();
        let walks = random_walks(&g, 20, 20, &mut ChaCha8Rng::seed_from_u64(3));
        let config = SkipGramConfig {
            dim: 16,
            window: 3,
            epochs: 3,
            ..Default::default()
        };
        let z = train_skipgram(&walks, 12, &config, &mut ChaCha8Rng::seed_from_u64(4));

        let (mut inside, mut n_inside, mut across, mut n_across) = (0.0, 0.0, 0.0, 0.0);
        for a in 0..12 {
            for b in a + 1..12 {
                if (a < 6) == (b < 6) {
                    inside += cosine(&z[a], &z[b]);
                    n_inside += 1.0;
                } else {
                    across += cosine(&z[a], &z[b]);
                    n_across += 1.0;
                }
            }
        }
        let (inside, across) = (inside / n_inside, across / n_across);
        assert!(
            inside > across + 0.25,
            "inside {inside:.3}, across {across:.3}"
        );
    }

    #[test]
    fn each_drones_nearest_vector_is_a_group_mate() {
        // What a listener would use: for every drone, the most similar other vector (cosine)
        // should belong to a drone of the same group.
        let g = two_groups();
        let walks = random_walks(&g, 10, 10, &mut ChaCha8Rng::seed_from_u64(5));
        let config = SkipGramConfig {
            dim: 16,
            window: 2,
            epochs: 3,
            ..Default::default()
        };
        let z = train_skipgram(&walks, 12, &config, &mut ChaCha8Rng::seed_from_u64(6));
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

    const EPS: f64 = 1e-6;

    fn loss_of(z_u: &[f64], z_v: &[f64], c_neg: &[Vec<f64>]) -> f64 {
        let negs: Vec<&[f64]> = c_neg.iter().map(|c| c.as_slice()).collect();
        pair_loss_and_grads(z_u, z_v, &negs).0
    }

    #[test]
    fn sigmoid_values() {
        assert_eq!(sigmoid(0.0), 0.5);
        for x in [-3.0, -0.4, 1.0, 7.5] {
            assert!((sigmoid(-x) - (1.0 - sigmoid(x))).abs() < 1e-12);
        }
        assert!(sigmoid(40.0) <= 1.0 && sigmoid(-40.0) > 0.0);
    }

    #[test]
    fn dot_by_hand() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 0.5]), 4.0 - 10.0 + 1.5);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn at_the_start_the_loss_is_6_ln_2() {
        // C starts at zero: every dot product is 0, every σ is 0.5. One positive and five
        // negatives each cost −ln 0.5 = ln 2. C gets a push at once (−0.5 · z_u), Z doesn't.
        let z_u = [0.3, -0.1, 0.2];
        let zero = [0.0; 3];
        let negs: Vec<&[f64]> = vec![&zero; 5];
        let (loss, grad_z, grad_zv, grad_neg) = pair_loss_and_grads(&z_u, &zero, &negs);

        assert!((loss - 6.0 * 2.0_f64.ln()).abs() < 1e-12, "loss {loss}");
        assert_eq!(grad_z, [0.0; 3]);
        assert_eq!(grad_zv, [-0.15, 0.05, -0.1]);
        assert_eq!(grad_neg.len(), 5);
        for g in &grad_neg {
            assert_eq!(*g, [0.15, -0.05, 0.1]);
        }
    }

    #[test]
    fn gradients_match_the_numerical_derivative() {
        // Nudge every single number by ±EPS and measure how the loss moves:
        // (L(x + EPS) − L(x − EPS)) / 2·EPS must equal the formula's gradient.
        let z_u = vec![0.4, -0.7, 0.1, 0.9];
        let z_v = vec![-0.2, 0.5, 0.8, -0.3];
        let c_neg = vec![
            vec![0.6, 0.1, -0.4, 0.2],
            vec![-0.9, 0.3, 0.0, 0.7],
            vec![0.05, -0.6, 0.5, -0.1],
        ];
        let negs: Vec<&[f64]> = c_neg.iter().map(|c| c.as_slice()).collect();
        let (_, grad_z, grad_zv, grad_neg) = pair_loss_and_grads(&z_u, &z_v, &negs);

        let close = |numeric: f64, formula: f64, what: &str| {
            assert!(
                (numeric - formula).abs() < 1e-7,
                "{what}: numeric {numeric:.9}, formula {formula:.9}"
            );
        };

        for i in 0..z_u.len() {
            let (mut up, mut down) = (z_u.clone(), z_u.clone());
            up[i] += EPS;
            down[i] -= EPS;
            let numeric = (loss_of(&up, &z_v, &c_neg) - loss_of(&down, &z_v, &c_neg)) / (2.0 * EPS);
            close(numeric, grad_z[i], &format!("∂L/∂z_u[{i}]"));
        }
        for i in 0..z_v.len() {
            let (mut up, mut down) = (z_v.clone(), z_v.clone());
            up[i] += EPS;
            down[i] -= EPS;
            let numeric = (loss_of(&z_u, &up, &c_neg) - loss_of(&z_u, &down, &c_neg)) / (2.0 * EPS);
            close(numeric, grad_zv[i], &format!("∂L/∂z_v[{i}]"));
        }
        for k in 0..c_neg.len() {
            for i in 0..c_neg[k].len() {
                let (mut up, mut down) = (c_neg.clone(), c_neg.clone());
                up[k][i] += EPS;
                down[k][i] -= EPS;
                let numeric = (loss_of(&z_u, &z_v, &up) - loss_of(&z_u, &z_v, &down)) / (2.0 * EPS);
                close(numeric, grad_neg[k][i], &format!("∂L/∂c_n{k}[{i}]"));
            }
        }
    }

    #[test]
    fn a_small_step_against_the_gradient_lowers_the_loss() {
        let z_u = vec![0.4, -0.7, 0.1];
        let z_v = vec![-0.2, 0.5, 0.8];
        let c_neg = [vec![0.6, 0.1, -0.4], vec![-0.9, 0.3, 0.0]];
        let negs: Vec<&[f64]> = c_neg.iter().map(|c| c.as_slice()).collect();
        let (before, grad_z, grad_zv, grad_neg) = pair_loss_and_grads(&z_u, &z_v, &negs);

        let lr = 0.025;
        let step = |v: &[f64], g: &[f64]| -> Vec<f64> {
            v.iter().zip(g).map(|(x, g)| x - lr * g).collect()
        };
        let z_u = step(&z_u, &grad_z);
        let z_v = step(&z_v, &grad_zv);
        let c_neg: Vec<Vec<f64>> = c_neg
            .iter()
            .zip(&grad_neg)
            .map(|(c, g)| step(c, g))
            .collect();

        let after = loss_of(&z_u, &z_v, &c_neg);
        assert!(after < before, "loss went from {before} to {after}");
    }
}
