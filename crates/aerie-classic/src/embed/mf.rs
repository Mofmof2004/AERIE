//! Matrix factorisation: Z such that z_u · z_v ≈ A_uv, the simplest shallow embedding.
//!
//! Same encoder and decoder as DeepWalk (a lookup table Z, a dot product), but no walks:
//! the vectors are fitted to the adjacency matrix itself, so they only see direct links.
//! The plan's baseline: does looking further than one hop (DeepWalk) help?
//!
//! ```text
//! e_uv = z_u · z_v − A_uv
//! L    = Σ_u Σ_{v≠u} e_uv² + l2 · Σ_u |z_u|²
//! ```
//!
//! The diagonal is left out: a drone isn't linked to itself, and asking |z_u|² ≈ 0 would
//! fight everything else. Trained by full-batch gradient descent: every pair, every epoch.

use faer::Mat;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::skipgram::dot;

/// Settings for matrix factorisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfConfig {
    /// Numbers per vector.
    pub dim: usize,
    /// Full gradient steps.
    pub epochs: usize,
    /// Step size. The gradient sums over every other drone, so it grows with n and with
    /// the size of A: the default suits A = ln(1 + messages) on a few hundred drones.
    pub lr: f64,
    /// Weight of the penalty on long vectors.
    pub l2: f64,
}

impl Default for MfConfig {
    fn default() -> Self {
        MfConfig {
            dim: 32,
            epochs: 300,
            lr: 0.001,
            l2: 0.01,
        }
    }
}

/// L = Σ_u Σ_{v≠u} (z_u · z_v − A_uv)² + l2 · Σ_u |z_u|²
pub fn mf_loss(a: &Mat<f64>, z: &[Vec<f64>], l2: f64) -> f64 {
    let n = z.len();
    let mut loss = 0.0;
    for u in 0..n {
        for v in 0..n {
            if v == u {
                continue; // a drone isn't linked to itself
            }
            let e = dot(&z[u], &z[v]) - a[(u, v)];
            loss += e * e;
        }
        loss += l2 * dot(&z[u], &z[u]);
    }
    loss
}

/// ∂L/∂z_u for every drone u: one vector of `dim` numbers each.
///
/// ∂L/∂z_u = 4 Σ_{v≠u} e_uv · z_v + 2 · l2 · z_u, with e_uv = z_u · z_v − A_uv.
/// The 4: z_u is in e_uv (as the first vector) and in e_vu (as the second), and the two
/// are equal because A is symmetric; each e² gives 2e. Every gradient is computed from
/// the current Z: nothing is updated inside the loop.
pub fn mf_gradient(a: &Mat<f64>, z: &[Vec<f64>], l2: f64) -> Vec<Vec<f64>> {
    let n = z.len();
    let dim = z[0].len();
    let mut grad = vec![vec![0.0; dim]; n];
    for u in 0..n {
        for v in 0..n {
            if v == u {
                continue; // a drone isn't linked to itself
            }
            let e = dot(&z[u], &z[v]) - a[(u, v)];
            for (g, &zv) in grad[u].iter_mut().zip(&z[v]) {
                *g += 4.0 * e * zv;
            }
        }
        for (g, &zu) in grad[u].iter_mut().zip(&z[u]) {
            *g += 2.0 * l2 * zu;
        }
    }
    grad
}

/// Z from A by gradient descent: Z random in ±0.5/dim (stream 2 of `seed`, like
/// skip-gram), then `epochs` full steps against the gradient. `a` must be symmetric.
/// Returns n vectors of `dim` numbers, in drone order. Same arguments, same Z.
///
/// Z can't start at zero: then every z_v is 0, so every gradient is 0 and nothing moves.
pub fn matrix_factorization(a: &Mat<f64>, config: &MfConfig, seed: u64) -> Vec<Vec<f64>> {
    let n = a.nrows();
    let half = 0.5 / config.dim as f64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(2);
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..config.dim)
                .map(|_| rng.random_range(-half..half))
                .collect()
        })
        .collect();

    for _ in 0..config.epochs {
        let grad = mf_gradient(a, &z, config.l2);
        for (zu, gu) in z.iter_mut().zip(&grad) {
            for (x, g) in zu.iter_mut().zip(gu) {
                *x -= config.lr * g;
            }
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two 6-drone cliques (0–5 and 6–11) joined by one bridge 5–6, weights 1.
    fn two_groups() -> Mat<f64> {
        let mut a = Mat::zeros(12, 12);
        for g in [0, 6] {
            for i in g..g + 6 {
                for j in g..g + 6 {
                    if i != j {
                        a[(i, j)] = 1.0;
                    }
                }
            }
        }
        a[(5, 6)] = 1.0;
        a[(6, 5)] = 1.0;
        a
    }

    fn small_z(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random_range(-0.5..0.5)).collect())
            .collect()
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
    }

    #[test]
    fn loss_by_hand() {
        // 3 drones, dim 1, a link 0–1 only. z = (0.9, 0.8, 0.5).
        // Pairs: (0,1) 0.72 − 1 = −0.28, (0,2) 0.45, (1,2) 0.40, each counted twice.
        let mut a = Mat::zeros(3, 3);
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 1.0;
        let z = vec![vec![0.9], vec![0.8], vec![0.5]];
        let pairs = 2.0 * (0.28_f64.powi(2) + 0.45_f64.powi(2) + 0.40_f64.powi(2));
        assert!((mf_loss(&a, &z, 0.0) - pairs).abs() < 1e-12);
        let lengths = 0.81 + 0.64 + 0.25;
        assert!((mf_loss(&a, &z, 0.1) - pairs - 0.1 * lengths).abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_the_numerical_derivative() {
        // Nudge each number of Z by ±EPS and compare the change of L with the formula.
        const EPS: f64 = 1e-6;
        let a = two_groups();
        let z = small_z(12, 3, 1);
        let grad = mf_gradient(&a, &z, 0.05);
        for u in [0, 5, 6, 11] {
            for i in 0..3 {
                let mut plus = z.clone();
                plus[u][i] += EPS;
                let mut minus = z.clone();
                minus[u][i] -= EPS;
                let numeric = (mf_loss(&a, &plus, 0.05) - mf_loss(&a, &minus, 0.05)) / (2.0 * EPS);
                assert!(
                    (numeric - grad[u][i]).abs() < 1e-6,
                    "z[{u}][{i}]: formula {} vs numeric {numeric}",
                    grad[u][i]
                );
            }
        }
    }

    #[test]
    fn a_zero_start_never_moves() {
        let a = two_groups();
        let zero = vec![vec![0.0; 4]; 12];
        assert!(
            mf_gradient(&a, &zero, 0.01)
                .iter()
                .flatten()
                .all(|&g| g == 0.0)
        );
    }

    #[test]
    fn training_lowers_the_loss_and_is_reproducible() {
        let a = two_groups();
        let config = MfConfig {
            dim: 4,
            epochs: 200,
            lr: 0.01,
            l2: 0.01,
        };
        let start = MfConfig {
            epochs: 0,
            ..config
        };
        let before = mf_loss(&a, &matrix_factorization(&a, &start, 3), config.l2);
        let z = matrix_factorization(&a, &config, 3);
        let after = mf_loss(&a, &z, config.l2);
        assert!(after < 0.2 * before, "loss {before} -> {after}");
        assert_eq!((z.len(), z[0].len()), (12, 4));
        assert_eq!(z, matrix_factorization(&a, &config, 3));
    }

    #[test]
    fn the_two_cliques_get_their_own_directions() {
        // Inside a clique the dot products should be near 1, across near 0.
        let a = two_groups();
        let config = MfConfig {
            dim: 4,
            epochs: 500,
            lr: 0.01,
            l2: 0.01,
        };
        let z = matrix_factorization(&a, &config, 4);
        let (mut inside, mut across) = (0.0, 0.0);
        for i in 0..12 {
            for j in 0..12 {
                if i == j {
                    continue;
                }
                if (i < 6) == (j < 6) {
                    inside += cosine(&z[i], &z[j]) / 60.0;
                } else {
                    across += cosine(&z[i], &z[j]) / 72.0;
                }
            }
        }
        assert!(
            inside > 0.9 && across < 0.2,
            "inside {inside}, across {across}"
        );
        let fit = dot(&z[0], &z[1]);
        assert!((fit - 1.0).abs() < 0.15, "z_0 · z_1 = {fit}");
    }
}
