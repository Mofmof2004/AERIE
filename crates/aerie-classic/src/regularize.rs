//! Mission 3: rebuild a signal on a graph from noisy, partial readings.
//!
//! - `tikhonov`: min ‖M(x − y)‖² + γ xᵀLx, one linear solve. Smooth maps; blurs edges.
//! - `trend_filtering` (order 1): min ½‖M(x − y)‖² + λ ‖Lx‖₁, by ADMM. Few drones bend:
//!   straight pieces joined at kinks.
//! - `total_variation` (trend filtering of order 0): λ ‖Bᵀx‖₁ instead, by ADMM. Few links
//!   jump: flat pieces, sharp edges.
//! - `sparse_total_variation_run`: total variation + λ₂ Σ_i |x_i|, by ADMM: flat pieces,
//!   most of the map exactly 0.
//! - `Regularizer` + `reconstruct`: any of them through one entry point.
//!
//! In all of them, `y[i]` is drone i's reading, used only where `has_sensor[i]` is true (M picks
//! those drones out). A connected piece with no sensor at all has nothing to pin its level:
//! the matrix can't be inverted there, and the answer is garbage (inf or NaN). Check the
//! pieces first.

use aerie_graph::EdgeList;
use faer::Mat;
use faer::linalg::solvers::Solve;

/// Tikhonov with a sensor mask: x̂ = (MᵀM + γL)⁻¹ MᵀM y.
/// `y[i]` is drone i's reading, used only where `has_sensor[i]` is true.
pub fn tikhonov(graph: &EdgeList, y: &[f64], has_sensor: &[bool], gamma: f64) -> Vec<f64> {
    let n = graph.n as usize;
    let laplacian = graph.laplacian();

    // We want x̂ such that   K · x̂ = b,
    // with                  K = MᵀM + γL   and   b = MᵀM y.

    // ---- Step 1: the matrix K ------------------------------------------------------
    // Start from γ × L.
    let mut k: Mat<f64> = Mat::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            k[(row, col)] = gamma * laplacian[(row, col)];
        }
    }
    // Then add MᵀM: a 1 on the diagonal for every drone that has a sensor.
    for drone in 0..n {
        if has_sensor[drone] {
            k[(drone, drone)] += 1.0;
        }
    }

    // ---- Step 2: the right-hand side b ---------------------------------------------
    // b = MᵀM y: the drone's reading if it has a sensor, 0 if it has none.
    let mut b: Mat<f64> = Mat::zeros(n, 1);
    for drone in 0..n {
        if has_sensor[drone] {
            b[(drone, 0)] = y[drone];
        }
    }

    // ---- Step 3: solve K · x̂ = b ---------------------------------------------------
    // faer solves the system of equations (Gaussian elimination, as by hand).
    let solution: Mat<f64> = k.partial_piv_lu().solve(&b);

    // ---- Step 4: return x̂ as a list, one number per drone --------------------------
    let mut x_hat: Vec<f64> = Vec::new();
    for drone in 0..n {
        x_hat.push(solution[(drone, 0)]);
    }
    x_hat
}

/// Moves v toward 0 by k, stopping at 0: small values become exactly 0.
///
/// The answer to min_z k|z| + ½(z − v)²: the step of ADMM that keeps edges. A jump
/// smaller than k is erased (a flat region), a bigger one only shrinks by k (an edge).
pub fn soft_threshold(v: f64, k: f64) -> f64 {
    if v > k {
        v - k
    } else if v < -k {
        v + k
    } else {
        0.0
    }
}

/// Bᵀx: the jump √w (x_src − x_dst) on every link, straight from the links (no matrix).
fn jumps(graph: &EdgeList, x: &[f64]) -> Vec<f64> {
    graph
        .edges
        .iter()
        .map(|e| e.weight.sqrt() * (x[e.src as usize] - x[e.dst as usize]))
        .collect()
}

/// B v for one value per link: each link adds √w v to its first drone and takes it from
/// its second (no matrix).
fn spread(graph: &EdgeList, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; graph.n as usize];
    for (e, &ve) in graph.edges.iter().zip(v) {
        let s = e.weight.sqrt() * ve;
        out[e.src as usize] += s;
        out[e.dst as usize] -= s;
    }
    out
}

/// What an iterative (ADMM) reconstruction gives: the map, and how the loop converged.
#[derive(Debug, Clone, PartialEq)]
pub struct Reconstruction {
    /// The map x̂: one value per drone.
    pub x: Vec<f64>,
    /// After each round: |Dx − z|, how far the map's differences still are from the ones
    /// the shrinking step allows. Falls toward 0 as the loop converges. Empty for the
    /// closed forms.
    pub gaps: Vec<f64>,
}

/// The older name, kept for the playground.
pub type TrendFiltering = Reconstruction;

/// K = MᵀM + c · L: a 1 on the diagonal for each sensor drone, plus c times the Laplacian.
/// Every reconstruction here solves a system with this matrix (c = γ, ρ, …).
fn mask_plus_laplacian(graph: &EdgeList, has_sensor: &[bool], c: f64) -> Mat<f64> {
    let n = graph.n as usize;
    let l = graph.laplacian();
    let mut k: Mat<f64> = Mat::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            k[(row, col)] = c * l[(row, col)];
        }
        if has_sensor[row] {
            k[(row, row)] += 1.0;
        }
    }
    k
}

/// MᵀM y: the reading for a sensor drone, 0 otherwise.
fn masked_readings(y: &[f64], has_sensor: &[bool]) -> Vec<f64> {
    y.iter()
        .zip(has_sensor)
        .map(|(&v, &s)| if s { v } else { 0.0 })
        .collect()
}

/// Solves K x = rhs with a prepared solver, as a `Vec`.
fn solve_with(solver: &impl Solve<f64>, rhs: &[f64]) -> Vec<f64> {
    let mut b: Mat<f64> = Mat::zeros(rhs.len(), 1);
    for (i, &v) in rhs.iter().enumerate() {
        b[(i, 0)] = v;
    }
    let x = solver.solve(&b);
    (0..rhs.len()).map(|i| x[(i, 0)]).collect()
}

// ------------------------------------------------------------------------------ penalties

/// Total variation ‖Bᵀx‖₁ = Σ over links √w |x_i − x_j|: the sum of the jumps, not
/// squared. Small when few links jump (flat pieces, sharp edges).
pub fn tv_penalty(graph: &EdgeList, x: &[f64]) -> f64 {
    jumps(graph, x).iter().map(|j| j.abs()).sum()
}

/// Sparse total variation λ₁ ‖Bᵀx‖₁ + λ₂ ‖x‖₁: the jumps and the values themselves, not
/// squared. Small when the map is flat in pieces and 0 almost everywhere.
pub fn stv_penalty(graph: &EdgeList, x: &[f64], lambda1: f64, lambda2: f64) -> f64 {
    lambda1 * tv_penalty(graph, x) + lambda2 * x.iter().map(|v| v.abs()).sum::<f64>()
}

// ---------------------------------------------------------------------- reconstructions

/// Total variation denoising (trend filtering of order 0): min ½‖M(x − y)‖² + λ ‖Bᵀx‖₁.
/// Keeps sharp edges: small jumps become 0, big ones survive. See `total_variation_run`.
pub fn total_variation(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Vec<f64> {
    total_variation_run(graph, y, has_sensor, lambda, rho, iterations).x
}

/// Total variation by ADMM, keeping the gap after each round.
///
/// The jumps get their own variable z (one per link), tied to the map by z = Bᵀx, and three
/// easy steps take turns:
///
/// ```text
/// x ← solve (MᵀM + ρL) x = MᵀM y + ρ B (z − u)     the fit, a Tikhonov-like solve
/// z ← soft_threshold(Bᵀx + u, λ/ρ)  link by link     the jumps: small ones to 0
/// u ← u + Bᵀx − z                                     the running tally of disagreement
/// ```
///
/// ρ only sets how hard each round pulls x and z together: any ρ > 0 reaches the same
/// answer (1 is a good start). K = MᵀM + ρL never changes, so its solver is prepared once.
/// Starts from x = z = u = 0; `iterations` rounds.
pub fn total_variation_run(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Reconstruction {
    let n = graph.n as usize;
    let m = graph.edges.len();
    let solver = mask_plus_laplacian(graph, has_sensor, rho).partial_piv_lu();
    let readings = masked_readings(y, has_sensor);

    let mut x = vec![0.0; n];
    let mut z = vec![0.0; m];
    let mut u = vec![0.0; m];
    let mut gaps = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // 1. x: the fit, pulled toward the jumps z (minus the tally u).
        let z_minus_u: Vec<f64> = z.iter().zip(&u).map(|(zi, ui)| zi - ui).collect();
        let pull = spread(graph, &z_minus_u);
        let rhs: Vec<f64> = readings
            .iter()
            .zip(&pull)
            .map(|(r, p)| r + rho * p)
            .collect();
        x = solve_with(&solver, &rhs);

        // 2. z: the map's jumps (plus the tally), small ones erased.
        let bx = jumps(graph, &x);
        for ((ze, &jump), &ue) in z.iter_mut().zip(&bx).zip(&u) {
            *ze = soft_threshold(jump + ue, lambda / rho);
        }

        // 3. u: add what is still left between the map's jumps and z.
        let mut gap = 0.0;
        for ((ue, &jump), &ze) in u.iter_mut().zip(&bx).zip(&z) {
            *ue += jump - ze;
            gap += (jump - ze).powi(2);
        }
        gaps.push(gap.sqrt());
    }
    Reconstruction { x, gaps }
}

/// Trend filtering (order 1): min ½‖M(x − y)‖² + λ ‖Lx‖₁. See `trend_filtering_run`.
pub fn trend_filtering(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Vec<f64> {
    trend_filtering_run(graph, y, has_sensor, lambda, rho, iterations).x
}

/// Trend filtering of order 1 by ADMM: min ½‖M(x − y)‖² + λ ‖Lx‖₁.
///
/// (Lx)_i = Σ_j w_ij (x_i − x_j) is how far drone i sits above its neighbours: 0 where the
/// signal is locally a straight slope, big at a kink. So most drones get (Lx)_i = 0 and the
/// map comes out as straight pieces joined at a few kinks (total variation, order 0, gives
/// flat pieces instead).
///
/// ```text
/// x ← solve (MᵀM + ρ L²) x = MᵀM y + ρ L (z − u)
/// z ← soft_threshold(Lx + u, λ/ρ)   drone by drone
/// u ← u + Lx − z
/// ```
pub fn trend_filtering_run(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Reconstruction {
    let n = graph.n as usize;
    let l = graph.laplacian();
    let l2 = &l * &l;
    let mut k: Mat<f64> = Mat::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            k[(row, col)] = rho * l2[(row, col)];
        }
        if has_sensor[row] {
            k[(row, row)] += 1.0;
        }
    }
    let solver = k.partial_piv_lu();
    let readings = masked_readings(y, has_sensor);
    let times_l = |v: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| l[(i, j)] * v[j]).sum())
            .collect()
    };

    let mut x = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut gaps = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // 1. x: the fit, pulled toward the bends z (minus the tally u).
        let z_minus_u: Vec<f64> = z.iter().zip(&u).map(|(zi, ui)| zi - ui).collect();
        let pull = times_l(&z_minus_u);
        let rhs: Vec<f64> = readings
            .iter()
            .zip(&pull)
            .map(|(r, p)| r + rho * p)
            .collect();
        x = solve_with(&solver, &rhs);
        // 2. z: each drone's bend (plus the tally), small ones erased.
        let lx = times_l(&x);
        for ((zi, &v), &ui) in z.iter_mut().zip(&lx).zip(&u) {
            *zi = soft_threshold(v + ui, lambda / rho);
        }
        // 3. u.
        let mut gap = 0.0;
        for ((ui, &v), &zi) in u.iter_mut().zip(&lx).zip(&z) {
            *ui += v - zi;
            gap += (v - zi).powi(2);
        }
        gaps.push(gap.sqrt());
    }
    Reconstruction { x, gaps }
}

/// Theory for denoising with Tikhonov (a reading at every drone, noise σ on each):
/// returns (bias², variance) at this γ, with H = (I + γL)⁻¹.
///
/// x̂ = H y = H (x + noise), so on average x̂ = H x: the bias is (H − I) x (the blur), and
/// bias² = |H x − x|². The noise goes through H too: its total variance is
/// σ² · trace(H Hᵀ), the sum of the squares of every entry of H. Their sum is the expected
/// error E|x̂ − x|². γ = 0: no bias, variance n σ²; γ → ∞ (connected graph): bias²
/// Σ (x_i − mean)², variance σ².
pub fn bias_variance(graph: &EdgeList, x_true: &[f64], sigma: f64, gamma: f64) -> (f64, f64) {
    let n = graph.n as usize;
    let laplacian = graph.laplacian();

    let mut k: Mat<f64> = Mat::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            k[(row, col)] = gamma * laplacian[(row, col)];
        }
        k[(row, row)] += 1.0;
    }

    let identity: Mat<f64> = Mat::identity(n, n);
    let h: Mat<f64> = k.partial_piv_lu().solve(&identity);

    let mut bias2 = 0.0;
    for i in 0..n {
        let mut hx_i = 0.0;
        for j in 0..n {
            hx_i += h[(i, j)] * x_true[j];
        }
        bias2 += (hx_i - x_true[i]).powi(2);
    }

    // ---- Step 4: variance = σ² · (sum of the squares of every entry of H) ----------
    let mut sum_squares = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum_squares += h[(i, j)] * h[(i, j)];
        }
    }
    let variance = sigma * sigma * sum_squares;

    (bias2, variance)
}

/// ‖Lx‖₁ = Σ_i |Σ_j w_ij (x_i − x_j)|: how much the slope bends, drone by drone.
pub fn tf_penalty(graph: &EdgeList, x: &[f64]) -> f64 {
    let l = graph.laplacian();
    let n = x.len();
    (0..n)
        .map(|i| (0..n).map(|j| l[(i, j)] * x[j]).sum::<f64>().abs())
        .sum()
}

/// Sparse total variation by ADMM: min ½‖M(x − y)‖² + λ₁ ‖Bᵀx‖₁ + λ₂ ‖x‖₁.
///
/// Total variation with a second term that pulls each value itself toward 0: the map comes
/// out flat in pieces (the jumps) and exactly 0 almost everywhere (the values), with a few
/// patches that survive. λ₂ = 0 is total variation.
///
/// Two helper variables, one for the jumps (z₁ = Bᵀx, one per link) and one for the values
/// (z₂ = x, one per drone), each with its own tally (u₁, u₂):
///
/// ```text
/// x  ← solve (MᵀM + ρ(L + I)) x = MᵀM y + ρ B (z₁ − u₁) + ρ (z₂ − u₂)
/// z₁ ← soft_threshold(Bᵀx + u₁, λ₁/ρ)   link by link
/// z₂ ← soft_threshold(x + u₂, λ₂/ρ)      drone by drone
/// u₁ ← u₁ + Bᵀx − z₁,   u₂ ← u₂ + x − z₂
/// ```
///
/// The I in the x-step's matrix makes it invertible even in a piece with no sensor (the
/// sparsity term pins such a piece to 0). The gap is |Bᵀx − z₁| and |x − z₂| together.
pub fn sparse_total_variation_run(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda1: f64,
    lambda2: f64,
    rho: f64,
    iterations: usize,
) -> Reconstruction {
    let n = graph.n as usize;
    let m = graph.edges.len();
    let mut k = mask_plus_laplacian(graph, has_sensor, rho);
    for i in 0..n {
        k[(i, i)] += rho;
    }
    let solver = k.partial_piv_lu();
    let readings = masked_readings(y, has_sensor);

    let mut x = vec![0.0; n];
    let (mut z1, mut u1) = (vec![0.0; m], vec![0.0; m]);
    let (mut z2, mut u2) = (vec![0.0; n], vec![0.0; n]);
    let mut gaps = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // 1. x: the fit, pulled toward the jumps z₁ and the values z₂ (minus their tallies).
        let z1_minus_u1: Vec<f64> = z1.iter().zip(&u1).map(|(z, u)| z - u).collect();
        let pull = spread(graph, &z1_minus_u1);
        let rhs: Vec<f64> = (0..n)
            .map(|i| readings[i] + rho * pull[i] + rho * (z2[i] - u2[i]))
            .collect();
        x = solve_with(&solver, &rhs);

        // 2. z₁: the jumps, small ones erased; z₂: the values, small ones erased.
        let bx = jumps(graph, &x);
        for ((z, &j), &u) in z1.iter_mut().zip(&bx).zip(&u1) {
            *z = soft_threshold(j + u, lambda1 / rho);
        }
        for ((z, &xi), &u) in z2.iter_mut().zip(&x).zip(&u2) {
            *z = soft_threshold(xi + u, lambda2 / rho);
        }

        // 3. The tallies.
        let mut gap = 0.0;
        for ((u, &j), &z) in u1.iter_mut().zip(&bx).zip(&z1) {
            *u += j - z;
            gap += (j - z).powi(2);
        }
        for ((u, &xi), &z) in u2.iter_mut().zip(&x).zip(&z2) {
            *u += xi - z;
            gap += (xi - z).powi(2);
        }
        gaps.push(gap.sqrt());
    }
    Reconstruction { x, gaps }
}

// ---------------------------------------------------------------------- one entry point

/// Every regularizer of Mission 3, with its parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularizer {
    /// ‖M(x − y)‖² + γ xᵀLx: smooth slopes.
    Tikhonov { gamma: f64 },
    /// ½‖M(x − y)‖² + λ ‖Lx‖₁ (trend filtering, order 1): straight pieces, few kinks.
    TrendFiltering { lambda: f64 },
    /// ½‖M(x − y)‖² + λ ‖Bᵀx‖₁ (total variation): flat pieces, sharp edges.
    TotalVariation { lambda: f64 },
    /// ½‖M(x − y)‖² + λ₁ ‖Bᵀx‖₁ + λ₂ ‖x‖₁: flat pieces, most of the map exactly 0.
    SparseTotalVariation { lambda1: f64, lambda2: f64 },
}

impl Regularizer {
    pub fn name(self) -> &'static str {
        match self {
            Regularizer::Tikhonov { .. } => "Tikhonov",
            Regularizer::TrendFiltering { .. } => "trend filtering",
            Regularizer::TotalVariation { .. } => "total variation",
            Regularizer::SparseTotalVariation { .. } => "sparse total variation",
        }
    }

    /// Solved by ADMM (rounds and a convergence curve) rather than one linear solve.
    pub fn is_iterative(self) -> bool {
        matches!(
            self,
            Regularizer::TrendFiltering { .. }
                | Regularizer::TotalVariation { .. }
                | Regularizer::SparseTotalVariation { .. }
        )
    }

    /// The penalty term at x (without the fit).
    pub fn penalty(self, graph: &EdgeList, x: &[f64]) -> f64 {
        match self {
            Regularizer::Tikhonov { gamma } => gamma * graph.variation(x),
            Regularizer::TrendFiltering { lambda } => lambda * tf_penalty(graph, x),
            Regularizer::TotalVariation { lambda } => lambda * tv_penalty(graph, x),
            Regularizer::SparseTotalVariation { lambda1, lambda2 } => {
                stv_penalty(graph, x, lambda1, lambda2)
            }
        }
    }
}

/// Rebuilds the map with any regularizer (ADMM ones with ρ = 1 and `iterations` rounds).
pub fn reconstruct(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    reg: Regularizer,
    iterations: usize,
) -> Reconstruction {
    let closed = |x| Reconstruction {
        x,
        gaps: Vec::new(),
    };
    match reg {
        Regularizer::Tikhonov { gamma } => closed(tikhonov(graph, y, has_sensor, gamma)),
        Regularizer::TrendFiltering { lambda } => {
            trend_filtering_run(graph, y, has_sensor, lambda, 1.0, iterations)
        }
        Regularizer::TotalVariation { lambda } => {
            total_variation_run(graph, y, has_sensor, lambda, 1.0, iterations)
        }
        Regularizer::SparseTotalVariation { lambda1, lambda2 } => {
            sparse_total_variation_run(graph, y, has_sensor, lambda1, lambda2, 1.0, iterations)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_graph::Edge;

    fn path(n: u32) -> EdgeList {
        EdgeList {
            n,
            edges: (0..n - 1)
                .map(|i| Edge {
                    src: i,
                    dst: i + 1,
                    weight: 1.0,
                })
                .collect(),
        }
    }

    /// 1 on the first half of the path, 0 on the second: one sharp edge.
    fn step(n: usize) -> Vec<f64> {
        (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect()
    }

    /// ½‖M(x − y)‖² + λ ‖Bᵀx‖₁: what trend filtering minimises.
    fn tf_objective(g: &EdgeList, x: &[f64], y: &[f64], has: &[bool], lambda: f64) -> f64 {
        let fit: f64 = (0..x.len())
            .filter(|&i| has[i])
            .map(|i| 0.5 * (x[i] - y[i]).powi(2))
            .sum();
        fit + lambda * jumps(g, x).iter().map(|j| j.abs()).sum::<f64>()
    }

    #[test]
    fn the_three_drone_example() {
        // 0 — 1 — 2, weights 1, sensors on 0 and 2 reading 1 and 3, γ = 1.
        let g = path(3);
        let x = tikhonov(&g, &[1.0, 99.0, 3.0], &[true, false, true], 1.0);
        for (got, want) in x.iter().zip([1.5, 2.0, 2.5]) {
            assert!((got - want).abs() < 1e-12, "{x:?}");
        }
    }

    #[test]
    fn tikhonov_at_both_ends_of_gamma() {
        let g = path(6);
        let y = [0.3, 1.2, -0.4, 2.0, 0.9, 1.1];
        let all = [true; 6];
        let keep = tikhonov(&g, &y, &all, 1e-10);
        assert!(
            keep.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-8),
            "γ → 0 keeps y"
        );
        let flat = tikhonov(&g, &y, &all, 1e10);
        let mean = y.iter().sum::<f64>() / 6.0;
        assert!(
            flat.iter().all(|v| (v - mean).abs() < 1e-6),
            "γ → ∞ flattens to the mean"
        );
    }

    #[test]
    fn soft_threshold_by_hand() {
        assert!((soft_threshold(0.5, 0.1) - 0.4).abs() < 1e-12);
        assert!((soft_threshold(-0.3, 0.1) + 0.2).abs() < 1e-12);
        assert_eq!(soft_threshold(0.05, 0.1), 0.0);
        assert_eq!(soft_threshold(-0.1, 0.1), 0.0);
    }

    #[test]
    fn total_variation_converges_and_beats_its_neighbours() {
        let g = path(20);
        let y: Vec<f64> = step(20)
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.1 * ((i * 7 % 5) as f64 - 2.0))
            .collect();
        let has = vec![true; 20];
        let run = total_variation_run(&g, &y, &has, 0.3, 1.0, 2000);
        assert!(
            *run.gaps.last().unwrap() < 1e-8,
            "gap {}",
            run.gaps.last().unwrap()
        );
        // A minimum: nudging any drone up or down makes the objective worse.
        let best = tf_objective(&g, &run.x, &y, &has, 0.3);
        for i in 0..20 {
            for d in [-1e-3, 1e-3] {
                let mut x = run.x.clone();
                x[i] += d;
                assert!(
                    tf_objective(&g, &x, &y, &has, 0.3) >= best - 1e-9,
                    "drone {i}"
                );
            }
        }
    }

    #[test]
    fn total_variation_keeps_the_edge_that_tikhonov_blurs() {
        // Clean step on a path of 30, a reading everywhere.
        let g = path(30);
        let y = step(30);
        let has = vec![true; 30];
        let tf = total_variation(&g, &y, &has, 0.5, 1.0, 3000);
        let tk = tikhonov(&g, &y, &has, 1.0);
        let tf_jumps = jumps(&g, &tf);
        let tk_jumps = jumps(&g, &tk);
        // Trend filtering: one jump of about 1 − 2λ/15 at the edge, flat everywhere else.
        assert!(tf_jumps[14] > 0.9, "edge {}", tf_jumps[14]);
        let others = tf_jumps
            .iter()
            .enumerate()
            .filter(|&(e, _)| e != 14)
            .map(|(_, j)| j.abs())
            .fold(0.0, f64::max);
        assert!(others < 1e-4, "flat pieces, largest other jump {others}");
        // Tikhonov: the edge spreads over several links.
        assert!(tk_jumps[14] < 0.6, "Tikhonov edge {}", tk_jumps[14]);
        assert!(
            tk_jumps[12] > 0.05,
            "Tikhonov blurs the neighbouring links too"
        );
    }

    #[test]
    fn total_variation_with_lambda_0_keeps_the_readings_and_fills_the_gaps() {
        // λ = 0: no penalty on jumps, so sensor drones keep their readings exactly.
        let g = path(5);
        let y = [1.0, 0.0, 3.0, 0.0, 2.0];
        let has = [true, false, true, false, true];
        let x = total_variation(&g, &y, &has, 0.0, 1.0, 500);
        for i in [0, 2, 4] {
            assert!((x[i] - y[i]).abs() < 1e-6, "drone {i}: {}", x[i]);
        }
        // Drones without a sensor sit between their neighbours.
        assert!(x[1] > 1.0 - 1e-6 && x[1] < 3.0 + 1e-6);
        assert!(x[3] > 2.0 - 1e-6 && x[3] < 3.0 + 1e-6);
    }

    #[test]
    fn any_rho_reaches_the_same_answer() {
        let g = path(12);
        let y = step(12);
        let has = vec![true; 12];
        let a = total_variation(&g, &y, &has, 0.2, 0.5, 4000);
        let b = total_variation(&g, &y, &has, 0.2, 2.0, 4000);
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-6));
    }

    fn triangle() -> EdgeList {
        EdgeList {
            n: 3,
            edges: vec![
                Edge {
                    src: 0,
                    dst: 1,
                    weight: 1.0,
                },
                Edge {
                    src: 1,
                    dst: 2,
                    weight: 4.0,
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
    fn penalties_by_hand() {
        // Triangle, x = (1, 2, 4): jumps √1·(1−2) = −1, √4·(2−4) = −4, √1·(1−4) = −3.
        let g = triangle();
        let x = [1.0, 2.0, 4.0];
        assert!((tv_penalty(&g, &x) - 8.0).abs() < 1e-12);
        // Sparse TV: 0.5 × 8 + 0.1 × (1 + 2 + 4).
        assert!((stv_penalty(&g, &x, 0.5, 0.1) - 4.7).abs() < 1e-12);
    }

    /// ½‖M(x − y)‖² + λ₁ ‖Bᵀx‖₁ + λ₂ ‖x‖₁.
    fn stv_objective(g: &EdgeList, x: &[f64], y: &[f64], has: &[bool], l1: f64, l2: f64) -> f64 {
        let fit: f64 = (0..x.len())
            .filter(|&i| has[i])
            .map(|i| 0.5 * (x[i] - y[i]).powi(2))
            .sum();
        fit + stv_penalty(g, x, l1, l2)
    }

    #[test]
    fn sparse_total_variation_converges_to_a_minimum() {
        let g = path(16);
        let y: Vec<f64> = step(16)
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.08 * ((i * 3 % 5) as f64 - 2.0))
            .collect();
        let has = vec![true; 16];
        let run = sparse_total_variation_run(&g, &y, &has, 0.2, 0.05, 1.0, 4000);
        assert!(
            *run.gaps.last().unwrap() < 1e-7,
            "gap {}",
            run.gaps.last().unwrap()
        );
        let best = stv_objective(&g, &run.x, &y, &has, 0.2, 0.05);
        for i in 0..16 {
            for d in [-1e-3, 1e-3] {
                let mut x = run.x.clone();
                x[i] += d;
                assert!(
                    stv_objective(&g, &x, &y, &has, 0.2, 0.05) >= best - 1e-9,
                    "drone {i}"
                );
            }
        }
    }

    #[test]
    fn sparse_total_variation_contains_total_variation_and_soft_thresholding() {
        let g = path(12);
        let y: Vec<f64> = step(12).iter().map(|v| v + 0.05).collect();
        let has = vec![true; 12];
        // λ₂ = 0: total variation.
        let a = sparse_total_variation_run(&g, &y, &has, 0.2, 0.0, 1.0, 3000).x;
        let b = total_variation(&g, &y, &has, 0.2, 1.0, 3000);
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-6));
        // λ₁ = 0, every drone measured: each value on its own, soft-thresholded by λ₂.
        let x = sparse_total_variation_run(&g, &y, &has, 0.0, 0.3, 1.0, 3000).x;
        for (xi, yi) in x.iter().zip(&y) {
            assert!((xi - soft_threshold(*yi, 0.3)).abs() < 1e-6);
        }
    }

    #[test]
    fn sparse_total_variation_keeps_a_patch_and_zeroes_the_rest() {
        // Path of 30: 0 everywhere except a patch of 1 on drones 12..18, slightly noisy.
        let g = path(30);
        let y: Vec<f64> = (0..30)
            .map(|i| if (12..18).contains(&i) { 1.0 } else { 0.0 } + 0.05 * ((i * 7 % 5) as f64 - 2.0))
            .collect();
        let has = vec![true; 30];
        let x = sparse_total_variation_run(&g, &y, &has, 0.1, 0.1, 1.0, 4000).x;
        let zeros = x.iter().filter(|v| v.abs() < 1e-6).count();
        assert!(zeros >= 20, "{zeros} drones exactly 0: {x:?}");
        assert!(x[14] > 0.6, "the patch survives: {}", x[14]);
        // Total variation alone leaves the background at small non-zero levels.
        let tv = total_variation(&g, &y, &has, 0.1, 1.0, 4000);
        assert!(tv.iter().filter(|v| v.abs() < 1e-6).count() < zeros);
    }

    #[test]
    fn reconstruct_runs_every_regularizer() {
        let g = path(10);
        let y = step(10);
        let has = vec![true; 10];
        for reg in [
            Regularizer::Tikhonov { gamma: 1.0 },
            Regularizer::TrendFiltering { lambda: 0.2 },
            Regularizer::TotalVariation { lambda: 0.2 },
            Regularizer::SparseTotalVariation {
                lambda1: 0.2,
                lambda2: 0.05,
            },
        ] {
            let r = reconstruct(&g, &y, &has, reg, 300);
            assert_eq!(r.x.len(), 10, "{}", reg.name());
            assert!(r.x.iter().all(|v| v.is_finite()), "{}", reg.name());
            assert_eq!(r.gaps.is_empty(), !reg.is_iterative(), "{}", reg.name());
            assert!(reg.penalty(&g, &r.x) >= 0.0);
        }
    }

    #[test]
    fn trend_filtering_keeps_a_kinked_slope_that_total_variation_turns_into_steps() {
        // A tent on a path of 31: up 0..15, down 15..30 (one kink), a reading everywhere.
        let g = path(31);
        let y: Vec<f64> = (0..31)
            .map(|i| 1.0 - ((i as f64) - 15.0).abs() / 15.0)
            .collect();
        let has = vec![true; 31];
        let tf = trend_filtering_run(&g, &y, &has, 0.05, 1.0, 4000);
        assert!(*tf.gaps.last().unwrap() < 1e-6, "converged");
        let err = |x: &[f64]| x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        let tv = total_variation(&g, &y, &has, 0.05, 1.0, 4000);
        assert!(
            err(&tf.x) < 0.2 * err(&tv),
            "tf {} vs tv {}",
            err(&tf.x),
            err(&tv)
        );
    }

    #[test]
    fn bias_variance_at_both_ends_of_gamma() {
        let g = path(8);
        let x: Vec<f64> = (0..8).map(|i| (i as f64 * 0.7).sin()).collect();
        let sigma = 0.3;
        // γ = 0: H = I, no blur, all the noise: n σ².
        let (b, v) = bias_variance(&g, &x, sigma, 0.0);
        assert!(b.abs() < 1e-20 && (v - 8.0 * sigma * sigma).abs() < 1e-12);
        // γ huge: everyone at the mean, noise averaged over all 8: σ².
        let (b, v) = bias_variance(&g, &x, sigma, 1e9);
        let mean = x.iter().sum::<f64>() / 8.0;
        let spread: f64 = x.iter().map(|xi| (xi - mean).powi(2)).sum();
        assert!((b - spread).abs() < 1e-6 && (v - sigma * sigma).abs() < 1e-6);
    }

    #[test]
    fn bias_rises_and_variance_falls_with_gamma() {
        let g = path(10);
        let x: Vec<f64> = (0..10).map(|i| (i as f64 * 0.5).sin()).collect();
        let curve: Vec<(f64, f64)> = [0.01, 0.1, 1.0, 10.0]
            .iter()
            .map(|&gamma| bias_variance(&g, &x, 0.4, gamma))
            .collect();
        assert!(
            curve.windows(2).all(|w| w[1].0 > w[0].0),
            "bias² goes up: {curve:?}"
        );
        assert!(
            curve.windows(2).all(|w| w[1].1 < w[0].1),
            "variance goes down: {curve:?}"
        );
    }

    #[test]
    fn bias_variance_matches_500_simulated_noise_draws() {
        // Experiment 1 in miniature: rebuild 500 noisy copies with tikhonov, measure the
        // blur of their mean and their spread, compare with the formulas.
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let g = path(12);
        let x: Vec<f64> = (0..12).map(|i| if i < 6 { 1.0 } else { 0.0 }).collect();
        let (sigma, gamma, draws) = (0.5, 0.8, 500);
        let all = vec![true; 12];
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut gauss = || {
            let u1: f64 = rng.random_range(1e-12..1.0);
            let u2: f64 = rng.random_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let maps: Vec<Vec<f64>> = (0..draws)
            .map(|_| {
                let y: Vec<f64> = x.iter().map(|v| v + sigma * gauss()).collect();
                tikhonov(&g, &y, &all, gamma)
            })
            .collect();
        let mean: Vec<f64> = (0..12)
            .map(|i| maps.iter().map(|m| m[i]).sum::<f64>() / draws as f64)
            .collect();
        let measured_bias: f64 = mean.iter().zip(&x).map(|(m, t)| (m - t).powi(2)).sum();
        let measured_var: f64 = (0..12)
            .map(|i| {
                maps.iter().map(|m| (m[i] - mean[i]).powi(2)).sum::<f64>() / (draws - 1) as f64
            })
            .sum();
        let (bias2, variance) = bias_variance(&g, &x, sigma, gamma);
        assert!(
            (measured_var - variance).abs() < 0.12 * variance,
            "variance {measured_var} vs {variance}"
        );
        // The measured bias also carries the mean's own noise (variance / draws).
        assert!(
            (measured_bias - bias2).abs() < 0.1 * bias2 + 3.0 * variance / draws as f64,
            "bias² {measured_bias} vs {bias2}"
        );
    }
}
