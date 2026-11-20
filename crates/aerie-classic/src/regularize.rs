//! Mission 3: rebuild a signal on a graph from noisy, partial readings.
//!
//! - `tikhonov`: min ‖M(x − y)‖² + γ xᵀLx, one linear solve. Smooth maps; blurs edges.
//! - `trend_filtering` (order 1): min ½‖M(x − y)‖² + λ ‖Lx‖₁, by ADMM. Few drones bend:
//!   straight pieces joined at kinks.
//! - `total_variation` (trend filtering of order 0): λ ‖Bᵀx‖₁ instead, by ADMM. Few links
//!   jump: flat pieces, sharp edges.
//! - `sparse_local_variation_run`: λ Σ_i ‖∇_i x‖₂ instead, by ADMM: few drones see change.
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

/// Each drone's local variation ‖∇_i x‖₂ = √(Σ over its links w (x_i − x_j)²): how much
/// the signal changes around it, as one number.
pub fn local_variation(graph: &EdgeList, x: &[f64]) -> Vec<f64> {
    let mut sq = vec![0.0; graph.n as usize];
    for e in &graph.edges {
        let (a, b) = (e.src as usize, e.dst as usize);
        let c = e.weight * (x[a] - x[b]).powi(2);
        sq[a] += c;
        sq[b] += c;
    }
    sq.into_iter().map(f64::sqrt).collect()
}

/// Sparse local variation Σ_i ‖∇_i x‖₂: the local variations added up, not squared.
/// Small when few drones see any change around them.
pub fn slv_penalty(graph: &EdgeList, x: &[f64]) -> f64 {
    local_variation(graph, x).iter().sum()
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

/// ‖Lx‖₁ = Σ_i |Σ_j w_ij (x_i − x_j)|: how much the slope bends, drone by drone.
pub fn tf_penalty(graph: &EdgeList, x: &[f64]) -> f64 {
    let l = graph.laplacian();
    let n = x.len();
    (0..n)
        .map(|i| (0..n).map(|j| l[(i, j)] * x[j]).sum::<f64>().abs())
        .sum()
}

/// A link seen from one end: (the drone, its neighbour, √w).
type HalfLink = (usize, usize, f64);

/// Every link seen from each of its ends: (the drone, its neighbour, √w), grouped by drone.
/// Drone i's local gradient ∇_i x is the list √w (x_neighbour − x_i) over its half-links.
fn half_links(graph: &EdgeList) -> (Vec<HalfLink>, Vec<std::ops::Range<usize>>) {
    let n = graph.n as usize;
    let mut half: Vec<HalfLink> = Vec::with_capacity(2 * graph.edges.len());
    for e in &graph.edges {
        let (a, b, s) = (e.src as usize, e.dst as usize, e.weight.sqrt());
        half.push((a, b, s));
        half.push((b, a, s));
    }
    half.sort_by_key(|h| h.0);
    let mut groups = vec![0..0; n];
    let mut start = 0;
    for (i, g) in groups.iter_mut().enumerate() {
        let mut end = start;
        while end < half.len() && half[end].0 == i {
            end += 1;
        }
        *g = start..end;
        start = end;
    }
    (half, groups)
}

/// Sparse local variation by ADMM: min ½‖M(x − y)‖² + λ Σ_i ‖∇_i x‖₂.
///
/// Like trend filtering, but the differences are shrunk drone by drone instead of link by
/// link: a drone's whole local gradient is set to 0 when it is small (block soft
/// threshold), so most drones end up with a flat neighbourhood.
///
/// ```text
/// x ← solve (MᵀM + 2ρ L) x = MᵀM y + ρ Dᵀ(z − u)     D x lists every ∇_i x (DᵀD = 2L)
/// z_i ← max(0, 1 − (λ/ρ) / |v_i|) · v_i, v_i = ∇_i x + u_i   drone by drone
/// u ← u + D x − z
/// ```
pub fn sparse_local_variation_run(
    graph: &EdgeList,
    y: &[f64],
    has_sensor: &[bool],
    lambda: f64,
    rho: f64,
    iterations: usize,
) -> Reconstruction {
    let n = graph.n as usize;
    let (half, groups) = half_links(graph);
    let h = half.len();
    let solver = mask_plus_laplacian(graph, has_sensor, 2.0 * rho).partial_piv_lu();
    let readings = masked_readings(y, has_sensor);
    // D x: every drone's local gradient, half-link by half-link.
    let d = |x: &[f64]| -> Vec<f64> { half.iter().map(|&(a, b, s)| s * (x[b] - x[a])).collect() };

    let mut x = vec![0.0; n];
    let mut z = vec![0.0; h];
    let mut u = vec![0.0; h];
    let mut gaps = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // 1. x: Dᵀ(z − u) puts each half-link's value back on its two drones.
        let mut pull = vec![0.0; n];
        for (k, &(a, b, s)) in half.iter().enumerate() {
            let v = s * (z[k] - u[k]);
            pull[a] -= v;
            pull[b] += v;
        }
        let rhs: Vec<f64> = readings
            .iter()
            .zip(&pull)
            .map(|(r, p)| r + rho * p)
            .collect();
        x = solve_with(&solver, &rhs);

        // 2. z: each drone's gradient (plus the tally), shrunk as a whole.
        let dx = d(&x);
        for range in &groups {
            let norm = range
                .clone()
                .map(|k| (dx[k] + u[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            let keep = if norm > 0.0 {
                (1.0 - (lambda / rho) / norm).max(0.0)
            } else {
                0.0
            };
            for k in range.clone() {
                z[k] = keep * (dx[k] + u[k]);
            }
        }

        // 3. u.
        let mut gap = 0.0;
        for ((uk, &dk), &zk) in u.iter_mut().zip(&dx).zip(&z) {
            *uk += dk - zk;
            gap += (dk - zk).powi(2);
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
    /// ½‖M(x − y)‖² + λ Σ_i ‖∇_i x‖₂: most drones see no change around them.
    SparseLocalVariation { lambda: f64 },
}

impl Regularizer {
    pub fn name(self) -> &'static str {
        match self {
            Regularizer::Tikhonov { .. } => "Tikhonov",
            Regularizer::TrendFiltering { .. } => "trend filtering",
            Regularizer::TotalVariation { .. } => "total variation",
            Regularizer::SparseLocalVariation { .. } => "sparse local variation",
        }
    }

    /// Solved by ADMM (rounds and a convergence curve) rather than one linear solve.
    pub fn is_iterative(self) -> bool {
        matches!(
            self,
            Regularizer::TrendFiltering { .. }
                | Regularizer::TotalVariation { .. }
                | Regularizer::SparseLocalVariation { .. }
        )
    }

    /// The penalty term at x (without the fit).
    pub fn penalty(self, graph: &EdgeList, x: &[f64]) -> f64 {
        match self {
            Regularizer::Tikhonov { gamma } => gamma * graph.variation(x),
            Regularizer::TrendFiltering { lambda } => lambda * tf_penalty(graph, x),
            Regularizer::TotalVariation { lambda } => lambda * tv_penalty(graph, x),
            Regularizer::SparseLocalVariation { lambda } => lambda * slv_penalty(graph, x),
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
        Regularizer::SparseLocalVariation { lambda } => {
            sparse_local_variation_run(graph, y, has_sensor, lambda, 1.0, iterations)
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
        // Local variations: drone 0 √(1 + 9), drone 1 √(1 + 16), drone 2 √(16 + 9).
        let lv = local_variation(&g, &x);
        for (got, want) in lv.iter().zip([10f64.sqrt(), 17f64.sqrt(), 5.0]) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!((slv_penalty(&g, &x) - (10f64.sqrt() + 17f64.sqrt() + 5.0)).abs() < 1e-12);
    }

    /// ½‖M(x − y)‖² + λ Σ_i ‖∇_i x‖₂.
    fn slv_objective(g: &EdgeList, x: &[f64], y: &[f64], has: &[bool], lambda: f64) -> f64 {
        let fit: f64 = (0..x.len())
            .filter(|&i| has[i])
            .map(|i| 0.5 * (x[i] - y[i]).powi(2))
            .sum();
        fit + lambda * slv_penalty(g, x)
    }

    #[test]
    fn sparse_local_variation_converges_to_a_minimum() {
        let g = path(16);
        let y: Vec<f64> = step(16)
            .iter()
            .enumerate()
            .map(|(i, v)| v + 0.08 * ((i * 3 % 5) as f64 - 2.0))
            .collect();
        let has = vec![true; 16];
        let run = sparse_local_variation_run(&g, &y, &has, 0.2, 1.0, 4000);
        assert!(
            *run.gaps.last().unwrap() < 1e-7,
            "gap {}",
            run.gaps.last().unwrap()
        );
        let best = slv_objective(&g, &run.x, &y, &has, 0.2);
        for i in 0..16 {
            for d in [-1e-3, 1e-3] {
                let mut x = run.x.clone();
                x[i] += d;
                assert!(
                    slv_objective(&g, &x, &y, &has, 0.2) >= best - 1e-9,
                    "drone {i}"
                );
            }
        }
    }

    #[test]
    fn sparse_local_variation_leaves_most_drones_with_a_flat_neighbourhood() {
        // Clean step on a path of 30.
        let g = path(30);
        let y = step(30);
        let has = vec![true; 30];
        let x = sparse_local_variation_run(&g, &y, &has, 0.3, 1.0, 3000).x;
        // The change stays next to the edge (drones 14 and 15) and dies off within a few
        // drones: each link is in the neighbourhood of both its drones, so the drones right
        // beside the edge can't be made perfectly flat, but those 4 or more away are.
        let lv = local_variation(&g, &x);
        assert!(lv[14] > 0.5 && lv[15] > 0.5, "the edge survives: {lv:?}");
        for (i, v) in lv.iter().enumerate() {
            if !(11..=18).contains(&i) {
                assert!(*v < 1e-3, "drone {i} still sees {v}");
            }
        }
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
            Regularizer::SparseLocalVariation { lambda: 0.2 },
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
}
