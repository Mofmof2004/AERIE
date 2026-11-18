//! Random walk with restart (Mission 1): like label propagation, but nothing is clamped.
//! Every round each drone listens to its neighbours (weight α) and goes back to its own
//! report (weight 1 − α), so a wrong report can be outvoted.

use faer::Mat;
use faer::linalg::solvers::Solve;

/// Random walk with restart: F <- αPF + (1−α)Y₀, until nothing moves.
///
/// - `p`: the transition matrix P = D⁻¹A (n x n, `EdgeList::transition_matrix`).
/// - `labels`: `Some(class)` for drones that reported, `None` for silent ones.
/// - `alpha`: in [0, 1). How much a drone listens to its neighbours each round; 1 − α is
///   how much it is pulled back to its own report (nothing for a silent drone).
///
/// Y₀ is the one-hot matrix of the reports (zeros for silent drones) and F starts at Y₀.
/// It stops when no cell moves by more than `tol`, or after `max_iter` rounds. Returns
/// the final F (n x n_classes) and the number of rounds.
///
/// Walker view: start at drone i; each step, stop with chance 1 − α, else move to a
/// neighbour picked by P. F[i][c] is the chance of stopping on a drone that reported c.
/// So, unlike label propagation:
/// - a reporter's row is not fixed: it drifts toward its neighbours;
/// - rows do not sum to 1 (the walker can stop on a silent drone): the guess is still
///   the largest column;
/// - a lie loses only when honest reporters around it outweigh its own restart term. In
///   a star, the centre's own report always wins (see the tests).
///
/// A silent drone with no path to any report keeps a row of zeros (no guess).
pub fn random_walk_with_restart(
    p: &Mat<f64>,
    labels: &[Option<usize>],
    n_classes: usize,
    alpha: f64,
    tol: f64,
    max_iter: usize,
) -> (Mat<f64>, usize) {
    let n = p.nrows();
    let mut y_0: Mat<f64> = Mat::zeros(n, n_classes);

    for i in 0..labels.len() {
        if let Some(c) = labels[i] {
            y_0[(i, c)] = 1.0;
        }
    }

    let mut y = y_0.clone();
    let mut y_old = y_0.clone();
    let mut iteration_count: usize = 0;
    for _ in 0..max_iter {
        iteration_count += 1;
        // Converged unless a cell below moves by more than tol.
        let mut break_tf = false;

        y = alpha * p * &y_old + ((1.0 - alpha) * &y_0);

        'y_iterator: for row in 0..labels.len() {
            for col in 0..n_classes {
                if (y[(row, col)] - y_old[(row, col)]).abs() > tol {
                    break_tf = true;
                    break 'y_iterator;
                }
            }
        }

        y_old = y.clone();

        if !break_tf {
            break;
        }
    }

    (y, iteration_count)
}

/// Random walk with restart in one linear solve: the F that `random_walk_with_restart`
/// converges to, F(∞).
///
/// At convergence F = αPF + (1−α)Y₀, so (I − αP) F = (1−α) Y₀: one n x n system
/// M F = B with M = I − αP and B = (1−α)Y₀, solved with an LU factorization. Unrolled,
/// F = (1−α)(Y₀ + αPY₀ + α²P²Y₀ + ...): term k is the reports that travel k hops,
/// weighted by αᵏ.
///
/// Unlike `label_propagation_closed_form`, there are no special rows (nothing is
/// clamped) and no BFS: for α < 1 every row of αP sums to α < 1, so the diagonal of M
/// outweighs the rest of its row and M is never singular. A drone that no report can
/// reach has a zero row in B, so it gets a zero row in F.
///
/// `alpha` must be in [0, 1): at α = 1, M = I − P has rows summing to 0 and is singular.
pub fn random_walk_with_restart_closed_form(
    p: &Mat<f64>,
    labels: &[Option<usize>],
    n_classes: usize,
    alpha: f64,
) -> Mat<f64> {
    let n = p.nrows();
    let mut index_matrix: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        index_matrix[(i, i)] = 1.0;
    }

    let mut y: Mat<f64> = Mat::zeros(n, n_classes);

    for i in 0..labels.len() {
        if let Some(c) = labels[i] {
            y[(i, c)] = 1.0;
        }
    }
    let m = &index_matrix - alpha * p; // M = I − αP
    let b = (1.0 - alpha) * &y; // B = (1−α)Y₀

    m.partial_piv_lu().solve(&b) // F such that M · F = B
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::label_propagation;
    use aerie_graph::{Edge, EdgeList};

    /// P of a graph given as (src, dst, weight) links.
    fn p_of(n: u32, links: &[(u32, u32, f64)]) -> Mat<f64> {
        let edges = links
            .iter()
            .map(|&(src, dst, weight)| Edge { src, dst, weight })
            .collect();
        EdgeList { n, edges }.transition_matrix()
    }

    /// The path 0 - 1 - ... - (n-1), weight 1.
    fn path(n: u32) -> Mat<f64> {
        let links: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        p_of(n, &links)
    }

    fn row(y: &Mat<f64>, i: usize) -> Vec<f64> {
        (0..y.ncols()).map(|c| y[(i, c)]).collect()
    }

    fn close(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn path_of_three_by_hand() {
        // F0 = ½F1 + ½[1,0], F2 = ½F1 + ½[0,1], F1 = ½(½F0 + ½F2)
        // => F1 = [1/6, 1/6], F0 = [7/12, 1/12], F2 = [1/12, 7/12].
        let (f, _) =
            random_walk_with_restart(&path(3), &[Some(0), None, Some(1)], 2, 0.5, 1e-12, 1000);
        assert!(close(&row(&f, 0), &[7.0 / 12.0, 1.0 / 12.0], 1e-9));
        assert!(close(&row(&f, 1), &[1.0 / 6.0, 1.0 / 6.0], 1e-9));
        assert!(close(&row(&f, 2), &[1.0 / 12.0, 7.0 / 12.0], 1e-9));
    }

    #[test]
    fn reporters_are_not_clamped_and_rows_sum_to_at_most_one() {
        let labels = [Some(0), None, None, Some(1), None];
        let (f, _) = random_walk_with_restart(&path(5), &labels, 2, 0.85, 1e-10, 10_000);
        assert!(
            f[(0, 0)] < 1.0,
            "the reporter drifted toward its neighbours"
        );
        for i in 0..5 {
            let s: f64 = row(&f, i).iter().sum();
            assert!((0.0..=1.0 + 1e-12).contains(&s), "row {i} sums to {s}");
        }
        // The walker from drone 4 must pass drone 3 (a report) before anything else.
        assert!(f[(4, 1)] > f[(4, 0)]);
    }

    #[test]
    fn alpha_zero_keeps_the_reports() {
        // α = 0: nobody listens to anybody, F = Y₀ after one round.
        let labels = [Some(1), None, Some(0)];
        let (f, rounds) = random_walk_with_restart(&path(3), &labels, 2, 0.0, 1e-9, 100);
        assert_eq!(rounds, 1);
        assert_eq!(row(&f, 0), [0.0, 1.0]);
        assert_eq!(row(&f, 1), [0.0, 0.0]);
        assert_eq!(row(&f, 2), [1.0, 0.0]);
    }

    #[test]
    fn a_star_cannot_outvote_its_centre() {
        // Centre 0 says 1, the three leaves say 0. Leaves only listen to the centre, so
        // they echo it back: by symmetry F0 = [α/(1+α), 1/(1+α)] for every α, the
        // centre's lie always wins. At α = 0.5: F0 = [1/3, 2/3], leaves [2/3, 1/3].
        let p = p_of(4, &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
        let labels = [Some(1), Some(0), Some(0), Some(0)];
        for alpha in [0.5, 0.9, 0.99] {
            let (f, _) = random_walk_with_restart(&p, &labels, 2, alpha, 1e-13, 100_000);
            let want = [alpha / (1.0 + alpha), 1.0 / (1.0 + alpha)];
            assert!(
                close(&row(&f, 0), &want, 1e-9),
                "α {alpha}: {:?}",
                row(&f, 0)
            );
        }
        let (f, _) = random_walk_with_restart(&p, &labels, 2, 0.5, 1e-13, 1000);
        assert!(close(&row(&f, 1), &[2.0 / 3.0, 1.0 / 3.0], 1e-9));
    }

    #[test]
    fn a_lie_between_honest_neighbours_is_outvoted() {
        // Drone 2 says 1; it sits in a clique of five drones whose other four say 0.
        // Label propagation keeps the lie (clamped), RWR overrules it.
        let mut links = Vec::new();
        for a in 0..5 {
            for b in a + 1..5 {
                links.push((a, b, 1.0));
            }
        }
        let p = p_of(5, &links);
        let labels = [Some(0), Some(0), Some(1), Some(0), Some(0)];
        let (f, _) = random_walk_with_restart(&p, &labels, 2, 0.85, 1e-12, 10_000);
        assert!(f[(2, 0)] > f[(2, 1)], "RWR: {:?}", row(&f, 2));
        let (y, _) = label_propagation(&p, &labels, 2, 1e-12, 10_000);
        assert_eq!(row(&y, 2), [0.0, 1.0], "LP: the lie stays");
    }

    #[test]
    fn a_silent_drone_with_no_path_to_a_report_gets_no_guess() {
        let p = p_of(4, &[(0, 1, 1.0)]);
        let (f, _) =
            random_walk_with_restart(&p, &[Some(1), None, None, Some(0)], 2, 0.85, 1e-9, 1000);
        assert_eq!(row(&f, 2), [0.0, 0.0]);
        // Drone 3 is isolated with a self-loop: it only hears itself, F3 = [1, 0].
        assert!(close(&row(&f, 3), &[1.0, 0.0], 1e-6));
    }

    /// Largest difference between two matrices of the same size.
    fn max_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
        let mut d = 0.0_f64;
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                d = d.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        d
    }

    #[test]
    fn closed_form_path_of_three_by_hand() {
        // Same numbers as the iteration, exact this time.
        let f = random_walk_with_restart_closed_form(&path(3), &[Some(0), None, Some(1)], 2, 0.5);
        assert!(close(&row(&f, 0), &[7.0 / 12.0, 1.0 / 12.0], 1e-12));
        assert!(close(&row(&f, 1), &[1.0 / 6.0, 1.0 / 6.0], 1e-12));
        assert!(close(&row(&f, 2), &[1.0 / 12.0, 7.0 / 12.0], 1e-12));
    }

    #[test]
    fn closed_form_star_centre_is_exact() {
        // The star from `a_star_cannot_outvote_its_centre`: F0 = [α/(1+α), 1/(1+α)].
        let p = p_of(4, &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
        let labels = [Some(1), Some(0), Some(0), Some(0)];
        for alpha in [0.0, 0.3, 0.5, 0.9, 0.999] {
            let f = random_walk_with_restart_closed_form(&p, &labels, 2, alpha);
            let want = [alpha / (1.0 + alpha), 1.0 / (1.0 + alpha)];
            assert!(
                close(&row(&f, 0), &want, 1e-9),
                "α {alpha}: {:?}",
                row(&f, 0)
            );
        }
    }

    #[test]
    fn closed_form_matches_the_iteration() {
        // Weighted graphs, several α: the iteration with a tight tolerance and the solve
        // agree to 1e-6. The last graph has an isolated silent drone (5) and a cut-off
        // silent pair (6, 7): no BFS needed, they come out as zeros in both.
        let cases: Vec<(Mat<f64>, Vec<Option<usize>>, usize)> = vec![
            (
                p_of(3, &[(0, 1, 3.0), (1, 2, 1.0)]),
                vec![Some(0), None, Some(1)],
                2,
            ),
            (
                path(6),
                vec![Some(0), None, Some(2), None, None, Some(1)],
                3,
            ),
            (
                p_of(
                    8,
                    &[
                        (0, 1, 1.0),
                        (1, 2, 0.4),
                        (0, 2, 2.0),
                        (2, 3, 0.7),
                        (3, 4, 1.5),
                        (1, 4, 0.9),
                        (6, 7, 1.0),
                    ],
                ),
                vec![Some(1), None, Some(0), None, Some(1), None, None, None],
                2,
            ),
        ];
        for (k, (p, labels, c)) in cases.iter().enumerate() {
            for alpha in [0.1, 0.5, 0.85, 0.95] {
                let (iterated, _) = random_walk_with_restart(p, labels, *c, alpha, 1e-13, 100_000);
                let solved = random_walk_with_restart_closed_form(p, labels, *c, alpha);
                let d = max_diff(&iterated, &solved);
                assert!(d < 1e-6, "case {k}, α {alpha}: differ by {d}");
            }
        }
        let (p, labels, c) = &cases[2];
        let f = random_walk_with_restart_closed_form(p, labels, *c, 0.85);
        for i in [5, 6, 7] {
            assert_eq!(row(&f, i), [0.0, 0.0], "drone {i}");
        }
    }

    #[test]
    fn closed_form_alpha_zero_is_the_reports() {
        let labels = [Some(1), None, Some(0)];
        let f = random_walk_with_restart_closed_form(&path(3), &labels, 2, 0.0);
        assert_eq!(row(&f, 0), [0.0, 1.0]);
        assert_eq!(row(&f, 1), [0.0, 0.0]);
        assert_eq!(row(&f, 2), [1.0, 0.0]);
    }

    #[test]
    fn max_iter_caps_the_rounds() {
        let labels = [Some(0), None, None, None, None, None, Some(1)];
        let (_, rounds) = random_walk_with_restart(&path(7), &labels, 2, 0.99, 1e-15, 7);
        assert_eq!(rounds, 7);
    }
}
