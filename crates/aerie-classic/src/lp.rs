//! Label propagation (Mission 1): a few drones report their label, the rest get theirs
//! by repeatedly averaging their neighbours, Y <- P Y, with the known rows clamped.

use faer::Mat;
use faer::linalg::solvers::Solve;
use std::collections::VecDeque;

/// Label propagation: spreads the known labels over the graph until nothing moves.
///
/// - `p`: the transition matrix P = D⁻¹A (n x n, `EdgeList::transition_matrix`).
/// - `labels`: one slot per drone, `Some(class)` for drones that reported, `None` for
///   silent ones. Classes are `0..n_classes`.
///
/// Y starts with a one-hot row for every labelled drone and zeros elsewhere. Each round
/// computes Y <- P' Y, where P' is P with every labelled row replaced by an identity row
/// (1 on the diagonal): a labelled drone only listens to itself, so its row is clamped.
/// It stops when no cell moves by more than `tol`, or after `max_iter` rounds.
///
/// Returns the final Y (n x n_classes) and the number of rounds. A silent drone's guess
/// is the column with the largest value in its row; a silent drone with no path to any
/// labelled drone keeps a row of zeros (no guess).
pub fn label_propagation(
    p: &Mat<f64>,             // transition matrix, n × n
    labels: &[Option<usize>], // Some(class) for reporting drones, None for silent ones
    n_classes: usize,
    tol: f64,        // plan: 1e-6
    max_iter: usize, // plan: 1000
) -> (Mat<f64>, usize) {
    let n = p.nrows();
    let mut y_old: Mat<f64> = Mat::zeros(n, n_classes);
    let mut y: Mat<f64> = Mat::zeros(n, n_classes);
    let mut new_p: Mat<f64> = Mat::zeros(n, n);

    for i in 0..labels.len() {
        if let Some(c) = labels[i] {
            y_old[(i, c)] = 1.0;
        }
    }

    for row in 0..labels.len() {
        for col in 0..labels.len() {
            // If both labelled
            if labels[row].is_some() && labels[col].is_some() {
                if row == col {
                    new_p[(row, col)] = 1.0;
                }
            } else if labels[row].is_some() && labels[col].is_none() {
                new_p[(row, col)] = 0.0;
            } else {
                new_p[(row, col)] = p[(row, col)];
            }
        }
    }

    let mut iteration_count: usize = 0;
    for _ in 0..max_iter {
        iteration_count += 1;
        // Converged unless a cell below moves by more than tol.
        let mut break_tf = false;

        y = &new_p * &y_old;

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

/// Label propagation in one linear solve: the answer `label_propagation` converges to.
///
/// At convergence the silent rows satisfy Y_U = P_UU Y_U + P_UL Y_L, so
/// (I - P_UU) Y_U = P_UL Y_L. Instead of cutting out the blocks, this keeps all n rows
/// in their drone order and solves one n x n system M Y = B:
/// - labelled drone i: M row = identity row, B row = its one-hot (so Y_i is its label);
/// - silent drone i: M[i][j] = delta_ij - P[i][j] over silent j (0 over labelled j),
///   B row = sum over labelled j of P[i][j] times j's one-hot.
///
/// Solved with an LU factorization (`partial_piv_lu().solve`), never forming an inverse.
///
/// A silent drone with no path to any labelled drone (isolated, or a silent squad cut off
/// from every report) would make M singular. A BFS from all labelled drones finds them
/// first; they keep an identity row in M and a zero row in B, so they get a row of zeros,
/// like the iteration.
pub fn label_propagation_closed_form(
    p: &Mat<f64>,
    labels: &[Option<usize>],
    n_classes: usize,
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

    // Which drones can reach a label: a BFS started from every labelled drone at once.
    let mut reached = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    for i in 0..n {
        if labels[i].is_some() {
            reached[i] = true;
            queue.push_back(i);
        }
    }

    while let Some(i) = queue.pop_front() {
        for j in 0..n {
            if p[(i, j)] != 0.0 && !reached[j] {
                reached[j] = true;
                queue.push_back(j);
            }
        }
    }

    let mut new_y: Mat<f64> = y.clone();
    for i in 0..n {
        // Unreachable silent drones skip this: identity row in M, zero row in B, so Y_i = 0.
        if labels[i].is_none() && reached[i] {
            for j in 0..n {
                if labels[j].is_none() {
                    index_matrix[(i, j)] -= p[(i, j)];
                } else {
                    for class in 0..n_classes {
                        new_y[(i, class)] += p[(i, j)] * y[(j, class)];
                    }
                }
            }
        }
    }

    index_matrix.partial_piv_lu().solve(&new_y) // gives Y such that M · Y = B
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn prediction_path_of_three() {
        // 0 is class 0, 2 is class 1, 1 is silent. Round 1: row 1 = half of each
        // neighbour = [0.5, 0.5]. Round 2: nothing moves, so it stops after 2 rounds.
        let (y, rounds) = label_propagation(&path(3), &[Some(0), None, Some(1)], 2, 1e-6, 1000);
        assert!(close(&row(&y, 1), &[0.5, 0.5], 1e-12));
        assert_eq!(rounds, 2);
    }

    #[test]
    fn labelled_rows_stay_clamped() {
        let labels = [Some(0), None, None, None, Some(1)];
        let (y, _) = label_propagation(&path(5), &labels, 2, 1e-6, 1000);
        assert_eq!(row(&y, 0), [1.0, 0.0]);
        assert_eq!(row(&y, 4), [0.0, 1.0]);
    }

    #[test]
    fn path_of_five_is_a_straight_line_between_the_ends() {
        // Each silent drone ends up as the average of its two neighbours, so the
        // share of class 0 falls in equal steps: 1, 0.75, 0.5, 0.25, 0.
        let labels = [Some(0), None, None, None, Some(1)];
        let (y, rounds) = label_propagation(&path(5), &labels, 2, 1e-9, 1000);
        assert!(close(&row(&y, 1), &[0.75, 0.25], 1e-6));
        assert!(close(&row(&y, 2), &[0.5, 0.5], 1e-6));
        assert!(close(&row(&y, 3), &[0.25, 0.75], 1e-6));
        // It takes many small steps, but stops well before the cap.
        assert!(rounds > 5 && rounds < 1000, "{rounds} rounds");
    }

    #[test]
    fn heavier_links_pull_harder() {
        // Drone 1 links to 0 (class 0) with weight 3 and to 2 (class 1) with weight 1:
        // P[1] = [0.75, 0, 0.25], so row 1 of Y = [0.75, 0.25].
        let p = p_of(3, &[(0, 1, 3.0), (1, 2, 1.0)]);
        let (y, _) = label_propagation(&p, &[Some(0), None, Some(1)], 2, 1e-9, 1000);
        assert!(close(&row(&y, 1), &[0.75, 0.25], 1e-12));
    }

    #[test]
    fn rows_of_reached_drones_sum_to_one() {
        // Three classes on a path of 6; every silent drone can reach a label.
        let labels = [Some(0), None, Some(2), None, None, Some(1)];
        let (y, _) = label_propagation(&path(6), &labels, 3, 1e-9, 1000);
        for i in 0..6 {
            let s: f64 = row(&y, i).iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {i} sums to {s}");
        }
    }

    #[test]
    fn a_silent_drone_with_no_path_to_a_label_gets_no_guess() {
        // 0 - 1 linked (0 labelled); 2 isolated and silent: it keeps zeros. 3 isolated
        // but labelled: it keeps its one-hot (the self-loop in P).
        let p = p_of(4, &[(0, 1, 1.0)]);
        let labels = [Some(1), None, None, Some(0)];
        let (y, _) = label_propagation(&p, &labels, 2, 1e-9, 1000);
        assert!(close(&row(&y, 1), &[0.0, 1.0], 1e-12));
        assert_eq!(row(&y, 2), [0.0, 0.0]);
        assert_eq!(row(&y, 3), [1.0, 0.0]);
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
    fn closed_form_path_of_three() {
        // One solve gives what the iteration reaches after 2 rounds.
        let y = label_propagation_closed_form(&path(3), &[Some(0), None, Some(1)], 2);
        assert!(close(&row(&y, 0), &[1.0, 0.0], 1e-12));
        assert!(close(&row(&y, 1), &[0.5, 0.5], 1e-12));
        assert!(close(&row(&y, 2), &[0.0, 1.0], 1e-12));
    }

    #[test]
    fn closed_form_path_of_five_is_exact() {
        // The iteration only gets close to 0.75 / 0.5 / 0.25; the solve hits it.
        let labels = [Some(0), None, None, None, Some(1)];
        let y = label_propagation_closed_form(&path(5), &labels, 2);
        assert!(close(&row(&y, 1), &[0.75, 0.25], 1e-12));
        assert!(close(&row(&y, 2), &[0.5, 0.5], 1e-12));
        assert!(close(&row(&y, 3), &[0.25, 0.75], 1e-12));
    }

    #[test]
    fn closed_form_matches_the_iteration() {
        // Weighted graphs where every silent drone can reach a label: the iteration with
        // a tight tolerance and the solve agree to 1e-6.
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
                // A triangle with a tail and a chord, uneven weights, labels at both ends.
                p_of(
                    6,
                    &[
                        (0, 1, 1.0),
                        (1, 2, 0.4),
                        (0, 2, 2.0),
                        (2, 3, 0.7),
                        (3, 4, 1.5),
                        (4, 5, 0.2),
                        (1, 4, 0.9),
                    ],
                ),
                vec![Some(1), None, None, None, None, Some(0)],
                2,
            ),
        ];
        for (k, (p, labels, c)) in cases.iter().enumerate() {
            let (iterated, _) = label_propagation(p, labels, *c, 1e-12, 100_000);
            let solved = label_propagation_closed_form(p, labels, *c);
            let d = max_diff(&iterated, &solved);
            assert!(d < 1e-6, "case {k}: differ by {d}");
        }
    }

    #[test]
    fn closed_form_keeps_the_labelled_rows() {
        let labels = [Some(0), None, Some(1), None, Some(0)];
        let y = label_propagation_closed_form(&path(5), &labels, 2);
        assert_eq!(row(&y, 0), [1.0, 0.0]);
        assert_eq!(row(&y, 2), [0.0, 1.0]);
        assert_eq!(row(&y, 4), [1.0, 0.0]);
    }

    #[test]
    fn closed_form_gives_zeros_where_no_label_can_reach() {
        // Same answer as the iteration: drone 2 is isolated and silent, so its row is
        // zero. Today the solve returns inf or NaN there.
        let p = p_of(4, &[(0, 1, 1.0)]);
        let labels = [Some(1), None, None, Some(0)];
        let y = label_propagation_closed_form(&p, &labels, 2);
        assert_eq!(row(&y, 2), [0.0, 0.0]);
        assert!(close(&row(&y, 1), &[0.0, 1.0], 1e-12));
    }

    #[test]
    fn max_iter_caps_the_rounds() {
        let labels = [Some(0), None, None, None, Some(1)];
        let (y, rounds) = label_propagation(&path(5), &labels, 2, 1e-12, 1);
        assert_eq!(rounds, 1);
        // After one round only the neighbours of the labels have moved.
        assert!(close(&row(&y, 1), &[0.5, 0.0], 1e-12));
        assert_eq!(row(&y, 2), [0.0, 0.0]);
    }
}
