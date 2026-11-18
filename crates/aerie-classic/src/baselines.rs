//! Baselines for Mission 1: what label propagation and RWR have to beat.
//!
//! Both return the same shape as the methods (n x n_classes, one-hot rows, zeros = no
//! guess), so the harness scores everything the same way. Reporting drones keep their
//! own report, as in label propagation.

use std::cmp::Reverse;
use std::collections::VecDeque;

use faer::Mat;

/// Every silent drone guesses the class reported most often (ties: smallest class).
/// Ignores the graph: a method that cannot beat this is not using the links.
/// With no reports at all, every silent row is zero.
pub fn majority_class(labels: &[Option<usize>], n_classes: usize) -> Mat<f64> {
    let mut counts = vec![0usize; n_classes];
    for &c in labels.iter().flatten() {
        counts[c] += 1;
    }
    let winner = (0..n_classes)
        .max_by_key(|&c| (counts[c], Reverse(c)))
        .filter(|&c| counts[c] > 0);

    let guesses: Vec<Option<usize>> = labels.iter().map(|&l| l.or(winner)).collect();
    one_hot_rows(&guesses, n_classes)
}

/// Every silent drone takes the class of the reporter the fewest hops away on the graph
/// (links = nonzero entries of `p`; weights are ignored). One BFS from all reporters at
/// once: each drone inherits the class of whoever reaches it first. Ties at the same
/// hop count go to the reporter with the smallest index (it enters the queue first).
/// A drone no reporter can reach gets a zero row.
pub fn nearest_reporter(p: &Mat<f64>, labels: &[Option<usize>], n_classes: usize) -> Mat<f64> {
    let n = p.nrows();
    assert_eq!(labels.len(), n, "one label slot per drone");

    let mut class = labels.to_vec();
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| labels[i].is_some()).collect();
    while let Some(i) = queue.pop_front() {
        for j in 0..n {
            if p[(i, j)] != 0.0 && class[j].is_none() {
                class[j] = class[i];
                queue.push_back(j);
            }
        }
    }
    one_hot_rows(&class, n_classes)
}

/// Every silent drone takes the class of the closest reporter in km, ignoring the graph
/// (the plan's nearest-neighbour baseline). `positions[i]` is drone i's (x, y). Ties go
/// to the reporter with the smallest index. With no reporters, every silent row is zero.
pub fn nearest_reporter_by_distance(
    positions: &[[f64; 2]],
    labels: &[Option<usize>],
    n_classes: usize,
) -> Mat<f64> {
    assert_eq!(labels.len(), positions.len(), "one label slot per drone");
    let reporters: Vec<(usize, usize)> = labels
        .iter()
        .enumerate()
        .filter_map(|(j, l)| l.map(|c| (j, c)))
        .collect();
    let dist2 = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);

    let guesses: Vec<Option<usize>> = labels
        .iter()
        .zip(positions)
        .map(|(&own, &at)| {
            own.or_else(|| {
                reporters
                    .iter()
                    .min_by(|a, b| dist2(at, positions[a.0]).total_cmp(&dist2(at, positions[b.0])))
                    .map(|&(_, c)| c)
            })
        })
        .collect();
    one_hot_rows(&guesses, n_classes)
}

/// Row i is the one-hot of `guesses[i]`, or all zeros for `None`.
fn one_hot_rows(guesses: &[Option<usize>], n_classes: usize) -> Mat<f64> {
    Mat::from_fn(guesses.len(), n_classes, |i, c| {
        if guesses[i] == Some(c) { 1.0 } else { 0.0 }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::label_propagation;
    use aerie_graph::{Edge, EdgeList};

    fn p_of(n: u32, links: &[(u32, u32, f64)]) -> Mat<f64> {
        let edges = links
            .iter()
            .map(|&(src, dst, weight)| Edge { src, dst, weight })
            .collect();
        EdgeList { n, edges }.transition_matrix()
    }

    fn path(n: u32) -> Mat<f64> {
        let links: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
        p_of(n, &links)
    }

    fn row(y: &Mat<f64>, i: usize) -> Vec<f64> {
        (0..y.ncols()).map(|c| y[(i, c)]).collect()
    }

    #[test]
    fn majority_fills_the_silent_drones_and_keeps_the_reports() {
        let y = majority_class(&[Some(1), None, Some(0), Some(1), None], 2);
        assert_eq!(row(&y, 0), [0.0, 1.0]);
        assert_eq!(row(&y, 1), [0.0, 1.0]);
        assert_eq!(row(&y, 2), [1.0, 0.0], "a reporter keeps its own class");
        assert_eq!(row(&y, 4), [0.0, 1.0]);
    }

    #[test]
    fn majority_tie_goes_to_the_smallest_class() {
        let y = majority_class(&[Some(2), Some(1), None], 3);
        assert_eq!(row(&y, 2), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn majority_without_reports_has_no_guess() {
        let y = majority_class(&[None, None], 2);
        assert_eq!(row(&y, 0), [0.0, 0.0]);
        assert_eq!(row(&y, 1), [0.0, 0.0]);
    }

    #[test]
    fn nearest_reporter_on_a_path() {
        // 0 (says 0) - 1 - 2 - 3 - 4 (says 1): 1 and 3 are one hop from their end; 2 is
        // two hops from both, the tie goes to reporter 0.
        let y = nearest_reporter(&path(5), &[Some(0), None, None, None, Some(1)], 2);
        assert_eq!(row(&y, 1), [1.0, 0.0]);
        assert_eq!(row(&y, 2), [1.0, 0.0]);
        assert_eq!(row(&y, 3), [0.0, 1.0]);
        assert_eq!(row(&y, 4), [0.0, 1.0]);
    }

    #[test]
    fn nearest_reporter_cannot_reach_a_cut_off_drone() {
        let p = p_of(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        let y = nearest_reporter(&p, &[Some(1), None, None, None], 2);
        assert_eq!(row(&y, 1), [0.0, 1.0]);
        assert_eq!(row(&y, 2), [0.0, 0.0]);
        assert_eq!(row(&y, 3), [0.0, 0.0]);
    }

    #[test]
    fn nearest_by_distance_ignores_the_graph() {
        // Drone 1 is 1 km from reporter 0 and 3 km from reporter 2; drone 3 is 0.5 km
        // from reporter 2. No graph needed, even a cut-off drone gets a guess.
        let positions = [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [4.0, 0.5]];
        let y = nearest_reporter_by_distance(&positions, &[Some(0), None, Some(1), None], 2);
        assert_eq!(row(&y, 1), [1.0, 0.0]);
        assert_eq!(row(&y, 3), [0.0, 1.0]);
        assert_eq!(row(&y, 2), [0.0, 1.0], "a reporter keeps its own class");
    }

    #[test]
    fn nearest_by_distance_tie_and_no_reports() {
        let positions = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let y = nearest_reporter_by_distance(&positions, &[Some(1), None, Some(0)], 2);
        assert_eq!(row(&y, 1), [0.0, 1.0], "tie: the smaller index wins");
        let y = nearest_reporter_by_distance(&positions, &[None, None, None], 2);
        assert_eq!(row(&y, 1), [0.0, 0.0]);
    }

    #[test]
    fn nearest_reporter_ignores_weights_label_propagation_does_not() {
        // Drone 1 hangs on a weak link to 0 and a strong one to 2: same hop count, so the
        // baseline breaks the tie by index (class 0); LP follows the strong link (class 1).
        let p = p_of(3, &[(0, 1, 0.1), (1, 2, 10.0)]);
        let labels = [Some(0), None, Some(1)];
        let y = nearest_reporter(&p, &labels, 2);
        assert_eq!(row(&y, 1), [1.0, 0.0]);
        let (lp, _) = label_propagation(&p, &labels, 2, 1e-9, 1000);
        assert!(lp[(1, 1)] > lp[(1, 0)]);
    }
}
