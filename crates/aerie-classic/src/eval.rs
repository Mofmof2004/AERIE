//! Mission 1 evaluation: draw which drones report, run a method, score the silent ones.
//!
//! Knows nothing about SwarmSim: a graph is its transition matrix P, a drone is an index,
//! a label is a class number. The runner (`examples/m1.rs`) brings the snapshots.

use faer::Mat;
use rand::seq::index::sample;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::baselines::{majority_class, nearest_reporter, nearest_reporter_by_distance};
use crate::lp::label_propagation_closed_form;
use crate::rwr::random_walk_with_restart_closed_form;

/// Every method Mission 1 compares. LP and RWR use their closed forms: the exact answer
/// the iterations converge to (tests check they agree), and one solve instead of up to
/// 1000 rounds, so a full sweep runs in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    Majority,
    /// Nearest reporter in km (the plan's baseline).
    NearestKm,
    /// Nearest reporter in hops on the graph.
    NearestHops,
    LabelPropagation,
    Rwr {
        alpha: f64,
    },
}

impl Method {
    /// The five methods with the plan's defaults (α = 0.85).
    pub const ALL: [Method; 5] = [
        Method::Majority,
        Method::NearestKm,
        Method::NearestHops,
        Method::LabelPropagation,
        Method::Rwr { alpha: 0.85 },
    ];

    /// Short name for tables and CSV files.
    pub fn name(self) -> &'static str {
        match self {
            Method::Majority => "majority",
            Method::NearestKm => "nearest_km",
            Method::NearestHops => "nearest_hops",
            Method::LabelPropagation => "lp",
            Method::Rwr { .. } => "rwr",
        }
    }

    /// Runs the method: one row per drone, the guess is the largest column.
    pub fn run(self, problem: &Problem) -> Mat<f64> {
        let Problem {
            p,
            positions,
            labels,
            n_classes,
        } = *problem;
        match self {
            Method::Majority => majority_class(labels, n_classes),
            Method::NearestKm => nearest_reporter_by_distance(positions, labels, n_classes),
            Method::NearestHops => nearest_reporter(p, labels, n_classes),
            Method::LabelPropagation => label_propagation_closed_form(p, labels, n_classes),
            Method::Rwr { alpha } => {
                random_walk_with_restart_closed_form(p, labels, n_classes, alpha)
            }
        }
    }
}

/// One instance to solve: a graph, where the drones are, and the reports.
#[derive(Debug, Clone, Copy)]
pub struct Problem<'a> {
    pub p: &'a Mat<f64>,
    pub positions: &'a [[f64; 2]],
    /// `Some(class)` for reporting drones, `None` for silent ones.
    pub labels: &'a [Option<usize>],
    pub n_classes: usize,
}

/// Which drones report: `round(share · n)` of them (at least 1), drawn without
/// replacement. Draw `draw` of seed `seed` always picks the same drones: its RNG is the
/// seed's ChaCha8 on stream `1 + draw` (stream 0 is the one that built the snapshot).
pub fn draw_reporters(n: usize, share: f64, seed: u64, draw: u64) -> Vec<bool> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(1 + draw);
    let k = ((share * n as f64).round() as usize).clamp(1, n.max(1));
    let mut reports = vec![false; n];
    for i in sample(&mut rng, n, k.min(n)) {
        reports[i] = true;
    }
    reports
}

/// The label slots the methods see: the true class where the drone reports, else `None`.
pub fn hide(truth: &[usize], reports: &[bool]) -> Vec<Option<usize>> {
    truth
        .iter()
        .zip(reports)
        .map(|(&c, &r)| r.then_some(c))
        .collect()
}

/// Makes `round(wrong_share · reporters)` of the reporters say a wrong class, picked
/// uniformly among the other classes (wrong reports: a drone whose report got through,
/// but says the wrong thing). Fixed by seed and draw: the RNG is the seed's ChaCha8 on
/// stream 2³² + draw, apart from the reporter draws. With fewer than 2 classes, nothing
/// changes.
pub fn corrupt(
    labels: &[Option<usize>],
    n_classes: usize,
    wrong_share: f64,
    seed: u64,
    draw: u64,
) -> Vec<Option<usize>> {
    let reporters: Vec<usize> = (0..labels.len()).filter(|&i| labels[i].is_some()).collect();
    let k = ((wrong_share * reporters.len() as f64).round() as usize).min(reporters.len());
    let mut out = labels.to_vec();
    if k == 0 || n_classes < 2 {
        return out;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream((1 << 32) + draw);
    let picks = sample(&mut rng, reporters.len(), k).into_vec();
    for idx in picks {
        let i = reporters[idx];
        let truth = labels[i].expect("a reporter");
        // Uniform over the n_classes - 1 other classes.
        let r = rng.random_range(0..n_classes - 1);
        out[i] = Some(if r >= truth { r + 1 } else { r });
    }
    out
}

/// The guess for every row: the largest column, `None` for a zero row (no label reached
/// it). Ties go to the smallest class.
pub fn guesses(y: &Mat<f64>) -> Vec<Option<usize>> {
    (0..y.nrows())
        .map(|i| {
            (0..y.ncols())
                .filter(|&c| y[(i, c)] > 0.0)
                .max_by(|&a, &b| y[(i, a)].total_cmp(&y[(i, b)]).then(b.cmp(&a)))
        })
        .collect()
}

/// Accuracy on the silent drones. A drone with no guess counts as wrong (decision for
/// Mission 1: keep split seeds, report the no-guess share next to the accuracy).
/// Wrong reporters are scored apart: did the method overrule the lie?
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Score {
    pub right: usize,
    /// Silent drones, the ones being scored.
    pub scored: usize,
    /// Silent drones with no guess (also counted in `scored`, not in `right`).
    pub no_guess: usize,
    /// Reporters whose report was wrong.
    pub liars: usize,
    /// Of those, how many the method gives their true class anyway.
    pub overruled: usize,
}

impl Score {
    pub fn accuracy(self) -> f64 {
        self.right as f64 / self.scored.max(1) as f64
    }

    pub fn no_guess_share(self) -> f64 {
        self.no_guess as f64 / self.scored.max(1) as f64
    }
}

/// Scores `y` against the truth. `labels` is what the method saw: silent drones (`None`)
/// are scored for accuracy; reporters that said something else than their truth are
/// counted as liars, and as overruled when `y` gives them their true class.
pub fn score(y: &Mat<f64>, truth: &[usize], labels: &[Option<usize>]) -> Score {
    let mut s = Score::default();
    for ((guess, &t), &said) in guesses(y).into_iter().zip(truth).zip(labels) {
        match said {
            None => {
                s.scored += 1;
                match guess {
                    Some(c) if c == t => s.right += 1,
                    Some(_) => {}
                    None => s.no_guess += 1,
                }
            }
            Some(c) if c != t => {
                s.liars += 1;
                if guess == Some(t) {
                    s.overruled += 1;
                }
            }
            Some(_) => {}
        }
    }
    s
}

/// Mean and sample standard deviation.
pub fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_graph::{Edge, EdgeList};

    fn path(n: u32) -> Mat<f64> {
        let edges = (0..n - 1)
            .map(|i| Edge {
                src: i,
                dst: i + 1,
                weight: 1.0,
            })
            .collect();
        EdgeList { n, edges }.transition_matrix()
    }

    #[test]
    fn draws_are_fixed_by_seed_and_draw_and_have_the_right_size() {
        let a = draw_reporters(100, 0.1, 90, 3);
        assert_eq!(a, draw_reporters(100, 0.1, 90, 3));
        assert_ne!(a, draw_reporters(100, 0.1, 90, 4));
        assert_ne!(a, draw_reporters(100, 0.1, 91, 3));
        assert_eq!(a.iter().filter(|&&r| r).count(), 10);
        assert_eq!(
            draw_reporters(100, 0.001, 0, 0)
                .iter()
                .filter(|&&r| r)
                .count(),
            1
        );
        assert!(draw_reporters(5, 1.0, 0, 0).iter().all(|&r| r));
    }

    #[test]
    fn hide_keeps_only_the_reports() {
        assert_eq!(
            hide(&[2, 0, 1], &[true, false, true]),
            [Some(2), None, Some(1)]
        );
    }

    #[test]
    fn guesses_take_the_largest_column_and_none_for_zero_rows() {
        let mut y: Mat<f64> = Mat::zeros(3, 3);
        y[(0, 1)] = 0.4;
        y[(0, 2)] = 0.6;
        y[(2, 0)] = 0.5;
        y[(2, 2)] = 0.5;
        assert_eq!(guesses(&y), [Some(2), None, Some(0)]);
    }

    #[test]
    fn score_counts_silent_drones_only_and_no_guess_as_wrong() {
        // Rows: drone 0 reports; 1 right; 2 wrong; 3 no guess.
        let mut y: Mat<f64> = Mat::zeros(4, 2);
        y[(0, 0)] = 1.0;
        y[(1, 1)] = 0.7;
        y[(2, 0)] = 0.9;
        let s = score(&y, &[0, 1, 1, 0], &[Some(0), None, None, None]);
        assert_eq!(
            s,
            Score {
                right: 1,
                scored: 3,
                no_guess: 1,
                liars: 0,
                overruled: 0,
            }
        );
        assert!((s.accuracy() - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn liars_are_counted_and_overruled_when_given_their_truth() {
        // Drone 0 says 1 but is class 0, and y gives it 0: overruled. Drone 1 says 0 but
        // is class 1, and y keeps 0: not overruled.
        let mut y: Mat<f64> = Mat::zeros(3, 2);
        y[(0, 0)] = 0.6;
        y[(0, 1)] = 0.4;
        y[(1, 0)] = 1.0;
        y[(2, 1)] = 1.0;
        let s = score(&y, &[0, 1, 1], &[Some(1), Some(0), None]);
        assert_eq!((s.liars, s.overruled, s.scored, s.right), (2, 1, 1, 1));
    }

    #[test]
    fn corrupt_flips_the_right_number_of_reports_to_another_class() {
        let truth: Vec<usize> = (0..100).map(|i| i % 4).collect();
        let reports = draw_reporters(100, 0.2, 91, 5);
        let labels = hide(&truth, &reports);
        let bad = corrupt(&labels, 4, 0.25, 91, 5);
        assert_eq!(
            bad,
            corrupt(&labels, 4, 0.25, 91, 5),
            "fixed by seed and draw"
        );
        let flipped: Vec<usize> = (0..100).filter(|&i| bad[i] != labels[i]).collect();
        assert_eq!(flipped.len(), 5, "25 % of 20 reporters");
        for i in flipped {
            assert!(labels[i].is_some(), "only reporters lie");
            assert_ne!(bad[i], Some(truth[i]), "a lie is a different class");
            assert!(bad[i].unwrap() < 4);
        }
        assert_eq!(corrupt(&labels, 4, 0.0, 91, 5), labels);
        assert_eq!(
            corrupt(&labels, 1, 0.5, 91, 5),
            labels,
            "one class: nothing to lie about"
        );
    }

    #[test]
    fn every_method_runs_on_a_path() {
        // 0 (says 0) - 1 - 2 - 3 - 4 (says 1): drone 1 is class 0 for everyone.
        let p = path(5);
        let positions: Vec<[f64; 2]> = (0..5).map(|i| [i as f64, 0.0]).collect();
        let labels = [Some(0), None, None, None, Some(1)];
        let problem = Problem {
            p: &p,
            positions: &positions,
            labels: &labels,
            n_classes: 2,
        };
        for m in Method::ALL {
            let y = m.run(&problem);
            assert_eq!((y.nrows(), y.ncols()), (5, 2), "{}", m.name());
            let g = guesses(&y);
            assert_eq!(g[0], Some(0), "{}", m.name());
            if m != Method::Majority {
                assert_eq!(g[1], Some(0), "{}", m.name());
                assert_eq!(g[3], Some(1), "{}", m.name());
            }
        }
    }

    #[test]
    fn mean_sd_of_a_few_values() {
        let (m, sd) = mean_sd(&[1.0, 2.0, 3.0]);
        assert!((m - 2.0).abs() < 1e-12 && (sd - 1.0).abs() < 1e-12);
    }
}
