//! The Mission 1 experiment: every method, on every graph, for every label, report share
//! and share of wrong reports, over the test seeds × 20 draws of which drones report.
//!
//! Used by the runner (`examples/m1.rs`) and by the swarm viewer's Mission 1 view. Seeds
//! run on parallel threads; the results are merged in seed order, so the output does not
//! depend on the number of cores.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use aerie_graph::EdgeList;
use faer::Mat;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::graphs::knn::knn_graph;
use swarmsim::snapshot::Snapshot;

use crate::eval::{Method, Problem, Score, corrupt, draw_reporters, hide, mean_sd, score};

/// Report shares for experiment 1 (the plan: 1 % to 50 %).
pub const SHARES: [f64; 7] = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
/// The plan's main setting: 10 % of the drones get a report through.
pub const MAIN_SHARE: f64 = 0.1;
/// Random draws of the reporters per seed and share.
pub const DRAWS: u64 = 20;
/// Shares of the reporters that say a wrong class (0 = every report is true).
pub const WRONG_SHARES: [f64; 4] = [0.0, 0.1, 0.25, 0.4];

/// The labels Mission 1 propagates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LabelKind {
    /// Inside a jammed zone or not (2 classes: 0 outside, 1 inside).
    Zone,
    /// Which squad (one class per squad).
    Squad,
    /// Which role (4 classes, in the order of `Role::ALL`).
    Role,
}

impl LabelKind {
    pub const ALL: [LabelKind; 3] = [LabelKind::Zone, LabelKind::Squad, LabelKind::Role];

    pub fn name(self) -> &'static str {
        match self {
            LabelKind::Zone => "zone",
            LabelKind::Squad => "squad",
            LabelKind::Role => "role",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.name() == name)
    }

    pub fn n_classes(self, config: &SimConfig) -> usize {
        match self {
            LabelKind::Zone => 2,
            LabelKind::Squad => config.n_squads,
            LabelKind::Role => Role::num_roles(),
        }
    }

    /// The true class of every drone. `jammed` is the zone label of each drone.
    pub fn truth(self, swarm: &[Drone], jammed: &[bool]) -> Vec<usize> {
        match self {
            LabelKind::Zone => jammed.iter().map(|&j| usize::from(j)).collect(),
            LabelKind::Squad => swarm.iter().map(|d| d.squad.0 as usize).collect(),
            LabelKind::Role => swarm.iter().map(|d| d.role as usize).collect(),
        }
    }
}

/// The graphs experiment 1 compares (the correlation graph comes with Mission 0 step 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GraphKind {
    /// The comm graph under jamming: the default.
    Live,
    /// The geometric epsilon graph, no jamming.
    Epsilon,
    /// The kNN graph (union rule).
    Knn,
}

impl GraphKind {
    pub const ALL: [GraphKind; 3] = [GraphKind::Live, GraphKind::Epsilon, GraphKind::Knn];

    pub fn name(self) -> &'static str {
        match self {
            GraphKind::Live => "live",
            GraphKind::Epsilon => "epsilon",
            GraphKind::Knn => "knn",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.name() == name)
    }

    /// This graph for a swarm, from the pieces a snapshot (or the viewer) already holds.
    pub fn build(
        self,
        swarm: &[Drone],
        geometric: &EdgeList,
        live: &EdgeList,
        config: &SimConfig,
    ) -> EdgeList {
        match self {
            GraphKind::Live => live.clone(),
            GraphKind::Epsilon => geometric.clone(),
            GraphKind::Knn => knn_graph(swarm, config),
        }
    }
}

/// The methods of a sweep: `Method::ALL` with RWR at `alpha`.
pub fn methods(alpha: f64) -> [Method; 5] {
    Method::ALL.map(|m| match m {
        Method::Rwr { .. } => Method::Rwr { alpha },
        other => other,
    })
}

/// Method by its `name()`; RWR gets `alpha`.
pub fn method_from_name(name: &str, alpha: f64) -> Option<Method> {
    methods(alpha).into_iter().find(|m| m.name() == name)
}

/// (x, y) of every drone, for the nearest-by-distance baseline.
pub fn positions(swarm: &[Drone]) -> Vec<[f64; 2]> {
    swarm.iter().map(|d| [d.position.x, d.position.y]).collect()
}

/// One run: one seed, draw, graph, report share, wrong share, label and method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunRow {
    pub seed: u64,
    pub draw: u64,
    pub graph: GraphKind,
    pub share: f64,
    /// Share of the reporters that said a wrong class.
    pub wrong: f64,
    pub label: LabelKind,
    pub method: Method,
    pub score: Score,
}

/// Mean over the runs of one setting (graph, share, wrong, label, method).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SummaryRow {
    pub graph: GraphKind,
    pub share: f64,
    pub wrong: f64,
    pub label: LabelKind,
    pub method: Method,
    pub runs: usize,
    pub accuracy: f64,
    pub accuracy_sd: f64,
    /// Mean share of the silent drones that got no guess.
    pub no_guess: f64,
    /// Share of all wrong reports the method overruled (gave the drone its true class);
    /// NaN when the setting has no wrong reports.
    pub overruled: f64,
}

/// What a sweep produces.
#[derive(Debug, Clone, Default)]
pub struct Sweep {
    pub runs: Vec<RunRow>,
    pub summary: Vec<SummaryRow>,
}

/// Shared with the threads of a sweep: progress for a progress bar, and a stop button.
#[derive(Debug, Default)]
pub struct Progress {
    /// Finished (seed, share, draw) units, out of `units(seeds)`.
    pub done: AtomicUsize,
    pub cancel: AtomicBool,
}

/// How many (seed, share, draw) units a sweep over `seeds` has.
pub fn units(seeds: &Range<u64>) -> usize {
    seeds.clone().count() * SHARES.len() * DRAWS as usize
}

/// Runs the whole experiment. Returns `Err` if cancelled or if a seed failed (a config
/// where `place_jamming` finds no placement).
pub fn sweep(
    config: &SimConfig,
    seeds: Range<u64>,
    alpha: f64,
    progress: &Progress,
) -> Result<Sweep, String> {
    let per_seed: Vec<Result<Vec<RunRow>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = seeds
            .clone()
            .map(|seed| scope.spawn(move || run_seed(seed, config, alpha, progress)))
            .collect();
        handles
            .into_iter()
            .zip(seeds.clone())
            .map(|(h, seed)| {
                h.join()
                    .map_err(|_| format!("seed {seed} failed (no jamming placement?)"))
                    .and_then(|r| r.ok_or_else(|| "cancelled".to_string()))
            })
            .collect()
    });

    let mut runs = Vec::new();
    for r in per_seed {
        runs.extend(r?);
    }
    let summary = summarize(&runs, alpha);
    Ok(Sweep { runs, summary })
}

/// Every run of one seed; `None` if cancelled. The same reporters for every label,
/// graph, wrong share and method of a draw, so the comparisons are paired.
fn run_seed(seed: u64, config: &SimConfig, alpha: f64, progress: &Progress) -> Option<Vec<RunRow>> {
    let snap = Snapshot::build(seed, config);
    let positions = positions(&snap.swarm);
    let graphs: Vec<(GraphKind, Mat<f64>)> = GraphKind::ALL
        .iter()
        .map(|&g| {
            let graph = g.build(&snap.swarm, &snap.geometric, &snap.live.graph, config);
            (g, graph.transition_matrix())
        })
        .collect();
    let truths: Vec<(LabelKind, Vec<usize>)> = LabelKind::ALL
        .iter()
        .map(|&l| (l, l.truth(&snap.swarm, &snap.jammed)))
        .collect();

    let mut rows = Vec::new();
    for &share in &SHARES {
        for draw in 0..DRAWS {
            if progress.cancel.load(Ordering::Relaxed) {
                return None;
            }
            let reports = draw_reporters(snap.n(), share, seed, draw);
            for (label, truth) in &truths {
                let n_classes = label.n_classes(config);
                let honest = hide(truth, &reports);
                for &wrong in &WRONG_SHARES {
                    let labels = corrupt(&honest, n_classes, wrong, seed, draw);
                    for (graph, p) in &graphs {
                        let problem = Problem {
                            p,
                            positions: &positions,
                            labels: &labels,
                            n_classes,
                        };
                        for method in methods(alpha) {
                            rows.push(RunRow {
                                seed,
                                draw,
                                graph: *graph,
                                share,
                                wrong,
                                label: *label,
                                method,
                                score: score(&method.run(&problem), truth, &labels),
                            });
                        }
                    }
                }
            }
            progress.done.fetch_add(1, Ordering::Relaxed);
        }
    }
    Some(rows)
}

/// Shares as whole permille, so they can be map keys (f64 is not `Ord`).
fn permille(share: f64) -> u64 {
    (share * 1000.0).round() as u64
}

/// Groups the runs by setting and averages them, sorted by graph, share, wrong share,
/// label, method.
pub fn summarize(runs: &[RunRow], alpha: f64) -> Vec<SummaryRow> {
    type Key = (GraphKind, u64, u64, LabelKind, Option<usize>);
    let index = |m: Method| methods(alpha).iter().position(|&x| x.name() == m.name());
    let mut groups: BTreeMap<Key, Vec<Score>> = BTreeMap::new();
    for r in runs {
        let key = (
            r.graph,
            permille(r.share),
            permille(r.wrong),
            r.label,
            index(r.method),
        );
        groups.entry(key).or_default().push(r.score);
    }
    groups
        .into_iter()
        .filter_map(|((graph, share, wrong, label, m), scores)| {
            let acc: Vec<f64> = scores.iter().map(|s| s.accuracy()).collect();
            let none: Vec<f64> = scores.iter().map(|s| s.no_guess_share()).collect();
            let (accuracy, accuracy_sd) = mean_sd(&acc);
            let liars: usize = scores.iter().map(|s| s.liars).sum();
            let overruled: usize = scores.iter().map(|s| s.overruled).sum();
            Some(SummaryRow {
                graph,
                share: share as f64 / 1000.0,
                wrong: wrong as f64 / 1000.0,
                label,
                method: methods(alpha)[m?],
                runs: scores.len(),
                accuracy,
                accuracy_sd,
                no_guess: mean_sd(&none).0,
                overruled: if liars == 0 {
                    f64::NAN
                } else {
                    overruled as f64 / liars as f64
                },
            })
        })
        .collect()
}

/// The summary row of one setting, if the sweep has it.
pub fn find(
    summary: &[SummaryRow],
    graph: GraphKind,
    share: f64,
    wrong: f64,
    label: LabelKind,
    method: &str,
) -> Option<SummaryRow> {
    summary.iter().copied().find(|r| {
        r.graph == graph
            && (r.share - share).abs() < 1e-9
            && (r.wrong - wrong).abs() < 1e-9
            && r.label == label
            && r.method.name() == method
    })
}

/// Where the results go: `results/m1` at the repo root.
pub fn results_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../results/m1")
}

pub fn runs_csv(runs: &[RunRow]) -> String {
    let mut out = String::from(
        "seed,draw,graph,share,wrong,label,method,right,scored,no_guess,liars,overruled\n",
    );
    for r in runs {
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            r.seed,
            r.draw,
            r.graph.name(),
            r.share,
            r.wrong,
            r.label.name(),
            r.method.name(),
            r.score.right,
            r.score.scored,
            r.score.no_guess,
            r.score.liars,
            r.score.overruled
        )
        .unwrap();
    }
    out
}

pub fn summary_csv(summary: &[SummaryRow]) -> String {
    let mut out = String::from(
        "graph,share,wrong,label,method,alpha,runs,accuracy,accuracy_sd,no_guess,overruled\n",
    );
    for r in summary {
        let alpha = match r.method {
            Method::Rwr { alpha } => format!("{alpha}"),
            _ => String::new(),
        };
        writeln!(
            out,
            "{},{},{},{},{},{alpha},{},{:.5},{:.5},{:.5},{:.5}",
            r.graph.name(),
            r.share,
            r.wrong,
            r.label.name(),
            r.method.name(),
            r.runs,
            r.accuracy,
            r.accuracy_sd,
            r.no_guess,
            r.overruled
        )
        .unwrap();
    }
    out
}

/// Reads `summary_csv` back. Lines that do not parse (or come from an older format) are
/// skipped.
pub fn parse_summary(text: &str) -> Vec<SummaryRow> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.trim().split(',').collect();
            let [
                graph,
                share,
                wrong,
                label,
                method,
                alpha,
                runs,
                acc,
                sd,
                none,
                over,
            ] = f.as_slice()
            else {
                return None;
            };
            let alpha = alpha.parse().unwrap_or(0.85);
            Some(SummaryRow {
                graph: GraphKind::from_name(graph)?,
                share: share.parse().ok()?,
                wrong: wrong.parse().ok()?,
                label: LabelKind::from_name(label)?,
                method: method_from_name(method, alpha)?,
                runs: runs.parse().ok()?,
                accuracy: acc.parse().ok()?,
                accuracy_sd: sd.parse().ok()?,
                no_guess: none.parse().ok()?,
                overruled: over.parse().ok()?,
            })
        })
        .collect()
}

/// Writes runs.csv and summary.csv to `results_dir()`; returns the folder.
pub fn write_results(sweep: &Sweep) -> std::io::Result<PathBuf> {
    let dir = results_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("runs.csv"), runs_csv(&sweep.runs))?;
    std::fs::write(dir.join("summary.csv"), summary_csv(&sweep.summary))?;
    dir.canonicalize()
}

/// Reads `results_dir()/summary.csv`, if it exists.
pub fn read_summary() -> Option<Vec<SummaryRow>> {
    let text = std::fs::read_to_string(results_dir().join("summary.csv")).ok()?;
    Some(parse_summary(&text))
}

/// The main table as text: one graph, one share, no wrong reports; accuracy (no-guess
/// share) per label and method.
pub fn main_table(summary: &[SummaryRow], graph: GraphKind, share: f64) -> String {
    let mut out = String::new();
    write!(out, "{:<7}", "label").unwrap();
    for m in Method::ALL {
        write!(out, "{:>16}", m.name()).unwrap();
    }
    out.push('\n');
    for label in LabelKind::ALL {
        write!(out, "{:<7}", label.name()).unwrap();
        for m in Method::ALL {
            let cell = find(summary, graph, share, 0.0, label, m.name())
                .map_or("—".into(), |r| {
                    format!("{:.1} ({:.1})", r.accuracy * 100.0, r.no_guess * 100.0)
                });
            write!(out, "{cell:>16}").unwrap();
        }
        out.push('\n');
    }
    out
}

/// Experiment 1 as text: one method's accuracy against the share, per graph, for `label`,
/// with no wrong reports.
pub fn share_table(summary: &[SummaryRow], label: LabelKind, method: &str) -> String {
    let mut out = String::new();
    write!(out, "{:<8}", "graph").unwrap();
    for share in SHARES {
        write!(out, "{:>8}", format!("{:.0}%", share * 100.0)).unwrap();
    }
    out.push('\n');
    for graph in GraphKind::ALL {
        write!(out, "{:<8}", graph.name()).unwrap();
        for share in SHARES {
            let cell = find(summary, graph, share, 0.0, label, method)
                .map_or("—".into(), |r| format!("{:.1}", r.accuracy * 100.0));
            write!(out, "{cell:>8}").unwrap();
        }
        out.push('\n');
    }
    out
}

/// Wrong reports as text: for one graph, share and label, each method's accuracy on the
/// silent drones at every wrong share, and in brackets the share of lies it overruled.
pub fn wrong_table(
    summary: &[SummaryRow],
    graph: GraphKind,
    share: f64,
    label: LabelKind,
) -> String {
    let mut out = String::new();
    write!(out, "{:<14}", "method").unwrap();
    for wrong in WRONG_SHARES {
        write!(out, "{:>18}", format!("{:.0}% wrong", wrong * 100.0)).unwrap();
    }
    out.push('\n');
    for m in Method::ALL {
        write!(out, "{:<14}", m.name()).unwrap();
        for wrong in WRONG_SHARES {
            let cell =
                find(summary, graph, share, wrong, label, m.name()).map_or("—".into(), |r| {
                    if r.overruled.is_nan() {
                        format!("{:.1}", r.accuracy * 100.0)
                    } else {
                        format!("{:.1} ({:.0})", r.accuracy * 100.0, r.overruled * 100.0)
                    }
                });
            write!(out, "{cell:>18}").unwrap();
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for l in LabelKind::ALL {
            assert_eq!(LabelKind::from_name(l.name()), Some(l));
        }
        for g in GraphKind::ALL {
            assert_eq!(GraphKind::from_name(g.name()), Some(g));
        }
        for m in methods(0.7) {
            assert_eq!(method_from_name(m.name(), 0.7), Some(m));
        }
    }

    #[test]
    fn summary_csv_round_trips() {
        let row = SummaryRow {
            graph: GraphKind::Knn,
            share: 0.05,
            wrong: 0.1,
            label: LabelKind::Role,
            method: Method::Rwr { alpha: 0.6 },
            runs: 200,
            accuracy: 0.31234,
            accuracy_sd: 0.1,
            no_guess: 0.02,
            overruled: 0.25,
        };
        let back = parse_summary(&summary_csv(&[row]));
        assert_eq!(back, [row]);
    }

    #[test]
    fn summarize_averages_per_setting() {
        let mk = |seed, right| RunRow {
            seed,
            draw: 0,
            graph: GraphKind::Live,
            share: 0.1,
            wrong: 0.0,
            label: LabelKind::Squad,
            method: Method::LabelPropagation,
            score: Score {
                right,
                scored: 10,
                ..Score::default()
            },
        };
        let s = summarize(&[mk(90, 4), mk(91, 6)], 0.85);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].runs, 2);
        assert!((s[0].accuracy - 0.5).abs() < 1e-12);
    }

    #[test]
    #[ignore = "slow in a debug build: cargo test --release -p aerie-classic -- --ignored"]
    fn a_small_sweep_is_complete_and_deterministic() {
        // One seed: every (graph, share, label, method) gets DRAWS runs, and running it
        // again gives the same numbers.
        let config = SimConfig::default();
        let progress = Progress::default();
        let a = sweep(&config, 90..91, 0.85, &progress).unwrap();
        assert_eq!(progress.done.load(Ordering::Relaxed), units(&(90..91)));
        assert_eq!(
            a.summary.len(),
            GraphKind::ALL.len()
                * SHARES.len()
                * WRONG_SHARES.len()
                * LabelKind::ALL.len()
                * Method::ALL.len()
        );
        assert!(a.summary.iter().all(|r| r.runs == DRAWS as usize));
        let b = sweep(&config, 90..91, 0.85, &Progress::default()).unwrap();
        assert_eq!(a.summary, b.summary);
    }

    #[test]
    fn cancel_stops_the_sweep() {
        let progress = Progress::default();
        progress.cancel.store(true, Ordering::Relaxed);
        assert!(sweep(&SimConfig::default(), 90..92, 0.85, &progress).is_err());
    }
}
