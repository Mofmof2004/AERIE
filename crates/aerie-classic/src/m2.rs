//! The Mission 2 experiment: what a listener learns from one flight's traffic.
//!
//! For one seed (`run_seed`): build the swarm and its traffic log, embed the undirected
//! traffic graph with DeepWalk and with Node2Vec at each (p, q) asked for, and score each
//! embedding three ways (`embed::score`):
//! - **role F1**: 5-fold logistic regression on Z, macro-F1;
//! - **link AUC**: 10 % of the links hidden *before* a second run on the rest;
//! - **squad NMI**: k-means (k = number of squads, 10 restarts) on the unit-length vectors.
//!
//! Baselines, to know what a number is worth:
//! - **random vectors**: a Z that learned nothing (same size). Its NMI is not 0: random
//!   clusters of ~10 drones overlap the squads by chance.
//! - **traffic counts** (role only): per drone, messages sent, received, and number of
//!   partners (log scale). What a listener gets without any embedding.
//! - **majority class** (role only): every drone called recon.
//!
//! `sweep` runs seeds on parallel threads. The runner (`examples/m2.rs`) and the swarm
//! viewer's Mission 2 view both call this module.

use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use aerie_graph::EdgeList;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;
use swarmsim::traffic::TrafficConfig;

use crate::embed::deepwalk::{DeepWalkConfig, deepwalk};
use crate::embed::node2vec::{Node2VecConfig, node2vec};
use crate::embed::score::{
    LinkSplit, cross_val_predict, hide_links, kmeans, link_auc, macro_f1, nmi, normalized,
};
use crate::embed::skipgram::dot;

/// Share of links hidden for the link test (plan).
pub const HIDDEN: f64 = 0.1;
/// k-means restarts.
pub const RESTARTS: usize = 10;
/// Folds of the role test (plan).
pub const FOLDS: usize = 5;

// RNG streams of the scoring, apart from DeepWalk's own (1 = walks, 2 = training).
const FOLD_STREAM: u64 = 3;
const HIDE_STREAM: u64 = 4;
const KMEANS_STREAM: u64 = 5;
const RANDOM_Z_STREAM: u64 = 6;

/// Experiment 1's Node2Vec settings: q below 1 and q above 1, p = 1 (the paper's values).
pub const EXPERIMENT_BIASES: [(f64, f64); 2] = [(1.0, 0.5), (1.0, 2.0)];

/// The plan's swarm for Mission 2: 30 squads, 300 drones ("100 nodes give noisy results").
pub fn plan_config() -> SimConfig {
    SimConfig::with_squads(30)
}

/// What gets scored: an embedding or a baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Method {
    DeepWalk,
    /// Node2Vec with its return (p) and in-out (q) parameters.
    Node2Vec {
        p: f64,
        q: f64,
    },
    RandomVectors,
    TrafficCounts,
    Majority,
}

impl Method {
    /// The baselines, in table order (after the embeddings).
    pub const BASELINES: [Method; 3] = [
        Method::RandomVectors,
        Method::TrafficCounts,
        Method::Majority,
    ];

    /// The name in tables and in the CSV, e.g. `node2vec p=1 q=0.5`.
    pub fn name(self) -> String {
        match self {
            Method::DeepWalk => "deepwalk".into(),
            Method::Node2Vec { p, q } => format!("node2vec p={p} q={q}"),
            Method::RandomVectors => "random vectors".into(),
            Method::TrafficCounts => "traffic counts".into(),
            Method::Majority => "majority class".into(),
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        if let Some(rest) = name.strip_prefix("node2vec p=") {
            let (p, q) = rest.split_once(" q=")?;
            return Some(Method::Node2Vec {
                p: p.parse().ok()?,
                q: q.parse().ok()?,
            });
        }
        [Method::DeepWalk]
            .into_iter()
            .chain(Method::BASELINES)
            .find(|m| m.name() == name)
    }

    /// An embedding (DeepWalk or Node2Vec), as opposed to a baseline.
    pub fn is_embedding(self) -> bool {
        matches!(self, Method::DeepWalk | Method::Node2Vec { .. })
    }
}

/// One method's scores on one seed. `None` where the method has nothing to say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    pub seed: u64,
    pub method: Method,
    pub role_f1: f64,
    pub link_auc: Option<f64>,
    pub squad_nmi: Option<f64>,
    /// Embedding time, seconds (both runs for an embedding; 0 for baselines).
    pub seconds: f64,
}

/// One embedding of one seed, and what the tests made of it (for the viewer's map).
pub struct Embedded {
    /// `DeepWalk` or `Node2Vec { p, q }`.
    pub method: Method,
    /// Z on the full traffic graph.
    pub z: Vec<Vec<f64>>,
    /// k-means cluster of each drone (on the unit-length rows of `z`).
    pub clusters: Vec<usize>,
    /// Each drone's role as guessed from `z` by a model that never saw it (5-fold).
    pub role_guess: Vec<usize>,
    /// The link test with Z trained without the hidden links: z_a · z_b of each hidden link
    /// and of each non-link, in the order of `SeedRun::split`.
    pub hidden_scores: Vec<f64>,
    pub non_link_scores: Vec<f64>,
}

/// Everything one seed gives: the scores, and what the viewer draws on the map.
pub struct SeedRun {
    pub seed: u64,
    /// The undirected traffic graph (weight = messages both ways).
    pub graph: EdgeList,
    /// Messages each drone sent and received.
    pub sent: Vec<u32>,
    pub received: Vec<u32>,
    /// True role (index into `Role::ALL`) and squad of each drone.
    pub roles: Vec<usize>,
    pub squads: Vec<usize>,
    /// DeepWalk first, then Node2Vec for each (p, q) asked for.
    pub embeddings: Vec<Embedded>,
    /// Each drone's role as guessed from the traffic counts.
    pub counts_guess: Vec<usize>,
    /// The link test: hidden links and non-links (the same for every embedding).
    pub split: LinkSplit,
    /// One row per method: the embeddings in order, then `Method::BASELINES`.
    pub rows: Vec<Row>,
}

fn rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

/// Log counts per drone: messages sent, received, number of partners.
fn traffic_counts(graph: &EdgeList, sent: &[u32], received: &[u32]) -> Vec<Vec<f64>> {
    let mut partners = vec![0.0_f64; graph.n as usize];
    for e in &graph.edges {
        partners[e.src as usize] += 1.0;
        partners[e.dst as usize] += 1.0;
    }
    (0..graph.n as usize)
        .map(|i| {
            vec![
                (1.0 + sent[i] as f64).ln(),
                (1.0 + received[i] as f64).ln(),
                (1.0 + partners[i]).ln(),
            ]
        })
        .collect()
}

/// Runs the whole Mission 2 test on seed `seed`: DeepWalk, Node2Vec at each of `biases`
/// ((p, q) pairs, with DeepWalk's walk and skip-gram settings), and the baselines. Same
/// arguments, same result. Every embedding sees the same hidden links, folds and k-means
/// starts, so their scores differ only by the vectors.
///
/// Panics if the snapshot can't be built (`place_jamming` finds no placement), as
/// `Snapshot::build` does.
pub fn run_seed(seed: u64, sim: &SimConfig, dw: &DeepWalkConfig, biases: &[(f64, f64)]) -> SeedRun {
    let snap = Snapshot::build(seed, sim);
    let log = snap.traffic(&TrafficConfig::default());
    let graph = log.undirected();
    let (sent, received) = log.sent_received();
    let n = snap.n();
    let classes = Role::ALL.len();
    let roles: Vec<usize> = snap.swarm.iter().map(|d| d.role as usize).collect();
    let squads: Vec<usize> = snap.swarm.iter().map(|d| d.squad.0 as usize).collect();
    let k = sim.n_squads.clamp(1, n);
    let split = hide_links(&graph, HIDDEN, &mut rng(seed, HIDE_STREAM));
    let squad_test = |z: &[Vec<f64>]| {
        let clusters = kmeans(&normalized(z), k, RESTARTS, &mut rng(seed, KMEANS_STREAM));
        let score = nmi(&clusters, &squads);
        (clusters, score)
    };
    let role_test = |x: &[Vec<f64>]| {
        let guess = cross_val_predict(x, &roles, classes, FOLDS, &mut rng(seed, FOLD_STREAM));
        let score = macro_f1(&roles, &guess, classes);
        (guess, score)
    };
    let embed = |method: Method, g: &EdgeList| match method {
        Method::Node2Vec { p, q } => {
            let config = Node2VecConfig {
                p,
                q,
                walks_per_node: dw.walks_per_node,
                walk_length: dw.walk_length,
                skipgram: dw.skipgram,
            };
            node2vec(g, &config, seed)
        }
        _ => deepwalk(g, dw, seed),
    };
    let mut rows = Vec::new();
    let mut embeddings = Vec::new();

    // Each embedding: once on the full graph (roles, squads), once without the hidden links.
    let methods = [Method::DeepWalk]
        .into_iter()
        .chain(biases.iter().map(|&(p, q)| Method::Node2Vec { p, q }));
    for method in methods {
        let start = Instant::now();
        let z = embed(method, &graph);
        let z_hidden = embed(method, &split.train);
        let seconds = start.elapsed().as_secs_f64();
        let (role_guess, f1) = role_test(&z);
        let (clusters, squad_nmi) = squad_test(&z);
        let pair_score = |&(a, b): &(usize, usize)| dot(&z_hidden[a], &z_hidden[b]);
        rows.push(Row {
            seed,
            method,
            role_f1: f1,
            link_auc: Some(link_auc(&z_hidden, &split)),
            squad_nmi: Some(squad_nmi),
            seconds,
        });
        embeddings.push(Embedded {
            method,
            hidden_scores: split.hidden.iter().map(pair_score).collect(),
            non_link_scores: split.non_links.iter().map(pair_score).collect(),
            z,
            clusters,
            role_guess,
        });
    }

    // Random vectors: the same size as Z, learned nothing.
    let mut r = rng(seed, RANDOM_Z_STREAM);
    let dim = dw.skipgram.dim;
    let random: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..dim).map(|_| r.random_range(-1.0..1.0)).collect())
        .collect();
    rows.push(Row {
        seed,
        method: Method::RandomVectors,
        role_f1: role_test(&random).1,
        link_auc: Some(link_auc(&random, &split)),
        squad_nmi: Some(squad_test(&random).1),
        seconds: 0.0,
    });

    // Traffic counts: what each drone sends, receives, and to how many partners.
    let (counts_guess, f1) = role_test(&traffic_counts(&graph, &sent, &received));
    rows.push(Row {
        seed,
        method: Method::TrafficCounts,
        role_f1: f1,
        link_auc: None,
        squad_nmi: None,
        seconds: 0.0,
    });

    // Majority class: everyone is a recon drone.
    rows.push(Row {
        seed,
        method: Method::Majority,
        role_f1: macro_f1(&roles, &vec![Role::Recon as usize; n], classes),
        link_auc: None,
        squad_nmi: None,
        seconds: 0.0,
    });

    SeedRun {
        seed,
        graph,
        sent,
        received,
        roles,
        squads,
        embeddings,
        counts_guess,
        split,
        rows,
    }
}

/// Shared with the threads of a sweep: progress for a progress bar, and a stop button.
#[derive(Debug, Default)]
pub struct Progress {
    /// Seeds finished.
    pub done: AtomicUsize,
    pub cancel: AtomicBool,
}

/// `run_seed` on every seed of `seeds`, one thread per seed. Rows come back in seed
/// order whatever thread finishes first. `Err` if stopped or if a seed panicked.
pub fn sweep(
    seeds: Range<u64>,
    sim: &SimConfig,
    dw: &DeepWalkConfig,
    biases: &[(f64, f64)],
    progress: &Progress,
) -> Result<Vec<Row>, String> {
    let per_seed: Vec<Option<Vec<Row>>> = std::thread::scope(|scope| {
        let jobs: Vec<_> = seeds
            .map(|seed| {
                scope.spawn(move || {
                    if progress.cancel.load(Ordering::Relaxed) {
                        return None;
                    }
                    let rows = run_seed(seed, sim, dw, biases).rows;
                    progress.done.fetch_add(1, Ordering::Relaxed);
                    Some(rows)
                })
            })
            .collect();
        jobs.into_iter().map(|j| j.join().ok().flatten()).collect()
    });
    if progress.cancel.load(Ordering::Relaxed) {
        return Err("stopped".into());
    }
    per_seed
        .into_iter()
        .map(|rows| rows.ok_or_else(|| "a seed failed to build".to_string()))
        .collect::<Result<Vec<_>, _>>()
        .map(|v| v.into_iter().flatten().collect())
}

/// Mean and spread (sample standard deviation) of `values`; spread 0 for one value.
pub fn mean_spread(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = if values.len() > 1 {
        (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    (mean, sd)
}

/// One method over all seeds: (mean, spread) of each score, `None` where it has none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub method: Method,
    pub seeds: usize,
    pub role_f1: (f64, f64),
    pub link_auc: Option<(f64, f64)>,
    pub squad_nmi: Option<(f64, f64)>,
}

/// Averages `rows` per method, in the order each method first appears.
pub fn summarize(rows: &[Row]) -> Vec<Summary> {
    let mut methods: Vec<Method> = Vec::new();
    for r in rows {
        if !methods.contains(&r.method) {
            methods.push(r.method);
        }
    }
    methods
        .into_iter()
        .map(|method| {
            let mine: Vec<&Row> = rows.iter().filter(|r| r.method == method).collect();
            let of = |f: fn(&Row) -> Option<f64>| {
                let v: Vec<f64> = mine.iter().filter_map(|r| f(r)).collect();
                (!v.is_empty()).then(|| mean_spread(&v))
            };
            Summary {
                method,
                seeds: mine.len(),
                role_f1: of(|r| Some(r.role_f1)).unwrap(),
                link_auc: of(|r| r.link_auc),
                squad_nmi: of(|r| r.squad_nmi),
            }
        })
        .collect()
}

/// The summary as a text table, for the terminal.
pub fn table(summary: &[Summary]) -> String {
    let cell = |v: Option<(f64, f64)>| v.map_or("—".into(), |(m, s)| format!("{m:.3} ± {s:.3}"));
    let mut out = format!(
        "{:<22} {:<15} {:<15} {:<15}\n",
        "method", "role F1", "link AUC", "squad NMI"
    );
    for s in summary {
        out += &format!(
            "{:<22} {:<15} {:<15} {:<15}\n",
            s.method.name(),
            cell(Some(s.role_f1)),
            cell(s.link_auc),
            cell(s.squad_nmi)
        );
    }
    out
}

/// `results/m2` at the repo root.
pub fn results_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../results/m2")
}

/// The rows as CSV: one line per seed and method.
pub fn rows_csv(rows: &[Row]) -> String {
    let opt = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.4}"));
    let mut csv = String::from("seed,method,role_f1,link_auc,squad_nmi,seconds\n");
    for r in rows {
        csv += &format!(
            "{},{},{:.4},{},{},{:.3}\n",
            r.seed,
            r.method.name(),
            r.role_f1,
            opt(r.link_auc),
            opt(r.squad_nmi),
            r.seconds
        );
    }
    csv
}

/// Reads `rows_csv` back. Lines that don't parse are skipped.
pub fn parse_rows(text: &str) -> Vec<Row> {
    let opt = |s: &str| {
        if s.is_empty() {
            Some(None)
        } else {
            s.parse().ok().map(Some)
        }
    };
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let [seed, method, role_f1, link_auc, squad_nmi, seconds] = f[..] else {
                return None;
            };
            Some(Row {
                seed: seed.parse().ok()?,
                method: Method::from_name(method)?,
                role_f1: role_f1.parse().ok()?,
                link_auc: opt(link_auc)?,
                squad_nmi: opt(squad_nmi)?,
                seconds: seconds.parse().ok()?,
            })
        })
        .collect()
}

/// The file the rows go to.
pub fn rows_path() -> PathBuf {
    results_dir().join("embeddings.csv")
}

/// Writes the rows to `rows_path()`; returns that path.
pub fn write_rows(rows: &[Row]) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(results_dir())?;
    std::fs::write(rows_path(), rows_csv(rows))?;
    Ok(rows_path())
}

/// Reads `rows_path()`, if it exists.
pub fn read_rows() -> Option<Vec<Row>> {
    let rows = parse_rows(&std::fs::read_to_string(rows_path()).ok()?);
    (!rows.is_empty()).then_some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::skipgram::SkipGramConfig;

    /// 100 drones and a tiny DeepWalk: fast enough for a debug test.
    fn small() -> (SimConfig, DeepWalkConfig) {
        let dw = DeepWalkConfig {
            walks_per_node: 2,
            walk_length: 10,
            skipgram: SkipGramConfig {
                dim: 8,
                window: 3,
                ..Default::default()
            },
        };
        (SimConfig::default(), dw)
    }

    #[test]
    fn one_seed_gives_every_method_and_a_guess_per_drone() {
        let (sim, dw) = small();
        let run = run_seed(0, &sim, &dw, &[(1.0, 0.5)]);
        let n = 100;
        let n2v = Method::Node2Vec { p: 1.0, q: 0.5 };
        assert_eq!(
            run.rows.iter().map(|r| r.method).collect::<Vec<_>>(),
            [vec![Method::DeepWalk, n2v], Method::BASELINES.to_vec()].concat()
        );
        assert_eq!(run.embeddings.len(), 2);
        for v in [&run.roles, &run.squads, &run.counts_guess] {
            assert_eq!(v.len(), n);
        }
        for (e, row) in run.embeddings.iter().zip(&run.rows) {
            assert_eq!(e.method, row.method);
            assert_eq!((e.z.len(), e.clusters.len(), e.role_guess.len()), (n, n, n));
            assert!(e.clusters.iter().all(|&c| c < sim.n_squads));
            assert_eq!(e.hidden_scores.len(), run.split.hidden.len());
            assert_eq!(e.non_link_scores.len(), run.split.non_links.len());
            // The guesses on the map are the ones the scores were computed from.
            assert_eq!(macro_f1(&run.roles, &e.role_guess, 4), row.role_f1);
            assert_eq!(nmi(&e.clusters, &run.squads), row.squad_nmi.unwrap());
        }
        for r in &run.rows {
            assert!((0.0..=1.0).contains(&r.role_f1));
            assert!(r.link_auc.is_none_or(|a| (0.0..=1.0).contains(&a)));
            assert!(r.squad_nmi.is_none_or(|a| (0.0..=1.0 + 1e-9).contains(&a)));
        }
    }

    #[test]
    fn node2vec_with_p_and_q_at_1_scores_exactly_like_deepwalk() {
        let (sim, dw) = small();
        let run = run_seed(2, &sim, &dw, &[(1.0, 1.0)]);
        let (a, b) = (&run.rows[0], &run.rows[1]);
        assert_eq!(run.embeddings[0].z, run.embeddings[1].z);
        assert_eq!(
            (a.role_f1, a.link_auc, a.squad_nmi),
            (b.role_f1, b.link_auc, b.squad_nmi)
        );
    }

    #[test]
    fn a_sweep_is_the_seeds_in_order_and_reproducible() {
        let (sim, dw) = small();
        let rows = sweep(3..5, &sim, &dw, &[], &Progress::default()).unwrap();
        assert_eq!(rows.len(), 2 * 4);
        assert!(rows[..4].iter().all(|r| r.seed == 3));
        assert!(rows[4..].iter().all(|r| r.seed == 4));
        let again = run_seed(4, &sim, &dw, &[]).rows;
        for (a, b) in rows[4..].iter().zip(&again) {
            assert_eq!(
                (a.role_f1, a.link_auc, a.squad_nmi),
                (b.role_f1, b.link_auc, b.squad_nmi)
            );
        }
    }

    #[test]
    fn a_stopped_sweep_says_so() {
        let (sim, dw) = small();
        let progress = Progress::default();
        progress.cancel.store(true, Ordering::Relaxed);
        assert!(sweep(0..2, &sim, &dw, &[], &progress).is_err());
    }

    #[test]
    fn rows_csv_round_trips() {
        let rows = vec![
            Row {
                seed: 7,
                method: Method::DeepWalk,
                role_f1: 0.5,
                link_auc: Some(0.875),
                squad_nmi: Some(0.25),
                seconds: 1.5,
            },
            Row {
                seed: 7,
                method: Method::Node2Vec { p: 1.0, q: 0.25 },
                role_f1: 0.4,
                link_auc: Some(0.9),
                squad_nmi: Some(0.8),
                seconds: 2.0,
            },
            Row {
                seed: 7,
                method: Method::Majority,
                role_f1: 0.1667,
                link_auc: None,
                squad_nmi: None,
                seconds: 0.0,
            },
        ];
        assert_eq!(parse_rows(&rows_csv(&rows)), rows);
        assert!(rows_csv(&rows).contains("node2vec p=1 q=0.25"));
    }

    #[test]
    fn summary_averages_per_method() {
        let row = |seed, f1| Row {
            seed,
            method: Method::TrafficCounts,
            role_f1: f1,
            link_auc: None,
            squad_nmi: None,
            seconds: 0.0,
        };
        let s = summarize(&[row(0, 0.4), row(1, 0.6)]);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].seeds, 2);
        assert!((s[0].role_f1.0 - 0.5).abs() < 1e-12);
        assert!((s[0].role_f1.1 - 0.02_f64.sqrt()).abs() < 1e-12);
        assert_eq!(s[0].link_auc, None);
        assert!(table(&s).contains("traffic counts"));
    }
}
