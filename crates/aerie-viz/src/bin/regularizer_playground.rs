//! Regularizer playground (Mission 3): rebuild a signal from noisy sensors with four
//! regularizers side by side, and see the maths of each.
//!
//! Run with `cargo run --release -p aerie-viz --bin regularizer_playground`.
//!
//! The four, all from `aerie_classic::regularize`:
//! - **Tikhonov**: γ Σ_links w (x_i − x_j)², one linear solve: smooth slopes.
//! - **Trend filtering** (order 1): λ Σ_drones |(Lx)_i|, ADMM: straight pieces, few kinks.
//! - **Total variation** (trend filtering of order 0): λ Σ_links √w |x_i − x_j|, ADMM: flat
//!   pieces, sharp edges.
//! - **Sparse total variation**: total variation + λ₂ Σ_drones |x_i|, ADMM: flat pieces,
//!   most of the map exactly 0.
//!
//! - Left: a scenario (one made for each regularizer's strength, checked with real numbers,
//!   or your own), the settings (graph, truth, sensors, noise), each regularizer on or off with
//!   its parameter, what the map shows; then the results: NMSE of each, tuning every
//!   parameter (by the truth or by held-out sensors), the ADMM loops' convergence.
//! - Centre: one map per regularizer. Bottom: the signal along the path, or each map against
//!   the truth.
//! - Right: the maths: the common shape, then each regularizer's penalty, how it is solved,
//!   and its numbers at the selected drone; Tikhonov's matrices written out.

use aerie_classic::regularize::{Reconstruction, Regularizer, reconstruct};
use aerie_graph::{Edge, EdgeList};
use aerie_viz::grid::{Focus, fmt, matrix_grid, mix};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use faer::Mat;
use faer::linalg::solvers::Solve;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::Snapshot;

/// Colour ramp of the signal: low, middle, high.
const LOW: Color32 = Color32::from_rgb(45, 80, 170);
const MID: Color32 = Color32::from_rgb(240, 215, 90);
const HIGH: Color32 = Color32::from_rgb(205, 50, 40);
const TRUTH_COLOR: Color32 = Color32::from_rgb(160, 160, 160);
const READING_COLOR: Color32 = Color32::from_rgb(90, 170, 240);
const WARN_COLOR: Color32 = Color32::from_rgb(235, 90, 80);
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 205, 60);
const L_COLOR: Color32 = Color32::from_rgb(176, 122, 161);
/// The parameter grid of the tuning: 16 values from 0.001 to 100, log-spaced.
const TUNE_STEPS: usize = 16;
const PARAM_MIN: f64 = 0.001;
const PARAM_MAX: f64 = 100.0;
/// Share of the sensors held out when tuning without the truth (plan).
const HOLDOUT: f64 = 0.2;
/// At most this many drones in the written-out matrices.
const WINDOW: usize = 8;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE regularizer playground")
            .with_inner_size([1750.0, 1020.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE regularizer playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

fn param_grid() -> Vec<f64> {
    (0..TUNE_STEPS)
        .map(|k| {
            let t = k as f64 / (TUNE_STEPS - 1) as f64;
            (PARAM_MIN.ln() + t * (PARAM_MAX.ln() - PARAM_MIN.ln())).exp()
        })
        .collect()
}

// ------------------------------------------------------------------------------ methods

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Tikhonov,
    TrendFiltering,
    TotalVariation,
    SparseTv,
}

const METHODS: [Method; 4] = [
    Method::Tikhonov,
    Method::TrendFiltering,
    Method::TotalVariation,
    Method::SparseTv,
];

impl Method {
    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Method::Tikhonov => "Tikhonov",
            Method::TrendFiltering => "Trend filtering",
            Method::TotalVariation => "Total variation",
            Method::SparseTv => "Sparse total variation",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Method::Tikhonov => Color32::from_rgb(255, 180, 50),
            Method::TrendFiltering => Color32::from_rgb(90, 200, 140),
            Method::TotalVariation => Color32::from_rgb(225, 100, 180),
            Method::SparseTv => Color32::from_rgb(170, 130, 255),
        }
    }

    fn param(self) -> &'static str {
        match self {
            Method::Tikhonov => "γ",
            Method::SparseTv => "λ₁",
            _ => "λ",
        }
    }

    /// The regularizer at parameter p (λ₂ is sparse total variation's second parameter).
    fn regularizer(self, p: f64, lambda2: f64) -> Regularizer {
        match self {
            Method::Tikhonov => Regularizer::Tikhonov { gamma: p },
            Method::TrendFiltering => Regularizer::TrendFiltering { lambda: p },
            Method::TotalVariation => Regularizer::TotalVariation { lambda: p },
            Method::SparseTv => Regularizer::SparseTotalVariation {
                lambda1: p,
                lambda2,
            },
        }
    }

    fn penalty(self) -> &'static str {
        match self {
            Method::Tikhonov => "γ Σ_links w (x_i − x_j)²",
            Method::TrendFiltering => "λ Σ_drones |(Lx)_i|",
            Method::TotalVariation => "λ Σ_links √w |x_i − x_j|",
            Method::SparseTv => "λ₁ Σ_links √w |x_i − x_j| + λ₂ Σ_drones |x_i|",
        }
    }

    fn favours(self) -> &'static str {
        match self {
            Method::Tikhonov => "smooth slopes; squares the jumps, so it blurs edges",
            Method::TrendFiltering => "straight pieces joined at a few kinks",
            Method::TotalVariation => "flat pieces with sharp edges (stairs on slopes)",
            Method::SparseTv => "flat pieces, and most of the map exactly 0",
        }
    }
}

// ------------------------------------------------------------------------------- graphs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphKind {
    TinyPath,
    Path,
    Grid,
    Swarm,
}

impl GraphKind {
    const ALL: [GraphKind; 4] = [
        GraphKind::TinyPath,
        GraphKind::Path,
        GraphKind::Grid,
        GraphKind::Swarm,
    ];

    fn name(self) -> &'static str {
        match self {
            GraphKind::TinyPath => "Tiny path of 5 (see the matrices)",
            GraphKind::Path => "Path of 40",
            GraphKind::Grid => "Grid 12 × 12",
            GraphKind::Swarm => "Swarm ε-graph (100 drones)",
        }
    }

    fn is_path(self) -> bool {
        matches!(self, GraphKind::TinyPath | GraphKind::Path)
    }
}

struct World {
    pos: Vec<[f64; 2]>,
    graph: EdgeList,
    jamming: Option<Vec<f64>>,
    rf: Option<Vec<bool>>,
}

impl World {
    fn build(kind: GraphKind, seed: u64) -> World {
        let link = |a: usize, b: usize| Edge {
            src: a.min(b) as u32,
            dst: a.max(b) as u32,
            weight: 1.0,
        };
        let path = |n: usize| World {
            pos: (0..n).map(|i| [i as f64, 0.0]).collect(),
            graph: EdgeList {
                n: n as u32,
                edges: (0..n - 1).map(|i| link(i, i + 1)).collect(),
            },
            jamming: None,
            rf: None,
        };
        match kind {
            GraphKind::TinyPath => path(5),
            GraphKind::Path => path(40),
            GraphKind::Grid => {
                let k = 12;
                let mut edges = Vec::new();
                for r in 0..k {
                    for c in 0..k {
                        let i = r * k + c;
                        if c + 1 < k {
                            edges.push(link(i, i + 1));
                        }
                        if r + 1 < k {
                            edges.push(link(i, i + k));
                        }
                    }
                }
                World {
                    pos: (0..k * k)
                        .map(|i| [(i % k) as f64, (i / k) as f64])
                        .collect(),
                    graph: EdgeList {
                        n: (k * k) as u32,
                        edges,
                    },
                    jamming: None,
                    rf: None,
                }
            }
            GraphKind::Swarm => {
                let snap = Snapshot::build(seed, &SimConfig::default());
                World {
                    pos: snap
                        .swarm
                        .iter()
                        .map(|d| [d.position.x, -d.position.y])
                        .collect(),
                    graph: snap.geometric.clone(),
                    rf: Some(
                        snap.swarm
                            .iter()
                            .map(|d| d.role == Role::RfSensing)
                            .collect(),
                    ),
                    jamming: Some(snap.truth.clone()),
                }
            }
        }
    }

    fn n(&self) -> usize {
        self.pos.len()
    }

    /// Positions rescaled to [0, 1] along the longer side.
    fn unit(&self) -> Vec<[f64; 2]> {
        let (lo, hi) = bounds(&self.pos);
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-9);
        self.pos
            .iter()
            .map(|p| [(p[0] - lo[0]) / span, (p[1] - lo[1]) / span])
            .collect()
    }
}

fn bounds(pos: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in pos {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if pos.is_empty() {
        ([0.0; 2], [1.0; 2])
    } else {
        (lo, hi)
    }
}

// ------------------------------------------------------------------------------- truths

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Truth {
    RandomSmooth,
    Bump,
    Kinks,
    Steps,
    Disc,
    Hotspots,
    Jamming,
}

impl Truth {
    const ALL: [Truth; 7] = [
        Truth::RandomSmooth,
        Truth::Bump,
        Truth::Kinks,
        Truth::Steps,
        Truth::Disc,
        Truth::Hotspots,
        Truth::Jamming,
    ];

    fn name(self) -> &'static str {
        match self {
            Truth::RandomSmooth => "random smooth field",
            Truth::Bump => "smooth bump",
            Truth::Kinks => "slopes with kinks",
            Truth::Steps => "plateaus (steps)",
            Truth::Disc => "a disc on a raised floor",
            Truth::Hotspots => "small hotspots",
            Truth::Jamming => "true jamming power",
        }
    }

    fn make(self, world: &World, seed: u64) -> Vec<f64> {
        let unit = world.unit();
        let w = unit.iter().map(|p| p[0]).fold(0.0, f64::max).max(1e-9);
        let h = unit.iter().map(|p| p[1]).fold(0.0, f64::max);
        let (cx, cy) = (w / 2.0, h / 2.0);
        let d2 = |p: &[f64; 2], x: f64, y: f64| (p[0] - x).powi(2) + (p[1] - y).powi(2);
        match self {
            Truth::RandomSmooth => random_smooth(&world.graph, 0.01, seed),
            Truth::Bump => {
                let s = 0.2 * w.max(h);
                unit.iter()
                    .map(|p| (-d2(p, cx, cy) / (2.0 * s * s)).exp())
                    .collect()
            }
            Truth::Kinks => unit
                .iter()
                .map(|p| {
                    if h < 1e-9 {
                        // A zigzag along the path: up, down, up.
                        let x = p[0] / w;
                        if x < 0.33 {
                            x * 3.0
                        } else if x < 0.66 {
                            1.0 - (x - 0.33) * 1.5
                        } else {
                            0.5 + (x - 0.66) * 1.5
                        }
                    } else {
                        // A pyramid: straight faces, kinks on the diagonals.
                        1.0 - 2.0 * ((p[0] - cx).abs() / w).max((p[1] - cy).abs() / w)
                    }
                })
                .collect(),
            Truth::Steps => unit
                .iter()
                .map(|p| {
                    let x = p[0] / w;
                    if x < 0.3 {
                        0.0
                    } else if x < 0.7 {
                        1.0
                    } else {
                        0.4
                    }
                })
                .collect(),
            Truth::Disc => unit
                .iter()
                .map(|p| {
                    let r = 0.3 * w.max(h).max(1e-9);
                    let inside = if h < 1e-9 {
                        (p[0] - cx).abs() < r
                    } else {
                        d2(p, cx, cy) < r * r
                    };
                    // On a floor of 0.4, not 0: two flat levels, neither of them 0.
                    if inside { 1.0 } else { 0.4 }
                })
                .collect(),
            Truth::Hotspots => unit
                .iter()
                .map(|p| {
                    let r2 = (0.08 * w.max(h)).powi(2);
                    let spots = [(0.25, 0.3), (0.7, 0.7), (0.75, 0.2)];
                    if spots.iter().any(|&(x, y)| {
                        d2(p, x * w, y * h.max(1e-9)) < r2
                            || (h < 1e-9 && (p[0] - x * w).abs() < 0.04 * w)
                    }) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            Truth::Jamming => world
                .jamming
                .clone()
                .unwrap_or_else(|| vec![0.0; world.n()]),
        }
    }
}

/// A random smooth field: covariance (L + εI)⁻¹, drawn as (L + εI)⁻¹ (B w + √ε g) with w, g
/// standard normal. Neighbours differ by small random amounts: Tikhonov's own assumption.
fn random_smooth(g: &EdgeList, eps: f64, seed: u64) -> Vec<f64> {
    let n = g.n as usize;
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1000);
    let mut rhs: Mat<f64> = Mat::zeros(n, 1);
    for e in &g.edges {
        let w = gaussian(&mut rng) * e.weight.sqrt();
        rhs[(e.src as usize, 0)] += w;
        rhs[(e.dst as usize, 0)] -= w;
    }
    for i in 0..n {
        rhs[(i, 0)] += eps.sqrt() * gaussian(&mut rng);
    }
    let mut q = g.laplacian();
    for i in 0..n {
        q[(i, i)] += eps;
    }
    let x = q.partial_piv_lu().solve(&rhs);
    (0..n).map(|i| x[(i, 0)]).collect()
}

/// A standard normal number (Box–Muller).
fn gaussian(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ---------------------------------------------------------------------------- scenarios

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scenario {
    TikhonovCase,
    TrendCase,
    TvCase,
    StvCase,
    Custom,
}

impl Scenario {
    const ALL: [Scenario; 5] = [
        Scenario::TikhonovCase,
        Scenario::TrendCase,
        Scenario::TvCase,
        Scenario::StvCase,
        Scenario::Custom,
    ];

    fn name(self) -> &'static str {
        match self {
            Scenario::TikhonovCase => "Random smooth field: Tikhonov's case",
            Scenario::TrendCase => "Slopes with kinks: trend filtering's case",
            Scenario::TvCase => "A disc with a sharp edge: total variation's case",
            Scenario::StvCase => {
                "Small hotspots on a quiet background: sparse total variation's case"
            }
            Scenario::Custom => "Your own settings",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Scenario::TikhonovCase => {
                "Neighbours differ by small random amounts everywhere: exactly what Tikhonov \
                 assumes, so with its best γ it is the most accurate (path of 40, a reading \
                 everywhere, σ 0.5; this draw: 11 % better than the next one). The edge is                  real but small: on the grid Tikhonov wins 10 draws in 12, by about 4 %. The                  ADMM methods look for pieces and kinks that aren't there."
            }
            Scenario::TrendCase => {
                "Straight slopes that change direction twice. Trend filtering keeps the \
                 slopes straight and the kinks sharp; total variation turns slopes into \
                 stairs; Tikhonov rounds the kinks."
            }
            Scenario::TvCase => {
                "A flat disc on a raised flat floor (1 on 0.4), one sharp edge all around. \
                 Total variation keeps both levels flat and the edge sharp; Tikhonov blurs the \
                 edge; trend filtering makes a cone; sparse total variation's λ₂ drags the \
                 floor toward 0, which it isn't."
            }
            Scenario::StvCase => {
                "Three small hot patches, 0 everywhere else, a reading at every drone. Sparse \
                 total variation keeps the patches flat and puts the background at exactly 0; \
                 total variation leaves small levels everywhere, Tikhonov blurs the patches. \
                 Careful: with few sensors, λ₂ pulls every drone without a sensor to 0 (try \
                 the swarm with RF sensors only)."
            }
            Scenario::Custom => "Change anything below.",
        }
    }
}

// ----------------------------------------------------------------------------- measures

/// Drones in a connected piece with no sensor: there the matrices can't be inverted.
fn sensorless(graph: &EdgeList, has_sensor: &[bool]) -> Vec<usize> {
    let comp = graph.components();
    let pieces = comp.iter().max().map_or(0, |m| m + 1);
    let mut covered = vec![false; pieces];
    for (i, &c) in comp.iter().enumerate() {
        if has_sensor[i] {
            covered[c] = true;
        }
    }
    (0..comp.len()).filter(|&i| !covered[comp[i]]).collect()
}

/// Σ (a − b)² / Σ b² over the drones where `keep` is true.
fn nmse(estimate: &[f64], truth: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
    let (mut err, mut size) = (0.0, 0.0);
    for i in (0..truth.len().min(estimate.len())).filter(|&i| keep(i)) {
        err += (estimate[i] - truth[i]).powi(2);
        size += truth[i].powi(2);
    }
    if size > 0.0 { err / size } else { err }
}

/// One parameter of the tuning, for one method.
#[derive(Debug, Clone, Copy)]
struct TunePoint {
    param: f64,
    /// NMSE against the truth, or the error on held-out sensors.
    error: f64,
}

struct Tuning {
    version: u64,
    by_truth: bool,
    curves: [Vec<TunePoint>; 4],
}

// ---------------------------------------------------------------------------------- app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Estimate,
    Truth,
    Readings,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    ToggleSensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SensorRule {
    Share,
    RfDrones,
}

struct Playground {
    scenario: Scenario,
    kind: GraphKind,
    seed: u64,
    world: World,
    truth_kind: Truth,
    rule: SensorRule,
    share: f64,
    blind_third: bool,
    sensor_seed: u64,
    noise: f64,
    noise_seed: u64,
    /// Bumped when the data changes: every map is recomputed.
    version: u64,
    truth: Vec<f64>,
    has_sensor: Vec<bool>,
    y: Vec<f64>,
    params: [f64; 4],
    /// Sparse total variation's λ₂ (its λ₁ is in `params`).
    lambda2: f64,
    shown: [bool; 4],
    iterations: usize,
    results: [Option<Reconstruction>; 4],
    result_for: [Option<(u64, u64, u64, usize)>; 4],
    tuning: Option<Tuning>,
    selected: usize,
    hover: Option<usize>,
    view: View,
    tool: Tool,
}

impl Default for Playground {
    fn default() -> Self {
        let mut p = Playground {
            scenario: Scenario::TrendCase,
            kind: GraphKind::Path,
            seed: 0,
            world: World::build(GraphKind::Path, 0),
            truth_kind: Truth::Kinks,
            rule: SensorRule::Share,
            share: 1.0,
            blind_third: true,
            sensor_seed: 1,
            noise: 0.4,
            noise_seed: 7,
            version: 0,
            truth: Vec::new(),
            has_sensor: Vec::new(),
            y: Vec::new(),
            params: [1.0, 0.1, 0.1, 0.1],
            lambda2: 0.05,
            shown: [true; 4],
            iterations: 800,
            results: [None, None, None, None],
            result_for: [None; 4],
            tuning: None,
            selected: 0,
            hover: None,
            view: View::Estimate,
            tool: Tool::Select,
        };
        p.apply(Scenario::TrendCase);
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.solve();
        egui::Panel::left("settings")
            .resizable(true)
            .default_size(410.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.settings(ui);
                    self.solve();
                    ui.separator();
                    self.results(ui);
                });
            });
        egui::Panel::right("math")
            .resizable(true)
            .default_size(540.0)
            .show(ui, |ui| {
                egui::ScrollArea::both().show(ui, |ui| self.math(ui));
            });
        egui::Panel::bottom("profile")
            .resizable(true)
            .default_size(230.0)
            .show(ui, |ui| self.profile(ui));
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl Playground {
    fn n(&self) -> usize {
        self.world.n()
    }

    /// A scenario's graph, truth, sensors, noise and parameters (the best ones found by
    /// tuning each regularizer against the truth).
    fn apply(&mut self, s: Scenario) {
        self.scenario = s;
        let (kind, truth, rule, share, noise, params): (
            GraphKind,
            Truth,
            SensorRule,
            f64,
            f64,
            [f64; 4],
        ) = match s {
            Scenario::TikhonovCase => (
                GraphKind::Path,
                Truth::RandomSmooth,
                SensorRule::Share,
                1.0,
                0.5,
                [2.15, 2.15, 1.0, 0.46],
            ),
            Scenario::TrendCase => (
                GraphKind::Path,
                Truth::Kinks,
                SensorRule::Share,
                1.0,
                0.4,
                [2.15, 0.46, 0.1, 0.046],
            ),
            Scenario::TvCase => (
                GraphKind::Grid,
                Truth::Disc,
                SensorRule::Share,
                1.0,
                0.5,
                [0.22, 0.046, 0.1, 0.1],
            ),
            Scenario::StvCase => (
                GraphKind::Grid,
                Truth::Hotspots,
                SensorRule::Share,
                1.0,
                0.5,
                [0.1, 0.022, 0.046, 0.022],
            ),
            Scenario::Custom => return,
        };
        self.kind = kind;
        self.truth_kind = truth;
        self.rule = rule;
        self.share = share;
        self.noise = noise;
        self.params = params;
        self.lambda2 = if s == Scenario::StvCase { 0.1 } else { 0.05 };
        // The random smooth field's draw (and the swarm's seed for the others).
        self.seed = if s == Scenario::TikhonovCase { 11 } else { 0 };
        self.sensor_seed = 1;
        self.noise_seed = 7;
        self.blind_third = true;
        self.world = World::build(kind, self.seed);
        self.rebuild();
    }

    /// New truth, sensors or readings.
    fn rebuild(&mut self) {
        let n = self.n();
        if self.truth_kind == Truth::Jamming && self.kind != GraphKind::Swarm {
            self.truth_kind = Truth::Bump;
        }
        if self.kind != GraphKind::Swarm {
            self.rule = SensorRule::Share;
        }
        self.truth = self.truth_kind.make(&self.world, self.seed);
        let mut rng = ChaCha8Rng::seed_from_u64(self.sensor_seed);
        self.has_sensor = match (self.rule, &self.world.rf) {
            (SensorRule::RfDrones, Some(rf)) => {
                let mut s = rf.clone();
                if self.blind_third {
                    let mut ids: Vec<usize> = (0..n).filter(|&i| rf[i]).collect();
                    ids.shuffle(&mut rng);
                    for &i in ids.iter().take(ids.len() / 3) {
                        s[i] = false;
                    }
                }
                s
            }
            _ => (0..n)
                .map(|_| rng.random_range(0.0..1.0) < self.share)
                .collect(),
        };
        if !self.has_sensor.iter().any(|&s| s) && n > 0 {
            self.has_sensor[0] = true;
        }
        let mean = self.truth.iter().sum::<f64>() / n.max(1) as f64;
        let spread = (self.truth.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1) as f64)
            .sqrt()
            .max(1e-9);
        let mut rng = ChaCha8Rng::seed_from_u64(self.noise_seed);
        self.y = self
            .truth
            .iter()
            .map(|v| v + self.noise * spread * gaussian(&mut rng))
            .collect();
        self.selected = self.selected.min(n.saturating_sub(1));
        self.version += 1;
        self.tuning = None;
        self.solve(); // at once: panels drawn later this frame must see the new sizes
    }

    fn set_graph(&mut self, kind: GraphKind) {
        self.kind = kind;
        self.world = World::build(kind, self.seed);
        if kind == GraphKind::Swarm {
            self.truth_kind = Truth::Jamming;
            self.rule = SensorRule::RfDrones;
        }
        self.rebuild();
    }

    /// Each shown regularizer's map, recomputed when its data or parameter changed.
    fn solve(&mut self) {
        let ok = sensorless(&self.world.graph, &self.has_sensor).is_empty();
        for m in METHODS {
            let i = m.index();
            if !self.shown[i] || !ok {
                self.results[i] = None;
                self.result_for[i] = None;
                continue;
            }
            let key = (
                self.version,
                self.params[i].to_bits(),
                self.lambda2.to_bits(),
                self.iterations,
            );
            if self.result_for[i] != Some(key) {
                self.results[i] = Some(reconstruct(
                    &self.world.graph,
                    &self.y,
                    &self.has_sensor,
                    m.regularizer(self.params[i], self.lambda2),
                    self.iterations,
                ));
                self.result_for[i] = Some(key);
            }
        }
    }

    fn map(&self, m: Method) -> Option<&[f64]> {
        self.results[m.index()]
            .as_ref()
            .map(|r| r.x.as_slice())
            .filter(|x| x.len() == self.n())
    }

    /// Every method on the parameter grid: NMSE against the truth, or the error on 20 %
    /// held-out sensors (what a real swarm can measure).
    fn tune(&mut self, by_truth: bool) {
        let g = &self.world.graph;
        let mut kept = self.has_sensor.clone();
        let mut held = Vec::new();
        if !by_truth {
            let mut sensors: Vec<usize> = (0..self.n()).filter(|&i| self.has_sensor[i]).collect();
            sensors.shuffle(&mut ChaCha8Rng::seed_from_u64(self.sensor_seed + 99));
            let k = ((HOLDOUT * sensors.len() as f64).round() as usize).max(1);
            held = sensors[..k.min(sensors.len().saturating_sub(1))].to_vec();
            for &i in &held {
                kept[i] = false;
            }
            if held.is_empty() || !sensorless(g, &kept).is_empty() {
                return;
            }
        }
        let curves = METHODS.map(|m| {
            param_grid()
                .into_iter()
                .map(|p| {
                    let x = reconstruct(
                        g,
                        &self.y,
                        &kept,
                        m.regularizer(p, self.lambda2),
                        self.iterations,
                    )
                    .x;
                    let error = if by_truth {
                        nmse(&x, &self.truth, |_| true)
                    } else {
                        held.iter()
                            .map(|&i| (x[i] - self.y[i]).powi(2))
                            .sum::<f64>()
                            / held.len() as f64
                    };
                    TunePoint { param: p, error }
                })
                .collect()
        });
        self.tuning = Some(Tuning {
            version: self.version,
            by_truth,
            curves,
        });
    }

    // ------------------------------------------------------------------ left: settings

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Regularizer playground");
        ui.weak("Four ways to rebuild a map from noisy sensors, side by side.");
        let mut scenario = self.scenario;
        egui::ComboBox::from_id_salt("scenario")
            .selected_text(scenario.name())
            .width(370.0)
            .show_ui(ui, |ui| {
                for s in Scenario::ALL {
                    ui.selectable_value(&mut scenario, s, s.name());
                }
            });
        if scenario != self.scenario {
            self.apply(scenario);
        }
        ui.weak(self.scenario.about());

        let mut changed = false;
        egui::CollapsingHeader::new(RichText::new("Graph, truth, sensors, noise").strong())
            .default_open(false)
            .show(ui, |ui| {
                let mut kind = self.kind;
                egui::ComboBox::from_id_salt("graph")
                    .selected_text(kind.name())
                    .width(300.0)
                    .show_ui(ui, |ui| {
                        for k in GraphKind::ALL {
                            ui.selectable_value(&mut kind, k, k.name());
                        }
                    });
                if kind != self.kind {
                    self.scenario = Scenario::Custom;
                    self.set_graph(kind);
                }
                if self.kind == GraphKind::Swarm {
                    ui.horizontal(|ui| {
                        ui.label("Swarm seed");
                        let mut seed = self.seed;
                        ui.add(egui::DragValue::new(&mut seed).range(0..=99));
                        if seed != self.seed {
                            self.seed = seed;
                            self.scenario = Scenario::Custom;
                            self.set_graph(GraphKind::Swarm);
                        }
                    });
                }
                ui.horizontal_wrapped(|ui| {
                    ui.label("Truth:");
                    for t in Truth::ALL {
                        if t == Truth::Jamming && self.kind != GraphKind::Swarm {
                            continue;
                        }
                        changed |= ui
                            .selectable_value(&mut self.truth_kind, t, t.name())
                            .changed();
                    }
                });
                if self.kind == GraphKind::Swarm {
                    ui.horizontal(|ui| {
                        ui.label("Sensors:");
                        changed |= ui
                            .selectable_value(
                                &mut self.rule,
                                SensorRule::RfDrones,
                                "RF sensing drones",
                            )
                            .changed();
                        changed |= ui
                            .selectable_value(&mut self.rule, SensorRule::Share, "random share")
                            .changed();
                    });
                    if self.rule == SensorRule::RfDrones {
                        changed |= ui
                            .checkbox(&mut self.blind_third, "a third of them blinded")
                            .changed();
                    }
                }
                if self.rule == SensorRule::Share {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.share, 0.05..=1.0)
                                .text("share with a sensor"),
                        )
                        .changed();
                }
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.noise, 0.0..=1.5)
                            .text("noise σ (× signal spread)"),
                    )
                    .changed();
                ui.horizontal(|ui| {
                    if ui.button("New sensors").clicked() {
                        self.sensor_seed += 1;
                        changed = true;
                    }
                    if ui.button("New noise").clicked() {
                        self.noise_seed += 1;
                        changed = true;
                    }
                });
            });
        if changed {
            self.scenario = Scenario::Custom;
            self.rebuild();
        }

        ui.add_space(4.0);
        ui.label(RichText::new("Regularizers").strong());
        for m in METHODS {
            let i = m.index();
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.shown[i], RichText::new(m.name()).color(m.color()));
            });
            if self.shown[i] {
                ui.add(
                    egui::Slider::new(&mut self.params[i], PARAM_MIN..=PARAM_MAX)
                        .logarithmic(true)
                        .text(m.param()),
                );
                if m == Method::SparseTv {
                    ui.add(
                        egui::Slider::new(&mut self.lambda2, PARAM_MIN..=10.0)
                            .logarithmic(true)
                            .text("λ₂ (sparsity)"),
                    );
                }
            }
        }
        ui.add(
            egui::Slider::new(&mut self.iterations, 10..=3000)
                .logarithmic(true)
                .text("ADMM rounds"),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Maps show:");
            ui.selectable_value(&mut self.view, View::Estimate, "x̂");
            ui.selectable_value(&mut self.view, View::Truth, "truth");
            ui.selectable_value(&mut self.view, View::Readings, "readings");
            ui.selectable_value(&mut self.view, View::Error, "error x̂ − x");
        });
        ui.horizontal(|ui| {
            ui.label("Click a drone to");
            ui.selectable_value(&mut self.tool, Tool::Select, "select it");
            ui.selectable_value(&mut self.tool, Tool::ToggleSensor, "toggle its sensor");
        });
    }

    // ------------------------------------------------------------------- left: results

    fn results(&mut self, ui: &mut egui::Ui) {
        ui.heading("Which map is best?");
        let lost = sensorless(&self.world.graph, &self.has_sensor);
        if !lost.is_empty() {
            ui.colored_label(
                WARN_COLOR,
                format!(
                    "{} drones sit in a piece with no sensor (crossed on the maps): nothing pins \
                     their level. Add a sensor there, or new sensors.",
                    lost.len()
                ),
            );
            return;
        }
        let scores: Vec<(Method, f64, f64)> = METHODS
            .iter()
            .filter_map(|&m| {
                self.map(m).map(|x| {
                    (
                        m,
                        nmse(x, &self.truth, |_| true),
                        nmse(x, &self.truth, |i| !self.has_sensor[i]),
                    )
                })
            })
            .collect();
        let best = scores
            .iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|s| s.0);
        egui::Grid::new("scores").striped(true).show(ui, |ui| {
            ui.label("NMSE");
            ui.label("every drone");
            ui.label("no sensor");
            ui.end_row();
            for &(m, all, unsensed) in &scores {
                let name = RichText::new(m.name()).color(m.color());
                ui.label(if Some(m) == best { name.strong() } else { name });
                let v = RichText::new(fmt(all));
                ui.label(if Some(m) == best { v.strong() } else { v });
                ui.label(if self.has_sensor.iter().all(|&s| s) {
                    "—".to_string()
                } else {
                    fmt(unsensed)
                });
                ui.end_row();
            }
            ui.label("raw readings");
            ui.label(fmt(nmse(&self.y, &self.truth, |i| self.has_sensor[i])));
            ui.end_row();
        });
        if let Some(b) = best {
            ui.label(RichText::new(format!("Lowest error now: {}", b.name())).color(b.color()));
        }
        ui.weak("NMSE = Σ (x̂ − x)² / Σ x²: 0 is perfect. Each parameter matters: tune them below.");

        ui.add_space(6.0);
        ui.label(RichText::new("Tune every parameter").strong());
        ui.horizontal(|ui| {
            if ui.button("By the truth").on_hover_text("Only possible in a simulation.").clicked() {
                self.tune(true);
            }
            if ui
                .button("By held-out sensors")
                .on_hover_text("Hide 20 % of the sensors, rebuild from the rest, score on the hidden readings: what a real swarm can do.")
                .clicked()
            {
                self.tune(false);
            }
        });
        ui.weak(format!(
            "{TUNE_STEPS} values from 0.001 to 100 for each regularizer (a second or two with \
             ADMM)."
        ));
        let mut set: Option<[f64; 4]> = None;
        if let Some(t) = &self.tuning
            && t.version == self.version
        {
            tuning_plot(ui, t);
            let bests: Vec<(Method, TunePoint)> = METHODS
                .iter()
                .filter_map(|&m| {
                    t.curves[m.index()]
                        .iter()
                        .filter(|p| p.error.is_finite())
                        .min_by(|a, b| a.error.total_cmp(&b.error))
                        .map(|p| (m, *p))
                })
                .collect();
            egui::Grid::new("bests").striped(true).show(ui, |ui| {
                ui.label("best");
                ui.label("parameter");
                ui.label(if t.by_truth { "NMSE" } else { "held-out error" });
                ui.end_row();
                for (m, p) in &bests {
                    ui.colored_label(m.color(), m.name());
                    ui.label(format!("{} = {}", m.param(), fmt(p.param)));
                    ui.label(fmt(p.error));
                    ui.end_row();
                }
            });
            if ui.button("Use these parameters").clicked() {
                let mut params = self.params;
                for (m, p) in &bests {
                    params[m.index()] = p.param;
                }
                set = Some(params);
            }
        }
        if let Some(p) = set {
            self.params = p;
        }

        let loops: Vec<(Method, &Vec<f64>)> = METHODS
            .iter()
            .filter_map(|&m| {
                self.results[m.index()]
                    .as_ref()
                    .filter(|r| !r.gaps.is_empty())
                    .map(|r| (m, &r.gaps))
            })
            .collect();
        if !loops.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Are the ADMM loops done?").strong());
            convergence_plot(ui, &loops);
            ui.weak("The gap after each round, log scale: it should fall toward 0. If a curve is still falling at the end, give it more rounds.");
        }
    }

    // ------------------------------------------------------------------------- maths

    fn window(&self) -> Vec<usize> {
        let n = self.n();
        if n <= WINDOW {
            return (0..n).collect();
        }
        let i = self.selected.min(n - 1);
        let mut win = vec![i];
        for e in &self.world.graph.edges {
            let (a, b) = (e.src as usize, e.dst as usize);
            for (me, other) in [(a, b), (b, a)] {
                if me == i && !win.contains(&other) && win.len() < WINDOW {
                    win.push(other);
                }
            }
        }
        let mut k = 1;
        while win.len() < WINDOW && k < n {
            for j in [i.wrapping_sub(k), i + k] {
                if j < n && !win.contains(&j) && win.len() < WINDOW {
                    win.push(j);
                }
            }
            k += 1;
        }
        win.sort_unstable();
        win
    }

    fn math(&self, ui: &mut egui::Ui) {
        let g = &self.world.graph;
        let n = self.n();
        ui.heading("The maths");
        section(ui, "The common shape", SELECT_COLOR);
        mono(
            ui,
            "x̂ = argmin   fit to the readings   +   penalty on change",
        );
        ui.weak(
            "All four keep x̂ close to the readings at the sensor drones; they differ only in \
             how they measure \"change\", which decides what kind of map comes out:",
        );
        egui::Grid::new("penalties").striped(true).show(ui, |ui| {
            for m in METHODS {
                ui.colored_label(m.color(), m.name());
                ui.label(RichText::new(m.penalty()).monospace());
                ui.end_row();
                ui.label("");
                ui.weak(m.favours());
                ui.end_row();
            }
        });
        ui.weak(
            "Squared (Tikhonov): one big change costs far more than many small ones, so change \
             spreads out. Not squared (the other three): change is cheap where it is big, so \
             it gathers on a few links or drones and the rest goes exactly flat.",
        );

        let i = self.selected.min(n.saturating_sub(1));
        if n == 0 {
            return;
        }
        // Tikhonov, written out.
        if self.shown[0] {
            self.math_tikhonov(ui);
        }
        let tiny = {
            let (lo, hi) = self.value_range();
            1e-4 * (hi - lo).max(1e-12)
        };
        // Trend filtering.
        if let Some(x) = self.map(Method::TrendFiltering) {
            let m = Method::TrendFiltering;
            section(ui, "Trend filtering", m.color());
            mono(ui, "x̂ = argmin ½‖M(x − y)‖² + λ ‖Lx‖₁");
            ui.weak(
                "(Lx)_i = Σ_j w_ij (x_i − x_j): how far drone i sits above its neighbours. It \
                 is 0 on a straight slope and big at a kink, so penalising |Lx| keeps slopes \
                 and allows only a few kinks.",
            );
            mono(ui, "x ← solve (MᵀM + ρL²) x = MᵀM y + ρ L (z − u)");
            mono(ui, "z ← soft_threshold(Lx + u, λ/ρ)    drone by drone");
            mono(ui, "u ← u + Lx − z");
            let lx = times_l(g, x);
            let bends = lx.iter().filter(|v| v.abs() > tiny).count();
            mono(ui, &format!("drones with a bend (Lx ≠ 0): {bends} of {n}"));
            mono(ui, &format!("drone {i}: (Lx)_{i} = {}", fmt(lx[i])));
            self.loop_line(ui, m);
        }
        // Total variation.
        if let Some(x) = self.map(Method::TotalVariation) {
            let m = Method::TotalVariation;
            section(ui, "Total variation", m.color());
            mono(ui, "x̂ = argmin ½‖M(x − y)‖² + λ ‖Bᵀx‖₁");
            ui.weak(
                "(Bᵀx)_e = √w (x_i − x_j): the jump on link e. Penalising |jump| lets a few \
                 links take all the change: flat pieces, sharp edges (and stairs on slopes).",
            );
            mono(ui, "x ← solve (MᵀM + ρL) x = MᵀM y + ρ B (z − u)");
            mono(ui, "z ← soft_threshold(Bᵀx + u, λ/ρ)   link by link");
            mono(ui, "u ← u + Bᵀx − z");
            let jumping = g
                .edges
                .iter()
                .filter(|e| {
                    (e.weight.sqrt() * (x[e.src as usize] - x[e.dst as usize])).abs() > tiny
                })
                .count();
            mono(
                ui,
                &format!("links with a jump: {jumping} of {}", g.edges.len()),
            );
            let mine: Vec<String> = g
                .edges
                .iter()
                .filter(|e| e.src as usize == i || e.dst as usize == i)
                .take(6)
                .map(|e| {
                    let j = e.weight.sqrt() * (x[e.src as usize] - x[e.dst as usize]);
                    format!(
                        "{}–{}: {}",
                        e.src,
                        e.dst,
                        if j.abs() > tiny { fmt(j) } else { "0".into() }
                    )
                })
                .collect();
            mono(ui, &format!("drone {i}'s links: {}", mine.join(", ")));
            self.loop_line(ui, m);
        }
        // Sparse total variation.
        if let Some(x) = self.map(Method::SparseTv) {
            let m = Method::SparseTv;
            section(ui, "Sparse total variation", m.color());
            mono(ui, "x̂ = argmin ½‖M(x − y)‖² + λ₁ ‖Bᵀx‖₁ + λ₂ ‖x‖₁");
            ui.weak(
                "Total variation plus a second |·| on the values themselves: the jumps are \
                 shrunk link by link (flat pieces) and the values drone by drone (most of the \
                 map exactly 0). Two helper variables, one per link and one per drone:",
            );
            mono(
                ui,
                "x  ← solve (MᵀM + ρ(L + I)) x = MᵀM y + ρB(z₁ − u₁) + ρ(z₂ − u₂)",
            );
            mono(ui, "z₁ ← soft_threshold(Bᵀx + u₁, λ₁/ρ)   link by link");
            mono(ui, "z₂ ← soft_threshold(x + u₂, λ₂/ρ)      drone by drone");
            mono(ui, "u₁ ← u₁ + Bᵀx − z₁,   u₂ ← u₂ + x − z₂");
            let zeros = x.iter().filter(|v| v.abs() <= tiny).count();
            let jumping = g
                .edges
                .iter()
                .filter(|e| {
                    (e.weight.sqrt() * (x[e.src as usize] - x[e.dst as usize])).abs() > tiny
                })
                .count();
            mono(ui, &format!("drones exactly 0: {zeros} of {n}"));
            mono(
                ui,
                &format!("links with a jump: {jumping} of {}", g.edges.len()),
            );
            mono(
                ui,
                &format!("λ₂ = {}   drone {i}: x̂ = {}", fmt(self.lambda2), fmt(x[i])),
            );
            self.loop_line(ui, m);
        }
        ui.add_space(4.0);
        ui.weak("ρ = 1 in every loop: it only sets the speed, never the answer.");
    }

    fn loop_line(&self, ui: &mut egui::Ui, m: Method) {
        if let Some(r) = &self.results[m.index()] {
            mono(
                ui,
                &format!(
                    "{} = {}, {} rounds, last gap {:.1e}",
                    m.param(),
                    fmt(self.params[m.index()]),
                    r.gaps.len(),
                    r.gaps.last().copied().unwrap_or(0.0)
                ),
            );
        }
    }

    /// Tikhonov's matrices for the selected drone's block, and its row solved by hand.
    fn math_tikhonov(&self, ui: &mut egui::Ui) {
        let m = Method::Tikhonov;
        let Some(x) = self.map(m) else {
            return;
        };
        let g = &self.world.graph;
        let n = self.n();
        let gamma = self.params[0];
        section(ui, "Tikhonov, computed", m.color());
        mono(
            ui,
            "x̂ = argmin ‖M(x − y)‖² + γ xᵀLx   →   (MᵀM + γL) x̂ = MᵀM y",
        );
        let win = self.window();
        let w = win.len();
        if n > w {
            ui.weak(format!(
                "{n} × {n} matrices: the block of drones {win:?} around the selected one."
            ));
        }
        let strong = ui.visuals().strong_text_color();
        let heads: Vec<(String, Color32)> = win
            .iter()
            .map(|&k| {
                (
                    k.to_string(),
                    if self.has_sensor[k] {
                        READING_COLOR
                    } else {
                        strong
                    },
                )
            })
            .collect();
        let one = vec![(String::new(), strong)];
        let focus = Focus {
            row: win.iter().position(|&k| k == self.selected),
            col: None,
            cell: None,
        };
        let l = g.laplacian();
        let sensor = |k: usize| if self.has_sensor[k] { 1.0 } else { 0.0 };
        let k_entry = |a: usize, b: usize| gamma * l[(a, b)] + if a == b { sensor(a) } else { 0.0 };
        let block = |f: &dyn Fn(usize, usize) -> f64| {
            let mut mat: Mat<f64> = Mat::zeros(w, w);
            for (r, &a) in win.iter().enumerate() {
                for (c, &b) in win.iter().enumerate() {
                    mat[(r, c)] = f(a, b);
                }
            }
            mat
        };
        let col = |f: &dyn Fn(usize) -> f64| {
            let mut mat: Mat<f64> = Mat::zeros(w, 1);
            for (r, &a) in win.iter().enumerate() {
                mat[(r, 0)] = f(a);
            }
            mat
        };
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.label("L");
                matrix_grid(
                    ui,
                    &block(&|a, b| l[(a, b)]),
                    &heads,
                    &heads,
                    focus,
                    L_COLOR,
                    None,
                );
            });
            ui.vertical(|ui| {
                ui.label(format!("K = MᵀM + {}·L", fmt(gamma)));
                matrix_grid(ui, &block(&k_entry), &heads, &heads, focus, m.color(), None);
            });
        });
        ui.horizontal_top(|ui| {
            for (title, mat, color) in [
                ("b = MᵀM y", col(&|a| sensor(a) * self.y[a]), READING_COLOR),
                ("x̂", col(&|a| x[a]), m.color()),
                ("truth", col(&|a| self.truth[a]), TRUTH_COLOR),
            ] {
                ui.vertical(|ui| {
                    ui.label(title);
                    matrix_grid(ui, &mat, &heads, &one, focus, color, None);
                });
                ui.add_space(6.0);
            }
        });
        let i = self.selected.min(n - 1);
        let (mut deg, mut pull) = (0.0, 0.0);
        for e in &g.edges {
            let (a, b) = (e.src as usize, e.dst as usize);
            if a == i || b == i {
                let j = if a == i { b } else { a };
                deg += e.weight;
                pull += e.weight * x[j];
            }
        }
        let mi = sensor(i);
        let by_hand = (mi * self.y[i] + gamma * pull) / (mi + gamma * deg);
        mono(ui, "x̂_i = (m_i y_i + γ Σ_j w_ij x̂_j) / (m_i + γ deg_i)");
        mono(
            ui,
            &format!(
                "x̂_{i} = ({}·{} + {}·{}) / ({} + {}·{}) = {}  {}",
                fmt(mi),
                fmt(self.y[i]),
                fmt(gamma),
                fmt(pull),
                fmt(mi),
                fmt(gamma),
                fmt(deg),
                fmt(by_hand),
                if (by_hand - x[i]).abs() < 1e-6 {
                    "✔ = solver"
                } else {
                    "✘"
                }
            ),
        );
        ui.weak(
            "A weighted average of the drone's reading and its neighbours: one solve, no loop.",
        );
    }

    // ------------------------------------------------------------------------ canvas

    fn value_range(&self) -> (f64, f64) {
        let n = self.n();
        let mut all: Vec<f64> = self.truth.clone();
        all.extend((0..n).filter(|&i| self.has_sensor[i]).map(|i| self.y[i]));
        for m in METHODS {
            if let Some(x) = self.map(m) {
                all.extend(x.iter().copied());
            }
        }
        range(&all)
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let n = self.n();
        let shown: Vec<Method> = METHODS
            .iter()
            .copied()
            .filter(|m| self.shown[m.index()])
            .collect();
        let cells: Vec<Rect> = match shown.len() {
            0 | 1 => vec![rect],
            2 => {
                let mid = rect.center().x;
                vec![
                    Rect::from_min_max(rect.min, pos2(mid - 2.0, rect.max.y)),
                    Rect::from_min_max(pos2(mid + 2.0, rect.min.y), rect.max),
                ]
            }
            _ => {
                let c = rect.center();
                vec![
                    Rect::from_min_max(rect.min, c - vec2(2.0, 2.0)),
                    Rect::from_min_max(pos2(c.x + 2.0, rect.min.y), pos2(rect.max.x, c.y - 2.0)),
                    Rect::from_min_max(pos2(rect.min.x, c.y + 2.0), pos2(c.x - 2.0, rect.max.y)),
                    Rect::from_min_max(c + vec2(2.0, 2.0), rect.max),
                ]
            }
        };
        let r = if n > 60 {
            if shown.len() > 2 { 4.0 } else { 6.0 }
        } else if n > 10 {
            if shown.len() > 2 { 5.0 } else { 7.0 }
        } else {
            12.0
        };
        let (hover, clicked) = {
            let pos = &self.world.pos;
            let fits: Vec<Fit> = cells
                .iter()
                .take(shown.len().max(1))
                .map(|c| Fit::new(pos, c.shrink2(vec2(14.0, 26.0))))
                .collect();
            let near = |p: Pos2| {
                fits.iter()
                    .flat_map(|f| (0..n).map(move |i| (i, f.screen(pos[i]).distance(p))))
                    .filter(|&(_, d)| d < r + 5.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
            };
            let hover = response.hover_pos().and_then(near);
            let clicked = if response.clicked() {
                response.interact_pointer_pos().and_then(near)
            } else {
                None
            };
            let (lo, hi) = self.value_range();
            let truth = &self.truth;
            let err_top = shown
                .iter()
                .filter_map(|&m| self.map(m))
                .flat_map(|x| (0..n).map(move |i| (x[i] - truth[i]).abs()))
                .fold(1e-12, f64::max);
            if shown.is_empty() {
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Tick a regularizer on the left.",
                    FontId::proportional(15.0),
                    visuals.text_color(),
                );
            }
            for ((m, cell), fit) in shown.iter().zip(&cells).zip(&fits) {
                let x = self.map(*m);
                let score = x.map_or(String::new(), |x| {
                    format!(" · NMSE {}", fmt(nmse(x, truth, |_| true)))
                });
                let title = format!(
                    "{} ({} = {}){score}",
                    m.name(),
                    m.param(),
                    fmt(self.params[m.index()])
                );
                self.draw_map(
                    &painter,
                    *cell,
                    fit,
                    &title,
                    m.color(),
                    x,
                    (lo, hi),
                    err_top,
                    r,
                    &visuals,
                );
                for (k, color) in [
                    (Some(self.selected), SELECT_COLOR),
                    (hover, visuals.text_color()),
                ] {
                    if let Some(k) = k.filter(|&k| k < n) {
                        painter.circle_stroke(fit.screen(pos[k]), r + 4.0, Stroke::new(2.0, color));
                    }
                }
            }
            if let Some(i) = hover {
                let mut lines = vec![format!(
                    "Drone {i}: truth {} · {}",
                    fmt(self.truth[i]),
                    if self.has_sensor[i] {
                        format!("reading {}", fmt(self.y[i]))
                    } else {
                        "no sensor".into()
                    }
                )];
                for m in &shown {
                    if let Some(x) = self.map(*m) {
                        lines.push(format!("{}: {}", m.name(), fmt(x[i])));
                    }
                }
                response.on_hover_text_at_pointer(lines.join("\n"));
            }
            (hover, clicked)
        };
        self.hover = hover;
        if let Some(i) = clicked {
            match self.tool {
                Tool::Select => self.selected = i,
                Tool::ToggleSensor => {
                    self.has_sensor[i] = !self.has_sensor[i];
                    self.version += 1;
                    self.tuning = None;
                    self.scenario = Scenario::Custom;
                    self.selected = i;
                    self.solve();
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_map(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        fit: &Fit,
        title: &str,
        color: Color32,
        values: Option<&[f64]>,
        (lo, hi): (f64, f64),
        err_top: f64,
        r: f32,
        visuals: &egui::Visuals,
    ) {
        let text = visuals.text_color();
        let n = self.n();
        let at = |i: usize| fit.screen(self.world.pos[i]);
        let scale = |v: f64| ramp((v - lo) / (hi - lo).max(1e-12));
        painter.rect_stroke(
            rect,
            2.0,
            Stroke::new(1.0, color.gamma_multiply(0.5)),
            egui::StrokeKind::Inside,
        );
        for e in &self.world.graph.edges {
            painter.line_segment(
                [at(e.src as usize), at(e.dst as usize)],
                Stroke::new(1.0, text.gamma_multiply(0.15)),
            );
        }
        let lost = sensorless(&self.world.graph, &self.has_sensor);
        for i in 0..n {
            let c = at(i);
            let fill = match self.view {
                View::Truth => Some(scale(self.truth[i])),
                View::Readings => self.has_sensor[i].then(|| scale(self.y[i])),
                View::Estimate => values.map(|x| scale(x[i])),
                View::Error => values.map(|x| {
                    let e = x[i] - self.truth[i];
                    let t = (e.abs() / err_top) as f32;
                    if e >= 0.0 {
                        mix(visuals.faint_bg_color, HIGH, t)
                    } else {
                        mix(visuals.faint_bg_color, LOW, t)
                    }
                }),
            };
            match fill {
                Some(f) => {
                    painter.circle_filled(c, r, f);
                }
                None => {
                    painter.circle_stroke(c, r, Stroke::new(1.0, text.gamma_multiply(0.4)));
                }
            }
            if self.has_sensor[i] && n <= 60 {
                painter.circle_stroke(c, r + 1.5, Stroke::new(1.5, READING_COLOR));
            }
            if lost.contains(&i) {
                let k = r * 0.8;
                let cross = Stroke::new(2.0, WARN_COLOR);
                painter.line_segment([c - vec2(k, k), c + vec2(k, k)], cross);
                painter.line_segment([c + vec2(-k, k), c + vec2(k, -k)], cross);
            }
        }
        painter.text(
            rect.center_top() + vec2(0.0, 5.0),
            Align2::CENTER_TOP,
            title,
            FontId::proportional(13.0),
            color,
        );
    }

    // ----------------------------------------------------------------------- profile

    fn profile(&self, ui: &mut egui::Ui) {
        let path = self.kind.is_path();
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(if path {
                    "Along the path"
                } else {
                    "Each map against the truth (on the diagonal = perfect)"
                })
                .strong(),
            );
            ui.colored_label(TRUTH_COLOR, "■ truth");
            ui.colored_label(READING_COLOR, "● readings");
            for m in METHODS {
                if self.map(m).is_some() {
                    ui.colored_label(m.color(), format!("■ {}", m.name()));
                }
            }
        });
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let text = visuals.text_color();
        let plot = Rect::from_min_max(rect.min + vec2(46.0, 6.0), rect.max - vec2(10.0, 16.0));
        painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
        let n = self.n();
        if n == 0 {
            return;
        }
        let (lo, hi) = self.value_range();
        let pad = 0.06 * (hi - lo);
        let (lo, hi) = (lo - pad, hi + pad);
        let fy = |v: f64| plot.bottom() - plot.height() * ((v - lo) / (hi - lo)) as f32;
        for k in 0..=4 {
            let v = lo + (hi - lo) * k as f64 / 4.0;
            painter.line_segment(
                [pos2(plot.left(), fy(v)), pos2(plot.right(), fy(v))],
                Stroke::new(1.0, text.gamma_multiply(0.1)),
            );
            painter.text(
                pos2(plot.left() - 4.0, fy(v)),
                Align2::RIGHT_CENTER,
                fmt(v),
                FontId::proportional(10.0),
                text,
            );
        }
        if path {
            let fx = |i: usize| plot.left() + plot.width() * (i as f32 / (n - 1).max(1) as f32);
            painter.add(Shape::line(
                (0..n).map(|i| pos2(fx(i), fy(self.truth[i]))).collect(),
                Stroke::new(3.0, TRUTH_COLOR),
            ));
            for i in (0..n).filter(|&i| self.has_sensor[i]) {
                painter.circle_filled(pos2(fx(i), fy(self.y[i])), 3.0, READING_COLOR);
            }
            for m in METHODS {
                if let Some(x) = self.map(m) {
                    painter.add(Shape::line(
                        (0..n).map(|i| pos2(fx(i), fy(x[i]))).collect(),
                        Stroke::new(2.0, m.color()),
                    ));
                }
            }
            painter.circle_stroke(
                pos2(fx(self.selected.min(n - 1)), plot.bottom() - 4.0),
                4.0,
                Stroke::new(2.0, SELECT_COLOR),
            );
        } else {
            let fx = |v: f64| plot.left() + plot.width() * ((v - lo) / (hi - lo)) as f32;
            painter.line_segment(
                [pos2(fx(lo), fy(lo)), pos2(fx(hi), fy(hi))],
                Stroke::new(1.0, TRUTH_COLOR),
            );
            for m in METHODS {
                if let Some(x) = self.map(m) {
                    for (i, &xi) in x.iter().enumerate() {
                        painter.circle_filled(pos2(fx(self.truth[i]), fy(xi)), 2.5, m.color());
                    }
                }
            }
            painter.text(
                plot.center_bottom() + vec2(0.0, 2.0),
                Align2::CENTER_TOP,
                "truth x →",
                FontId::proportional(10.0),
                text,
            );
        }
    }
}

/// Lx as a list, straight from the links: Σ_j w_ij (x_i − x_j).
fn times_l(g: &EdgeList, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for e in &g.edges {
        let (a, b) = (e.src as usize, e.dst as usize);
        let d = e.weight * (x[a] - x[b]);
        out[a] += d;
        out[b] -= d;
    }
    out
}

// ---------------------------------------------------------------------------- drawing

fn section(ui: &mut egui::Ui, title: &str, color: Color32) {
    ui.add_space(8.0);
    ui.label(RichText::new(title).strong().color(color));
}

fn mono(ui: &mut egui::Ui, t: &str) {
    ui.label(RichText::new(t).monospace());
}

/// Every method's tuning curve on one log-x plot, each best marked.
fn tuning_plot(ui: &mut egui::Ui, t: &Tuning) {
    let size = vec2(ui.available_width().min(390.0), 160.0);
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let plot = Rect::from_min_max(rect.min + vec2(44.0, 4.0), rect.max - vec2(6.0, 16.0));
    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let values: Vec<f64> = t
        .curves
        .iter()
        .flatten()
        .map(|p| p.error)
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return;
    }
    let top = values.iter().copied().fold(0.0, f64::max).max(1e-12) * 1.05;
    let (g0, g1) = (PARAM_MIN.log10(), PARAM_MAX.log10());
    let fx = |g: f64| plot.left() + plot.width() * ((g.log10() - g0) / (g1 - g0)) as f32;
    let fy = |v: f64| plot.bottom() - plot.height() * (v / top) as f32;
    for d in (g0 as i32)..=(g1 as i32) {
        let x = fx(10f64.powi(d));
        painter.line_segment(
            [pos2(x, plot.top()), pos2(x, plot.bottom())],
            Stroke::new(1.0, text.gamma_multiply(0.08)),
        );
        painter.text(
            pos2(x, plot.bottom() + 2.0),
            Align2::CENTER_TOP,
            fmt(10f64.powi(d)),
            FontId::proportional(10.0),
            text,
        );
    }
    for k in 0..=2 {
        let v = top * k as f64 / 2.0;
        painter.text(
            pos2(plot.left() - 4.0, fy(v)),
            Align2::RIGHT_CENTER,
            fmt(v),
            FontId::proportional(10.0),
            text,
        );
    }
    for m in METHODS {
        let curve = &t.curves[m.index()];
        let pts: Vec<Pos2> = curve
            .iter()
            .filter(|p| p.error.is_finite())
            .map(|p| pos2(fx(p.param), fy(p.error)))
            .collect();
        painter.add(Shape::line(pts, Stroke::new(2.0, m.color())));
        if let Some(b) = curve
            .iter()
            .filter(|p| p.error.is_finite())
            .min_by(|a, b| a.error.total_cmp(&b.error))
        {
            painter.circle_filled(pos2(fx(b.param), fy(b.error)), 4.0, m.color());
        }
    }
    painter.text(
        plot.right_top() + vec2(-4.0, 2.0),
        Align2::RIGHT_TOP,
        if t.by_truth {
            "NMSE vs parameter"
        } else {
            "held-out error vs parameter"
        },
        FontId::proportional(10.0),
        text,
    );
}

/// The ADMM gaps per round, log scale, one curve per loop.
fn convergence_plot(ui: &mut egui::Ui, loops: &[(Method, &Vec<f64>)]) {
    let size = vec2(ui.available_width().min(390.0), 120.0);
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let plot = Rect::from_min_max(rect.min + vec2(44.0, 4.0), rect.max - vec2(6.0, 16.0));
    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let logs: Vec<f64> = loops
        .iter()
        .flat_map(|(_, g)| g.iter().map(|v| v.max(1e-16).log10()))
        .collect();
    if logs.is_empty() {
        return;
    }
    let lo = logs.iter().copied().fold(f64::MAX, f64::min).floor();
    let hi = logs
        .iter()
        .copied()
        .fold(f64::MIN, f64::max)
        .ceil()
        .max(lo + 1.0);
    let rounds = loops.iter().map(|(_, g)| g.len()).max().unwrap_or(1).max(2);
    let fx = |k: usize| plot.left() + plot.width() * (k as f32 / (rounds - 1) as f32);
    let fy = |l: f64| plot.bottom() - plot.height() * ((l - lo) / (hi - lo)) as f32;
    for d in (lo as i32)..=(hi as i32) {
        let y = fy(d as f64);
        painter.line_segment(
            [pos2(plot.left(), y), pos2(plot.right(), y)],
            Stroke::new(1.0, text.gamma_multiply(0.08)),
        );
        painter.text(
            pos2(plot.left() - 4.0, y),
            Align2::RIGHT_CENTER,
            format!("1e{d}"),
            FontId::proportional(10.0),
            text,
        );
    }
    for (m, gaps) in loops {
        painter.add(Shape::line(
            gaps.iter()
                .enumerate()
                .map(|(k, v)| pos2(fx(k), fy(v.max(1e-16).log10())))
                .collect(),
            Stroke::new(2.0, m.color()),
        ));
    }
    painter.text(
        plot.right_bottom() + vec2(0.0, 2.0),
        Align2::RIGHT_TOP,
        format!("round {rounds}"),
        FontId::proportional(10.0),
        text,
    );
}

fn range(v: &[f64]) -> (f64, f64) {
    let lo = v
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::MAX, f64::min);
    let hi = v
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::MIN, f64::max);
    if !(lo.is_finite() && hi.is_finite()) {
        (0.0, 1.0)
    } else if hi > lo {
        (lo, hi)
    } else {
        (lo - 0.5, hi + 0.5)
    }
}

fn ramp(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        mix(LOW, MID, t * 2.0)
    } else {
        mix(MID, HIGH, (t - 0.5) * 2.0)
    }
}

/// World ↔ screen: fits the drones, same scale on both axes (a path gets some height).
struct Fit {
    min: [f64; 2],
    scale: f64,
    offset: egui::Vec2,
}

impl Fit {
    fn new(pos: &[[f64; 2]], rect: Rect) -> Fit {
        let (lo, hi) = bounds(pos);
        let span = [(hi[0] - lo[0]).max(1e-9), (hi[1] - lo[1]).max(1.0)];
        let scale = (rect.width() as f64 / span[0]).min(rect.height() as f64 / span[1]);
        let used = vec2((span[0] * scale) as f32, ((hi[1] - lo[1]) * scale) as f32);
        Fit {
            min: lo,
            scale,
            offset: rect.min.to_vec2() + (rect.size() - used) / 2.0,
        }
    }

    fn screen(&self, p: [f64; 2]) -> Pos2 {
        pos2(
            ((p[0] - self.min[0]) * self.scale) as f32,
            ((p[1] - self.min[1]) * self.scale) as f32,
        ) + self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn best_nmse(p: &Playground, m: Method) -> f64 {
        param_grid()
            .into_iter()
            .map(|q| {
                let x = reconstruct(
                    &p.world.graph,
                    &p.y,
                    &p.has_sensor,
                    m.regularizer(q, p.lambda2),
                    800,
                )
                .x;
                nmse(&x, &p.truth, |_| true)
            })
            .fold(f64::MAX, f64::min)
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "hundreds of ADMM solves: run with cargo test --release"
    )]
    fn each_scenario_is_won_by_its_regularizer() {
        for (s, winner) in [
            (Scenario::TikhonovCase, Method::Tikhonov),
            (Scenario::TrendCase, Method::TrendFiltering),
            (Scenario::TvCase, Method::TotalVariation),
            (Scenario::StvCase, Method::SparseTv),
        ] {
            let mut p = Playground::default();
            p.apply(s);
            let scores: Vec<(Method, f64)> =
                METHODS.iter().map(|&m| (m, best_nmse(&p, m))).collect();
            let best = scores.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
            assert_eq!(best, winner, "{}: {scores:?}", s.name());
        }
    }

    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "hundreds of ADMM solves: run with cargo test --release"
    )]
    fn the_presets_parameters_are_close_to_their_best() {
        // The scenario's own parameter for its winner is within 10 % of the tuned best.
        for (s, m) in [
            (Scenario::TikhonovCase, Method::Tikhonov),
            (Scenario::TrendCase, Method::TrendFiltering),
            (Scenario::TvCase, Method::TotalVariation),
            (Scenario::StvCase, Method::SparseTv),
        ] {
            let mut p = Playground::default();
            p.apply(s);
            let now = nmse(p.map(m).unwrap(), &p.truth, |_| true);
            assert!(now <= 1.1 * best_nmse(&p, m), "{}: {now}", s.name());
        }
    }

    #[test]
    fn tuning_finds_a_best_for_every_method_both_ways() {
        let mut p = Playground::default();
        p.apply(Scenario::TrendCase); // path of 40: cheap
        p.iterations = 30;
        for by_truth in [true, false] {
            p.tune(by_truth);
            let t = p.tuning.as_ref().unwrap();
            assert_eq!(t.by_truth, by_truth);
            for c in &t.curves {
                assert_eq!(c.len(), TUNE_STEPS);
                assert!(c.iter().any(|q| q.error.is_finite()));
            }
        }
    }

    #[test]
    fn switching_graph_mid_frame_never_mixes_sizes() {
        let mut p = Playground::default();
        for kind in GraphKind::ALL {
            p.set_graph(kind);
            for m in METHODS {
                assert_eq!(p.map(m).map_or(p.n(), <[f64]>::len), p.n());
            }
            let mut out = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                p.results(ui);
                p.math(ui);
            });
            out.textures_delta.clear();
        }
    }

    #[test]
    fn a_piece_with_no_sensor_gives_no_maps() {
        let mut p = Playground::default();
        p.apply(Scenario::Custom);
        p.set_graph(GraphKind::TinyPath);
        p.world.graph.edges.remove(1);
        p.has_sensor = vec![true, false, false, false, false];
        p.version += 1;
        p.solve();
        assert!(METHODS.iter().all(|&m| p.map(m).is_none()));
        assert_eq!(sensorless(&p.world.graph, &p.has_sensor), vec![2, 3, 4]);
    }

    #[test]
    fn everything_draws_without_panicking() {
        let mut p = Playground {
            iterations: 20,
            ..Default::default()
        };
        for s in Scenario::ALL {
            p.apply(s);
            for kind in [GraphKind::TinyPath, GraphKind::Grid] {
                if s == Scenario::Custom {
                    p.set_graph(kind);
                }
                for shown in [
                    [true; 4],
                    [true, false, false, false],
                    [true, true, false, false],
                ] {
                    p.shown = shown;
                    p.solve();
                    for view in [View::Estimate, View::Error, View::Readings] {
                        p.view = view;
                        let mut out =
                            egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                                p.settings(ui);
                                p.results(ui);
                                p.math(ui);
                                p.profile(ui);
                                p.canvas(ui);
                            });
                        out.textures_delta.clear();
                    }
                }
            }
        }
    }
}
