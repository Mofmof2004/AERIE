//! Bias–variance playground (Mission 3, experiment 1): how Tikhonov's γ trades noise for
//! blur, in theory and in simulation.
//!
//! Run with `cargo run --release -p aerie-viz --bin bias_variance_playground`.
//!
//! Denoising with a reading at every drone: y = x + noise (σ per drone), x̂ = H y with
//! H = (I + γL)⁻¹. Then
//! - bias = (H − I) x: the blur, the same on every draw; bias² = |Hx − x|²;
//! - variance = σ² trace(H Hᵀ): the noise that survives the smoothing;
//! - expected error = bias² + variance.
//!
//! - Left: the graph (a tiny path of 6 where H is written out, a path of 40, a 12 × 12 grid,
//!   the swarm's ε-graph), the truth, σ, γ, the number of simulated draws, and *Run
//!   experiment 1* (every γ of the plan's grid, draws at each).
//! - Centre: a map of the truth, the average map Hx, each drone's bias or variance, or one
//!   noisy draw and its reconstruction. Bottom: along the path, the truth, the average map,
//!   a ±2 standard deviation band and a few single reconstructions (or, off the path, the
//!   average map against the truth).
//! - Right: the curves (theory from `aerie_classic::regularize::bias_variance`, simulation as
//!   dots), the numbers at this γ, and the maths with live numbers, including the selected
//!   drone's row of H: the weights it averages the readings with.

use aerie_classic::regularize::{bias_variance, tikhonov};
use aerie_graph::{Edge, EdgeList};
use aerie_viz::grid::{Focus, fmt, matrix_grid, mix};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use faer::Mat;
use faer::linalg::solvers::Solve;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::snapshot::Snapshot;

const LOW: Color32 = Color32::from_rgb(45, 80, 170);
const MID: Color32 = Color32::from_rgb(240, 215, 90);
const HIGH: Color32 = Color32::from_rgb(205, 50, 40);
const TRUTH_COLOR: Color32 = Color32::from_rgb(160, 160, 160);
const READING_COLOR: Color32 = Color32::from_rgb(90, 170, 240);
const BIAS_COLOR: Color32 = Color32::from_rgb(235, 110, 70);
const VAR_COLOR: Color32 = Color32::from_rgb(90, 170, 240);
const TOTAL_COLOR: Color32 = Color32::from_rgb(255, 205, 60);
const SELECT_COLOR: Color32 = Color32::from_rgb(255, 205, 60);
/// The plan's γ grid: 20 values from 0.001 to 100, log-spaced.
const GAMMAS: usize = 20;
const GAMMA_MIN: f64 = 0.001;
const GAMMA_MAX: f64 = 100.0;
/// Single reconstructions drawn along the path.
const SAMPLES: usize = 6;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE bias–variance playground")
            .with_inner_size([1700.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE bias–variance playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

fn gamma_grid() -> Vec<f64> {
    (0..GAMMAS)
        .map(|k| {
            let t = k as f64 / (GAMMAS - 1) as f64;
            (GAMMA_MIN.ln() + t * (GAMMA_MAX.ln() - GAMMA_MIN.ln())).exp()
        })
        .collect()
}

fn gaussian(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ------------------------------------------------------------------------- graph, truth

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
            GraphKind::TinyPath => "Tiny path of 6 (H written out)",
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
        };
        match kind {
            GraphKind::TinyPath => path(6),
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
                    jamming: Some(snap.truth.clone()),
                }
            }
        }
    }

    fn n(&self) -> usize {
        self.pos.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Truth {
    Bump,
    Edge,
    Kinks,
    Jamming,
}

impl Truth {
    const ALL: [Truth; 4] = [Truth::Bump, Truth::Edge, Truth::Kinks, Truth::Jamming];

    fn name(self) -> &'static str {
        match self {
            Truth::Bump => "smooth bump",
            Truth::Edge => "sharp edge",
            Truth::Kinks => "slopes with kinks",
            Truth::Jamming => "true jamming power",
        }
    }

    fn make(self, world: &World) -> Vec<f64> {
        let (lo, hi) = bounds(&world.pos);
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-9);
        let unit: Vec<[f64; 2]> = world
            .pos
            .iter()
            .map(|p| [(p[0] - lo[0]) / span, (p[1] - lo[1]) / span])
            .collect();
        let w = unit.iter().map(|p| p[0]).fold(0.0, f64::max).max(1e-9);
        let h = unit.iter().map(|p| p[1]).fold(0.0, f64::max);
        match self {
            Truth::Bump => {
                let s = 0.2 * w.max(h);
                unit.iter()
                    .map(|p| {
                        (-((p[0] - w / 2.0).powi(2) + (p[1] - h / 2.0).powi(2)) / (2.0 * s * s))
                            .exp()
                    })
                    .collect()
            }
            Truth::Edge => unit
                .iter()
                .map(|p| if p[0] < w * 0.45 { 1.0 } else { 0.0 })
                .collect(),
            Truth::Kinks => unit
                .iter()
                .map(|p| {
                    let x = p[0] / w;
                    if x < 0.33 {
                        x * 3.0
                    } else if x < 0.66 {
                        1.0 - (x - 0.33) * 1.5
                    } else {
                        0.5 + (x - 0.66) * 1.5
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

// ---------------------------------------------------------------------------- the maths

/// H = (I + γL)⁻¹, the whole matrix (the same solve as `bias_variance`).
fn smoother(graph: &EdgeList, gamma: f64) -> Mat<f64> {
    let n = graph.n as usize;
    let l = graph.laplacian();
    let mut k: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            k[(i, j)] = gamma * l[(i, j)];
        }
        k[(i, i)] += 1.0;
    }
    k.partial_piv_lu().solve(Mat::<f64>::identity(n, n))
}

fn times(h: &Mat<f64>, v: &[f64]) -> Vec<f64> {
    (0..h.nrows())
        .map(|i| (0..h.ncols()).map(|j| h[(i, j)] * v[j]).sum())
        .collect()
}

/// Everything at one γ.
struct AtGamma {
    gamma: f64,
    h: Mat<f64>,
    /// Hx: the average reconstruction over infinitely many draws.
    mean: Vec<f64>,
    /// Each drone's bias (Hx − x)_i and variance σ² Σ_j H_ij².
    bias: Vec<f64>,
    var: Vec<f64>,
    /// The totals from `bias_variance`.
    bias2: f64,
    variance: f64,
}

/// (γ, bias², variance) from the theory.
type CurvePoint = (f64, f64, f64);
/// The simulation at the current γ: (data key, γ bits, draws, totals, maps, readings).
type SimNow = (u64, u64, usize, Measured, Vec<Vec<f64>>, Vec<Vec<f64>>);

/// What the simulation measured at one γ.
#[derive(Debug, Clone, Copy)]
struct Measured {
    gamma: f64,
    bias2: f64,
    variance: f64,
}

/// `draws` noisy copies y = x + σ·noise, each rebuilt by `tikhonov`: the measured bias² (of
/// their mean) and variance (their spread, summed over drones), and the first few maps.
fn simulate(
    graph: &EdgeList,
    x: &[f64],
    sigma: f64,
    gamma: f64,
    draws: usize,
    seed: u64,
) -> (Measured, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = x.len();
    let all = vec![true; n];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sum = vec![0.0; n];
    let mut sum_sq = vec![0.0; n];
    let mut samples = Vec::new();
    let mut readings = Vec::new();
    for d in 0..draws {
        let y: Vec<f64> = x.iter().map(|v| v + sigma * gaussian(&mut rng)).collect();
        let xh = tikhonov(graph, &y, &all, gamma);
        for i in 0..n {
            sum[i] += xh[i];
            sum_sq[i] += xh[i] * xh[i];
        }
        if d < SAMPLES {
            samples.push(xh);
            readings.push(y);
        }
    }
    let dn = draws as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / dn).collect();
    let bias2 = mean.iter().zip(x).map(|(m, t)| (m - t).powi(2)).sum();
    let variance = (0..n)
        .map(|i| (sum_sq[i] - dn * mean[i] * mean[i]) / (dn - 1.0).max(1.0))
        .sum();
    (
        Measured {
            gamma,
            bias2,
            variance,
        },
        samples,
        readings,
    )
}

// ---------------------------------------------------------------------------------- app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Truth,
    Mean,
    Bias,
    Variance,
    Draw,
    DrawMap,
}

impl View {
    const ALL: [View; 6] = [
        View::Truth,
        View::Mean,
        View::Bias,
        View::Variance,
        View::Draw,
        View::DrawMap,
    ];

    fn name(self) -> &'static str {
        match self {
            View::Truth => "truth x",
            View::Mean => "average map Hx",
            View::Bias => "bias (Hx − x)_i",
            View::Variance => "variance σ² Σ_j H_ij²",
            View::Draw => "one noisy draw y",
            View::DrawMap => "its map x̂ = H y",
        }
    }
}

struct Playground {
    kind: GraphKind,
    world: World,
    truth_kind: Truth,
    truth: Vec<f64>,
    /// Noise, as a share of the truth's spread.
    noise: f64,
    gamma: f64,
    draws: usize,
    seed: u64,
    version: u64,
    at: Option<(u64, u64, AtGamma)>,
    /// (key, the theory at every γ of the grid: (γ, bias², variance)).
    curve: Option<(u64, Vec<CurvePoint>)>,
    /// The simulation at the current γ: key, measured totals, sample maps, their readings.
    measured_now: Option<SimNow>,
    experiment: Option<(u64, Vec<Measured>)>,
    view: View,
    selected: usize,
    hover: Option<usize>,
}

impl Default for Playground {
    fn default() -> Self {
        let world = World::build(GraphKind::Path, 0);
        let mut p = Playground {
            kind: GraphKind::Path,
            world,
            truth_kind: Truth::Edge,
            truth: Vec::new(),
            noise: 0.4,
            gamma: 1.0,
            draws: 200,
            seed: 1,
            version: 0,
            at: None,
            curve: None,
            measured_now: None,
            experiment: None,
            view: View::Mean,
            selected: 20,
            hover: None,
        };
        p.rebuild();
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.update();
        egui::Panel::left("settings")
            .resizable(true)
            .default_size(330.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.settings(ui));
            });
        self.update();
        egui::Panel::right("curves")
            .resizable(true)
            .default_size(560.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.right(ui));
            });
        egui::Panel::bottom("profile")
            .resizable(true)
            .default_size(250.0)
            .show(ui, |ui| self.profile(ui));
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl Playground {
    fn n(&self) -> usize {
        self.world.n()
    }

    /// σ itself: the noise share × the truth's spread.
    fn sigma(&self) -> f64 {
        let n = self.n().max(1) as f64;
        let mean = self.truth.iter().sum::<f64>() / n;
        let spread = (self.truth.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n)
            .sqrt()
            .max(1e-9);
        self.noise * spread
    }

    fn rebuild(&mut self) {
        if self.truth_kind == Truth::Jamming && self.kind != GraphKind::Swarm {
            self.truth_kind = Truth::Bump;
        }
        self.truth = self.truth_kind.make(&self.world);
        self.selected = self.selected.min(self.n().saturating_sub(1));
        self.version += 1;
        self.experiment = None;
        self.update();
    }

    fn set_graph(&mut self, kind: GraphKind) {
        self.kind = kind;
        self.world = World::build(kind, 0);
        if kind == GraphKind::Swarm {
            self.truth_kind = Truth::Jamming;
        }
        self.selected = self.n() / 2;
        self.rebuild();
    }

    /// Recomputes what depends on γ, σ and the data.
    fn update(&mut self) {
        let g = &self.world.graph;
        let sigma = self.sigma();
        let key = (self.version, self.gamma.to_bits() ^ sigma.to_bits());
        if self.at.as_ref().is_none_or(|(v, k, _)| (*v, *k) != key) {
            let h = smoother(g, self.gamma);
            let mean = times(&h, &self.truth);
            let bias: Vec<f64> = mean.iter().zip(&self.truth).map(|(m, t)| m - t).collect();
            let n = self.n();
            let var: Vec<f64> = (0..n)
                .map(|i| sigma * sigma * (0..n).map(|j| h[(i, j)].powi(2)).sum::<f64>())
                .collect();
            let (bias2, variance) = bias_variance(g, &self.truth, sigma, self.gamma);
            self.at = Some((
                key.0,
                key.1,
                AtGamma {
                    gamma: self.gamma,
                    h,
                    mean,
                    bias,
                    var,
                    bias2,
                    variance,
                },
            ));
        }
        let curve_key = self.version ^ sigma.to_bits();
        if self.curve.as_ref().is_none_or(|(k, _)| *k != curve_key) {
            let curve = gamma_grid()
                .into_iter()
                .map(|gm| {
                    let (b, v) = bias_variance(g, &self.truth, sigma, gm);
                    (gm, b, v)
                })
                .collect();
            self.curve = Some((curve_key, curve));
        }
        let sim_key = (
            self.version ^ sigma.to_bits(),
            self.gamma.to_bits(),
            self.draws,
        );
        if self
            .measured_now
            .as_ref()
            .is_none_or(|m| (m.0, m.1, m.2) != sim_key)
        {
            let (m, samples, readings) =
                simulate(g, &self.truth, sigma, self.gamma, self.draws, self.seed);
            self.measured_now = Some((sim_key.0, sim_key.1, sim_key.2, m, samples, readings));
        }
        if self
            .experiment
            .as_ref()
            .is_some_and(|(k, _)| *k != curve_key)
        {
            self.experiment = None;
        }
    }

    /// Experiment 1: `draws` noisy copies at every γ of the grid.
    fn run_experiment(&mut self) {
        let sigma = self.sigma();
        let points = gamma_grid()
            .into_iter()
            .map(|gm| {
                simulate(
                    &self.world.graph,
                    &self.truth,
                    sigma,
                    gm,
                    self.draws,
                    self.seed,
                )
                .0
            })
            .collect();
        self.experiment = Some((self.version ^ sigma.to_bits(), points));
    }

    fn at(&self) -> &AtGamma {
        &self.at.as_ref().expect("update() runs first").2
    }

    // ---------------------------------------------------------------------- settings

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Bias–variance playground");
        ui.weak(
            "Experiment 1: how Tikhonov's γ trades noise for blur. A noisy reading at every drone.",
        );
        ui.separator();
        let mut kind = self.kind;
        egui::ComboBox::from_id_salt("graph")
            .selected_text(kind.name())
            .width(280.0)
            .show_ui(ui, |ui| {
                for k in GraphKind::ALL {
                    ui.selectable_value(&mut kind, k, k.name());
                }
            });
        if kind != self.kind {
            self.set_graph(kind);
        }
        let mut changed = false;
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
        if changed {
            self.rebuild();
        }
        ui.add(egui::Slider::new(&mut self.noise, 0.0..=1.5).text("noise σ (× signal spread)"));
        ui.weak(format!("σ = {}", fmt(self.sigma())));
        ui.add(
            egui::Slider::new(&mut self.gamma, GAMMA_MIN..=GAMMA_MAX)
                .logarithmic(true)
                .text("γ"),
        );
        ui.add(
            egui::Slider::new(&mut self.draws, 20..=1000)
                .logarithmic(true)
                .text("simulated draws"),
        );
        ui.horizontal(|ui| {
            if ui.button("New noise").clicked() {
                self.seed += 1;
                self.measured_now = None;
                self.experiment = None;
            }
            if ui.button("▶ Run experiment 1").on_hover_text("The plan's 20 values of γ, the draws at each: measured bias², variance and total next to the theory.").clicked() {
                self.run_experiment();
            }
        });
        ui.separator();
        ui.label(RichText::new("The map shows").strong());
        for v in View::ALL {
            ui.radio_value(&mut self.view, v, v.name());
        }
        ui.weak("Click a drone to follow it in the maths.");
    }

    // ------------------------------------------------------------------- right panel

    fn right(&mut self, ui: &mut egui::Ui) {
        ui.heading("Noise against blur");
        let mut clicked = None;
        if let Some((_, curve)) = &self.curve {
            clicked = curves_plot(
                ui,
                curve,
                self.experiment.as_ref().map(|e| e.1.as_slice()),
                self.gamma,
            );
        }
        if let Some(g) = clicked {
            self.gamma = g;
            self.update();
        }
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(BIAS_COLOR, "━ bias²");
            ui.colored_label(VAR_COLOR, "━ variance");
            ui.colored_label(TOTAL_COLOR, "━ total");
            ui.weak("lines: theory, dots: simulation. Click the plot to set γ.");
        });
        let best = self.curve.as_ref().and_then(|(_, curve)| {
            curve
                .iter()
                .min_by(|a, b| (a.1 + a.2).total_cmp(&(b.1 + b.2)))
                .copied()
        });
        if let Some(best) = best {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Lowest expected error: γ = {} (total {})",
                    fmt(best.0),
                    fmt(best.1 + best.2)
                ));
                if ui.small_button("use").clicked() {
                    self.gamma = best.0;
                    self.update();
                }
            });
        }

        let a = self.at();
        let m = &self.measured_now.as_ref().unwrap().3;
        ui.add_space(6.0);
        ui.label(RichText::new(format!("At γ = {}", fmt(a.gamma))).strong());
        egui::Grid::new("now").striped(true).show(ui, |ui| {
            ui.label("");
            ui.label("theory");
            ui.label(format!("{} draws", self.draws));
            ui.end_row();
            ui.colored_label(BIAS_COLOR, "bias²");
            ui.label(fmt(a.bias2));
            ui.label(fmt(m.bias2));
            ui.end_row();
            ui.colored_label(VAR_COLOR, "variance");
            ui.label(fmt(a.variance));
            ui.label(fmt(m.variance));
            ui.end_row();
            ui.colored_label(TOTAL_COLOR, "total");
            ui.label(RichText::new(fmt(a.bias2 + a.variance)).strong());
            ui.label(fmt(m.bias2 + m.variance));
            ui.end_row();
        });
        ui.weak("The simulated bias² also carries a little noise of its own (variance ÷ draws): more draws, closer to the theory.");

        self.math(ui);
    }

    fn math(&self, ui: &mut egui::Ui) {
        let a = self.at();
        let n = self.n();
        let sigma = self.sigma();
        let x = &self.truth;
        section(ui, "1. One reading per drone, noisy");
        mono(ui, "y = x + ε,   ε_i independent, variance σ²");
        mono(ui, &format!("σ = {},  n = {n} drones", fmt(sigma)));
        section(ui, "2. Tikhonov is a fixed smoothing matrix H");
        mono(ui, "x̂ = (I + γL)⁻¹ y = H y");
        ui.weak("Each drone's x̂_i is a weighted average of all the readings: row i of H holds the weights (they add up to 1).");
        let i = self.selected.min(n.saturating_sub(1));
        let mut row: Vec<(usize, f64)> = (0..n).map(|j| (j, a.h[(i, j)])).collect();
        row.sort_by(|p, q| q.1.abs().total_cmp(&p.1.abs()));
        let shown: Vec<String> = row
            .iter()
            .take(6)
            .map(|(j, w)| format!("{}·y_{j}", fmt(*w)))
            .collect();
        mono(ui, &format!("x̂_{i} = {} + …", shown.join(" + ")));
        let row_sum: f64 = (0..n).map(|j| a.h[(i, j)]).sum();
        mono(
            ui,
            &format!("Σ_j H_{i}j = {}   (weights add up to 1)", fmt(row_sum)),
        );
        if n <= 8 {
            let heads: Vec<(String, Color32)> = (0..n)
                .map(|k| (k.to_string(), ui.visuals().strong_text_color()))
                .collect();
            ui.label("H (γ moves every entry):");
            matrix_grid(
                ui,
                &a.h,
                &heads,
                &heads,
                Focus {
                    row: Some(i),
                    col: None,
                    cell: None,
                },
                TOTAL_COLOR,
                Some("Σ"),
            );
        }
        section(ui, "3. On average: the bias (blur)");
        mono(ui, "E[x̂] = H E[y] = H x     bias = H x − x");
        mono(
            ui,
            &format!(
                "drone {i}: (Hx)_{i} = {} vs x_{i} = {}  →  bias {}",
                fmt(a.mean[i]),
                fmt(x[i]),
                fmt(a.bias[i])
            ),
        );
        mono(ui, &format!("bias² = Σ_i (Hx − x)_i² = {}", fmt(a.bias2)));
        ui.weak("The same on every draw: noise can't fix it. It grows with γ (more averaging, more blur).");
        section(ui, "4. Around the average: the variance (noise left)");
        mono(ui, "Cov(x̂) = H Cov(y) Hᵀ = σ² H Hᵀ");
        mono(ui, "variance = trace(σ² H Hᵀ) = σ² Σ_i Σ_j H_ij²");
        let sq: f64 = (0..n).map(|j| a.h[(i, j)].powi(2)).sum();
        mono(
            ui,
            &format!(
                "drone {i}: σ² Σ_j H_{i}j² = {} × {} = {}",
                fmt(sigma * sigma),
                fmt(sq),
                fmt(a.var[i])
            ),
        );
        mono(
            ui,
            &format!("variance = Σ over drones = {}", fmt(a.variance)),
        );
        ui.weak("Averaging many readings with small weights shrinks the noise: Σ_j H_ij² falls as γ spreads the weights out.");
        section(ui, "5. The expected error, and the two ends");
        mono(
            ui,
            &format!(
                "E|x̂ − x|² = bias² + variance = {} + {} = {}",
                fmt(a.bias2),
                fmt(a.variance),
                fmt(a.bias2 + a.variance)
            ),
        );
        let mean = x.iter().sum::<f64>() / n.max(1) as f64;
        mono(
            ui,
            &format!(
                "γ = 0 :  H = I, bias² 0, variance n σ² = {}",
                fmt(n as f64 * sigma * sigma)
            ),
        );
        mono(
            ui,
            &format!(
                "γ → ∞ :  everyone at the mean, bias² {}, variance σ² = {}",
                fmt(x.iter().map(|v| (v - mean).powi(2)).sum::<f64>()),
                fmt(sigma * sigma)
            ),
        );
        ui.weak("Between the two, the total dips: the best γ is where adding blur stops buying less noise.");
    }

    // ------------------------------------------------------------------------ canvas

    fn values(&self) -> Vec<f64> {
        let a = self.at();
        let m = self.measured_now.as_ref().unwrap();
        match self.view {
            View::Truth => self.truth.clone(),
            View::Mean => a.mean.clone(),
            View::Bias => a.bias.clone(),
            View::Variance => a.var.clone(),
            View::Draw => m.5[0].clone(),
            View::DrawMap => m.4[0].clone(),
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let n = self.n();
        let fit = Fit::new(&self.world.pos, rect.shrink2(vec2(24.0, 36.0)));
        let at = |i: usize| fit.screen(self.world.pos[i]);
        let r = if n > 60 {
            7.0
        } else if n > 10 {
            9.0
        } else {
            16.0
        };
        for e in &self.world.graph.edges {
            painter.line_segment(
                [at(e.src as usize), at(e.dst as usize)],
                Stroke::new(1.0, visuals.text_color().gamma_multiply(0.15)),
            );
        }
        let v = self.values();
        let (lo, hi) = match self.view {
            View::Bias => {
                let top = v.iter().map(|b| b.abs()).fold(1e-12, f64::max);
                (-top, top)
            }
            View::Variance => (0.0, v.iter().copied().fold(1e-12, f64::max)),
            _ => {
                let mut all = self.truth.clone();
                all.extend(self.at().mean.iter().copied());
                all.extend(self.measured_now.as_ref().unwrap().5[0].iter().copied());
                range(&all)
            }
        };
        for (i, &vi) in v.iter().enumerate() {
            let t = (vi - lo) / (hi - lo).max(1e-12);
            let fill = match self.view {
                View::Variance => mix(visuals.faint_bg_color, VAR_COLOR, t as f32),
                _ => ramp(t),
            };
            painter.circle_filled(at(i), r, fill);
            painter.circle_stroke(at(i), r, Stroke::new(1.0, Color32::from_black_alpha(140)));
            if n <= 8 {
                painter.text(
                    at(i),
                    Align2::CENTER_CENTER,
                    i.to_string(),
                    FontId::proportional(12.0),
                    Color32::BLACK,
                );
            }
        }
        painter.circle_stroke(
            at(self.selected.min(n - 1)),
            r + 5.0,
            Stroke::new(2.5, SELECT_COLOR),
        );
        painter.text(
            rect.center_top() + vec2(0.0, 8.0),
            Align2::CENTER_TOP,
            format!("{} — from {} to {}", self.view.name(), fmt(lo), fmt(hi)),
            FontId::proportional(13.0),
            visuals.text_color(),
        );
        let near = |p: Pos2| {
            (0..n)
                .map(|i| (i, at(i).distance(p)))
                .filter(|&(_, d)| d < r + 5.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        };
        self.hover = response.hover_pos().and_then(near);
        if let Some(i) = self.hover {
            let a = self.at();
            response.clone().on_hover_text_at_pointer(format!(
                "Drone {i}: truth {} · average map {} · bias {} · variance {}",
                fmt(self.truth[i]),
                fmt(a.mean[i]),
                fmt(a.bias[i]),
                fmt(a.var[i])
            ));
        }
        if response.clicked()
            && let Some(i) = response.interact_pointer_pos().and_then(near)
        {
            self.selected = i;
        }
    }

    // ----------------------------------------------------------------------- profile

    fn profile(&self, ui: &mut egui::Ui) {
        let path = self.kind.is_path();
        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new(if path {
                    "Along the path"
                } else {
                    "Average map against the truth (±2 standard deviations)"
                })
                .strong(),
            );
            ui.colored_label(TRUTH_COLOR, "━ truth");
            ui.colored_label(TOTAL_COLOR, "━ average map Hx");
            ui.colored_label(VAR_COLOR, "▒ ±2 std (variance)");
            if path {
                ui.colored_label(TRUTH_COLOR.gamma_multiply(0.6), "─ single reconstructions");
                ui.colored_label(READING_COLOR, "● one draw's readings");
            }
        });
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let text = visuals.text_color();
        let plot = Rect::from_min_max(rect.min + vec2(46.0, 6.0), rect.max - vec2(10.0, 16.0));
        painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
        let n = self.n();
        let a = self.at();
        let m = self.measured_now.as_ref().unwrap();
        let sd: Vec<f64> = a.var.iter().map(|v| v.sqrt()).collect();
        let mut all = self.truth.clone();
        all.extend((0..n).map(|i| a.mean[i] + 2.0 * sd[i]));
        all.extend((0..n).map(|i| a.mean[i] - 2.0 * sd[i]));
        if path {
            all.extend(m.5[0].iter().copied());
        }
        let (lo, hi) = range(&all);
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
            // The band: Hx ± 2 standard deviations.
            let mut band: Vec<Pos2> = (0..n)
                .map(|i| pos2(fx(i), fy(a.mean[i] + 2.0 * sd[i])))
                .collect();
            band.extend(
                (0..n)
                    .rev()
                    .map(|i| pos2(fx(i), fy(a.mean[i] - 2.0 * sd[i]))),
            );
            for i in 0..n.saturating_sub(1) {
                let quad = vec![
                    pos2(fx(i), fy(a.mean[i] + 2.0 * sd[i])),
                    pos2(fx(i + 1), fy(a.mean[i + 1] + 2.0 * sd[i + 1])),
                    pos2(fx(i + 1), fy(a.mean[i + 1] - 2.0 * sd[i + 1])),
                    pos2(fx(i), fy(a.mean[i] - 2.0 * sd[i])),
                ];
                painter.add(Shape::convex_polygon(
                    quad,
                    VAR_COLOR.gamma_multiply(0.18),
                    Stroke::NONE,
                ));
            }
            for s in &m.4 {
                painter.add(Shape::line(
                    (0..n).map(|i| pos2(fx(i), fy(s[i]))).collect(),
                    Stroke::new(1.0, TRUTH_COLOR.gamma_multiply(0.5)),
                ));
            }
            for i in 0..n {
                painter.circle_filled(pos2(fx(i), fy(m.5[0][i])), 2.5, READING_COLOR);
            }
            painter.add(Shape::line(
                (0..n).map(|i| pos2(fx(i), fy(self.truth[i]))).collect(),
                Stroke::new(2.5, TRUTH_COLOR),
            ));
            painter.add(Shape::line(
                (0..n).map(|i| pos2(fx(i), fy(a.mean[i]))).collect(),
                Stroke::new(2.5, TOTAL_COLOR),
            ));
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
            for (i, &sdi) in sd.iter().enumerate() {
                let x = fx(self.truth[i]);
                painter.line_segment(
                    [
                        pos2(x, fy(a.mean[i] - 2.0 * sdi)),
                        pos2(x, fy(a.mean[i] + 2.0 * sdi)),
                    ],
                    Stroke::new(1.0, VAR_COLOR.gamma_multiply(0.5)),
                );
                painter.circle_filled(pos2(x, fy(a.mean[i])), 2.5, TOTAL_COLOR);
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

// ---------------------------------------------------------------------------- drawing

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(title).strong().color(SELECT_COLOR));
}

fn mono(ui: &mut egui::Ui, t: &str) {
    ui.label(RichText::new(t).monospace());
}

/// Bias², variance and total against γ (log x): theory as lines, simulation as dots.
/// Returns the γ clicked, if any.
fn curves_plot(
    ui: &mut egui::Ui,
    curve: &[(f64, f64, f64)],
    measured: Option<&[Measured]>,
    gamma: f64,
) -> Option<f64> {
    let size = vec2(ui.available_width().min(540.0), 260.0);
    let (response, painter) = ui.allocate_painter(size, Sense::click());
    let rect = response.rect;
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let plot = Rect::from_min_max(rect.min + vec2(50.0, 6.0), rect.max - vec2(8.0, 18.0));
    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let mut top = curve.iter().map(|c| c.1 + c.2).fold(1e-12, f64::max);
    if let Some(ms) = measured {
        top = ms.iter().map(|m| m.bias2 + m.variance).fold(top, f64::max);
    }
    let top = top * 1.05;
    let (g0, g1) = (GAMMA_MIN.log10(), GAMMA_MAX.log10());
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
    for k in 0..=4 {
        let v = top * k as f64 / 4.0;
        painter.text(
            pos2(plot.left() - 4.0, fy(v)),
            Align2::RIGHT_CENTER,
            fmt(v),
            FontId::proportional(10.0),
            text,
        );
    }
    for (color, f) in [
        (
            BIAS_COLOR,
            (|c: &(f64, f64, f64)| c.1) as fn(&(f64, f64, f64)) -> f64,
        ),
        (VAR_COLOR, |c| c.2),
        (TOTAL_COLOR, |c| c.1 + c.2),
    ] {
        painter.add(Shape::line(
            curve.iter().map(|c| pos2(fx(c.0), fy(f(c)))).collect(),
            Stroke::new(2.5, color),
        ));
    }
    if let Some(best) = curve
        .iter()
        .min_by(|a, b| (a.1 + a.2).total_cmp(&(b.1 + b.2)))
    {
        painter.circle_filled(pos2(fx(best.0), fy(best.1 + best.2)), 5.0, TOTAL_COLOR);
    }
    if let Some(ms) = measured {
        for m in ms {
            painter.circle_stroke(
                pos2(fx(m.gamma), fy(m.bias2)),
                3.5,
                Stroke::new(1.5, BIAS_COLOR),
            );
            painter.circle_stroke(
                pos2(fx(m.gamma), fy(m.variance)),
                3.5,
                Stroke::new(1.5, VAR_COLOR),
            );
            painter.circle_stroke(
                pos2(fx(m.gamma), fy(m.bias2 + m.variance)),
                3.5,
                Stroke::new(1.5, TOTAL_COLOR),
            );
        }
    }
    let x = fx(gamma);
    painter.line_segment(
        [pos2(x, plot.top()), pos2(x, plot.bottom())],
        Stroke::new(1.5, SELECT_COLOR),
    );
    painter.text(
        plot.center_bottom() + vec2(0.0, 10.0),
        Align2::CENTER_TOP,
        "γ (log scale)",
        FontId::proportional(10.0),
        text,
    );
    if response.clicked()
        && let Some(p) = response.interact_pointer_pos()
    {
        let t = ((p.x - plot.left()) / plot.width()).clamp(0.0, 1.0) as f64;
        return Some(10f64.powf(g0 + t * (g1 - g0)));
    }
    None
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

    #[test]
    fn the_per_drone_numbers_add_up_to_bias_variance() {
        let p = Playground::default();
        let a = p.at();
        let b: f64 = a.bias.iter().map(|v| v * v).sum();
        let v: f64 = a.var.iter().sum();
        assert!((b - a.bias2).abs() < 1e-9 * a.bias2.max(1.0));
        assert!((v - a.variance).abs() < 1e-9 * a.variance.max(1.0));
        // Every row of H adds up to 1 (constants pass through untouched).
        for i in 0..p.n() {
            let s: f64 = (0..p.n()).map(|j| a.h[(i, j)]).sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_theory_curve_dips_and_the_simulation_follows_it() {
        let mut p = Playground {
            draws: 300,
            ..Default::default()
        };
        p.run_experiment();
        let (_, curve) = p.curve.as_ref().unwrap();
        let totals: Vec<f64> = curve.iter().map(|c| c.1 + c.2).collect();
        let best = totals.iter().copied().fold(f64::MAX, f64::min);
        assert!(
            best < totals[0] && best < totals[GAMMAS - 1],
            "U-shaped: {totals:?}"
        );
        let (_, measured) = p.experiment.as_ref().unwrap();
        for (m, c) in measured.iter().zip(curve) {
            let theory = c.1 + c.2;
            assert!(
                (m.bias2 + m.variance - theory).abs() < 0.15 * theory + 0.02,
                "γ {}: {} vs {theory}",
                c.0,
                m.bias2 + m.variance
            );
        }
    }

    #[test]
    fn everything_draws_without_panicking() {
        let mut p = Playground {
            draws: 30,
            ..Default::default()
        };
        for kind in GraphKind::ALL {
            p.set_graph(kind);
            for view in View::ALL {
                p.view = view;
                p.update();
                let mut out = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                    p.settings(ui);
                    p.update();
                    p.right(ui);
                    p.profile(ui);
                    p.canvas(ui);
                });
                out.textures_delta.clear();
            }
        }
    }
}
