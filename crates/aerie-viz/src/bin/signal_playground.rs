//! Signal playground (Mission 3): how bumpy is a signal on a graph, and what does
//! smoothing do to it?
//!
//! Run with `cargo run --release -p aerie-viz --bin signal_playground`.
//!
//! - Left: the graph (a path of 30, a 12 × 12 grid, or a seed's 100-drone ε-graph from
//!   Mission 0), the signal (constant, ramp, bump, sharp edge, noise; on the swarm, the true
//!   jamming power and the noisy sensor readings), "add noise", a brush to paint values,
//!   smoothing steps x ← x − α L x, and *Pin* to keep a signal's numbers for comparison.
//! - Centre: the graph, drones coloured by x (or by their local variation), links by the
//!   size of their jump √w |x_i − x_j|. Below: the signal's profile (along the path, or
//!   against the horizontal position).
//! - Right: the measures (xᵀLx, the total variation Σ √w |x_i − x_j|, and xᵀLx / xᵀx),
//!   which links carry the most variation, and the pinned signals side by side.
//!
//! The measures come from `aerie_graph::EdgeList`: `variation` (xᵀLx), `incidence_matrix`
//! (the jumps Bᵀx, whose absolute values add up to the total variation) and `laplacian`
//! (the smoothing step).

use aerie_graph::{Edge, EdgeList};
use aerie_viz::grid::{fmt, mix};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::snapshot::Snapshot;

/// Colour ramp of the signal: low, middle, high.
const LOW: Color32 = Color32::from_rgb(45, 80, 170);
const MID: Color32 = Color32::from_rgb(240, 215, 90);
const HIGH: Color32 = Color32::from_rgb(205, 50, 40);
/// Links with a big jump.
const JUMP_COLOR: Color32 = Color32::from_rgb(235, 90, 80);
/// The brush's ring.
const BRUSH_COLOR: Color32 = Color32::from_rgb(255, 205, 60);
/// Drone marker radius, pixels.
const NODE_PX: f32 = 7.0;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE signal playground")
            .with_inner_size([1650.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE signal playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

// ------------------------------------------------------------------------------- graphs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphKind {
    Path,
    Grid,
    Swarm,
}

impl GraphKind {
    const ALL: [GraphKind; 3] = [GraphKind::Path, GraphKind::Grid, GraphKind::Swarm];

    fn name(self) -> &'static str {
        match self {
            GraphKind::Path => "Path of 30",
            GraphKind::Grid => "Grid 12 × 12",
            GraphKind::Swarm => "Swarm ε-graph (100 drones)",
        }
    }
}

/// A graph with positions, and the swarm's jamming field when it is the swarm.
struct World {
    kind: GraphKind,
    pos: Vec<[f64; 2]>,
    graph: EdgeList,
    /// Swarm only: true jamming power and the sensor readings at each drone.
    truth: Option<Vec<f64>>,
    readings: Option<Vec<f64>>,
}

impl World {
    fn build(kind: GraphKind, seed: u64) -> World {
        let link = |a: usize, b: usize| Edge {
            src: a.min(b) as u32,
            dst: a.max(b) as u32,
            weight: 1.0,
        };
        match kind {
            GraphKind::Path => {
                let n = 30;
                World {
                    kind,
                    pos: (0..n).map(|i| [i as f64, 0.0]).collect(),
                    graph: EdgeList {
                        n: n as u32,
                        edges: (0..n - 1).map(|i| link(i, i + 1)).collect(),
                    },
                    truth: None,
                    readings: None,
                }
            }
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
                    kind,
                    pos: (0..k * k)
                        .map(|i| [(i % k) as f64, (i / k) as f64])
                        .collect(),
                    graph: EdgeList {
                        n: (k * k) as u32,
                        edges,
                    },
                    truth: None,
                    readings: None,
                }
            }
            GraphKind::Swarm => {
                let snap = Snapshot::build(seed, &SimConfig::default());
                World {
                    kind,
                    pos: snap
                        .swarm
                        .iter()
                        .map(|d| [d.position.x, -d.position.y])
                        .collect(),
                    graph: snap.geometric.clone(),
                    truth: Some(snap.truth.clone()),
                    readings: Some(snap.readings.clone()),
                }
            }
        }
    }

    fn n(&self) -> usize {
        self.pos.len()
    }

    /// Positions rescaled to [0, 1] along x (and the same scale along y).
    fn unit(&self) -> Vec<[f64; 2]> {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &self.pos {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let span = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1e-9);
        self.pos
            .iter()
            .map(|p| [(p[0] - lo[0]) / span, (p[1] - lo[1]) / span])
            .collect()
    }
}

// ------------------------------------------------------------------------------ signals

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signal {
    Constant,
    Ramp,
    Bump,
    Edge,
    Noise,
    Truth,
    Readings,
}

impl Signal {
    const ALL: [Signal; 7] = [
        Signal::Constant,
        Signal::Ramp,
        Signal::Bump,
        Signal::Edge,
        Signal::Noise,
        Signal::Truth,
        Signal::Readings,
    ];

    fn name(self) -> &'static str {
        match self {
            Signal::Constant => "constant",
            Signal::Ramp => "ramp (left → right)",
            Signal::Bump => "bump",
            Signal::Edge => "sharp edge",
            Signal::Noise => "pure noise",
            Signal::Truth => "true jamming power",
            Signal::Readings => "sensor readings (noisy)",
        }
    }

    fn swarm_only(self) -> bool {
        matches!(self, Signal::Truth | Signal::Readings)
    }

    /// The signal on `world`, from 0 to 1 for the shapes.
    fn make(self, world: &World, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let unit = world.unit();
        let width = unit.iter().map(|p| p[0]).fold(0.0, f64::max);
        let height = unit.iter().map(|p| p[1]).fold(0.0, f64::max);
        match self {
            Signal::Constant => vec![1.0; world.n()],
            Signal::Ramp => unit.iter().map(|p| p[0] / width.max(1e-9)).collect(),
            Signal::Bump => {
                let (cx, cy) = (width / 2.0, height / 2.0);
                let s = 0.18 * width.max(height);
                unit.iter()
                    .map(|p| (-((p[0] - cx).powi(2) + (p[1] - cy).powi(2)) / (2.0 * s * s)).exp())
                    .collect()
            }
            Signal::Edge => unit
                .iter()
                .map(|p| if p[0] < width / 2.0 { 1.0 } else { 0.0 })
                .collect(),
            Signal::Noise => (0..world.n()).map(|_| gaussian(rng)).collect(),
            Signal::Truth => world.truth.clone().unwrap_or_else(|| vec![0.0; world.n()]),
            Signal::Readings => world
                .readings
                .clone()
                .unwrap_or_else(|| vec![0.0; world.n()]),
        }
    }
}

/// A standard normal number (Box–Muller).
fn gaussian(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random_range(1e-12..1.0);
    let u2: f64 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ----------------------------------------------------------------------------- measures

/// The numbers the right panel shows for a signal.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Measures {
    /// xᵀLx = Σ w (x_i − x_j)²: Tikhonov's penalty.
    l2: f64,
    /// Σ √w |x_i − x_j| = |Bᵀx|₁: trend filtering's penalty.
    tv: f64,
    /// xᵀx.
    energy: f64,
}

impl Measures {
    fn of(graph: &EdgeList, x: &[f64]) -> Measures {
        Measures {
            l2: graph.variation(x),
            tv: jumps(graph, x).iter().map(|j| j.abs()).sum(),
            energy: x.iter().map(|v| v * v).sum(),
        }
    }

    /// xᵀLx / xᵀx: bumpiness per unit of signal, comparable between signals of different
    /// sizes (0 = flat; at most the largest eigenvalue of L).
    fn rayleigh(&self) -> f64 {
        if self.energy > 0.0 {
            self.l2 / self.energy
        } else {
            0.0
        }
    }
}

/// Bᵀx: the jump √w (x_src − x_dst) on every link.
fn jumps(graph: &EdgeList, x: &[f64]) -> Vec<f64> {
    let b = graph.incidence_matrix();
    (0..b.ncols())
        .map(|e| (0..b.nrows()).map(|i| b[(i, e)] * x[i]).sum())
        .collect()
}

/// Each drone's local variation: Σ over its links w (x_i − x_j)². Half their sum is xᵀLx.
fn local_variation(graph: &EdgeList, x: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; graph.n as usize];
    for e in &graph.edges {
        let (a, b) = (e.src as usize, e.dst as usize);
        let c = e.weight * (x[a] - x[b]).powi(2);
        v[a] += c;
        v[b] += c;
    }
    v
}

/// The step size of x ← x − α L x: α = 1 / (2 · largest degree) keeps every eigenvalue of
/// I − αL between 0 and 1 (L's largest eigenvalue is at most twice the largest degree), so
/// each step only damps, never amplifies.
fn smoothing_alpha(graph: &EdgeList) -> f64 {
    let top = graph.degrees().into_iter().fold(0.0, f64::max);
    if top > 0.0 { 0.5 / top } else { 0.0 }
}

/// One smoothing step x ← x − α L x: each drone moves toward the weighted average of its
/// neighbours (heat diffusion).
fn smooth(graph: &EdgeList, x: &[f64], alpha: f64) -> Vec<f64> {
    let l = graph.laplacian();
    let n = x.len();
    (0..n)
        .map(|i| x[i] - alpha * (0..n).map(|j| l[(i, j)] * x[j]).sum::<f64>())
        .collect()
}

// ---------------------------------------------------------------------------------- app

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColorBy {
    Signal,
    LocalVariation,
}

/// A signal kept for comparison.
struct Pinned {
    name: String,
    m: Measures,
}

struct Playground {
    kind: GraphKind,
    seed: u64,
    world: World,
    signal: Signal,
    /// What has been done to the signal since the preset (noise, paint, smoothing).
    history: Vec<String>,
    x: Vec<f64>,
    noise: f64,
    noise_seed: u64,
    brush: bool,
    brush_value: f64,
    color_by: ColorBy,
    color_links: bool,
    steps: usize,
    pinned: Vec<Pinned>,
    hover: Option<usize>,
}

impl Default for Playground {
    fn default() -> Self {
        let world = World::build(GraphKind::Path, 0);
        let mut p = Playground {
            kind: GraphKind::Path,
            seed: 0,
            world,
            signal: Signal::Ramp,
            history: Vec::new(),
            x: Vec::new(),
            noise: 0.2,
            noise_seed: 1,
            brush: false,
            brush_value: 1.0,
            color_by: ColorBy::Signal,
            color_links: true,
            steps: 0,
            pinned: Vec::new(),
            hover: None,
        };
        p.set_signal(Signal::Ramp);
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::Panel::right("measures")
            .resizable(true)
            .default_size(460.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.measures(ui));
            });
        egui::Panel::bottom("profile")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| self.profile(ui));
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl Playground {
    fn set_graph(&mut self, kind: GraphKind) {
        self.kind = kind;
        self.world = World::build(kind, self.seed);
        if self.signal.swarm_only() && kind != GraphKind::Swarm {
            self.signal = Signal::Ramp;
        }
        self.set_signal(self.signal);
    }

    fn set_signal(&mut self, s: Signal) {
        self.signal = s;
        let mut rng = ChaCha8Rng::seed_from_u64(self.noise_seed);
        self.x = s.make(&self.world, &mut rng);
        self.history.clear();
        self.steps = 0;
    }

    fn add_noise(&mut self) {
        self.noise_seed += 1;
        let mut rng = ChaCha8Rng::seed_from_u64(self.noise_seed);
        for v in &mut self.x {
            *v += self.noise * gaussian(&mut rng);
        }
        self.history.push(format!("noise σ {}", fmt(self.noise)));
    }

    fn smooth_steps(&mut self, k: usize) {
        let alpha = smoothing_alpha(&self.world.graph);
        for _ in 0..k {
            self.x = smooth(&self.world.graph, &self.x, alpha);
        }
        self.steps += k;
    }

    /// A short name for the current signal, for the pinned table.
    fn describe(&self) -> String {
        let mut parts = vec![self.signal.name().to_string()];
        parts.extend(self.history.iter().cloned());
        if self.steps > 0 {
            parts.push(format!("{} smoothing steps", self.steps));
        }
        format!("{} · {}", self.world.kind.name(), parts.join(" + "))
    }

    // ---------------------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Signal playground");
        ui.weak("A signal = one number per drone. How bumpy is it on the graph?");
        ui.separator();

        ui.label(RichText::new("Graph").strong());
        let mut kind = self.kind;
        egui::ComboBox::from_id_salt("graph")
            .selected_text(kind.name())
            .width(240.0)
            .show_ui(ui, |ui| {
                for k in GraphKind::ALL {
                    ui.selectable_value(&mut kind, k, k.name());
                }
            });
        if kind != self.kind {
            self.set_graph(kind);
        }
        if self.kind == GraphKind::Swarm {
            ui.horizontal(|ui| {
                ui.label("Seed");
                let mut seed = self.seed;
                ui.add(egui::DragValue::new(&mut seed).range(0..=99));
                if seed != self.seed {
                    self.seed = seed;
                    self.set_graph(GraphKind::Swarm);
                }
            });
            ui.weak(
                "Mission 0's distance graph: drones within radio range, Gaussian weights \
                 (the graph Mission 3 rebuilds the jamming map on).",
            );
        }
        ui.label(format!(
            "{} drones, {} links",
            self.world.n(),
            self.world.graph.edges.len()
        ));
        ui.separator();

        ui.label(RichText::new("Signal").strong());
        ui.horizontal_wrapped(|ui| {
            for s in Signal::ALL {
                if s.swarm_only() && self.kind != GraphKind::Swarm {
                    continue;
                }
                if ui
                    .selectable_label(
                        self.signal == s && self.history.is_empty() && self.steps == 0,
                        s.name(),
                    )
                    .clicked()
                {
                    self.set_signal(s);
                }
            }
        });
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.noise, 0.0..=1.0).text("σ"));
            if ui.button("Add noise").clicked() {
                self.add_noise();
            }
        });
        ui.checkbox(
            &mut self.brush,
            "🖌 Brush: click or drag over drones to paint",
        );
        if self.brush {
            ui.add(egui::Slider::new(&mut self.brush_value, -2.0..=2.0).text("value"));
        }
        ui.separator();

        ui.label(RichText::new("Smooth").strong());
        ui.weak(format!(
            "x ← x − α L x, α = 1 / (2 · largest degree) = {}: each drone moves toward \
             the weighted average of its neighbours. Watch xᵀLx fall.",
            fmt(smoothing_alpha(&self.world.graph))
        ));
        ui.horizontal(|ui| {
            for k in [1, 10, 100] {
                if ui
                    .button(format!("{k} step{}", if k > 1 { "s" } else { "" }))
                    .clicked()
                {
                    self.smooth_steps(k);
                }
            }
        });
        ui.label(format!("{} steps so far", self.steps));
        if ui.button("Back to the preset signal").clicked() {
            self.set_signal(self.signal);
        }
        ui.separator();

        ui.label(RichText::new("Display").strong());
        ui.horizontal(|ui| {
            ui.label("Drones by");
            ui.selectable_value(&mut self.color_by, ColorBy::Signal, "x");
            ui.selectable_value(
                &mut self.color_by,
                ColorBy::LocalVariation,
                "local variation",
            );
        });
        ui.checkbox(&mut self.color_links, "Links coloured by their jump");
        ui.separator();
        if ui.button("📌 Pin this signal's numbers").clicked() {
            let m = Measures::of(&self.world.graph, &self.x);
            self.pinned.push(Pinned {
                name: self.describe(),
                m,
            });
        }
        if !self.pinned.is_empty() && ui.small_button("Clear the pins").clicked() {
            self.pinned.clear();
        }
    }

    // ------------------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let n = self.world.n();
        let fit = Fit::new(&self.world.pos, rect.shrink(30.0));
        let at = |i: usize| fit.screen(self.world.pos[i]);
        let r = if n > 60 { NODE_PX * 0.8 } else { NODE_PX * 1.3 };

        let near = |p: Pos2| {
            (0..n)
                .map(|i| (i, at(i).distance(p)))
                .filter(|&(_, d)| d < r + 5.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        };
        self.hover = response.hover_pos().and_then(near);
        if self.brush
            && (response.clicked() || response.dragged())
            && let Some(i) = response.interact_pointer_pos().and_then(near)
            && self.x[i] != self.brush_value
        {
            self.x[i] = self.brush_value;
            if self.history.last().map(String::as_str) != Some("painted") {
                self.history.push("painted".into());
            }
        }

        // Links, coloured by their jump.
        let j = jumps(&self.world.graph, &self.x);
        let top = j.iter().map(|v| v.abs()).fold(1e-12, f64::max);
        for (e, edge) in self.world.graph.edges.iter().enumerate() {
            let (a, b) = (at(edge.src as usize), at(edge.dst as usize));
            let t = (j[e].abs() / top) as f32;
            let (color, width) = if self.color_links {
                (mix(text.gamma_multiply(0.2), JUMP_COLOR, t), 1.0 + 3.5 * t)
            } else {
                (text.gamma_multiply(0.25), 1.0)
            };
            painter.line_segment([a, b], Stroke::new(width, color));
        }

        // Drones.
        let values = match self.color_by {
            ColorBy::Signal => self.x.clone(),
            ColorBy::LocalVariation => local_variation(&self.world.graph, &self.x),
        };
        let (lo, hi) = range(&values);
        for (i, &v) in values.iter().enumerate() {
            let c = at(i);
            painter.circle_filled(c, r, ramp((v - lo) / (hi - lo).max(1e-12)));
            painter.circle_stroke(c, r, Stroke::new(1.0, Color32::from_black_alpha(140)));
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(
                at(i),
                r + 3.0,
                Stroke::new(2.0, if self.brush { BRUSH_COLOR } else { text }),
            );
            let lv = local_variation(&self.world.graph, &self.x)[i];
            response.on_hover_text_at_pointer(format!(
                "Drone {i}: x = {} · local variation {}",
                fmt(self.x[i]),
                fmt(lv)
            ));
        }
        // Colour scale.
        let bar = Rect::from_min_size(rect.right_top() + vec2(-170.0, 10.0), vec2(150.0, 10.0));
        for k in 0..30 {
            let t = k as f64 / 29.0;
            let cell = Rect::from_min_size(
                bar.min + vec2(bar.width() * k as f32 / 30.0, 0.0),
                vec2(bar.width() / 30.0 + 0.5, bar.height()),
            );
            painter.rect_filled(cell, 0.0, ramp(t));
        }
        let what = match self.color_by {
            ColorBy::Signal => "x",
            ColorBy::LocalVariation => "local variation",
        };
        painter.text(
            bar.left_bottom() + vec2(0.0, 3.0),
            Align2::LEFT_TOP,
            fmt(lo),
            FontId::proportional(10.0),
            text,
        );
        painter.text(
            bar.right_bottom() + vec2(0.0, 3.0),
            Align2::RIGHT_TOP,
            fmt(hi),
            FontId::proportional(10.0),
            text,
        );
        painter.text(
            bar.center_bottom() + vec2(0.0, 3.0),
            Align2::CENTER_TOP,
            what,
            FontId::proportional(10.0),
            text,
        );
    }

    // -------------------------------------------------------------------- right panel

    fn measures(&mut self, ui: &mut egui::Ui) {
        let g = &self.world.graph;
        let m = Measures::of(g, &self.x);
        ui.heading("How bumpy is it?");
        ui.label(RichText::new(self.describe()).italics());
        ui.add_space(4.0);
        egui::Grid::new("measures").striped(true).show(ui, |ui| {
            ui.label(RichText::new("xᵀLx = Σ w (x_i − x_j)²").monospace());
            ui.label(RichText::new(fmt(m.l2)).strong());
            ui.end_row();
            ui.label(RichText::new("total variation = Σ √w |x_i − x_j|").monospace());
            ui.label(RichText::new(fmt(m.tv)).strong());
            ui.end_row();
            ui.label(RichText::new("xᵀx").monospace());
            ui.label(fmt(m.energy));
            ui.end_row();
            ui.label(RichText::new("xᵀLx / xᵀx").monospace());
            ui.label(fmt(m.rayleigh()));
            ui.end_row();
        });
        ui.weak(
            "xᵀLx squares each jump: one big jump costs far more than many small ones. \
             Tikhonov minimises it, so it would rather blur a sharp edge into a slope. The \
             total variation adds the jumps as they are: an edge costs the same as a slope \
             with the same rise, so trend filtering keeps edges. xᵀLx / xᵀx compares signals \
             of different sizes.",
        );

        ui.add_space(8.0);
        ui.label(RichText::new("Where the variation is").strong());
        let j = jumps(g, &self.x);
        let mut links: Vec<(usize, f64)> = g
            .edges
            .iter()
            .enumerate()
            .map(|(e, edge)| {
                (
                    e,
                    edge.weight * (self.x[edge.src as usize] - self.x[edge.dst as usize]).powi(2),
                )
            })
            .collect();
        links.sort_by(|a, b| b.1.total_cmp(&a.1));
        if m.l2 > 0.0 {
            let top5: f64 = links.iter().take(5).map(|l| l.1).sum();
            ui.weak(format!(
                "The 5 biggest links carry {:.0} % of xᵀLx ({} links in all).",
                100.0 * top5 / m.l2,
                links.len()
            ));
            egui::Grid::new("links").striped(true).show(ui, |ui| {
                for h in ["link", "w", "jump √w Δx", "w Δx²", "share"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for &(e, c) in links.iter().take(8) {
                    let edge = &g.edges[e];
                    ui.label(format!("{}–{}", edge.src, edge.dst));
                    ui.label(fmt(edge.weight));
                    ui.label(fmt(j[e]));
                    ui.label(fmt(c));
                    ui.label(format!("{:.1} %", 100.0 * c / m.l2));
                    ui.end_row();
                }
            });
        } else {
            ui.weak("No variation at all: every link joins two equal values.");
        }

        ui.add_space(8.0);
        ui.label(RichText::new("Pinned signals").strong());
        if self.pinned.is_empty() {
            ui.weak(
                "📌 Pin a signal to keep its numbers. Try on the path: pin the ramp, then \
                 the sharp edge. Same rise from 0 to 1: compare their xᵀLx and their total \
                 variation.",
            );
        } else {
            egui::Grid::new("pinned").striped(true).show(ui, |ui| {
                for h in ["signal", "xᵀLx", "total var.", "xᵀLx / xᵀx"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for p in &self.pinned {
                    ui.label(&p.name);
                    ui.label(fmt(p.m.l2));
                    ui.label(fmt(p.m.tv));
                    ui.label(fmt(p.m.rayleigh()));
                    ui.end_row();
                }
            });
        }
    }

    // ------------------------------------------------------------------ bottom panel

    fn profile(&self, ui: &mut egui::Ui) {
        let along_path = self.kind == GraphKind::Path;
        ui.label(
            RichText::new(if along_path {
                "The signal along the path (drone 0 → 29)"
            } else {
                "The signal against the horizontal position (each dot a drone)"
            })
            .strong(),
        );
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let text = visuals.text_color();
        let plot = Rect::from_min_max(rect.min + vec2(46.0, 6.0), rect.max - vec2(10.0, 18.0));
        painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
        let n = self.world.n();
        if n == 0 {
            return;
        }
        let xs: Vec<f64> = self.world.pos.iter().map(|p| p[0]).collect();
        let (x0, x1) = range(&xs);
        let (lo, hi) = range(&self.x);
        let pad = 0.08 * (hi - lo).max(1e-9);
        let (lo, hi) = (lo - pad, hi + pad);
        let at = |px: f64, v: f64| {
            pos2(
                plot.left() + plot.width() * ((px - x0) / (x1 - x0).max(1e-12)) as f32,
                plot.bottom() - plot.height() * ((v - lo) / (hi - lo).max(1e-12)) as f32,
            )
        };
        for k in 0..=4 {
            let v = lo + (hi - lo) * k as f64 / 4.0;
            let y = at(x0, v).y;
            painter.line_segment(
                [pos2(plot.left(), y), pos2(plot.right(), y)],
                Stroke::new(1.0, text.gamma_multiply(0.1)),
            );
            painter.text(
                pos2(plot.left() - 4.0, y),
                Align2::RIGHT_CENTER,
                fmt(v),
                FontId::proportional(10.0),
                text,
            );
        }
        if along_path {
            let pts: Vec<Pos2> = (0..n).map(|i| at(xs[i], self.x[i])).collect();
            painter.add(Shape::line(pts.clone(), Stroke::new(2.0, MID)));
            for (i, p) in pts.into_iter().enumerate() {
                painter.circle_filled(p, 3.5, ramp((self.x[i] - lo) / (hi - lo).max(1e-12)));
            }
        } else {
            for (&px, &v) in xs.iter().zip(&self.x) {
                let c = ramp((v - lo) / (hi - lo).max(1e-12));
                painter.circle_filled(at(px, v), 3.0, c);
            }
        }
        if let Some(i) = self.hover {
            painter.circle_stroke(at(xs[i], self.x[i]), 6.0, Stroke::new(2.0, text));
        }
    }
}

/// (min, max) of the values (0, 1 for none).
fn range(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 1.0);
    }
    let lo = v.iter().copied().fold(f64::MAX, f64::min);
    let hi = v.iter().copied().fold(f64::MIN, f64::max);
    if hi > lo {
        (lo, hi)
    } else {
        (lo - 0.5, hi + 0.5)
    }
}

/// Blue (0) → yellow (0.5) → red (1).
fn ramp(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        mix(LOW, MID, t * 2.0)
    } else {
        mix(MID, HIGH, (t - 0.5) * 2.0)
    }
}

/// World ↔ screen for the canvas: fits the drones, same scale on both axes.
struct Fit {
    min: [f64; 2],
    scale: f64,
    offset: egui::Vec2,
}

impl Fit {
    fn new(pos: &[[f64; 2]], rect: Rect) -> Fit {
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in pos {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        if pos.is_empty() {
            (lo, hi) = ([0.0; 2], [1.0; 2]);
        }
        // A path is flat: give it some height so the scale comes from its length.
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

    fn measures(kind: GraphKind, s: Signal) -> Measures {
        let w = World::build(kind, 0);
        let x = s.make(&w, &mut ChaCha8Rng::seed_from_u64(1));
        Measures::of(&w.graph, &x)
    }

    #[test]
    fn on_the_path_an_edge_and_a_ramp_have_the_same_total_variation_but_not_the_same_l2() {
        // Both rise by 1 overall. The ramp: 29 jumps of 1/29; the edge: one jump of 1.
        let ramp = measures(GraphKind::Path, Signal::Ramp);
        let edge = measures(GraphKind::Path, Signal::Edge);
        assert!((ramp.tv - 1.0).abs() < 1e-12 && (edge.tv - 1.0).abs() < 1e-12);
        assert!((ramp.l2 - 1.0 / 29.0).abs() < 1e-12);
        assert!((edge.l2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_signals_have_no_variation() {
        for kind in GraphKind::ALL {
            let m = measures(kind, Signal::Constant);
            assert_eq!((m.l2, m.tv), (0.0, 0.0), "{}", kind.name());
        }
    }

    #[test]
    fn local_variations_add_up_to_twice_x_l_x() {
        for kind in GraphKind::ALL {
            let w = World::build(kind, 0);
            let x = Signal::Noise.make(&w, &mut ChaCha8Rng::seed_from_u64(2));
            let total: f64 = local_variation(&w.graph, &x).iter().sum();
            assert!(
                (total - 2.0 * w.graph.variation(&x)).abs() < 1e-9,
                "{}",
                kind.name()
            );
        }
    }

    #[test]
    fn smoothing_lowers_the_variation_and_keeps_the_mean() {
        for kind in GraphKind::ALL {
            let w = World::build(kind, 0);
            let mut x = Signal::Noise.make(&w, &mut ChaCha8Rng::seed_from_u64(3));
            let alpha = smoothing_alpha(&w.graph);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (before, m0) = (w.graph.variation(&x), mean(&x));
            for _ in 0..20 {
                let next = smooth(&w.graph, &x, alpha);
                assert!(w.graph.variation(&next) <= w.graph.variation(&x) + 1e-12);
                x = next;
            }
            assert!(w.graph.variation(&x) < 0.5 * before, "{}", kind.name());
            assert!(
                (mean(&x) - m0).abs() < 1e-9,
                "{}: smoothing moved the mean",
                kind.name()
            );
        }
    }

    #[test]
    fn the_noisy_readings_are_bumpier_than_the_truth() {
        let truth = measures(GraphKind::Swarm, Signal::Truth);
        let readings = measures(GraphKind::Swarm, Signal::Readings);
        assert!(readings.rayleigh() > truth.rayleigh());
    }

    #[test]
    fn every_graph_and_signal_draws_without_panicking() {
        let mut p = Playground::default();
        for kind in GraphKind::ALL {
            p.set_graph(kind);
            for s in Signal::ALL {
                if s.swarm_only() && kind != GraphKind::Swarm {
                    continue;
                }
                p.set_signal(s);
                p.add_noise();
                p.smooth_steps(3);
                for color_by in [ColorBy::Signal, ColorBy::LocalVariation] {
                    p.color_by = color_by;
                    let mut out =
                        egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                            p.controls(ui);
                            p.measures(ui);
                            p.profile(ui);
                            p.canvas(ui);
                        });
                    out.textures_delta.clear();
                }
                p.pinned.push(Pinned {
                    name: p.describe(),
                    m: Measures::of(&p.world.graph, &p.x),
                });
            }
        }
    }
}
