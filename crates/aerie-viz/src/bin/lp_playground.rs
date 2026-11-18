//! Label propagation playground (Mission 1): place drones, choose which ones report their
//! class, and watch the labels spread over the comm graph round by round.
//!
//! Run with `cargo run --release -p aerie-viz --bin lp_playground`.
//!
//! - Links: the swarm rule (`epsilon_graph`, D00), Gaussian weights or all 1.
//! - Each drone has a true class. Reporting drones are labelled with it (thick ring);
//!   silent drones get their class from `aerie_classic::lp::label_propagation`.
//! - The round slider (or Play) shows Y after each round: colours mix as the labels
//!   arrive. The right panel lists every row of Y, the guess, and whether it is right.
//!
//! Canvas: click empty space to add a drone with the current brush; click a drone to
//! paint it with the brush; drag to move; right-click (or Delete) to remove.

use aerie_classic::lp::label_propagation;
use aerie_graph::EdgeList;
use aerie_viz::grid::{Focus, fmt, matrix_grid, show_tip};
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, pos2, vec2,
};
use faer::Mat;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, DroneId, Role, SquadId, Vec2};
use swarmsim::graphs::epsilon::epsilon_graph;

/// Size of the playground, km.
const WORLD_W: f64 = 16.0;
const WORLD_H: f64 = 10.0;
/// How close the pointer must be to a drone to grab it, in pixels.
const GRAB_PX: f32 = 12.0;
/// Drone marker radius, pixels.
const NODE_PX: f32 = 9.0;
/// The most classes the playground offers.
const MAX_CLASSES: usize = 4;
/// Colour of each class.
const CLASS_COLORS: [Color32; MAX_CLASSES] = [
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(89, 161, 79),
    Color32::from_rgb(176, 122, 161),
];
/// Silent drones that no label has reached yet.
const NO_GUESS: Color32 = Color32::from_rgb(120, 120, 120);
const RIGHT: Color32 = Color32::from_rgb(89, 161, 79);
const WRONG: Color32 = Color32::from_rgb(225, 87, 89);
/// Accent colours of the matrices in the Matrices tab.
const P_ACCENT: Color32 = Color32::from_rgb(118, 183, 178);
const CLAMPED_ACCENT: Color32 = Color32::from_rgb(237, 201, 72);
const Y_ACCENT: Color32 = Color32::from_rgb(170, 170, 190);
/// Rounds shown at most (the slider and the plot).
const MAX_ROUNDS: usize = 500;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE label propagation playground")
            .with_inner_size([1600.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE label propagation playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

/// What a click on a drone (or on empty space) does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Brush {
    /// The drone's true class becomes k and it reports it.
    Report(usize),
    /// The drone's true class becomes k but it stays silent.
    Silent(usize),
    /// Keep its class, switch it between reporting and silent.
    Toggle,
}

/// What the right panel shows under the plot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Rows,
    Matrices,
}

/// The matrices of the Matrices tab (to link a hovered cell to the others).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Which {
    P,
    Clamped,
    YPrev,
    Y,
}

/// One drone of the playground.
#[derive(Debug, Clone)]
struct Node {
    drone: Drone,
    /// Its true class (what it would report).
    truth: usize,
    /// Whether it gets its report through (labelled) or is silent.
    reports: bool,
}

/// Everything computed from the drones for one frame.
struct Run {
    graph: EdgeList,
    /// P = D⁻¹A.
    p: Mat<f64>,
    /// P′: P with every reporting drone's row replaced by an identity row.
    clamped: Mat<f64>,
    /// Y after round 0 (the one-hots), 1, 2, ... up to the last round.
    ys: Vec<Mat<f64>>,
    /// Largest change of any cell in each round (round 1 first).
    changes: Vec<f64>,
    /// Final Y and rounds from the real `label_propagation`.
    final_y: Mat<f64>,
    rounds: usize,
}

struct Playground {
    nodes: Vec<Node>,
    config: SimConfig,
    n_classes: usize,
    binary: bool,
    tol: f64,
    max_iter: usize,

    brush: Brush,
    snap: bool,
    selected: Option<usize>,
    dragging: Option<usize>,
    hover_node: Option<usize>,
    hover_row: Option<usize>,
    /// Right panel tab, and the matrix cell hovered last frame.
    tab: Tab,
    mat_hover: Option<(Which, usize, usize)>,

    /// The round shown on the map (clamped to the last round).
    round: usize,
    /// Always show the last round (follows edits).
    show_final: bool,
    playing: bool,
    /// Rounds per second while playing.
    speed: f32,
    play_clock: f32,

    show_truth: bool,
    show_bars: bool,
    show_weights: bool,

    // Random preset.
    seed: u64,
    n_random: usize,
    report_share: f64,

    zoom: f32,
    pan: egui::Vec2,
    panning: bool,
}

impl Default for Playground {
    fn default() -> Self {
        let mut p = Playground {
            nodes: Vec::new(),
            config: SimConfig::default(),
            n_classes: 2,
            binary: true,
            tol: 1e-6,
            max_iter: 1000,
            brush: Brush::Report(0),
            snap: true,
            selected: None,
            dragging: None,
            hover_node: None,
            hover_row: None,
            tab: Tab::Rows,
            mat_hover: None,
            round: 0,
            show_final: true,
            playing: false,
            speed: 4.0,
            play_clock: 0.0,
            show_truth: true,
            show_bars: true,
            show_weights: false,
            seed: 0,
            n_random: 30,
            report_share: 0.1,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            panning: false,
        };
        p.load_two_sides();
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Advance the animation.
        if self.playing {
            self.show_final = false;
            self.play_clock += ui.input(|i| i.stable_dt);
            let step = 1.0 / self.speed.max(0.1);
            while self.play_clock >= step {
                self.play_clock -= step;
                self.round += 1;
            }
            ui.ctx().request_repaint();
        }

        let run = self.run();
        let last = run.ys.len() - 1;
        if self.show_final || self.round > last {
            if self.playing && self.round > last {
                self.playing = false;
            }
            self.round = last;
        }

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(290.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui, &run));
            });
        egui::Panel::right("table")
            .resizable(true)
            .default_size(470.0)
            .show(ui, |ui| {
                egui::ScrollArea::both().show(ui, |ui| self.table(ui, &run));
            });
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui, &run));
    }
}

impl Playground {
    // ---------------------------------------------------------------- model

    fn drones(&self) -> Vec<Drone> {
        self.nodes.iter().map(|n| n.drone.clone()).collect()
    }

    /// The label slots passed to `label_propagation`: `Some(truth)` for reporting drones.
    fn labels(&self) -> Vec<Option<usize>> {
        self.nodes
            .iter()
            .map(|n| n.reports.then_some(n.truth))
            .collect()
    }

    fn graph(&self) -> EdgeList {
        let mut g = epsilon_graph(&self.drones(), &self.config);
        if self.binary {
            for e in &mut g.edges {
                e.weight = 1.0;
            }
        }
        g
    }

    /// Graph, every round of Y for the slider, and the real function's answer.
    fn run(&self) -> Run {
        let graph = self.graph();
        let p = graph.transition_matrix();
        let labels = self.labels();
        let (ys, changes) = trajectory(&p, &labels, self.n_classes, self.tol, MAX_ROUNDS);
        let (final_y, rounds) =
            label_propagation(&p, &labels, self.n_classes, self.tol, self.max_iter);
        let clamped = clamp_rows(&p, &labels);
        Run {
            graph,
            p,
            clamped,
            ys,
            changes,
            final_y,
            rounds,
        }
    }

    fn place(&self, p: Vec2) -> Vec2 {
        let snap = |v: f64| {
            if self.snap {
                (v * 2.0).round() / 2.0
            } else {
                v
            }
        };
        Vec2 {
            x: snap(p.x).clamp(0.0, WORLD_W),
            y: snap(p.y).clamp(0.0, WORLD_H),
        }
    }

    fn add(&mut self, role: Role, position: Vec2, truth: usize, reports: bool) {
        let id = DroneId(self.nodes.len() as u32);
        self.nodes.push(Node {
            drone: Drone::new(id, SquadId(truth as u16), role, position),
            truth,
            reports,
        });
    }

    /// Applies the brush to drone `i`.
    fn paint(&mut self, i: usize) {
        let node = &mut self.nodes[i];
        match self.brush {
            Brush::Report(k) => {
                node.truth = k;
                node.reports = true;
            }
            Brush::Silent(k) => {
                node.truth = k;
                node.reports = false;
            }
            Brush::Toggle => node.reports = !node.reports,
        }
        node.drone.squad = SquadId(node.truth as u16);
    }

    fn remove(&mut self, i: usize) {
        self.nodes.remove(i);
        for (k, n) in self.nodes.iter_mut().enumerate() {
            n.drone.id = DroneId(k as u32);
        }
        self.selected = match self.selected {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        self.dragging = None;
        self.hover_node = None;
        self.hover_row = None;
    }

    fn node_at(&self, t: &Transform, p: Pos2) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, t.to_screen(n.drone.position).distance(p)))
            .filter(|&(_, px)| px <= GRAB_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Replaces the drones: (x, y, truth, reports).
    fn set(&mut self, n_classes: usize, drones: &[(f64, f64, usize, bool)]) {
        self.config = SimConfig::default();
        self.n_classes = n_classes;
        self.nodes.clear();
        for &(x, y, truth, reports) in drones {
            self.add(Role::Recon, Vec2 { x, y }, truth, reports);
        }
        self.selected = None;
        self.hover_node = None;
        self.hover_row = None;
        self.show_final = true;
        self.playing = false;
    }

    /// A path of 5, 2 km apart, the two ends reporting different classes.
    fn load_path(&mut self) {
        self.set(
            2,
            &[
                (3.0, 5.0, 0, true),
                (5.0, 5.0, 0, false),
                (7.0, 5.0, 0, false),
                (9.0, 5.0, 1, false),
                (11.0, 5.0, 1, true),
            ],
        );
    }

    /// Two groups (west class 0, east class 1) joined by one link, one reporter each.
    fn load_two_sides(&mut self) {
        let mut d = Vec::new();
        for (cx, class) in [(4.5, 0), (11.5, 1)] {
            for k in 0..6 {
                let a = k as f64 / 6.0 * std::f64::consts::TAU;
                d.push((cx + 1.5 * a.cos(), 5.0 + 1.5 * a.sin(), class, k == 3));
            }
        }
        // The bridge: three silent drones in a row between the groups, 1 km apart.
        d.push((7.0, 5.0, 0, false));
        d.push((8.0, 5.0, 0, false));
        d.push((9.0, 5.0, 1, false));
        self.set(2, &d);
    }

    /// Three squads (classes 0, 1, 2). Squads 0 and 1 have a reporter; squad 2 has none
    /// and hangs off squad 1 (Mission 1, experiment 4: the silent squad).
    fn load_silent_squad(&mut self) {
        let mut d = Vec::new();
        for (cx, cy, class, reporter) in [
            (3.5, 6.5, 0, true),
            (8.0, 3.5, 1, true),
            (12.5, 6.5, 2, false),
        ] {
            for k in 0..5 {
                let a = k as f64 / 5.0 * std::f64::consts::TAU;
                let reports = reporter && k == 0;
                d.push((cx + 1.2 * a.cos(), cy + 1.2 * a.sin(), class, reports));
            }
        }
        // Links between the squads (the drones in between belong to squad 1).
        d.push((5.8, 5.0, 1, false));
        d.push((10.3, 5.0, 1, false));
        self.set(3, &d);
    }

    /// `n_random` drones; true class = west or east half; `report_share` of them report.
    fn load_random(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut d = Vec::new();
        for _ in 0..self.n_random {
            let x = rng.random_range(0.5..WORLD_W - 0.5);
            let y = rng.random_range(0.5..WORLD_H - 0.5);
            let class = usize::from(x > WORLD_W / 2.0);
            d.push((x, y, class, rng.random_range(0.0..1.0) < self.report_share));
        }
        self.set(2, &d);
        // Wider range so random drones form one graph more often.
        self.config.range_short = 3.5;
    }

    // ---------------------------------------------------------------- panels

    fn controls(&mut self, ui: &mut egui::Ui, run: &Run) {
        ui.heading("Label propagation");
        ui.weak("Mission 1: spread the few reports over the comm graph.");
        ui.add_space(6.0);

        // Rounds.
        ui.label(RichText::new("Rounds").strong());
        let last = run.ys.len() - 1;
        ui.horizontal(|ui| {
            if ui
                .button("⏮")
                .on_hover_text("Round 0: only the reports")
                .clicked()
            {
                self.round = 0;
                self.show_final = false;
                self.playing = false;
            }
            if ui.button("◀").clicked() {
                self.round = self.round.saturating_sub(1);
                self.show_final = false;
                self.playing = false;
            }
            let play = if self.playing { "Pause" } else { "Play" };
            if ui.button(play).clicked() {
                if !self.playing && self.round >= last {
                    self.round = 0;
                }
                self.playing = !self.playing;
                self.show_final = false;
                self.play_clock = 0.0;
            }
            if ui.button("▶").clicked() {
                self.round = (self.round + 1).min(last);
                self.show_final = false;
                self.playing = false;
            }
            if ui
                .button("⏭")
                .on_hover_text("The converged answer")
                .clicked()
            {
                self.show_final = true;
                self.playing = false;
            }
        });
        let mut r = self.round;
        if ui
            .add(egui::Slider::new(&mut r, 0..=last).text("round"))
            .changed()
        {
            self.round = r;
            self.show_final = false;
            self.playing = false;
        }
        ui.add(
            egui::Slider::new(&mut self.speed, 0.5..=30.0)
                .logarithmic(true)
                .text("rounds / s"),
        );
        ui.weak(format!(
            "label_propagation stopped after {} rounds (tol {:.0e}, max {}).",
            run.rounds, self.tol, self.max_iter
        ));

        ui.separator();
        ui.label(RichText::new("Brush").strong());
        ui.weak("Click a drone to paint it. Click empty space: a new drone.");
        for k in 0..self.n_classes {
            ui.horizontal(|ui| {
                class_dot(ui, k, true);
                ui.radio_value(
                    &mut self.brush,
                    Brush::Report(k),
                    format!("Class {k}, reports"),
                );
            });
            ui.horizontal(|ui| {
                class_dot(ui, k, false);
                ui.radio_value(
                    &mut self.brush,
                    Brush::Silent(k),
                    format!("Class {k}, silent"),
                );
            });
        }
        ui.radio_value(&mut self.brush, Brush::Toggle, "Switch reports / silent");
        ui.add(egui::Slider::new(&mut self.n_classes, 2..=MAX_CLASSES).text("classes"));
        // Drones whose class no longer exists move to the last class.
        let top = self.n_classes - 1;
        for n in &mut self.nodes {
            n.truth = n.truth.min(top);
        }
        if let Brush::Report(k) | Brush::Silent(k) = &mut self.brush {
            *k = (*k).min(top);
        }
        ui.checkbox(&mut self.snap, "Snap to 0.5 km");

        ui.separator();
        ui.label(RichText::new("Graph").strong());
        ui.checkbox(&mut self.binary, "All links weight 1")
            .on_hover_text("Untick for the Gaussian weights: closer neighbours pull harder.");
        ui.add(
            egui::Slider::new(&mut self.config.range_short, 0.5..=10.0)
                .suffix(" km")
                .text("radio range"),
        );
        ui.horizontal(|ui| {
            ui.label("tol");
            let mut exp = self.tol.log10().round() as i32;
            if ui
                .add(egui::DragValue::new(&mut exp).range(-12..=-1).prefix("1e"))
                .changed()
            {
                self.tol = 10f64.powi(exp);
            }
            ui.label("max rounds");
            ui.add(egui::DragValue::new(&mut self.max_iter).range(1..=10_000));
        });

        ui.separator();
        ui.label(RichText::new("Display").strong());
        ui.checkbox(&mut self.show_truth, "True class as an outer ring");
        ui.checkbox(&mut self.show_bars, "Y as a bar under each drone");
        ui.checkbox(&mut self.show_weights, "Link weights");

        ui.separator();
        ui.label(RichText::new("Presets").strong());
        if ui.button("Two sides joined by a bridge").clicked() {
            self.load_two_sides();
        }
        if ui.button("Path of 5").clicked() {
            self.load_path();
        }
        if ui
            .button("The silent squad")
            .on_hover_text("Experiment 4: squad 2 has no reporter. What does it get?")
            .clicked()
        {
            self.load_silent_squad();
        }
        ui.horizontal(|ui| {
            if ui.button("Random").clicked() {
                self.load_random();
            }
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.seed));
        });
        ui.add(egui::Slider::new(&mut self.n_random, 5..=60).text("drones"));
        ui.add(
            egui::Slider::new(&mut self.report_share, 0.01..=0.5)
                .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                .text("report"),
        );
        if ui.button("Clear").clicked() {
            self.set(self.n_classes, &[]);
        }
    }

    /// Right panel: accuracy, the convergence plot, and every row of Y.
    fn table(&mut self, ui: &mut egui::Ui, run: &Run) {
        let y = &run.ys[self.round];
        let n = self.nodes.len();
        ui.heading(format!("Y after round {}", self.round));

        // Accuracy on the silent drones, at this round.
        let silent: Vec<usize> = (0..n).filter(|&i| !self.nodes[i].reports).collect();
        let guesses: Vec<Option<usize>> = (0..n).map(|i| guess(y, i)).collect();
        let right = silent
            .iter()
            .filter(|&&i| guesses[i] == Some(self.nodes[i].truth))
            .count();
        let none = silent.iter().filter(|&&i| guesses[i].is_none()).count();
        ui.label(format!(
            "{} reporting, {} silent · right {right} of {} ({:.0} %) · no guess {none}",
            n - silent.len(),
            silent.len(),
            silent.len(),
            100.0 * right as f64 / silent.len().max(1) as f64
        ));
        let same = (0..n).all(|i| {
            (0..self.n_classes)
                .all(|c| (run.final_y[(i, c)] - run.ys.last().unwrap()[(i, c)]).abs() < 1e-9)
        });
        ui.weak(format!(
            "Last round matches label_propagation: {}",
            if same {
                "yes"
            } else {
                "no (raise the round cap)"
            }
        ));

        ui.add_space(4.0);
        ui.label(RichText::new("Largest change per round (log scale)").strong());
        change_plot(ui, &run.changes, self.tol, self.round);

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Rows, "Rows of Y");
            ui.selectable_value(&mut self.tab, Tab::Matrices, "Matrices");
        });
        if self.tab == Tab::Matrices {
            self.matrices(ui, run);
            return;
        }
        self.mat_hover = None;
        ui.label(RichText::new("Rows of Y").strong());
        ui.weak("Hover a row to find the drone on the map.");
        let mut hovered = None;
        egui::Grid::new("rows").striped(true).show(ui, |ui| {
            ui.label("Drone");
            ui.label("Status");
            ui.label("Y");
            ui.label("Values");
            ui.label("Guess");
            ui.label("Truth");
            ui.end_row();
            for i in 0..n {
                let node = &self.nodes[i];
                let mut cells = Vec::new();
                cells.push(ui.label(format!("{i}")));
                cells.push(if node.reports {
                    ui.label(
                        RichText::new(format!("reports {}", node.truth))
                            .color(CLASS_COLORS[node.truth])
                            .strong(),
                    )
                } else {
                    ui.weak("silent")
                });
                let (rect, bar) = ui.allocate_exact_size(vec2(90.0, 12.0), Sense::hover());
                draw_bar(ui.painter(), rect, y, i, self.n_classes);
                cells.push(bar);
                let values: Vec<String> = (0..self.n_classes)
                    .map(|c| format!("{:.2}", y[(i, c)]))
                    .collect();
                cells.push(ui.monospace(values.join(" ")));
                cells.push(match guesses[i] {
                    Some(c) => ui.label(RichText::new(format!("{c}")).color(CLASS_COLORS[c])),
                    None => ui.weak("—"),
                });
                let mark = if node.reports {
                    ui.label(format!("{}", node.truth))
                } else if guesses[i] == Some(node.truth) {
                    ui.colored_label(RIGHT, format!("{} ✓", node.truth))
                } else {
                    ui.colored_label(WRONG, format!("{} ×", node.truth))
                };
                cells.push(mark);
                if cells.iter().any(|r| r.hovered()) {
                    hovered = Some(i);
                }
                ui.end_row();
            }
        });
        self.hover_row = hovered;
    }

    /// The Matrices tab: P, P′, and Y before and after the shown round, with the
    /// product Y(t) = P′ × Y(t−1) spelled out on hover.
    fn matrices(&mut self, ui: &mut egui::Ui, run: &Run) {
        let n = self.nodes.len();
        let c_n = self.n_classes;
        if n == 0 {
            ui.weak("Add drones on the map.");
            self.hover_row = None;
            self.mat_hover = None;
            return;
        }
        let t = self.round.min(run.ys.len() - 1);
        let weak = ui.visuals().weak_text_color();
        let drones: Vec<(String, Color32)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                if node.reports {
                    (format!("{i}R"), CLASS_COLORS[node.truth])
                } else {
                    (i.to_string(), weak)
                }
            })
            .collect();
        let classes: Vec<(String, Color32)> = (0..c_n)
            .map(|c| (format!("c{c}"), CLASS_COLORS[c]))
            .collect();
        let prev = self.mat_hover;
        let mut now: Option<(Which, usize, usize)> = None;
        ui.weak("Row i = drone i (R = reports, in its class colour). Hover a cell.");

        // P.
        ui.add_space(6.0);
        ui.label(RichText::new("P = D⁻¹A").strong().color(P_ACCENT));
        ui.weak("Row i: how drone i averages its neighbours. Every row sums to 1.");
        let (h, resp) = matrix_grid(
            ui,
            &run.p,
            &drones,
            &drones,
            focus_for(Which::P, prev),
            P_ACCENT,
            Some("Σ"),
        );
        if let Some((i, j)) = h {
            now = Some((Which::P, i, j));
            let text = if run.p[(i, j)] == 0.0 {
                format!("P[{i}][{j}] = 0: drone {j} is not a neighbour of {i}.")
            } else {
                format!(
                    "P[{i}][{j}] = {}: the share of drone {i}'s average taken from {j}.",
                    fmt(run.p[(i, j)])
                )
            };
            show_tip(resp, &text);
        }

        // P′.
        ui.add_space(8.0);
        ui.label(
            RichText::new("P′ = P with the reporters' rows clamped")
                .strong()
                .color(CLAMPED_ACCENT),
        );
        ui.weak("A reporting drone only listens to itself: 1 on the diagonal, 0 elsewhere.");
        let (h, resp) = matrix_grid(
            ui,
            &run.clamped,
            &drones,
            &drones,
            focus_for(Which::Clamped, prev),
            CLAMPED_ACCENT,
            Some("Σ"),
        );
        if let Some((i, j)) = h {
            now = Some((Which::Clamped, i, j));
            let text = if self.nodes[i].reports {
                format!("Drone {i} reports: P′[{i}][{i}] = 1, so its row of Y never changes.")
            } else if run.clamped[(i, j)] == 0.0 {
                format!("P′[{i}][{j}] = 0: drone {i} does not listen to {j}.")
            } else {
                format!(
                    "P′[{i}][{j}] = P[{i}][{j}] = {}: drone {i} is silent, it averages.",
                    fmt(run.clamped[(i, j)])
                )
            };
            show_tip(resp, &text);
        }

        if t == 0 {
            ui.add_space(8.0);
            ui.label(RichText::new("Y after round 0").strong());
            ui.weak("One-hot rows for the reporters, zeros for the silent drones.");
            let (h, _) = matrix_grid(
                ui,
                &run.ys[0],
                &drones,
                &classes,
                focus_for(Which::Y, prev),
                Y_ACCENT,
                Some("Σ"),
            );
            if let Some((i, c)) = h {
                now = Some((Which::Y, i, c));
            }
            ui.weak("Step the round slider to see Y(1) = P′ × Y(0).");
        } else {
            let (y_prev, y) = (&run.ys[t - 1], &run.ys[t]);
            ui.add_space(8.0);
            ui.label(RichText::new(format!("Y after round {}", t - 1)).strong());
            let (h, resp) = matrix_grid(
                ui,
                y_prev,
                &drones,
                &classes,
                focus_for(Which::YPrev, prev),
                Y_ACCENT,
                Some("Σ"),
            );
            if let Some((j, c)) = h {
                now = Some((Which::YPrev, j, c));
                show_tip(
                    resp,
                    &format!(
                        "Y{}[{j}][c{c}] = {}: drone {j}'s belief in class {c} before round {t}.",
                        t - 1,
                        fmt(y_prev[(j, c)])
                    ),
                );
            }

            ui.add_space(8.0);
            ui.label(
                RichText::new(format!("Y after round {t} = P′ × Y after round {}", t - 1)).strong(),
            );
            ui.weak("Hover a cell: row i of P′ times column c of the Y above.");
            let (h, resp) = matrix_grid(
                ui,
                y,
                &drones,
                &classes,
                focus_for(Which::Y, prev),
                Y_ACCENT,
                Some("Σ"),
            );
            if let Some((i, c)) = h {
                now = Some((Which::Y, i, c));
                let terms: Vec<String> = (0..n)
                    .filter(|&j| run.clamped[(i, j)] != 0.0)
                    .map(|j| format!("{}·{}", fmt(run.clamped[(i, j)]), fmt(y_prev[(j, c)])))
                    .collect();
                let used: Vec<String> = (0..n)
                    .filter(|&j| run.clamped[(i, j)] != 0.0)
                    .map(|j| j.to_string())
                    .collect();
                let text = if terms.is_empty() {
                    format!("Y{t}[{i}][c{c}] = 0: drone {i} has no row in P′.")
                } else {
                    format!(
                        "Y{t}[{i}][c{c}] = Σⱼ P′[{i}][j] · Y{}[j][c{c}]  (j = {})\n= {} = {}",
                        t - 1,
                        used.join(", "),
                        terms.join(" + "),
                        fmt(y[(i, c)])
                    )
                };
                show_tip(resp, &text);
            }
        }

        self.mat_hover = now;
        // A hovered row is a drone: outline it on the map.
        self.hover_row = now.map(|(w, i, j)| match w {
            Which::YPrev | Which::P | Which::Clamped | Which::Y if i < n => i,
            _ => j,
        });
    }

    // ---------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui, run: &Run) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        if let Some(pointer) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let before = Transform::fit(response.rect, self.zoom, self.pan).to_km(pointer);
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.5, 20.0);
                let moved = Transform::fit(response.rect, self.zoom, self.pan).to_screen(before);
                self.pan += pointer - moved;
            }
        }
        let t = Transform::fit(response.rect, self.zoom, self.pan);
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();
        let weak = visuals.weak_text_color();

        // --- Input.
        self.hover_node = response.hover_pos().and_then(|p| self.node_at(&t, p));
        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.dragging = origin.and_then(|p| self.node_at(&t, p));
            if self.dragging.is_some() {
                self.selected = self.dragging;
            }
            self.panning = self.dragging.is_none();
        }
        if self.panning && response.dragged() {
            self.pan += response.drag_delta();
        }
        if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos())
            && response.dragged()
            && i < self.nodes.len()
        {
            self.nodes[i].drone.position = self.place(t.to_km(p));
        }
        if response.drag_stopped() {
            self.dragging = None;
            self.panning = false;
        }
        let mut changed = false;
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.node_at(&t, p) {
                Some(i) => {
                    self.paint(i);
                    self.selected = Some(i);
                    changed = true;
                }
                None if t.world_rect().contains(p) => {
                    let pos = self.place(t.to_km(p));
                    let (truth, reports) = match self.brush {
                        Brush::Report(k) => (k, true),
                        Brush::Silent(k) => (k, false),
                        Brush::Toggle => (0, false),
                    };
                    self.add(Role::Recon, pos, truth, reports);
                    self.selected = Some(self.nodes.len() - 1);
                    changed = true;
                }
                None => {}
            }
        }
        if response.secondary_clicked()
            && let Some(i) = response
                .interact_pointer_pos()
                .and_then(|p| self.node_at(&t, p))
        {
            self.remove(i);
            changed = true;
        }
        if !ui.ctx().egui_wants_keyboard_input() {
            if let Some(i) = self.selected
                && ui.input(|inp| inp.key_pressed(Key::Delete))
            {
                self.remove(i);
                changed = true;
            }
            if ui.input(|inp| inp.key_pressed(Key::Escape)) {
                self.selected = None;
            }
        }
        // An edit makes this frame's run stale: draw from a fresh one.
        let fresh;
        let run = if changed || self.dragging.is_some() {
            fresh = self.run();
            &fresh
        } else {
            run
        };
        let y = &run.ys[self.round.min(run.ys.len() - 1)];

        // --- World and grid.
        let world = t.world_rect();
        painter.rect_filled(world, 2.0, visuals.extreme_bg_color);
        for k in 0..=(WORLD_W as i32) {
            let x = t
                .to_screen(Vec2 {
                    x: k as f64,
                    y: 0.0,
                })
                .x;
            painter.vline(
                x,
                world.y_range(),
                Stroke::new(1.0, weak.gamma_multiply(0.12)),
            );
        }
        for k in 0..=(WORLD_H as i32) {
            let yy = t
                .to_screen(Vec2 {
                    x: 0.0,
                    y: k as f64,
                })
                .y;
            painter.hline(
                world.x_range(),
                yy,
                Stroke::new(1.0, weak.gamma_multiply(0.12)),
            );
        }

        // --- Links.
        let focus = self.hover_node.or(self.hover_row).or(self.selected);
        for e in &run.graph.edges {
            let (s, d) = (e.src as usize, e.dst as usize);
            if s >= self.nodes.len() || d >= self.nodes.len() {
                continue;
            }
            let (ps, pd) = (
                t.to_screen(self.nodes[s].drone.position),
                t.to_screen(self.nodes[d].drone.position),
            );
            let touches = focus.is_some_and(|k| k == s || k == d);
            let color = text.gamma_multiply(if touches { 0.9 } else { 0.35 });
            painter.line_segment([ps, pd], Stroke::new(1.0 + e.weight as f32, color));
            if self.show_weights {
                painter.text(
                    ps + (pd - ps) * 0.5 + vec2(0.0, -6.0),
                    Align2::CENTER_BOTTOM,
                    format!("{:.2}", e.weight),
                    FontId::monospace(10.0),
                    weak,
                );
            }
        }

        // --- Drones: fill = the classes mixed by row i of Y.
        for (i, node) in self.nodes.iter().enumerate() {
            if y.nrows() <= i {
                continue;
            }
            let c = t.to_screen(node.drone.position);
            let fill = row_color(y, i, self.n_classes);
            if self.show_truth && !node.reports {
                painter.circle_stroke(
                    c,
                    NODE_PX + 4.0,
                    Stroke::new(2.0, CLASS_COLORS[node.truth].gamma_multiply(0.8)),
                );
            }
            painter.circle(
                c,
                NODE_PX,
                fill,
                Stroke::new(1.0, Color32::from_black_alpha(180)),
            );
            if node.reports {
                // Reporting drone: thick ring in the text colour, letter R.
                painter.circle_stroke(c, NODE_PX + 1.5, Stroke::new(3.0, text));
                painter.text(
                    c,
                    Align2::CENTER_CENTER,
                    "R",
                    FontId::proportional(11.0),
                    Color32::WHITE,
                );
            } else if guess(y, i).is_none() {
                painter.text(
                    c,
                    Align2::CENTER_CENTER,
                    "?",
                    FontId::proportional(12.0),
                    Color32::WHITE,
                );
            }
            if self.show_bars && !node.reports {
                let r = Rect::from_center_size(c + vec2(0.0, NODE_PX + 8.0), vec2(26.0, 5.0));
                draw_bar(&painter, r, y, i, self.n_classes);
            }
            if focus == Some(i) {
                painter.circle_stroke(c, NODE_PX + 8.0, Stroke::new(2.0, text));
            }
            painter.text(
                c + vec2(NODE_PX + 4.0, -NODE_PX - 1.0),
                Align2::LEFT_BOTTOM,
                format!("{i}"),
                FontId::proportional(12.0),
                weak,
            );
        }

        painter.text(
            world.left_top() + vec2(8.0, 6.0),
            Align2::LEFT_TOP,
            format!(
                "round {} of {}",
                self.round.min(run.ys.len() - 1),
                run.ys.len() - 1
            ),
            FontId::proportional(14.0),
            text,
        );

        if let Some(i) = self.hover_node
            && i < self.nodes.len()
            && i < y.nrows()
        {
            let node = &self.nodes[i];
            let values: Vec<String> = (0..self.n_classes)
                .map(|c| format!("{:.3}", y[(i, c)]))
                .collect();
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!(
                    "Drone {i} · {}",
                    if node.reports { "reports" } else { "silent" }
                ));
                ui.label(format!("Y row: [{}]", values.join(", ")));
                ui.label(format!("True class: {}", node.truth));
                match guess(y, i) {
                    Some(c) if !node.reports => {
                        ui.label(format!(
                            "Guess: {c} ({})",
                            if c == node.truth { "right" } else { "wrong" }
                        ));
                    }
                    None => {
                        ui.label("No guess yet: no label has reached it.");
                    }
                    _ => {}
                }
            });
        }
    }
}

/// P′: `p` with each labelled drone's row replaced by an identity row (1 on the
/// diagonal). Multiplying by it leaves the labelled rows of Y as they are.
fn clamp_rows(p: &Mat<f64>, labels: &[Option<usize>]) -> Mat<f64> {
    let n = p.nrows();
    let mut clamped = p.clone();
    for (i, label) in labels.iter().enumerate() {
        if label.is_some() {
            for j in 0..n {
                clamped[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    clamped
}

/// What to outline in matrix `target` when a cell of another one is hovered, so the
/// cells that go into the product light up together:
/// Y(t)[i][c] = row i of P′ · column c of Y(t−1).
fn focus_for(target: Which, hover: Option<(Which, usize, usize)>) -> Focus {
    let mut f = Focus::default();
    let Some((w, i, j)) = hover else {
        return f;
    };
    match (w, target) {
        (a, b) if a == b => f.cell = Some((i, j)),
        (Which::Y, Which::Clamped) => f.row = Some(i),
        (Which::Y, Which::YPrev) => f.col = Some(j),
        (Which::Clamped, Which::P) | (Which::P, Which::Clamped) => f.cell = Some((i, j)),
        (Which::Clamped, Which::YPrev) => f.row = Some(j),
        (Which::Clamped, Which::Y) => f.row = Some(i),
        (Which::YPrev, Which::Clamped) => f.col = Some(i),
        (Which::YPrev, Which::Y) => f.col = Some(j),
        _ => {}
    }
    f
}

/// Y after every round, from round 0 (the one-hots) until no cell moves by more than
/// `tol`, or `max_rounds`. The same steps as `label_propagation` (a test checks that
/// they agree round by round), kept here so the slider can show every round.
/// Also returns the largest change in each round.
fn trajectory(
    p: &Mat<f64>,
    labels: &[Option<usize>],
    n_classes: usize,
    tol: f64,
    max_rounds: usize,
) -> (Vec<Mat<f64>>, Vec<f64>) {
    let n = p.nrows();
    let clamped = clamp_rows(p, labels);
    let mut y: Mat<f64> = Mat::zeros(n, n_classes);
    for (i, label) in labels.iter().enumerate() {
        if let Some(c) = label {
            y[(i, *c)] = 1.0;
        }
    }
    let mut ys = vec![y];
    let mut changes = Vec::new();
    for _ in 0..max_rounds {
        let next = &clamped * ys.last().unwrap();
        let prev = ys.last().unwrap();
        let mut change = 0.0_f64;
        for i in 0..n {
            for c in 0..n_classes {
                change = change.max((next[(i, c)] - prev[(i, c)]).abs());
            }
        }
        ys.push(next);
        changes.push(change);
        if change <= tol {
            break;
        }
    }
    (ys, changes)
}

/// The class with the largest share in row `i` of Y, or `None` if the row is all zero.
fn guess(y: &Mat<f64>, i: usize) -> Option<usize> {
    if i >= y.nrows() {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for c in 0..y.ncols() {
        let v = y[(i, c)];
        if v > 0.0 && best.is_none_or(|(_, b)| v > b) {
            best = Some((c, v));
        }
    }
    best.map(|(c, _)| c)
}

/// The class colours mixed by the shares in row `i` of Y; grey when the row is zero.
fn row_color(y: &Mat<f64>, i: usize, n_classes: usize) -> Color32 {
    let total: f64 = (0..n_classes).map(|c| y[(i, c)]).sum();
    if total <= 0.0 {
        return NO_GUESS;
    }
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
    for c in 0..n_classes {
        let w = (y[(i, c)] / total) as f32;
        let col = CLASS_COLORS[c];
        r += w * col.r() as f32;
        g += w * col.g() as f32;
        b += w * col.b() as f32;
    }
    Color32::from_rgb(r as u8, g as u8, b as u8)
}

/// Row `i` of Y as a stacked bar in `rect`: one segment per class, widths = values
/// (out of 1), the rest empty.
fn draw_bar(painter: &egui::Painter, rect: Rect, y: &Mat<f64>, i: usize, n_classes: usize) {
    painter.rect_filled(rect, 1.0, Color32::from_gray(60));
    let mut x = rect.left();
    for c in 0..n_classes {
        let w = (y[(i, c)].clamp(0.0, 1.0) as f32) * rect.width();
        if w > 0.0 {
            let seg = Rect::from_min_size(pos2(x, rect.top()), vec2(w, rect.height()));
            painter.rect_filled(seg, 0.0, CLASS_COLORS[c]);
            x += w;
        }
    }
    painter.rect_stroke(
        rect,
        1.0,
        Stroke::new(1.0, Color32::from_black_alpha(140)),
        StrokeKind::Inside,
    );
}

/// A small dot in class `k`'s colour: filled for reporting, hollow for silent.
fn class_dot(ui: &mut egui::Ui, k: usize, filled: bool) {
    let (rect, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
    let c = rect.center();
    if filled {
        ui.painter().circle(
            c,
            6.0,
            CLASS_COLORS[k],
            Stroke::new(2.0, ui.visuals().text_color()),
        );
    } else {
        ui.painter()
            .circle_stroke(c, 6.0, Stroke::new(2.0, CLASS_COLORS[k]));
    }
}

/// log10 of the largest change per round, with the tolerance as a dashed line and the
/// shown round marked.
fn change_plot(ui: &mut egui::Ui, changes: &[f64], tol: f64, round: usize) {
    let (rect, _) =
        ui.allocate_exact_size(vec2(ui.available_width().min(420.0), 110.0), Sense::hover());
    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();
    painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
    if changes.is_empty() {
        return;
    }
    let lo = tol.max(1e-15).log10().floor() - 1.0;
    let hi = 0.5_f64;
    let n = changes.len().max(2);
    let to = |k: usize, v: f64| {
        let x = rect.left() + 6.0 + (k as f32) / ((n - 1) as f32) * (rect.width() - 12.0);
        let lv = v.max(1e-15).log10().clamp(lo, hi);
        let yy = rect.bottom() - 6.0 - ((lv - lo) / (hi - lo)) as f32 * (rect.height() - 12.0);
        pos2(x, yy)
    };
    // Tolerance line.
    let ty = to(0, tol).y;
    painter.extend(egui::Shape::dashed_line(
        &[pos2(rect.left(), ty), pos2(rect.right(), ty)],
        Stroke::new(1.0, WRONG.gamma_multiply(0.8)),
        4.0,
        3.0,
    ));
    painter.text(
        pos2(rect.right() - 4.0, ty - 2.0),
        Align2::RIGHT_BOTTOM,
        "tol",
        FontId::proportional(10.0),
        WRONG,
    );
    let pts: Vec<Pos2> = changes.iter().enumerate().map(|(k, &v)| to(k, v)).collect();
    painter.add(egui::Shape::line(pts, Stroke::new(1.5, CLASS_COLORS[0])));
    // Round k's change is changes[k - 1].
    if round >= 1 && round <= changes.len() {
        let p = to(round - 1, changes[round - 1]);
        painter.circle_filled(p, 4.0, visuals.text_color());
    }
    painter.text(
        rect.left_top() + vec2(6.0, 4.0),
        Align2::LEFT_TOP,
        format!("{} rounds", changes.len()),
        FontId::proportional(10.0),
        visuals.weak_text_color(),
    );
}

/// km to screen and back. The world is fitted into the panel, y pointing up.
struct Transform {
    left: f32,
    bottom: f32,
    scale: f32,
}

impl Transform {
    fn fit(rect: Rect, zoom: f32, pan: egui::Vec2) -> Self {
        let margin = 30.0;
        let fit = ((rect.width() - 2.0 * margin) / WORLD_W as f32)
            .min((rect.height() - 2.0 * margin) / WORLD_H as f32)
            .max(1.0);
        let scale = fit * zoom;
        Transform {
            left: rect.center().x - WORLD_W as f32 * scale / 2.0 + pan.x,
            bottom: rect.center().y + WORLD_H as f32 * scale / 2.0 + pan.y,
            scale,
        }
    }

    fn to_screen(&self, p: Vec2) -> Pos2 {
        pos2(
            self.left + p.x as f32 * self.scale,
            self.bottom - p.y as f32 * self.scale,
        )
    }

    fn to_km(&self, p: Pos2) -> Vec2 {
        Vec2 {
            x: ((p.x - self.left) / self.scale) as f64,
            y: ((self.bottom - p.y) / self.scale) as f64,
        }
    }

    fn world_rect(&self) -> Rect {
        Rect::from_min_max(
            self.to_screen(Vec2 { x: 0.0, y: WORLD_H }),
            self.to_screen(Vec2 { x: WORLD_W, y: 0.0 }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &Mat<f64>, b: &Mat<f64>) -> bool {
        a.nrows() == b.nrows()
            && a.ncols() == b.ncols()
            && (0..a.nrows()).all(|i| (0..a.ncols()).all(|c| (a[(i, c)] - b[(i, c)]).abs() < 1e-12))
    }

    #[test]
    fn trajectory_matches_label_propagation_round_by_round() {
        // On every preset: Y after round t from the slider equals the real function run
        // with max_iter = t, and the last round equals its converged answer.
        let mut p = Playground::default();
        for load in [
            Playground::load_two_sides,
            Playground::load_path,
            Playground::load_silent_squad,
            Playground::load_random,
        ] {
            load(&mut p);
            let pm = p.graph().transition_matrix();
            let labels = p.labels();
            let (ys, changes) = trajectory(&pm, &labels, p.n_classes, p.tol, MAX_ROUNDS);
            assert_eq!(ys.len(), changes.len() + 1);
            for (t, expected) in ys.iter().enumerate().take(30).skip(1) {
                let (y, rounds) = label_propagation(&pm, &labels, p.n_classes, p.tol, t);
                assert_eq!(rounds, t);
                assert!(same(&y, expected), "round {t}");
            }
            let (y, rounds) = label_propagation(&pm, &labels, p.n_classes, p.tol, 1000);
            assert_eq!(rounds, ys.len() - 1);
            assert!(same(&y, ys.last().unwrap()));
        }
    }

    #[test]
    fn clamped_rows_and_the_product_shown_in_the_matrices_tab() {
        let mut p = Playground::default();
        p.load_two_sides();
        let run = p.run();
        let n = p.nodes.len();
        for i in 0..n {
            for j in 0..n {
                let want = if p.nodes[i].reports {
                    if i == j { 1.0 } else { 0.0 }
                } else {
                    run.p[(i, j)]
                };
                assert_eq!(run.clamped[(i, j)], want, "({i}, {j})");
            }
        }
        // Each shown round is P′ times the one before.
        for t in 1..run.ys.len().min(10) {
            let product = &run.clamped * &run.ys[t - 1];
            assert!(same(&product, &run.ys[t]), "round {t}");
        }
    }

    #[test]
    fn hovering_y_lights_up_the_row_of_p_prime_and_the_column_of_y_before() {
        let hover = Some((Which::Y, 3, 1));
        assert_eq!(focus_for(Which::Clamped, hover).row, Some(3));
        assert_eq!(focus_for(Which::YPrev, hover).col, Some(1));
        assert_eq!(focus_for(Which::Y, hover).cell, Some((3, 1)));
        assert_eq!(focus_for(Which::P, hover), Focus::default());
    }

    #[test]
    fn path_preset_converges_to_a_straight_line() {
        let mut p = Playground::default();
        p.load_path();
        let run = p.run();
        let y = &run.final_y;
        assert!((y[(2, 0)] - 0.5).abs() < 1e-5);
        assert!((y[(1, 0)] - 0.75).abs() < 1e-5);
        assert_eq!(guess(y, 1), Some(0));
        assert_eq!(guess(y, 3), Some(1));
    }

    #[test]
    fn silent_squad_gets_its_neighbours_class() {
        // Squad 2 (drones 10..15) has no reporter and hangs off squad 1: label
        // propagation calls all of it class 1, which is wrong. Experiment 4's point.
        let mut p = Playground::default();
        p.load_silent_squad();
        let y = p.run().final_y;
        for i in 10..15 {
            assert_eq!(p.nodes[i].truth, 2);
            assert_eq!(guess(&y, i), Some(1), "drone {i}");
        }
    }

    #[test]
    fn guess_is_none_for_a_zero_row_and_the_largest_otherwise() {
        let mut y: Mat<f64> = Mat::zeros(2, 3);
        assert_eq!(guess(&y, 0), None);
        y[(1, 0)] = 0.2;
        y[(1, 2)] = 0.7;
        assert_eq!(guess(&y, 1), Some(2));
    }

    #[test]
    fn painting_and_removing() {
        let mut p = Playground {
            brush: Brush::Silent(1),
            ..Default::default()
        };
        p.paint(0);
        assert_eq!((p.nodes[0].truth, p.nodes[0].reports), (1, false));
        p.brush = Brush::Toggle;
        p.paint(0);
        assert!(p.nodes[0].reports);
        let n = p.nodes.len();
        p.remove(0);
        assert_eq!(p.nodes.len(), n - 1);
        assert!(
            p.nodes
                .iter()
                .enumerate()
                .all(|(k, d)| d.drone.id.0 as usize == k)
        );
        assert_eq!(p.labels().len(), n - 1);
    }
}
