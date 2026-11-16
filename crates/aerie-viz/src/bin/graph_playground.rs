//! Graph playground: place drones by hand and watch the comm graph form.
//!
//! Run with `cargo run --release -p aerie-viz --bin graph_playground`.
//!
//! Every mode calls the real builder in `swarmsim::graphs`:
//! - Textbook: one epsilon and one Gaussian width sigma for every pair
//!   (`textbook_epsilon_graph`). Roles are ignored.
//! - Swarm (D00): epsilon = the smaller of the two radio ranges and
//!   sigma = half of it (`epsilon_graph`). Roles matter.
//! - kNN (D00): each drone picks its k nearest, the graph keeps the union
//!   (`knn_graph`). Same kernel, no cut-off: links can exceed radio range.
//!
//! Canvas: click empty space to add a drone, drag a drone to move it,
//! right-click a drone (or select it and press Delete) to remove it.
//! Scroll to zoom, drag empty space to pan.
//! The right panel plots the Gaussian kernel and lists every edge.

use aerie_graph::EdgeList;
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, DroneId, Role, SquadId, Vec2};
use swarmsim::graphs::epsilon::{epsilon_graph, textbook_epsilon_graph};
use swarmsim::graphs::knn::knn_graph;
use swarmsim::graphs::stats::stats;

use aerie_viz::style::{
    draw_drone, ramp, role_color, role_name, role_shape_name, squad_color, swatch,
};

/// Size of the playground, km.
const WORLD_W: f64 = 24.0;
const WORLD_H: f64 = 16.0;
/// How close the pointer must be to a drone to grab it, in pixels.
const GRAB_PX: f32 = 12.0;
/// Drone marker radius, pixels.
const NODE_PX: f32 = 7.0;
/// Colour of every drone in textbook mode, where roles do not matter.
const TEXTBOOK_FILL: Color32 = Color32::from_rgb(78, 121, 167);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE graph playground")
            .with_inner_size([1500.0, 880.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE graph playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

/// Which builder makes the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Textbook,
    Swarm,
    Knn,
}

/// Which drones get their epsilon (or radio range) circle drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Circles {
    None,
    Selected,
    All,
}

impl Circles {
    const ALL: [Circles; 3] = [Circles::None, Circles::Selected, Circles::All];

    fn label(self) -> &'static str {
        match self {
            Circles::None => "None",
            Circles::Selected => "Selected drone",
            Circles::All => "All drones",
        }
    }
}

/// One kernel curve on the side plot.
struct Curve {
    label: String,
    /// Cut-off (epsilon modes) or radio range (kNN mode), km.
    epsilon: f64,
    sigma: f64,
    color: Color32,
    /// True when no edge can exist beyond `epsilon`. False for kNN.
    cut: bool,
}

/// Gaussian kernel weight at distance `d`.
fn gaussian(d: f64, sigma: f64) -> f64 {
    (-(d * d) / (2.0 * sigma * sigma)).exp()
}

struct Playground {
    mode: Mode,
    /// The drones on the canvas. Their index is their node number in the graph.
    nodes: Vec<Drone>,

    // Textbook parameters.
    epsilon: f64,
    sigma: f64,
    lock_sigma: bool,

    // Swarm parameters: only the two radio ranges are used.
    config: SimConfig,

    // Editing.
    new_role: Role,
    /// Squad given to new drones. Squads only matter for the cross-squad stat.
    new_squad: u16,
    snap: bool,
    selected: Option<usize>,
    dragging: Option<usize>,
    seed: u64,
    n_random: usize,

    // Display.
    circles: Circles,
    show_weights: bool,
    show_distances: bool,
    show_one_way: bool,
    show_sigma_ring: bool,
    /// kNN mode: draw each drone's picks as arrows.
    show_picks: bool,

    // View: zoom factor (1 = whole world fits) and pan offset in pixels.
    zoom: f32,
    pan: egui::Vec2,
    /// True while dragging empty space (panning) rather than a drone.
    panning: bool,
}

impl Default for Playground {
    fn default() -> Self {
        let mut p = Playground {
            mode: Mode::Textbook,
            nodes: Vec::new(),
            epsilon: 5.0,
            sigma: 2.5,
            lock_sigma: false,
            config: SimConfig::default(),
            new_role: Role::Recon,
            new_squad: 0,
            snap: true,
            selected: None,
            dragging: None,
            seed: 0,
            n_random: 12,
            circles: Circles::Selected,
            show_weights: true,
            show_distances: false,
            show_one_way: true,
            show_sigma_ring: true,
            show_picks: true,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            panning: false,
        };
        p.load_test_cases();
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.lock_sigma {
            self.sigma = self.epsilon / 2.0;
        }

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        // Built once per frame from the current drones and parameters.
        let graph = self.graph();

        egui::Panel::right("kernel")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side(ui, &graph));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui, &graph));
    }
}

impl Playground {
    // ---------------------------------------------------------------- model

    /// Calls the builder for the current mode.
    fn graph(&self) -> EdgeList {
        match self.mode {
            Mode::Textbook => textbook_epsilon_graph(&self.nodes, self.epsilon, self.sigma),
            Mode::Swarm => epsilon_graph(&self.nodes, &self.config),
            Mode::Knn => knn_graph(&self.nodes, &self.config),
        }
    }

    /// kNN mode: every pick as (picker, picked), before the union.
    /// Same rule as `knn_graph`: closest first, ties to the lower index.
    fn picks(&self) -> Vec<(usize, usize)> {
        let n = self.nodes.len();
        if n < 2 {
            return Vec::new();
        }
        let k = self.config.knn_k.min(n - 1);
        let mut out = Vec::new();
        for i in 0..n {
            let mut candidates: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (self.nodes[i].position.dist(self.nodes[j].position), j))
                .collect();
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            out.extend(candidates.iter().take(k).map(|&(_, j)| (i, j)));
        }
        out
    }

    /// The epsilon a pair is judged against in the current mode.
    fn pair_epsilon(&self, a: &Drone, b: &Drone) -> f64 {
        match self.mode {
            Mode::Textbook => self.epsilon,
            Mode::Swarm | Mode::Knn => a
                .role
                .radio_range(&self.config)
                .min(b.role.radio_range(&self.config)),
        }
    }

    /// The radius of a drone's circle: epsilon (textbook) or its radio range (swarm).
    fn circle_radius(&self, d: &Drone) -> f64 {
        match self.mode {
            Mode::Textbook => self.epsilon,
            Mode::Swarm | Mode::Knn => d.role.radio_range(&self.config),
        }
    }

    /// Kernel curves to plot: one in textbook mode, one per distinct link range in swarm mode.
    fn curves(&self) -> Vec<Curve> {
        match self.mode {
            Mode::Textbook => vec![Curve {
                label: "every pair".to_string(),
                epsilon: self.epsilon,
                sigma: self.sigma,
                color: TEXTBOOK_FILL,
                cut: true,
            }],
            Mode::Swarm | Mode::Knn => {
                let cut = self.mode == Mode::Swarm;
                let s = self.config.range_short;
                let l = self.config.range_long;
                let (short_label, long_label) = if s <= l {
                    ("pair with a non-relay", "relay - relay")
                } else {
                    ("non-relay - non-relay", "pair with a relay")
                };
                let mut out = vec![Curve {
                    label: short_label.to_string(),
                    epsilon: s,
                    sigma: s / 2.0,
                    color: role_color(Role::Recon),
                    cut,
                }];
                if (s - l).abs() > 1e-9 {
                    out.push(Curve {
                        label: long_label.to_string(),
                        epsilon: l,
                        sigma: l / 2.0,
                        color: role_color(Role::Relay),
                        cut,
                    });
                }
                out
            }
        }
    }

    /// Colour of the kernel curve a pair belongs to.
    fn curve_color(&self, curves: &[Curve], eps: f64) -> Color32 {
        curves
            .iter()
            .find(|c| (c.epsilon - eps).abs() < 1e-9)
            .map_or(TEXTBOOK_FILL, |c| c.color)
    }

    /// Clamps to the world and, if snapping is on, rounds to the nearest 0.5 km.
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

    fn add_node(&mut self, role: Role, squad: u16, position: Vec2) {
        let id = DroneId(self.nodes.len() as u32);
        self.nodes
            .push(Drone::new(id, SquadId(squad), role, position));
    }

    /// Removes drone `i` and renumbers the rest so ids match indices again.
    fn remove_node(&mut self, i: usize) {
        self.nodes.remove(i);
        for (k, d) in self.nodes.iter_mut().enumerate() {
            d.id = DroneId(k as u32);
        }
        self.selected = match self.selected {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        self.dragging = None;
    }

    /// The drone under screen point `p`, if any (the closest one within GRAB_PX).
    fn node_at(&self, t: &Transform, p: Pos2) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, d)| (i, t.to_screen(d.position).distance(p)))
            .filter(|&(_, px)| px <= GRAB_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// The three planned unit tests, laid out so the pairs do not interfere:
    /// recon pair 4 km apart, recon-relay 6 km apart, relay pair 8 km apart.
    /// The expected values assume ranges of 5 and 9 km, so it sets those too.
    fn load_test_cases(&mut self) {
        self.config.range_short = 5.0;
        self.config.range_long = 9.0;
        self.nodes.clear();
        self.selected = None;
        self.dragging = None;
        // One squad per pair.
        let pts = [
            (Role::Recon, 0, 1.0, 2.0),
            (Role::Recon, 0, 5.0, 2.0),
            (Role::Recon, 1, 1.0, 9.0),
            (Role::Relay, 1, 7.0, 9.0),
            (Role::Relay, 2, 14.0, 2.0),
            (Role::Relay, 2, 22.0, 2.0),
        ];
        for (role, squad, x, y) in pts {
            self.add_node(role, squad, Vec2 { x, y });
        }
    }

    /// kNN example from the tests: four recon drones on a line at relative
    /// x = 0, 1, 3, 7 with k = 1. Union: a path. Mutual would keep only 0-1.
    fn load_knn_line(&mut self) {
        self.nodes.clear();
        self.selected = None;
        self.dragging = None;
        self.mode = Mode::Knn;
        self.config.knn_k = 1;
        for x in [0.0, 1.0, 3.0, 7.0] {
            self.add_node(Role::Recon, 0, Vec2 { x: 8.0 + x, y: 8.0 });
        }
    }

    /// kNN hub: a centre and four drones 1 km away, k = 1. All four pick the
    /// centre, which ends up with 4 links, more than 2k.
    fn load_knn_hub(&mut self) {
        self.nodes.clear();
        self.selected = None;
        self.dragging = None;
        self.mode = Mode::Knn;
        self.config.knn_k = 1;
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            self.add_node(
                Role::Recon,
                0,
                Vec2 {
                    x: 12.0 + dx,
                    y: 8.0 + dy,
                },
            );
        }
    }

    /// `n_random` drones at random positions with random roles and one of
    /// three squads, from `seed`.
    fn load_random(&mut self) {
        self.nodes.clear();
        self.selected = None;
        self.dragging = None;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        for _ in 0..self.n_random {
            let role = Role::ALL[rng.random_range(0..Role::ALL.len())];
            let p = Vec2 {
                x: rng.random_range(0.0..WORLD_W),
                y: rng.random_range(0.0..WORLD_H),
            };
            let squad = rng.random_range(0..3);
            let p = self.place(p);
            self.add_node(role, squad, p);
        }
    }

    // ---------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph playground");
        ui.weak("Place drones, move them, watch links appear.");
        ui.separator();

        // Mode.
        ui.label(RichText::new("Builder").strong());
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, Mode::Textbook, "Textbook ε-graph");
            ui.selectable_value(&mut self.mode, Mode::Swarm, "Swarm (D00)");
            ui.selectable_value(&mut self.mode, Mode::Knn, "kNN (union)");
        });
        match self.mode {
            Mode::Textbook => {
                ui.weak("textbook_epsilon_graph: one ε and one σ for every pair. Roles ignored.");
            }
            Mode::Swarm => {
                ui.weak("epsilon_graph: ε = min(range_i, range_j), σ = ε / 2. Roles matter.");
            }
            Mode::Knn => {
                ui.weak(
                    "knn_graph: each drone picks its k nearest; linked if either picked. \
                     σ = min range / 2, no cut-off.",
                );
            }
        }
        ui.add_space(6.0);

        // Parameters.
        ui.label(RichText::new("Parameters").strong());
        match self.mode {
            Mode::Textbook => {
                ui.add(
                    egui::Slider::new(&mut self.epsilon, 0.5..=15.0)
                        .text("ε  cut-off")
                        .suffix(" km"),
                );
                ui.add_enabled(
                    !self.lock_sigma,
                    egui::Slider::new(&mut self.sigma, 0.2..=10.0)
                        .text("σ  kernel width")
                        .suffix(" km"),
                );
                ui.checkbox(&mut self.lock_sigma, "Lock σ = ε / 2 (the swarm's choice)");
                ui.weak(format!(
                    "ε / σ = {:.2}. A pair right at the cut-off weighs {:.3}.",
                    self.epsilon / self.sigma,
                    gaussian(self.epsilon, self.sigma)
                ));
            }
            Mode::Swarm | Mode::Knn => {
                if self.mode == Mode::Knn {
                    ui.add(
                        egui::Slider::new(&mut self.config.knn_k, 1..=10)
                            .text("k  picks per drone"),
                    );
                    ui.weak("In kNN mode the ranges below only set σ (the weights).");
                }
                ui.add(
                    egui::Slider::new(&mut self.config.range_short, 0.5..=15.0)
                        .text("short range")
                        .suffix(" km"),
                );
                ui.add(
                    egui::Slider::new(&mut self.config.range_long, 0.5..=20.0)
                        .text("relay range")
                        .suffix(" km"),
                );
                let d = SimConfig::default();
                let reset = format!("Reset ranges ({} / {} km)", d.range_short, d.range_long);
                if ui.button(reset).clicked() {
                    self.config.range_short = d.range_short;
                    self.config.range_long = d.range_long;
                }
            }
        }
        ui.separator();

        // Drones.
        ui.label(RichText::new("Drones").strong());
        ui.weak("Click: add · drag: move · right-click or Delete: remove · Esc: deselect");
        ui.horizontal_wrapped(|ui| {
            ui.label("New drones:");
            for role in Role::ALL {
                ui.selectable_value(&mut self.new_role, role, role_name(role));
            }
        });
        ui.horizontal(|ui| {
            ui.label("Squad:");
            ui.add(
                egui::DragValue::new(&mut self.new_squad)
                    .range(0..=9)
                    .prefix("S"),
            );
            swatch(ui, Role::Recon, squad_color(self.new_squad as usize));
            ui.weak("only used by the cross-squad stat");
        });
        ui.checkbox(&mut self.snap, "Snap to 0.5 km");

        if let Some(i) = self.selected {
            ui.add_space(4.0);
            ui.label(format!("Selected: drone {i}"));
            let mut role = self.nodes[i].role;
            egui::ComboBox::from_label("Role")
                .selected_text(role_name(role))
                .show_ui(ui, |ui| {
                    for r in Role::ALL {
                        ui.selectable_value(&mut role, r, role_name(r));
                    }
                });
            self.nodes[i].role = role;
            ui.horizontal(|ui| {
                ui.label("Squad");
                ui.add(
                    egui::DragValue::new(&mut self.nodes[i].squad.0)
                        .range(0..=9)
                        .prefix("S"),
                );
            });
            let mut x = self.nodes[i].position.x;
            let mut y = self.nodes[i].position.y;
            ui.horizontal(|ui| {
                ui.label("x");
                ui.add(
                    egui::DragValue::new(&mut x)
                        .range(0.0..=WORLD_W)
                        .speed(0.1)
                        .suffix(" km"),
                );
                ui.label("y");
                ui.add(
                    egui::DragValue::new(&mut y)
                        .range(0.0..=WORLD_H)
                        .speed(0.1)
                        .suffix(" km"),
                );
            });
            self.nodes[i].position = Vec2 { x, y };
            ui.horizontal(|ui| {
                if ui.button("Delete").clicked() {
                    self.remove_node(i);
                }
                if ui.button("Deselect").clicked() {
                    self.selected = None;
                }
            });
        }
        ui.add_space(4.0);

        ui.label(RichText::new("Presets").strong());
        if ui
            .button("Test cases: 4, 6, 8 km pairs (ranges 5 / 9)")
            .on_hover_text(
                "Recon-recon 4 km, recon-relay 6 km, relay-relay 8 km.\n\
                 Sets the ranges to 5 / 9 km, where swarm mode
                 expects 0.278, no link, 0.206.",
            )
            .clicked()
        {
            self.load_test_cases();
        }
        ui.horizontal(|ui| {
            if ui
                .button("kNN line (k = 1)")
                .on_hover_text("x = 0, 1, 3, 7. Union: a path. Mutual would keep only 0-1.")
                .clicked()
            {
                self.load_knn_line();
            }
            if ui
                .button("kNN hub (k = 1)")
                .on_hover_text("All four outer drones pick the centre: 4 links > 2k.")
                .clicked()
            {
                self.load_knn_hub();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Random").clicked() {
                self.load_random();
            }
            ui.add(
                egui::DragValue::new(&mut self.n_random)
                    .range(2..=60)
                    .suffix(" drones"),
            );
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut self.seed).range(0..=9999));
        });
        if ui.button("Clear").clicked() {
            self.nodes.clear();
            self.selected = None;
            self.dragging = None;
        }
        ui.separator();

        // Display.
        ui.label(RichText::new("Display").strong());
        ui.horizontal(|ui| {
            if ui.button("Reset view").clicked() {
                self.zoom = 1.0;
                self.pan = egui::Vec2::ZERO;
            }
            ui.weak(format!("zoom ×{:.1}", self.zoom));
        });
        let circle_label = match self.mode {
            Mode::Textbook => "ε circles",
            Mode::Swarm | Mode::Knn => "Radio range circles",
        };
        egui::ComboBox::from_label(circle_label)
            .selected_text(self.circles.label())
            .show_ui(ui, |ui| {
                for c in Circles::ALL {
                    ui.selectable_value(&mut self.circles, c, c.label());
                }
            });
        ui.checkbox(&mut self.show_sigma_ring, "σ ring on the selected drone")
            .on_hover_text("Dashed ring where the weight is exp(-1/2) = 0.607.");
        ui.checkbox(&mut self.show_weights, "Edge weights");
        ui.checkbox(&mut self.show_distances, "Edge distances");
        if self.mode == Mode::Swarm {
            ui.checkbox(&mut self.show_one_way, "One-way pairs (dashed)")
                .on_hover_text(
                    "Within the longer range but not the shorter one:\n\
                     one drone hears the other, but no link (min rule).",
                );
        }
        if self.mode == Mode::Knn {
            ui.checkbox(&mut self.show_picks, "Picks (arrows)")
                .on_hover_text(
                    "Each drone's own k picks, before the union.\n\
                     Two arrows on one link = a mutual pick.",
                );
        }
        ui.separator();

        // Legend.
        ui.label(RichText::new("Legend").strong());
        if self.mode != Mode::Textbook {
            for role in Role::ALL {
                ui.horizontal(|ui| {
                    swatch(ui, role, role_color(role));
                    ui.label(format!(
                        "{} ({}), range {:.1} km",
                        role_name(role),
                        role_shape_name(role),
                        role.radio_range(&self.config)
                    ));
                });
            }
        } else {
            ui.horizontal(|ui| {
                swatch(ui, Role::Recon, TEXTBOOK_FILL);
                ui.label("drone (roles ignored in textbook mode)");
            });
        }
        ui.add_space(4.0);
        ui.weak("Edge colour and width = weight");
        ui.horizontal(|ui| {
            for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
                swatch(ui, Role::Recon, ramp(w));
                ui.label(format!("{w}"));
            }
        });
    }

    // --------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui, graph: &EdgeList) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());

        // Zoom with the scroll wheel, keeping the point under the cursor fixed.
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
        let hovered = response.hover_pos().and_then(|p| self.node_at(&t, p));

        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.dragging = origin.and_then(|p| self.node_at(&t, p));
            if self.dragging.is_some() {
                self.selected = self.dragging;
            }
            // Dragging empty space pans the view instead of moving a drone.
            self.panning = self.dragging.is_none();
        }
        if self.panning && response.dragged() {
            self.pan += response.drag_delta();
        }
        if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos())
            && response.dragged()
            && i < self.nodes.len()
        {
            self.nodes[i].position = self.place(t.to_km(p));
        }
        if response.drag_stopped() {
            self.dragging = None;
            self.panning = false;
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.node_at(&t, p) {
                Some(i) => self.selected = Some(i),
                None if t.world_rect().contains(p) => {
                    let pos = self.place(t.to_km(p));
                    self.add_node(self.new_role, self.new_squad, pos);
                    self.selected = Some(self.nodes.len() - 1);
                }
                None => {}
            }
        }
        if response.secondary_clicked()
            && let Some(i) = response
                .interact_pointer_pos()
                .and_then(|p| self.node_at(&t, p))
        {
            self.remove_node(i);
        }
        let typing = ui.ctx().egui_wants_keyboard_input();
        if !typing {
            if let Some(i) = self.selected
                && ui.input(|inp| inp.key_pressed(Key::Delete))
            {
                self.remove_node(i);
            }
            if ui.input(|inp| inp.key_pressed(Key::Escape)) {
                self.selected = None;
            }
        }
        // Input may have removed a drone: the graph passed in is stale, rebuild it.
        let rebuilt;
        let graph = if graph.n as usize == self.nodes.len() {
            graph
        } else {
            rebuilt = self.graph();
            &rebuilt
        };

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
            let major = k % 5 == 0;
            let stroke = Stroke::new(1.0, weak.gamma_multiply(if major { 0.45 } else { 0.15 }));
            painter.vline(x, world.y_range(), stroke);
            if major {
                painter.text(
                    pos2(x, world.bottom() + 4.0),
                    Align2::CENTER_TOP,
                    format!("{k}"),
                    FontId::proportional(11.0),
                    weak,
                );
            }
        }
        for k in 0..=(WORLD_H as i32) {
            let y = t
                .to_screen(Vec2 {
                    x: 0.0,
                    y: k as f64,
                })
                .y;
            let major = k % 5 == 0;
            let stroke = Stroke::new(1.0, weak.gamma_multiply(if major { 0.45 } else { 0.15 }));
            painter.hline(world.x_range(), y, stroke);
            if major {
                painter.text(
                    pos2(world.left() - 4.0, y),
                    Align2::RIGHT_CENTER,
                    format!("{k}"),
                    FontId::proportional(11.0),
                    weak,
                );
            }
        }
        painter.text(
            pos2(world.right(), world.bottom() + 4.0),
            Align2::RIGHT_TOP,
            "km (grid: 1 km)",
            FontId::proportional(11.0),
            weak,
        );

        // --- Circles: epsilon (textbook) or radio range (swarm).
        let fill_color = |d: &Drone| match self.mode {
            Mode::Textbook => TEXTBOOK_FILL,
            Mode::Swarm | Mode::Knn => role_color(d.role),
        };
        for (i, d) in self.nodes.iter().enumerate() {
            let show = match self.circles {
                Circles::None => false,
                Circles::Selected => self.selected == Some(i) || hovered == Some(i),
                Circles::All => true,
            };
            if show {
                let c = fill_color(d);
                let r = t.len(self.circle_radius(d));
                painter.circle(
                    t.to_screen(d.position),
                    r,
                    c.gamma_multiply(0.06),
                    Stroke::new(1.0, c.gamma_multiply(0.6)),
                );
            }
        }
        if self.show_sigma_ring
            && let Some(i) = self.selected
        {
            let d = &self.nodes[i];
            let sigma = self.circle_radius(d) / 2.0;
            let sigma = match self.mode {
                Mode::Textbook => self.sigma,
                Mode::Swarm | Mode::Knn => sigma,
            };
            let ring = circle_points(t.to_screen(d.position), t.len(sigma), 96);
            painter.extend(Shape::dashed_line(
                &ring,
                Stroke::new(1.0, fill_color(d).gamma_multiply(0.8)),
                5.0,
                4.0,
            ));
            painter.text(
                t.to_screen(Vec2 {
                    x: d.position.x + sigma * std::f64::consts::FRAC_1_SQRT_2,
                    y: d.position.y + sigma * std::f64::consts::FRAC_1_SQRT_2,
                }),
                Align2::LEFT_BOTTOM,
                "σ",
                FontId::proportional(12.0),
                fill_color(d),
            );
        }

        // --- One-way pairs (swarm mode): one drone hears the other, but no link.
        if self.mode == Mode::Swarm && self.show_one_way {
            for i in 0..self.nodes.len() {
                for j in (i + 1)..self.nodes.len() {
                    let (a, b) = (&self.nodes[i], &self.nodes[j]);
                    let dist = a.position.dist(b.position);
                    let ra = a.role.radio_range(&self.config);
                    let rb = b.role.radio_range(&self.config);
                    if dist > ra.min(rb) && dist <= ra.max(rb) {
                        painter.extend(Shape::dashed_line(
                            &[t.to_screen(a.position), t.to_screen(b.position)],
                            Stroke::new(1.0, weak.gamma_multiply(0.8)),
                            4.0,
                            4.0,
                        ));
                    }
                }
            }
        }

        // --- Edges.
        let focus = self.dragging.or(self.selected).or(hovered);
        for e in &graph.edges {
            let (s, d) = (e.src as usize, e.dst as usize);
            let a = t.to_screen(self.nodes[s].position);
            let b = t.to_screen(self.nodes[d].position);
            let incident = focus.is_none_or(|f| f == s || f == d);
            let mut color = ramp(e.weight);
            if !incident {
                color = color.gamma_multiply(0.3);
            }
            let stroke = Stroke::new(1.0 + 4.0 * e.weight as f32, color);
            // kNN links longer than the pair's radio range are dashed.
            let (na, nb) = (&self.nodes[s], &self.nodes[d]);
            let beyond = na.position.dist(nb.position) > self.pair_epsilon(na, nb);
            if self.mode == Mode::Knn && beyond {
                let dashed = Stroke::new(stroke.width.max(1.5), color);
                painter.extend(Shape::dashed_line(&[a, b], dashed, 6.0, 4.0));
            } else {
                painter.line_segment([a, b], stroke);
            }
        }

        // --- kNN picks: an arrow from each picker towards what it picked,
        // stopping halfway, so a mutual pick shows two arrows meeting.
        if self.mode == Mode::Knn && self.show_picks {
            let arrow = Stroke::new(1.5, text.gamma_multiply(0.85));
            for (i, j) in self.picks() {
                let from = t.to_screen(self.nodes[i].position);
                let to = t.to_screen(self.nodes[j].position);
                draw_pick(&painter, from, to, arrow);
            }
        }

        // --- Edge labels (after all lines so lines never cover them).
        if self.show_weights || self.show_distances {
            for e in &graph.edges {
                let (s, d) = (e.src as usize, e.dst as usize);
                let incident = focus.is_none_or(|f| f == s || f == d);
                let dist = self.nodes[s].position.dist(self.nodes[d].position);
                let label = match (self.show_weights, self.show_distances) {
                    (true, true) => format!("{:.3} · {dist:.1} km", e.weight),
                    (true, false) => format!("{:.3}", e.weight),
                    _ => format!("{dist:.1} km"),
                };
                let mid = t
                    .to_screen(self.nodes[s].position)
                    .lerp(t.to_screen(self.nodes[d].position), 0.5);
                let color = if incident {
                    text
                } else {
                    weak.gamma_multiply(0.6)
                };
                let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), color);
                let rect = Rect::from_center_size(mid, galley.size()).expand(2.0);
                let bg = visuals
                    .extreme_bg_color
                    .gamma_multiply(if incident { 0.9 } else { 0.6 });
                painter.rect_filled(rect, 3.0, bg);
                painter.galley(rect.min + vec2(2.0, 2.0), galley, color);
            }
        }

        // --- Drones.
        for (i, d) in self.nodes.iter().enumerate() {
            let c = t.to_screen(d.position);
            let role = match self.mode {
                Mode::Textbook => Role::Recon,
                Mode::Swarm | Mode::Knn => d.role,
            };
            let outline = Stroke::new(1.2, visuals.extreme_bg_color);
            draw_drone(&painter, c, NODE_PX, role, fill_color(d), outline);
            if self.selected == Some(i) {
                painter.circle_stroke(c, NODE_PX + 6.0, Stroke::new(2.0, text));
            } else if hovered == Some(i) {
                painter.circle_stroke(c, NODE_PX + 5.0, Stroke::new(1.0, text.gamma_multiply(0.6)));
            }
            let label = painter.text(
                c + vec2(NODE_PX + 3.0, -NODE_PX - 1.0),
                Align2::LEFT_BOTTOM,
                format!("{i}"),
                FontId::proportional(12.0),
                text,
            );
            painter.text(
                label.right_bottom() + vec2(2.0, 0.0),
                Align2::LEFT_BOTTOM,
                format!("S{}", d.squad.0),
                FontId::proportional(10.0),
                squad_color(d.squad.0 as usize),
            );
        }

        // --- Overlays: hint and pointer coordinates.
        painter.text(
            response.rect.left_top() + vec2(8.0, 6.0),
            Align2::LEFT_TOP,
            "click: add · drag: move · right-click: delete · scroll: zoom · drag empty: pan",
            FontId::proportional(12.0),
            weak,
        );
        if let Some(p) = response.hover_pos()
            && world.contains(p)
        {
            let km = self.place(t.to_km(p));
            painter.text(
                response.rect.right_top() + vec2(-8.0, 6.0),
                Align2::RIGHT_TOP,
                format!("({:.1}, {:.1}) km", km.x, km.y),
                FontId::monospace(12.0),
                weak,
            );
        }

        // --- Hover tooltip.
        if let Some(i) = hovered
            && self.dragging.is_none()
        {
            let d = &self.nodes[i];
            let neighbours = graph.neighbours();
            let degree = graph.degrees()[i];
            let n_links = neighbours[i].len();
            let my_picks: Vec<String> = if self.mode == Mode::Knn {
                self.picks()
                    .into_iter()
                    .filter(|&(a, _)| a == i)
                    .map(|(_, b)| b.to_string())
                    .collect()
            } else {
                Vec::new()
            };
            response.on_hover_ui_at_pointer(|ui| {
                ui.label(RichText::new(format!("Drone {i} · squad S{}", d.squad.0)).strong());
                if self.mode != Mode::Textbook {
                    ui.label(format!(
                        "{}, range {:.1} km",
                        role_name(d.role),
                        d.role.radio_range(&self.config)
                    ));
                }
                ui.label(format!("({:.2}, {:.2}) km", d.position.x, d.position.y));
                ui.label(format!("{n_links} links, weighted degree {degree:.3}"));
                if !my_picks.is_empty() {
                    ui.label(format!("picks: {}", my_picks.join(", ")));
                }
            });
        }
    }

    // ----------------------------------------------------------- right panel

    fn side(&self, ui: &mut egui::Ui, graph: &EdgeList) {
        let curves = self.curves();

        ui.label(RichText::new("Gaussian kernel").strong());
        let formula = match self.mode {
            Mode::Knn => "w(d) = exp(−d² / 2σ²)   for every kNN link, no cut-off",
            _ => "w(d) = exp(−d² / 2σ²)   if d ≤ ε,   else no edge",
        };
        ui.label(RichText::new(formula).monospace());
        ui.add_space(4.0);

        // One dot per edge, on the curve of its pair type.
        let points: Vec<(f64, f64, Color32)> = graph
            .edges
            .iter()
            .map(|e| {
                let (a, b) = (&self.nodes[e.src as usize], &self.nodes[e.dst as usize]);
                let eps = self.pair_epsilon(a, b);
                (
                    a.position.dist(b.position),
                    e.weight,
                    self.curve_color(&curves, eps),
                )
            })
            .collect();
        kernel_plot(ui, &curves, &points);

        // Key values for each curve.
        egui::Grid::new("kernel_values")
            .striped(true)
            .show(ui, |ui| {
                let eps_name = if self.mode == Mode::Knn {
                    "range"
                } else {
                    "ε"
                };
                ui.label("");
                ui.label(eps_name);
                ui.label("σ");
                ui.label("w(σ)");
                ui.label("w(2σ)");
                ui.label(format!("w({eps_name})"));
                ui.end_row();
                for c in &curves {
                    ui.label(RichText::new(&c.label).color(c.color));
                    ui.label(format!("{:.1}", c.epsilon));
                    ui.label(format!("{:.2}", c.sigma));
                    ui.label(format!("{:.3}", gaussian(c.sigma, c.sigma)));
                    ui.label(format!("{:.3}", gaussian(2.0 * c.sigma, c.sigma)));
                    ui.label(format!("{:.3}", gaussian(c.epsilon, c.sigma)));
                    ui.end_row();
                }
            });
        if self.mode == Mode::Textbook {
            let ratio = self.epsilon / self.sigma;
            let note = if ratio < 1.0 {
                "ε < σ: every edge weighs almost 1, so the graph is nearly unweighted."
            } else if ratio > 3.0 {
                "ε > 3σ: far edges exist but weigh almost 0. They add links but no information."
            } else {
                "ε between σ and 3σ: weights span a useful range."
            };
            ui.weak(note);
        } else if self.mode == Mode::Swarm {
            ui.weak("σ = ε / 2 everywhere, so a pair at the edge of its range always weighs exp(−2) = 0.135.");
        } else {
            ui.weak(
                "kNN has no cut-off: a drone is linked to its k nearest however far they are. \
                 Links beyond the radio range (dashed on the map) weigh less than exp(−2) = 0.135.",
            );
        }
        ui.separator();

        // Stats, from swarmsim::graphs::stats::stats, against the plan's targets.
        ui.label(RichText::new("Graph stats").strong());
        ui.weak("From your stats() function. Targets are for the full 100-drone swarm.");
        let st = stats(graph, &self.nodes);
        let good = Color32::from_rgb(89, 161, 79);
        let bad = Color32::from_rgb(225, 87, 89);
        let verdict = |ok: bool| {
            if ok {
                RichText::new("✔").color(good)
            } else {
                RichText::new("✘").color(bad)
            }
        };
        egui::Grid::new("stats").striped(true).show(ui, |ui| {
            ui.label("");
            ui.label("value");
            ui.label("target");
            ui.label("");
            ui.end_row();

            ui.label("Nodes");
            ui.label(st.n_nodes.to_string());
            ui.end_row();
            ui.label("Edges");
            ui.label(st.n_edges.to_string());
            ui.end_row();

            ui.label("Mean degree");
            ui.label(format!("{:.2}", st.mean_degree));
            ui.label("8 to 10");
            ui.label(verdict((8.0..=10.0).contains(&st.mean_degree)));
            ui.end_row();

            ui.label("Mean weight");
            ui.label(format!("{:.3}", st.mean_weight));
            ui.end_row();

            ui.label("Components");
            ui.label(st.n_components.to_string());
            ui.label("1");
            ui.label(verdict(st.n_components == 1));
            ui.end_row();

            ui.label("Largest component");
            ui.label(format!("{} of {}", st.largest_component, st.n_nodes));
            ui.end_row();

            ui.label("Isolated drones");
            ui.label(st.isolated.to_string());
            ui.label("0");
            ui.label(verdict(st.isolated == 0));
            ui.end_row();

            ui.label("Between squads");
            ui.label(format!("{:.0} %", 100.0 * st.cross_squad_share));
            ui.label("about 25 %");
            ui.label(verdict((0.15..=0.35).contains(&st.cross_squad_share)));
            ui.end_row();
        });
        ui.weak("Mean degree below 4 fragments the graph; above 20, GNNs oversmooth.");
        ui.separator();

        // Edge list.
        ui.label(RichText::new("Edge list").strong());
        ui.weak("Exactly what the builder returned: src < dst, node = index.");
        egui::Grid::new("edges").striped(true).show(ui, |ui| {
            ui.label("src");
            ui.label("dst");
            ui.label("d (km)");
            ui.label(if self.mode == Mode::Knn {
                "range (km)"
            } else {
                "ε (km)"
            });
            ui.label("weight");
            ui.end_row();
            for e in &graph.edges {
                let (a, b) = (&self.nodes[e.src as usize], &self.nodes[e.dst as usize]);
                let strong = self
                    .selected
                    .is_some_and(|s| s == e.src as usize || s == e.dst as usize);
                let cell = |s: String| {
                    if strong {
                        RichText::new(s).strong()
                    } else {
                        RichText::new(s)
                    }
                };
                ui.label(cell(e.src.to_string()));
                ui.label(cell(e.dst.to_string()));
                ui.label(cell(format!("{:.2}", a.position.dist(b.position))));
                ui.label(cell(format!("{:.1}", self.pair_epsilon(a, b))));
                ui.label(cell(format!("{:.3}", e.weight)).color(ramp(e.weight)));
                ui.end_row();
            }
        });
        if graph.edges.is_empty() {
            ui.weak("No edges yet.");
        }
    }
}

/// Plots each kernel curve: solid up to ε, faint and dashed beyond (no edge there).
/// Dots are the actual edges at (distance, weight). Hover for a readout.
fn kernel_plot(ui: &mut egui::Ui, curves: &[Curve], points: &[(f64, f64, Color32)]) {
    let d_max = curves
        .iter()
        .map(|c| (c.epsilon * 1.25).max(c.sigma * 3.5))
        .chain(points.iter().map(|p| p.0 * 1.05))
        .fold(1.0, f64::max);
    let width = ui.available_width().max(200.0);
    let (response, painter) = ui.allocate_painter(vec2(width, 240.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_max(rect.min + vec2(36.0, 10.0), rect.max - vec2(10.0, 30.0));
    let x = |d: f64| plot.left() + (d / d_max) as f32 * plot.width();
    let y = |w: f64| plot.bottom() - w as f32 * plot.height();

    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    // Frame and grid.
    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    for w in [0.0, 0.25, 0.5, 0.75, 1.0] {
        painter.hline(
            plot.x_range(),
            y(w),
            Stroke::new(1.0, weak.gamma_multiply(0.25)),
        );
        painter.text(
            pos2(plot.left() - 4.0, y(w)),
            Align2::RIGHT_CENTER,
            format!("{w}"),
            small.clone(),
            weak,
        );
    }
    let step = if d_max <= 6.0 {
        1.0
    } else if d_max <= 15.0 {
        2.0
    } else {
        5.0
    };
    let mut d = 0.0;
    while d <= d_max + 1e-9 {
        painter.vline(
            x(d),
            plot.y_range(),
            Stroke::new(1.0, weak.gamma_multiply(0.15)),
        );
        painter.text(
            pos2(x(d), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{d}"),
            small.clone(),
            weak,
        );
        d += step;
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "distance d (km)",
        small.clone(),
        weak,
    );

    // Curves.
    let samples = 200;
    for c in curves {
        let inside: Vec<Pos2> = (0..=samples)
            .map(|k| c.epsilon * k as f64 / samples as f64)
            .map(|d| pos2(x(d), y(gaussian(d, c.sigma))))
            .collect();
        painter.line(inside, Stroke::new(2.5, c.color));

        if c.epsilon < d_max {
            let outside: Vec<Pos2> = (0..=samples)
                .map(|k| c.epsilon + (d_max - c.epsilon) * k as f64 / samples as f64)
                .map(|d| pos2(x(d), y(gaussian(d, c.sigma))))
                .collect();
            if c.cut {
                // Beyond the cut-off no edge exists: faint and dashed.
                painter.extend(Shape::dashed_line(
                    &outside,
                    Stroke::new(1.0, c.color.gamma_multiply(0.45)),
                    4.0,
                    4.0,
                ));
            } else {
                // kNN: edges can exist here too, just with small weights.
                painter.line(outside, Stroke::new(1.5, c.color.gamma_multiply(0.7)));
            }
        }

        let w_eps = gaussian(c.epsilon, c.sigma);
        let marker = [pos2(x(c.epsilon), y(w_eps)), pos2(x(c.epsilon), y(0.0))];
        if c.cut {
            // The cut-off: the weight drops to "no edge" at d = ε.
            painter.line_segment(marker, Stroke::new(1.5, c.color));
        } else {
            // kNN: only a marker for the radio range, nothing is cut.
            painter.extend(Shape::dashed_line(
                &marker,
                Stroke::new(1.0, c.color),
                3.0,
                3.0,
            ));
        }
        painter.text(
            pos2(x(c.epsilon) + 3.0, y(w_eps) - 2.0),
            Align2::LEFT_BOTTOM,
            if c.cut { "ε" } else { "range" },
            FontId::proportional(13.0),
            c.color,
        );

        // sigma: the curve's inflection point, weight exp(-1/2).
        let p_sigma = pos2(x(c.sigma), y(gaussian(c.sigma, c.sigma)));
        painter.circle_filled(p_sigma, 3.0, c.color);
        painter.text(
            p_sigma + vec2(4.0, -2.0),
            Align2::LEFT_BOTTOM,
            "σ",
            FontId::proportional(13.0),
            c.color,
        );
    }

    // Edges as dots on their curve.
    for &(d, w, col) in points {
        if d <= d_max {
            painter.circle(pos2(x(d), y(w)), 4.0, col, Stroke::new(1.0, text));
        }
    }

    // Hover readout.
    if let Some(p) = response.hover_pos()
        && plot.contains(p)
    {
        let d = ((p.x - plot.left()) / plot.width()) as f64 * d_max;
        painter.vline(
            p.x,
            plot.y_range(),
            Stroke::new(1.0, text.gamma_multiply(0.5)),
        );
        let mut lines = vec![format!("d = {d:.2} km")];
        for c in curves {
            let w = gaussian(d, c.sigma);
            if d <= c.epsilon {
                lines.push(format!("{}: w = {w:.3}", c.label));
            } else if c.cut {
                lines.push(format!("{}: no edge (would be {w:.3})", c.label));
            } else {
                lines.push(format!("{}: w = {w:.3} (beyond radio range)", c.label));
            }
        }
        let galley = painter.layout_no_wrap(lines.join("\n"), FontId::monospace(11.0), text);
        let mut pos = p + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = p.x - 8.0 - galley.size().x;
        }
        let bg = Rect::from_min_size(pos, galley.size()).expand(3.0);
        painter.rect_filled(bg, 3.0, visuals.window_fill);
        painter.galley(pos, galley, text);
    }
}

/// An arrow from just outside drone `from` to halfway towards drone `to`.
fn draw_pick(painter: &egui::Painter, from: Pos2, to: Pos2, stroke: Stroke) {
    let delta = to - from;
    let len = delta.length();
    if len < 2.0 * NODE_PX + 8.0 {
        return;
    }
    let dir = delta / len;
    let start = from + dir * (NODE_PX + 3.0);
    let tip = from + delta * 0.5 - dir * 3.0;
    painter.line_segment([start, tip], stroke);
    for angle in [0.45_f32, -0.45] {
        let back = egui::emath::Rot2::from_angle(angle) * (-dir) * 7.0;
        painter.line_segment([tip, tip + back], stroke);
    }
}

/// Points on a circle, closed (first point repeated at the end).
fn circle_points(c: Pos2, r: f32, n: usize) -> Vec<Pos2> {
    (0..=n)
        .map(|k| {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            pos2(c.x + r * a.cos(), c.y + r * a.sin())
        })
        .collect()
}

/// km to screen and back. The world is fitted into the panel, y pointing up.
struct Transform {
    left: f32,
    bottom: f32,
    scale: f32,
}

impl Transform {
    /// Fits the world into `rect`, then applies `zoom` (1 = fit) and `pan` (pixels).
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

    fn len(&self, km: f64) -> f32 {
        km as f32 * self.scale
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

    /// Finds the weight of edge (src, dst) in a graph, if it exists.
    fn weight(g: &EdgeList, src: u32, dst: u32) -> Option<f64> {
        g.edges
            .iter()
            .find(|e| e.src == src && e.dst == dst)
            .map(|e| e.weight)
    }

    #[test]
    fn test_cases_in_swarm_mode_match_the_planned_values() {
        let p = Playground {
            mode: Mode::Swarm,
            ..Default::default()
        };
        let g = p.graph();
        // Only the recon pair (0, 1) and the relay pair (4, 5) are linked.
        assert_eq!(g.edges.len(), 2);
        assert!((weight(&g, 0, 1).unwrap() - 0.278).abs() < 1e-3);
        assert!(weight(&g, 2, 3).is_none());
        assert!((weight(&g, 4, 5).unwrap() - 0.206).abs() < 1e-3);
    }

    #[test]
    fn test_cases_in_textbook_mode_ignore_roles() {
        let mut p = Playground {
            mode: Mode::Textbook,
            epsilon: 5.0,
            sigma: 2.5,
            ..Default::default()
        };
        // eps = 5 everywhere: only the 4 km pair links; the 8 km relay pair is lost.
        let g = p.graph();
        assert_eq!(g.edges.len(), 1);
        assert!(weight(&g, 4, 5).is_none());
        // eps = 9 everywhere: the 6 km recon-relay pair now links too.
        p.epsilon = 9.0;
        let g = p.graph();
        assert!(weight(&g, 2, 3).is_some());
    }

    #[test]
    fn knn_presets_match_the_knn_tests() {
        let mut p = Playground::default();
        p.load_knn_line();
        assert_eq!(p.mode, Mode::Knn);
        let g = p.graph();
        let pairs: Vec<(u32, u32)> = g.edges.iter().map(|e| (e.src, e.dst)).collect();
        assert_eq!(pairs, vec![(0, 1), (1, 2), (2, 3)]);
        // Before the union: 0 -> 1, 1 -> 0, 2 -> 1, 3 -> 2.
        assert_eq!(p.picks(), vec![(0, 1), (1, 0), (2, 1), (3, 2)]);

        p.load_knn_hub();
        let counts: Vec<usize> = p.graph().neighbours().iter().map(|v| v.len()).collect();
        assert_eq!(counts, vec![4, 1, 1, 1, 1]);
    }

    #[test]
    fn removing_a_drone_renumbers_and_fixes_selection() {
        let mut p = Playground {
            selected: Some(4),
            ..Default::default()
        };
        p.remove_node(1);
        assert_eq!(p.nodes.len(), 5);
        assert!(
            p.nodes
                .iter()
                .enumerate()
                .all(|(i, d)| d.id == DroneId(i as u32))
        );
        assert_eq!(p.selected, Some(3));
        p.remove_node(3);
        assert_eq!(p.selected, None);
    }

    #[test]
    fn transform_round_trips() {
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0));
        let t = Transform::fit(rect, 3.0, vec2(40.0, -25.0));
        let p = Vec2 { x: 3.5, y: 7.25 };
        let back = t.to_km(t.to_screen(p));
        assert!((back.x - p.x).abs() < 1e-4 && (back.y - p.y).abs() < 1e-4);
    }
}
