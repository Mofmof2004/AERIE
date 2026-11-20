//! Graph matrices playground (Mission 3): build a graph by hand, give every drone a value,
//! and see every matrix of the graph at once.
//!
//! Run with `cargo run --release -p aerie-viz --bin graph_matrices_playground`.
//!
//! - Left: presets (including the worked examples: the triangle with x = (5, 3, 4), the
//!   weighted path with x = (1, 2, 4)), tools (select, add, link, delete; drag to move),
//!   the selected drone's value x or the selected link's weight, signal presets, and which
//!   matrices to show.
//! - Centre: the graph. Links as wide as their weight, labelled with it (or with the jump
//!   Bᵀx across them); drones filled by their value x (blue below 0, red above).
//! - Right: the matrices (A, D, B, L = D − A, B Bᵀ, P = D⁻¹A, the normalised Laplacian),
//!   then the signal: x, Bᵀx, Lx and xᵀLx three ways. Hover a cell: its drone or link
//!   lights up on the graph, and a tooltip says where the number comes from. Hover a drone
//!   or a link: its row or column lights up in every matrix.
//!
//! Every matrix comes from `aerie_graph::EdgeList`: A, D and P from Mission 1, and B, L and
//! the variation from Mission 3's step 1 (`incidence_matrix`, `laplacian`, `variation`). The
//! B Bᵀ panel and the three ways of computing xᵀLx check them against each other.

use aerie_graph::{Edge, EdgeList};
use aerie_viz::grid::{Focus, HOVER_COLOR, fmt, matrix_grid, mix, show_tip};
use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, pos2, vec2,
};
use faer::Mat;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Accent colours of the matrices.
const A_COLOR: Color32 = Color32::from_rgb(78, 121, 167);
const D_COLOR: Color32 = Color32::from_rgb(242, 142, 43);
const B_COLOR: Color32 = Color32::from_rgb(225, 87, 89);
const L_COLOR: Color32 = Color32::from_rgb(176, 122, 161);
const P_COLOR: Color32 = Color32::from_rgb(89, 161, 79);
const SIGNAL_COLOR: Color32 = Color32::from_rgb(118, 183, 178);
/// Signal colours: below 0 and above 0.
const LOW_COLOR: Color32 = Color32::from_rgb(70, 120, 220);
const HIGH_COLOR: Color32 = Color32::from_rgb(225, 80, 70);
/// Drone marker radius and grab distance, pixels.
const NODE_PX: f32 = 13.0;
const GRAB_PX: f32 = 16.0;
/// The canvas always shows at least ±BOX.
const BOX: f32 = 3.0;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE graph matrices playground")
            .with_inner_size([1700.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE graph matrices playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

// ------------------------------------------------------------------------------ matrices

/// B, from `EdgeList::incidence_matrix` (Mission 3, step 1).
fn incidence(g: &EdgeList) -> Mat<f64> {
    g.incidence_matrix()
}

/// L = D − A, from `EdgeList::laplacian`.
fn laplacian(g: &EdgeList) -> Mat<f64> {
    g.laplacian()
}

/// xᵀLx, from `EdgeList::variation`.
fn variation(g: &EdgeList, x: &[f64]) -> f64 {
    g.variation(x)
}

/// M Mᵀ (small matrices: plain loops).
fn times_own_transpose(m: &Mat<f64>) -> Mat<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = Mat::zeros(r, r);
    for i in 0..r {
        for j in 0..r {
            out[(i, j)] = (0..c).map(|k| m[(i, k)] * m[(j, k)]).sum();
        }
    }
    out
}

/// M v as a column (k × 1).
fn times(m: &Mat<f64>, v: &[f64]) -> Mat<f64> {
    let mut out = Mat::zeros(m.nrows(), 1);
    for i in 0..m.nrows() {
        out[(i, 0)] = (0..m.ncols()).map(|k| m[(i, k)] * v[k]).sum();
    }
    out
}

/// Mᵀ v as a column.
fn transpose_times(m: &Mat<f64>, v: &[f64]) -> Mat<f64> {
    let mut out = Mat::zeros(m.ncols(), 1);
    for j in 0..m.ncols() {
        out[(j, 0)] = (0..m.nrows()).map(|k| m[(k, j)] * v[k]).sum();
    }
    out
}

/// P = D⁻¹A: row i of A divided by i's degree (an isolated drone: a self-loop, as in
/// `EdgeList::transition_matrix`).
fn transition(g: &EdgeList) -> Mat<f64> {
    g.transition_matrix()
}

/// The normalised Laplacian D^-1/2 L D^-1/2 (rows and columns of isolated drones: 0).
fn normalized_laplacian(g: &EdgeList) -> Mat<f64> {
    let l = laplacian(g);
    let deg = g.degrees();
    let s: Vec<f64> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let n = l.nrows();
    let mut out = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = s[i] * l[(i, j)] * s[j];
        }
    }
    out
}

fn column(v: &[f64]) -> Mat<f64> {
    let mut m = Mat::zeros(v.len(), 1);
    for (i, &x) in v.iter().enumerate() {
        m[(i, 0)] = x;
    }
    m
}

fn same(a: &Mat<f64>, b: &Mat<f64>) -> bool {
    a.nrows() == b.nrows()
        && a.ncols() == b.ncols()
        && (0..a.nrows()).all(|i| (0..a.ncols()).all(|j| (a[(i, j)] - b[(i, j)]).abs() < 1e-9))
}

// --------------------------------------------------------------------------------- graph

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Preset {
    Triangle,
    WeightedPath,
    Star,
    SquareDiagonal,
    TwoTriangles,
    TwoPieces,
    Empty,
}

impl Preset {
    const ALL: [Preset; 7] = [
        Preset::Triangle,
        Preset::WeightedPath,
        Preset::Star,
        Preset::SquareDiagonal,
        Preset::TwoTriangles,
        Preset::TwoPieces,
        Preset::Empty,
    ];

    fn name(self) -> &'static str {
        match self {
            Preset::Triangle => "Triangle, x = (5, 3, 4)",
            Preset::WeightedPath => "Weighted path, x = (1, 2, 4)",
            Preset::Star => "Star (a hub and 4 leaves)",
            Preset::SquareDiagonal => "Square with a diagonal",
            Preset::TwoTriangles => "Two triangles + a bridge",
            Preset::TwoPieces => "Two pieces (not connected)",
            Preset::Empty => "Empty: build your own",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Preset::Triangle => "The incidence example: Bᵀx = (2, 1, −1), so xᵀLx = 4 + 1 + 1 = 6.",
            Preset::WeightedPath => {
                "The worked example, weights 1 and 4: B has ±√4 = ±2 in its second column, \
                 L = [[1, −1, 0], [−1, 5, −4], [0, −4, 4]], xᵀLx = 1 + 16 = 17."
            }
            Preset::Star => "The hub has degree 4. Raise its value: every link lights up at once.",
            Preset::SquareDiagonal => {
                "4 drones, 5 links: B is 4 × 5. Drones 0 and 2 have degree 3, the others 2."
            }
            Preset::TwoTriangles => {
                "Two groups and one bridge: give each group its own value, and only the \
                 bridge has a jump."
            }
            Preset::TwoPieces => {
                "Two pieces with no link between them. Give each piece a different constant: \
                 what is xᵀLx? What does that say about L?"
            }
            Preset::Empty => "Nothing yet: pick ➕ Add and click to place drones, then 🔗 Link.",
        }
    }
}

/// A preset: positions, links (src, dst, weight) and the signal x.
type Layout = (Vec<[f32; 2]>, Vec<(u32, u32, f64)>, Vec<f64>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Add,
    Link,
    Delete,
}

/// What the pointer is on, in the matrices or on the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Drone(usize),
    Link(usize),
    /// A cell of an n × n matrix: drones i and j (and their link, if any).
    Pair(usize, usize),
    /// A cell of B: drone i and link e.
    DroneLink(usize, usize),
}

/// Which matrices to show.
#[derive(Debug, Clone, Copy)]
struct Shown {
    a: bool,
    d: bool,
    b: bool,
    l: bool,
    bbt: bool,
    p: bool,
    l_norm: bool,
    signal: bool,
}

impl Default for Shown {
    fn default() -> Self {
        Shown {
            a: true,
            d: true,
            b: true,
            l: true,
            bbt: true,
            p: false,
            l_norm: false,
            signal: true,
        }
    }
}

struct Playground {
    preset: Preset,
    /// Drone positions (any units; the canvas fits them).
    pos: Vec<[f32; 2]>,
    /// The signal: one value per drone.
    x: Vec<f64>,
    graph: EdgeList,
    tool: Tool,
    /// Weight of new links.
    new_weight: f64,
    selected: Option<Target>,
    link_from: Option<usize>,
    dragging: Option<usize>,
    /// Hovered last frame (matrix cell or graph element): drawn on both sides.
    hover: Option<Target>,
    shown: Shown,
    /// Link labels: the weight (false) or the jump Bᵀx (true).
    show_jumps: bool,
    seed: u64,
}

impl Default for Playground {
    fn default() -> Self {
        let mut p = Playground {
            preset: Preset::WeightedPath,
            pos: Vec::new(),
            x: Vec::new(),
            graph: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            tool: Tool::Select,
            new_weight: 1.0,
            selected: None,
            link_from: None,
            dragging: None,
            hover: None,
            shown: Shown::default(),
            show_jumps: true,
            seed: 1,
        };
        p.load(Preset::WeightedPath);
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let previous = self.hover.take();
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });
        egui::Panel::right("matrices")
            .resizable(true)
            .default_size(760.0)
            .show(ui, |ui| {
                egui::ScrollArea::both().show(ui, |ui| self.matrices(ui, previous));
            });
        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui, previous));
    }
}

impl Playground {
    fn n(&self) -> usize {
        self.pos.len()
    }

    fn load(&mut self, preset: Preset) {
        self.preset = preset;
        let (pos, links, x): Layout = match preset {
            Preset::Triangle => (
                vec![[-1.5, 1.0], [1.5, 1.0], [0.0, -1.6]],
                vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)],
                vec![5.0, 3.0, 4.0],
            ),
            Preset::WeightedPath => (
                vec![[-2.2, 0.0], [0.0, 0.0], [2.2, 0.0]],
                vec![(0, 1, 1.0), (1, 2, 4.0)],
                vec![1.0, 2.0, 4.0],
            ),
            Preset::Star => (
                vec![[0.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]],
                vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
                vec![3.0, 1.0, 1.0, 1.0, 1.0],
            ),
            Preset::SquareDiagonal => (
                vec![[-1.6, -1.6], [1.6, -1.6], [1.6, 1.6], [-1.6, 1.6]],
                vec![
                    (0, 1, 1.0),
                    (1, 2, 1.0),
                    (2, 3, 1.0),
                    (0, 3, 1.0),
                    (0, 2, 2.0),
                ],
                vec![1.0, 2.0, 1.0, 0.0],
            ),
            Preset::TwoTriangles => (
                vec![
                    [-2.6, -1.0],
                    [-2.6, 1.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [2.6, -1.0],
                    [2.6, 1.0],
                ],
                vec![
                    (0, 1, 1.0),
                    (0, 2, 1.0),
                    (1, 2, 1.0),
                    (2, 3, 1.0),
                    (3, 4, 1.0),
                    (3, 5, 1.0),
                    (4, 5, 1.0),
                ],
                vec![1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            ),
            Preset::TwoPieces => (
                vec![
                    [-2.4, -1.0],
                    [-2.4, 1.0],
                    [-0.8, 0.0],
                    [1.2, -1.0],
                    [2.6, 0.5],
                ],
                vec![(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (3, 4, 1.0)],
                vec![2.0, 2.0, 2.0, 5.0, 5.0],
            ),
            Preset::Empty => (Vec::new(), Vec::new(), Vec::new()),
        };
        self.pos = pos;
        self.x = x;
        self.graph = EdgeList {
            n: self.pos.len() as u32,
            edges: links
                .into_iter()
                .map(|(src, dst, weight)| Edge { src, dst, weight })
                .collect(),
        };
        self.selected = None;
        self.link_from = None;
    }

    fn add_drone(&mut self, p: [f32; 2]) {
        self.pos.push(p);
        self.x.push(0.0);
        self.graph.n = self.pos.len() as u32;
    }

    /// Deletes drone `i` and its links; drones after it move down by one.
    fn remove_drone(&mut self, i: usize) {
        self.pos.remove(i);
        self.x.remove(i);
        let i = i as u32;
        self.graph.edges.retain(|e| e.src != i && e.dst != i);
        for e in &mut self.graph.edges {
            if e.src > i {
                e.src -= 1;
            }
            if e.dst > i {
                e.dst -= 1;
            }
        }
        self.graph.n = self.pos.len() as u32;
        self.selected = None;
    }

    /// Adds the link a–b (weight `new_weight`) if it is missing, removes it if it is there.
    fn toggle_link(&mut self, a: usize, b: usize) {
        let (src, dst) = (a.min(b) as u32, a.max(b) as u32);
        let before = self.graph.edges.len();
        self.graph.edges.retain(|e| (e.src, e.dst) != (src, dst));
        if self.graph.edges.len() == before {
            self.graph.edges.push(Edge {
                src,
                dst,
                weight: self.new_weight,
            });
        }
        self.selected = None;
    }

    fn link_name(&self, e: usize) -> String {
        let edge = &self.graph.edges[e];
        format!("{}–{}", edge.src, edge.dst)
    }

    fn set_signal(&mut self, kind: &str) {
        let n = self.n();
        self.x = match kind {
            "constant" => vec![1.0; n],
            "left to right" => self
                .pos
                .iter()
                .map(|p| (p[0] as f64 * 2.0).round() / 2.0)
                .collect(),
            "alternating" => (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            "spike" => {
                let at = match self.selected {
                    Some(Target::Drone(i)) => i,
                    _ => 0,
                };
                (0..n).map(|i| if i == at { 1.0 } else { 0.0 }).collect()
            }
            _ => {
                self.seed += 1;
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                (0..n)
                    .map(|_| (rng.random_range(-3.0..3.0_f64) * 2.0).round() / 2.0)
                    .collect()
            }
        };
    }

    // ---------------------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph matrices playground");
        ui.weak("Build a graph, give each drone a value x, and see every matrix of it.");
        ui.separator();

        ui.label(RichText::new("Graph").strong());
        let mut chosen = self.preset;
        egui::ComboBox::from_id_salt("preset")
            .selected_text(self.preset.name())
            .width(260.0)
            .show_ui(ui, |ui| {
                for p in Preset::ALL {
                    ui.selectable_value(&mut chosen, p, p.name());
                }
            });
        if chosen != self.preset {
            self.load(chosen);
        }
        ui.weak(self.preset.about());
        if ui.small_button("Reload preset").clicked() {
            self.load(self.preset);
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Tool:");
            ui.selectable_value(&mut self.tool, Tool::Select, "👆 Select");
            ui.selectable_value(&mut self.tool, Tool::Add, "➕ Add");
            ui.selectable_value(&mut self.tool, Tool::Link, "🔗 Link");
            ui.selectable_value(&mut self.tool, Tool::Delete, "✖ Delete");
        });
        ui.weak(match self.tool {
            Tool::Select => "Click a drone (its value x) or a link (its weight) to edit it.",
            Tool::Add => "Click empty space: a new drone with x = 0.",
            Tool::Link => "Click a drone, then another: adds the link, or removes it.",
            Tool::Delete => "Click a drone or a link: deletes it.",
        });
        ui.weak("Any tool: drag a drone to move it.");
        ui.horizontal(|ui| {
            ui.label("New links get weight");
            ui.add(
                egui::DragValue::new(&mut self.new_weight)
                    .range(0.01..=100.0)
                    .speed(0.1),
            );
        });
        ui.label(format!(
            "{} drones, {} links: A, D, L are {n} × {n}, B is {n} × {}.",
            self.n(),
            self.graph.edges.len(),
            self.graph.edges.len(),
            n = self.n()
        ));
        ui.separator();

        ui.label(RichText::new("Selected").strong());
        match self.selected {
            Some(Target::Drone(i)) if i < self.n() => {
                ui.horizontal(|ui| {
                    ui.label(format!("drone {i}: x ="));
                    ui.add(egui::DragValue::new(&mut self.x[i]).speed(0.1));
                });
                ui.add(egui::Slider::new(&mut self.x[i], -10.0..=10.0).step_by(0.5));
            }
            Some(Target::Link(e)) if e < self.graph.edges.len() => {
                let name = self.link_name(e);
                ui.horizontal(|ui| {
                    ui.label(format!("link {name}: w ="));
                    ui.add(
                        egui::DragValue::new(&mut self.graph.edges[e].weight)
                            .range(0.01..=100.0)
                            .speed(0.1),
                    );
                });
                ui.add(
                    egui::Slider::new(&mut self.graph.edges[e].weight, 0.01..=16.0)
                        .logarithmic(true),
                );
            }
            _ => {
                ui.weak("Nothing: 👆 Select a drone or a link.");
            }
        }
        ui.separator();

        ui.label(RichText::new("Signal x").strong());
        ui.weak("One value per drone (in Mission 3: the jamming power each drone feels).");
        ui.horizontal_wrapped(|ui| {
            for kind in [
                "constant",
                "left to right",
                "alternating",
                "spike",
                "random",
            ] {
                if ui.small_button(kind).clicked() {
                    self.set_signal(kind);
                }
            }
        });
        ui.weak("spike: 1 at the selected drone, 0 elsewhere.");
        ui.checkbox(
            &mut self.show_jumps,
            "Label links with their jump Bᵀx (else the weight)",
        );
        ui.separator();

        ui.label(RichText::new("Show").strong());
        let s = &mut self.shown;
        ui.checkbox(&mut s.a, "A · adjacency");
        ui.checkbox(&mut s.d, "D · degree");
        ui.checkbox(&mut s.b, "B · incidence");
        ui.checkbox(&mut s.l, "L = D − A · Laplacian");
        ui.checkbox(&mut s.bbt, "B Bᵀ (should equal L)");
        ui.checkbox(&mut s.p, "P = D⁻¹A · transition (Mission 1)");
        ui.checkbox(&mut s.l_norm, "D^-1/2 L D^-1/2 · normalised Laplacian");
        ui.checkbox(&mut s.signal, "The signal: x, Bᵀx, Lx, xᵀLx");
    }

    // ------------------------------------------------------------------------ matrices

    fn matrices(&mut self, ui: &mut egui::Ui, previous: Option<Target>) {
        let n = self.n();
        let m = self.graph.edges.len();
        ui.heading("Matrices");
        if n == 0 {
            ui.weak("Add drones on the graph.");
            return;
        }
        let heads: Vec<(String, Color32)> = (0..n)
            .map(|i| (i.to_string(), ui.visuals().strong_text_color()))
            .collect();
        let link_heads: Vec<(String, Color32)> = (0..m)
            .map(|e| (self.link_name(e), ui.visuals().strong_text_color()))
            .collect();
        // Focus: the hovered or selected drone's row and column, the hovered cell.
        let drone = match previous.or(self.selected) {
            Some(Target::Drone(i)) => Some(i),
            _ => None,
        };
        let link = match previous.or(self.selected) {
            Some(Target::Link(e)) | Some(Target::DroneLink(_, e)) => Some(e),
            _ => None,
        };
        let pair_cell = match previous {
            Some(Target::Pair(i, j)) => Some((i, j)),
            _ => None,
        };
        let square = Focus {
            row: drone,
            col: drone,
            cell: pair_cell,
        };
        let b_focus = Focus {
            row: drone,
            col: link,
            cell: match previous {
                Some(Target::DroneLink(i, e)) => Some((i, e)),
                _ => None,
            },
        };
        let a = self.graph.adjacency_matrix();
        let d = self.graph.degree_matrix();
        let b = incidence(&self.graph);
        let l = laplacian(&self.graph);
        let deg = self.graph.degrees();
        let mut hovered: Option<Target> = None;
        let w = |i: usize, j: usize| a[(i, j)];

        if self.shown.a {
            title(
                ui,
                "A · adjacency",
                A_COLOR,
                "A[i][j] = weight of the link i–j, 0 when there is none.",
            );
            let (h, resp) = matrix_grid(ui, &a, &heads, &heads, square, A_COLOR, Some("Σ = deg"));
            if let Some((i, j)) = h {
                show_tip(
                    resp,
                    &if i == j {
                        format!("A[{i}][{i}] = 0: no self-loops.")
                    } else if w(i, j) == 0.0 {
                        format!("A[{i}][{j}] = 0: no link.")
                    } else {
                        format!(
                            "A[{i}][{j}] = {}: the weight of link {}–{}.",
                            fmt(w(i, j)),
                            i.min(j),
                            i.max(j)
                        )
                    },
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.d {
            title(
                ui,
                "D · degree",
                D_COLOR,
                "Diagonal: D[i][i] = Σ_j A[i][j], the weighted degree.",
            );
            let (h, resp) = matrix_grid(ui, &d, &heads, &heads, square, D_COLOR, None);
            if let Some((i, j)) = h {
                show_tip(
                    resp,
                    &if i == j {
                        format!("D[{i}][{i}] = sum of row {i} of A = {}", fmt(deg[i]))
                    } else {
                        "Off the diagonal: always 0. A degree belongs to one drone.".to_string()
                    },
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.b {
            title(
                ui,
                "B · incidence (rows: drones, columns: links)",
                B_COLOR,
                "Column e = link src–dst of weight w: +√w at src, −√w at dst, 0 elsewhere.",
            );
            if m == 0 {
                ui.weak("No link: B has no column.");
            } else {
                let (h, resp) = matrix_grid(ui, &b, &heads, &link_heads, b_focus, B_COLOR, None);
                if let Some((i, e)) = h {
                    let edge = &self.graph.edges[e];
                    let text = if i == edge.src as usize {
                        format!(
                            "B[{i}][{}] = +√{} = {}: drone {i} is the first end (src) of this link.",
                            self.link_name(e),
                            fmt(edge.weight),
                            fmt(b[(i, e)])
                        )
                    } else if i == edge.dst as usize {
                        format!(
                            "B[{i}][{}] = −√{} = {}: drone {i} is the second end (dst).",
                            self.link_name(e),
                            fmt(edge.weight),
                            fmt(b[(i, e)])
                        )
                    } else {
                        format!(
                            "B[{i}][{}] = 0: drone {i} is not on this link.",
                            self.link_name(e)
                        )
                    };
                    show_tip(resp, &text);
                    hovered = Some(Target::DroneLink(i, e));
                }
            }
        }
        if self.shown.l {
            title(
                ui,
                "L = D − A · Laplacian",
                L_COLOR,
                "Diagonal: the degree. Off it: −weight of the link (0 if none). Every row sums to 0.",
            );
            let (h, resp) = matrix_grid(ui, &l, &heads, &heads, square, L_COLOR, Some("Σ"));
            if let Some((i, j)) = h {
                show_tip(
                    resp,
                    &if i == j {
                        format!(
                            "L[{i}][{i}] = D[{i}][{i}] − A[{i}][{i}] = {} − 0 = {}",
                            fmt(deg[i]),
                            fmt(l[(i, i)])
                        )
                    } else {
                        format!(
                            "L[{i}][{j}] = D[{i}][{j}] − A[{i}][{j}] = 0 − {} = {}",
                            fmt(w(i, j)),
                            fmt(l[(i, j)])
                        )
                    },
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.bbt {
            let bbt = times_own_transpose(&b);
            let ok = same(&bbt, &l);
            title(
                ui,
                &format!("B Bᵀ {}", if ok { "= L ✔" } else { "≠ L ✘" }),
                L_COLOR,
                "(B Bᵀ)[i][j] = Σ over links e of B[i][e] · B[j][e]: row i of B times row j.",
            );
            let (h, resp) = matrix_grid(ui, &bbt, &heads, &heads, square, L_COLOR, None);
            if let Some((i, j)) = h {
                let terms: Vec<String> = (0..m)
                    .filter(|&e| b[(i, e)] != 0.0 && b[(j, e)] != 0.0)
                    .map(|e| {
                        format!(
                            "({})·({}) on {}",
                            fmt(b[(i, e)]),
                            fmt(b[(j, e)]),
                            self.link_name(e)
                        )
                    })
                    .collect();
                show_tip(
                    resp,
                    &if terms.is_empty() {
                        format!("(B Bᵀ)[{i}][{j}] = 0: no link has both {i} and {j}.")
                    } else {
                        format!(
                            "(B Bᵀ)[{i}][{j}] = {} = {}",
                            terms.join(" + "),
                            fmt(bbt[(i, j)])
                        )
                    },
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.p {
            let p = transition(&self.graph);
            title(
                ui,
                "P = D⁻¹A · transition",
                P_COLOR,
                "Row i of A divided by i's degree: where a random walk at i steps next. Rows sum to 1.",
            );
            let (h, resp) = matrix_grid(ui, &p, &heads, &heads, square, P_COLOR, Some("Σ"));
            if let Some((i, j)) = h {
                show_tip(
                    resp,
                    &if deg[i] == 0.0 {
                        format!("Drone {i} has no link: it stays put (P[{i}][{i}] = 1).")
                    } else {
                        format!(
                            "P[{i}][{j}] = A[{i}][{j}] / D[{i}][{i}] = {} / {} = {}",
                            fmt(w(i, j)),
                            fmt(deg[i]),
                            fmt(p[(i, j)])
                        )
                    },
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.l_norm {
            let ln = normalized_laplacian(&self.graph);
            title(
                ui,
                "D^-1/2 L D^-1/2 · normalised Laplacian",
                L_COLOR,
                "L[i][j] / √(deg i · deg j): 1 on the diagonal, so busy drones don't dominate.",
            );
            let (h, resp) = matrix_grid(ui, &ln, &heads, &heads, square, L_COLOR, None);
            if let Some((i, j)) = h {
                show_tip(
                    resp,
                    &format!(
                        "L[{i}][{j}] / √({} · {}) = {} / {} = {}",
                        fmt(deg[i]),
                        fmt(deg[j]),
                        fmt(l[(i, j)]),
                        fmt((deg[i] * deg[j]).sqrt()),
                        fmt(ln[(i, j)])
                    ),
                );
            }
            hovered = hovered.or(h.map(|(i, j)| Target::Pair(i, j)));
        }
        if self.shown.signal {
            self.signal_panel(ui, &b, &l, &heads, &link_heads, drone, link, &mut hovered);
        }
        self.hover = self.hover.or(hovered);
    }

    #[allow(clippy::too_many_arguments)]
    fn signal_panel(
        &self,
        ui: &mut egui::Ui,
        b: &Mat<f64>,
        l: &Mat<f64>,
        heads: &[(String, Color32)],
        link_heads: &[(String, Color32)],
        drone: Option<usize>,
        link: Option<usize>,
        hovered: &mut Option<Target>,
    ) {
        let x = &self.x;
        let m = self.graph.edges.len();
        ui.add_space(10.0);
        ui.separator();
        ui.heading("The signal");
        egui::Grid::new("signal_meaning")
            .num_columns(3)
            .spacing(vec2(12.0, 2.0))
            .show(ui, |ui| {
                ui.label(RichText::new("Bᵀx").strong().color(B_COLOR));
                ui.label("per link");
                ui.label(
                    "the jump √w (x_i − x_j) across it: where x changes (a slope); \
                     the sign says the direction of change (+ : x_i is higher)",
                );
                ui.end_row();
                ui.label(RichText::new("Lx").strong().color(L_COLOR));
                ui.label("per drone");
                ui.label("deg · (x_i − neighbours' average): > 0 peak, < 0 dip (a curvature)");
                ui.end_row();
                ui.label(RichText::new("xᵀLx").strong());
                ui.label("one number");
                ui.label("x · Lx = |Bᵀx|²: how bumpy x is overall");
                ui.end_row();
            });
        ui.add_space(4.0);
        let one = vec![(String::new(), ui.visuals().strong_text_color())];
        let bx = transpose_times(b, x);
        let lx = times(l, x);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("x").strong().color(SIGNAL_COLOR));
                ui.weak("per drone");
                let (h, _) = matrix_grid(ui, &column(x), heads, &one, Focus { row: drone, col: None, cell: None }, SIGNAL_COLOR, None);
                if let Some((i, _)) = h {
                    *hovered = Some(Target::Drone(i));
                }
            });
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.label(RichText::new("Bᵀx").strong().color(B_COLOR));
                ui.weak("jump per link");
                if m == 0 {
                    ui.weak("no link");
                } else {
                    let (h, resp) = matrix_grid(ui, &bx, link_heads, &one, Focus { row: link, col: None, cell: None }, B_COLOR, None);
                    if let Some((e, _)) = h {
                        let edge = &self.graph.edges[e];
                        let (s, t) = (edge.src as usize, edge.dst as usize);
                        show_tip(resp, &format!(
                            "(Bᵀx)[{}] = √{} · (x_{s} − x_{t}) = {} · ({} − {}) = {}",
                            self.link_name(e), fmt(edge.weight), fmt(edge.weight.sqrt()), fmt(x[s]), fmt(x[t]), fmt(bx[(e, 0)])
                        ));
                        *hovered = Some(Target::Link(e));
                    }
                }
            });
            ui.add_space(16.0);
            ui.vertical(|ui| {
                ui.label(RichText::new("Lx").strong().color(L_COLOR));
                ui.weak("peak / dip");
                let (h, resp) = matrix_grid(ui, &lx, heads, &one, Focus { row: drone, col: None, cell: None }, L_COLOR, None);
                if let Some((i, _)) = h {
                    let terms: Vec<String> = self
                        .graph
                        .edges
                        .iter()
                        .filter(|e| e.src as usize == i || e.dst as usize == i)
                        .map(|e| {
                            let j = if e.src as usize == i { e.dst } else { e.src } as usize;
                            format!("{}·({} − {})", fmt(e.weight), fmt(x[i]), fmt(x[j]))
                        })
                        .collect();
                    show_tip(resp, &if terms.is_empty() {
                        format!("(Lx)[{i}] = 0: drone {i} has no link.")
                    } else {
                        format!(
                            "(Lx)[{i}] = Σ over i's links w (x_i − x_j) = {} = {}\n= degree · x_i − Σ w x_j: how far drone {i} sits above the weighted sum of its neighbours.",
                            terms.join(" + "),
                            fmt(lx[(i, 0)])
                        )
                    });
                    *hovered = Some(Target::Drone(i));
                }
            });
        });

        ui.add_space(8.0);
        ui.label(RichText::new("xᵀLx: how bumpy x is on this graph, three ways").strong());
        let by_links = variation(&self.graph, x);
        let terms: Vec<String> = self
            .graph
            .edges
            .iter()
            .map(|e| {
                format!(
                    "{}·({} − {})²",
                    fmt(e.weight),
                    fmt(x[e.src as usize]),
                    fmt(x[e.dst as usize])
                )
            })
            .collect();
        mono(
            ui,
            format!(
                "Σ over links w (x_i − x_j)² = {} = {}",
                if terms.is_empty() {
                    "0".into()
                } else {
                    terms.join(" + ")
                },
                fmt(by_links)
            ),
        );
        let squares: Vec<String> = (0..m).map(|e| format!("({})²", fmt(bx[(e, 0)]))).collect();
        let norm: f64 = (0..m).map(|e| bx[(e, 0)].powi(2)).sum();
        mono(
            ui,
            format!(
                "|Bᵀx|² = {} = {}",
                if squares.is_empty() {
                    "0".into()
                } else {
                    squares.join(" + ")
                },
                fmt(norm)
            ),
        );
        let products: Vec<String> = (0..x.len())
            .map(|i| format!("{}·{}", fmt(x[i]), fmt(lx[(i, 0)])))
            .collect();
        let xlx: f64 = (0..x.len()).map(|i| x[i] * lx[(i, 0)]).sum();
        mono(
            ui,
            format!("x · (Lx) = {} = {}", products.join(" + "), fmt(xlx)),
        );
        let ok = (by_links - norm).abs() < 1e-9 && (by_links - xlx).abs() < 1e-9;
        ui.colored_label(
            if ok { P_COLOR } else { B_COLOR },
            if ok {
                "✔ All three agree."
            } else {
                "✘ They don't agree."
            },
        );
        ui.weak(
            "0 means linked drones all agree. Tikhonov (Mission 3) keeps the reconstructed \
             map close to the readings while making this number small.",
        );
    }

    // -------------------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui, previous: Option<Target>) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();
        painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
        let fit = Fit::new(&self.pos, rect.shrink(40.0));
        let at = |p: [f32; 2]| fit.screen(p);

        // What is under the pointer: a drone first, else a link.
        let node_at = |p: Pos2, pos: &[[f32; 2]]| {
            (0..pos.len())
                .map(|i| (i, at(pos[i]).distance(p)))
                .filter(|&(_, d)| d < GRAB_PX)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        };
        let link_at = |p: Pos2, pos: &[[f32; 2]], g: &EdgeList| {
            g.edges
                .iter()
                .enumerate()
                .map(|(e, edge)| {
                    (
                        e,
                        segment_distance(p, at(pos[edge.src as usize]), at(pos[edge.dst as usize])),
                    )
                })
                .filter(|&(_, d)| d < 7.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(e, _)| e)
        };
        let pointer = response.hover_pos();
        let hit_node = pointer.and_then(|p| node_at(p, &self.pos));
        let hit_link = if hit_node.is_none() {
            pointer.and_then(|p| link_at(p, &self.pos, &self.graph))
        } else {
            None
        };
        if let Some(i) = hit_node {
            self.hover = Some(Target::Drone(i));
        } else if let Some(e) = hit_link {
            self.hover = Some(Target::Link(e));
        }

        // Input.
        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.dragging = origin.and_then(|p| node_at(p, &self.pos));
        }
        if let (Some(i), Some(p)) = (self.dragging, response.interact_pointer_pos())
            && response.dragged()
            && i < self.n()
        {
            self.pos[i] = fit.world(p);
        }
        if response.drag_stopped() {
            self.dragging = None;
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let node = node_at(p, &self.pos);
            let link = if node.is_none() {
                link_at(p, &self.pos, &self.graph)
            } else {
                None
            };
            match (self.tool, node, link) {
                (Tool::Select, Some(i), _) => self.selected = Some(Target::Drone(i)),
                (Tool::Select, None, Some(e)) => self.selected = Some(Target::Link(e)),
                (Tool::Add, None, _) => {
                    self.add_drone(fit.world(p));
                    self.selected = Some(Target::Drone(self.n() - 1));
                }
                (Tool::Link, Some(b), _) => match self.link_from {
                    Some(a) if a != b => {
                        self.toggle_link(a, b);
                        self.link_from = None;
                    }
                    Some(_) => self.link_from = None,
                    None => self.link_from = Some(b),
                },
                (Tool::Delete, Some(i), _) => self.remove_drone(i),
                (Tool::Delete, None, Some(e)) => {
                    self.graph.edges.remove(e);
                    self.selected = None;
                }
                _ => {
                    self.selected = None;
                    self.link_from = None;
                }
            }
        }

        let n = self.n();
        if n == 0 {
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                "Empty graph: pick ➕ Add and click here.",
                FontId::proportional(15.0),
                text,
            );
            return;
        }
        let fit = Fit::new(&self.pos, rect.shrink(40.0));
        let at = |i: usize| fit.screen(self.pos[i]);
        let lit = previous.or(self.hover);
        let lit_drones: Vec<usize> = match lit {
            Some(Target::Drone(i)) => vec![i],
            Some(Target::Pair(i, j)) => vec![i, j],
            Some(Target::DroneLink(i, _)) => vec![i],
            _ => Vec::new(),
        };
        let lit_link: Option<usize> = match lit {
            Some(Target::Link(e)) | Some(Target::DroneLink(_, e)) => Some(e),
            Some(Target::Pair(i, j)) => {
                let key = (i.min(j) as u32, i.max(j) as u32);
                self.graph.edges.iter().position(|e| (e.src, e.dst) == key)
            }
            _ => None,
        };

        // Links: as wide as their weight, labelled with their weight or their jump.
        let max_w = self
            .graph
            .edges
            .iter()
            .map(|e| e.weight)
            .fold(1e-12, f64::max);
        let max_jump = self
            .graph
            .edges
            .iter()
            .map(|e| (e.weight.sqrt() * (self.x[e.src as usize] - self.x[e.dst as usize])).abs())
            .fold(1e-12, f64::max);
        for (k, e) in self.graph.edges.iter().enumerate() {
            let (s, d) = (e.src as usize, e.dst as usize);
            let (a, b) = (at(s), at(d));
            let jump = e.weight.sqrt() * (self.x[s] - self.x[d]);
            let mut color = if self.show_jumps {
                mix(
                    text.gamma_multiply(0.35),
                    B_COLOR,
                    (jump.abs() / max_jump) as f32,
                )
            } else {
                text.gamma_multiply(0.4)
            };
            if lit_link == Some(k) || self.selected == Some(Target::Link(k)) {
                color = HOVER_COLOR;
            }
            painter.line_segment(
                [a, b],
                Stroke::new(1.5 + 5.0 * (e.weight / max_w).sqrt() as f32, color),
            );
            let label = if self.show_jumps {
                format!("{}: Bᵀx = {}", self.link_name(k), fmt(jump))
            } else {
                format!("{}: w = {}", self.link_name(k), fmt(e.weight))
            };
            let mid = a + (b - a) * 0.5;
            let normal = vec2(-(b - a).y, (b - a).x).normalized() * 12.0;
            painter.text(
                mid + normal,
                Align2::CENTER_CENTER,
                label,
                FontId::proportional(12.0),
                color,
            );
        }

        // Drones: filled by their value.
        let top = self.x.iter().map(|v| v.abs()).fold(1e-12, f64::max);
        for i in 0..n {
            let c = at(i);
            let v = self.x[i];
            let fill = if v >= 0.0 {
                mix(visuals.faint_bg_color, HIGH_COLOR, (v / top) as f32)
            } else {
                mix(visuals.faint_bg_color, LOW_COLOR, (-v / top) as f32)
            };
            painter.circle_filled(c, NODE_PX, fill);
            let ring = if lit_drones.contains(&i) || self.selected == Some(Target::Drone(i)) {
                Stroke::new(3.0, HOVER_COLOR)
            } else {
                Stroke::new(1.0, text.gamma_multiply(0.6))
            };
            painter.circle_stroke(c, NODE_PX, ring);
            painter.text(
                c,
                Align2::CENTER_CENTER,
                i.to_string(),
                FontId::proportional(13.0),
                text,
            );
            painter.text(
                c + vec2(0.0, NODE_PX + 10.0),
                Align2::CENTER_CENTER,
                format!("x = {}", fmt(v)),
                FontId::proportional(12.0),
                text,
            );
        }
        if let Some(a) = self.link_from {
            painter.circle_stroke(at(a), NODE_PX + 6.0, Stroke::new(1.5, text));
            painter.text(
                rect.left_bottom() + vec2(8.0, -8.0),
                Align2::LEFT_BOTTOM,
                format!("Drone {a} picked: click another drone."),
                FontId::proportional(12.0),
                text,
            );
        }
    }
}

fn title(ui: &mut egui::Ui, name: &str, color: Color32, about: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(name).strong().color(color));
    ui.weak(about);
}

fn mono(ui: &mut egui::Ui, t: String) {
    ui.label(RichText::new(t).monospace());
}

/// Distance from p to the segment a–b, pixels.
fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// World ↔ screen for the canvas: fits the drones and at least ±BOX.
struct Fit {
    min: [f32; 2],
    scale: f32,
    offset: egui::Vec2,
}

impl Fit {
    fn new(pos: &[[f32; 2]], rect: Rect) -> Fit {
        let (mut lo, mut hi) = ([-BOX; 2], [BOX; 2]);
        for p in pos {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let span = [hi[0] - lo[0], hi[1] - lo[1]];
        let scale = (rect.width() / span[0]).min(rect.height() / span[1]);
        let used = vec2(span[0] * scale, span[1] * scale);
        Fit {
            min: lo,
            scale,
            offset: rect.min.to_vec2() + (rect.size() - used) / 2.0,
        }
    }

    fn screen(&self, p: [f32; 2]) -> Pos2 {
        pos2(
            (p[0] - self.min[0]) * self.scale,
            (p[1] - self.min[1]) * self.scale,
        ) + self.offset
    }

    fn world(&self, p: Pos2) -> [f32; 2] {
        let q = p - self.offset;
        [
            q.x / self.scale + self.min[0],
            q.y / self.scale + self.min[1],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(m: &Mat<f64>) -> Vec<Vec<f64>> {
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
            .collect()
    }

    #[test]
    fn the_worked_example() {
        let mut p = Playground::default();
        p.load(Preset::WeightedPath);
        assert_eq!(
            rows(&incidence(&p.graph)),
            vec![vec![1.0, 0.0], vec![-1.0, 2.0], vec![0.0, -2.0]]
        );
        assert_eq!(
            rows(&laplacian(&p.graph)),
            vec![
                vec![1.0, -1.0, 0.0],
                vec![-1.0, 5.0, -4.0],
                vec![0.0, -4.0, 4.0]
            ]
        );
        assert_eq!(variation(&p.graph, &p.x), 17.0);
        p.load(Preset::Triangle);
        assert_eq!(
            rows(&transpose_times(&incidence(&p.graph), &p.x)),
            vec![vec![2.0], vec![1.0], vec![-1.0]]
        );
        assert_eq!(variation(&p.graph, &p.x), 6.0);
    }

    #[test]
    fn b_bt_is_l_and_the_three_variations_agree_on_every_preset() {
        let mut p = Playground::default();
        for preset in Preset::ALL {
            p.load(preset);
            p.set_signal("random");
            let b = incidence(&p.graph);
            let l = laplacian(&p.graph);
            assert!(same(&times_own_transpose(&b), &l), "{}", preset.name());
            let lx = times(&l, &p.x);
            let xlx: f64 = (0..p.n()).map(|i| p.x[i] * lx[(i, 0)]).sum();
            let bx = transpose_times(&b, &p.x);
            let norm: f64 = (0..bx.nrows()).map(|e| bx[(e, 0)].powi(2)).sum();
            let v = variation(&p.graph, &p.x);
            assert!(
                (v - xlx).abs() < 1e-9 && (v - norm).abs() < 1e-9,
                "{}",
                preset.name()
            );
            for i in 0..p.n() {
                let row: f64 = (0..p.n()).map(|j| l[(i, j)]).sum();
                assert!(
                    row.abs() < 1e-12,
                    "{}: row {i} of L sums to {row}",
                    preset.name()
                );
            }
        }
    }

    #[test]
    fn the_normalised_laplacian_has_ones_on_its_diagonal() {
        let mut p = Playground::default();
        p.load(Preset::SquareDiagonal);
        let ln = normalized_laplacian(&p.graph);
        for i in 0..4 {
            assert!((ln[(i, i)] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn editing_the_graph() {
        let mut p = Playground::default();
        p.load(Preset::Triangle);
        p.toggle_link(0, 1);
        assert_eq!(p.graph.edges.len(), 2);
        p.toggle_link(1, 0);
        assert_eq!(p.graph.edges.len(), 3);
        p.remove_drone(0); // links 0–1 and 0–2 go, 1–2 becomes 0–1
        assert_eq!((p.n(), p.graph.n, p.x.len()), (2, 2, 2));
        assert_eq!(p.graph.edges.len(), 1);
        assert_eq!((p.graph.edges[0].src, p.graph.edges[0].dst), (0, 1));
        p.add_drone([0.0, 0.0]);
        assert_eq!((p.n(), p.graph.n, p.x[2]), (3, 3, 0.0));
    }

    #[test]
    fn everything_draws_without_panicking() {
        let mut p = Playground::default();
        p.shown = Shown {
            a: true,
            d: true,
            b: true,
            l: true,
            bbt: true,
            p: true,
            l_norm: true,
            signal: true,
        };
        for preset in Preset::ALL {
            p.load(preset);
            for previous in [
                None,
                Some(Target::Drone(0)),
                Some(Target::Pair(0, 1)),
                Some(Target::Link(0)),
                Some(Target::DroneLink(1, 0)),
            ] {
                let mut out = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                    p.controls(ui);
                    p.matrices(ui, previous);
                    p.canvas(ui, previous);
                });
                out.textures_delta.clear();
            }
        }
    }
}
