//! Matrix playground: place drones and see the matrices of their comm graph (Mission 1).
//!
//! Run with `cargo run --release -p aerie-viz --bin matrix_playground`.
//!
//! The graph is the swarm rule (`epsilon_graph`, D00): two drones are linked when they
//! are within the smaller of their radio ranges, with a Gaussian weight. The right panel
//! shows the matrices of that graph, from the real functions in `aerie-graph`:
//! - A, `EdgeList::adjacency_matrix`, optionally binary (every weight 1);
//! - D, `EdgeList::degree_matrix`;
//! - A^k = A × … × A (optional, k chosen): with a binary A, the number of k-hop walks;
//! - P = D⁻¹A, the transition matrix (shown once `EdgeList::transition_matrix` exists).
//!
//! Canvas: click empty space to add a drone, drag a drone to move it, right-click a drone
//! (or select it and press Delete) to remove it. Scroll to zoom, drag empty space to pan.
//! Hover a matrix cell to see its link on the map; hover or select a drone to see its row.

use aerie_graph::EdgeList;
use aerie_viz::grid::{Focus, HOVER_COLOR, fmt, matrix_grid, show_tip};
use aerie_viz::style::{draw_drone, role_color, role_name, swatch};
use eframe::egui::{
    self, Align2, Color32, FontId, Key, Pos2, Rect, RichText, Sense, Stroke, pos2, vec2,
};
use faer::Mat;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, DroneId, Role, SquadId, Vec2};
use swarmsim::graphs::epsilon::epsilon_graph;

/// Size of the playground, km.
const WORLD_W: f64 = 16.0;
const WORLD_H: f64 = 10.0;
/// How close the pointer must be to a drone to grab it, in pixels.
const GRAB_PX: f32 = 12.0;
/// Drone marker radius, pixels.
const NODE_PX: f32 = 8.0;

/// Accent colours of the matrices.
const A_COLOR: Color32 = Color32::from_rgb(78, 121, 167);
const D_COLOR: Color32 = Color32::from_rgb(242, 142, 43);
const A2_COLOR: Color32 = Color32::from_rgb(176, 122, 161);
const P_COLOR: Color32 = Color32::from_rgb(89, 161, 79);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE matrix playground")
            .with_inner_size([1600.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE matrix playground",
        options,
        Box::new(|_cc| Ok(Box::new(Playground::default()))),
    )
}

/// P = D⁻¹A, from `EdgeList::transition_matrix` (Mission 1).
fn transition(graph: &EdgeList) -> Option<Mat<f64>> {
    Some(graph.transition_matrix())
}

struct Playground {
    /// The drones on the canvas. Their index is their row and column in every matrix.
    nodes: Vec<Drone>,
    /// Only the two radio ranges are used (D00 link rule).
    config: SimConfig,
    /// Every link gets weight 1, so A is 0/1.
    binary: bool,
    /// Show A^k.
    show_power: bool,
    /// The power k of A^k.
    power_k: usize,

    // Editing.
    new_role: Role,
    snap: bool,
    selected: Option<usize>,
    dragging: Option<usize>,

    // Display.
    show_weights: bool,
    show_range: bool,

    // Hover links between the map and the matrices (set each frame).
    hover_node: Option<usize>,
    hover_cell: Option<(usize, usize)>,

    // View: zoom factor (1 = whole world fits) and pan offset in pixels.
    zoom: f32,
    pan: egui::Vec2,
    panning: bool,
}

impl Default for Playground {
    fn default() -> Self {
        let mut p = Playground {
            nodes: Vec::new(),
            config: SimConfig::default(),
            binary: false,
            show_power: false,
            power_k: 2,
            new_role: Role::Recon,
            snap: true,
            selected: None,
            dragging: None,
            show_weights: true,
            show_range: true,
            hover_node: None,
            hover_cell: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            panning: false,
        };
        p.load_triangle();
        p
    }
}

impl eframe::App for Playground {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        // Built once per frame from the current drones and settings.
        let graph = self.graph();

        egui::Panel::right("matrices")
            .resizable(true)
            .default_size(620.0)
            .show(ui, |ui| {
                egui::ScrollArea::both().show(ui, |ui| self.matrices(ui, &graph));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui, &graph));
    }
}

impl Playground {
    // ---------------------------------------------------------------- model

    /// The swarm graph (`epsilon_graph`), with every weight set to 1 when `binary` is on.
    fn graph(&self) -> EdgeList {
        let mut g = epsilon_graph(&self.nodes, &self.config);
        if self.binary {
            for e in &mut g.edges {
                e.weight = 1.0;
            }
        }
        g
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

    fn add_node(&mut self, role: Role, position: Vec2) {
        let id = DroneId(self.nodes.len() as u32);
        self.nodes.push(Drone::new(id, SquadId(0), role, position));
    }

    /// Removes drone `i` and renumbers the rest so ids match indices (matrix rows) again.
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
        self.hover_cell = None;
        self.hover_node = None;
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

    fn set(&mut self, drones: &[(Role, f64, f64)]) {
        self.nodes.clear();
        for &(role, x, y) in drones {
            self.add_node(role, Vec2 { x, y });
        }
        self.selected = None;
        self.hover_cell = None;
        self.hover_node = None;
    }

    /// The test graph of `aerie-graph`: a triangle 0-1-2 (recon, 2 km sides) and node 3
    /// alone. With "A is 0/1" ticked, A, D and P are exactly the grids of the tests.
    fn load_triangle(&mut self) {
        self.config = SimConfig::default();
        let h = 3.0_f64.sqrt(); // height of a triangle with 2 km sides
        self.set(&[
            (Role::Recon, 3.0, 3.0),
            (Role::Recon, 5.0, 3.0),
            (Role::Recon, 4.0, 3.0 + h),
            (Role::Recon, 12.0, 7.0),
        ]);
    }

    /// A path 0 - 1 - 2 - 3, recon drones 2 km apart.
    fn load_path(&mut self) {
        self.config = SimConfig::default();
        self.set(&[
            (Role::Recon, 3.0, 5.0),
            (Role::Recon, 5.0, 5.0),
            (Role::Recon, 7.0, 5.0),
            (Role::Recon, 9.0, 5.0),
        ]);
    }

    /// A relay in the middle, five recon drones around it: a star (P shows the hub).
    fn load_star(&mut self) {
        self.config = SimConfig::default();
        let mut d = vec![(Role::Relay, 8.0, 5.0)];
        for k in 0..5 {
            let a = k as f64 / 5.0 * std::f64::consts::TAU;
            d.push((Role::Recon, 8.0 + 2.0 * a.cos(), 5.0 + 2.0 * a.sin()));
        }
        self.set(&d);
    }

    /// Two squads of three recon drones, each with a relay; the relays link the squads.
    fn load_two_squads(&mut self) {
        self.config = SimConfig::default();
        self.set(&[
            (Role::Recon, 2.0, 4.0),
            (Role::Recon, 3.5, 6.0),
            (Role::Recon, 2.0, 6.5),
            (Role::Relay, 3.5, 4.5),
            (Role::Relay, 12.5, 4.5),
            (Role::Recon, 14.0, 4.0),
            (Role::Recon, 12.5, 6.0),
            (Role::Recon, 14.0, 6.5),
        ]);
    }

    // ---------------------------------------------------------------- panels

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrix playground");
        ui.weak("Mission 1: the matrices of the comm graph.");
        ui.add_space(6.0);

        ui.label(RichText::new("New drones").strong());
        for role in Role::ALL {
            ui.horizontal(|ui| {
                swatch(ui, role, role_color(role));
                ui.radio_value(&mut self.new_role, role, role_name(role));
            });
        }
        ui.checkbox(&mut self.snap, "Snap to 0.5 km");
        ui.weak("Click: add · drag: move · right-click or Delete: remove.");

        ui.separator();
        ui.label(RichText::new("Matrices").strong());
        ui.checkbox(&mut self.binary, "A is 0/1 (ignore the weights)")
            .on_hover_text(
                "Every link gets weight 1. A then only says who is linked, D counts \
                 neighbours, and P spreads evenly over them.",
            );
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_power, "Show A^k, k =")
                .on_hover_text(
                    "With a 0/1 A, (A^k)[i][j] counts the walks of k hops from i to j. \
                     A walk may go back and forth, so k = 2 puts each drone's number \
                     of neighbours on the diagonal.",
                );
            ui.add_enabled(
                self.show_power,
                egui::DragValue::new(&mut self.power_k).range(1..=8),
            );
        });

        ui.separator();
        ui.label(RichText::new("Links (D00 rule)").strong());
        ui.add(
            egui::Slider::new(&mut self.config.range_short, 0.5..=10.0)
                .suffix(" km")
                .text("short range"),
        )
        .on_hover_text("Recon, RF sensing and compute drones.");
        ui.add(
            egui::Slider::new(&mut self.config.range_long, 0.5..=20.0)
                .suffix(" km")
                .text("long range"),
        )
        .on_hover_text("Relay drones.");
        ui.weak("Linked when the distance is at most the smaller of the two ranges.");
        ui.weak("Weight exp(−d² / 2σ²), σ = half that range.");

        ui.separator();
        ui.label(RichText::new("Display").strong());
        ui.checkbox(&mut self.show_weights, "Weights on links");
        ui.checkbox(&mut self.show_range, "Range circle of the selected drone");

        ui.separator();
        ui.label(RichText::new("Presets").strong());
        if ui
            .button("Triangle + isolated")
            .on_hover_text("The aerie-graph test graph. Tick \"A is 0/1\" to get the test grids.")
            .clicked()
        {
            self.load_triangle();
        }
        if ui.button("Path of 4").clicked() {
            self.load_path();
        }
        if ui.button("Star (relay hub)").clicked() {
            self.load_star();
        }
        if ui.button("Two squads + relay link").clicked() {
            self.load_two_squads();
        }
        if ui.button("Clear").clicked() {
            self.set(&[]);
        }
    }

    /// Right panel: A, D, A² and P side by side with their row sums.
    fn matrices(&mut self, ui: &mut egui::Ui, graph: &EdgeList) {
        let n = self.nodes.len();
        let prev_hover = self.hover_cell;
        let focus = self.hover_node.or(self.selected);
        let mut hovered: Option<(usize, usize)> = None;

        ui.heading("Matrices");
        ui.weak(format!(
            "{n} drones, {} links. Row and column i = drone i. Σ = row sum.",
            graph.edges.len()
        ));
        if n == 0 {
            ui.weak("Add drones on the map.");
            self.hover_cell = None;
            return;
        }
        if n > 20 {
            ui.weak("Many drones: scroll the panel to see the whole matrices.");
        }
        // Row and column headers: the drone number in its role colour.
        let heads: Vec<(String, Color32)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(k, d)| (k.to_string(), role_color(d.role)))
            .collect();
        let at = |cell| Focus {
            row: focus,
            col: focus,
            cell,
        };
        let a = graph.adjacency_matrix();
        let d = graph.degree_matrix();
        let dist = |i: usize, j: usize| self.nodes[i].position.dist(self.nodes[j].position);

        // A.
        ui.add_space(6.0);
        ui.label(RichText::new("A · adjacency").strong().color(A_COLOR));
        ui.weak(if self.binary {
            "A[i][j] = 1 when i and j are linked, 0 otherwise."
        } else {
            "A[i][j] = weight of the link i–j, 0 when there is none."
        });
        let (h, resp) = matrix_grid(
            ui,
            &a,
            &heads,
            &heads,
            at(prev_hover),
            A_COLOR,
            Some("Σ = deg"),
        );
        if let Some((i, j)) = h {
            hovered = Some((i, j));
            let text = if i == j {
                format!("A[{i}][{i}] = 0: no self-loops.")
            } else if a[(i, j)] == 0.0 {
                format!("A[{i}][{j}] = 0: no link ({:.2} km apart).", dist(i, j))
            } else {
                format!(
                    "A[{i}][{j}] = {}: link {i}–{j}, {:.2} km. Same as A[{j}][{i}] (symmetric).",
                    fmt(a[(i, j)]),
                    dist(i, j)
                )
            };
            show_tip(resp, &text);
        }

        // D.
        ui.add_space(8.0);
        ui.label(RichText::new("D · degree").strong().color(D_COLOR));
        ui.weak("Diagonal: D[i][i] = sum of row i of A. Zero everywhere else.");
        let (h, resp) = matrix_grid(ui, &d, &heads, &heads, at(prev_hover), D_COLOR, Some("Σ"));
        if let Some((i, j)) = h {
            hovered = Some((i, j));
            let text = if i == j {
                let row: Vec<String> = (0..n)
                    .filter(|&k| a[(i, k)] != 0.0)
                    .map(|k| fmt(a[(i, k)]))
                    .collect();
                if row.is_empty() {
                    format!("D[{i}][{i}] = 0: drone {i} has no links (isolated).")
                } else {
                    format!(
                        "D[{i}][{i}] = row {i} of A summed = {} = {}",
                        row.join(" + "),
                        fmt(d[(i, i)])
                    )
                }
            } else {
                format!("D[{i}][{j}] = 0: a degree belongs to one drone, so D is diagonal.")
            };
            show_tip(resp, &text);
        }

        // A^k.
        if self.show_power {
            let k = self.power_k;
            let ak = mat_power(&a, k);
            ui.add_space(8.0);
            ui.label(
                RichText::new(format!("A^{k} = A × … × A ({k} times)"))
                    .strong()
                    .color(A2_COLOR),
            );
            ui.weak(if self.binary {
                format!(
                    "(A^{k})[i][j] = number of {k}-hop walks from i to j. Non-zero = j is \
                     reachable from i in exactly {k} hops (walks may go back and forth)."
                )
            } else {
                format!(
                    "(A^{k})[i][j] = sum over {k}-hop walks i → … → j of the product of \
                     their link weights."
                )
            });
            let (h, resp) =
                matrix_grid(ui, &ak, &heads, &heads, at(prev_hover), A2_COLOR, Some("Σ"));
            if let Some((i, j)) = h {
                hovered = Some((i, j));
                let (walks, more) = walks_between(&a, i, j, k, 6);
                let text = if walks.is_empty() {
                    format!("(A^{k})[{i}][{j}] = 0: no {k}-hop walk from {i} to {j}.")
                } else {
                    let list: Vec<String> = walks
                        .iter()
                        .map(|w| {
                            w.iter()
                                .map(|v| v.to_string())
                                .collect::<Vec<_>>()
                                .join("→")
                        })
                        .collect();
                    format!(
                        "(A^{k})[{i}][{j}] = {}: walks {}{}",
                        fmt(ak[(i, j)]),
                        list.join(", "),
                        if more { ", …" } else { "" }
                    )
                };
                show_tip(resp, &text);
            }
        }

        // P.
        ui.add_space(8.0);
        ui.label(
            RichText::new("P = D⁻¹A · transition")
                .strong()
                .color(P_COLOR),
        );
        match transition(graph) {
            Some(p) => {
                ui.weak("P[i][j] = A[i][j] / D[i][i]: chance a random walk at i steps to j.");
                let (h, resp) = matrix_grid(
                    ui,
                    &p,
                    &heads,
                    &heads,
                    at(prev_hover),
                    P_COLOR,
                    Some("Σ = 1?"),
                );
                if let Some((i, j)) = h {
                    hovered = Some((i, j));
                    let text = if d[(i, i)] == 0.0 {
                        format!("P[{i}][{j}] = {}: drone {i} is isolated.", fmt(p[(i, j)]))
                    } else {
                        format!(
                            "P[{i}][{j}] = A[{i}][{j}] / D[{i}][{i}] = {} / {} = {}",
                            fmt(a[(i, j)]),
                            fmt(d[(i, i)]),
                            fmt(p[(i, j)])
                        )
                    };
                    show_tip(resp, &text);
                }
                let bad: Vec<usize> = (0..n)
                    .filter(|&i| {
                        let s: f64 = (0..n).map(|j| p[(i, j)]).sum();
                        (s - 1.0).abs() > 1e-9
                    })
                    .collect();
                if bad.is_empty() {
                    ui.colored_label(P_COLOR, "Every row sums to 1.");
                } else {
                    ui.colored_label(
                        Color32::from_rgb(225, 87, 89),
                        format!("Rows that do not sum to 1: {bad:?}"),
                    );
                }
            }
            None => {
                ui.weak(
                    "Shows here once EdgeList::transition_matrix exists in aerie-graph. \
                     Each row of A divided by that drone's degree; an isolated drone \
                     needs your rule (the plan: decide before it is a division by zero).",
                );
            }
        }

        // Checks that tie the matrices together.
        ui.add_space(8.0);
        ui.separator();
        let symmetric = (0..n).all(|i| (0..n).all(|j| a[(i, j)] == a[(j, i)]));
        let deg = graph.degrees();
        let diag_ok = (0..n).all(|i| (d[(i, i)] - deg[i]).abs() < 1e-9);
        ui.label(format!(
            "A symmetric: {} · D diagonal = degrees(): {}",
            yes(symmetric),
            yes(diag_ok)
        ));

        self.hover_cell = hovered;
    }

    // ---------------------------------------------------------------- canvas

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
                    self.add_node(self.new_role, pos);
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
        if !ui.ctx().egui_wants_keyboard_input() {
            if let Some(i) = self.selected
                && ui.input(|inp| inp.key_pressed(Key::Delete))
            {
                self.remove_node(i);
            }
            if ui.input(|inp| inp.key_pressed(Key::Escape)) {
                self.selected = None;
            }
        }
        // Input may have added, moved or removed drones: rebuild the graph if so.
        let rebuilt;
        let graph = if graph.n as usize == self.nodes.len() && self.dragging.is_none() {
            graph
        } else {
            rebuilt = self.graph();
            &rebuilt
        };

        // --- World and grid (1 km lines, labels every 2 km).
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
                Stroke::new(1.0, weak.gamma_multiply(0.15)),
            );
            if k % 2 == 0 {
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
            painter.hline(
                world.x_range(),
                y,
                Stroke::new(1.0, weak.gamma_multiply(0.15)),
            );
            if k % 2 == 0 {
                painter.text(
                    pos2(world.left() - 4.0, y),
                    Align2::RIGHT_CENTER,
                    format!("{k}"),
                    FontId::proportional(11.0),
                    weak,
                );
            }
        }

        // --- Range circle of the selected drone.
        if self.show_range
            && let Some(d) = self.selected.and_then(|i| self.nodes.get(i))
        {
            let r = t.len(d.role.radio_range(&self.config));
            painter.circle_stroke(
                t.to_screen(d.position),
                r,
                Stroke::new(1.0, role_color(d.role).gamma_multiply(0.6)),
            );
        }

        // --- Links. The hovered matrix cell (i, j) lights up link i–j.
        let cell = self.hover_cell;
        let max_w = graph
            .edges
            .iter()
            .map(|e| e.weight)
            .fold(0.0_f64, f64::max)
            .max(1e-9);
        for e in &graph.edges {
            let (s, d) = (e.src as usize, e.dst as usize);
            let (ps, pd) = (
                t.to_screen(self.nodes[s].position),
                t.to_screen(self.nodes[d].position),
            );
            let lit = cell.is_some_and(|(i, j)| (i, j) == (s, d) || (i, j) == (d, s));
            let touches = self
                .hover_node
                .or(self.selected)
                .is_some_and(|k| k == s || k == d);
            let k = (e.weight / max_w) as f32;
            let (width, color) = if lit {
                (4.0, HOVER_COLOR)
            } else if touches {
                (1.0 + 3.0 * k, text.gamma_multiply(0.9))
            } else {
                (1.0 + 3.0 * k, text.gamma_multiply(0.45))
            };
            painter.line_segment([ps, pd], Stroke::new(width, color));
            if self.show_weights {
                painter.text(
                    ps + (pd - ps) * 0.5 + vec2(0.0, -8.0),
                    Align2::CENTER_BOTTOM,
                    fmt(e.weight),
                    FontId::monospace(11.0),
                    if lit { HOVER_COLOR } else { weak },
                );
            }
        }

        // --- Drones, with their index (= matrix row) next to them.
        for (i, d) in self.nodes.iter().enumerate() {
            let c = t.to_screen(d.position);
            let in_cell = cell.is_some_and(|(a, b)| a == i || b == i);
            draw_drone(
                &painter,
                c,
                NODE_PX,
                d.role,
                role_color(d.role),
                Stroke::new(1.0, Color32::from_black_alpha(180)),
            );
            if in_cell {
                painter.circle_stroke(c, NODE_PX + 5.0, Stroke::new(2.5, HOVER_COLOR));
            }
            if self.selected == Some(i) {
                painter.circle_stroke(c, NODE_PX + 8.0, Stroke::new(2.0, text));
            }
            painter.text(
                c + vec2(NODE_PX + 3.0, -NODE_PX - 1.0),
                Align2::LEFT_BOTTOM,
                format!("{i}"),
                FontId::proportional(14.0),
                if in_cell { HOVER_COLOR } else { text },
            );
        }

        if let Some(i) = self.hover_node {
            let d = &self.nodes[i];
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!("Drone {i} · {}", role_name(d.role)));
                ui.label(format!("({:.1}, {:.1}) km", d.position.x, d.position.y));
                ui.weak("Its row and column are outlined in the matrices.");
            });
        }
    }
}

/// A^k by repeated multiplication (A^1 = A, A^0 = identity).
fn mat_power(a: &Mat<f64>, k: usize) -> Mat<f64> {
    let n = a.nrows();
    let mut result: Mat<f64> = Mat::identity(n, n);
    for _ in 0..k {
        result = &result * a;
    }
    result
}

/// Up to `limit` walks of exactly `k` hops from `i` to `j` on the links of `a`, and
/// whether there are more. Each walk lists its drones, `i` first, `j` last.
fn walks_between(
    a: &Mat<f64>,
    i: usize,
    j: usize,
    k: usize,
    limit: usize,
) -> (Vec<Vec<usize>>, bool) {
    struct Search<'m> {
        a: &'m Mat<f64>,
        j: usize,
        limit: usize,
        out: Vec<Vec<usize>>,
        more: bool,
    }
    fn go(s: &mut Search, walk: &mut Vec<usize>, left: usize) {
        let at = *walk.last().unwrap();
        if left == 0 {
            if at == s.j {
                if s.out.len() < s.limit {
                    s.out.push(walk.clone());
                } else {
                    s.more = true;
                }
            }
            return;
        }
        for next in 0..s.a.nrows() {
            if s.more {
                return;
            }
            if s.a[(at, next)] != 0.0 {
                walk.push(next);
                go(s, walk, left - 1);
                walk.pop();
            }
        }
    }
    let mut search = Search {
        a,
        j,
        limit,
        out: Vec::new(),
        more: false,
    };
    go(&mut search, &mut vec![i], k);
    (search.out, search.more)
}

fn yes(ok: bool) -> &'static str {
    if ok { "yes" } else { "NO" }
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

    fn grid(m: &Mat<f64>) -> Vec<Vec<f64>> {
        (0..m.nrows())
            .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
            .collect()
    }

    #[test]
    fn triangle_preset_with_binary_a_is_the_test_graph() {
        // The same A and D as the aerie-graph tests on triangle_plus_isolated().
        let p = Playground {
            binary: true,
            ..Default::default()
        };
        let g = p.graph();
        assert_eq!(
            grid(&g.adjacency_matrix()),
            vec![
                vec![0.0, 1.0, 1.0, 0.0],
                vec![1.0, 0.0, 1.0, 0.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            ]
        );
        assert_eq!(g.degrees(), vec![2.0, 2.0, 2.0, 0.0]);
    }

    #[test]
    fn weighted_a_uses_the_gaussian_weights_of_epsilon_graph() {
        // 2 km sides, recon range 2.5 km, sigma 1.25: weight exp(-4 / 3.125).
        let p = Playground::default();
        let a = p.graph().adjacency_matrix();
        let w = (-4.0_f64 / (2.0 * 1.25 * 1.25)).exp();
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            assert!((a[(i, j)] - w).abs() < 1e-9, "({i}, {j}) = {}", a[(i, j)]);
        }
        assert_eq!(a[(0, 3)], 0.0);
    }

    #[test]
    fn binary_keeps_the_same_links() {
        let mut p = Playground::default();
        p.load_two_squads();
        let weighted = p.graph();
        p.binary = true;
        let binary = p.graph();
        assert_eq!(weighted.edges.len(), binary.edges.len());
        for (w, b) in weighted.edges.iter().zip(&binary.edges) {
            assert_eq!((w.src, w.dst), (b.src, b.dst));
            assert_eq!(b.weight, 1.0);
        }
        // The two relays (3 and 4, 9 km apart) are linked by the long range.
        assert!(binary.edges.iter().any(|e| (e.src, e.dst) == (3, 4)));
    }

    #[test]
    fn a_squared_counts_two_hop_walks_on_the_path() {
        // Path 0-1-2-3, binary: (A²)[0][2] = 1 (via 1), (A²)[1][1] = 2 neighbours,
        // (A²)[0][3] = 0 (3 hops away).
        let mut p = Playground::default();
        p.load_path();
        p.binary = true;
        let a = p.graph().adjacency_matrix();
        let a2 = &a * &a;
        assert_eq!(a2[(0, 2)], 1.0);
        assert_eq!(a2[(1, 1)], 2.0);
        assert_eq!(a2[(0, 0)], 1.0);
        assert_eq!(a2[(0, 3)], 0.0);
    }

    #[test]
    fn removing_a_drone_renumbers_rows() {
        let mut p = Playground {
            selected: Some(3),
            ..Default::default()
        };
        p.remove_node(1);
        assert_eq!(p.nodes.len(), 3);
        assert!(
            p.nodes
                .iter()
                .enumerate()
                .all(|(k, d)| d.id.0 as usize == k)
        );
        assert_eq!(p.selected, Some(2));
        assert_eq!(p.graph().adjacency_matrix().nrows(), 3);
    }

    #[test]
    fn p_panel_uses_the_real_transition_matrix() {
        // Binary triangle: rows of 0.5, and the isolated drone 3 keeps a self-loop.
        let p = Playground {
            binary: true,
            ..Default::default()
        };
        let m = transition(&p.graph()).unwrap();
        assert_eq!(m[(0, 1)], 0.5);
        assert_eq!(m[(3, 3)], 1.0);
    }

    #[test]
    fn a_to_the_k_counts_k_hop_walks() {
        // Path 0-1-2-3, binary. A^3[0][3] = 1 (0→1→2→3); A^1 = A; A^0 = I.
        let mut p = Playground {
            binary: true,
            ..Default::default()
        };
        p.load_path();
        let a = p.graph().adjacency_matrix();
        assert_eq!(mat_power(&a, 1)[(0, 1)], 1.0);
        assert_eq!(mat_power(&a, 0)[(2, 2)], 1.0);
        assert_eq!(mat_power(&a, 3)[(0, 3)], 1.0);
        // The walk lister finds the same number as the matrix, cell by cell.
        for k in 1..=4 {
            let ak = mat_power(&a, k);
            for i in 0..4 {
                for j in 0..4 {
                    let (walks, more) = walks_between(&a, i, j, k, 100);
                    assert!(!more);
                    assert_eq!(walks.len() as f64, ak[(i, j)], "k {k} ({i}, {j})");
                }
            }
        }
        assert_eq!(walks_between(&a, 0, 3, 3, 10).0, vec![vec![0, 1, 2, 3]]);
    }
}
