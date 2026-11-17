//! The arena map: cells, squads, radio ranges and drones, with zoom and pan. With a
//! snapshot (step 2 and 3a): the jammer, the ridge, the jammed zone, the live links
//! after jamming, and the tasks.

use aerie_graph::EdgeList;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Vec2,
    pos2, vec2,
};
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::jamming::{Jammer, Ridge, jam_power};
use swarmsim::links::LiveGraph;
use swarmsim::tasks::{Task, TaskKind};

use crate::squads::SquadInfo;
use crate::style::{draw_drone, ramp, role_color, role_name, squad_color};

/// What the drone fill colour encodes. The shape always encodes the role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorBy {
    Squad,
    Role,
    Battery,
    SensorQuality,
    ComputeCapacity,
}

impl ColorBy {
    pub const ALL: [ColorBy; 5] = [
        ColorBy::Squad,
        ColorBy::Role,
        ColorBy::Battery,
        ColorBy::SensorQuality,
        ColorBy::ComputeCapacity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ColorBy::Squad => "Squad",
            ColorBy::Role => "Role",
            ColorBy::Battery => "Battery",
            ColorBy::SensorQuality => "Sensor quality",
            ColorBy::ComputeCapacity => "Compute capacity",
        }
    }
}

/// Which radio-range circles to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeCircles {
    None,
    Selected,
    Relays,
    All,
}

impl RangeCircles {
    pub const ALL: [RangeCircles; 4] = [
        RangeCircles::None,
        RangeCircles::Selected,
        RangeCircles::Relays,
        RangeCircles::All,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RangeCircles::None => "None",
            RangeCircles::Selected => "Selected drone",
            RangeCircles::Relays => "Relays",
            RangeCircles::All => "All drones",
        }
    }
}

/// What the colour of a link encodes. Width always encodes the weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkColor {
    /// Red (weak) to green (strong).
    Weight,
    /// Links inside a squad in the squad colour, links between squads highlighted.
    Squad,
}

impl LinkColor {
    pub const ALL: [LinkColor; 2] = [LinkColor::Weight, LinkColor::Squad];

    pub fn label(self) -> &'static str {
        match self {
            LinkColor::Weight => "Weight",
            LinkColor::Squad => "Inside / between squads",
        }
    }
}

/// Which links to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkSource {
    /// Every pair within range (`epsilon_graph`), as if there were no jammer.
    Geometric,
    /// The links that survive jamming (`live_graph`, D03).
    Live,
}

impl LinkSource {
    pub const ALL: [LinkSource; 2] = [LinkSource::Geometric, LinkSource::Live];

    pub fn label(self) -> &'static str {
        match self {
            LinkSource::Geometric => "Geometric (no jamming)",
            LinkSource::Live => "Live (after jamming)",
        }
    }
}

/// What the dataset adds on top of the swarm for one seed, built in the dataset order
/// with one RNG: spawn, place_jamming, truth, readings, live graph, tasks.
pub struct Snapshot {
    pub jammer: Jammer,
    pub ridge: Ridge,
    /// True jamming power at each drone.
    pub truth: Vec<f64>,
    /// What each drone's sensor reads.
    pub readings: Vec<f64>,
    /// Zone label: true power at or above the threshold.
    pub labels: Vec<bool>,
    pub live: LiveGraph,
    pub tasks: Vec<Task>,
    /// Jamming power on a grid of FIELD_RES km cells (row 0 at y = 0), for the shading.
    pub field: Vec<f64>,
}

/// Size of a cell of the jamming-field grid, km.
pub const FIELD_RES: f64 = 0.25;

/// Grid size (columns, rows) of the jamming field for this arena.
pub fn field_dims(cfg: &SimConfig) -> (usize, usize) {
    (
        (cfg.arena_w / FIELD_RES).ceil() as usize,
        (cfg.arena_h / FIELD_RES).ceil() as usize,
    )
}

/// Jamming power at the centre of every FIELD_RES cell, from `jam_power` (so the ridge
/// shadow is included). Row-major, row 0 at y = 0. Computed once per seed.
pub fn jam_field(jammer: &Jammer, ridge: &Ridge, cfg: &SimConfig) -> Vec<f64> {
    let (w, h) = field_dims(cfg);
    let mut field = Vec::with_capacity(w * h);
    for r in 0..h {
        for c in 0..w {
            let at = swarmsim::drone::Vec2 {
                x: (c as f64 + 0.5) * FIELD_RES,
                y: (r as f64 + 0.5) * FIELD_RES,
            };
            field.push(jam_power(jammer, ridge, at, cfg));
        }
    }
    field
}

/// Colour of a task kind on the map and in the legend.
pub fn task_color(kind: TaskKind) -> Color32 {
    match kind {
        TaskKind::Observe => Color32::from_rgb(70, 190, 255),
        TaskKind::Track => Color32::from_rgb(255, 150, 50),
        TaskKind::Relay => Color32::from_rgb(190, 120, 255),
        TaskKind::Fuse => Color32::from_rgb(110, 210, 110),
    }
}

/// One-letter label of a task kind.
pub fn task_letter(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Observe => "O",
        TaskKind::Track => "T",
        TaskKind::Relay => "R",
        TaskKind::Fuse => "F",
    }
}

pub fn task_name(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Observe => "Observe",
        TaskKind::Track => "Track",
        TaskKind::Relay => "Relay",
        TaskKind::Fuse => "Fuse",
    }
}

/// How many of the tasks were placed near a drone (the first ones, see `spawn_tasks`).
pub fn n_near_tasks(cfg: &SimConfig) -> usize {
    (cfg.n_tasks as f64 * cfg.task_near_frac).round() as usize
}

const JAMMER_COLOR: Color32 = Color32::from_rgb(230, 60, 60);
const RIDGE_COLOR: Color32 = Color32::from_rgb(170, 120, 70);
const ZONE_COLOR: Color32 = Color32::from_rgb(255, 70, 70);
/// Drones in the jammed zone: lime, used by no squad, role or task kind.
pub const JAMMED_DRONE_COLOR: Color32 = Color32::from_rgb(200, 255, 0);

/// Map layers the user can switch on and off.
pub struct Layers {
    pub color_by: ColorBy,
    pub grid: bool,
    pub occupied_cells: bool,
    pub centres: bool,
    pub footprints: bool,
    pub ids: bool,
    pub ranges: RangeCircles,
    pub dim_other_squads: bool,
    /// Draw the comm graph links.
    pub links: bool,
    pub link_color: LinkColor,
    /// Geometric links, or the live ones after jamming (needs a snapshot).
    pub link_source: LinkSource,
    /// With live links: also draw the geometric links that jamming killed.
    pub dead_links: bool,
    /// The jammer, the ridge and the jammed drones (lime).
    pub jamming: bool,
    /// Fill the jammed zone with a dark shade, darker where jamming is stronger.
    pub zone_shade: bool,
    /// The tasks of the snapshot.
    pub tasks: bool,
}

impl Default for Layers {
    fn default() -> Self {
        Layers {
            color_by: ColorBy::Squad,
            grid: true,
            occupied_cells: true,
            centres: true,
            footprints: true,
            ids: false,
            ranges: RangeCircles::Selected,
            dim_other_squads: true,
            links: true,
            link_color: LinkColor::Squad,
            link_source: LinkSource::Live,
            dead_links: true,
            jamming: true,
            zone_shade: true,
            tasks: true,
        }
    }
}

/// Zoom and pan state of the map.
pub struct MapView {
    pub zoom: f32,
    pub pan: Vec2,
}

impl Default for MapView {
    fn default() -> Self {
        MapView {
            zoom: 1.0,
            pan: Vec2::ZERO,
        }
    }
}

/// Converts arena km to screen pixels (y up in km, y down on screen).
struct Transform {
    left: f32,
    bottom: f32,
    scale: f32,
}

impl Transform {
    fn new(rect: Rect, view: &MapView, cfg: &SimConfig) -> Self {
        let margin = 36.0;
        let w = cfg.arena_w as f32;
        let h = cfg.arena_h as f32;
        let fit = ((rect.width() - 2.0 * margin) / w).min((rect.height() - 2.0 * margin) / h);
        let scale = fit.max(1.0) * view.zoom;
        Transform {
            left: rect.center().x - w * scale / 2.0 + view.pan.x,
            bottom: rect.center().y + h * scale / 2.0 + view.pan.y,
            scale,
        }
    }

    fn to_screen(&self, x: f64, y: f64) -> Pos2 {
        pos2(
            self.left + x as f32 * self.scale,
            self.bottom - y as f32 * self.scale,
        )
    }

    fn to_km(&self, p: Pos2) -> (f64, f64) {
        (
            ((p.x - self.left) / self.scale) as f64,
            ((self.bottom - p.y) / self.scale) as f64,
        )
    }

    fn len(&self, km: f64) -> f32 {
        km as f32 * self.scale
    }
}

/// Draws the map and handles zoom, pan, hover and click-to-select (a drone or a task).
// Each input is a separate piece of viewer state; bundling them would only move the list.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    view: &mut MapView,
    layers: &Layers,
    cfg: &SimConfig,
    swarm: &[Drone],
    graph: &EdgeList,
    squads: &[SquadInfo],
    snap: Option<&Snapshot>,
    selected: &mut Option<usize>,
    selected_task: &mut Option<usize>,
) {
    let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
    let rect = response.rect;

    // Zoom with the scroll wheel, keeping the point under the cursor fixed.
    if let Some(pointer) = response.hover_pos() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            let before = Transform::new(rect, view, cfg).to_km(pointer);
            view.zoom = (view.zoom * (scroll * 0.002).exp()).clamp(0.5, 25.0);
            let moved = Transform::new(rect, view, cfg).to_screen(before.0, before.1);
            view.pan += pointer - moved;
        }
    }
    if response.dragged() {
        view.pan += response.drag_delta();
    }

    let t = Transform::new(rect, view, cfg);
    let visuals = ui.visuals().clone();
    let text = visuals.text_color();
    let faint = text.gamma_multiply(0.15);

    // Arena background and border.
    let arena = Rect::from_two_pos(t.to_screen(0.0, 0.0), t.to_screen(cfg.arena_w, cfg.arena_h));
    painter.rect_filled(arena, 0.0, visuals.extreme_bg_color);

    let selected_squad = selected
        .and_then(|i| swarm.get(i))
        .map(|d| d.squad.0 as usize);

    // Cells that hold a squad, tinted in the squad colour.
    if layers.occupied_cells {
        for s in squads {
            let col = (s.centroid.0 / cfg.cell_width).floor();
            let row = (s.centroid.1 / cfg.cell_height).floor();
            let cell = Rect::from_two_pos(
                t.to_screen(col * cfg.cell_width, row * cfg.cell_height),
                t.to_screen((col + 1.0) * cfg.cell_width, (row + 1.0) * cfg.cell_height),
            );
            painter.rect_filled(cell, 0.0, squad_color(s.squad).gamma_multiply(0.10));
        }
    }

    // The jammed zone (true power at or above the threshold), shaded darker as the power
    // grows towards P0. One mesh of small squares, from the field computed per seed.
    if layers.jamming
        && layers.zone_shade
        && let Some(s) = snap
    {
        let (w, h) = field_dims(cfg);
        let thr = cfg.jam_threshold * cfg.jam_p0;
        let span = (cfg.jam_p0 - thr).max(1e-9);
        let mut mesh = egui::Mesh::default();
        for r in 0..h {
            for c in 0..w {
                let p = s.field[r * w + c];
                if p < thr {
                    continue;
                }
                let k = ((p - thr) / span).clamp(0.0, 1.0) as f32;
                let alpha = (90.0 + 90.0 * k) as u8;
                let (x0, y0) = (c as f64 * FIELD_RES, r as f64 * FIELD_RES);
                let cell = Rect::from_two_pos(
                    t.to_screen(x0, y0),
                    t.to_screen(x0 + FIELD_RES, y0 + FIELD_RES),
                );
                mesh.add_colored_rect(cell.expand(0.3), Color32::from_black_alpha(alpha));
            }
        }
        painter.add(Shape::mesh(mesh));
    }

    if layers.grid {
        draw_grid(&painter, &t, cfg, faint);
    }
    draw_axes(&painter, &t, cfg, text.gamma_multiply(0.7));
    painter.rect_stroke(
        arena,
        0.0,
        Stroke::new(1.0, text.gamma_multiply(0.5)),
        StrokeKind::Middle,
    );

    // Squad footprints: a circle around each squad reaching its furthest member.
    if layers.footprints {
        for s in squads {
            let c = t.to_screen(s.centroid.0, s.centroid.1);
            let color = squad_color(s.squad);
            painter.circle_filled(c, t.len(s.footprint), color.gamma_multiply(0.06));
            painter.circle_stroke(
                c,
                t.len(s.footprint),
                Stroke::new(1.0, color.gamma_multiply(0.5)),
            );
        }
    }

    // Radio-range circles.
    for (i, d) in swarm.iter().enumerate() {
        let show = match layers.ranges {
            RangeCircles::None => false,
            RangeCircles::Selected => *selected == Some(i),
            RangeCircles::Relays => d.role == Role::Relay,
            RangeCircles::All => true,
        };
        if show {
            let c = t.to_screen(d.position.x, d.position.y);
            let r = t.len(d.role.radio_range(cfg));
            let color = fill_color(d, layers.color_by);
            painter.circle_filled(c, r, color.gamma_multiply(0.05));
            painter.circle_stroke(c, r, Stroke::new(1.0, color.gamma_multiply(0.6)));
        }
    }

    // Ridge and jammer, with a dashed ring at d0 (half power).
    if layers.jamming
        && let Some(s) = snap
    {
        let (a, b) = (
            t.to_screen(s.ridge.a.x, s.ridge.a.y),
            t.to_screen(s.ridge.b.x, s.ridge.b.y),
        );
        painter.line_segment([a, b], Stroke::new(6.0, Color32::from_black_alpha(140)));
        painter.line_segment([a, b], Stroke::new(4.0, RIDGE_COLOR));
        let c = t.to_screen(s.jammer.position.x, s.jammer.position.y);
        painter.extend(Shape::dashed_line(
            &circle_points(c, t.len(cfg.jam_d0), 96),
            Stroke::new(1.0, JAMMER_COLOR.gamma_multiply(0.8)),
            5.0,
            4.0,
        ));
    }

    // Links of the comm graph. Width = weight. With a drone selected, its links stay
    // bright and the others fade. Skipped if the graph is out of date with the swarm.
    let live_links = layers.link_source == LinkSource::Live && snap.is_some();
    let link_graph: &EdgeList = match snap {
        Some(s) if live_links => &s.live.graph,
        _ => graph,
    };
    // Geometric links that jamming killed (live edges keep the geometric order).
    if layers.links && layers.dead_links && live_links && graph.n as usize == swarm.len() {
        let mut l = 0;
        for e in &graph.edges {
            let alive = link_graph
                .edges
                .get(l)
                .is_some_and(|x| (x.src, x.dst) == (e.src, e.dst));
            if alive {
                l += 1;
                continue;
            }
            let (a, b) = (&swarm[e.src as usize], &swarm[e.dst as usize]);
            painter.extend(Shape::dashed_line(
                &[
                    t.to_screen(a.position.x, a.position.y),
                    t.to_screen(b.position.x, b.position.y),
                ],
                Stroke::new(1.0, ZONE_COLOR.gamma_multiply(0.55)),
                4.0,
                3.0,
            ));
        }
    }
    if layers.links && link_graph.n as usize == swarm.len() {
        for e in &link_graph.edges {
            let (s, d) = (e.src as usize, e.dst as usize);
            let (a, b) = (&swarm[s], &swarm[d]);
            let mut color = match layers.link_color {
                LinkColor::Weight => ramp(e.weight),
                LinkColor::Squad if a.squad == b.squad => {
                    squad_color(a.squad.0 as usize).gamma_multiply(0.6)
                }
                LinkColor::Squad => text,
            };
            if selected.is_some_and(|i| i != s && i != d) {
                color = color.gamma_multiply(0.15);
            }
            let width = 0.5 + 2.5 * e.weight as f32;
            painter.line_segment(
                [
                    t.to_screen(a.position.x, a.position.y),
                    t.to_screen(b.position.x, b.position.y),
                ],
                Stroke::new(width, color),
            );
        }
    }

    // Drones. Size grows a little with zoom so they stay readable.
    let radius = (5.0 * view.zoom.sqrt()).clamp(4.0, 12.0);
    let outline = Stroke::new(1.0, Color32::from_black_alpha(160));
    // With a task selected, the drones whose role can take it get a ring in its colour.
    let task_kind = selected_task
        .and_then(|k| snap.and_then(|s| s.tasks.get(k)))
        .map(|task| task.kind);
    for (i, d) in swarm.iter().enumerate() {
        let c = t.to_screen(d.position.x, d.position.y);
        let mut fill = fill_color(d, layers.color_by);
        if layers.dim_other_squads
            && selected_squad.is_some_and(|sq| sq != d.squad.0 as usize)
            && *selected != Some(i)
        {
            fill = fill.gamma_multiply(0.25);
        }
        let jammed = layers.jamming && snap.is_some_and(|s| s.labels.get(i) == Some(&true));
        if jammed {
            fill = JAMMED_DRONE_COLOR;
        }
        if task_kind.is_some_and(|k| !k.can_do(d.role)) {
            fill = fill.gamma_multiply(0.2);
        }
        draw_drone(&painter, c, radius, d.role, fill, outline);
        if let Some(k) = task_kind
            && k.can_do(d.role)
        {
            painter.circle_stroke(c, radius + 4.0, Stroke::new(2.0, task_color(k)));
        }
        if jammed {
            painter.circle_stroke(c, radius + 2.5, Stroke::new(1.5, JAMMED_DRONE_COLOR));
        }
        if layers.ids {
            painter.text(
                c + vec2(radius + 2.0, -radius - 2.0),
                Align2::LEFT_BOTTOM,
                d.id.0,
                FontId::monospace(10.0),
                text.gamma_multiply(0.8),
            );
        }
    }

    // Squad centre markers and labels.
    if layers.centres {
        for s in squads {
            let c = t.to_screen(s.centroid.0, s.centroid.1);
            let color = squad_color(s.squad);
            let k = 6.0;
            let stroke = Stroke::new(2.0, color);
            painter.line_segment([c - vec2(k, 0.0), c + vec2(k, 0.0)], stroke);
            painter.line_segment([c - vec2(0.0, k), c + vec2(0.0, k)], stroke);
            painter.text(
                c + vec2(0.0, -t.len(s.footprint) - 4.0),
                Align2::CENTER_BOTTOM,
                format!("S{}", s.squad),
                FontId::proportional(13.0),
                color,
            );
        }
    }

    // Jammer marker, on top of the drones.
    if layers.jamming
        && let Some(s) = snap
    {
        draw_jammer(
            &painter,
            t.to_screen(s.jammer.position.x, s.jammer.position.y),
        );
    }

    // Tasks: a diamond in the kind's colour with its letter. Filled = placed near a
    // drone, hollow = uniform over the arena.
    if layers.tasks
        && let Some(s) = snap
    {
        let n_near = n_near_tasks(cfg);
        for (k, task) in s.tasks.iter().enumerate() {
            let c = t.to_screen(task.position.x, task.position.y);
            let color = task_color(task.kind);
            let r = 6.0;
            let diamond = vec![
                c + vec2(0.0, -r),
                c + vec2(r, 0.0),
                c + vec2(0.0, r),
                c + vec2(-r, 0.0),
            ];
            let near = k < n_near;
            painter.add(Shape::convex_polygon(
                diamond,
                if near { color } else { Color32::TRANSPARENT },
                Stroke::new(if near { 1.0 } else { 2.0 }, color),
            ));
            painter.text(
                c + vec2(r + 2.0, 0.0),
                Align2::LEFT_CENTER,
                task_letter(task.kind),
                FontId::proportional(11.0),
                color,
            );
            if *selected_task == Some(k) {
                painter.circle_stroke(c, r + 5.0, Stroke::new(2.5, text));
            }
        }
    }

    // Selection ring.
    if let Some(d) = selected.and_then(|i| swarm.get(i)) {
        let c = t.to_screen(d.position.x, d.position.y);
        painter.circle_stroke(c, radius + 5.0, Stroke::new(2.5, text));
    }

    // Hover and click. A task under the cursor wins over a drone.
    let near = |p: Pos2| nearest_drone(&t, swarm, p, radius + 6.0);
    let near_task = |p: Pos2| {
        snap.filter(|_| layers.tasks)
            .and_then(|s| nearest_task(&t, &s.tasks, p, 9.0))
    };
    if response.clicked() {
        let p = response.interact_pointer_pos();
        if let Some(k) = p.and_then(near_task) {
            *selected_task = Some(k);
            *selected = None;
        } else {
            *selected = p.and_then(near);
            *selected_task = None;
        }
    }
    if let (Some(k), Some(s)) = (response.hover_pos().and_then(near_task), snap) {
        let task = &s.tasks[k];
        let roles: Vec<&str> = Role::ALL
            .into_iter()
            .filter(|&r| task.kind.can_do(r))
            .map(role_name)
            .collect();
        let qualified = swarm.iter().filter(|d| task.kind.can_do(d.role)).count();
        let nearest = swarm
            .iter()
            .filter(|d| task.kind.can_do(d.role))
            .map(|d| (d.id.0, d.position.dist(task.position)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let near = k < n_near_tasks(cfg);
        response.on_hover_ui_at_pointer(|ui| {
            ui.strong(format!("Task {} · {}", task.id.0, task_name(task.kind)));
            ui.label(format!(
                "({:.2}, {:.2}) km · {}",
                task.position.x,
                task.position.y,
                if near {
                    "placed near a drone"
                } else {
                    "placed uniformly"
                }
            ));
            ui.label(format!("Roles: {}  ({qualified} drones)", roles.join(", ")));
            if let Some((id, dist)) = nearest {
                ui.label(format!("Nearest of them: drone {id}, {dist:.2} km"));
            }
            if task.kind.needs_relay() {
                ui.weak("Relay rule applies (step 3b): needs a relay within 2 hops.");
            }
            ui.weak("Click to highlight the drones that can take it");
        });
    } else if let Some(i) = response.hover_pos().and_then(near) {
        let d = &swarm[i];
        // Links of the hovered drone: count and weighted degree.
        let (n_links, w_degree) = graph
            .edges
            .iter()
            .filter(|e| e.src as usize == i || e.dst as usize == i)
            .fold((0, 0.0), |(n, w), e| (n + 1, w + e.weight));
        response.on_hover_ui_at_pointer(|ui| {
            ui.strong(format!("Drone {}  ·  squad S{}", d.id.0, d.squad.0));
            ui.label(format!("Role: {}", role_name(d.role)));
            ui.label(format!(
                "Position: ({:.2}, {:.2}) km",
                d.position.x, d.position.y
            ));
            ui.label(format!("Radio range: {:.1} km", d.role.radio_range(cfg)));
            ui.label(format!("Links: {n_links} · weighted degree {w_degree:.2}"));
            ui.label(format!(
                "Battery {:.2} · sensor {:.2} · compute {:.2}",
                d.battery, d.sensor_quality, d.compute_capacity
            ));
            if let Some(s) = snap {
                ui.label(format!(
                    "Jamming {:.3} (reading {:.3}) · {}",
                    s.truth[i],
                    s.readings[i],
                    if s.labels[i] {
                        "in the zone"
                    } else {
                        "outside the zone"
                    }
                ));
                let live = s
                    .live
                    .graph
                    .edges
                    .iter()
                    .filter(|e| e.src as usize == i || e.dst as usize == i)
                    .count();
                ui.label(format!("Live links: {live} of {n_links}"));
            }
            ui.weak("Click to select");
        });
    } else if let Some(p) = response.hover_pos() {
        // Cursor position in km, bottom-left of the map.
        let (x, y) = t.to_km(p);
        painter.text(
            rect.left_bottom() + vec2(8.0, -6.0),
            Align2::LEFT_BOTTOM,
            format!("{x:.1} km, {y:.1} km"),
            FontId::monospace(11.0),
            text.gamma_multiply(0.7),
        );
    }
}

/// Fill colour of a drone for the chosen encoding.
pub fn fill_color(d: &Drone, by: ColorBy) -> Color32 {
    match by {
        ColorBy::Squad => squad_color(d.squad.0 as usize),
        ColorBy::Role => role_color(d.role),
        ColorBy::Battery => ramp(d.battery),
        ColorBy::SensorQuality => ramp(d.sensor_quality),
        ColorBy::ComputeCapacity => ramp(d.compute_capacity),
    }
}

/// Index of the drone closest to `p` on screen, if within `max_px`.
fn nearest_drone(t: &Transform, swarm: &[Drone], p: Pos2, max_px: f32) -> Option<usize> {
    swarm
        .iter()
        .enumerate()
        .map(|(i, d)| (i, t.to_screen(d.position.x, d.position.y).distance(p)))
        .filter(|&(_, dist)| dist <= max_px)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Index of the task closest to `p` on screen, if within `max_px`.
fn nearest_task(t: &Transform, tasks: &[Task], p: Pos2, max_px: f32) -> Option<usize> {
    tasks
        .iter()
        .enumerate()
        .map(|(k, task)| (k, t.to_screen(task.position.x, task.position.y).distance(p)))
        .filter(|&(_, dist)| dist <= max_px)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(k, _)| k)
}

/// A jammer marker: a red disc with four short rays.
fn draw_jammer(painter: &Painter, c: Pos2) {
    let stroke = Stroke::new(2.0, Color32::WHITE);
    for k in 0..4 {
        let a = std::f32::consts::FRAC_PI_4 + k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = vec2(a.cos(), a.sin());
        painter.line_segment([c + dir * 9.0, c + dir * 14.0], stroke);
    }
    painter.circle(c, 7.0, JAMMER_COLOR, stroke);
}

/// Points on a circle, closed.
fn circle_points(c: Pos2, r: f32, n: usize) -> Vec<Pos2> {
    (0..=n)
        .map(|k| {
            let a = k as f32 / n as f32 * std::f32::consts::TAU;
            pos2(c.x + r * a.cos(), c.y + r * a.sin())
        })
        .collect()
}

/// Thin lines on the squad-cell boundaries.
fn draw_grid(painter: &Painter, t: &Transform, cfg: &SimConfig, color: Color32) {
    let stroke = Stroke::new(1.0, color);
    let cols = (cfg.arena_w / cfg.cell_width).round() as usize;
    let rows = (cfg.arena_h / cfg.cell_height).round() as usize;
    for c in 1..cols {
        let x = c as f64 * cfg.cell_width;
        painter.line_segment([t.to_screen(x, 0.0), t.to_screen(x, cfg.arena_h)], stroke);
    }
    for r in 1..rows {
        let y = r as f64 * cfg.cell_height;
        painter.line_segment([t.to_screen(0.0, y), t.to_screen(cfg.arena_w, y)], stroke);
    }
}

/// Tick labels in km along the bottom and left edges of the arena.
fn draw_axes(painter: &Painter, t: &Transform, cfg: &SimConfig, color: Color32) {
    // Pick a tick step that keeps labels at least ~45 px apart.
    let step = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        .into_iter()
        .find(|s| t.len(*s) >= 45.0)
        .unwrap_or(100.0);
    let font = FontId::monospace(10.0);
    let stroke = Stroke::new(1.0, color);

    let mut x = 0.0;
    while x <= cfg.arena_w + 1e-9 {
        let p = t.to_screen(x, 0.0);
        painter.line_segment([p, p + vec2(0.0, 4.0)], stroke);
        painter.text(
            p + vec2(0.0, 6.0),
            Align2::CENTER_TOP,
            format!("{x}"),
            font.clone(),
            color,
        );
        x += step;
    }
    let mut y = 0.0;
    while y <= cfg.arena_h + 1e-9 {
        let p = t.to_screen(0.0, y);
        painter.line_segment([p, p - vec2(4.0, 0.0)], stroke);
        painter.text(
            p - vec2(6.0, 0.0),
            Align2::RIGHT_CENTER,
            format!("{y}"),
            font.clone(),
            color,
        );
        y += step;
    }
    painter.text(
        t.to_screen(cfg.arena_w, 0.0) + vec2(0.0, 20.0),
        Align2::RIGHT_TOP,
        "km",
        font,
        color,
    );
}
