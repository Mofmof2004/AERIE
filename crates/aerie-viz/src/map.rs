//! The arena map: cells, squads, radio ranges and drones, with zoom and pan.

use aerie_graph::EdgeList;
use eframe::egui::{
    self, Align2, Color32, FontId, Painter, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2,
};
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};

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
    /// Draw the comm graph (epsilon graph) links.
    pub links: bool,
    pub link_color: LinkColor,
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

/// Draws the map and handles zoom, pan, hover and click-to-select.
// Eight inputs, each a separate piece of viewer state; bundling them would only move the list.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    ui: &mut egui::Ui,
    view: &mut MapView,
    layers: &Layers,
    cfg: &SimConfig,
    swarm: &[Drone],
    graph: &EdgeList,
    squads: &[SquadInfo],
    selected: &mut Option<usize>,
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

    // Links of the comm graph. Width = weight. With a drone selected, its links stay
    // bright and the others fade. Skipped if the graph is out of date with the swarm.
    if layers.links && graph.n as usize == swarm.len() {
        for e in &graph.edges {
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
    for (i, d) in swarm.iter().enumerate() {
        let c = t.to_screen(d.position.x, d.position.y);
        let mut fill = fill_color(d, layers.color_by);
        if layers.dim_other_squads
            && selected_squad.is_some_and(|sq| sq != d.squad.0 as usize)
            && *selected != Some(i)
        {
            fill = fill.gamma_multiply(0.25);
        }
        draw_drone(&painter, c, radius, d.role, fill, outline);
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

    // Selection ring.
    if let Some(d) = selected.and_then(|i| swarm.get(i)) {
        let c = t.to_screen(d.position.x, d.position.y);
        painter.circle_stroke(c, radius + 5.0, Stroke::new(2.5, text));
    }

    // Hover and click.
    let near = |p: Pos2| nearest_drone(&t, swarm, p, radius + 6.0);
    if response.clicked() {
        *selected = response.interact_pointer_pos().and_then(near);
    }
    if let Some(i) = response.hover_pos().and_then(near) {
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
