//! Colours, shapes and names shared by the map and the side panels.

use eframe::egui::{self, Color32, Painter, Pos2, Sense, Shape, Stroke, pos2, vec2};
use swarmsim::drone::Role;

/// Ten well-separated colours for the first ten squads (Tableau 10).
const SQUAD_COLORS: [Color32; 10] = [
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(225, 87, 89),
    Color32::from_rgb(118, 183, 178),
    Color32::from_rgb(89, 161, 79),
    Color32::from_rgb(237, 201, 72),
    Color32::from_rgb(176, 122, 161),
    Color32::from_rgb(255, 157, 167),
    Color32::from_rgb(156, 117, 95),
    Color32::from_rgb(186, 176, 172),
];

/// Colour of squad `k`. Beyond ten squads, hues are spread by the golden ratio.
pub fn squad_color(k: usize) -> Color32 {
    if k < SQUAD_COLORS.len() {
        SQUAD_COLORS[k]
    } else {
        hsv(((k as f32) * 0.618_034).fract(), 0.55, 0.9)
    }
}

/// Colour of a role, used when the map is coloured by role.
pub fn role_color(role: Role) -> Color32 {
    match role {
        Role::Recon => Color32::from_rgb(78, 121, 167),
        Role::RfSensing => Color32::from_rgb(225, 87, 89),
        Role::Relay => Color32::from_rgb(89, 161, 79),
        Role::Compute => Color32::from_rgb(176, 122, 161),
    }
}

/// Human-readable role name.
pub fn role_name(role: Role) -> &'static str {
    match role {
        Role::Recon => "Recon",
        Role::RfSensing => "RF sensing",
        Role::Relay => "Relay",
        Role::Compute => "Compute",
    }
}

/// Shape of a role, in words, for the legend.
pub fn role_shape_name(role: Role) -> &'static str {
    match role {
        Role::Recon => "circle",
        Role::RfSensing => "triangle",
        Role::Relay => "square",
        Role::Compute => "diamond",
    }
}

/// Red (0) to amber (0.5) to green (1), for battery, sensor quality, compute.
pub fn ramp(t: f64) -> Color32 {
    let t = t.clamp(0.0, 1.0) as f32;
    let (a, b, u) = if t < 0.5 {
        ([215.0, 48.0, 39.0], [240.0, 190.0, 60.0], t * 2.0)
    } else {
        ([240.0, 190.0, 60.0], [26.0, 152.0, 80.0], (t - 0.5) * 2.0)
    };
    let mix = |i: usize| (a[i] + (b[i] - a[i]) * u) as u8;
    Color32::from_rgb(mix(0), mix(1), mix(2))
}

/// Draws one drone: the shape says the role, the fill is chosen by the caller.
/// `r` is the radius in pixels.
pub fn draw_drone(painter: &Painter, c: Pos2, r: f32, role: Role, fill: Color32, stroke: Stroke) {
    let points = match role {
        Role::Recon => {
            painter.circle(c, r, fill, stroke);
            return;
        }
        Role::RfSensing => vec![
            pos2(c.x, c.y - 1.25 * r),
            pos2(c.x + 1.15 * r, c.y + 0.8 * r),
            pos2(c.x - 1.15 * r, c.y + 0.8 * r),
        ],
        Role::Relay => {
            let s = 0.9 * r;
            vec![
                pos2(c.x - s, c.y - s),
                pos2(c.x + s, c.y - s),
                pos2(c.x + s, c.y + s),
                pos2(c.x - s, c.y + s),
            ]
        }
        Role::Compute => {
            let s = 1.3 * r;
            vec![
                pos2(c.x, c.y - s),
                pos2(c.x + s, c.y),
                pos2(c.x, c.y + s),
                pos2(c.x - s, c.y),
            ]
        }
    };
    painter.add(Shape::convex_polygon(points, fill, stroke));
}

/// A small inline swatch for legends: the role's shape in the given colour.
pub fn swatch(ui: &mut egui::Ui, role: Role, fill: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
    let outline = Stroke::new(1.0, ui.visuals().text_color().gamma_multiply(0.6));
    draw_drone(ui.painter(), rect.center(), 6.0, role, fill, outline);
}

/// HSV to RGB, all inputs in 0..1.
fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (i as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}
