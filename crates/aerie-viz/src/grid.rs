//! A matrix drawn as a grid of coloured cells, shared by the playgrounds that show
//! matrices (matrix playground, label propagation playground).

use eframe::egui::{self, Align2, Color32, FontId, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use faer::Mat;

/// Cell size, pixels.
pub const CELL_W: f32 = 42.0;
pub const CELL_H: f32 = 20.0;
/// Size of the row and column headers, pixels.
pub const HEAD: f32 = 26.0;
/// Width of the row-sum column, pixels.
pub const SUM_W: f32 = 48.0;
/// Outline of the hovered cell and of focused rows and columns.
pub const HOVER_COLOR: Color32 = Color32::from_rgb(255, 200, 40);

/// What to outline in a grid.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Focus {
    pub row: Option<usize>,
    pub col: Option<usize>,
    pub cell: Option<(usize, usize)>,
}

/// Draws `m` with row and column headers (text and colour each) and, if `sum_label` is
/// given, a row-sum column. Cells are shaded by |value| in `accent`. Returns the cell
/// under the pointer and the grid's response (for a tooltip).
pub fn matrix_grid(
    ui: &mut egui::Ui,
    m: &Mat<f64>,
    row_heads: &[(String, Color32)],
    col_heads: &[(String, Color32)],
    focus: Focus,
    accent: Color32,
    sum_label: Option<&str>,
) -> (Option<(usize, usize)>, egui::Response) {
    let (rows, cols) = (m.nrows(), m.ncols());
    let sum_w = if sum_label.is_some() {
        6.0 + SUM_W
    } else {
        0.0
    };
    let size = vec2(
        HEAD + cols as f32 * CELL_W + sum_w,
        HEAD + rows as f32 * CELL_H,
    );
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    let visuals = ui.visuals();
    let bg = visuals.extreme_bg_color;
    let text = visuals.strong_text_color();
    let weak = visuals.weak_text_color();
    let origin = rect.min + vec2(HEAD, HEAD);
    let cell_rect = |i: usize, j: usize| {
        Rect::from_min_size(
            origin + vec2(j as f32 * CELL_W, i as f32 * CELL_H),
            vec2(CELL_W, CELL_H),
        )
    };
    let max = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| m[(i, j)].abs()))
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    // Headers.
    for (j, (label, color)) in col_heads.iter().enumerate().take(cols) {
        painter.text(
            pos2(
                origin.x + (j as f32 + 0.5) * CELL_W,
                rect.min.y + HEAD / 2.0,
            ),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(12.0),
            *color,
        );
    }
    for (i, (label, color)) in row_heads.iter().enumerate().take(rows) {
        painter.text(
            pos2(
                rect.min.x + HEAD / 2.0,
                origin.y + (i as f32 + 0.5) * CELL_H,
            ),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(12.0),
            *color,
        );
    }
    let sum_x = origin.x + cols as f32 * CELL_W + 6.0;
    if let Some(label) = sum_label {
        painter.text(
            pos2(sum_x + SUM_W / 2.0, rect.min.y + HEAD / 2.0),
            Align2::CENTER_CENTER,
            label,
            FontId::proportional(11.0),
            weak,
        );
    }

    // Cells, then the row sums.
    for i in 0..rows {
        for j in 0..cols {
            let r = cell_rect(i, j);
            let v = m[(i, j)];
            let fill = if v == 0.0 {
                bg
            } else {
                mix(bg, accent, 0.25 + 0.65 * (v.abs() / max) as f32)
            };
            painter.rect_filled(r.shrink(0.5), 2.0, fill);
            let label = if v == 0.0 { "·".to_string() } else { fmt(v) };
            painter.text(
                r.center(),
                Align2::CENTER_CENTER,
                label,
                FontId::monospace(11.0),
                if v == 0.0 { weak } else { text },
            );
        }
        if sum_label.is_some() {
            let s: f64 = (0..cols).map(|j| m[(i, j)]).sum();
            painter.text(
                pos2(sum_x + SUM_W / 2.0, origin.y + (i as f32 + 0.5) * CELL_H),
                Align2::CENTER_CENTER,
                fmt(s),
                FontId::monospace(11.0),
                weak,
            );
        }
    }

    // Outlines.
    let soft = Stroke::new(1.5, HOVER_COLOR.gamma_multiply(0.7));
    if rows > 0 && cols > 0 {
        if let Some(f) = focus.row.filter(|&f| f < rows) {
            let r = cell_rect(f, 0).union(cell_rect(f, cols - 1));
            painter.rect_stroke(r, 2.0, soft, StrokeKind::Inside);
        }
        if let Some(f) = focus.col.filter(|&f| f < cols) {
            let r = cell_rect(0, f).union(cell_rect(rows - 1, f));
            painter.rect_stroke(r, 2.0, soft, StrokeKind::Inside);
        }
        if let Some((i, j)) = focus.cell.filter(|&(i, j)| i < rows && j < cols) {
            painter.rect_stroke(
                cell_rect(i, j),
                2.0,
                Stroke::new(2.0, HOVER_COLOR),
                StrokeKind::Inside,
            );
        }
    }

    let cell = response.hover_pos().and_then(|p| {
        let local = p - origin;
        if local.x < 0.0 || local.y < 0.0 {
            return None;
        }
        let (j, i) = ((local.x / CELL_W) as usize, (local.y / CELL_H) as usize);
        (i < rows && j < cols).then_some((i, j))
    });
    (cell, response)
}

/// A tooltip at the pointer with one line of text.
pub fn show_tip(response: egui::Response, text: &str) {
    response.on_hover_ui_at_pointer(|ui| {
        ui.label(text);
    });
}

/// Short number: whole numbers without decimals, others with 2 (or 3 when small).
pub fn fmt(v: f64) -> String {
    if v == v.round() && v.abs() < 1e6 {
        format!("{v:.0}")
    } else if v.abs() < 0.1 {
        format!("{v:.3}")
    } else {
        format!("{v:.2}")
    }
}

/// Linear mix of two colours, `t` = 0 gives `a`, 1 gives `b`.
pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(f(a.r(), b.r()), f(a.g(), b.g()), f(a.b(), b.b()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_is_short() {
        assert_eq!(fmt(1.0), "1");
        assert_eq!(fmt(0.5), "0.50");
        assert_eq!(fmt(0.04), "0.040");
    }

    #[test]
    fn mix_ends_are_the_two_colours() {
        let (a, b) = (Color32::from_rgb(0, 0, 0), Color32::from_rgb(200, 100, 50));
        assert_eq!(mix(a, b, 0.0), a);
        assert_eq!(mix(a, b, 1.0), b);
        assert_eq!(mix(a, b, 0.5), Color32::from_rgb(100, 50, 25));
    }
}
