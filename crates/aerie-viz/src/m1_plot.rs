//! Mission 1 results: running the sweep in the background, and drawing its summary as a
//! plot (accuracy against report share) and a table. Shared by the swarm viewer's
//! Mission 1 view and the `m1_results` GUI.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use aerie_classic::m1::{
    GraphKind, LabelKind, Progress, SHARES, SummaryRow, Sweep, WRONG_SHARES, find, read_summary,
    sweep, units, write_results,
};
use eframe::egui::{
    self, Align2, Color32, FontId, Mesh, Pos2, Rect, RichText, Sense, Shape, Stroke, pos2, vec2,
};
use swarmsim::config::SimConfig;
use swarmsim::snapshot::TEST_SEEDS;

/// Method names, in the order of `aerie_classic::eval::Method::ALL`.
pub const METHODS: [&str; 5] = ["majority", "nearest_km", "nearest_hops", "lp", "rwr"];
pub const METHOD_COLORS: [Color32; 5] = [
    Color32::from_rgb(150, 150, 150),
    Color32::from_rgb(242, 142, 43),
    Color32::from_rgb(237, 201, 72),
    Color32::from_rgb(78, 121, 167),
    Color32::from_rgb(225, 87, 89),
];
/// How close the pointer must be to a point to show its numbers, pixels.
const HOVER_PX: f32 = 10.0;

/// What the plot and the table show.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotState {
    pub label: LabelKind,
    /// One per `GraphKind::ALL`.
    pub graphs: [bool; 3],
    /// One per `METHODS`.
    pub methods: [bool; 5],
    /// The ±1 sd band.
    pub band: bool,
    /// Index into `SHARES` for the table.
    pub table_share: usize,
    /// Index into `WRONG_SHARES`: which share of wrong reports the plot and table show.
    pub wrong: usize,
}

impl Default for PlotState {
    fn default() -> Self {
        PlotState {
            label: LabelKind::Squad,
            graphs: [true, false, false],
            methods: [true; 5],
            band: true,
            table_share: 3,
            wrong: 0,
        }
    }
}

/// A sweep running on its own thread, with its progress counter.
type Job = (Arc<Progress>, JoinHandle<Result<Sweep, String>>);

/// The sweep in a background thread, and the last results.
pub struct Runner {
    pub summary: Vec<SummaryRow>,
    pub status: String,
    job: Option<Job>,
}

impl Default for Runner {
    /// Starts with the results on disk, if any.
    fn default() -> Self {
        let summary = read_summary().unwrap_or_default();
        let status = if summary.is_empty() {
            "No results yet: press Run.".into()
        } else {
            format!("Loaded results/m1/summary.csv ({} rows).", summary.len())
        };
        Runner {
            summary,
            status,
            job: None,
        }
    }
}

impl Runner {
    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    /// Starts the full sweep (test seeds × 20 draws) with this config and α.
    pub fn start(&mut self, config: SimConfig, alpha: f64) {
        if self.running() {
            return;
        }
        let progress = Arc::new(Progress::default());
        let p = Arc::clone(&progress);
        let handle = std::thread::spawn(move || sweep(&config, TEST_SEEDS, alpha, &p));
        self.job = Some((progress, handle));
        self.status = "Running…".into();
    }

    pub fn cancel(&self) {
        if let Some((p, _)) = &self.job {
            p.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Share of the sweep done, while it runs.
    pub fn fraction(&self) -> Option<f32> {
        self.job
            .as_ref()
            .map(|(p, _)| p.done.load(Ordering::Relaxed) as f32 / units(&TEST_SEEDS).max(1) as f32)
    }

    /// Collects a finished sweep: keeps the summary and writes the CSV files.
    pub fn poll(&mut self, ctx: &egui::Context) {
        let Some((_, handle)) = &self.job else {
            return;
        };
        if !handle.is_finished() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            return;
        }
        let (_, handle) = self.job.take().unwrap();
        self.status = match handle.join() {
            Ok(Ok(result)) => {
                self.summary = result.summary.clone();
                match write_results(&result) {
                    Ok(dir) => format!("Done. Wrote {}.", dir.display()),
                    Err(e) => format!("Done, but could not write the CSV files: {e}"),
                }
            }
            Ok(Err(e)) => format!("Stopped: {e}."),
            Err(_) => "The sweep crashed.".into(),
        };
    }

    /// Run / Stop buttons, the progress bar and the status line.
    pub fn buttons(&mut self, ui: &mut egui::Ui, config: &SimConfig, alpha: f64) {
        ui.horizontal(|ui| {
            if self.running() {
                if ui.button("Stop").clicked() {
                    self.cancel();
                }
            } else if ui
                .button("▶ Run the experiment")
                .on_hover_text(
                    "Every method × graph × label × share on test seeds 90–99 × 20 draws, \
                     with the current settings and α. About 45 s. Writes results/m1/.",
                )
                .clicked()
            {
                self.start(*config, alpha);
            }
            if let Some(f) = self.fraction() {
                ui.add(
                    egui::ProgressBar::new(f)
                        .show_percentage()
                        .desired_width(160.0),
                );
            }
        });
        ui.weak(&self.status);
    }
}

/// Line style of each graph: live solid, epsilon dashed, kNN dotted.
pub fn graph_dash(graph: usize) -> Option<(f32, f32)> {
    match graph {
        0 => None,
        1 => Some((8.0, 5.0)),
        _ => Some((2.0, 4.0)),
    }
}

pub fn polyline(painter: &egui::Painter, pts: &[Pos2], stroke: Stroke, dash: Option<(f32, f32)>) {
    match dash {
        None => {
            painter.add(Shape::line(pts.to_vec(), stroke));
        }
        Some((d, g)) => {
            painter.extend(Shape::dashed_line(pts, stroke, d, g));
        }
    }
}

/// Label, graph and method pickers for the plot.
pub fn plot_controls(ui: &mut egui::Ui, st: &mut PlotState) {
    ui.horizontal(|ui| {
        ui.label("Label");
        for l in LabelKind::ALL {
            ui.selectable_value(&mut st.label, l, l.name());
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Graphs");
        for (g, graph) in GraphKind::ALL.iter().enumerate() {
            ui.checkbox(&mut st.graphs[g], graph.name());
            let (rect, _) = ui.allocate_exact_size(vec2(26.0, 12.0), Sense::hover());
            let y = rect.center().y;
            polyline(
                ui.painter(),
                &[pos2(rect.left(), y), pos2(rect.right(), y)],
                Stroke::new(2.0, ui.visuals().text_color()),
                graph_dash(g),
            );
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Methods");
        for (m, name) in METHODS.iter().enumerate() {
            let (rect, _) = ui.allocate_exact_size(vec2(10.0, 14.0), Sense::hover());
            ui.painter()
                .circle_filled(rect.center(), 4.5, METHOD_COLORS[m]);
            ui.checkbox(&mut st.methods[m], *name);
        }
        ui.checkbox(&mut st.band, "±1 sd");
    });
    ui.horizontal(|ui| {
        ui.label("Wrong reports");
        for (k, w) in WRONG_SHARES.iter().enumerate() {
            ui.selectable_value(&mut st.wrong, k, format!("{:.0} %", w * 100.0))
                .on_hover_text("Share of the reporters that say a wrong class.");
        }
    });
}

/// The rows of one line, sorted by share.
fn series<'a>(
    rows: &'a [SummaryRow],
    st: &PlotState,
    graph: GraphKind,
    method: &str,
) -> Vec<&'a SummaryRow> {
    let mut v: Vec<&SummaryRow> = rows
        .iter()
        .filter(|r| {
            r.label == st.label
                && r.graph == graph
                && r.method.name() == method
                && (r.wrong - WRONG_SHARES[st.wrong]).abs() < 1e-9
        })
        .collect();
    v.sort_by(|a, b| a.share.total_cmp(&b.share));
    v
}

/// Accuracy against the report share (log scale), filling the available space.
pub fn plot(ui: &mut egui::Ui, rows: &[SummaryRow], st: &PlotState) {
    let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
    let visuals = ui.visuals().clone();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let full = response.rect;
    painter.rect_filled(full, 4.0, visuals.extreme_bg_color);
    if rows.is_empty() {
        painter.text(
            full.center(),
            Align2::CENTER_CENTER,
            "No results yet: run the experiment.",
            FontId::proportional(15.0),
            weak,
        );
        return;
    }
    let area = Rect::from_min_max(full.min + vec2(52.0, 26.0), full.max - vec2(16.0, 34.0));
    if area.width() < 40.0 || area.height() < 40.0 {
        return;
    }
    let (lo, hi) = (SHARES[0].log10(), SHARES[SHARES.len() - 1].log10());
    let to_screen = |share: f64, acc: f64| {
        let t = ((share.log10() - lo) / (hi - lo)) as f32;
        pos2(
            area.left() + t * area.width(),
            area.bottom() - acc.clamp(0.0, 1.0) as f32 * area.height(),
        )
    };

    for k in 0..=4 {
        let acc = k as f64 / 4.0;
        let y = to_screen(SHARES[0], acc).y;
        painter.hline(
            area.x_range(),
            y,
            Stroke::new(1.0, weak.gamma_multiply(0.25)),
        );
        painter.text(
            pos2(area.left() - 6.0, y),
            Align2::RIGHT_CENTER,
            format!("{:.0} %", acc * 100.0),
            FontId::proportional(11.0),
            weak,
        );
    }
    for s in SHARES {
        let x = to_screen(s, 0.0).x;
        painter.vline(
            x,
            area.y_range(),
            Stroke::new(1.0, weak.gamma_multiply(0.15)),
        );
        painter.text(
            pos2(x, area.bottom() + 4.0),
            Align2::CENTER_TOP,
            format!("{:.0} %", s * 100.0),
            FontId::proportional(11.0),
            weak,
        );
    }
    painter.text(
        pos2(area.center().x, full.bottom() - 4.0),
        Align2::CENTER_BOTTOM,
        "share of drones that report (log scale)",
        FontId::proportional(11.0),
        weak,
    );
    painter.text(
        pos2(area.left(), full.top() + 6.0),
        Align2::LEFT_TOP,
        format!(
            "label = {}, {:.0} % wrong reports: accuracy on the silent drones (no guess counts as wrong)",
            st.label.name(),
            WRONG_SHARES[st.wrong] * 100.0
        ),
        FontId::proportional(13.0),
        text,
    );

    let pointer = response.hover_pos();
    let mut nearest: Option<(f32, SummaryRow)> = None;
    for (g, graph) in GraphKind::ALL.iter().enumerate() {
        if !st.graphs[g] {
            continue;
        }
        for (m, method) in METHODS.iter().enumerate() {
            if !st.methods[m] {
                continue;
            }
            let line = series(rows, st, *graph, method);
            if line.is_empty() {
                continue;
            }
            let color = METHOD_COLORS[m];
            if st.band {
                band(&painter, &line, &to_screen, color.gamma_multiply(0.12));
            }
            let pts: Vec<Pos2> = line
                .iter()
                .map(|r| to_screen(r.share, r.accuracy))
                .collect();
            polyline(&painter, &pts, Stroke::new(2.0, color), graph_dash(g));
            for (p, r) in pts.iter().zip(&line) {
                painter.circle_filled(*p, 3.0, color);
                if let Some(at) = pointer {
                    let d = p.distance(at);
                    if d < HOVER_PX && nearest.is_none_or(|(best, _)| d < best) {
                        nearest = Some((d, **r));
                    }
                }
            }
        }
    }

    if let Some((_, r)) = nearest {
        painter.circle_stroke(to_screen(r.share, r.accuracy), 6.0, Stroke::new(2.0, text));
        response.on_hover_ui_at_pointer(|ui| {
            ui.strong(format!(
                "{} on {}, {:.0} % report",
                r.method.name(),
                r.graph.name(),
                r.share * 100.0
            ));
            ui.label(format!(
                "accuracy {:.1} % ± {:.1}",
                r.accuracy * 100.0,
                r.accuracy_sd * 100.0
            ));
            ui.label(format!(
                "no guess {:.1} % of the silent drones",
                r.no_guess * 100.0
            ));
            if !r.overruled.is_nan() {
                ui.label(format!(
                    "overruled {:.0} % of the wrong reports",
                    r.overruled * 100.0
                ));
            }
            ui.weak(format!("{} runs", r.runs));
        });
    }
}

/// Every method on every graph at one share: accuracy ± sd (no-guess share).
pub fn table(ui: &mut egui::Ui, rows: &[SummaryRow], st: &mut PlotState) {
    if rows.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("label = {}", st.label.name())).strong());
        let k = &mut st.table_share;
        *k = (*k).min(SHARES.len() - 1);
        egui::ComboBox::from_id_salt("m1_table_share")
            .selected_text(format!("{:.0} % report", SHARES[*k] * 100.0))
            .show_ui(ui, |ui| {
                for (i, s) in SHARES.iter().enumerate() {
                    ui.selectable_value(k, i, format!("{:.0} %", s * 100.0));
                }
            });
        ui.label(format!("{:.0} % wrong", WRONG_SHARES[st.wrong] * 100.0));
        ui.weak("accuracy ± sd (no guess) [lies overruled], %; bold = best graph for the method");
    });
    let share = SHARES[st.table_share];
    let wrong = WRONG_SHARES[st.wrong.min(WRONG_SHARES.len() - 1)];
    egui::Grid::new("m1_table")
        .striped(true)
        .min_col_width(130.0)
        .show(ui, |ui| {
            ui.label("");
            for graph in GraphKind::ALL {
                ui.strong(graph.name());
            }
            ui.end_row();
            for (m, method) in METHODS.iter().enumerate() {
                ui.colored_label(METHOD_COLORS[m], *method);
                let cells: Vec<Option<SummaryRow>> = GraphKind::ALL
                    .iter()
                    .map(|&g| find(rows, g, share, wrong, st.label, method))
                    .collect();
                let best = cells
                    .iter()
                    .flatten()
                    .map(|r| r.accuracy)
                    .fold(f64::NEG_INFINITY, f64::max);
                for cell in cells {
                    match cell {
                        Some(r) => {
                            let mut s = format!(
                                "{:.1} ± {:.1}  ({:.1})",
                                r.accuracy * 100.0,
                                r.accuracy_sd * 100.0,
                                r.no_guess * 100.0
                            );
                            if !r.overruled.is_nan() {
                                s.push_str(&format!("  [{:.0}]", r.overruled * 100.0));
                            }
                            if r.accuracy == best {
                                ui.label(RichText::new(s).strong());
                            } else {
                                ui.label(s);
                            }
                        }
                        None => {
                            ui.weak("—");
                        }
                    }
                }
                ui.end_row();
            }
        });
}

/// The ±1 sd band of a series as a strip of triangles.
fn band(
    painter: &egui::Painter,
    series: &[&SummaryRow],
    to_screen: &impl Fn(f64, f64) -> Pos2,
    color: Color32,
) {
    let mut mesh = Mesh::default();
    for r in series {
        mesh.colored_vertex(to_screen(r.share, r.accuracy + r.accuracy_sd), color);
        mesh.colored_vertex(to_screen(r.share, r.accuracy - r.accuracy_sd), color);
    }
    for k in 0..series.len().saturating_sub(1) as u32 {
        let (a, b, c, d) = (2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3);
        mesh.add_triangle(a, b, c);
        mesh.add_triangle(b, d, c);
    }
    painter.add(Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_names_match_the_experiment() {
        let names: Vec<&str> = aerie_classic::eval::Method::ALL
            .iter()
            .map(|m| m.name())
            .collect();
        assert_eq!(names, METHODS);
    }
}
