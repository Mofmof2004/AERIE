//! The swarm viewer's Mission 1 view: who reports on this seed (and who reports
//! something wrong), what each method guesses, and the whole experiment with its plot.
//!
//! - Left panel (`controls`): label, graph, method, α, who reports (random draw with a
//!   share of wrong reports, or picked by hand), display options.
//! - Map (`overlay`): drones filled with their propagated label, reporters ringed (gold:
//!   true report, red: wrong report), wrong guesses crossed, the graph the method ran on.
//! - Right panel (`details`): the reporters, every method's score on this draw, and the
//!   selected drone's guess under each method.
//! - Bottom panel (`bottom`): run the experiment over the test seeds, plot and table.

use aerie_classic::eval::{Method, Problem, Score, corrupt, draw_reporters, guesses, hide, score};
use aerie_classic::m1::{GraphKind, LabelKind, SHARES, methods, positions};
use aerie_graph::EdgeList;
use eframe::egui::{self, Color32, RichText};
use faer::Mat;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::snapshot::TEST_SEEDS;

use crate::m1_plot::{METHOD_COLORS, METHODS, PlotState, Runner, plot, plot_controls, table};
use crate::map::{JAMMED_DRONE_COLOR, LabelOverlay, RELAY_FAIL_COLOR, Snapshot};
use crate::style::{role_color, role_name, squad_color};

/// Grey for a drone no label reached.
const NO_GUESS: Color32 = Color32::from_rgb(110, 110, 110);
/// Zone class 0 (outside the jammed zone).
const FREE_COLOR: Color32 = Color32::from_rgb(90, 140, 210);
/// Silent drones in the "Reports only" view.
const SILENT_COLOR: Color32 = Color32::from_rgb(55, 55, 60);
/// The reporters' marker (same gold as on the map).
pub const REPORT_COLOR: Color32 = crate::map::REPORT_COLOR;
/// Wrong reports (same red as on the map).
pub const LIE_COLOR: Color32 = RELAY_FAIL_COLOR;

/// Colour of class `c` of a label.
pub fn class_color(label: LabelKind, c: usize) -> Color32 {
    match label {
        LabelKind::Zone => [FREE_COLOR, JAMMED_DRONE_COLOR][c.min(1)],
        LabelKind::Squad => squad_color(c),
        LabelKind::Role => role_color(Role::ALL[c.min(3)]),
    }
}

/// Name of class `c` of a label.
pub fn class_name(label: LabelKind, c: usize) -> String {
    match label {
        LabelKind::Zone => ["outside", "jammed"][c.min(1)].to_string(),
        LabelKind::Squad => format!("S{c}"),
        LabelKind::Role => role_name(Role::ALL[c.min(3)]).to_string(),
    }
}

/// The wrong class a hand-picked liar reports: the next class after its true one.
pub fn wrong_class(truth: usize, n_classes: usize) -> usize {
    (truth + 1) % n_classes.max(1)
}

/// Who reports: a random draw (as in the experiment) or drones picked by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportSource {
    Draw,
    Hand,
}

/// A hand-picked drone's state. A click moves it to the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandReport {
    Silent,
    /// Reports its true class.
    Honest,
    /// Reports a wrong class (`wrong_class`).
    Wrong,
}

impl HandReport {
    pub fn next(self) -> Self {
        match self {
            HandReport::Silent => HandReport::Honest,
            HandReport::Honest => HandReport::Wrong,
            HandReport::Wrong => HandReport::Silent,
        }
    }
}

/// Where the reports come from, as far as the computation is concerned.
#[derive(Debug, Clone, PartialEq)]
enum Who {
    Draw { share: f64, draw: u64, wrong: f64 },
    Hand(Vec<HandReport>),
}

/// Everything that decides this seed's picture; recomputed only when it changes.
#[derive(Debug, Clone, PartialEq)]
struct Key {
    seed: u64,
    config: SimConfig,
    label: LabelKind,
    graph: GraphKind,
    alpha: f64,
    who: Who,
}

/// One seed and draw, every method run.
struct Computed {
    key: Key,
    links: EdgeList,
    truth: Vec<usize>,
    /// What each drone told the methods: `None` silent, `Some(class)` reported.
    said: Vec<Option<usize>>,
    /// `said[i].is_some()`, for the overlay.
    reports: Vec<bool>,
    /// Reporters that said a wrong class.
    lies: Vec<bool>,
    /// Y of each method, in the order of `METHODS`.
    ys: Vec<Mat<f64>>,
    scores: Vec<Score>,
}

pub struct M1View {
    pub label: LabelKind,
    pub graph: GraphKind,
    /// Index into `METHODS` of the method shown on the map.
    pub method: usize,
    pub alpha: f64,
    pub share: f64,
    pub draw: u64,
    /// Random draw: share of the reporters that say a wrong class.
    pub wrong: f64,
    pub source: ReportSource,
    /// The hand-picked reports, one per drone; reset when the seed or swarm changes.
    pub hand: Vec<HandReport>,
    /// The (seed, number of drones) `hand` belongs to.
    hand_for: Option<(u64, usize)>,
    pub mark_wrong: bool,
    pub truth_ring: bool,
    /// Show only who reports (round 0) instead of the propagated labels.
    pub reports_only: bool,
    pub runner: Runner,
    pub plot: PlotState,
    computed: Option<Computed>,
    // The overlay's per-drone slices, rebuilt with `computed`.
    fill: Vec<Color32>,
    wrong_marks: Vec<bool>,
    no_guess: Vec<bool>,
    truth_colors: Vec<Color32>,
    hover: Vec<String>,
}

impl Default for M1View {
    fn default() -> Self {
        M1View {
            label: LabelKind::Squad,
            graph: GraphKind::Live,
            method: 3,
            alpha: 0.85,
            share: 0.1,
            draw: 0,
            wrong: 0.0,
            source: ReportSource::Draw,
            hand: Vec::new(),
            hand_for: None,
            mark_wrong: true,
            truth_ring: false,
            reports_only: false,
            runner: Runner::default(),
            plot: PlotState::default(),
            computed: None,
            fill: Vec::new(),
            wrong_marks: Vec::new(),
            no_guess: Vec::new(),
            truth_colors: Vec::new(),
            hover: Vec::new(),
        }
    }
}

/// The current random draw as hand-picked reports (all honest).
fn hand_from_draw(n: usize, share: f64, seed: u64, draw: u64) -> Vec<HandReport> {
    draw_reporters(n, share, seed, draw)
        .into_iter()
        .map(|r| {
            if r {
                HandReport::Honest
            } else {
                HandReport::Silent
            }
        })
        .collect()
}

impl M1View {
    /// Reruns every method if the seed, config or a setting changed.
    pub fn update(
        &mut self,
        seed: u64,
        config: &SimConfig,
        swarm: &[Drone],
        geometric: &EdgeList,
        snap: Option<&Snapshot>,
    ) {
        let Some(snap) = snap else {
            self.computed = None;
            return;
        };
        let n = swarm.len();
        if self.hand_for != Some((seed, n)) {
            // A new swarm: start the hand-picked set from the current draw.
            self.hand = hand_from_draw(n, self.share, seed, self.draw);
            self.hand_for = Some((seed, n));
        }
        let who = match self.source {
            ReportSource::Draw => Who::Draw {
                share: self.share,
                draw: self.draw,
                wrong: self.wrong,
            },
            ReportSource::Hand => Who::Hand(self.hand.clone()),
        };
        let key = Key {
            seed,
            config: *config,
            label: self.label,
            graph: self.graph,
            alpha: self.alpha,
            who,
        };
        if self.computed.as_ref().is_none_or(|c| c.key != key) {
            self.computed = Some(compute(key, swarm, geometric, snap));
        }
        self.refresh_overlay();
    }

    /// A drone was clicked on the map: in hand-picked mode it moves on to the next state,
    /// silent → reports → wrong report → silent.
    pub fn clicked(&mut self, drone: usize) {
        if self.source == ReportSource::Hand
            && let Some(r) = self.hand.get_mut(drone)
        {
            *r = r.next();
        }
    }

    /// The per-drone colours and marks of the shown method.
    fn refresh_overlay(&mut self) {
        let Some(c) = &self.computed else { return };
        let y = &c.ys[self.method];
        let g = guesses(y);
        let n = c.truth.len();
        let n_classes = y.ncols();
        if self.reports_only {
            // Round 0: only what got through. Reporters in the class they said, silent
            // drones dark.
            self.fill = c
                .said
                .iter()
                .map(|s| s.map_or(SILENT_COLOR, |k| class_color(self.label, k)))
                .collect();
            self.no_guess = vec![false; n];
            self.wrong_marks = vec![false; n];
        } else {
            self.fill = (0..n).map(|i| mix_row(y, i, self.label)).collect();
            self.no_guess = (0..n).map(|i| !c.reports[i] && g[i].is_none()).collect();
            // Crossed: a silent drone guessed wrong, or a lie the method kept.
            self.wrong_marks = (0..n)
                .map(|i| {
                    self.mark_wrong
                        && (!c.reports[i] || c.lies[i])
                        && g[i].is_some_and(|k| k != c.truth[i])
                })
                .collect();
        }
        self.truth_colors = c
            .truth
            .iter()
            .map(|&t| class_color(self.label, t))
            .collect();
        let name = |k: usize| class_name(self.label, k);
        self.hover = (0..n)
            .map(|i| {
                let row: Vec<String> = (0..n_classes)
                    .filter(|&k| y[(i, k)] > 0.005)
                    .map(|k| format!("{} {:.2}", name(k), y[(i, k)]))
                    .collect();
                let status = match (c.said[i], g[i]) {
                    (Some(s), Some(k)) if c.lies[i] && k == c.truth[i] => {
                        format!("WRONG report: says {}; overruled ✓", name(s))
                    }
                    (Some(s), _) if c.lies[i] => format!("WRONG report: says {}", name(s)),
                    (Some(s), _) => format!("reports {}", name(s)),
                    (None, None) => "silent, no guess".to_string(),
                    (None, Some(k)) if k == c.truth[i] => format!("guess {} ✓", name(k)),
                    (None, Some(k)) => format!("guess {} ✗", name(k)),
                };
                format!(
                    "Mission 1 ({}, {}): true {} · {status}\nY: {}",
                    self.label.name(),
                    METHODS[self.method],
                    name(c.truth[i]),
                    if row.is_empty() {
                        "all zero".into()
                    } else {
                        row.join(", ")
                    }
                )
            })
            .collect();
    }

    /// Each method's score on this seed and draw, in the order of `METHODS`.
    pub fn scores(&self) -> Option<&[Score]> {
        self.computed.as_ref().map(|c| c.scores.as_slice())
    }

    pub fn overlay(&self) -> Option<LabelOverlay<'_>> {
        let c = self.computed.as_ref()?;
        Some(LabelOverlay {
            fill: &self.fill,
            reports: &c.reports,
            lies: &c.lies,
            wrong: &self.wrong_marks,
            no_guess: &self.no_guess,
            truth: self.truth_ring.then_some(self.truth_colors.as_slice()),
            links: &c.links,
            hover: &self.hover,
            marked_links: &[],
        })
    }

    /// Left panel: what to propagate, on which graph, with which method, from which
    /// reports.
    pub fn controls(&mut self, ui: &mut egui::Ui, seed: &mut u64) {
        ui.label(RichText::new("Mission 1: label propagation").strong());
        if !TEST_SEEDS.contains(seed) {
            ui.horizontal(|ui| {
                ui.weak("Not a test seed.");
                if ui.small_button("Go to seed 90").clicked() {
                    *seed = TEST_SEEDS.start;
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Label");
            for l in LabelKind::ALL {
                ui.selectable_value(&mut self.label, l, l.name());
            }
        });
        ui.horizontal(|ui| {
            ui.label("Graph");
            for g in GraphKind::ALL {
                ui.selectable_value(&mut self.graph, g, g.name());
            }
        });
        egui::ComboBox::from_label("Method on the map")
            .selected_text(METHODS[self.method])
            .show_ui(ui, |ui| {
                for (m, name) in METHODS.iter().enumerate() {
                    ui.selectable_value(
                        &mut self.method,
                        m,
                        RichText::new(*name).color(METHOD_COLORS[m]),
                    );
                }
            });
        ui.add(egui::Slider::new(&mut self.alpha, 0.0..=0.99).text("RWR α"));

        ui.horizontal(|ui| {
            ui.label("Who reports");
            ui.selectable_value(&mut self.source, ReportSource::Draw, "Random draw")
                .on_hover_text("As in the experiment: a share of the drones, drawn at random.");
            ui.selectable_value(&mut self.source, ReportSource::Hand, "Pick by hand")
                .on_hover_text("Click drones on the map: silent → reports → wrong report.");
        });
        match self.source {
            ReportSource::Draw => {
                ui.add(
                    egui::Slider::new(&mut self.share, SHARES[0]..=0.5)
                        .logarithmic(true)
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .text("report"),
                );
                ui.add(
                    egui::Slider::new(&mut self.wrong, 0.0..=0.5)
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .text("of them wrong"),
                )
                .on_hover_text(
                    "Share of the reporters that say a wrong class (picked at random, fixed by \
                     seed and draw). The experiment runs 0, 10 and 25 %.",
                );
                ui.horizontal(|ui| {
                    ui.label("Draw");
                    ui.add(egui::DragValue::new(&mut self.draw).range(0..=999));
                    if ui.small_button("◀").clicked() {
                        self.draw = self.draw.saturating_sub(1);
                    }
                    if ui.small_button("▶").clicked() {
                        self.draw += 1;
                    }
                });
                ui.weak("Draw = which drones report (the experiment uses draws 0–19).");
            }
            ReportSource::Hand => {
                ui.label(
                    RichText::new("Click a drone: silent → reports → wrong report → silent.")
                        .color(REPORT_COLOR),
                );
                ui.weak("A wrong report says the next class after the true one.");
                ui.horizontal(|ui| {
                    if ui.button("Clear all").clicked() {
                        self.hand.fill(HandReport::Silent);
                    }
                    if ui
                        .button("Copy the random draw")
                        .on_hover_text("Start from the draw set by the share and draw number.")
                        .clicked()
                        && let Some((seed, n)) = self.hand_for
                    {
                        self.hand = hand_from_draw(n, self.share, seed, self.draw);
                    }
                });
                ui.weak("The experiment (Run) still uses random draws.");
            }
        }

        ui.horizontal(|ui| {
            ui.label("Show");
            ui.selectable_value(&mut self.reports_only, true, "Reports only")
                .on_hover_text("Who got a report through, and what they said. Silent drones dark.");
            ui.selectable_value(&mut self.reports_only, false, "Propagated labels")
                .on_hover_text("Every drone filled with the label the method gave it.");
        });
        if let Some(c) = &self.computed {
            let n_rep = c.reports.iter().filter(|&&r| r).count();
            let n_lie = c.lies.iter().filter(|&&l| l).count();
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{n_rep} of {} drones report", c.reports.len()))
                        .color(REPORT_COLOR)
                        .strong(),
                );
                if n_lie > 0 {
                    ui.label(
                        RichText::new(format!("· {n_lie} wrong"))
                            .color(LIE_COLOR)
                            .strong(),
                    );
                }
            });
        }
        ui.add_enabled(
            !self.reports_only,
            egui::Checkbox::new(&mut self.mark_wrong, "Cross wrong guesses and kept lies"),
        );
        ui.checkbox(&mut self.truth_ring, "Ring in the true class colour");
        ui.weak(
            "Gold ring + R = reports, red ring + R = wrong report. Fill = class (mixed by Y \
             when propagated). ? = no guess.",
        );
        if let Some(c) = &self.computed {
            ui.horizontal_wrapped(|ui| {
                for k in 0..c.ys[self.method].ncols() {
                    ui.label(RichText::new("●").color(class_color(self.label, k)));
                    ui.label(class_name(self.label, k));
                }
            });
        }
        self.refresh_overlay();
    }

    /// Right panel: who reports, this draw's scores per method, and the selected drone.
    /// Clicking a reporter in the list selects it on the map.
    pub fn details(&self, ui: &mut egui::Ui, selected: &mut Option<usize>) {
        ui.label(RichText::new("Mission 1, this seed and draw").strong());
        let Some(c) = &self.computed else {
            ui.weak("No snapshot for this seed.");
            return;
        };
        let name = |k: usize| class_name(self.label, k);
        let silent = c.reports.iter().filter(|&&r| !r).count();
        ui.label(format!(
            "{} report, {silent} silent · label {} on the {} graph",
            c.reports.len() - silent,
            self.label.name(),
            self.graph.name()
        ));

        // Who reports, and what each one said. Click to select it on the map.
        ui.add_space(4.0);
        ui.label(
            RichText::new("Reporting drones")
                .color(REPORT_COLOR)
                .strong(),
        );
        ui.horizontal_wrapped(|ui| {
            for (i, said) in c.said.iter().enumerate() {
                let Some(k) = *said else { continue };
                let text = if c.lies[i] {
                    RichText::new(format!("{i}: {} ✗ (is {})", name(k), name(c.truth[i])))
                        .color(LIE_COLOR)
                } else {
                    RichText::new(format!("{i}: {}", name(k))).color(class_color(self.label, k))
                };
                if ui
                    .selectable_label(*selected == Some(i), text)
                    .on_hover_text("Select this drone on the map")
                    .clicked()
                {
                    *selected = Some(i);
                }
            }
        });
        // Which classes nobody reported: label propagation can never guess them.
        let n_classes = c.ys[0].ncols();
        let missing: Vec<String> = (0..n_classes)
            .filter(|&k| !c.said.contains(&Some(k)))
            .map(name)
            .collect();
        if !missing.is_empty() {
            ui.weak(format!("Nobody reported: {}", missing.join(", ")));
        }

        ui.add_space(4.0);
        let any_lie = c.lies.iter().any(|&l| l);
        egui::Grid::new("m1_this_draw")
            .striped(true)
            .show(ui, |ui| {
                ui.label("method");
                ui.label("right");
                ui.label("accuracy");
                ui.label("no guess");
                if any_lie {
                    ui.label("lies overruled");
                }
                ui.end_row();
                for (m, s) in c.scores.iter().enumerate() {
                    let label = RichText::new(METHODS[m]).color(METHOD_COLORS[m]);
                    ui.label(if m == self.method {
                        label.strong()
                    } else {
                        label
                    });
                    ui.label(format!("{}/{}", s.right, s.scored));
                    ui.label(format!("{:.1} %", s.accuracy() * 100.0));
                    ui.label(format!("{}", s.no_guess));
                    if any_lie {
                        ui.label(format!("{}/{}", s.overruled, s.liars));
                    }
                    ui.end_row();
                }
            });

        if let Some(i) = selected.filter(|&i| i < c.truth.len()) {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("Drone {i}")).strong());
            ui.label(format!(
                "true {} · {}",
                name(c.truth[i]),
                match c.said[i] {
                    None => "silent".to_string(),
                    Some(k) if c.lies[i] => format!("reports {} (wrong)", name(k)),
                    Some(_) => "reports".to_string(),
                }
            ));
            for (m, y) in c.ys.iter().enumerate() {
                let g = guesses(y)[i];
                let mark = match g {
                    _ if c.reports[i] && !c.lies[i] => "",
                    None => " · no guess",
                    Some(k) if k == c.truth[i] => " ✓",
                    Some(_) => " ✗",
                };
                ui.label(format!(
                    "{:<12} → {}{mark}",
                    METHODS[m],
                    g.map_or("—".into(), name)
                ));
            }
        }
        ui.separator();
    }

    /// Bottom panel: the experiment over the test seeds, its plot and table.
    pub fn bottom(&mut self, ui: &mut egui::Ui, config: &SimConfig) {
        self.runner.poll(ui.ctx());
        ui.horizontal(|ui| {
            ui.label(RichText::new("Mission 1 experiment").strong());
            ui.weak(
                "test seeds 90–99 × 20 draws, every method × graph × label × share × wrong \
                 share (0, 10, 25 %)",
            );
        });
        self.runner.buttons(ui, config, self.alpha);
        plot_controls(ui, &mut self.plot);
        ui.columns(2, |cols| {
            plot(&mut cols[0], &self.runner.summary, &self.plot);
            egui::ScrollArea::both()
                .id_salt("m1_table_scroll")
                .show(&mut cols[1], |ui| {
                    table(ui, &self.runner.summary, &mut self.plot)
                });
        });
    }
}

/// Runs every method on one seed and set of reports.
fn compute(key: Key, swarm: &[Drone], geometric: &EdgeList, snap: &Snapshot) -> Computed {
    let links = key
        .graph
        .build(swarm, geometric, &snap.live.graph, &key.config);
    let p = links.transition_matrix();
    let truth = key.label.truth(swarm, &snap.labels);
    let n_classes = key.label.n_classes(&key.config);
    let said: Vec<Option<usize>> = match &key.who {
        // Exactly what the experiment does for this seed and draw.
        Who::Draw { share, draw, wrong } => {
            let reports = draw_reporters(swarm.len(), *share, key.seed, *draw);
            corrupt(&hide(&truth, &reports), n_classes, *wrong, key.seed, *draw)
        }
        Who::Hand(hand) => hand
            .iter()
            .zip(&truth)
            .map(|(h, &t)| match h {
                HandReport::Silent => None,
                HandReport::Honest => Some(t),
                HandReport::Wrong => Some(wrong_class(t, n_classes)),
            })
            .collect(),
    };
    let reports: Vec<bool> = said.iter().map(Option::is_some).collect();
    let lies: Vec<bool> = said
        .iter()
        .zip(&truth)
        .map(|(s, &t)| s.is_some_and(|k| k != t))
        .collect();
    let pos = positions(swarm);
    let problem = Problem {
        p: &p,
        positions: &pos,
        labels: &said,
        n_classes,
    };
    let ys: Vec<Mat<f64>> = methods(key.alpha)
        .iter()
        .map(|m: &Method| m.run(&problem))
        .collect();
    let scores = ys.iter().map(|y| score(y, &truth, &said)).collect();
    Computed {
        key,
        links,
        truth,
        said,
        reports,
        lies,
        ys,
        scores,
    }
}

/// The class colours mixed by the shares in row `i` of Y; grey when the row is zero.
fn mix_row(y: &Mat<f64>, i: usize, label: LabelKind) -> Color32 {
    let total: f64 = (0..y.ncols()).map(|c| y[(i, c)].max(0.0)).sum();
    if total <= 0.0 {
        return NO_GUESS;
    }
    let (mut r, mut g, mut b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for c in 0..y.ncols() {
        let w = (y[(i, c)].max(0.0) / total) as f32;
        let col = class_color(label, c);
        r += w * col.r() as f32;
        g += w * col.g() as f32;
        b += w * col.b() as f32;
    }
    Color32::from_rgb(r as u8, g as u8, b as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixing_a_one_hot_row_gives_the_class_colour() {
        let mut y: Mat<f64> = Mat::zeros(2, 3);
        y[(0, 2)] = 1.0;
        assert_eq!(mix_row(&y, 0, LabelKind::Squad), squad_color(2));
        assert_eq!(mix_row(&y, 1, LabelKind::Squad), NO_GUESS);
    }

    #[test]
    fn class_names_and_colours_cover_every_class() {
        let config = SimConfig::default();
        for label in LabelKind::ALL {
            for c in 0..label.n_classes(&config) {
                assert!(!class_name(label, c).is_empty());
                let _ = class_color(label, c);
            }
        }
        assert_eq!(class_color(LabelKind::Zone, 1), JAMMED_DRONE_COLOR);
    }

    #[test]
    fn a_click_cycles_silent_honest_wrong() {
        let s = HandReport::Silent;
        assert_eq!(s.next(), HandReport::Honest);
        assert_eq!(s.next().next(), HandReport::Wrong);
        assert_eq!(s.next().next().next(), HandReport::Silent);
    }

    #[test]
    fn the_wrong_class_is_never_the_truth() {
        for n in 2..12 {
            for t in 0..n {
                let w = wrong_class(t, n);
                assert!(w < n && w != t);
            }
        }
    }
}
