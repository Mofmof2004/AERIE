//! The swarm viewer's Mission 2 view: what a listener learns from one flight's traffic.
//!
//! - Left panel (`controls`): the plan's 300-drone swarm, DeepWalk's settings, what the
//!   map shows.
//! - Map (`overlay`): the traffic graph (width = log messages). Drones filled by k-means
//!   cluster or guessed role, the truth as an outer ring, wrong ones crossed. The hidden
//!   links of the link test dashed, green when DeepWalk ranks them above most non-links,
//!   red when below.
//! - Right panel (`details`): this seed's scores, the selected drone, its nearest vectors.
//! - Bottom panel (`bottom`): run the test seeds, chart and table.
//!
//! Embedding a seed takes about a second, so it runs on a background thread: the map
//! keeps the last result until the new one is ready.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use aerie_classic::embed::deepwalk::DeepWalkConfig;
use aerie_classic::embed::skipgram::dot;
use aerie_classic::m2::{
    EXPERIMENT_BIASES, Method, Progress, Row, SeedRun, Summary, run_seed, summarize, sweep,
};
use aerie_classic::m2::{plan_config, read_rows, rows_path, write_rows};
use aerie_graph::{Edge, EdgeList};
use eframe::egui::{self, Color32, RichText, Sense, Stroke, pos2, vec2};
use swarmsim::config::SimConfig;
use swarmsim::drone::Role;
use swarmsim::snapshot::TEST_SEEDS;

use crate::map::{LabelOverlay, RELAY_FAIL_COLOR};
use crate::style::{ramp, role_color, role_name, squad_color};

/// What the drones' fill shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Show {
    /// k-means clusters of Z, each in the colour of the squad most of its drones are in.
    Clusters,
    /// The true squads.
    Squads,
    /// Role guessed from Z (5-fold logistic regression).
    RoleFromZ,
    /// Role guessed from the traffic counts.
    RoleFromCounts,
    /// The true roles.
    Roles,
}

impl Show {
    pub const ALL: [Show; 5] = [
        Show::Clusters,
        Show::Squads,
        Show::RoleFromZ,
        Show::RoleFromCounts,
        Show::Roles,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Show::Clusters => "k-means clusters of Z",
            Show::Squads => "true squads",
            Show::RoleFromZ => "role guessed from Z",
            Show::RoleFromCounts => "role guessed from counts",
            Show::Roles => "true roles",
        }
    }

    fn about(self) -> &'static str {
        match self {
            Show::Clusters => {
                "Each cluster takes the colour of the squad most of its drones belong to. \
                 Ring: the drone's true squad. Cross: its cluster is led by another squad."
            }
            Show::Squads => "What the clusters should find.",
            Show::RoleFromZ => {
                "Logistic regression on DeepWalk's vectors, each drone guessed by a model \
                 that never saw it (5 folds). Ring: true role. Cross: wrong guess."
            }
            Show::RoleFromCounts => {
                "Logistic regression on 3 numbers per drone: messages sent, received, and \
                 partners. No embedding at all."
            }
            Show::Roles => "What the role guesses should find.",
        }
    }

    fn is_role(self) -> bool {
        matches!(self, Show::RoleFromZ | Show::RoleFromCounts | Show::Roles)
    }
}

/// Colour of each method in the chart and table.
pub fn method_color(m: Method) -> Color32 {
    match m {
        Method::DeepWalk => Color32::from_rgb(78, 121, 167),
        // Node2Vec by q: outward (q < 1) teal, local (q > 1) purple, q = 1 light blue.
        Method::Node2Vec { q, .. } if q < 1.0 => Color32::from_rgb(44, 170, 150),
        Method::Node2Vec { q, .. } if q > 1.0 => Color32::from_rgb(176, 122, 200),
        Method::Node2Vec { .. } => Color32::from_rgb(130, 175, 225),
        Method::Mf => Color32::from_rgb(156, 117, 95),
        Method::RandomVectors => Color32::from_rgb(150, 150, 150),
        Method::TrafficCounts => Color32::from_rgb(242, 142, 43),
        Method::Majority => Color32::from_rgb(100, 100, 110),
    }
}

/// Which embedding the map and the drone details use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shown {
    DeepWalk,
    Node2Vec,
    Mf,
}

/// What one embedding was computed from.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Key {
    seed: u64,
    config: SimConfig,
    dw: DeepWalkConfig,
    /// Node2Vec's p and q (it runs next to DeepWalk).
    p: f64,
    q: f64,
}

/// For each cluster, the squad most of its drones belong to (ties: the lower squad).
pub fn leading_squad(clusters: &[usize], squads: &[usize]) -> Vec<usize> {
    let k = clusters.iter().max().map_or(0, |m| m + 1);
    let s = squads.iter().max().map_or(0, |m| m + 1);
    let mut counts = vec![vec![0usize; s]; k];
    for (&c, &q) in clusters.iter().zip(squads) {
        counts[c][q] += 1;
    }
    counts
        .iter()
        .map(|row| {
            (0..s)
                .max_by_key(|&q| (row[q], std::cmp::Reverse(q)))
                .unwrap_or(0)
        })
        .collect()
}

/// Share of `others` that `x` beats (a tie counts half): 1 = above them all.
fn share_beaten(x: f64, others: &[f64]) -> f64 {
    if others.is_empty() {
        return 0.5;
    }
    let wins: f64 = others
        .iter()
        .map(|&o| {
            if x > o {
                1.0
            } else if x == o {
                0.5
            } else {
                0.0
            }
        })
        .sum();
    wins / others.len() as f64
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let d = (dot(a, a) * dot(b, b)).sqrt();
    if d > 0.0 { dot(a, b) / d } else { 0.0 }
}

pub struct M2View {
    pub dw: DeepWalkConfig,
    /// Node2Vec's return and in-out parameters (same walks and skip-gram as `dw`).
    pub p: f64,
    pub q: f64,
    /// The embedding the map, the drone details and the nearest vectors use.
    pub shown: Shown,
    pub show: Show,
    pub truth_ring: bool,
    pub mark_wrong: bool,
    pub hidden_links: bool,
    /// The viewer already switched to the 300-drone swarm once (it does on first visit).
    pub moved_to_plan: bool,
    /// What the map should show now.
    wanted: Option<Key>,
    /// The last finished embedding.
    current: Option<(Key, SeedRun)>,
    /// The embedding being computed.
    job: Option<(Key, JoinHandle<SeedRun>)>,
    status: String,
    pub runner: Runner,
    // The overlay's per-drone slices, rebuilt with `current` or a display setting.
    fill: Vec<Color32>,
    wrong: Vec<bool>,
    truth_colors: Vec<Color32>,
    hover: Vec<String>,
    links: EdgeList,
    marked: Vec<(usize, usize, Color32)>,
    /// The settings `fill` etc. were built for.
    overlay_for: Option<(Key, Show, bool, bool, Shown)>,
}

impl Default for M2View {
    fn default() -> Self {
        M2View {
            dw: DeepWalkConfig::default(),
            p: 1.0,
            q: 0.5,
            shown: Shown::DeepWalk,
            show: Show::Clusters,
            truth_ring: true,
            mark_wrong: true,
            hidden_links: false,
            moved_to_plan: false,
            wanted: None,
            current: None,
            job: None,
            status: String::new(),
            runner: Runner::default(),
            fill: Vec::new(),
            wrong: Vec::new(),
            truth_colors: Vec::new(),
            hover: Vec::new(),
            links: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            marked: Vec::new(),
            overlay_for: None,
        }
    }
}

impl M2View {
    /// Starts embedding the current seed if it changed; collects a finished embedding.
    /// `snapshot_ok`: the viewer could build this seed's snapshot (else nothing to embed).
    pub fn update(
        &mut self,
        ctx: &egui::Context,
        seed: u64,
        config: &SimConfig,
        snapshot_ok: bool,
    ) {
        self.runner.poll(ctx);
        if !snapshot_ok {
            self.wanted = None;
            self.status = "No snapshot for this seed (jamming could not be placed).".into();
            return;
        }
        let key = Key {
            seed,
            config: *config,
            dw: self.dw,
            p: self.p,
            q: self.q,
        };
        self.wanted = Some(key);

        if self.job.as_ref().is_some_and(|(_, h)| h.is_finished()) {
            let (done, handle) = self.job.take().unwrap();
            match handle.join() {
                Ok(run) => {
                    self.current = Some((done, run));
                    self.status.clear();
                }
                Err(_) => self.status = format!("Embedding seed {} crashed.", done.seed),
            }
        }
        let up_to_date = self.current.as_ref().is_some_and(|(k, _)| *k == key);
        if !up_to_date && self.job.is_none() {
            // One job at a time: if the settings move while it runs, the next one starts
            // when it ends, with the latest settings.
            self.job = Some((
                key,
                std::thread::spawn(move || {
                    run_seed(key.seed, &key.config, &key.dw, &[(key.p, key.q)])
                }),
            ));
        }
        if let Some((k, _)) = &self.job {
            self.status = format!(
                "Embedding seed {} ({} drones)…",
                k.seed,
                k.config.n_squads * k.config.squad_size
            );
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.refresh_overlay();
    }

    /// Which of the seed's embeddings the map uses: 0 DeepWalk, 1 Node2Vec.
    fn shown_index(&self, run: &SeedRun) -> usize {
        // The seed's embeddings: DeepWalk, Node2Vec (the sliders' p, q), MF.
        let last = run.embeddings.len() - 1;
        match self.shown {
            Shown::DeepWalk => 0,
            Shown::Node2Vec => 1.min(last),
            Shown::Mf => last,
        }
    }

    /// The finished embedding, if it is the one for the current seed and settings.
    fn run(&self) -> Option<&SeedRun> {
        let (k, run) = self.current.as_ref()?;
        (Some(*k) == self.wanted).then_some(run)
    }

    fn refresh_overlay(&mut self) {
        let Some((key, run)) = &self.current else {
            return;
        };
        let wanted = (
            *key,
            self.show,
            self.mark_wrong,
            self.hidden_links,
            self.shown,
        );
        if self.overlay_for == Some(wanted) {
            return;
        }
        let e = &run.embeddings[self.shown_index(run)];
        let n = run.roles.len();
        let leaders = leading_squad(&e.clusters, &run.squads);
        let role_of = |k: usize| Role::ALL[k];
        let (fill, truth, wrong): (Vec<Color32>, Vec<Color32>, Vec<bool>) = match self.show {
            Show::Clusters => (
                e.clusters
                    .iter()
                    .map(|&c| squad_color(leaders[c]))
                    .collect(),
                run.squads.iter().map(|&q| squad_color(q)).collect(),
                (0..n)
                    .map(|i| leaders[e.clusters[i]] != run.squads[i])
                    .collect(),
            ),
            Show::Squads => (
                run.squads.iter().map(|&q| squad_color(q)).collect(),
                run.squads.iter().map(|&q| squad_color(q)).collect(),
                vec![false; n],
            ),
            Show::RoleFromZ | Show::RoleFromCounts => {
                let guess = if self.show == Show::RoleFromZ {
                    &e.role_guess
                } else {
                    &run.counts_guess
                };
                (
                    guess.iter().map(|&k| role_color(role_of(k))).collect(),
                    run.roles.iter().map(|&k| role_color(role_of(k))).collect(),
                    (0..n).map(|i| guess[i] != run.roles[i]).collect(),
                )
            }
            Show::Roles => (
                run.roles.iter().map(|&k| role_color(role_of(k))).collect(),
                run.roles.iter().map(|&k| role_color(role_of(k))).collect(),
                vec![false; n],
            ),
        };
        self.fill = fill;
        self.truth_colors = truth;
        self.wrong = if self.mark_wrong {
            wrong
        } else {
            vec![false; n]
        };

        // Link widths: log messages, scaled so the busiest link is 2 px wide.
        let top = run.graph.edges.iter().map(|e| e.weight).fold(1.0, f64::max);
        self.links = EdgeList {
            n: run.graph.n,
            edges: run
                .graph
                .edges
                .iter()
                .map(|e| Edge {
                    src: e.src,
                    dst: e.dst,
                    weight: 0.6 * (1.0 + e.weight).ln() / (1.0 + top).ln(),
                })
                .collect(),
        };
        self.marked = if self.hidden_links {
            run.split
                .hidden
                .iter()
                .zip(&e.hidden_scores)
                .map(|(&(a, b), &s)| (a, b, ramp(share_beaten(s, &e.non_link_scores))))
                .collect()
        } else {
            Vec::new()
        };

        let mut partners = vec![0usize; n];
        for e in &run.graph.edges {
            partners[e.src as usize] += 1;
            partners[e.dst as usize] += 1;
        }
        let mark = |ok: bool| if ok { "✓" } else { "✗" };
        self.hover = (0..n)
            .map(|i| {
                let c = e.clusters[i];
                format!(
                    "squad {} · cluster {c} (led by squad {}) {}\n\
                     role {} · from Z: {} {} · from counts: {} {}\n\
                     sent {} · received {} · {} partners",
                    run.squads[i],
                    leaders[c],
                    mark(leaders[c] == run.squads[i]),
                    role_name(role_of(run.roles[i])),
                    role_name(role_of(e.role_guess[i])),
                    mark(e.role_guess[i] == run.roles[i]),
                    role_name(role_of(run.counts_guess[i])),
                    mark(run.counts_guess[i] == run.roles[i]),
                    run.sent[i],
                    run.received[i],
                    partners[i],
                )
            })
            .collect();
        self.overlay_for = Some(wanted);
    }

    /// What the map draws, once the current seed is embedded.
    pub fn overlay(&self) -> Option<LabelOverlay<'_>> {
        self.run()?;
        Some(LabelOverlay {
            fill: &self.fill,
            reports: &[],
            lies: &[],
            wrong: &self.wrong,
            no_guess: &[],
            truth: (self.truth_ring && !matches!(self.show, Show::Squads | Show::Roles))
                .then_some(self.truth_colors.as_slice()),
            links: &self.links,
            hover: &self.hover,
            marked_links: &self.marked,
        })
    }

    /// Left panel: the swarm, DeepWalk's settings, what the map shows.
    pub fn controls(&mut self, ui: &mut egui::Ui, config: &mut SimConfig) {
        ui.label(RichText::new("Mission 2: Fingerprints").strong());
        ui.weak(
            "A listener records who talks to whom during one flight. DeepWalk learns a \
             vector per drone from that log alone; the scores say what it gives away.",
        );
        let n = config.n_squads * config.squad_size;
        let plan = plan_config();
        if *config != plan {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(RELAY_FAIL_COLOR, format!("{n} drones: the plan uses 300."));
                if ui
                    .button("Use the 300-drone swarm")
                    .on_hover_text("30 squads on a 70 × 35 km arena (SimConfig::with_squads(30)).")
                    .clicked()
                {
                    *config = plan;
                }
            });
        }

        ui.add_space(4.0);
        ui.label(RichText::new("DeepWalk").strong());
        egui::Grid::new("m2_dw").num_columns(2).show(ui, |ui| {
            let dw = &mut self.dw;
            ui.label("walks per drone");
            ui.add(egui::DragValue::new(&mut dw.walks_per_node).range(1..=50));
            ui.end_row();
            ui.label("walk length");
            ui.add(egui::DragValue::new(&mut dw.walk_length).range(2..=100));
            ui.end_row();
            ui.label("dimension");
            ui.add(egui::DragValue::new(&mut dw.skipgram.dim).range(2..=128));
            ui.end_row();
            ui.label("window");
            ui.add(egui::DragValue::new(&mut dw.skipgram.window).range(1..=20));
            ui.end_row();
            ui.label("negatives");
            ui.add(egui::DragValue::new(&mut dw.skipgram.negatives).range(1..=20));
            ui.end_row();
            ui.label("epochs");
            ui.add(egui::DragValue::new(&mut dw.skipgram.epochs).range(1..=10));
            ui.end_row();
            ui.label("learning rate");
            ui.add(
                egui::DragValue::new(&mut dw.skipgram.lr)
                    .range(0.001..=0.5)
                    .speed(0.001),
            );
            ui.end_row();
        });
        if self.dw != DeepWalkConfig::default() && ui.small_button("Plan's settings").clicked() {
            self.dw = DeepWalkConfig::default();
        }

        ui.add_space(4.0);
        ui.label(RichText::new("Node2Vec").strong());
        ui.add(
            egui::Slider::new(&mut self.p, 0.1..=10.0)
                .logarithmic(true)
                .text("p (return)"),
        );
        ui.add(
            egui::Slider::new(&mut self.q, 0.1..=10.0)
                .logarithmic(true)
                .text("q (in-out)"),
        );
        ui.weak(
            "Same walks and skip-gram settings as DeepWalk, the walks biased by p and q \
             (p = q = 1 is DeepWalk exactly). The Node2Vec playground shows what they do to \
             a walk.",
        );
        ui.horizontal(|ui| {
            ui.label("Map uses");
            ui.selectable_value(&mut self.shown, Shown::DeepWalk, "DeepWalk");
            ui.selectable_value(
                &mut self.shown,
                Shown::Node2Vec,
                Method::Node2Vec {
                    p: self.p,
                    q: self.q,
                }
                .name(),
            );
            ui.selectable_value(&mut self.shown, Shown::Mf, "MF");
        });
        if !self.status.is_empty() {
            ui.horizontal(|ui| {
                if self.job.is_some() {
                    ui.add(egui::Spinner::new());
                }
                ui.weak(&self.status);
            });
        }

        ui.add_space(4.0);
        ui.label(RichText::new("Map").strong());
        egui::ComboBox::from_id_salt("m2_show")
            .selected_text(self.show.name())
            .width(220.0)
            .show_ui(ui, |ui| {
                for s in Show::ALL {
                    ui.selectable_value(&mut self.show, s, s.name());
                }
            });
        ui.weak(self.show.about());
        ui.checkbox(&mut self.truth_ring, "Ring: the truth");
        ui.checkbox(&mut self.mark_wrong, "Cross: wrong");
        ui.checkbox(&mut self.hidden_links, "Hidden links of the link test")
            .on_hover_text(
                "The 10 % of links DeepWalk didn't see in its second run. Green: it scores \
                 the pair above most pairs that never talked (a hit); red: below.",
            );
        if self.show.is_role() {
            ui.horizontal_wrapped(|ui| {
                for role in Role::ALL {
                    ui.colored_label(role_color(role), "●");
                    ui.label(role_name(role));
                }
            });
        }
        self.refresh_overlay();
    }

    /// Right panel: this seed's scores, the selected drone and its nearest vectors.
    pub fn details(&self, ui: &mut egui::Ui, selected: &mut Option<usize>) {
        ui.label(RichText::new("Mission 2, this seed").strong());
        let Some(run) = self.run() else {
            ui.weak(if self.status.is_empty() {
                "Embedding…"
            } else {
                &self.status
            });
            ui.separator();
            return;
        };
        ui.label(format!(
            "{} drones · {} links that carried traffic · {} hidden for the link test",
            run.roles.len(),
            run.graph.edges.len(),
            run.split.hidden.len()
        ));
        score_grid(ui, "m2_seed_scores", &summarize(&run.rows));
        let e = &run.embeddings[self.shown_index(run)];
        ui.label(format!("Map and drone details: {}", e.method.name()));
        let leaders = leading_squad(&e.clusters, &run.squads);
        let placed = (0..run.roles.len())
            .filter(|&i| leaders[e.clusters[i]] == run.squads[i])
            .count();
        ui.weak(format!(
            "{placed}/{} drones sit in a cluster led by their own squad.",
            run.roles.len()
        ));

        if let Some(i) = selected.filter(|&i| i < run.roles.len()) {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("Drone {i}")).strong());
            ui.label(format!(
                "squad {} · {} · cluster {} (led by squad {})",
                run.squads[i],
                role_name(Role::ALL[run.roles[i]]),
                e.clusters[i],
                leaders[e.clusters[i]]
            ));
            ui.label(format!(
                "role from Z: {} · from counts: {}",
                role_name(Role::ALL[e.role_guess[i]]),
                role_name(Role::ALL[run.counts_guess[i]])
            ));
            ui.add_space(2.0);
            ui.label("Nearest vectors (cosine of z):");
            let mut near: Vec<(usize, f64)> = (0..e.z.len())
                .filter(|&j| j != i)
                .map(|j| (j, cosine(&e.z[i], &e.z[j])))
                .collect();
            near.sort_by(|a, b| b.1.total_cmp(&a.1));
            egui::Grid::new("m2_nearest").striped(true).show(ui, |ui| {
                for &(j, cos) in near.iter().take(8) {
                    let same = run.squads[j] == run.squads[i];
                    if ui
                        .selectable_label(false, format!("drone {j}"))
                        .on_hover_text("Select it")
                        .clicked()
                    {
                        *selected = Some(j);
                    }
                    ui.colored_label(
                        squad_color(run.squads[j]),
                        format!("squad {}{}", run.squads[j], if same { " ✓" } else { "" }),
                    );
                    ui.label(role_name(Role::ALL[run.roles[j]]));
                    ui.label(format!("{cos:.2}"));
                    ui.end_row();
                }
            });
        }
        ui.separator();
    }

    /// Bottom panel: run the test seeds with the current settings; chart and table.
    pub fn bottom(&mut self, ui: &mut egui::Ui, config: &SimConfig) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Mission 2 experiment").strong());
            ui.weak(
                "test seeds 90–99: DeepWalk, Node2Vec (p = 1 with q = 0.5 and q = 2, and the \
                 p, q above), the baselines; the plan's three scores",
            );
        });
        let mut biases = EXPERIMENT_BIASES.to_vec();
        if !biases.contains(&(self.p, self.q)) {
            biases.push((self.p, self.q));
        }
        self.runner.buttons(ui, config, &self.dw, &biases);
        results(ui, &self.runner.rows);
    }
}

/// The scores of each method as a grid: method, seeds, role F1, link AUC, squad NMI.
pub fn score_grid(ui: &mut egui::Ui, id: &str, summary: &[Summary]) {
    let cell = |v: Option<(f64, f64)>, seeds: usize| match v {
        None => "—".to_string(),
        Some((m, _)) if seeds == 1 => format!("{m:.3}"),
        Some((m, s)) => format!("{m:.3} ± {s:.3}"),
    };
    egui::Grid::new(id).striped(true).show(ui, |ui| {
        ui.label("method");
        ui.label("role F1");
        ui.label("link AUC");
        ui.label("squad NMI");
        ui.end_row();
        for s in summary {
            ui.colored_label(method_color(s.method), s.method.name());
            ui.label(cell(Some(s.role_f1), s.seeds));
            ui.label(cell(s.link_auc, s.seeds));
            ui.label(cell(s.squad_nmi, s.seeds));
            ui.end_row();
        }
    });
}

/// Chart (one group of bars per score, whiskers = spread over seeds) and table.
pub fn results(ui: &mut egui::Ui, rows: &[Row]) {
    let summary = summarize(rows);
    if summary.is_empty() {
        return;
    }
    ui.columns(2, |cols| {
        chart(&mut cols[0], &summary);
        egui::ScrollArea::vertical()
            .id_salt("m2_table")
            .show(&mut cols[1], |ui| {
                score_grid(ui, "m2_summary", &summary);
                ui.add_space(4.0);
                ui.weak(
                    "Role F1: macro-F1, 5-fold logistic regression (chance ≈ 0.25, all-recon \
                     0.17). Link AUC: hidden link vs never-talked pair, 0.5 = guessing. Squad \
                     NMI: k-means vs true squads; random vectors set the floor, not 0.",
                );
                egui::CollapsingHeader::new("Per seed").show(ui, |ui| {
                    egui::Grid::new("m2_per_seed").striped(true).show(ui, |ui| {
                        for h in ["seed", "method", "role F1", "link AUC", "squad NMI", "s"] {
                            ui.label(h);
                        }
                        ui.end_row();
                        let opt = |v: Option<f64>| v.map_or("—".into(), |x| format!("{x:.3}"));
                        for r in rows {
                            ui.label(r.seed.to_string());
                            ui.colored_label(method_color(r.method), r.method.name());
                            ui.label(format!("{:.3}", r.role_f1));
                            ui.label(opt(r.link_auc));
                            ui.label(opt(r.squad_nmi));
                            ui.label(format!("{:.2}", r.seconds));
                            ui.end_row();
                        }
                    });
                });
            });
    });
}

/// Grouped bars: role F1, link AUC, squad NMI; one bar per method that has the score.
fn chart(ui: &mut egui::Ui, summary: &[Summary]) {
    let size = vec2(ui.available_width().min(520.0), 210.0);
    let (response, painter) = ui.allocate_painter(size, Sense::hover());
    let rect = response.rect;
    let text = ui.visuals().text_color();
    let faint = text.gamma_multiply(0.2);
    let plot = egui::Rect::from_min_max(rect.min + vec2(34.0, 8.0), rect.max - vec2(8.0, 34.0));
    let y = |v: f64| plot.bottom() - (v.clamp(0.0, 1.0) as f32) * plot.height();
    for k in 0..=4 {
        let v = k as f64 / 4.0;
        painter.line_segment(
            [pos2(plot.left(), y(v)), pos2(plot.right(), y(v))],
            Stroke::new(1.0, faint),
        );
        painter.text(
            pos2(plot.left() - 4.0, y(v)),
            egui::Align2::RIGHT_CENTER,
            format!("{v:.2}"),
            egui::FontId::proportional(10.0),
            text,
        );
    }
    type Pick = fn(&Summary) -> Option<(f64, f64)>;
    let groups: [(&str, Pick); 3] = [
        ("role F1", |s| Some(s.role_f1)),
        ("link AUC", |s| s.link_auc),
        ("squad NMI", |s| s.squad_nmi),
    ];
    let group_w = plot.width() / groups.len() as f32;
    let mut hovered: Option<String> = None;
    for (g, (name, pick)) in groups.iter().enumerate() {
        let bars: Vec<(&Summary, (f64, f64))> = summary
            .iter()
            .filter_map(|s| pick(s).map(|v| (s, v)))
            .collect();
        let left = plot.left() + g as f32 * group_w + group_w * 0.12;
        let bar_w = group_w * 0.76 / bars.len().max(1) as f32;
        for (b, (s, (mean, sd))) in bars.iter().enumerate() {
            let x0 = left + b as f32 * bar_w;
            let bar = egui::Rect::from_min_max(
                pos2(x0 + 1.0, y(*mean)),
                pos2(x0 + bar_w - 1.0, plot.bottom()),
            );
            painter.rect_filled(bar, 2.0, method_color(s.method));
            let cx = bar.center().x;
            let whisker = Stroke::new(1.5, text.gamma_multiply(0.8));
            painter.line_segment([pos2(cx, y(mean - sd)), pos2(cx, y(mean + sd))], whisker);
            if response
                .hover_pos()
                .is_some_and(|p| bar.expand2(vec2(0.0, 6.0)).contains(p))
            {
                hovered = Some(format!("{} · {name}: {mean:.3} ± {sd:.3}", s.method.name()));
            }
        }
        painter.text(
            pos2(
                plot.left() + (g as f32 + 0.5) * group_w,
                plot.bottom() + 4.0,
            ),
            egui::Align2::CENTER_TOP,
            *name,
            egui::FontId::proportional(12.0),
            text,
        );
    }
    // Legend under the groups.
    let mut x = plot.left();
    for s in summary {
        let at = pos2(x, plot.bottom() + 22.0);
        painter.rect_filled(
            egui::Rect::from_center_size(at + vec2(5.0, 0.0), vec2(10.0, 10.0)),
            2.0,
            method_color(s.method),
        );
        let r = painter.text(
            at + vec2(14.0, 0.0),
            egui::Align2::LEFT_CENTER,
            s.method.name(),
            egui::FontId::proportional(11.0),
            text,
        );
        x = r.right() + 12.0;
    }
    if let Some(h) = hovered {
        response.on_hover_text(h);
    }
}

/// A sweep running on its own thread, with its progress counter.
type Job = (Arc<Progress>, JoinHandle<Result<Vec<Row>, String>>);

/// The test-seed sweep in a background thread, and the last results.
pub struct Runner {
    pub rows: Vec<Row>,
    pub status: String,
    job: Option<Job>,
}

impl Default for Runner {
    /// Starts with the results on disk, if any.
    fn default() -> Self {
        let rows = read_rows().unwrap_or_default();
        let status = if rows.is_empty() {
            "No results yet: press Run.".into()
        } else {
            format!("Loaded results/m2/embeddings.csv ({} rows).", rows.len())
        };
        Runner {
            rows,
            status,
            job: None,
        }
    }
}

impl Runner {
    pub fn running(&self) -> bool {
        self.job.is_some()
    }

    /// Starts the sweep over the test seeds with these settings.
    pub fn start(&mut self, config: SimConfig, dw: DeepWalkConfig, biases: Vec<(f64, f64)>) {
        if self.running() {
            return;
        }
        let progress = Arc::new(Progress::default());
        let p = Arc::clone(&progress);
        let handle = std::thread::spawn(move || sweep(TEST_SEEDS, &config, &dw, &biases, &p));
        self.job = Some((progress, handle));
        self.status = format!("Running on {} drones…", config.n_squads * config.squad_size);
    }

    pub fn cancel(&self) {
        if let Some((p, _)) = &self.job {
            p.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Share of the seeds done, while it runs.
    pub fn fraction(&self) -> Option<f32> {
        self.job
            .as_ref()
            .map(|(p, _)| p.done.load(Ordering::Relaxed) as f32 / TEST_SEEDS.count().max(1) as f32)
    }

    /// Collects a finished sweep: keeps the rows and writes the CSV.
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
            Ok(Ok(rows)) => {
                self.rows = rows;
                match write_rows(&self.rows) {
                    Ok(_) => format!("Done. Wrote {}.", rows_path().display()),
                    Err(e) => format!("Done, but could not write the CSV: {e}"),
                }
            }
            Ok(Err(e)) => format!("Stopped: {e}."),
            Err(_) => "The sweep crashed.".into(),
        };
    }

    /// Run / Stop buttons, the progress bar and the status line.
    pub fn buttons(
        &mut self,
        ui: &mut egui::Ui,
        config: &SimConfig,
        dw: &DeepWalkConfig,
        biases: &[(f64, f64)],
    ) {
        ui.horizontal(|ui| {
            if self.running() {
                if ui.button("Stop").clicked() {
                    self.cancel();
                }
            } else if ui
                .button("▶ Run the experiment")
                .on_hover_text(
                    "DeepWalk, Node2Vec at each (p, q) and the baselines on test seeds \
                     90–99 with the current swarm and walk settings, seeds in parallel. About \
                     10 seconds in release. Writes results/m2/embeddings.csv.",
                )
                .clicked()
            {
                self.start(*config, *dw, biases.to_vec());
            }
            if let Some(f) = self.fraction() {
                ui.add(
                    egui::ProgressBar::new(f)
                        .show_percentage()
                        .desired_width(160.0),
                );
            }
            ui.weak(&self.status);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_classic::embed::skipgram::SkipGramConfig;

    fn small() -> DeepWalkConfig {
        DeepWalkConfig {
            walks_per_node: 2,
            walk_length: 10,
            skipgram: SkipGramConfig {
                dim: 8,
                window: 3,
                ..Default::default()
            },
        }
    }

    #[test]
    fn leading_squad_is_the_majority_of_each_cluster() {
        // Cluster 0: squads 2, 2, 1 -> 2. Cluster 1: squads 0, 1 (tie) -> 0.
        assert_eq!(
            leading_squad(&[0, 0, 0, 1, 1], &[2, 2, 1, 0, 1]),
            vec![2, 0]
        );
    }

    #[test]
    fn share_beaten_counts_ties_half() {
        assert_eq!(share_beaten(2.0, &[1.0, 2.0, 3.0, 0.0]), 2.5 / 4.0);
        assert_eq!(share_beaten(9.0, &[1.0]), 1.0);
    }

    #[test]
    fn the_overlay_matches_the_embedding_and_waits_for_the_right_seed() {
        let config = SimConfig::default();
        let mut view = M2View {
            dw: small(),
            ..Default::default()
        };
        let ctx = egui::Context::default();
        view.update(&ctx, 1, &config, true);
        assert!(view.overlay().is_none(), "nothing until the job ends");
        while view.job.as_ref().is_some_and(|(_, h)| !h.is_finished()) {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        view.update(&ctx, 1, &config, true);
        let run = view.run().expect("seed 1 embedded");
        let e = &run.embeddings[0];
        let n = run.roles.len();
        let leaders = leading_squad(&e.clusters, &run.squads);
        let expected_wrong: Vec<bool> = (0..n)
            .map(|i| leaders[e.clusters[i]] != run.squads[i])
            .collect();
        let o = view.overlay().unwrap();
        assert_eq!(o.fill.len(), n);
        assert_eq!(o.hover.len(), n);
        assert_eq!(o.links.edges.len(), run.graph.edges.len());
        assert!(
            o.links
                .edges
                .iter()
                .all(|e| e.weight > 0.0 && e.weight <= 0.6 + 1e-12)
        );
        assert_eq!(o.wrong, expected_wrong.as_slice());
        assert!(o.marked_links.is_empty());

        // Role view: crosses are the wrong guesses from Z.
        view.show = Show::RoleFromZ;
        view.hidden_links = true;
        view.refresh_overlay();
        let run = view.run().unwrap();
        let e = &run.embeddings[0];
        let wrong: Vec<bool> = (0..n).map(|i| e.role_guess[i] != run.roles[i]).collect();
        let o = view.overlay().unwrap();
        assert_eq!(o.wrong, wrong.as_slice());
        assert_eq!(o.marked_links.len(), run.split.hidden.len());

        // The map draws this overlay (no reporters, no "no guess": empty slices) with
        // every layer on, and the drone tooltips, without panicking.
        let snap = swarmsim::snapshot::Snapshot::build(1, &config);
        let squads = crate::squads::squad_infos(&snap.swarm);
        let layers = crate::map::Layers {
            links: true,
            ids: true,
            ..Default::default()
        };
        let mut map_view = crate::map::MapView::default();
        let (mut selected, mut task) = (Some(0), None);
        for show in Show::ALL {
            view.show = show;
            view.refresh_overlay();
            let overlay = view.overlay();
            let mut out = egui::Context::default().run_ui(egui::RawInput::default(), |ui| {
                crate::map::draw(
                    ui,
                    &mut map_view,
                    &layers,
                    &config,
                    &snap.swarm,
                    &snap.geometric,
                    &squads,
                    None,
                    &mut selected,
                    &mut task,
                    overlay.as_ref(),
                );
            });
            out.textures_delta.clear(); // no window to upload the font texture to
        }

        // Another seed: the old embedding is not shown on the new swarm.
        view.update(&ctx, 2, &config, true);
        assert!(view.overlay().is_none());
    }
}
