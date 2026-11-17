//! AERIE viewer: an interactive window onto SwarmSim swarms.
//!
//! Run with `cargo run --release -p aerie-viz` (the default GUI)
//! or `cargo run --release -p aerie-viz --bin swarm_viewer`.
//! Left panel: mission, scenario and layers. Centre: the arena map
//! (scroll to zoom, drag to pan, click a drone or a task). Right panel: details.
//! Each seed is built in the dataset order with one RNG: spawn_swarm, place_jamming,
//! true_power, readings, live_graph, spawn_tasks.

use aerie_graph::EdgeList;
use eframe::egui::{self, RichText};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::jamming::{jammed_labels, place_jamming, readings, true_power};
use swarmsim::links::live_graph;
use swarmsim::spawn::spawn_swarm;
use swarmsim::tasks::{
    TaskKind, assignment_cost, cost_matrix, joint_winners, relay_within_two_hops, single_winners,
    spawn_tasks,
};

use aerie_viz::launch::{GUIS, launch};
use aerie_viz::map::{
    self, ColorBy, JAMMED_DRONE_COLOR, JOINT_COLOR, Layers, LinkColor, LinkSource, MapView,
    RELAY_FAIL_COLOR, RangeCircles, Snapshot, WINNER_COLOR, jam_field, n_near_tasks, task_color,
    task_letter, task_name,
};
use aerie_viz::mission::Mission;
use aerie_viz::squads::{SquadInfo, cell_count, cell_options, squad_infos};
use aerie_viz::style::{ramp, role_color, role_name, role_shape_name, squad_color, swatch};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE viewer")
            .with_inner_size([1450.0, 850.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE viewer",
        options,
        Box::new(|_cc| Ok(Box::new(Viewer::default()))),
    )
}

struct Viewer {
    mission: Mission,
    seed: u64,
    config: SimConfig,
    swarm: Vec<Drone>,
    /// The comm graph of `swarm` (epsilon graph, decision D00).
    graph: EdgeList,
    squads: Vec<SquadInfo>,
    /// The (seed, config) the current swarm was generated from.
    generated_for: Option<(u64, SimConfig)>,
    layers: Layers,
    view: MapView,
    /// Index into `swarm` of the selected drone.
    selected: Option<usize>,
    /// Jamming, live graph and tasks for this seed; `None` when place_jamming found no
    /// placement (possible with unusual configs).
    snapshot: Option<Snapshot>,
    /// Index into the snapshot's tasks of the selected task.
    selected_task: Option<usize>,
    /// The settings window (every SimConfig field) is open.
    show_settings: bool,
    /// Last message from the Playgrounds menu (which GUI was started).
    launch_msg: Option<String>,
}

impl Default for Viewer {
    fn default() -> Self {
        let mut viewer = Viewer {
            mission: Mission::M0,
            seed: 0,
            config: SimConfig::default(),
            swarm: Vec::new(),
            graph: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            squads: Vec::new(),
            generated_for: None,
            layers: Layers::default(),
            view: MapView::default(),
            selected: None,
            snapshot: None,
            selected_task: None,
            show_settings: false,
            launch_msg: None,
        };
        viewer.regenerate();
        viewer
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(290.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        self.settings_window(ui.ctx());
        self.regenerate();

        egui::Panel::right("details")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.details(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.central(ui));
    }
}

impl Viewer {
    /// Rebuilds the swarm if the seed or the config changed.
    fn regenerate(&mut self) {
        // Guards against configs that would make spawn_swarm loop or panic.
        let cells = cell_count(
            &self.config,
            self.config.cell_width,
            self.config.cell_height,
        );
        self.config.n_squads = self.config.n_squads.clamp(1, cells.max(1));
        self.config.squad_spread = self.config.squad_spread.max(0.05);
        // Values that would make the simulator panic (empty random ranges, division by 0).
        let c = &mut self.config;
        c.squad_size = c.squad_size.max(1);
        c.range_short = c.range_short.max(0.05);
        c.range_long = c.range_long.max(0.05);
        c.jam_d0 = c.jam_d0.max(0.05);
        c.noise_sd = c.noise_sd.max(0.0);
        c.noise_floor = c.noise_floor.max(1e-4);
        c.task_near_radius = c.task_near_radius.max(0.05);
        c.task_near_frac = c.task_near_frac.clamp(0.0, 1.0);
        for s in [
            &mut c.signal_scale_recon,
            &mut c.signal_scale_rf,
            &mut c.signal_scale_relay,
            &mut c.signal_scale_compute,
        ] {
            *s = s.max(0.01);
        }
        if c.jam_frac_min > c.jam_frac_max {
            std::mem::swap(&mut c.jam_frac_min, &mut c.jam_frac_max);
        }

        let key = (self.seed, self.config);
        if self.generated_for == Some(key) {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.swarm = spawn_swarm(&self.config, &mut rng);
        self.graph = epsilon_graph(&self.swarm, &self.config);
        self.squads = squad_infos(&self.swarm);
        self.snapshot = build_snapshot(&self.swarm, &self.graph, &self.config, &mut rng);
        self.generated_for = Some(key);
        if self.selected.is_some_and(|i| i >= self.swarm.len()) {
            self.selected = None;
        }
        let n_tasks = self.snapshot.as_ref().map_or(0, |s| s.tasks.len());
        if self.selected_task.is_some_and(|k| k >= n_tasks) {
            self.selected_task = None;
        }
    }

    /// Left panel: mission, scenario, layers, legend.
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("AERIE viewer");
        if ui
            .button("⚙ All settings…")
            .on_hover_text("Every SimConfig field in a window. Changes apply live.")
            .clicked()
        {
            self.show_settings = true;
        }
        ui.add_space(4.0);

        // Mission.
        ui.label(RichText::new("Mission").strong());
        egui::ComboBox::from_id_salt("mission")
            .selected_text(self.mission.label())
            .width(250.0)
            .show_ui(ui, |ui| {
                for m in Mission::ALL {
                    let text = if m.available() {
                        m.label().to_string()
                    } else {
                        format!("{}   (not built yet)", m.label())
                    };
                    ui.selectable_value(&mut self.mission, m, text);
                }
            });
        ui.weak(self.mission.description());
        ui.separator();

        // Scenario: seed and config.
        ui.label(RichText::new("Scenario").strong());
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut self.seed).range(0..=999_999));
            if ui
                .small_button("◀")
                .on_hover_text("Previous seed")
                .clicked()
            {
                self.seed = self.seed.saturating_sub(1);
            }
            if ui.small_button("▶").on_hover_text("Next seed").clicked() {
                self.seed += 1;
            }
        });
        ui.weak("Seeds 0-79 train, 80-89 validate, 90-99 test.");

        let cfg = &mut self.config;
        let cells = cell_count(cfg, cfg.cell_width, cfg.cell_height);

        egui::ComboBox::from_label("Cell size")
            .selected_text(format!("{} × {} km", cfg.cell_width, cfg.cell_height))
            .show_ui(ui, |ui| {
                for (w, h) in cell_options(cfg) {
                    let n = cell_count(cfg, w, h);
                    let text = format!("{w} × {h} km  ({n} cells)");
                    if ui
                        .selectable_label(cfg.cell_width == w && cfg.cell_height == h, text)
                        .clicked()
                    {
                        cfg.cell_width = w;
                        cfg.cell_height = h;
                    }
                }
            });
        ui.add(egui::Slider::new(&mut cfg.n_squads, 1..=cells.min(40)).text("squads"));
        ui.add(egui::Slider::new(&mut cfg.squad_size, 1..=30).text("drones per squad"));
        ui.add(
            egui::Slider::new(&mut cfg.squad_spread, 0.1..=6.0)
                .text("squad spread")
                .suffix(" km"),
        );
        ui.add(
            egui::Slider::new(&mut cfg.range_short, 0.5..=15.0)
                .text("short range")
                .suffix(" km"),
        );
        ui.add(
            egui::Slider::new(&mut cfg.range_long, 0.5..=25.0)
                .text("relay range")
                .suffix(" km"),
        );
        if ui.button("Reset config to defaults").clicked() {
            *cfg = SimConfig::default();
        }
        ui.separator();

        // Layers.
        ui.label(RichText::new("Layers").strong());
        egui::ComboBox::from_label("Colour by")
            .selected_text(self.layers.color_by.label())
            .show_ui(ui, |ui| {
                for c in ColorBy::ALL {
                    ui.selectable_value(&mut self.layers.color_by, c, c.label());
                }
            });
        egui::ComboBox::from_label("Radio range")
            .selected_text(self.layers.ranges.label())
            .show_ui(ui, |ui| {
                for r in RangeCircles::ALL {
                    ui.selectable_value(&mut self.layers.ranges, r, r.label());
                }
            });
        ui.checkbox(&mut self.layers.grid, "Cell grid");
        ui.checkbox(&mut self.layers.occupied_cells, "Tint occupied cells");
        ui.checkbox(&mut self.layers.centres, "Squad centres and labels");
        ui.checkbox(&mut self.layers.footprints, "Squad footprints");
        ui.checkbox(&mut self.layers.ids, "Drone IDs");
        ui.checkbox(
            &mut self.layers.dim_other_squads,
            "Dim other squads on select",
        );
        ui.checkbox(
            &mut self.layers.jamming,
            "Jammer, ridge, jammed zone (step 2)",
        )
        .on_hover_text("Lime: drones in the jammed zone. Dashed ring: d0, half power.");
        ui.add_enabled(
            self.layers.jamming,
            egui::Checkbox::new(&mut self.layers.zone_shade, "Shade the jammed zone"),
        )
        .on_hover_text(
            "Dark fill where the true power is at or above the threshold, darker towards P0. \
             The ridge shadow shows as a gap.",
        );
        ui.checkbox(&mut self.layers.links, "Links")
            .on_hover_text("Width = weight. Select a drone to highlight its links.");
        ui.add_enabled_ui(self.layers.links, |ui| {
            egui::ComboBox::from_label("Links shown")
                .selected_text(self.layers.link_source.label())
                .show_ui(ui, |ui| {
                    for s in LinkSource::ALL {
                        ui.selectable_value(&mut self.layers.link_source, s, s.label());
                    }
                });
            ui.add_enabled(
                self.layers.link_source == LinkSource::Live,
                egui::Checkbox::new(&mut self.layers.dead_links, "Dead links (red, dashed)"),
            );
            egui::ComboBox::from_label("Link colour")
                .selected_text(self.layers.link_color.label())
                .show_ui(ui, |ui| {
                    for c in LinkColor::ALL {
                        ui.selectable_value(&mut self.layers.link_color, c, c.label());
                    }
                });
        });
        ui.checkbox(&mut self.layers.tasks, "Tasks (step 3a)")
            .on_hover_text("Click a task to highlight the drones whose role can take it.");
        ui.horizontal(|ui| {
            if ui.button("Reset view").clicked() {
                self.view = MapView::default();
            }
            ui.weak("scroll: zoom · drag: pan");
        });
        ui.separator();

        // Legend.
        ui.label(RichText::new("Legend").strong());
        ui.weak("Shape = role");
        for role in Role::ALL {
            ui.horizontal(|ui| {
                swatch(ui, role, ui.visuals().text_color().gamma_multiply(0.5));
                ui.label(format!("{} ({})", role_name(role), role_shape_name(role)));
            });
        }
        ui.add_space(4.0);
        ui.weak(format!(
            "Colour = {}",
            self.layers.color_by.label().to_lowercase()
        ));
        match self.layers.color_by {
            ColorBy::Squad => {
                ui.horizontal_wrapped(|ui| {
                    for s in &self.squads {
                        swatch(ui, Role::Recon, squad_color(s.squad));
                        ui.label(format!("S{}", s.squad));
                    }
                });
            }
            ColorBy::Role => {
                for role in Role::ALL {
                    ui.horizontal(|ui| {
                        swatch(ui, role, role_color(role));
                        ui.label(role_name(role));
                    });
                }
            }
            ColorBy::Battery | ColorBy::SensorQuality | ColorBy::ComputeCapacity => {
                ui.horizontal(|ui| {
                    for v in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        swatch(ui, Role::Recon, ramp(v));
                        ui.label(format!("{v}"));
                    }
                });
            }
        }
        if self.layers.jamming {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                swatch(ui, Role::Recon, JAMMED_DRONE_COLOR);
                ui.label("In the jammed zone (lime, any squad)");
            });
        }
        if self.layers.tasks {
            ui.add_space(4.0);
            ui.weak("Tasks (diamonds): filled = near a drone, hollow = uniform");
            ui.horizontal_wrapped(|ui| {
                for kind in TaskKind::ALL {
                    ui.label(
                        RichText::new(format!("◆ {} {}", task_letter(kind), task_name(kind)))
                            .color(task_color(kind)),
                    );
                }
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("×").color(RELAY_FAIL_COLOR).strong());
                ui.label("Role fits, but no relay within 2 hops (Observe, Track)");
            });
        }
    }

    /// Centre: the map, with a banner for missions that have no view yet.
    fn central(&mut self, ui: &mut egui::Ui) {
        if !self.mission.available() {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{}: view not built yet.", self.mission.label()))
                        .strong(),
                );
                ui.weak(format!(
                    "It will show: {} Showing the M0 swarm meanwhile.",
                    self.mission.description()
                ));
            });
        }
        map::draw(
            ui,
            &mut self.view,
            &self.layers,
            &self.config,
            &self.swarm,
            &self.graph,
            &self.squads,
            self.snapshot.as_ref(),
            &mut self.selected,
            &mut self.selected_task,
        );
    }

    /// Top bar: a Playgrounds menu that opens each GUI in its own window, and the settings.
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Playgrounds", |ui| {
                ui.weak("Opens in a new window (built first if the code changed).");
                for gui in &GUIS {
                    if ui.button(gui.title).on_hover_text(gui.about).clicked() {
                        self.launch_msg = Some(launch(gui.bin));
                        ui.close();
                    }
                }
            });
            if ui.button("Settings").clicked() {
                self.show_settings = true;
            }
            if let Some(msg) = &self.launch_msg {
                ui.weak(msg);
            }
        });
    }

    /// The settings window: every `SimConfig` field, grouped. Each change rebuilds the seed
    /// live (spawn, jamming, live graph, tasks), in the dataset order.
    fn settings_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        egui::Window::new("Settings (SimConfig)")
            .open(&mut open)
            .resizable(true)
            .default_width(420.0)
            .vscroll(true)
            .show(ctx, |ui| self.settings(ui));
        self.show_settings = open;
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.weak("Changes apply live: the seed is rebuilt in the dataset order.");
        // Live result of the current config.
        match &self.snapshot {
            Some(s) => {
                let n = self.swarm.len().max(1);
                let zone = s.labels.iter().filter(|&&l| l).count();
                let live = stats(&s.live.graph, &self.swarm);
                let lost =
                    1.0 - s.live.graph.edges.len() as f64 / self.graph.edges.len().max(1) as f64;
                ui.label(
                    RichText::new(format!(
                        "{} drones · {:.0} % jammed · {:.0} % links lost · degree {:.2} · \
                         {} components · {} tasks",
                        self.swarm.len(),
                        100.0 * zone as f64 / n as f64,
                        100.0 * lost,
                        live.mean_degree,
                        live.n_components,
                        s.tasks.len()
                    ))
                    .monospace(),
                );
            }
            None => {
                ui.colored_label(
                    egui::Color32::from_rgb(255, 120, 80),
                    "place_jamming found no placement with these bounds: no jamming or tasks.",
                );
            }
        }
        ui.separator();

        let d = SimConfig::default();
        let cfg = &mut self.config;

        egui::CollapsingHeader::new("Swarm and arena")
            .default_open(true)
            .show(ui, |ui| {
                // The arena is a whole number of cells, so every cell size stays valid.
                let mut cols = (cfg.arena_w / cfg.cell_width).round().max(1.0) as usize;
                let mut rows = (cfg.arena_h / cfg.cell_height).round().max(1.0) as usize;
                ui.add(
                    egui::Slider::new(&mut cfg.cell_width, 1.0..=20.0)
                        .step_by(0.5)
                        .suffix(" km")
                        .text("cell width"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.cell_height, 1.0..=20.0)
                        .step_by(0.5)
                        .suffix(" km")
                        .text("cell height"),
                );
                ui.add(egui::Slider::new(&mut cols, 1..=20).text("cells across"));
                ui.add(egui::Slider::new(&mut rows, 1..=20).text("cells down"));
                cfg.arena_w = cols as f64 * cfg.cell_width;
                cfg.arena_h = rows as f64 * cfg.cell_height;
                ui.weak(format!(
                    "Arena {} × {} km, {} cells (one squad per cell at most).",
                    cfg.arena_w,
                    cfg.arena_h,
                    cols * rows
                ));
                ui.add(egui::Slider::new(&mut cfg.n_squads, 1..=(cols * rows)).text("squads"));
                ui.add(egui::Slider::new(&mut cfg.squad_size, 1..=30).text("drones per squad"));
                ui.add(
                    egui::Slider::new(&mut cfg.squad_spread, 0.1..=6.0)
                        .suffix(" km")
                        .text("squad spread"),
                );
                if ui.small_button("Reset group").clicked() {
                    cfg.arena_w = d.arena_w;
                    cfg.arena_h = d.arena_h;
                    cfg.cell_width = d.cell_width;
                    cfg.cell_height = d.cell_height;
                    cfg.n_squads = d.n_squads;
                    cfg.squad_size = d.squad_size;
                    cfg.squad_spread = d.squad_spread;
                }
            });

        egui::CollapsingHeader::new("Radio ranges (D01)")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut cfg.range_short, 0.5..=15.0)
                        .suffix(" km")
                        .text("short range (recon, RF, compute)"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.range_long, 0.5..=30.0)
                        .suffix(" km")
                        .text("relay range"),
                );
                ui.add(egui::Slider::new(&mut cfg.knn_k, 1..=20).text("kNN k"))
                    .on_hover_text("Used by knn_graph only; this viewer shows the ε-graph.");
                if ui.small_button("Reset group").clicked() {
                    cfg.range_short = d.range_short;
                    cfg.range_long = d.range_long;
                    cfg.knn_k = d.knn_k;
                }
            });

        egui::CollapsingHeader::new("Jamming (D02)")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut cfg.jam_p0, 0.1..=3.0).text("P0  jammer power"));
                ui.add(
                    egui::Slider::new(&mut cfg.jam_d0, 0.5..=10.0)
                        .suffix(" km")
                        .text("d0  half-power distance"),
                );
                ui.add(egui::Slider::new(&mut cfg.ridge_factor, 0.0..=1.0).text("ridge factor"));
                ui.add(
                    egui::Slider::new(&mut cfg.jam_threshold, 0.05..=1.0)
                        .text("zone threshold (× P0)"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.noise_sd, 0.0..=0.5).text("reading noise sd (× P0)"),
                );
                let before = (cfg.jam_frac_min, cfg.jam_frac_max);
                ui.add(
                    egui::Slider::new(&mut cfg.jam_frac_min, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .text("min jammed share"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.jam_frac_max, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .text("max jammed share"),
                );
                // Keep min <= max: whichever slider moved pushes the other one.
                if cfg.jam_frac_min > cfg.jam_frac_max {
                    if cfg.jam_frac_min != before.0 {
                        cfg.jam_frac_max = cfg.jam_frac_min;
                    } else {
                        cfg.jam_frac_min = cfg.jam_frac_max;
                    }
                }
                if ui.small_button("Reset group").clicked() {
                    cfg.jam_p0 = d.jam_p0;
                    cfg.jam_d0 = d.jam_d0;
                    cfg.ridge_factor = d.ridge_factor;
                    cfg.jam_threshold = d.jam_threshold;
                    cfg.noise_sd = d.noise_sd;
                    cfg.jam_frac_min = d.jam_frac_min;
                    cfg.jam_frac_max = d.jam_frac_max;
                }
            });

        egui::CollapsingHeader::new("Link model (D03)")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut cfg.noise_floor, 0.005..=5.0)
                        .logarithmic(true)
                        .text("N0  noise floor (× P0)"),
                )
                .on_hover_text("Lower N0 = FEWER links under jamming (tx power is tied to N0).");
                for (name, value) in [
                    ("recon scale", &mut cfg.signal_scale_recon),
                    ("RF sensing scale", &mut cfg.signal_scale_rf),
                    ("relay scale", &mut cfg.signal_scale_relay),
                    ("compute scale", &mut cfg.signal_scale_compute),
                ] {
                    ui.add(
                        egui::Slider::new(value, 0.05..=3.0)
                            .logarithmic(true)
                            .suffix(" km")
                            .text(name),
                    );
                }
                ui.weak("Smaller scale = louder transmitter = resists jamming.");
                if ui.small_button("Reset group").clicked() {
                    cfg.noise_floor = d.noise_floor;
                    cfg.signal_scale_recon = d.signal_scale_recon;
                    cfg.signal_scale_rf = d.signal_scale_rf;
                    cfg.signal_scale_relay = d.signal_scale_relay;
                    cfg.signal_scale_compute = d.signal_scale_compute;
                }
            });

        egui::CollapsingHeader::new("Tasks and costs (step 3)")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut cfg.n_tasks, 0..=200).text("tasks"));
                ui.add(
                    egui::Slider::new(&mut cfg.task_near_frac, 0.0..=1.0)
                        .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                        .text("placed near a drone"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.task_near_radius, 0.1..=10.0)
                        .suffix(" km")
                        .text("near radius"),
                );
                ui.add(
                    egui::Slider::new(&mut cfg.cost_beta, 0.0..=20.0)
                        .suffix(" km")
                        .text("β: cost of an empty battery"),
                )
                .on_hover_text("Cost = distance + β (1 − battery) + penalty.");
                ui.add(
                    egui::Slider::new(&mut cfg.cost_penalty, 10.0..=10000.0)
                        .logarithmic(true)
                        .text("penalty when not able"),
                )
                .on_hover_text("Wrong role, or no relay within 2 hops for Observe and Track.");
                if ui.small_button("Reset group").clicked() {
                    cfg.n_tasks = d.n_tasks;
                    cfg.task_near_frac = d.task_near_frac;
                    cfg.task_near_radius = d.task_near_radius;
                    cfg.cost_beta = d.cost_beta;
                    cfg.cost_penalty = d.cost_penalty;
                }
            });

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Reset everything to defaults").clicked() {
                *cfg = d;
            }
            ui.weak("The seed is kept.");
        });
    }

    /// Summary of the comm graph, from `swarmsim::graphs::stats::stats`.
    fn link_stats(&self, ui: &mut egui::Ui) {
        let st = stats(&self.graph, &self.swarm);
        egui::Grid::new("link_stats").striped(true).show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(k);
                ui.label(v);
                ui.end_row();
            };
            row("Links", st.n_edges.to_string());
            row(
                "Mean degree",
                format!("{:.2}  (target 8-10)", st.mean_degree),
            );
            row("Mean weight", format!("{:.3}", st.mean_weight));
            row(
                "Components",
                format!(
                    "{}  (largest {} of {})",
                    st.n_components, st.largest_component, st.n_nodes
                ),
            );
            row("Isolated drones", st.isolated.to_string());
            row(
                "Between squads",
                format!("{:.0} %  (target ~25 %)", 100.0 * st.cross_squad_share),
            );
        });
        if st.n_components > 1 {
            ui.weak("More than one component: some squads cannot reach each other.");
        }
    }

    /// Right panel: swarm summary, selected drone, squads table.
    fn details(&mut self, ui: &mut egui::Ui) {
        let cfg = &self.config;

        ui.label(RichText::new("Swarm").strong());
        ui.label(format!(
            "{} drones in {} squads · seed {}",
            self.swarm.len(),
            self.squads.len(),
            self.seed
        ));
        egui::Grid::new("role_counts").striped(true).show(ui, |ui| {
            for role in Role::ALL {
                swatch(ui, role, role_color(role));
                ui.label(role_name(role));
                ui.label(
                    self.swarm
                        .iter()
                        .filter(|d| d.role == role)
                        .count()
                        .to_string(),
                );
                ui.label(format!("{:.0} km", role.radio_range(cfg)));
                ui.end_row();
            }
        });
        ui.separator();

        // Comm graph.
        ui.label(RichText::new("Links (ε-graph)").strong());
        self.link_stats(ui);
        ui.separator();

        // Jamming and the live graph.
        ui.label(RichText::new("Jamming and live links (step 2)").strong());
        match &self.snapshot {
            None => {
                ui.weak(
                    "place_jamming found no placement for this config (5-30 % jammed). \
                     Try another seed or reset the config.",
                );
            }
            Some(s) => {
                let n = self.swarm.len().max(1);
                let zone = s.labels.iter().filter(|&&l| l).count();
                let live = stats(&s.live.graph, &self.swarm);
                let lost =
                    1.0 - s.live.graph.edges.len() as f64 / self.graph.edges.len().max(1) as f64;
                egui::Grid::new("jam_stats").striped(true).show(ui, |ui| {
                    let mut row = |k: &str, v: String| {
                        ui.label(k);
                        ui.label(v);
                        ui.end_row();
                    };
                    row(
                        "Jammer",
                        format!(
                            "({:.1}, {:.1}) km",
                            s.jammer.position.x, s.jammer.position.y
                        ),
                    );
                    row(
                        "In the jammed zone",
                        format!("{zone} drones ({:.0} %)", 100.0 * zone as f64 / n as f64),
                    );
                    row(
                        "Live links",
                        format!("{} ({:.0} % lost)", live.n_edges, 100.0 * lost),
                    );
                    row("Mean degree", format!("{:.2}", live.mean_degree));
                    row(
                        "Components",
                        format!(
                            "{}  (largest {} of {})",
                            live.n_components, live.largest_component, live.n_nodes
                        ),
                    );
                    row("Isolated drones", live.isolated.to_string());
                    row(
                        "One-way",
                        format!(
                            "{} cut by jamming · {} by range",
                            s.live.cut_one_way.len(),
                            s.live.range_one_way.len()
                        ),
                    );
                });
            }
        }
        ui.separator();

        // Tasks.
        ui.label(RichText::new("Tasks (step 3a)").strong());
        if let Some(s) = &self.snapshot {
            let n_near = n_near_tasks(cfg);
            egui::Grid::new("task_counts").striped(true).show(ui, |ui| {
                ui.label("");
                ui.label("Tasks");
                ui.label("Near");
                ui.label("By role");
                ui.label("Able");
                ui.end_row();
                for kind in TaskKind::ALL {
                    let of_kind: Vec<usize> = (0..s.tasks.len())
                        .filter(|&k| s.tasks[k].kind == kind)
                        .collect();
                    ui.label(RichText::new(task_name(kind)).color(task_color(kind)));
                    ui.label(of_kind.len().to_string());
                    ui.label(of_kind.iter().filter(|&&k| k < n_near).count().to_string());
                    let by_role: Vec<usize> = (0..self.swarm.len())
                        .filter(|&i| kind.can_do(self.swarm[i].role))
                        .collect();
                    let able = by_role
                        .iter()
                        .filter(|&&i| !kind.needs_relay() || s.relay_ok[i])
                        .count();
                    ui.label(by_role.len().to_string());
                    ui.label(able.to_string());
                    ui.end_row();
                }
            });
            ui.weak(
                "By role: the strict mapping (3a). Able: also passes the relay rule (3b), \
                 a relay within 2 hops on the live graph, for Observe and Track.",
            );
            let ok = s.relay_ok.iter().filter(|&&b| b).count();
            ui.label(format!(
                "Relay within 2 hops: {ok} of {} drones",
                s.relay_ok.len()
            ));
            // The two oracle labels over all tasks (step 3d).
            let open = s.joint.iter().filter(|w| w.is_none()).count();
            let differ = (0..s.tasks.len())
                .filter(|&k| s.joint[k].is_some() && s.joint[k] != Some(s.single[k]))
                .count();
            // Over the staffed tasks only, so both totals cover the same tasks.
            let singles: f64 = (0..s.tasks.len())
                .filter(|&k| s.joint[k].is_some())
                .map(|k| s.costs.get(k, s.single[k]))
                .sum();
            ui.label(format!(
                "Oracle: joint differs from single on {differ} of {} tasks, {open} open",
                s.tasks.len()
            ));
            ui.weak(format!(
                "Total cost: singles {singles:.1} km (may share drones), joint {:.1} km",
                assignment_cost(&s.costs, &s.joint)
            ));
            if let Some(task) = self.selected_task.and_then(|k| s.tasks.get(k)) {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!(
                        "Selected: task {} · {}",
                        task.id.0,
                        task_name(task.kind)
                    ))
                    .color(task_color(task.kind))
                    .strong(),
                );
                ui.label(format!(
                    "({:.2}, {:.2}) km · {}",
                    task.position.x,
                    task.position.y,
                    if (task.id.0 as usize) < n_near {
                        "near a drone"
                    } else {
                        "uniform"
                    }
                ));
                // Row of the cost matrix for this task, cheapest first (step 3c).
                let k = self.selected_task.unwrap_or(0);
                let mut order: Vec<usize> = (0..self.swarm.len()).collect();
                order.sort_by(|&a, &b| s.costs.get(k, a).total_cmp(&s.costs.get(k, b)));
                ui.weak(
                    "Cheapest drones (cost = km + β·(1 − battery), or the penalty if not able):",
                );
                for (rank, &i) in order.iter().take(5).enumerate() {
                    let d = &self.swarm[i];
                    let cost = s.costs.get(k, i);
                    let dist = d.position.dist(task.position);
                    let tired = cfg.cost_beta * (1.0 - d.battery);
                    let able = task.kind.able(d.role, s.relay_ok[i]);
                    let why = if able {
                        format!("{dist:.2} km + {tired:.2}")
                    } else if !task.kind.can_do(d.role) {
                        "penalty (role)".to_string()
                    } else {
                        "penalty (no relay)".to_string()
                    };
                    let text = format!(
                        "{}drone {} · {} · S{} · cost {cost:.2} = {why}",
                        if rank == 0 { "BEST  " } else { "" },
                        d.id.0,
                        role_name(d.role),
                        d.squad.0,
                    );
                    let mut text = RichText::new(text);
                    if !able {
                        text = text.color(RELAY_FAIL_COLOR);
                    } else if rank == 0 {
                        text = text.color(WINNER_COLOR).strong();
                    }
                    if ui.selectable_label(false, text).clicked() {
                        self.selected = Some(i);
                    }
                }
                ui.weak(
                    "BEST = cheapest drone, shown on the map with a gold star and a dashed \
                     line from the task. It is the single-winner label (step 3d).",
                );
                ui.add_space(2.0);
                let single = s.single[k];
                ui.label(
                    RichText::new(format!(
                        "Single winner: drone {} · cost {:.2}",
                        self.swarm[single].id.0,
                        s.costs.get(k, single)
                    ))
                    .color(WINNER_COLOR),
                );
                match s.joint[k] {
                    Some(j) if j == single => {
                        ui.label(RichText::new("Joint winner: the same drone").color(JOINT_COLOR));
                    }
                    Some(j) => {
                        let other =
                            (0..s.tasks.len()).find(|&t| t != k && s.joint[t] == Some(single));
                        ui.label(
                            RichText::new(format!(
                                "Joint winner: drone {} · cost {:.2} (cyan star)",
                                self.swarm[j].id.0,
                                s.costs.get(k, j)
                            ))
                            .color(JOINT_COLOR),
                        );
                        if let Some(o) = other {
                            ui.weak(format!(
                                "Drone {} went to task {} instead.",
                                self.swarm[single].id.0, s.tasks[o].id.0
                            ));
                        }
                    }
                    None => {
                        ui.label(
                            RichText::new(
                                "Joint winner: none, the task stays open (no able drone left)",
                            )
                            .color(JOINT_COLOR),
                        );
                    }
                }
                if ui.button("Deselect task").clicked() {
                    self.selected_task = None;
                }
            } else {
                ui.weak("Click a task on the map.");
            }
        } else {
            ui.weak("No snapshot for this config.");
        }
        ui.separator();

        // Selected drone.
        ui.label(RichText::new("Selected drone").strong());
        match self.selected.and_then(|i| self.swarm.get(i)) {
            None => {
                ui.weak("Click a drone on the map.");
            }
            Some(d) => {
                let squad = self.squads.iter().find(|s| s.squad == d.squad.0 as usize);
                egui::Grid::new("selected").striped(true).show(ui, |ui| {
                    let mut row = |k: &str, v: String| {
                        ui.label(k);
                        ui.label(v);
                        ui.end_row();
                    };
                    row("ID", d.id.0.to_string());
                    row("Squad", format!("S{}", d.squad.0));
                    row("Role", role_name(d.role).to_string());
                    row(
                        "Position",
                        format!("({:.2}, {:.2}) km", d.position.x, d.position.y),
                    );
                    row(
                        "Velocity",
                        format!("({:.2}, {:.2}) km/step", d.velocity.x, d.velocity.y),
                    );
                    row("Radio range", format!("{:.1} km", d.role.radio_range(cfg)));
                    let i = d.id.0 as usize;
                    let mine: Vec<_> = self
                        .graph
                        .edges
                        .iter()
                        .filter(|e| e.src as usize == i || e.dst as usize == i)
                        .collect();
                    let cross = mine
                        .iter()
                        .filter(|e| {
                            self.swarm[e.src as usize].squad != self.swarm[e.dst as usize].squad
                        })
                        .count();
                    row("Links", format!("{} ({cross} to other squads)", mine.len()));
                    row(
                        "Weighted degree",
                        format!("{:.2}", mine.iter().map(|e| e.weight).sum::<f64>()),
                    );
                    if let Some(s) = &self.snapshot {
                        let ok = s.relay_ok.get(i) == Some(&true);
                        row(
                            "Relay ≤ 2 hops",
                            if ok {
                                "yes"
                            } else {
                                "no (can't Observe or Track)"
                            }
                            .to_string(),
                        );
                    }
                    row("Battery", format!("{:.2}", d.battery));
                    row("Sensor quality", format!("{:.2}", d.sensor_quality));
                    row("Compute capacity", format!("{:.2}", d.compute_capacity));
                    if let Some(s) = squad {
                        let dx = d.position.x - s.centroid.0;
                        let dy = d.position.y - s.centroid.1;
                        row("From squad centre", format!("{:.2} km", dx.hypot(dy)));
                    }
                });
                if ui.button("Deselect").clicked() {
                    self.selected = None;
                }
            }
        }
        ui.separator();

        // Squads.
        ui.label(RichText::new("Squads").strong());
        ui.weak("Footprint: centre to furthest member. Nearest: closest other squad centre.");
        egui::Grid::new("squads").striped(true).show(ui, |ui| {
            ui.label("");
            ui.label("Cell");
            ui.label("Footprint");
            ui.label("Nearest");
            ui.label("R/S/L/C");
            ui.end_row();
            for s in &self.squads {
                let text = RichText::new(format!("S{}", s.squad)).color(squad_color(s.squad));
                if ui
                    .selectable_label(false, text)
                    .on_hover_text("Select its first drone")
                    .clicked()
                {
                    self.selected = s.members.first().copied();
                }
                let col = (s.centroid.0 / cfg.cell_width).floor() as i64;
                let row = (s.centroid.1 / cfg.cell_height).floor() as i64;
                ui.label(format!("({col}, {row})"));
                ui.label(format!("{:.2} km", s.footprint));
                ui.label(
                    s.nearest_other
                        .map_or("-".to_string(), |d| format!("{d:.1} km")),
                );
                let [r, f, l, c] = s.role_counts;
                ui.label(format!("{r}/{f}/{l}/{c}"));
                ui.end_row();
            }
        });
        ui.weak("R/S/L/C = recon / RF sensing / relay / compute.");
    }
}

/// Step 2 and 3a for one seed, continuing the RNG after `spawn_swarm` in the dataset
/// order. `place_jamming` panics when no placement fits the bounds (possible with the
/// sliders); that becomes `None` here, with the panic message silenced.
fn build_snapshot(
    swarm: &[Drone],
    graph: &EdgeList,
    cfg: &SimConfig,
    rng: &mut ChaCha8Rng,
) -> Option<Snapshot> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let placed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        place_jamming(swarm, cfg, rng)
    }));
    std::panic::set_hook(previous);
    let (jammer, ridge) = placed.ok()?;
    let truth = true_power(swarm, &jammer, &ridge, cfg);
    let reads = readings(&truth, cfg, rng);
    let labels = jammed_labels(&truth, cfg);
    let live = live_graph(graph, swarm, &truth, cfg);
    let tasks = spawn_tasks(swarm, cfg, rng);
    let relay_ok = relay_within_two_hops(&live.graph, swarm);
    let costs = cost_matrix(swarm, &tasks, &relay_ok, cfg);
    let single = single_winners(&costs);
    let joint = joint_winners(&costs, cfg);
    let field = jam_field(&jammer, &ridge, cfg);
    Some(Snapshot {
        jammer,
        ridge,
        truth,
        readings: reads,
        labels,
        live,
        tasks,
        relay_ok,
        costs,
        single,
        joint,
        field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_follows_the_dataset_order() {
        let v = Viewer::default();
        let s = v.snapshot.as_ref().expect("seed 0 places jamming");
        let cfg = SimConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let swarm = spawn_swarm(&cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, &cfg);
        let reads = readings(&truth, &cfg, &mut rng);
        let tasks = spawn_tasks(&swarm, &cfg, &mut rng);
        assert_eq!(s.jammer, jammer);
        assert_eq!(s.truth, truth);
        assert_eq!(s.readings, reads);
        assert_eq!(s.tasks, tasks);
        assert_eq!(s.live, live_graph(&v.graph, &swarm, &truth, &cfg));
        assert_eq!(s.labels, jammed_labels(&truth, &cfg));
        assert_eq!(s.relay_ok, relay_within_two_hops(&s.live.graph, &swarm));
        assert_eq!(s.costs, cost_matrix(&swarm, &tasks, &s.relay_ok, &cfg));
        assert_eq!(s.single, single_winners(&s.costs));
        assert_eq!(s.joint, joint_winners(&s.costs, &cfg));
    }

    #[test]
    fn gold_star_goes_to_the_cheapest_able_drone() {
        let v = Viewer::default();
        let s = v.snapshot.as_ref().unwrap();
        for (k, task) in s.tasks.iter().enumerate() {
            let w = map::cheapest_drone(s, k).unwrap();
            let row = s.costs.row(k);
            assert!(row.iter().all(|&c| row[w] <= c), "task {k}");
            assert!(task.kind.able(v.swarm[w].role, s.relay_ok[w]), "task {k}");
        }
        assert_eq!(map::cheapest_drone(s, s.tasks.len()), None);
    }

    #[test]
    fn shading_field_covers_the_arena_and_matches_jam_power() {
        let v = Viewer::default();
        let s = v.snapshot.as_ref().unwrap();
        let (w, h) = map::field_dims(&v.config);
        assert_eq!(s.field.len(), w * h);
        // The cell under the jammer is in the zone (power close to P0).
        let c = (s.jammer.position.x / map::FIELD_RES) as usize;
        let r = (s.jammer.position.y / map::FIELD_RES) as usize;
        assert!(s.field[r * w + c] >= v.config.jam_threshold * v.config.jam_p0);
        // Every value is the real jam_power at the cell centre.
        let at = swarmsim::drone::Vec2 {
            x: (c as f64 + 0.5) * map::FIELD_RES,
            y: (r as f64 + 0.5) * map::FIELD_RES,
        };
        let expected = swarmsim::jamming::jam_power(&s.jammer, &s.ridge, at, &v.config);
        assert_eq!(s.field[r * w + c], expected);
    }

    #[test]
    fn settings_that_would_panic_are_guarded() {
        // A zero task radius (empty random range), swapped share bounds, zero scales.
        let mut v = Viewer::default();
        v.config.task_near_radius = 0.0;
        v.config.jam_frac_min = 0.3;
        v.config.jam_frac_max = 0.05;
        v.config.signal_scale_relay = 0.0;
        v.config.squad_size = 0;
        v.regenerate();
        assert!(v.config.task_near_radius > 0.0);
        assert!(v.config.jam_frac_min <= v.config.jam_frac_max);
        assert!(v.config.signal_scale_relay > 0.0);
        assert!(v.snapshot.is_some());
        // A different arena (8 x 3 cells of 5 km) still spawns and places jamming.
        v.config = SimConfig {
            arena_w: 40.0,
            arena_h: 15.0,
            n_squads: 6,
            ..SimConfig::default()
        };
        v.regenerate();
        assert_eq!(v.swarm.len(), 60);
        assert!(v.snapshot.is_some());
    }

    #[test]
    fn impossible_jamming_bounds_give_no_snapshot_instead_of_a_crash() {
        let mut v = Viewer::default();
        v.config.jam_frac_min = 1.0;
        v.config.jam_frac_max = 1.0;
        v.selected_task = Some(3);
        v.regenerate();
        assert!(v.snapshot.is_none());
        assert_eq!(v.selected_task, None);
        assert_eq!(v.swarm.len(), 100); // the swarm itself is still shown
    }

    #[test]
    fn viewer_starts_and_regenerates_on_change() {
        let mut v = Viewer::default();
        assert_eq!(v.swarm.len(), 100);
        let first = v.swarm.clone();
        v.seed = 1;
        v.regenerate();
        assert_ne!(v.swarm, first);
        // The comm graph follows the swarm.
        assert_eq!(v.graph.n as usize, v.swarm.len());
        assert!(!v.graph.edges.is_empty());
        // An impossible squad count is clamped instead of hanging spawn_swarm.
        v.config.n_squads = 1000;
        v.regenerate();
        assert!(
            v.config.n_squads <= cell_count(&v.config, v.config.cell_width, v.config.cell_height)
        );
    }
}
