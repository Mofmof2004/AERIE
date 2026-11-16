//! AERIE viewer: an interactive window onto SwarmSim swarms.
//!
//! Run with `cargo run --release -p aerie-viz` (the default GUI)
//! or `cargo run --release -p aerie-viz --bin swarm_viewer`.
//! Left panel: mission, scenario and layers. Centre: the arena map
//! (scroll to zoom, drag to pan, click a drone). Right panel: details.

use aerie_graph::EdgeList;
use eframe::egui::{self, RichText};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::spawn::spawn_swarm;

use aerie_viz::map::{self, ColorBy, Layers, LinkColor, MapView, RangeCircles};
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
        };
        viewer.regenerate();
        viewer
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(290.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

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

        let key = (self.seed, self.config);
        if self.generated_for == Some(key) {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.swarm = spawn_swarm(&self.config, &mut rng);
        self.graph = epsilon_graph(&self.swarm, &self.config);
        self.squads = squad_infos(&self.swarm);
        self.generated_for = Some(key);
        if self.selected.is_some_and(|i| i >= self.swarm.len()) {
            self.selected = None;
        }
    }

    /// Left panel: mission, scenario, layers, legend.
    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("AERIE viewer");
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
        ui.checkbox(&mut self.layers.links, "Links (ε-graph)")
            .on_hover_text("Width = weight. Select a drone to highlight its links.");
        ui.add_enabled_ui(self.layers.links, |ui| {
            egui::ComboBox::from_label("Link colour")
                .selected_text(self.layers.link_color.label())
                .show_ui(ui, |ui| {
                    for c in LinkColor::ALL {
                        ui.selectable_value(&mut self.layers.link_color, c, c.label());
                    }
                });
        });
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
            &mut self.selected,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

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
