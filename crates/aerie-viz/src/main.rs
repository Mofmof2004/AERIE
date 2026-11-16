//! AERIE viewer: an interactive window onto SwarmSim swarms.
//!
//! Run with `cargo run --release -p aerie-viz`.
//! Left panel: mission, scenario and layers. Centre: the arena map
//! (scroll to zoom, drag to pan, click a drone). Right panel: details.

mod map;
mod mission;
mod style;

use std::collections::BTreeMap;

use eframe::egui::{self, RichText};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Role};
use swarmsim::spawn::spawn_swarm;

use map::{ColorBy, Layers, MapView, RangeCircles};
use mission::Mission;
use style::{ramp, role_color, role_name, role_shape_name, squad_color, swatch};

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

/// Summary of one squad, recomputed whenever the swarm changes.
pub struct SquadInfo {
    pub squad: usize,
    /// Indices into the swarm vector.
    pub members: Vec<usize>,
    /// Mean position of the members, km.
    pub centroid: (f64, f64),
    /// Distance from the centroid to the furthest member, km.
    pub footprint: f64,
    /// Distance to the closest other squad centroid, km.
    pub nearest_other: Option<f64>,
    /// Members per role, in the order of `Role::ALL`.
    pub role_counts: [usize; 4],
}

struct Viewer {
    mission: Mission,
    seed: u64,
    config: SimConfig,
    swarm: Vec<Drone>,
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
        let mut no_links = false;
        ui.add_enabled(false, egui::Checkbox::new(&mut no_links, "Links (ε-graph)"))
            .on_disabled_hover_text("Appears once epsilon_graph exists in swarmsim.");
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
            &self.squads,
            &mut self.selected,
        );
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
        let links = "Links: not built yet (epsilon_graph).";
        ui.weak(links);
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

/// Number of cells in the grid for a cell size.
fn cell_count(cfg: &SimConfig, w: f64, h: f64) -> usize {
    ((cfg.arena_w / w).round() as usize) * ((cfg.arena_h / h).round() as usize)
}

/// Cell sizes that divide the arena exactly and leave room for one squad each.
fn cell_options(cfg: &SimConfig) -> Vec<(f64, f64)> {
    let divides = |part: f64, whole: f64| {
        let r = whole / part;
        (r - r.round()).abs() < 1e-9
    };
    let mut options = Vec::new();
    for w in [2.0, 4.0, 5.0, 8.0, 10.0, 20.0, 40.0] {
        for h in [2.0, 4.0, 5.0, 10.0, 20.0] {
            if divides(w, cfg.arena_w) && divides(h, cfg.arena_h) {
                options.push((w, h));
            }
        }
    }
    options
}

/// Groups the swarm by squad and computes each squad's summary.
fn squad_infos(swarm: &[Drone]) -> Vec<SquadInfo> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, d) in swarm.iter().enumerate() {
        groups.entry(d.squad.0 as usize).or_default().push(i);
    }

    let mut infos: Vec<SquadInfo> = groups
        .into_iter()
        .map(|(squad, members)| {
            let n = members.len() as f64;
            let cx = members.iter().map(|&i| swarm[i].position.x).sum::<f64>() / n;
            let cy = members.iter().map(|&i| swarm[i].position.y).sum::<f64>() / n;
            let footprint = members
                .iter()
                .map(|&i| (swarm[i].position.x - cx).hypot(swarm[i].position.y - cy))
                .fold(0.0, f64::max);
            let mut role_counts = [0; 4];
            for &i in &members {
                let k = Role::ALL
                    .iter()
                    .position(|&r| r == swarm[i].role)
                    .unwrap_or(0);
                role_counts[k] += 1;
            }
            SquadInfo {
                squad,
                members,
                centroid: (cx, cy),
                footprint,
                nearest_other: None,
                role_counts,
            }
        })
        .collect();

    let centroids: Vec<(f64, f64)> = infos.iter().map(|s| s.centroid).collect();
    for (k, s) in infos.iter_mut().enumerate() {
        s.nearest_other = centroids
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != k)
            .map(|(_, c)| (c.0 - s.centroid.0).hypot(c.1 - s.centroid.1))
            .min_by(f64::total_cmp);
    }
    infos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_options_all_divide_the_arena() {
        let cfg = SimConfig::default();
        let options = cell_options(&cfg);
        assert!(options.contains(&(cfg.cell_width, cfg.cell_height)));
        for (w, h) in options {
            assert_eq!((cfg.arena_w / w).fract(), 0.0);
            assert_eq!((cfg.arena_h / h).fract(), 0.0);
        }
    }

    #[test]
    fn squad_infos_cover_every_drone_once() {
        let cfg = SimConfig::default();
        let swarm = spawn_swarm(&cfg, &mut ChaCha8Rng::seed_from_u64(0));
        let infos = squad_infos(&swarm);
        assert_eq!(infos.len(), cfg.n_squads);
        let total: usize = infos.iter().map(|s| s.members.len()).sum();
        assert_eq!(total, swarm.len());
        for s in &infos {
            assert_eq!(s.role_counts.iter().sum::<usize>(), s.members.len());
            assert!(s.nearest_other.is_some());
        }
    }

    #[test]
    fn viewer_starts_and_regenerates_on_change() {
        let mut v = Viewer::default();
        assert_eq!(v.swarm.len(), 100);
        let first = v.swarm.clone();
        v.seed = 1;
        v.regenerate();
        assert_ne!(v.swarm, first);
        // An impossible squad count is clamped instead of hanging spawn_swarm.
        v.config.n_squads = 1000;
        v.regenerate();
        assert!(
            v.config.n_squads <= cell_count(&v.config, v.config.cell_width, v.config.cell_height)
        );
    }
}
