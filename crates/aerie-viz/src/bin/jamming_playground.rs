//! Jamming playground: place jammers and a ridge, see the jamming power everywhere.
//!
//! Run with `cargo run --release -p aerie-viz --bin jamming_playground`.
//!
//! The heatmap calls the real `swarmsim::jamming::jam_power` at every pixel.
//! Tools: Move (drag jammers, ridge ends or the whole ridge; drag empty space to pan),
//! Add jammer (click), Draw ridge (drag from one end to the other).
//! Right-click a jammer or the ridge to delete it. Scroll to zoom.
//! Optional swarm layer: which drones are jammed and which links fail (previews 2c / 2d).
//! "Place from seed" runs `place_jamming` on the chosen seed's swarm, exactly as the
//! dataset will: spawn the swarm, then place jammer and ridge with the same RNG.
//! Readings (2c): each drone's true power comes from your `true_power`, its sensor
//! reading from your `readings`, its label from your `jammed_labels`. The "after"
//! graph in the right panel comes from your `fail_links`. Drones can be coloured by label against reading, and
//! a scatter plot shows truth against reading for every drone.

use aerie_graph::EdgeList;
use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Key, Pos2, Rect, RichText, Sense, Shape, Stroke,
    TextureHandle, TextureOptions, Vec2 as EVec2, pos2, vec2,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use swarmsim::config::SimConfig;
use swarmsim::drone::{Drone, Vec2};
use swarmsim::graphs::epsilon::epsilon_graph;
use swarmsim::graphs::stats::stats;
use swarmsim::jamming::{
    Jammer, Ridge, fail_links, jam_power, jammed_labels, place_jamming, readings, true_power,
};
use swarmsim::spawn::spawn_swarm;

use aerie_viz::style::{draw_drone, role_color, role_name};

/// Heatmap resolution, km per pixel.
const RES: f64 = 0.2;
/// Grab distances, pixels.
const GRAB_JAMMER: f32 = 12.0;
const GRAB_END: f32 = 10.0;
const GRAB_RIDGE: f32 = 6.0;

const RIDGE_COLOR: Color32 = Color32::from_rgb(170, 120, 70);
const JAMMER_COLOR: Color32 = Color32::from_rgb(230, 60, 60);
const BLOCKED_COLOR: Color32 = Color32::from_rgb(120, 180, 255);

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AERIE jamming playground")
            .with_inner_size([1500.0, 880.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AERIE jamming playground",
        options,
        Box::new(|_cc| Ok(Box::new(JamApp::default()))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Move,
    AddJammer,
    DrawRidge,
}

/// How the power of several jammers adds up at one point. Not decided yet (D02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Combine {
    Sum,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    Linear,
    Log,
}

/// How the swarm's drones are coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DroneColor {
    Role,
    Truth,
    Reading,
    Check,
}

/// A drone's label (from the truth) against what its reading says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Check {
    /// Jammed, and the reading is above the threshold too.
    Hit,
    /// Jammed, but the reading is below the threshold.
    Missed,
    /// Not jammed, but the reading is above the threshold.
    FalseAlarm,
    /// Not jammed, and the reading is below the threshold too.
    Clear,
}

impl Check {
    const ALL: [Check; 4] = [Check::Hit, Check::Missed, Check::FalseAlarm, Check::Clear];

    fn name(self) -> &'static str {
        match self {
            Check::Hit => "jammed, reading agrees",
            Check::Missed => "jammed, reading below threshold",
            Check::FalseAlarm => "not jammed, reading above threshold",
            Check::Clear => "not jammed, reading agrees",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Check::Hit => Color32::from_rgb(255, 70, 70),
            Check::Missed => Color32::from_rgb(255, 170, 0),
            Check::FalseAlarm => Color32::from_rgb(220, 90, 255),
            Check::Clear => Color32::from_rgb(160, 160, 160),
        }
    }
}

/// Sorts a drone by its label (truth at or above the threshold) and its reading.
fn check(label: bool, reading: f64, threshold: f64) -> Check {
    match (label, reading >= threshold) {
        (true, true) => Check::Hit,
        (true, false) => Check::Missed,
        (false, true) => Check::FalseAlarm,
        (false, false) => Check::Clear,
    }
}

/// What the current drag is doing.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Drag {
    Jammer(usize),
    RidgeA,
    RidgeB,
    RidgeBody,
    NewRidge(Vec2),
    Pan,
}

/// Everything the heatmap depends on. The texture is rebuilt only when this changes.
#[derive(Debug, Clone, PartialEq)]
struct HeatKey {
    jammers: Vec<Jammer>,
    ridge: Option<Ridge>,
    d0: f64,
    factor: f64,
    threshold: f64,
    p0: f64,
    combine: Combine,
    scale: Scale,
    contours: bool,
}

struct JamApp {
    config: SimConfig,
    jammers: Vec<Jammer>,
    ridge: Option<Ridge>,
    tool: Tool,
    combine: Combine,
    scale: Scale,
    snap: bool,
    selected: Option<usize>,
    drag: Option<Drag>,
    /// Pointer position in km, from the last frame (the side panel shows it).
    probe: Option<Vec2>,

    // View.
    zoom: f32,
    pan: EVec2,

    // Display.
    contours: bool,
    d0_rings: bool,
    probe_lines: bool,

    // Swarm layer.
    show_swarm: bool,
    show_failed: bool,
    seed: u64,
    swarm: Vec<Drone>,
    graph: EdgeList,
    swarm_seed: Option<u64>,
    /// Re-run `place_jamming` whenever the seed changes.
    auto_place: bool,
    /// The seed the last `place_jamming` run was for (to warn when the seed moved on).
    placed_for: Option<u64>,
    /// Message from the last placement (success or failure).
    place_msg: Option<String>,

    // Readings (2c).
    /// The seed's RNG right after spawn_swarm (and after place_jamming when placed):
    /// where the dataset's readings draw their noise from.
    rng_after: ChaCha8Rng,
    /// 0 = the dataset's noise; 1, 2, ... = other noise draws for the same truth.
    noise_draw: u64,
    drone_color: DroneColor,
    show_heat: bool,
    /// Drone under the cursor in the truth / reading plot (highlighted on the map).
    plot_hover: Option<usize>,

    // Heatmap texture cache.
    heat: Option<TextureHandle>,
    heat_key: Option<HeatKey>,
}

impl Default for JamApp {
    fn default() -> Self {
        let mut app = JamApp {
            config: SimConfig::default(),
            jammers: Vec::new(),
            ridge: None,
            tool: Tool::Move,
            combine: Combine::Sum,
            scale: Scale::Linear,
            snap: true,
            selected: None,
            drag: None,
            probe: None,
            zoom: 1.0,
            pan: EVec2::ZERO,
            contours: true,
            d0_rings: true,
            probe_lines: true,
            show_swarm: false,
            show_failed: true,
            seed: 0,
            swarm: Vec::new(),
            graph: EdgeList {
                n: 0,
                edges: Vec::new(),
            },
            swarm_seed: None,
            auto_place: false,
            placed_for: None,
            place_msg: None,
            rng_after: ChaCha8Rng::seed_from_u64(0),
            noise_draw: 0,
            drone_color: DroneColor::Check,
            show_heat: true,
            plot_hover: None,
            heat: None,
            heat_key: None,
        };
        app.load_test_setup();
        app
    }
}

impl eframe::App for JamApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
            });

        self.update_swarm();

        egui::Panel::right("details")
            .resizable(true)
            .default_size(380.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.details(ui));
            });

        egui::CentralPanel::default().show(ui, |ui| self.canvas(ui));
    }
}

impl JamApp {
    // ---------------------------------------------------------------- model

    /// The ridge to pass to `jam_power`. With no ridge, a zero-length one far away:
    /// `orient` is then always 0, so nothing is ever shadowed.
    fn ridge_or_none(&self) -> Ridge {
        self.ridge.unwrap_or(Ridge {
            a: Vec2 { x: -1e6, y: -1e6 },
            b: Vec2 { x: -1e6, y: -1e6 },
        })
    }

    /// Power of one jammer at `at`, from your `jam_power`.
    fn power_of(&self, j: &Jammer, at: Vec2) -> f64 {
        jam_power(j, &self.ridge_or_none(), at, &self.config)
    }

    /// Power of one jammer at `at` if there were no ridge (open ground).
    fn open_power(&self, j: &Jammer, at: Vec2) -> f64 {
        let r = j.position.dist(at) / self.config.jam_d0;
        j.power / (1.0 + r * r)
    }

    /// True when the ridge cuts the line from `j` to `at`.
    fn blocked(&self, j: &Jammer, at: Vec2) -> bool {
        self.ridge.is_some() && self.power_of(j, at) < self.open_power(j, at) * (1.0 - 1e-9)
    }

    /// Total jamming power at `at`, combining every jammer.
    fn total_power(&self, at: Vec2) -> f64 {
        let each = self.jammers.iter().map(|j| self.power_of(j, at));
        match self.combine {
            Combine::Sum => each.sum(),
            Combine::Max => each.fold(0.0, f64::max),
        }
    }

    fn clamp_to_arena(&self, p: Vec2) -> Vec2 {
        let snap = |v: f64| {
            if self.snap {
                (v * 2.0).round() / 2.0
            } else {
                v
            }
        };
        Vec2 {
            x: snap(p.x).clamp(0.0, self.config.arena_w),
            y: snap(p.y).clamp(0.0, self.config.arena_h),
        }
    }

    /// The jammed threshold in absolute power: `jam_threshold` is a fraction of P0,
    /// exactly as in `place_jamming`.
    fn threshold(&self) -> f64 {
        self.config.jam_threshold * self.config.jam_p0
    }

    /// Respawns the swarm and its epsilon graph when the seed changes.
    fn update_swarm(&mut self) {
        if self.swarm_seed == Some(self.seed) {
            return;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.swarm = spawn_swarm(&self.config, &mut rng);
        self.graph = epsilon_graph(&self.swarm, &self.config);
        self.swarm_seed = Some(self.seed);
        self.rng_after = rng;
    }

    /// True power per drone. With exactly one jammer this is your `true_power`;
    /// with several, the GUI combines them (sum or max), which `true_power` does not do.
    fn truth(&self) -> Vec<f64> {
        if let [jammer] = self.jammers.as_slice() {
            true_power(&self.swarm, jammer, &self.ridge_or_none(), &self.config)
        } else {
            self.swarm
                .iter()
                .map(|d| self.total_power(d.position))
                .collect()
        }
    }

    /// Sensor readings from your `readings`. Draw 0 continues the seed's RNG after
    /// spawn (and place_jamming), exactly as the dataset will; other draws use a fresh
    /// RNG. The noise depends only on the RNG, so moving the jammer changes the truth
    /// but not the noise on each drone.
    fn sensor_readings(&self, truth: &[f64]) -> Vec<f64> {
        let mut rng = if self.noise_draw == 0 {
            self.rng_after.clone()
        } else {
            ChaCha8Rng::seed_from_u64(
                self.seed
                    .wrapping_mul(1_000_003)
                    .wrapping_add(self.noise_draw),
            )
        };
        readings(truth, &self.config, &mut rng)
    }

    /// Jammed flag per drone (the zone label), from your `jammed_labels`.
    fn jammed(&self) -> Vec<bool> {
        jammed_labels(&self.truth(), &self.config)
    }

    /// Same geometry as the unit tests, moved by (10, 10): jammer at (10, 10),
    /// ridge at x = 12 from y = 5 to 15.
    fn load_test_setup(&mut self) {
        self.jammers = vec![Jammer {
            position: Vec2 { x: 10.0, y: 10.0 },
            power: 1.0,
        }];
        self.ridge = Some(Ridge {
            a: Vec2 { x: 12.0, y: 5.0 },
            b: Vec2 { x: 12.0, y: 15.0 },
        });
        self.selected = Some(0);
    }

    /// Spawns the swarm for `self.seed` and places one jammer and one ridge with
    /// `place_jamming`, using the same RNG after the spawn (as the dataset will).
    /// `place_jamming` panics when no placement works (for example with the jammed
    /// threshold slider at 1.0); the GUI catches that and shows a message instead.
    fn place_from_seed(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let swarm = spawn_swarm(&self.config, &mut rng);
        let config = self.config;
        // Silence the panic message while trying: the GUI shows its own.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            place_jamming(&swarm, &config, &mut rng)
        }));
        std::panic::set_hook(previous_hook);
        self.rng_after = rng;
        self.graph = epsilon_graph(&swarm, &self.config);
        self.swarm = swarm;
        self.swarm_seed = Some(self.seed);
        self.show_swarm = true;
        match result {
            Ok((jammer, ridge)) => {
                self.jammers = vec![jammer];
                self.ridge = Some(ridge);
                self.selected = Some(0);
                self.placed_for = Some(self.seed);
                let jammed = self.jammed().iter().filter(|&&j| j).count();
                self.place_msg = Some(format!(
                    "Seed {}: {} of {} drones jammed ({:.0} %).",
                    self.seed,
                    jammed,
                    self.swarm.len(),
                    100.0 * jammed as f64 / self.swarm.len().max(1) as f64
                ));
            }
            Err(_) => {
                self.placed_for = None;
                self.place_msg = Some(format!(
                    "Seed {}: place_jamming found no placement with {} to {} % jammed. \
                     Check the threshold and d0 sliders.",
                    self.seed,
                    100.0 * self.config.jam_frac_min,
                    100.0 * self.config.jam_frac_max
                ));
            }
        }
    }

    fn load_two_jammers(&mut self) {
        self.jammers = vec![
            Jammer {
                position: Vec2 { x: 13.0, y: 10.0 },
                power: 1.0,
            },
            Jammer {
                position: Vec2 { x: 21.0, y: 10.0 },
                power: 1.0,
            },
        ];
        self.ridge = Some(Ridge {
            a: Vec2 { x: 26.0, y: 3.0 },
            b: Vec2 { x: 29.0, y: 17.0 },
        });
        self.selected = None;
    }

    // ------------------------------------------------------------- heatmap

    fn heat_key_now(&self) -> HeatKey {
        HeatKey {
            jammers: self.jammers.clone(),
            ridge: self.ridge,
            d0: self.config.jam_d0,
            factor: self.config.ridge_factor,
            threshold: self.config.jam_threshold,
            p0: self.config.jam_p0,
            combine: self.combine,
            scale: self.scale,
            contours: self.contours,
        }
    }

    /// Maps a power to 0..1 for the colour ramp.
    fn scale_t(&self, power: f64) -> f32 {
        let t = match self.scale {
            Scale::Linear => power,
            // 0.01 P0 -> 0, 1 P0 -> 1.
            Scale::Log => (power.max(1e-9).log10() + 2.0) / 2.0,
        };
        t.clamp(0.0, 1.0) as f32
    }

    /// One pixel per RES km. Row 0 is the top of the arena (y = arena_h).
    fn heat_image(&self) -> ColorImage {
        let (aw, ah) = (self.config.arena_w, self.config.arena_h);
        let w = (aw / RES).round() as usize;
        let h = (ah / RES).round() as usize;
        let mut vals = vec![0.0; w * h];
        for r in 0..h {
            for c in 0..w {
                let at = Vec2 {
                    x: (c as f64 + 0.5) * RES,
                    y: ah - (r as f64 + 0.5) * RES,
                };
                vals[r * w + c] = self.total_power(at);
            }
        }
        let p0 = self.config.jam_p0.max(1e-9);
        let mut px: Vec<Color32> = vals
            .iter()
            .map(|&v| heat_color(self.scale_t(v / p0)))
            .collect();

        if self.contours {
            // A pixel is on a contour when it and its right or lower neighbour sit on
            // different sides of the level.
            let levels = [
                (self.threshold(), Color32::WHITE),
                (0.2 * p0, Color32::from_white_alpha(110)),
                (0.1 * p0, Color32::from_white_alpha(70)),
            ];
            for r in 0..h {
                for c in 0..w {
                    let v = vals[r * w + c];
                    let right = if c + 1 < w { vals[r * w + c + 1] } else { v };
                    let down = if r + 1 < h { vals[(r + 1) * w + c] } else { v };
                    for &(level, color) in levels.iter().rev() {
                        let above = v >= level;
                        if above != (right >= level) || above != (down >= level) {
                            px[r * w + c] = blend(px[r * w + c], color);
                        }
                    }
                }
            }
        }
        ColorImage::new([w, h], px)
    }

    // ---------------------------------------------------------- left panel

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Jamming playground");
        ui.weak("Every pixel calls your jam_power.");
        ui.separator();

        ui.label(RichText::new("Tool").strong());
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tool, Tool::Move, "Move");
            ui.selectable_value(&mut self.tool, Tool::AddJammer, "Add jammer");
            ui.selectable_value(&mut self.tool, Tool::DrawRidge, "Draw ridge");
        });
        ui.weak(match self.tool {
            Tool::Move => "Drag a jammer, a ridge end or the ridge. Drag empty space to pan.",
            Tool::AddJammer => "Click to add a jammer. Drag a jammer to move it.",
            Tool::DrawRidge => "Drag from one end of the ridge to the other (one ridge).",
        });
        ui.weak("Right-click a jammer or the ridge to delete it. Scroll: zoom.");
        ui.checkbox(&mut self.snap, "Snap to 0.5 km");
        ui.separator();

        // Jammers.
        ui.label(RichText::new("Jammers").strong());
        let mut remove = None;
        for (i, j) in self.jammers.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                let label = RichText::new(format!("J{i}")).color(JAMMER_COLOR).strong();
                if ui
                    .selectable_label(self.selected == Some(i), label)
                    .clicked()
                {
                    self.selected = Some(i);
                }
                ui.add(
                    egui::DragValue::new(&mut j.power)
                        .range(0.0..=5.0)
                        .speed(0.01)
                        .prefix("P0 "),
                );
                ui.add(
                    egui::DragValue::new(&mut j.position.x)
                        .range(0.0..=self.config.arena_w)
                        .speed(0.1)
                        .prefix("x "),
                );
                ui.add(
                    egui::DragValue::new(&mut j.position.y)
                        .range(0.0..=self.config.arena_h)
                        .speed(0.1)
                        .prefix("y "),
                );
                if ui.small_button("✖").on_hover_text("Delete").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.remove_jammer(i);
        }
        if self.jammers.is_empty() {
            ui.weak("No jammer. Pick \"Add jammer\" and click the map.");
        }
        ui.add_space(4.0);

        // Ridge.
        ui.label(RichText::new("Ridge").strong());
        match &mut self.ridge {
            Some(r) => {
                for (name, end) in [("a", &mut r.a), ("b", &mut r.b)] {
                    ui.horizontal(|ui| {
                        ui.label(name);
                        ui.add(
                            egui::DragValue::new(&mut end.x)
                                .range(0.0..=self.config.arena_w)
                                .speed(0.1)
                                .prefix("x "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut end.y)
                                .range(0.0..=self.config.arena_h)
                                .speed(0.1)
                                .prefix("y "),
                        );
                    });
                }
                let len = r.a.dist(r.b);
                ui.weak(format!("Length {len:.1} km"));
                if ui.button("Remove ridge").clicked() {
                    self.ridge = None;
                }
            }
            None => {
                ui.weak("No ridge. Pick \"Draw ridge\" and drag on the map.");
            }
        }
        ui.separator();

        // Physics. Moving any of these re-runs place_jamming when a placement is shown.
        ui.label(RichText::new("Physics (SimConfig)").strong());
        let physics_before = (
            self.config.jam_p0,
            self.config.jam_d0,
            self.config.ridge_factor,
            self.config.jam_threshold,
        );
        ui.add(egui::Slider::new(&mut self.config.jam_p0, 0.1..=3.0).text("P0  jammer power"));
        ui.add(
            egui::Slider::new(&mut self.config.jam_d0, 0.5..=10.0)
                .text("d0  half-power distance")
                .suffix(" km"),
        );
        ui.add(egui::Slider::new(&mut self.config.ridge_factor, 0.0..=1.0).text("ridge factor"));
        ui.add(
            egui::Slider::new(&mut self.config.jam_threshold, 0.05..=1.0).text("jammed threshold"),
        );
        ui.weak("Threshold, 0.2 and 0.1 contours and colours are fractions of P0.");
        let physics_after = (
            self.config.jam_p0,
            self.config.jam_d0,
            self.config.ridge_factor,
            self.config.jam_threshold,
        );
        if physics_after != physics_before {
            if self.placed_for.is_some() {
                // Live: a new placement under the new physics.
                self.place_from_seed();
            } else if physics_after.0 != physics_before.0 {
                // No placement shown: P0 applies to the jammers on the map.
                for j in &mut self.jammers {
                    j.power = self.config.jam_p0;
                }
            }
        }
        ui.horizontal(|ui| {
            ui.label("Several jammers:");
            ui.selectable_value(&mut self.combine, Combine::Sum, "sum");
            ui.selectable_value(&mut self.combine, Combine::Max, "max");
        });
        ui.weak("How jammers add up is not decided yet (D02). Noise powers usually add: sum.");
        if ui.button("Reset physics to defaults").clicked() {
            let d = SimConfig::default();
            self.config.jam_p0 = d.jam_p0;
            self.config.jam_d0 = d.jam_d0;
            self.config.ridge_factor = d.ridge_factor;
            self.config.jam_threshold = d.jam_threshold;
            self.config.jam_frac_min = d.jam_frac_min;
            self.config.jam_frac_max = d.jam_frac_max;
            self.config.noise_sd = d.noise_sd;
        }
        ui.separator();

        // Presets.
        ui.label(RichText::new("Presets").strong());
        ui.horizontal(|ui| {
            if ui
                .button("Test setup")
                .on_hover_text(
                    "The unit-test geometry moved by (10, 10): jammer at (10, 10),\n\
                     ridge at x = 12. Hover (10, 14): 0.5, (14, 10): 0.05, (12, 10): 0.8.",
                )
                .clicked()
            {
                self.load_test_setup();
            }
            if ui.button("Two jammers").clicked() {
                self.load_two_jammers();
            }
            if ui.button("Clear").clicked() {
                self.jammers.clear();
                self.ridge = None;
                self.selected = None;
            }
        });
        ui.separator();

        // Display.
        ui.label(RichText::new("Display").strong());
        ui.horizontal(|ui| {
            ui.label("Colour scale:");
            ui.selectable_value(&mut self.scale, Scale::Linear, "linear");
            ui.selectable_value(&mut self.scale, Scale::Log, "log");
        });
        ui.checkbox(
            &mut self.contours,
            "Contours (threshold white, 0.2 and 0.1 grey)",
        );
        ui.checkbox(&mut self.d0_rings, "d0 ring around each jammer");
        ui.checkbox(&mut self.probe_lines, "Lines from jammers to the cursor");
        ui.horizontal(|ui| {
            if ui.button("Reset view").clicked() {
                self.zoom = 1.0;
                self.pan = EVec2::ZERO;
            }
            ui.weak(format!("zoom ×{:.1}", self.zoom));
        });
        ui.separator();

        // Swarm.
        ui.label(RichText::new("Swarm").strong());
        ui.checkbox(&mut self.show_swarm, "Show a spawned swarm");
        let seed_before = self.seed;
        ui.horizontal(|ui| {
            ui.label("Seed");
            ui.add(egui::DragValue::new(&mut self.seed).range(0..=999_999));
            if ui.small_button("◀").clicked() {
                self.seed = self.seed.saturating_sub(1);
            }
            if ui.small_button("▶").clicked() {
                self.seed += 1;
            }
        });
        ui.horizontal(|ui| {
            if ui
                .button("Place from seed")
                .on_hover_text(
                    "Runs your place_jamming on this seed's swarm: one jammer and one \
                     ridge, redrawn until 5-30 % of the drones are jammed.\n\
                     Replaces the jammers and the ridge on the map.",
                )
                .clicked()
            {
                self.place_from_seed();
            }
            ui.checkbox(&mut self.auto_place, "on every seed change");
        });

        // Placement bounds (D02), live: moving a slider re-runs place_jamming.
        let bounds_before = (self.config.jam_frac_min, self.config.jam_frac_max);
        ui.add(
            egui::Slider::new(&mut self.config.jam_frac_min, 0.0..=1.0)
                .text("min jammed share")
                .custom_formatter(|v, _| format!("{:.0} %", v * 100.0)),
        );
        ui.add(
            egui::Slider::new(&mut self.config.jam_frac_max, 0.0..=1.0)
                .text("max jammed share")
                .custom_formatter(|v, _| format!("{:.0} %", v * 100.0)),
        );
        // Keep min <= max: whichever slider moved pushes the other one.
        if self.config.jam_frac_min > self.config.jam_frac_max {
            if self.config.jam_frac_min != bounds_before.0 {
                self.config.jam_frac_max = self.config.jam_frac_min;
            } else {
                self.config.jam_frac_min = self.config.jam_frac_max;
            }
        }
        let bounds_moved = (self.config.jam_frac_min, self.config.jam_frac_max) != bounds_before;
        if bounds_moved && self.placed_for.is_some() {
            self.place_from_seed();
        }
        ui.weak("place_jamming redraws until the jammed share is between these. Moving them re-places live.");
        if self.auto_place && self.seed != seed_before {
            self.place_from_seed();
        }
        if let Some(msg) = &self.place_msg {
            ui.weak(msg);
        }
        if self.placed_for.is_some() && self.placed_for != Some(self.seed) {
            ui.weak("The seed changed since the placement: press the button again.");
        }
        ui.add_enabled_ui(self.show_swarm, |ui| {
            ui.checkbox(&mut self.show_failed, "Show failed links (red, dashed)");
        });
        ui.weak("Jammed = power ≥ threshold. A link fails when either end is jammed.");
        ui.separator();

        // Readings (2c).
        ui.label(RichText::new("Readings (2c)").strong());
        ui.add(egui::Slider::new(&mut self.config.noise_sd, 0.0..=0.5).text("noise sd (× P0)"));
        ui.horizontal(|ui| {
            ui.label("Noise draw:");
            let name = if self.noise_draw == 0 {
                "dataset".to_string()
            } else {
                format!("#{}", self.noise_draw)
            };
            ui.label(RichText::new(name).monospace());
            if ui
                .button("New draw")
                .on_hover_text("Fresh noise on every drone, same truth.")
                .clicked()
            {
                self.noise_draw += 1;
            }
            if ui
                .add_enabled(self.noise_draw != 0, egui::Button::new("Dataset draw"))
                .on_hover_text(
                    "The noise the dataset gets: the seed's RNG, continued after \
                     spawn_swarm and place_jamming.",
                )
                .clicked()
            {
                self.noise_draw = 0;
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Colour drones by:");
            ui.selectable_value(&mut self.drone_color, DroneColor::Role, "role");
            ui.selectable_value(&mut self.drone_color, DroneColor::Truth, "true power");
            ui.selectable_value(&mut self.drone_color, DroneColor::Reading, "reading");
            ui.selectable_value(&mut self.drone_color, DroneColor::Check, "label vs reading");
        });
        if self.drone_color == DroneColor::Check {
            for c in Check::ALL {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, c.color());
                    ui.label(c.name());
                });
            }
        }
        ui.checkbox(
            &mut self.show_heat,
            "Heatmap (turn off to see drone colours)",
        );
        ui.weak(
            "Reading = truth + Normal(0, noise sd). The label (red ring) comes from the \
             truth; the model will only see the readings.",
        );
        ui.separator();

        // Legend.
        ui.label(RichText::new("Legend").strong());
        ui.horizontal(|ui| {
            for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let (rect, _) = ui.allocate_exact_size(vec2(18.0, 12.0), Sense::hover());
                ui.painter().rect_filled(rect, 2.0, heat_color(t));
                let label = match self.scale {
                    Scale::Linear => format!("{t}"),
                    Scale::Log => format!("{}", 10f64.powf(t as f64 * 2.0 - 2.0)),
                };
                ui.label(label);
            }
        });
        ui.weak("Power as a fraction of P0 = 1.");
    }

    fn remove_jammer(&mut self, i: usize) {
        self.jammers.remove(i);
        self.selected = match self.selected {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        self.drag = None;
    }

    // --------------------------------------------------------- right panel

    fn details(&mut self, ui: &mut egui::Ui) {
        self.plot_hover = None;
        ui.label(RichText::new("Power against distance (as a fraction of P0)").strong());
        ui.label(RichText::new("P(d) = P0 / (1 + (d / d0)²)").monospace());
        falloff_plot(ui, &self.config);
        let d0 = self.config.jam_d0;
        let f = self.config.ridge_factor;
        egui::Grid::new("falloff_values")
            .striped(true)
            .show(ui, |ui| {
                ui.label("d");
                ui.label("open ground");
                ui.label("behind ridge");
                ui.end_row();
                for k in [0.0, 1.0, 2.0, 3.0] {
                    let p = self.config.jam_p0 / (1.0 + k * k);
                    ui.label(format!("{k} d0 = {:.1} km", k * d0));
                    ui.label(format!("{p:.3}"));
                    ui.label(format!("{:.3}", p * f));
                    ui.end_row();
                }
            });
        let t = self.config.jam_threshold;
        if t < 1.0 {
            let r = d0 * (1.0 / t - 1.0).sqrt();
            ui.weak(format!(
                "Jammed zone in open ground (P0 = 1): radius d0·√(1/{t} − 1) = {r:.2} km."
            ));
        }
        ui.separator();

        // Probe.
        ui.label(RichText::new("At the cursor").strong());
        match self.probe {
            None => {
                ui.weak("Hover the map.");
            }
            Some(at) => {
                let total = self.total_power(at);
                let jammed = total >= self.threshold();
                ui.label(format!("({:.2}, {:.2}) km", at.x, at.y));
                ui.label(
                    RichText::new(format!(
                        "total {:.3}  →  {}",
                        total,
                        if jammed { "JAMMED" } else { "not jammed" }
                    ))
                    .strong()
                    .color(if jammed {
                        JAMMER_COLOR
                    } else {
                        ui.visuals().text_color()
                    }),
                );
                egui::Grid::new("probe").striped(true).show(ui, |ui| {
                    ui.label("");
                    ui.label("d (km)");
                    ui.label("open");
                    ui.label("ridge");
                    ui.label("power");
                    ui.end_row();
                    for (i, j) in self.jammers.iter().enumerate() {
                        let blocked = self.blocked(j, at);
                        ui.label(RichText::new(format!("J{i}")).color(JAMMER_COLOR));
                        ui.label(format!("{:.2}", j.position.dist(at)));
                        ui.label(format!("{:.3}", self.open_power(j, at)));
                        ui.label(if blocked {
                            RichText::new("blocked").color(BLOCKED_COLOR)
                        } else {
                            RichText::new("clear")
                        });
                        ui.label(format!("{:.3}", self.power_of(j, at)));
                        ui.end_row();
                    }
                });
            }
        }
        ui.separator();

        // Swarm effect.
        ui.label(RichText::new("Effect on the swarm").strong());
        if !self.show_swarm {
            ui.weak("Turn on \"Show a spawned swarm\" in the left panel.");
            return;
        }
        let jammed = self.jammed();
        let n_jammed = jammed.iter().filter(|&&j| j).count();
        let before = stats(&self.graph, &self.swarm);
        let after_graph = fail_links(&self.graph, &jammed);
        let after = stats(&after_graph, &self.swarm);
        egui::Grid::new("swarm_effect")
            .striped(true)
            .show(ui, |ui| {
                ui.label("");
                ui.label("before");
                ui.label("after jamming");
                ui.end_row();
                ui.label("Jammed drones");
                ui.label("");
                ui.label(format!(
                    "{n_jammed} ({:.0} %)",
                    100.0 * n_jammed as f64 / self.swarm.len().max(1) as f64
                ));
                ui.end_row();
                ui.label("Links");
                ui.label(before.n_edges.to_string());
                ui.label(after.n_edges.to_string());
                ui.end_row();
                ui.label("Mean degree");
                ui.label(format!("{:.2}", before.mean_degree));
                ui.label(format!("{:.2}", after.mean_degree));
                ui.end_row();
                ui.label("Components");
                ui.label(before.n_components.to_string());
                ui.label(after.n_components.to_string());
                ui.end_row();
                ui.label("Largest component");
                ui.label(before.largest_component.to_string());
                ui.label(after.largest_component.to_string());
                ui.end_row();
                ui.label("Isolated drones");
                ui.label(before.isolated.to_string());
                ui.label(after.isolated.to_string());
                ui.end_row();
            });
        ui.weak(
            "Every jammed drone loses all its links, so it becomes isolated. \
             The plan's 2b question: how many drones should a jammer hit (5 to 30 %?).",
        );
        ui.separator();

        // Readings against truth.
        ui.label(RichText::new("Readings against truth (2c)").strong());
        let truth = self.truth();
        let reads = self.sensor_readings(&truth);
        let thr = self.threshold();
        let sd = self.config.noise_sd * self.config.jam_p0;
        ui.label(format!(
            "noise sd = noise_sd × P0 = {sd:.3}   ·   draw: {}",
            if self.noise_draw == 0 {
                "dataset".to_string()
            } else {
                format!("#{}", self.noise_draw)
            }
        ));
        let mut counts = [0usize; 4];
        for (&t, &r) in truth.iter().zip(&reads) {
            counts[check(t >= thr, r, thr) as usize] += 1;
        }
        egui::Grid::new("checks").striped(true).show(ui, |ui| {
            for c in Check::ALL {
                let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
                ui.painter().circle_filled(rect.center(), 5.0, c.color());
                ui.label(c.name());
                ui.label(counts[c as usize].to_string());
                ui.end_row();
            }
        });
        let n = truth.len().max(1) as f64;
        let noise: Vec<f64> = reads.iter().zip(&truth).map(|(r, t)| r - t).collect();
        let mean = noise.iter().sum::<f64>() / n;
        let spread = (noise.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n).sqrt();
        ui.label(format!(
            "Measured noise over {} drones: mean {mean:+.3}, sd {spread:.3}",
            truth.len()
        ));
        self.plot_hover = truth_reading_plot(ui, &truth, &reads, thr, self.config.jam_p0);
        ui.weak(
            "Each dot is a drone. On the diagonal: no noise. Right of the vertical line: \
             jammed (label, from the truth). Above the horizontal line: the reading says \
             jammed. Top left: false alarms. Bottom right: missed. Hover a dot to find \
             the drone on the map.",
        );
    }

    // --------------------------------------------------------------- canvas

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let (aw, ah) = (self.config.arena_w, self.config.arena_h);

        // Zoom around the cursor.
        if let Some(pointer) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let before =
                    Transform::fit(response.rect, aw, ah, self.zoom, self.pan).to_km(pointer);
                self.zoom = (self.zoom * (scroll * 0.002).exp()).clamp(0.5, 25.0);
                let moved =
                    Transform::fit(response.rect, aw, ah, self.zoom, self.pan).to_screen(before);
                self.pan += pointer - moved;
            }
        }
        let t = Transform::fit(response.rect, aw, ah, self.zoom, self.pan);
        let visuals = ui.visuals().clone();
        let text = visuals.text_color();

        // --- Input.
        if response.drag_started() {
            let origin = ui
                .input(|i| i.pointer.press_origin())
                .or(response.interact_pointer_pos());
            self.drag = origin.map(|p| match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => {
                    self.selected = Some(i);
                    Drag::Jammer(i)
                }
                Some(Hit::RidgeA) => Drag::RidgeA,
                Some(Hit::RidgeB) => Drag::RidgeB,
                Some(Hit::Ridge) if self.tool != Tool::DrawRidge => Drag::RidgeBody,
                _ if self.tool == Tool::DrawRidge => {
                    Drag::NewRidge(self.clamp_to_arena(t.to_km(p)))
                }
                _ => Drag::Pan,
            });
        }
        if response.dragged()
            && let Some(drag) = self.drag
        {
            let pointer = response.interact_pointer_pos();
            let km = pointer.map(|p| self.clamp_to_arena(t.to_km(p)));
            match (drag, km) {
                (Drag::Jammer(i), Some(km)) if i < self.jammers.len() => {
                    self.jammers[i].position = km
                }
                (Drag::RidgeA, Some(km)) => {
                    if let Some(r) = &mut self.ridge {
                        r.a = km;
                    }
                }
                (Drag::RidgeB, Some(km)) => {
                    if let Some(r) = &mut self.ridge {
                        r.b = km;
                    }
                }
                (Drag::RidgeBody, _) => {
                    let d = response.drag_delta();
                    let (dx, dy) = ((d.x / t.scale) as f64, (-d.y / t.scale) as f64);
                    if let Some(r) = &mut self.ridge {
                        r.a.x += dx;
                        r.a.y += dy;
                        r.b.x += dx;
                        r.b.y += dy;
                    }
                }
                (Drag::Pan, _) => self.pan += response.drag_delta(),
                _ => {}
            }
        }
        if response.drag_stopped() {
            if let Some(Drag::NewRidge(start)) = self.drag
                && let Some(p) = response.interact_pointer_pos()
            {
                let end = self.clamp_to_arena(t.to_km(p));
                if start.dist(end) > 0.3 {
                    self.ridge = Some(Ridge { a: start, b: end });
                }
            }
            self.drag = None;
        }
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => self.selected = Some(i),
                _ if self.tool == Tool::AddJammer && t.arena_rect().contains(p) => {
                    let position = self.clamp_to_arena(t.to_km(p));
                    self.jammers.push(Jammer {
                        position,
                        power: self.config.jam_p0,
                    });
                    self.selected = Some(self.jammers.len() - 1);
                }
                _ => self.selected = None,
            }
        }
        if response.secondary_clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            match self.hit(&t, p) {
                Some(Hit::Jammer(i)) => self.remove_jammer(i),
                Some(Hit::Ridge | Hit::RidgeA | Hit::RidgeB) => self.ridge = None,
                None => {}
            }
        }
        if !ui.ctx().egui_wants_keyboard_input() {
            if let Some(i) = self.selected
                && ui.input(|inp| inp.key_pressed(Key::Delete))
            {
                self.remove_jammer(i);
            }
            if ui.input(|inp| inp.key_pressed(Key::Escape)) {
                self.selected = None;
            }
        }
        self.probe = response
            .hover_pos()
            .filter(|p| t.arena_rect().contains(*p))
            .map(|p| t.to_km(p));

        // --- Heatmap (rebuilt only when something it depends on changed).
        let key = self.heat_key_now();
        if self.heat.is_none() || self.heat_key.as_ref() != Some(&key) {
            let img = self.heat_image();
            match &mut self.heat {
                Some(tex) => tex.set(img, TextureOptions::LINEAR),
                None => {
                    self.heat = Some(ui.ctx().load_texture(
                        "jam_heat",
                        img,
                        TextureOptions::LINEAR,
                    ));
                }
            }
            self.heat_key = Some(key);
        }
        let arena = t.arena_rect();
        painter.rect_filled(arena, 0.0, Color32::BLACK);
        if self.show_heat
            && let Some(tex) = &self.heat
        {
            painter.image(
                tex.id(),
                arena,
                Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        // --- Grid every 5 km, with labels.
        let grid = Stroke::new(1.0, Color32::from_white_alpha(28));
        let label_color = text.gamma_multiply(0.7);
        let mut x = 0.0;
        while x <= aw + 1e-9 {
            let top = t.to_screen(Vec2 { x, y: ah });
            let bottom = t.to_screen(Vec2 { x, y: 0.0 });
            painter.line_segment([top, bottom], grid);
            painter.text(
                bottom + vec2(0.0, 4.0),
                Align2::CENTER_TOP,
                format!("{x}"),
                FontId::monospace(10.0),
                label_color,
            );
            x += 5.0;
        }
        let mut y = 0.0;
        while y <= ah + 1e-9 {
            let left = t.to_screen(Vec2 { x: 0.0, y });
            let right = t.to_screen(Vec2 { x: aw, y });
            painter.line_segment([left, right], grid);
            painter.text(
                left - vec2(4.0, 0.0),
                Align2::RIGHT_CENTER,
                format!("{y}"),
                FontId::monospace(10.0),
                label_color,
            );
            y += 5.0;
        }

        // --- Swarm layer.
        let truth = if self.show_swarm {
            self.truth()
        } else {
            Vec::new()
        };
        let reads = self.sensor_readings(&truth);
        let thr = self.threshold();
        if self.show_swarm {
            let jammed = jammed_labels(&truth, &self.config);
            for e in &self.graph.edges {
                let (s, d) = (e.src as usize, e.dst as usize);
                let a = t.to_screen(self.swarm[s].position);
                let b = t.to_screen(self.swarm[d].position);
                if jammed[s] || jammed[d] {
                    if self.show_failed {
                        painter.extend(Shape::dashed_line(
                            &[a, b],
                            Stroke::new(1.2, Color32::from_rgb(255, 70, 70)),
                            4.0,
                            3.0,
                        ));
                    }
                } else {
                    let alpha = (60.0 + 160.0 * e.weight) as u8;
                    painter
                        .line_segment([a, b], Stroke::new(1.0, Color32::from_white_alpha(alpha)));
                }
            }
            let radius = (4.5 * self.zoom.sqrt()).clamp(3.5, 10.0);
            let p0 = self.config.jam_p0.max(1e-9);
            let outline = match self.drone_color {
                DroneColor::Role | DroneColor::Check => Stroke::new(1.0, Color32::BLACK),
                DroneColor::Truth | DroneColor::Reading => {
                    Stroke::new(1.0, Color32::from_white_alpha(180))
                }
            };
            for (i, d) in self.swarm.iter().enumerate() {
                let c = t.to_screen(d.position);
                let fill = match self.drone_color {
                    DroneColor::Role => role_color(d.role),
                    DroneColor::Truth => heat_color(self.scale_t(truth[i] / p0)),
                    DroneColor::Reading => heat_color(self.scale_t(reads[i] / p0)),
                    DroneColor::Check => check(jammed[i], reads[i], thr).color(),
                };
                draw_drone(&painter, c, radius, d.role, fill, outline);
                if jammed[i] {
                    painter.circle_stroke(
                        c,
                        radius + 3.0,
                        Stroke::new(2.0, Color32::from_rgb(255, 70, 70)),
                    );
                }
                if self.plot_hover == Some(i) {
                    painter.circle_stroke(c, radius + 8.0, Stroke::new(2.5, Color32::WHITE));
                }
            }
        }

        // --- Ridge.
        if let Some(r) = self.ridge {
            let a = t.to_screen(r.a);
            let b = t.to_screen(r.b);
            painter.line_segment([a, b], Stroke::new(6.0, Color32::from_black_alpha(160)));
            painter.line_segment([a, b], Stroke::new(4.0, RIDGE_COLOR));
            for (end, name) in [(a, "a"), (b, "b")] {
                painter.circle(end, 5.0, RIDGE_COLOR, Stroke::new(1.5, Color32::WHITE));
                painter.text(
                    end + vec2(7.0, -7.0),
                    Align2::LEFT_BOTTOM,
                    name,
                    FontId::proportional(12.0),
                    Color32::WHITE,
                );
            }
        }
        if let Some(Drag::NewRidge(start)) = self.drag
            && let Some(p) = response.interact_pointer_pos()
        {
            let end = t.to_screen(self.clamp_to_arena(t.to_km(p)));
            painter.extend(Shape::dashed_line(
                &[t.to_screen(start), end],
                Stroke::new(3.0, RIDGE_COLOR),
                8.0,
                5.0,
            ));
        }

        // --- Lines from each jammer to the cursor: dashed blue when the ridge blocks them.
        if self.probe_lines
            && self.drag.is_none()
            && let Some(at) = self.probe
        {
            let cursor = t.to_screen(at);
            for j in &self.jammers {
                let from = t.to_screen(j.position);
                if self.blocked(j, at) {
                    painter.extend(Shape::dashed_line(
                        &[from, cursor],
                        Stroke::new(1.5, BLOCKED_COLOR),
                        6.0,
                        4.0,
                    ));
                } else {
                    painter.line_segment(
                        [from, cursor],
                        Stroke::new(1.0, Color32::from_white_alpha(140)),
                    );
                }
            }
        }

        // --- Jammers.
        for (i, j) in self.jammers.iter().enumerate() {
            let c = t.to_screen(j.position);
            if self.d0_rings {
                let ring = circle_points(c, t.len(self.config.jam_d0), 96);
                painter.extend(Shape::dashed_line(
                    &ring,
                    Stroke::new(1.0, Color32::from_white_alpha(150)),
                    5.0,
                    4.0,
                ));
            }
            draw_jammer(&painter, c, self.selected == Some(i));
            painter.text(
                c + vec2(10.0, -10.0),
                Align2::LEFT_BOTTOM,
                format!("J{i}  P0 {:.2}", j.power),
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }

        // --- Overlays.
        painter.text(
            response.rect.left_top() + vec2(8.0, 6.0),
            Align2::LEFT_TOP,
            match self.tool {
                Tool::Move => "Move: drag jammers / ridge · drag empty: pan · right-click: delete",
                Tool::AddJammer => "Add jammer: click the map",
                Tool::DrawRidge => "Draw ridge: drag from one end to the other",
            },
            FontId::proportional(12.0),
            text.gamma_multiply(0.8),
        );
        if let Some(at) = self.probe {
            painter.text(
                response.rect.right_top() + vec2(-8.0, 6.0),
                Align2::RIGHT_TOP,
                format!(
                    "({:.1}, {:.1}) km   power {:.3}",
                    at.x,
                    at.y,
                    self.total_power(at)
                ),
                FontId::monospace(12.0),
                text,
            );
        }
        if self.show_swarm
            && let Some(p) = response.hover_pos()
            && let Some(i) = self.drone_at(&t, p)
        {
            let d = &self.swarm[i];
            let (power, reading) = (truth[i], reads[i]);
            let label = power >= thr;
            let verdict = check(label, reading, thr);
            response.on_hover_ui_at_pointer(|ui| {
                ui.strong(format!(
                    "Drone {} · squad S{} · {}",
                    d.id.0,
                    d.squad.0,
                    role_name(d.role)
                ));
                ui.label(RichText::new(format!("True power {power:.3}")).monospace());
                ui.label(
                    RichText::new(format!(
                        "Reading    {reading:.3}  (noise {:+.3})",
                        reading - power
                    ))
                    .monospace(),
                );
                ui.label(if label {
                    "Label: jammed, all its links fail"
                } else {
                    "Label: not jammed"
                });
                ui.label(RichText::new(verdict.name()).color(verdict.color()));
            });
        }
    }

    /// What is under screen point `p`: a jammer first, then a ridge end, then the ridge.
    fn hit(&self, t: &Transform, p: Pos2) -> Option<Hit> {
        let nearest = self
            .jammers
            .iter()
            .enumerate()
            .map(|(i, j)| (i, t.to_screen(j.position).distance(p)))
            .filter(|&(_, d)| d <= GRAB_JAMMER)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = nearest {
            return Some(Hit::Jammer(i));
        }
        let r = self.ridge?;
        let (a, b) = (t.to_screen(r.a), t.to_screen(r.b));
        if a.distance(p) <= GRAB_END {
            Some(Hit::RidgeA)
        } else if b.distance(p) <= GRAB_END {
            Some(Hit::RidgeB)
        } else if point_segment_distance(p, a, b) <= GRAB_RIDGE {
            Some(Hit::Ridge)
        } else {
            None
        }
    }

    fn drone_at(&self, t: &Transform, p: Pos2) -> Option<usize> {
        self.swarm
            .iter()
            .enumerate()
            .map(|(i, d)| (i, t.to_screen(d.position).distance(p)))
            .filter(|&(_, d)| d <= 8.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Hit {
    Jammer(usize),
    RidgeA,
    RidgeB,
    Ridge,
}

/// Power against distance, open ground (solid) and behind the ridge (dashed),
/// with the jammed threshold and d0 marked. Hover for a readout.
fn falloff_plot(ui: &mut egui::Ui, cfg: &SimConfig) {
    let d0 = cfg.jam_d0;
    let d_max = 4.0 * d0;
    let width = ui.available_width().max(200.0);
    let (response, painter) = ui.allocate_painter(vec2(width, 220.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_max(rect.min + vec2(36.0, 10.0), rect.max - vec2(10.0, 28.0));
    let x = |d: f64| plot.left() + (d / d_max) as f32 * plot.width();
    let y = |p: f64| plot.bottom() - p as f32 * plot.height();
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    for p in [0.0, 0.25, 0.5, 0.75, 1.0] {
        painter.hline(
            plot.x_range(),
            y(p),
            Stroke::new(1.0, weak.gamma_multiply(0.25)),
        );
        painter.text(
            pos2(plot.left() - 4.0, y(p)),
            Align2::RIGHT_CENTER,
            format!("{p}"),
            small.clone(),
            weak,
        );
    }
    for k in 0..=4 {
        let d = k as f64 * d0;
        painter.vline(
            x(d),
            plot.y_range(),
            Stroke::new(1.0, weak.gamma_multiply(0.15)),
        );
        painter.text(
            pos2(x(d), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{d:.0}"),
            small.clone(),
            weak,
        );
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "distance d (km)",
        small.clone(),
        weak,
    );

    let open = |d: f64| 1.0 / (1.0 + (d / d0) * (d / d0));
    let n = 200;
    let pts = |f: &dyn Fn(f64) -> f64| -> Vec<Pos2> {
        (0..=n)
            .map(|k| d_max * k as f64 / n as f64)
            .map(|d| pos2(x(d), y(f(d))))
            .collect()
    };
    painter.line(pts(&open), Stroke::new(2.5, JAMMER_COLOR));
    let behind = pts(&|d| open(d) * cfg.ridge_factor);
    painter.extend(Shape::dashed_line(
        &behind,
        Stroke::new(1.5, BLOCKED_COLOR),
        5.0,
        4.0,
    ));

    // Threshold and d0.
    let thr = cfg.jam_threshold;
    painter.extend(Shape::dashed_line(
        &[pos2(plot.left(), y(thr)), pos2(plot.right(), y(thr))],
        Stroke::new(1.0, text.gamma_multiply(0.8)),
        4.0,
        3.0,
    ));
    painter.text(
        pos2(plot.right() - 2.0, y(thr) - 2.0),
        Align2::RIGHT_BOTTOM,
        "jammed above",
        small.clone(),
        text,
    );
    painter.circle_filled(pos2(x(d0), y(open(d0))), 3.5, JAMMER_COLOR);
    painter.text(
        pos2(x(d0) + 5.0, y(open(d0)) - 3.0),
        Align2::LEFT_BOTTOM,
        "d0",
        FontId::proportional(13.0),
        JAMMER_COLOR,
    );
    painter.text(
        plot.left_top() + vec2(6.0, 4.0),
        Align2::LEFT_TOP,
        "— open ground   - - behind ridge",
        small,
        text.gamma_multiply(0.8),
    );

    if let Some(p) = response.hover_pos()
        && plot.contains(p)
    {
        let d = ((p.x - plot.left()) / plot.width()) as f64 * d_max;
        painter.vline(
            p.x,
            plot.y_range(),
            Stroke::new(1.0, text.gamma_multiply(0.5)),
        );
        let label = format!(
            "d = {d:.2} km\nopen   {:.3}\nbehind {:.3}",
            open(d),
            open(d) * cfg.ridge_factor
        );
        let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), text);
        let mut pos = p + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = p.x - 8.0 - galley.size().x;
        }
        painter.rect_filled(
            Rect::from_min_size(pos, galley.size()).expand(3.0),
            3.0,
            visuals.window_fill,
        );
        painter.galley(pos, galley, text);
    }
}

/// Each drone as a dot: true power (x) against its reading (y), both as fractions of
/// P0, with the diagonal (no noise) and the threshold on both axes. Returns the drone
/// under the cursor.
fn truth_reading_plot(
    ui: &mut egui::Ui,
    truth: &[f64],
    reads: &[f64],
    threshold: f64,
    p0: f64,
) -> Option<usize> {
    let p0 = p0.max(1e-9);
    let thr = threshold / p0;
    let xs: Vec<f64> = truth.iter().map(|v| v / p0).collect();
    let ys: Vec<f64> = reads.iter().map(|v| v / p0).collect();
    let mut lo: f64 = 0.0;
    let mut hi: f64 = (thr * 1.6).max(0.2);
    for &v in xs.iter().chain(&ys) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let pad = 0.05 * (hi - lo);
    let (lo, hi) = (lo - pad, hi + pad);

    let width = ui.available_width().max(200.0);
    let side = (width - 50.0).min(300.0);
    let (response, painter) = ui.allocate_painter(vec2(width, side + 40.0), Sense::hover());
    let rect = response.rect;
    let plot = Rect::from_min_size(rect.min + vec2(40.0, 8.0), vec2(side, side));
    let sx = |v: f64| plot.left() + ((v - lo) / (hi - lo)) as f32 * plot.width();
    let sy = |v: f64| plot.bottom() - ((v - lo) / (hi - lo)) as f32 * plot.height();
    let visuals = ui.visuals();
    let text = visuals.text_color();
    let weak = visuals.weak_text_color();
    let small = FontId::proportional(11.0);

    painter.rect_filled(plot, 2.0, visuals.extreme_bg_color);
    let step = if hi - lo > 2.0 { 0.5 } else { 0.25 };
    let mut k = (lo / step).ceil() * step;
    while k <= hi {
        let grid = Stroke::new(1.0, weak.gamma_multiply(0.2));
        painter.hline(plot.x_range(), sy(k), grid);
        painter.vline(sx(k), plot.y_range(), grid);
        painter.text(
            pos2(plot.left() - 4.0, sy(k)),
            Align2::RIGHT_CENTER,
            format!("{k}"),
            small.clone(),
            weak,
        );
        painter.text(
            pos2(sx(k), plot.bottom() + 3.0),
            Align2::CENTER_TOP,
            format!("{k}"),
            small.clone(),
            weak,
        );
        k += step;
    }
    painter.text(
        pos2(plot.center().x, rect.bottom()),
        Align2::CENTER_BOTTOM,
        "x: true power   ·   y: reading   (× P0)",
        small.clone(),
        weak,
    );

    // Diagonal (reading = truth) and the threshold on both axes.
    painter.line_segment(
        [pos2(sx(lo), sy(lo)), pos2(sx(hi), sy(hi))],
        Stroke::new(1.0, text.gamma_multiply(0.35)),
    );
    let dash = Stroke::new(1.0, text.gamma_multiply(0.8));
    painter.extend(Shape::dashed_line(
        &[pos2(sx(thr), plot.top()), pos2(sx(thr), plot.bottom())],
        dash,
        4.0,
        3.0,
    ));
    painter.extend(Shape::dashed_line(
        &[pos2(plot.left(), sy(thr)), pos2(plot.right(), sy(thr))],
        dash,
        4.0,
        3.0,
    ));
    painter.text(
        plot.left_top() + vec2(4.0, 3.0),
        Align2::LEFT_TOP,
        "false alarm",
        small.clone(),
        Check::FalseAlarm.color(),
    );
    painter.text(
        plot.right_bottom() - vec2(4.0, 3.0),
        Align2::RIGHT_BOTTOM,
        "missed",
        small.clone(),
        Check::Missed.color(),
    );

    for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
        let c = check(truth[i] >= threshold, reads[i], threshold);
        painter.circle_filled(pos2(sx(x), sy(y)), 3.0, c.color());
    }

    let hovered = response.hover_pos().and_then(|p| {
        xs.iter()
            .zip(&ys)
            .enumerate()
            .map(|(i, (&x, &y))| (i, pos2(sx(x), sy(y)).distance(p)))
            .filter(|&(_, d)| d <= 6.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    });
    if let Some(i) = hovered {
        let c = pos2(sx(xs[i]), sy(ys[i]));
        painter.circle_stroke(c, 6.0, Stroke::new(1.5, text));
        let label = format!(
            "drone {i}\ntruth   {:.3}\nreading {:.3}",
            truth[i], reads[i]
        );
        let galley = painter.layout_no_wrap(label, FontId::monospace(11.0), text);
        let mut pos = c + vec2(8.0, 8.0);
        if pos.x + galley.size().x > rect.right() {
            pos.x = c.x - 8.0 - galley.size().x;
        }
        painter.rect_filled(
            Rect::from_min_size(pos, galley.size()).expand(3.0),
            3.0,
            visuals.window_fill,
        );
        painter.galley(pos, galley, text);
    }
    hovered
}

/// A jammer marker: a red disc with four short rays.
fn draw_jammer(painter: &egui::Painter, c: Pos2, selected: bool) {
    let stroke = Stroke::new(2.0, Color32::WHITE);
    for k in 0..4 {
        let a = std::f32::consts::FRAC_PI_4 + k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = vec2(a.cos(), a.sin());
        painter.line_segment([c + dir * 9.0, c + dir * 14.0], stroke);
    }
    painter.circle(c, 7.0, JAMMER_COLOR, stroke);
    if selected {
        painter.circle_stroke(c, 17.0, Stroke::new(2.0, Color32::WHITE));
    }
}

/// Dark purple (0) to orange to pale yellow (1), close to the "inferno" map.
fn heat_color(t: f32) -> Color32 {
    const STOPS: [[f32; 3]; 5] = [
        [10.0, 7.0, 30.0],
        [87.0, 16.0, 110.0],
        [188.0, 55.0, 84.0],
        [249.0, 142.0, 9.0],
        [252.0, 255.0, 164.0],
    ];
    let t = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f32;
    let i = (t.floor() as usize).min(STOPS.len() - 2);
    let u = t - i as f32;
    let mix = |k: usize| (STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * u) as u8;
    Color32::from_rgb(mix(0), mix(1), mix(2))
}

/// Paints `top` (with its alpha) over the opaque `base`.
fn blend(base: Color32, top: Color32) -> Color32 {
    let a = top.a() as f32 / 255.0;
    let mix = |b: u8, t: u8| (b as f32 * (1.0 - a) + t as f32 * a) as u8;
    // `top` is premultiplied; un-premultiply its colour before mixing.
    let un = |c: u8| {
        if a > 0.0 {
            (c as f32 / a).min(255.0) as u8
        } else {
            0
        }
    };
    Color32::from_rgb(
        mix(base.r(), un(top.r())),
        mix(base.g(), un(top.g())),
        mix(base.b(), un(top.b())),
    )
}

/// Distance in pixels from `p` to the segment a-b.
fn point_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    if len2 == 0.0 {
        return p.distance(a);
    }
    let u = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * u)
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

/// km to screen and back: the arena fitted into the panel, then zoom and pan; y up.
struct Transform {
    left: f32,
    bottom: f32,
    scale: f32,
    w: f64,
    h: f64,
}

impl Transform {
    fn fit(rect: Rect, w: f64, h: f64, zoom: f32, pan: EVec2) -> Self {
        let margin = 30.0;
        let fit = ((rect.width() - 2.0 * margin) / w as f32)
            .min((rect.height() - 2.0 * margin) / h as f32)
            .max(1.0);
        let scale = fit * zoom;
        Transform {
            left: rect.center().x - w as f32 * scale / 2.0 + pan.x,
            bottom: rect.center().y + h as f32 * scale / 2.0 + pan.y,
            scale,
            w,
            h,
        }
    }

    fn to_screen(&self, p: Vec2) -> Pos2 {
        pos2(
            self.left + p.x as f32 * self.scale,
            self.bottom - p.y as f32 * self.scale,
        )
    }

    fn to_km(&self, p: Pos2) -> Vec2 {
        Vec2 {
            x: ((p.x - self.left) / self.scale) as f64,
            y: ((self.bottom - p.y) / self.scale) as f64,
        }
    }

    fn len(&self, km: f64) -> f32 {
        km as f32 * self.scale
    }

    fn arena_rect(&self) -> Rect {
        Rect::from_min_max(
            self.to_screen(Vec2 { x: 0.0, y: self.h }),
            self.to_screen(Vec2 { x: self.w, y: 0.0 }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, y: f64) -> Vec2 {
        Vec2 { x, y }
    }

    #[test]
    fn test_setup_matches_the_unit_test_values() {
        let app = JamApp::default(); // loads the test setup
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(app.total_power(at(10.0, 14.0)), 0.5)); // d0, open
        assert!(close(app.total_power(at(14.0, 10.0)), 0.05)); // d0, behind the ridge
        assert!(close(app.total_power(at(12.0, 10.0)), 0.8)); // on the ridge
        assert!(app.blocked(&app.jammers[0], at(14.0, 10.0)));
        assert!(!app.blocked(&app.jammers[0], at(10.0, 14.0)));
    }

    #[test]
    fn no_ridge_means_nothing_is_blocked() {
        let app = JamApp {
            ridge: None,
            ..Default::default()
        };
        assert!((app.total_power(at(14.0, 10.0)) - 0.5).abs() < 1e-12);
        assert!(!app.blocked(&app.jammers[0], at(14.0, 10.0)));
    }

    #[test]
    fn sum_and_max_combine_two_jammers() {
        let mut app = JamApp {
            ridge: None,
            ..Default::default()
        };
        app.jammers = vec![
            Jammer {
                position: at(10.0, 10.0),
                power: 1.0,
            },
            Jammer {
                position: at(18.0, 10.0),
                power: 1.0,
            },
        ];
        // Midpoint: 4 km from both, 0.5 each.
        app.combine = Combine::Sum;
        assert!((app.total_power(at(14.0, 10.0)) - 1.0).abs() < 1e-12);
        app.combine = Combine::Max;
        assert!((app.total_power(at(14.0, 10.0)) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn heat_image_has_one_pixel_per_res_km() {
        let app = JamApp::default();
        let img = app.heat_image();
        let cfg = SimConfig::default();
        assert_eq!(
            img.size,
            [
                (cfg.arena_w / RES).round() as usize,
                (cfg.arena_h / RES).round() as usize
            ]
        );
    }

    #[test]
    fn jammed_drones_lose_all_their_links() {
        let mut app = JamApp {
            show_swarm: true,
            ..Default::default()
        };
        app.update_swarm();
        // Put the jammer on drone 0.
        app.jammers[0].position = app.swarm[0].position;
        let jammed = app.jammed();
        assert!(jammed[0]);
        let after = fail_links(&app.graph, &jammed);
        assert!(
            after
                .edges
                .iter()
                .all(|e| !jammed[e.src as usize] && !jammed[e.dst as usize])
        );
        assert!(after.edges.len() < app.graph.edges.len());
    }

    #[test]
    fn place_from_seed_matches_the_dataset_order() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        // Same calls, same RNG order as the dataset: spawn, then place.
        let cfg = SimConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let swarm = spawn_swarm(&cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
        assert_eq!(app.swarm, swarm);
        assert_eq!(app.jammers, vec![jammer]);
        assert_eq!(app.ridge, Some(ridge));
        assert_eq!(app.placed_for, Some(4));
        // The placement respects the bounds.
        let f = app.jammed().iter().filter(|&&j| j).count() as f64 / app.swarm.len() as f64;
        assert!((cfg.jam_frac_min..=cfg.jam_frac_max).contains(&f));
    }

    #[test]
    fn p0_cancels_out_of_who_is_jammed() {
        // The threshold is a fraction of P0 and the placed jammer has power P0,
        // so doubling P0 doubles every power and the threshold: same placement.
        let mut a = JamApp {
            seed: 9,
            ..Default::default()
        };
        a.place_from_seed();
        let mut b = JamApp {
            seed: 9,
            ..Default::default()
        };
        b.config.jam_p0 = 2.0;
        b.place_from_seed();
        assert_eq!(a.jammed(), b.jammed());
        assert_eq!(a.ridge, b.ridge);
        assert_eq!(a.jammers[0].position, b.jammers[0].position);
    }

    #[test]
    fn readings_follow_the_dataset_order() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        // The dataset: spawn, place, truth, readings, all with one RNG.
        let cfg = SimConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let swarm = spawn_swarm(&cfg, &mut rng);
        let (jammer, ridge) = place_jamming(&swarm, &cfg, &mut rng);
        let truth = true_power(&swarm, &jammer, &ridge, &cfg);
        let reads = readings(&truth, &cfg, &mut rng);
        assert_eq!(app.truth(), truth);
        assert_eq!(app.sensor_readings(&app.truth()), reads);
    }

    /// Reading minus truth, per drone.
    fn noise(app: &JamApp) -> Vec<f64> {
        let truth = app.truth();
        app.sensor_readings(&truth)
            .iter()
            .zip(&truth)
            .map(|(r, t)| r - t)
            .collect()
    }

    #[test]
    fn moving_the_jammer_changes_the_truth_not_the_noise() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let truth_before = app.truth();
        let noise_before = noise(&app);
        app.jammers[0].position.x += 3.0;
        assert_ne!(app.truth(), truth_before);
        for (a, b) in noise(&app).iter().zip(&noise_before) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn a_new_draw_changes_the_noise() {
        let mut app = JamApp {
            seed: 4,
            ..Default::default()
        };
        app.place_from_seed();
        let dataset = noise(&app);
        app.noise_draw = 1;
        assert_ne!(noise(&app), dataset);
        app.noise_draw = 0;
        assert_eq!(noise(&app), dataset);
    }

    #[test]
    fn check_sorts_label_against_reading() {
        assert_eq!(check(true, 0.6, 0.5), Check::Hit);
        assert_eq!(check(true, 0.4, 0.5), Check::Missed);
        assert_eq!(check(false, 0.6, 0.5), Check::FalseAlarm);
        assert_eq!(check(false, 0.4, 0.5), Check::Clear);
        assert_eq!(check(false, 0.5, 0.5), Check::FalseAlarm); // >= like the label
    }

    #[test]
    fn transform_round_trips() {
        let t = Transform::fit(
            Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 500.0)),
            40.0,
            20.0,
            2.5,
            vec2(30.0, -10.0),
        );
        let p = at(12.3, 7.7);
        let back = t.to_km(t.to_screen(p));
        assert!((back.x - p.x).abs() < 1e-3 && (back.y - p.y).abs() < 1e-3);
    }
}
